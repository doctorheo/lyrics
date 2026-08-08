import os
import re
import uuid
import json
import shutil
import tempfile
import asyncio
import subprocess
import threading
import warnings
import zipfile
from typing import List, Tuple, Dict, Any
from difflib import SequenceMatcher

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# Whisper AI & mutagen
import whisper
from mutagen import File as MutagenFile

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: Dict[str, Dict[str, Any]] = {}

def remove_dir(path: str):
    """임시 디렉토리 삭제"""
    shutil.rmtree(path, ignore_errors=True)

def log_task(task_id: str, message: str):
    """실시간 터미널 로그 버퍼에 저장"""
    if task_id in tasks:
        tasks[task_id]["logs"].append(message)

# ==========================================
# 1. .song 패키지 생성 및 Whisper 싱크 분석
# ==========================================

def get_similarity(str1, str2):
    s1 = str1.replace(" ", "")
    s2 = str2.replace(" ", "")
    return SequenceMatcher(None, s1, s2).ratio()

def extract_album_art(audio_path, output_dir, task_id: str):
    log_task(task_id, "🖼 미디어 파일에서 앨범 아트를 추출하고 있습니다...")
    try:
        audio = MutagenFile(audio_path)
        if audio is None:
            log_task(task_id, "⚠️ 메타데이터를 읽을 수 없는 파일입니다.")
            return None

        # 1. MP3 (ID3 태그)
        if audio.tags:
            for key in audio.tags.keys():
                if key.startswith('APIC'):
                    apic = audio.tags[key]
                    cover_data = apic.data
                    mime = getattr(apic, 'mime', '').lower()
                    ext = "png" if "png" in mime else "jpg"
                    cover_path = os.path.join(output_dir, f"cover.{ext}")
                    with open(cover_path, "wb") as f:
                        f.write(cover_data)
                    log_task(task_id, f"✅ MP3 앨범 아트 추출 완료: cover.{ext}")
                    return cover_path

        # 2. MP4 / M4A
        if hasattr(audio, 'get') and audio.get('covr'):
            covers = audio['covr']
            if covers:
                cover_data = covers[0]
                is_png = getattr(cover_data, 'imageformat', None) == 14
                ext = "png" if is_png else "jpg"
                cover_path = os.path.join(output_dir, f"cover.{ext}")
                with open(cover_path, "wb") as f:
                    f.write(bytes(cover_data))
                log_task(task_id, f"✅ M4A/MP4 앨범 아트 추출 완료: cover.{ext}")
                return cover_path

        # 3. FLAC
        if hasattr(audio, 'pictures') and audio.pictures:
            pic = audio.pictures[0]
            cover_data = pic.data
            mime = getattr(pic, 'mime', '').lower()
            ext = "png" if "png" in mime else "jpg"
            cover_path = os.path.join(output_dir, f"cover.{ext}")
            with open(cover_path, "wb") as f:
                f.write(cover_data)
            log_task(task_id, f"✅ FLAC 앨범 아트 추출 완료: cover.{ext}")
            return cover_path

        log_task(task_id, "⚠️ 미디어 파일 내에 내장된 앨범 아트를 찾을 수 없습니다.")
    except Exception as e:
        log_task(task_id, f"⚠️ 앨범 아트 추출 중 오류가 발생했으나 계속 진행합니다: {e}")
    
    return None

def transcribe_and_align_with_similarity(audio_file, lyrics_text, task_id: str):
    log_task(task_id, "🎵 다국어 음성 분석을 시작합니다. (Whisper 모델 로드 중)...")
    tasks[task_id]["progress"] = 20
    model = whisper.load_model("base")
    
    log_task(task_id, "🎤 오디오 음성 신호와 싱크를 정밀 분석하고 있습니다...")
    tasks[task_id]["progress"] = 45
    
    result = model.transcribe(audio_file, word_timestamps=True, fp16=False)
    detected_lang = result.get("language", "unknown")
    log_task(task_id, f"🌐 감지된 음성 언어: [{detected_lang.upper()}]")

    total_duration = get_audio_duration(audio_file)
    
    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                all_words.append({
                    "word": word_info["word"].strip(),
                    "start": word_info["start"],
                    "end": word_info["end"]
                })
                
    log_task(task_id, "📝 입력된 가사와의 관련도를 분석하여 싱크 매칭을 진행합니다...")
    tasks[task_id]["progress"] = 75
    
    lines = []
    for line in lyrics_text.split("\n"):
        cleaned_line = line.strip()
        cleaned_line = re.sub(r'\[.*?\]|【.*?】|「.*?」', '', cleaned_line)
        
        parentheses_matches = re.findall(r'[\(（](.*?)[\)）]', cleaned_line)
        for content in parentheses_matches:
            if ',' in content or any(word in content for word in ['소리', '웃음', '쾅', '배경', '음향', '笑', '声', '背景']):
                cleaned_line = cleaned_line.replace(f"({content})", "").replace(f"（{content}）", "")
            else:
                cleaned_line = cleaned_line.replace(f"({content})", content).replace(f"（{content}）", content)
        
        cleaned_line = cleaned_line.strip()
        if cleaned_line:
            lines.append(cleaned_line)
            
    lrc_lines = []
    word_idx = 0
    total_words = len(all_words)
    last_valid_time = 0.0
    
    for idx, line in enumerate(lines):
        best_similarity = -1
        best_start_time = None
        best_word_count = 0
        
        for check_count in range(1, min(16, total_words - word_idx + 1)):
            current_combination = all_words[word_idx : word_idx + check_count]
            combined_text = "".join([w["word"] for w in current_combination])
            
            sim = get_similarity(line, combined_text)
            
            if sim > best_similarity:
                best_similarity = sim
                best_start_time = current_combination[0]["start"]
                best_word_count = check_count
                
            if sim > 0.95:
                break
        
        if best_start_time is not None and best_similarity > 0.1:
            last_valid_time = best_start_time
            minutes = int(best_start_time // 60)
            seconds = best_start_time % 60
            timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
            lrc_lines.append(f"{timestamp} {line}")
            word_idx += best_word_count
        else:
            if word_idx < total_words:
                fallback_time = all_words[word_idx]["start"]
            else:
                remaining_lines = len(lines) - idx
                fallback_time = last_valid_time + ((total_duration - last_valid_time) / (remaining_lines + 1))
            
            last_valid_time = fallback_time
            minutes = int(fallback_time // 60)
            seconds = fallback_time % 60
            timestamp = f"[{minutes:02d}:{seconds:05.2f}]"
            lrc_lines.append(f"{timestamp} {line}")
            
            if word_idx < total_words:
                word_idx += 1
            
    return "\n".join(lrc_lines)

def make_song_package(media_path, lrc_content, output_path, task_id: str):
    log_task(task_id, "📦 플레이어 전용 패키지(.song) 파일을 생성하고 있습니다...")
    base_dir = os.path.dirname(output_path)
    
    temp_lrc_path = os.path.join(base_dir, "lyrics.lrc")
    with open(temp_lrc_path, "w", encoding="utf-8") as f:
        f.write(lrc_content)
        
    cover_path = extract_album_art(media_path, base_dir, task_id)
        
    media_filename = os.path.basename(media_path)
    _, ext = os.path.splitext(media_filename)
    
    target_media_name = "video.mp4" if ext.lower() == ".mp4" else "audio.mp3"

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as song_zip:
        song_zip.write(media_path, arcname=target_media_name)
        song_zip.write(temp_lrc_path, arcname="lyrics.lrc")
        
        if cover_path and os.path.exists(cover_path):
            cover_filename = os.path.basename(cover_path)
            song_zip.write(cover_path, arcname=cover_filename)
        
    if os.path.exists(temp_lrc_path):
        os.remove(temp_lrc_path)
    if cover_path and os.path.exists(cover_path):
        os.remove(cover_path)
        
    log_task(task_id, f"✨ .song 패키지 재패킹이 성공적으로 완료되었습니다!")
    tasks[task_id]["progress"] = 100

def process_song_creation(task_id: str, media_bytes: bytes, original_filename: str, lyrics_text: str):
    tmpdir = tempfile.mkdtemp()
    tasks[task_id]["tmpdir"] = tmpdir

    try:
        log_task(task_id, "[SYSTEM] .song 패키지 생성 프로세스 시작")
        original_media_path = os.path.join(tmpdir, original_filename)
        with open(original_media_path, "wb") as f:
            f.write(media_bytes)

        _, ext = os.path.splitext(original_filename)
        ext_lower = ext.lower()

        audio_for_whisper = original_media_path

        # MP4 비디오일 경우 native FFmpeg로 안전하게 오디오 추출
        if ext_lower == ".mp4":
            log_task(task_id, "🎬 MP4 비디오 파일 감지: FFmpeg로 오디오 추출 중...")
            tasks[task_id]["progress"] = 10
            extracted_mp3_path = os.path.join(tmpdir, "extracted_audio.mp3")
            
            ffmpeg_extract_cmd = [
                "ffmpeg", "-y",
                "-i", original_media_path,
                "-vn",
                "-c:a", "mp3",
                "-q:a", "2",
                extracted_mp3_path
            ]
            
            res = subprocess.run(
                ffmpeg_extract_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if res.returncode != 0:
                log_task(task_id, f"❌ FFmpeg 오디오 추출 실패 로그:\n{res.stderr}")
                raise Exception("MP4 파일에서 오디오 트랙을 추출할 수 없습니다. (소리가 없는 비디오이거나 손상된 파일입니다.)")
            
            log_task(task_id, "✅ 오디오 추출 완료!")
            audio_for_whisper = extracted_mp3_path

        # Whisper 음성 분석 및 가사 싱크 매칭
        lrc_content = transcribe_and_align_with_similarity(audio_for_whisper, lyrics_text, task_id)
        
        output_song = os.path.join(tmpdir, f"{os.path.splitext(original_filename)[0]}.song")
        tasks[task_id]["output_file"] = output_song

        make_song_package(original_media_path, lrc_content, output_song, task_id)
        tasks[task_id]["status"] = "completed"

    except Exception as e:
        log_task(task_id, f"Error: {str(e)}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error_msg"] = str(e)

# ==========================================
# 2. MP4 렌더링 엔진
# ==========================================

def parse_lrc(lrc_text: str) -> List[Tuple[float, str]]:
    lines = lrc_text.splitlines()
    result = []
    pattern = re.compile(r'\[(\d+):(\d+(?:\.\d+)?)\]')
    for line in lines:
        match = pattern.search(line)
        if match:
            min_val = int(match.group(1))
            sec_val = float(match.group(2))
            timestamp = min_val * 60 + sec_val
            text = pattern.sub('', line).strip()
            result.append((timestamp, text))
    result.sort(key=lambda x: x[0])
    return result

def get_audio_duration(audio_path: str) -> float:
    try:
        import mutagen
        audio_file = mutagen.File(audio_path)
        if audio_file is not None and audio_file.info and hasattr(audio_file.info, 'length'):
            return float(audio_file.info.length)
    except Exception:
        pass

    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprintwrappers=1:nokey=1", audio_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        if res.stdout and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass

    try:
        cmd_ffmpeg = ["ffmpeg", "-i", audio_path]
        res_ff = subprocess.run(cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res_ff.stderr)
        if match:
            return float(match.group(1)) * 3600 + float(match.group(2)) * 60 + float(match.group(3))
    except Exception:
        pass

    raise Exception("오디오 파일 길이를 읽을 수 없습니다.")

def render_frame(width: int, height: int, album_img: Image.Image, lrc_data: List[Tuple[float, str]], current_time: float, mode: str, attach_art: bool) -> Image.Image:
    bg_color = (18, 18, 18) if mode == "classic" else (0, 0, 0)
    base_img = Image.new("RGB", (width, height), bg_color)

    if mode == "new" and attach_art and album_img:
        bg_art = album_img.copy()
        img_w, img_h = bg_art.size
        aspect_target = width / height
        aspect_img = img_w / img_h
        
        if aspect_img > aspect_target:
            new_w = int(img_h * aspect_target)
            left = (img_w - new_w) // 2
            bg_art = bg_art.crop((left, 0, left + new_w, img_h))
        else:
            new_h = int(img_w / aspect_target)
            top = (img_h - new_h) // 2
            bg_art = bg_art.crop((0, top, img_w, top + new_h))
            
        bg_art = bg_art.resize((width, height), Image.Resampling.LANCZOS)
        bg_art = bg_art.filter(ImageFilter.GaussianBlur(35))
        enhancer = ImageEnhance.Brightness(bg_art)
        bg_art = enhancer.enhance(0.35)
        base_img.paste(bg_art, (0, 0))

    video_area_w = int(width * 0.45) if attach_art else 0
    
    if attach_art and album_img:
        art = album_img.copy()
        target_size = min(int(width * 0.38), int(height * 0.72))
        art.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        art_w, art_h = art.size
        
        art_x = (video_area_w - art_w) // 2
        art_y = (height - art_h) // 2

        if mode == "new":
            shadow = Image.new("RGBA", (art_w + 20, art_h + 20), (0, 0, 0, 0))
            s_draw = ImageDraw.Draw(shadow)
            s_draw.rectangle([10, 10, art_w + 10, art_h + 10], fill=(0, 0, 0, 180))
            shadow = shadow.filter(ImageFilter.GaussianBlur(10))
            base_img.paste(shadow, (art_x - 10, art_y - 10), shadow)

        base_img.paste(art, (art_x, art_y))

    lyrics_area_x = video_area_w
    lyrics_area_w = width - lyrics_area_x
    
    if mode == "new":
        overlay = Image.new("RGBA", (lyrics_area_w, height), (0, 0, 0, 64))
        base_img.paste(overlay, (lyrics_area_x, 0), overlay)

    draw = ImageDraw.Draw(base_img)

    active_idx = -1
    for i in range(len(lrc_data) - 1, -1, -1):
        if current_time >= lrc_data[i][0]:
            active_idx = i
            break

    try:
        font_active = ImageFont.truetype("malgunbd.ttf", 30) if mode == "classic" else ImageFont.truetype("malgunbd.ttf", 32)
        font_sub = ImageFont.truetype("malgun.ttf", 20) if mode == "classic" else ImageFont.truetype("malgun.ttf", 22)
    except Exception:
        font_active = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    center_x = lyrics_area_x + (lyrics_area_w // 2)
    center_y = height // 2
    max_text_w = lyrics_area_w - 60

    def wrap_text(text: str, font) -> List[str]:
        words = text.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = font.getbbox(test_line)
            if (bbox[2] - bbox[0]) <= max_text_w:
                current_line = test_line
            else:
                if current_line: lines.append(current_line)
                current_line = word
        if current_line: lines.append(current_line)
        return lines or [text]

    def get_font_line_height(font) -> int:
        if hasattr(font, 'size'):
            return int(font.size * 1.35)
        return 20

    def get_block_info(text: str, font) -> Tuple[List[str], int, int]:
        lines = wrap_text(text, font)
        lh = get_font_line_height(font)
        return lines, lh, len(lines) * lh

    def draw_lines_block(start_y: int, lines: List[str], line_h: int, font, fill_color: tuple, shadow_opacity: float):
        curr_y = start_y
        for line in lines:
            if curr_y + line_h > 0 and curr_y < height:
                bbox = font.getbbox(line)
                line_w = bbox[2] - bbox[0]
                x = center_x - (line_w // 2)

                if shadow_opacity > 0:
                    s_color = (0, 0, 0, int(255 * shadow_opacity))
                    draw.text((x + 1, curr_y + 2), line, fill=s_color, font=font)

                draw.text((x, curr_y), line, fill=fill_color, font=font)
            curr_y += line_h

    GAP = 28

    if lrc_data:
        curr_idx = active_idx if active_idx >= 0 else 0
        act_lines, act_lh, act_total_h = get_block_info(lrc_data[curr_idx][1], font_active)
        act_start_y = center_y - (act_total_h // 2)
        act_color = (255, 255, 255) if active_idx >= 0 else (180, 180, 180)

        if mode == "classic":
            draw_lines_block(act_start_y, act_lines, act_lh, font_active, act_color, shadow_opacity=0.0)

            if curr_idx > 0:
                prev_lines, prev_lh, prev_total_h = get_block_info(lrc_data[curr_idx - 1][1], font_sub)
                prev_start_y = act_start_y - GAP - prev_total_h
                draw_lines_block(prev_start_y, prev_lines, prev_lh, font_sub, (120, 120, 120), shadow_opacity=0.0)

            if curr_idx < len(lrc_data) - 1:
                next_lines, next_lh, next_total_h = get_block_info(lrc_data[curr_idx + 1][1], font_sub)
                next_start_y = act_start_y + act_total_h + GAP
                draw_lines_block(next_start_y, next_lines, next_lh, font_sub, (120, 120, 120), shadow_opacity=0.0)

        else:
            draw_lines_block(act_start_y, act_lines, act_lh, font_active, act_color, shadow_opacity=0.8)

            top_y = act_start_y
            idx_up = curr_idx - 1
            while idx_up >= 0 and top_y > -100:
                lines, lh, total_h = get_block_info(lrc_data[idx_up][1], font_sub)
                top_y -= (total_h + GAP)
                dist = abs(curr_idx - idx_up)
                alpha = max(70, 180 - (dist * 25))
                draw_lines_block(top_y, lines, lh, font_sub, (alpha, alpha, alpha), shadow_opacity=0.3)
                idx_up -= 1

            bot_y = act_start_y + act_total_h
            idx_down = curr_idx + 1
            while idx_down < len(lrc_data) and bot_y < height + 100:
                lines, lh, total_h = get_block_info(lrc_data[idx_down][1], font_sub)
                draw_y = bot_y + GAP
                dist = abs(curr_idx - idx_down)
                alpha = max(70, 180 - (dist * 25))
                draw_lines_block(draw_y, lines, lh, font_sub, (alpha, alpha, alpha), shadow_opacity=0.3)
                bot_y = draw_y + total_h
                idx_down += 1

    return base_img

def read_ffmpeg_stderr(proc, task_id: str):
    for line in iter(proc.stderr.readline, ''):
        if line:
            clean_line = line.rstrip()
            log_task(task_id, clean_line)

            if "frame=" in clean_line:
                match = re.search(r"frame=\s*(\d+)", clean_line)
                if match and tasks[task_id].get("total_frames", 0) > 0:
                    curr_frame = int(match.group(1))
                    tot_frames = tasks[task_id]["total_frames"]
                    pct = min(100, int((curr_frame / tot_frames) * 100))
                    tasks[task_id]["progress"] = pct
    proc.stderr.close()

def process_video_conversion(task_id: str, file_bytes: bytes, original_filename: str, mode: str, attach_art: bool):
    tmpdir = tempfile.mkdtemp()
    tasks[task_id]["tmpdir"] = tmpdir

    try:
        zip_path = os.path.join(tmpdir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(file_bytes)

        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        lrc_data = []
        audio_path = None
        cover_path = None

        for root, _, files in os.walk(tmpdir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if filename.endswith(".lrc"):
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        lrc_data = parse_lrc(f.read())
                elif filename.endswith(".mp3"):
                    audio_path = filepath
                elif re.search(r'cover.*\.(jpg|png)$', filename, re.IGNORECASE):
                    cover_path = filepath

        if not audio_path:
            raise Exception("ZIP 내 MP3 파일이 존재하지 않습니다.")

        duration = get_audio_duration(audio_path)
        album_img = Image.open(cover_path).convert("RGB") if cover_path else None
        output_mp4 = os.path.join(tmpdir, "output.mp4")
        tasks[task_id]["output_file"] = output_mp4

        fps = 10
        total_frames = int(duration * fps)
        tasks[task_id]["total_frames"] = total_frames

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", "1280x720",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            output_mp4
        ]

        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        stderr_thread = threading.Thread(target=read_ffmpeg_stderr, args=(proc, task_id))
        stderr_thread.daemon = True
        stderr_thread.start()

        for i in range(total_frames):
            t = i / fps
            frame_img = render_frame(1280, 720, album_img, lrc_data, t, mode, attach_art)
            proc.stdin.buffer.write(frame_img.tobytes())

        proc.stdin.close()
        proc.wait()
        stderr_thread.join()

        tasks[task_id]["progress"] = 100
        tasks[task_id]["status"] = "completed"

    except Exception as e:
        log_task(task_id, f"Error: {str(e)}")
        tasks[task_id]["status"] = "error"
        tasks[task_id]["error_msg"] = str(e)

# ==========================================
# 3. 라우팅 엔드포인트
# ==========================================

@app.post("/start_convert")
async def start_convert(
    file: UploadFile = File(...),
    mode: str = Form("new"),
    attach_art: str = Form("true")
):
    task_id = str(uuid.uuid4())
    contents = await file.read()
    is_attach_art = attach_art.lower() == "true"

    tasks[task_id] = {
        "progress": 0,
        "status": "processing",
        "logs": [],
        "filename": f"{os.path.splitext(file.filename)[0]}.mp4",
        "error_msg": "",
        "total_frames": 0
    }

    thread = threading.Thread(
        target=process_video_conversion,
        args=(task_id, contents, file.filename, mode, is_attach_art)
    )
    thread.start()

    return {"task_id": task_id}

@app.post("/start_create_song")
async def start_create_song(
    audio_file: UploadFile = File(...),
    lyrics: str = Form(...)
):
    task_id = str(uuid.uuid4())
    contents = await audio_file.read()

    tasks[task_id] = {
        "progress": 0,
        "status": "processing",
        "logs": [],
        "filename": f"{os.path.splitext(audio_file.filename)[0]}.song",
        "error_msg": ""
    }

    thread = threading.Thread(
        target=process_song_creation,
        args=(task_id, contents, audio_file.filename, lyrics)
    )
    thread.start()

    return {"task_id": task_id}

@app.get("/stream_logs/{task_id}")
async def stream_logs(task_id: str):
    async def event_generator():
        last_log_idx = 0
        while True:
            if task_id not in tasks:
                yield f"data: {json.dumps({'error': 'Task Not Found'})}\n\n"
                break

            task = tasks[task_id]
            current_logs = task["logs"][last_log_idx:]
            last_log_idx = len(task["logs"])

            payload = {
                "progress": task["progress"],
                "status": task["status"],
                "logs": current_logs,
                "error": task.get("error_msg", "")
            }

            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            if task["status"] in ("completed", "error"):
                break

            await asyncio.sleep(0.2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/download/{task_id}")
async def download_file(task_id: str, background_tasks: BackgroundTasks):
    if task_id not in tasks or tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    task = tasks[task_id]
    file_path = task.get("output_file")

    if "tmpdir" in task:
        background_tasks.add_task(remove_dir, task["tmpdir"])

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=task.get("filename", "output.file")
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
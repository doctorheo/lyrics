import os
import re
import tempfile
import subprocess
import zipfile
import shutil
from typing import List, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont

app = FastAPI()

# 프론트엔드 통신 허용 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def remove_dir(path: str):
    """파일 전송 후 임시 디렉토리를 삭제하는 백그라운드 작업 함수"""
    shutil.rmtree(path, ignore_errors=True)

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
    # 1. mutagen 시도
    try:
        import mutagen
        audio_file = mutagen.File(audio_path)
        if audio_file is not None and audio_file.info and hasattr(audio_file.info, 'length'):
            return float(audio_file.info.length)
    except Exception:
        pass

    # 2. ffprobe 시도
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprintwrappers=1:nokey=1",
            audio_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        if res.stdout and res.stdout.strip():
            return float(res.stdout.strip())
    except Exception:
        pass

    # 3. ffmpeg 직접 헤더 분석 시도
    try:
        cmd_ffmpeg = ["ffmpeg", "-i", audio_path]
        res_ff = subprocess.run(cmd_ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res_ff.stderr)
        if match:
            hours = float(match.group(1))
            minutes = float(match.group(2))
            seconds = float(match.group(3))
            return hours * 3600 + minutes * 60 + seconds
    except Exception:
        pass

    raise HTTPException(status_code=500, detail="오디오 파일 길이를 읽을 수 없습니다.")

def render_frame(width: int, height: int, album_img: Image.Image, lrc_data: List[Tuple[float, str]], current_time: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (17, 17, 17))
    draw = ImageDraw.Draw(img)

    # 1. 앨범아트 렌더링
    if album_img:
        art_size = 420
        art_x = 60
        art_y = (height - art_size) // 2
        resized_art = album_img.resize((art_size, art_size), Image.Resampling.LANCZOS)
        img.paste(resized_art, (art_x, art_y))

    # 2. 현재 가사 인덱스 계산
    active_idx = -1
    for i in range(len(lrc_data) - 1, -1, -1):
        if current_time >= lrc_data[i][0]:
            active_idx = i
            break

    # 가사 영역 가로폭 및 중심 위치 설정 (앨범아트 오른쪽 영역 전체 활용)
    lyrics_area_start_x = 510  # 앨범아트(60+420=480) 우측 여백 고려
    lyrics_area_width = width - lyrics_area_start_x - 40  # 우측 여백 40px
    center_x = lyrics_area_start_x + (lyrics_area_width // 2)
    center_y = height // 2

    # 폰트 설정
    try:
        font_active = ImageFont.truetype("malgun.ttf", 30)
        font_sub = ImageFont.truetype("malgun.ttf", 20)
    except Exception:
        font_active = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # 긴 가사 자동 줄바꿈 함수
    def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
        words = text.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            bbox = font.getbbox(test_line)
            w = bbox[2] - bbox[0]
            if w <= max_w:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines or [text]

    # 다중 줄 텍스트 중앙 렌더링 함수
    def draw_multiline_text(y_center: int, text: str, font: ImageFont.FreeTypeFont, fill_color: tuple):
        wrapped_lines = wrap_text(text, font, lyrics_area_width)
        
        # 줄 간격 계산
        line_heights = []
        for line in wrapped_lines:
            bbox = font.getbbox(line)
            line_heights.append(bbox[3] - bbox[1] + 6) # 6px 줄간격
            
        total_h = sum(line_heights)
        start_y = y_center - (total_h // 2)

        curr_y = start_y
        for i, line in enumerate(wrapped_lines):
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            x = center_x - (line_w // 2)
            draw.text((x, curr_y), line, fill=fill_color, font=font)
            curr_y += line_heights[i]

    if active_idx >= 0 and lrc_data:
        # 현재 가사 (하일라이트)
        draw_multiline_text(center_y, lrc_data[active_idx][1], font_active, (255, 255, 255))

        # 이전 가사 (상단)
        if active_idx > 0:
            draw_multiline_text(center_y - 75, lrc_data[active_idx - 1][1], font_sub, (140, 140, 140))

        # 다음 가사 (하단)
        if active_idx < len(lrc_data) - 1:
            draw_multiline_text(center_y + 75, lrc_data[active_idx + 1][1], font_sub, (140, 140, 140))

    return img

@app.post("/convert")
async def convert_to_mp4(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()

    # 안전하게 임시 폴더 직접 생성
    tmpdir = tempfile.mkdtemp()
    
    # 다운로드가 끝난 후 폴더 삭제되도록 백그라운드 등록
    background_tasks.add_task(remove_dir, tmpdir)

    zip_path = os.path.join(tmpdir, "uploaded.zip")
    with open(zip_path, "wb") as f:
        f.write(contents)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
    except Exception:
        raise HTTPException(status_code=400, detail="ZIP 파일 해제 실패")

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
        raise HTTPException(status_code=400, detail="ZIP 내 MP3 파일이 없습니다.")

    duration = get_audio_duration(audio_path)
    album_img = Image.open(cover_path).convert("RGB") if cover_path else None

    output_mp4 = os.path.join(tmpdir, "output.mp4")
    fps = 10
    total_frames = int(duration * fps)

    # FFmpeg 입출력 연결
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

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    for i in range(total_frames):
        t = i / fps
        frame_img = render_frame(1280, 720, album_img, lrc_data, t)
        proc.stdin.write(frame_img.tobytes())

    proc.stdin.close()
    proc.wait()

    return FileResponse(
        output_mp4,
        media_type="video/mp4",
        filename=f"{os.path.splitext(file.filename)[0]}.mp4"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
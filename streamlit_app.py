import base64
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import server

st.set_page_config(page_title="가사 플레이어", layout="wide")


def ensure_api_server():
    if st.session_state.get("api_server_started"):
        return

    def run_uvicorn():
        try:
            import uvicorn
            uvicorn.run(server.app, host="0.0.0.0", port=8000, log_level="warning")
        except Exception:
            server_path = Path(__file__).with_name("server.py")
            subprocess.Popen(
                [sys.executable, str(server_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(server_path.parent),
            )

    thread = threading.Thread(target=run_uvicorn, daemon=True)
    thread.start()
    st.session_state["api_server_started"] = True
    time.sleep(0.3)


def load_index_html() -> str:
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    st.markdown(
        """
        <style>
        html, body, .stApp, .css-1d391kg, .main, .block-container {
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            height: 100% !important;
            min-height: 100% !important;
        }
        .reportview-container .main .block-container {
            max-width: 100% !important;
        }
        .css-1v3fvcr, .css-1q1n7m7, .css-1d391kg > header, footer {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    html_content = load_index_html().encode("utf-8")
    encoded = base64.b64encode(html_content).decode("ascii")
    src = f"data:text/html;charset=utf-8;base64,{encoded}"

    component_html = f"""
        <style>
        html, body {{ margin: 0; padding: 0; height: 100%; overflow: hidden; background: transparent; }}
        iframe#fullScreenIndex {{
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            border: none;
            margin: 0;
            padding: 0;
            overflow: hidden;
            z-index: 999999;
        }}
        </style>
        <iframe id="fullScreenIndex" src="{src}"></iframe>
    """

    ensure_api_server()
    components.html(component_html, height=1600, scrolling=False)


if __name__ == "__main__":
    main()

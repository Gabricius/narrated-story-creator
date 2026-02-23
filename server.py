from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal, Optional, Annotated
from contextlib import asynccontextmanager
from pydantic import Field
from enum import Enum
import asyncio
import queue
import threading
import time
import uuid
import requests
import shelve
import os
import atexit
import signal
import sys
import base64
import json
import random
import re
import torch

def normalize_drive_url(url: str) -> str:
    """Convert Google Drive share/view links to direct download URLs.
    
    Input:  https://drive.google.com/file/d/FILE_ID/view?usp=drive_link
    Output: https://drive.google.com/uc?export=download&id=FILE_ID&confirm=t
    
    The confirm=t parameter helps bypass the virus scan confirmation page for large files.
    """
    if not url:
        return url
    match = re.search(r'drive\.google\.com/file/d/([^/?]+)', url)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    # Already a uc? link — add confirm=t if missing
    if 'drive.google.com/uc?' in url and 'confirm=' not in url:
        return url + ('&' if '?' in url else '?') + 'confirm=t'
    return url
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount
from video_maker import (
    create_overlay,
    create_tts_international,
    create_tts_english,
    create_subtitle_segments_international,
    create_subtitle_segments_english,
    create_subtitle,
    create_subtitle_v2_karaoke,
    render_video,
)
import shutil
import gc
import wave

# ═══════════════════════════════
# CHUNKED TTS — Process large texts in pieces to avoid OOM
# ═══════════════════════════════

TTS_CHUNK_CHARS = 1500  # Max chars per TTS chunk (~1.5 min audio each, safer for low RAM)

def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB → MB
    except:
        return 0

def split_text_into_chunks(text, max_chars=TTS_CHUNK_CHARS):
    """Split text into chunks at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current = current + " " + s if current else s
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]

def concatenate_wav_files(wav_paths, output_path):
    """Concatenate multiple WAV files into one."""
    if len(wav_paths) == 1:
        os.rename(wav_paths[0], output_path)
        return
    
    params = None
    frames = []
    for p in wav_paths:
        with wave.open(p, 'rb') as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    
    with wave.open(output_path, 'wb') as out:
        out.setparams(params)
        for f in frames:
            out.writeframes(f)
    
    # Cleanup chunk files
    for p in wav_paths:
        try:
            os.remove(p)
        except:
            pass

def create_tts_chunked(text, output_path, lang_code, voice, is_international=False):
    """Process TTS in isolated subprocesses to avoid OOM.
    
    Each chunk runs in a separate Python process that loads the model,
    generates audio, saves results, and exits — releasing ALL memory back to OS.
    """
    import subprocess as sp
    
    chunks = split_text_into_chunks(text)
    total_chunks = len(chunks)
    print(f"[TTS] Processing {len(text)} chars in {total_chunks} chunks (max {TTS_CHUNK_CHARS} chars/chunk)")
    print(f"[TTS] Using SUBPROCESS isolation to prevent OOM")
    print(f"[TTS] Memory before start: {get_memory_mb():.0f} MB")
    
    all_captions = []
    chunk_wav_paths = []
    cumulative_duration = 0.0
    
    video_dir = os.path.dirname(output_path)
    
    for i, chunk_text in enumerate(chunks):
        chunk_path = output_path.replace('.wav', f'_chunk{i}.wav')
        chunk_text_path = os.path.join(video_dir, f'chunk_{i}_text.txt')
        chunk_captions_path = os.path.join(video_dir, f'chunk_{i}_captions.json')
        
        print(f"[TTS] Chunk {i+1}/{total_chunks}: {len(chunk_text)} chars | Memory: {get_memory_mb():.0f} MB")
        
        # Write chunk text to file (avoids command line length limits)
        with open(chunk_text_path, 'w', encoding='utf-8') as f:
            f.write(chunk_text)
        
        # Build subprocess script
        tts_func = 'create_tts_international' if is_international else 'create_tts_english'
        worker_script = f'''
import json, sys
sys.path.insert(0, '/app')
from video_maker import {tts_func}

with open("{chunk_text_path}", "r", encoding="utf-8") as f:
    text = f.read()

captions, audio_length = {tts_func}(
    text=text, output_path="{chunk_path}",
    lang_code="{lang_code}", voice="{voice}",
)

# Serialize captions to JSON — handle ANY caption format
serializable = []
for cap in captions:
    if isinstance(cap, dict):
        serializable.append(cap)
    elif isinstance(cap, (list, tuple)):
        serializable.append(list(cap))
    elif hasattr(cap, '__dict__'):
        # Custom object (WordCaption, etc.) — convert to dict
        serializable.append(cap.__dict__)
    elif hasattr(cap, '_asdict'):
        # namedtuple — convert to dict
        serializable.append(cap._asdict())
    elif hasattr(cap, 'start') and hasattr(cap, 'end'):
        # Duck-type: anything with start/end/word
        d = {{"start": cap.start, "end": cap.end}}
        if hasattr(cap, 'word'): d["word"] = cap.word
        if hasattr(cap, 'text'): d["text"] = cap.text
        serializable.append(d)
    else:
        # Last resort — try to convert, log warning
        print(f"WARNING: Unknown caption type: {{type(cap).__name__}} — {{repr(cap)[:100]}}")
        serializable.append({{"text": str(cap), "start": 0, "end": 0}})

with open("{chunk_captions_path}", "w") as f:
    json.dump({{"captions": serializable, "audio_length": audio_length}}, f)

print(f"CHUNK_OK audio_length={{audio_length:.2f}} captions={{len(serializable)}}")
'''
        
        worker_script_path = os.path.join(video_dir, f'chunk_{i}_worker.py')
        with open(worker_script_path, 'w') as f:
            f.write(worker_script)
        
        # Run in subprocess — all model memory is freed when process exits
        try:
            result = sp.run(
                ['python3', worker_script_path],
                capture_output=True, text=True,
                timeout=600,  # 10 min max per chunk
                cwd='/app'
            )
            
            print(f"[TTS] Subprocess stdout: {result.stdout[-200:] if result.stdout else '(empty)'}")
            if result.stderr:
                # Filter out warnings, only show errors
                errors = [l for l in result.stderr.split('\n') 
                         if l and not any(w in l for w in ['Warning', 'WARNING', 'FutureWarning', 'UserWarning', 'notice', 'pip'])]
                if errors:
                    print(f"[TTS] Subprocess errors: {chr(10).join(errors[-5:])}")
            
            if result.returncode != 0:
                raise Exception(f"TTS subprocess failed (exit {result.returncode}): {result.stderr[-500:]}")
            
            # Read captions from JSON file
            if not os.path.exists(chunk_captions_path):
                raise Exception(f"TTS subprocess didn't produce captions file")
            
            with open(chunk_captions_path, 'r') as f:
                chunk_result = json.load(f)
            
            captions = chunk_result['captions']
            audio_length = chunk_result['audio_length']
            
            # Adjust timestamps by cumulative offset
            adjusted_captions = []
            
            # Log first caption structure for debugging
            if captions and i == 0:
                sample = captions[0]
                if isinstance(sample, dict):
                    print(f"[TTS] Caption format: dict with keys {list(sample.keys())}")
                else:
                    print(f"[TTS] Caption format: {type(sample).__name__} = {repr(sample)[:150]}")
            
            for cap in captions:
                if isinstance(cap, dict):
                    adj = dict(cap)
                    for start_key in ['start', 'start_ts', 'start_time', 's']:
                        if start_key in adj:
                            adj[start_key] = float(adj[start_key]) + cumulative_duration
                            break
                    for end_key in ['end', 'end_ts', 'end_time', 'e']:
                        if end_key in adj:
                            adj[end_key] = float(adj[end_key]) + cumulative_duration
                            break
                    adjusted_captions.append(adj)
                elif isinstance(cap, (list, tuple)) and len(cap) >= 2:
                    cap_list = list(cap)
                    cap_list[0] += cumulative_duration
                    cap_list[1] += cumulative_duration
                    adjusted_captions.append(cap_list)
                else:
                    adjusted_captions.append(cap)
            
            all_captions.extend(adjusted_captions)
            chunk_wav_paths.append(chunk_path)
            cumulative_duration += audio_length
            
            print(f"[TTS] Chunk {i+1} done: {audio_length:.1f}s (total: {cumulative_duration:.1f}s) | Memory: {get_memory_mb():.0f} MB")
            
        except sp.TimeoutExpired:
            print(f"[TTS] Chunk {i+1} timed out after 600s")
            raise Exception(f"TTS chunk {i+1} timed out")
        except Exception as e:
            print(f"[TTS] Chunk {i+1} failed: {e}")
            for p in chunk_wav_paths:
                try: os.remove(p)
                except: pass
            raise
        finally:
            # Clean up temp files for this chunk
            for tmp in [chunk_text_path, chunk_captions_path, worker_script_path]:
                try: os.remove(tmp)
                except: pass
    
    # Concatenate all chunks
    print(f"[TTS] Concatenating {len(chunk_wav_paths)} audio chunks...")
    concatenate_wav_files(chunk_wav_paths, output_path)
    
    print(f"[TTS] Complete: {cumulative_duration:.1f}s total audio | Memory: {get_memory_mb():.0f} MB")
    return all_captions, cumulative_duration

CUDA = os.environ.get("CUDA", "0")
if CUDA == "1" and torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using CPU")
    device = torch.device("cpu")
    num_cores = os.cpu_count()
    if os.path.exists("/sys/fs/cgroup/cpu.max"):
        with open("/sys/fs/cgroup/cpu.max", "r") as f:
            line = f.readline()
            if len(line.split()) == 2:
                if line.split()[0] == "max":
                    print("File /sys/fs/cgroup/cpu.max has max value, using os.cpu_count()")
                else:
                    cpu_max = int(line.split()[0])
                    cpu_period = int(line.split()[1])
                    num_cores = cpu_max // cpu_period
                    print(f"Using {num_cores} cores")
            else:
                print("File /sys/fs/cgroup/cpu.max does not have 2 values, using os.cpu_count()")
    else:
        print("File /sys/fs/cgroup/cpu.max not found, using os.cpu_count()")
    
    # Use fewer threads to reduce memory overhead on low-RAM containers
    num_threads = os.environ.get("NUM_THREADS", max(1, num_cores))
    torch.set_num_threads(int(num_threads))
    # Reduce interop threads too
    torch.set_num_interop_threads(1)
    print(f"[MEM] Torch threads: {num_threads}, interop: 1")

# Memory optimization for low-RAM environments
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
print(f"[MEM] Initial memory: {get_memory_mb():.0f} MB")

WORK_DIR = os.environ.get('WORK_DIR', os.getcwd())
TMP_DIR = os.path.join(WORK_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(WORK_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
SHELVE_FILE_PATH = os.path.join(WORK_DIR, "videos_db")

## Video storage — rclone uploads to Google Drive from VPS
## After rendering, rclone copies the video to Drive and returns a public URL
## Job 5 (n8n) only needs to read the drive_url from status, no file transfer needed

RCLONE_REMOTE = os.environ.get("RCLONE_REMOTE", "gdrive")
RCLONE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID", "")

def rclone_upload_video(local_path, filename, folder_id=None):
    """Upload video to Google Drive via rclone. Returns public URL or None."""
    target_folder = folder_id or RCLONE_FOLDER_ID
    if not target_folder:
        print("[RCLONE] No GDRIVE_FOLDER_ID configured, skipping upload")
        return None
    
    try:
        import subprocess as sp
        file_size = os.path.getsize(local_path) / 1024 / 1024
        print(f"[RCLONE] Uploading {filename} ({file_size:.1f} MB) to folder {target_folder}...")
        
        # Step 1: Copy file to Google Drive
        result = sp.run([
            "rclone", "copy",
            local_path,
            f"{RCLONE_REMOTE}:",
            "--drive-root-folder-id", target_folder,
            "--drive-acknowledge-abuse",
            "--progress",
            "--stats-one-line",
        ], capture_output=True, text=True, timeout=1800)  # 30 min max
        
        if result.returncode != 0:
            print(f"[RCLONE] Upload failed: {result.stderr[-300:]}")
            return None
        
        print(f"[RCLONE] Upload complete, getting public link...")
        
        # Step 2: Get public link
        link_result = sp.run([
            "rclone", "link",
            f"{RCLONE_REMOTE}:{filename}",
            "--drive-root-folder-id", target_folder,
        ], capture_output=True, text=True, timeout=30)
        
        if link_result.returncode == 0 and link_result.stdout.strip():
            public_url = link_result.stdout.strip()
            print(f"[RCLONE] Public URL: {public_url}")
            return public_url
        else:
            # Fallback: list files to find the ID
            list_result = sp.run([
                "rclone", "lsjson",
                f"{RCLONE_REMOTE}:",
                "--drive-root-folder-id", target_folder,
                "--no-modtime",
                "-f", f"+ {filename}",
                "-f", "- *",
            ], capture_output=True, text=True, timeout=30)
            
            if list_result.returncode == 0:
                import json as _json
                files = _json.loads(list_result.stdout)
                if files:
                    file_id = files[0].get("ID", "")
                    if file_id:
                        url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                        print(f"[RCLONE] Constructed URL: {url}")
                        return url
            
            print(f"[RCLONE] Could not get link: {link_result.stderr[:200]}")
            return None
    
    except Exception as e:
        print(f"[RCLONE] Error: {e}")
        return None

def setup_rclone():
    """Install rclone if needed and configure from environment variables."""
    import subprocess as sp
    
    # Check if already installed
    try:
        result = sp.run(["rclone", "version"], capture_output=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.decode().split('\n')[0] if result.stdout else 'unknown'
            print(f"[RCLONE] Already installed: {version}")
    except (FileNotFoundError, Exception):
        print("[RCLONE] Not found, installing...")
        try:
            install = sp.run(
                ["bash", "-c", "curl -s https://rclone.org/install.sh | bash"],
                capture_output=True, text=True, timeout=120
            )
            if install.returncode == 0:
                print("[RCLONE] Installed successfully")
            else:
                print(f"[RCLONE] Install failed: {install.stderr[-200:]}")
                return False
        except Exception as e:
            print(f"[RCLONE] Install error: {e}")
            return False
    
    # Write config from environment variables
    # Supports either RCLONE_CONFIG_GDRIVE_* env vars (native rclone)
    # or our GDRIVE_RCLONE_TOKEN env var
    rclone_token = os.environ.get("RCLONE_DRIVE_TOKEN", "")
    rclone_client_id = os.environ.get("RCLONE_DRIVE_CLIENT_ID", "")
    rclone_client_secret = os.environ.get("RCLONE_DRIVE_CLIENT_SECRET", "")
    
    if rclone_token:
        config_dir = os.path.expanduser("~/.config/rclone")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "rclone.conf")
        
        config_content = f"""[{RCLONE_REMOTE}]
type = drive
client_id = {rclone_client_id}
client_secret = {rclone_client_secret}
scope = drive
token = {rclone_token}
team_drive = 
"""
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"[RCLONE] Config written to {config_path}")
    else:
        # Check if config already exists (mounted volume or pre-installed)
        config_path = os.path.expanduser("~/.config/rclone/rclone.conf")
        if os.path.exists(config_path):
            print(f"[RCLONE] Using existing config: {config_path}")
        else:
            print("[RCLONE] No GDRIVE_RCLONE_TOKEN env var and no config file found")
            print("[RCLONE] Set GDRIVE_RCLONE_TOKEN with the token JSON from rclone.conf")
            return False
    
    # Verify it works
    try:
        test = sp.run(
            ["rclone", "about", f"{RCLONE_REMOTE}:", "--json"],
            capture_output=True, text=True, timeout=15
        )
        if test.returncode == 0:
            print(f"[RCLONE] Connection verified ✓")
            return True
        else:
            print(f"[RCLONE] Connection test failed: {test.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[RCLONE] Connection test error: {e}")
        return False

def rclone_available():
    """Check if rclone is installed and configured."""
    try:
        import subprocess as sp
        result = sp.run(["rclone", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except:
        return False

# Auto-setup rclone on startup
_rclone_ok = setup_rclone()
if _rclone_ok:
    print(f"[RCLONE] Ready — remote: {RCLONE_REMOTE}, folder: {RCLONE_FOLDER_ID or '(not set)'}")
else:
    print("[RCLONE] Not available — videos will stay local")

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def iterfile(path: str):
    with open(path, mode="rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            yield chunk

LANGUAGE_CONFIG = {
    'en-us': {'lang_code': 'a', 'international': False},
    'en': {'lang_code': 'a', 'international': False},
    'en-gb': {'lang_code': 'b', 'international': False},
    'es': {'lang_code': 'e', 'international': True},
    'fr': {'lang_code': 'f', 'international': True},
    'hi': {'lang_code': 'h', 'international': True},
    'it': {'lang_code': 'i', 'international': True},
    'pt': {'lang_code': 'p', 'international': True},
    'ja': {'lang_code': 'j', 'international': True},
    'zh': {'lang_code': 'z', 'international': True},
}
LANGUAGE_VOICE_CONFIG = {
    'en-us': ['af_heart','af_alloy','af_aoede','af_bella','af_jessica','af_kore','af_nicole','af_nova','af_river','af_sarah','af_sky','am_adam','am_echo','am_eric','am_fenrir','am_liam','am_michael','am_onyx','am_puck','am_santa'],
    'en-gb': ['bf_alice','bf_emma','bf_isabella','bf_lily','bm_daniel','bm_fable','bm_george','bm_lewis'],
    'zh': ['zf_xiaobei','zf_xiaoni','zf_xiaoxiao','zf_xiaoyi','zm_yunjian','zm_yunxi','zm_yunxia','zm_yunyang'],
    'es': ['ef_dora', 'em_alex', 'em_santa'],
    'fr': ['ff_siwis'],
    'it': ['if_sara', 'im_nicola'],
    'pt': ['pf_dora', 'pm_alex', 'pm_santa'],
    'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
}

LANGUAGE_VOICE_MAP = {}
for lang, voices in LANGUAGE_VOICE_CONFIG.items():
    for voice in voices:
        if lang in LANGUAGE_CONFIG:
            LANGUAGE_VOICE_MAP[voice] = LANGUAGE_CONFIG[lang]
        else:
            print(f"Warning: Language {lang} not found in LANGUAGE_CONFIG")

def signal_handler(sig, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_videos()
    worker_thread = threading.Thread(target=process_video_queue, daemon=True)
    worker_thread.start()
    yield
    global worker_running
    worker_running = False
    if worker_thread.is_alive():
        worker_thread.join(timeout=1.0)
    save_videos()

app = FastAPI(lifespan=lifespan)

# CORS — allow Pipeline Manager and any frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

mcp = FastMCP(name="NarratedStoryMakerMCP", stateless_http=True)
active_connections = set()

class VideoStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"
    NOT_FOUND = "not_found"

AvailableVoices = Enum('Voice', {
    voice.upper().replace('_', '-'): voice
    for lang in LANGUAGE_VOICE_CONFIG
    for voice in LANGUAGE_VOICE_CONFIG[lang]
})

def load_videos():
    global videos
    try:
        with shelve.open(SHELVE_FILE_PATH) as db:
            if 'videos' in db:
                videos = db['videos']
                print(f"Loaded {len(videos)} videos from persistent storage")
                for video_id, video_data in videos.items():
                    if video_data['status'] == VideoStatus.QUEUED:
                        video_queue.put(video_id)
                    elif video_data['status'] == VideoStatus.PROCESSING:
                        video_data['status'] = VideoStatus.QUEUED
                        video_queue.put(video_id)
    except Exception as e:
        print(f"Error loading videos from persistent storage: {e}")

def save_videos():
    try:
        with shelve.open(SHELVE_FILE_PATH) as db:
            db['videos'] = videos
            print(f"Saved {len(videos)} videos to persistent storage")
    except Exception as e:
        print(f"Error saving videos to persistent storage: {e}")

atexit.register(save_videos)

video_queue = queue.Queue()
videos = {}
worker_lock = threading.Lock()
worker_running = True

def process_video_queue():
    while worker_running:
        try:
            if not video_queue.empty():
                video_id = video_queue.get()
                if video_id in videos:
                    videos[video_id]["status"] = VideoStatus.PROCESSING
                    save_videos()
                    data = videos[video_id]["data"]
                    video_dir = os.path.join(TMP_DIR, video_id)
                    os.makedirs(video_dir, exist_ok=True)
                    
                    try:
                        # Download background video
                        print(f"Downloading background video for {video_id}")
                        download_url = normalize_drive_url(data["bg_video_url"])
                        bg_extension = os.path.splitext(download_url.split('?')[0])[1]
                        if not bg_extension or len(bg_extension) > 5:
                            bg_extension = ".mp4"
                        bg_video_path = os.path.join(video_dir, f"background{bg_extension}")
                        response = requests.get(download_url, stream=True, timeout=120, allow_redirects=True)
                        if response.status_code == 200:
                            with open(bg_video_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        else:
                            raise Exception(f"Failed to download background video: {response.status_code}")
                        
                        # Download person image
                        print(f"Downloading person image for {video_id}")
                        person_download_url = normalize_drive_url(data["person_image_url"])
                        person_extension = os.path.splitext(person_download_url.split('?')[0])[1]
                        if not person_extension or len(person_extension) > 5:
                            person_extension = ".png"
                        person_image_path = os.path.join(video_dir, f"person{person_extension}")
                        response = requests.get(person_download_url, stream=True, timeout=60, allow_redirects=True)
                        if response.status_code == 200:
                            with open(person_image_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        else:
                            raise Exception(f"Failed to download person image: {response.status_code}")
                    except Exception as download_error:
                        try:
                            shutil.rmtree(video_dir)
                        except:
                            pass
                        raise Exception(f"Download failed: {download_error}")
                    
                    overlay_path = os.path.join(video_dir, "overlay.png")
                    print("creating overlay")
                    font_path = "assets/noto.ttf"
                    if LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"] == "h":
                        font_path = "assets/noto_hindi.ttf"
                    
                    display_name = data.get("person_name") or "Narrator"
                    
                    create_overlay(
                        person_image_path=person_image_path,
                        volume_icon_path="assets/icon_volume.png",
                        display_name=display_name,
                        output_path=overlay_path,
                        subtitle_background_color=(0, 0, 0, 200),
                        font_path=font_path,
                    )
                    
                    print("creating narration")
                    # Free memory before heavy TTS processing
                    gc.collect()
                    print(f"[MEM] Before TTS: {get_memory_mb():.0f} MB")
                    sound_path = os.path.join(video_dir, "sound.wav")
                    segments = []
                    is_international = LANGUAGE_VOICE_MAP[data["voice"]]["international"]
                    text_len = len(data["text"])
                    use_chunked = text_len > TTS_CHUNK_CHARS
                    
                    if use_chunked:
                        print(f"[TTS] Large text ({text_len} chars), using chunked processing")
                        captions, audio_length = create_tts_chunked(
                            text=data["text"], output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                            is_international=is_international,
                        )
                    elif is_international:
                        captions, audio_length = create_tts_international(
                            text=data["text"], output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                        )
                    else:
                        captions, audio_length = create_tts_english(
                            text=data["text"], output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                        )
                    
                    # If captions came from chunked TTS, normalize to standard format.
                    # Downstream code uses BOTH cap['start'] and cap.start, so we need
                    # a hybrid that supports both access patterns.
                    if use_chunked and captions:
                        sample = captions[0]
                        print(f"[CAPTIONS] Raw type: {type(sample).__name__} | Sample: {repr(sample)[:200]}")
                        if isinstance(sample, dict):
                            print(f"[CAPTIONS] Dict keys: {list(sample.keys())}")
                        
                        # Hybrid dict that supports both cap['key'] and cap.key
                        class Cap(dict):
                            __getattr__ = dict.__getitem__
                            __setattr__ = dict.__setitem__
                        
                        normalized = []
                        for cap in captions:
                            if isinstance(cap, dict):
                                s = cap.get('start', cap.get('start_ts', cap.get('start_time', cap.get('s', 0))))
                                e = cap.get('end', cap.get('end_ts', cap.get('end_time', cap.get('e', 0))))
                                w = cap.get('word', cap.get('text', cap.get('w', '')))
                                normalized.append(Cap(start=float(s), end=float(e), word=str(w), text=str(w),
                                                      start_ts=float(s), end_ts=float(e)))
                            elif isinstance(cap, (list, tuple)) and len(cap) >= 2:
                                w = str(cap[2]) if len(cap) > 2 else ""
                                normalized.append(Cap(start=float(cap[0]), end=float(cap[1]), word=w, text=w,
                                                      start_ts=float(cap[0]), end_ts=float(cap[1])))
                            elif hasattr(cap, 'start') and hasattr(cap, 'end'):
                                normalized.append(cap)
                            else:
                                print(f"[CAPTIONS] WARNING: Unknown: {type(cap).__name__} = {repr(cap)[:100]}")
                        
                        captions = normalized
                        print(f"[CAPTIONS] Normalized {len(captions)} captions")
                        if len(captions) > 1:
                            print(f"[CAPTIONS] First: start={captions[0].start:.2f}s word='{captions[0].word}'")
                            print(f"[CAPTIONS] Last: start={captions[-1].start:.2f}s end={captions[-1].end:.2f}s word='{captions[-1].word}'")
                    
                    if is_international:
                        max_line_length = 30
                        if LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"] == "z":
                            max_line_length = 15
                        segments = create_subtitle_segments_international(
                            captions=captions, max_length=max_line_length, lines=2,
                        )
                    else:
                        segments = create_subtitle_segments_english(
                            captions=captions, max_length=30, lines=2
                        )
                    
                    version = data.get("version", "v1")
                    subtitle_path = os.path.join(video_dir, "subtitle.ass")
                    print(f"Creating subtitle (version: {version})")
                    
                    if version == "v2":
                        print("Using v2 karaoke subtitle style")
                        create_subtitle_v2_karaoke(
                            word_captions=captions, font_size=80, output_path=subtitle_path,
                        )
                    else:
                        print("Using v1 static subtitle style")
                        create_subtitle(
                            segments=segments, font_size=80, output_path=subtitle_path,
                        )
                    
                    # Background video loop is handled by concat demuxer in render_video
                    # No pre-trimming needed
                    
                    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
                    print("rendering video")
                    render_video(
                        sound_path=sound_path, subtitle_path=subtitle_path,
                        overlay_path=overlay_path, audio_length=audio_length,
                        bg_video_path=bg_video_path, output_path=video_path,
                    )
                    
                    try:
                        print(f"Cleaning up temporary files for video: {video_id}")
                        shutil.rmtree(video_dir)
                    except Exception as cleanup_error:
                        print(f"Warning: Failed to clean up: {cleanup_error}")
                    
                    # Upload to Google Drive via rclone if available
                    drive_url = None
                    if rclone_available() and RCLONE_FOLDER_ID:
                        # Use per-channel folder if specified, otherwise global
                        folder_id = data.get("gdrive_folder_id") or RCLONE_FOLDER_ID
                        drive_url = rclone_upload_video(video_path, f"{video_id}.mp4", folder_id=folder_id)
                        if drive_url:
                            videos[video_id]["drive_url"] = drive_url
                            # Delete local file to save disk
                            try:
                                os.remove(video_path)
                                print(f"[DISK] Deleted local: {video_path}")
                            except:
                                pass
                    
                    videos[video_id]["status"] = VideoStatus.COMPLETED
                    save_videos()
                    gc.collect()
                    print(f"Completed video: {video_id} | Storage: {('Drive: ' + drive_url) if drive_url else 'local'}")
                
                video_queue.task_done()
            else:
                time.sleep(0.5)
        except Exception as e:
            print(f"Error in worker thread: {e}")
            if 'video_id' in locals() and video_id in videos:
                videos[video_id]["status"] = VideoStatus.FAILED
                videos[video_id]["error"] = str(e)
                save_videos()
                try:
                    if 'video_dir' in locals():
                        shutil.rmtree(video_dir)
                except:
                    pass

load_videos()
worker_thread = threading.Thread(target=process_video_queue, daemon=True)

### REST API ###
@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.get("/api/languages")
def get_languages():
    return LANGUAGE_VOICE_CONFIG

@app.get("/api/videos")
def list_videos():
    return [{"video_id": vid, "status": vd["status"]} for vid, vd in videos.items()]

@app.post("/api/videos")
def create_video(video: dict):
    text_len = len(video.get("text", ""))
    print(f"[API] POST /api/videos received — text: {text_len} chars, version: {video.get('version', '?')}")
    version = video.get("version", "v1")
    voice = video.get("voice", "af_heart")
    overlay_bg_color = video.get("overlay_bg_color", (232, 14, 64))
    bg_video_url = video.get("bg_video_url", "")
    if not bg_video_url:
        return {"error": "bg_video_url is required"}
    
    print(f"[{version}] Creating video with background: {bg_video_url}")
    
    video_id, video_data, error = process_video_request(
        text=video.get("text", ""),
        person_image_url=video.get("person_image_url", ""),
        person_name=video.get("person_name", ""),
        bg_video_url=bg_video_url,
        voice=voice,
        overlay_bg_color=overlay_bg_color,
        version=version,
        gdrive_folder_id=video.get("gdrive_folder_id", "")
    )
    
    if error:
        return {"error": error}
    
    videos[video_id] = video_data
    save_videos()
    video_queue.put(video_id)
    return {"video_id": video_id, "status": VideoStatus.QUEUED}

@app.get("/api/videos/{video_id}/status")
def get_video(video_id: str):
    if video_id in videos:
        result = {"video_id": video_id, "status": videos[video_id]["status"]}
        if videos[video_id]["status"] == VideoStatus.COMPLETED:
            drive_url = videos[video_id].get("drive_url")
            if drive_url:
                result["video_url"] = drive_url
                result["storage"] = "drive"
            else:
                video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
                if os.path.exists(video_path):
                    result["video_url"] = f"/api/videos/{video_id}"
                    result["size_mb"] = round(os.path.getsize(video_path) / 1024 / 1024, 1)
                    result["storage"] = "local"
                else:
                    result["video_url"] = None
                    result["note"] = "File already cleaned up"
        return result
    return {"video_id": video_id, "status": "not_found"}

@app.get("/api/videos/{video_id}")
def download_video(video_id: str, download: bool = False):
    if video_id in videos and videos[video_id]["status"] == VideoStatus.COMPLETED:
        # If on Google Drive, redirect
        drive_url = videos[video_id].get("drive_url")
        if drive_url:
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=drive_url)
        # Otherwise serve local file
        video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return StreamingResponse(
                iterfile(video_path), media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{video_id}.mp4"'}
            )
        return JSONResponse(content={"video_id": video_id, "status": "file_cleaned"}, status_code=status.HTTP_410_GONE)
    elif video_id in videos:
        if videos[video_id]["status"] == VideoStatus.FAILED:
            return JSONResponse(content={"video_id": video_id, "status": VideoStatus.FAILED}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        if videos[video_id]["status"] == VideoStatus.PROCESSING:
            return JSONResponse(content={"video_id": video_id, "status": VideoStatus.PROCESSING}, status_code=status.HTTP_202_ACCEPTED)
    return JSONResponse(content={"video_id": video_id, "status": VideoStatus.NOT_FOUND}, status_code=status.HTTP_404_NOT_FOUND)

@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    """Delete video file and metadata. Called by n8n after uploading to Google Drive."""
    if video_id in videos:
        video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        freed = 0
        if os.path.exists(video_path):
            freed = os.path.getsize(video_path)
            os.remove(video_path)
            print(f"[DISK] Deleted {video_id}: {freed / 1024 / 1024:.1f} MB freed")
        del videos[video_id]
        save_videos()
        return {"video_id": video_id, "status": VideoStatus.DELETED, "freed_mb": round(freed / 1024 / 1024, 1)}
    return {"video_id": video_id, "status": VideoStatus.NOT_FOUND}

@app.get("/api/disk")
def disk_status():
    """Show disk usage for video storage."""
    entries = []
    total_size = 0
    drive_count = 0
    for vid, data in videos.items():
        path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        entry = {"video_id": vid, "status": data["status"], "local": exists, "size_mb": round(size / 1024 / 1024, 1)}
        if data.get("drive_url"):
            entry["drive_url"] = data["drive_url"]
            drive_count += 1
        entries.append(entry)
        total_size += size
    return {
        "total_videos": len(entries),
        "on_drive": drive_count,
        "local_only": len(entries) - drive_count,
        "total_local_mb": round(total_size / 1024 / 1024, 1),
        "rclone_available": rclone_available(),
        "videos": entries
    }

@app.post("/api/disk/upload-to-drive")
def upload_local_to_drive(video_id: str = None):
    """Manually upload local video(s) to Google Drive via rclone."""
    if not rclone_available():
        return JSONResponse(content={"error": "rclone not available"}, status_code=503)
    if not RCLONE_FOLDER_ID:
        return JSONResponse(content={"error": "GDRIVE_FOLDER_ID not set"}, status_code=503)
    
    results = []
    targets = []
    
    if video_id:
        if video_id in videos:
            targets.append(video_id)
        else:
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
    else:
        # All local completed videos without drive_url
        targets = [vid for vid, data in videos.items() 
                   if data["status"] == VideoStatus.COMPLETED and not data.get("drive_url")]
    
    for vid in targets:
        video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            results.append({"video_id": vid, "status": "file_missing"})
            continue
        
        size_mb = os.path.getsize(video_path) / 1024 / 1024
        folder_id = videos[vid].get("data", {}).get("gdrive_folder_id") or RCLONE_FOLDER_ID
        drive_url = rclone_upload_video(video_path, f"{vid}.mp4", folder_id=folder_id)
        if drive_url:
            videos[vid]["drive_url"] = drive_url
            save_videos()
            try:
                os.remove(video_path)
            except:
                pass
            results.append({"video_id": vid, "status": "uploaded", "drive_url": drive_url, "freed_mb": round(size_mb, 1)})
        else:
            results.append({"video_id": vid, "status": "upload_failed"})
    
    return {"uploaded": len([r for r in results if r["status"] == "uploaded"]), "results": results}

@app.get("/api/queue")
def get_queue_status():
    return {
        "queue_size": video_queue.qsize(),
        "queued": len([v for v in videos.values() if v["status"] == VideoStatus.QUEUED]),
        "processing": len([v for v in videos.values() if v["status"] == VideoStatus.PROCESSING])
    }

@app.get("/api/diagnostics")
def run_diagnostics():
    """Quick health check of all subsystems."""
    results = {}
    
    # 1. Server health
    results["server"] = {"status": "ok", "memory_mb": round(get_memory_mb(), 0)}
    
    # 2. Rclone
    results["rclone"] = {
        "installed": rclone_available(),
        "folder_id": RCLONE_FOLDER_ID or "(not set)"
    }
    if rclone_available() and RCLONE_FOLDER_ID:
        try:
            import subprocess as sp
            test = sp.run(
                ["rclone", "lsd", f"{RCLONE_REMOTE}:", "--drive-root-folder-id", RCLONE_FOLDER_ID],
                capture_output=True, text=True, timeout=15
            )
            results["rclone"]["connection"] = "ok" if test.returncode == 0 else f"error: {test.stderr[:100]}"
        except Exception as e:
            results["rclone"]["connection"] = f"error: {e}"
    
    # 3. Supabase Storage (for ImageFX)
    supabase_url = os.environ.get("SUPABASE_URL", "")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    results["supabase_storage"] = {
        "configured": bool(supabase_url and supabase_key),
        "url": supabase_url[:40] + "..." if supabase_url else "(not set)"
    }
    if supabase_url and supabase_key:
        try:
            # Actually test the connection — list buckets
            resp = requests.get(
                f"{supabase_url}/storage/v1/bucket",
                headers={"Authorization": f"Bearer {supabase_key}"},
                timeout=10
            )
            if resp.status_code == 200:
                buckets = [b["name"] for b in resp.json()]
                has_imagefx = "imagefx" in buckets
                results["supabase_storage"]["connection"] = "ok"
                results["supabase_storage"]["buckets"] = buckets
                results["supabase_storage"]["imagefx_bucket"] = "exists" if has_imagefx else "MISSING — create bucket 'imagefx' (public)"
                
                # If bucket exists, count files
                if has_imagefx:
                    try:
                        files_resp = requests.post(
                            f"{supabase_url}/storage/v1/object/list/{IMAGEFX_BUCKET}",
                            headers={"Authorization": f"Bearer {supabase_key}", "Content-Type": "application/json"},
                            json={"limit": 1, "offset": 0, "prefix": ""},
                            timeout=10
                        )
                        if files_resp.status_code == 200:
                            results["supabase_storage"]["files_sample"] = len(files_resp.json())
                    except:
                        pass
            else:
                results["supabase_storage"]["connection"] = f"error: {resp.status_code} {resp.text[:100]}"
        except requests.exceptions.ConnectionError as e:
            results["supabase_storage"]["connection"] = f"DNS/network error: {str(e)[:100]}"
        except Exception as e:
            results["supabase_storage"]["connection"] = f"error: {str(e)[:100]}"
    
    # 4. Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        results["disk"] = {
            "total_gb": round(total / (1024**3), 1),
            "used_gb": round(used / (1024**3), 1),
            "free_gb": round(free / (1024**3), 1),
            "used_pct": round(used / total * 100, 1)
        }
    except:
        results["disk"] = {"status": "error"}
    
    # 5. Videos summary
    statuses = {}
    for v in videos.values():
        s = v["status"]
        statuses[s] = statuses.get(s, 0) + 1
    results["videos"] = statuses
    
    # 6. TTS test (import only, no generation)
    try:
        import importlib
        vm = importlib.import_module("video_maker")
        results["tts"] = {"video_maker": "ok", "functions": [
            f for f in ["create_tts_english", "create_tts_international", "render_video", 
                       "create_subtitle_v2_karaoke", "create_overlay"]
            if hasattr(vm, f)
        ]}
    except Exception as e:
        results["tts"] = {"video_maker": f"error: {e}"}
    
    # 7. FFmpeg
    try:
        import subprocess as sp
        ff = sp.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        version_line = ff.stdout.split('\n')[0] if ff.stdout else "unknown"
        results["ffmpeg"] = {"status": "ok", "version": version_line}
    except:
        results["ffmpeg"] = {"status": "not found"}
    
    return results

@app.post("/api/test-video")
def create_test_video(params: dict = {}):
    """Create a test video to validate the full pipeline.
    Accepts: bg_video_url, gdrive_folder_id, person_image_url, voice, chunks (2 or 3)
    Tests: TTS, subtitle generation with timestamp offsets, overlay, rendering, rclone upload."""
    
    bg_video_url = params.get("bg_video_url", os.environ.get("TEST_BG_VIDEO_URL", ""))
    gdrive_folder_id = params.get("gdrive_folder_id", RCLONE_FOLDER_ID)
    person_image_url = params.get("person_image_url", "")
    voice = params.get("voice", "af_heart")
    num_chunks = int(params.get("chunks", 2))
    
    if not bg_video_url:
        return JSONResponse(content={"error": "bg_video_url is required"}, status_code=400)
    
    # Build test text that creates multiple TTS chunks
    # Each chunk ~1500 chars, so 2 chunks = ~3000 chars, 3 chunks = ~4500 chars
    chunk_texts = [
        # Chunk 1: ~1600 chars
        "This is the first part of the test video. We need enough text to fill an entire chunk of the text to speech system. The purpose of this test is to verify that the subtitle timestamps are correctly offset when multiple chunks are concatenated together. Each chunk processes independently and generates its own set of captions starting from zero seconds. The main process then adjusts the timestamps by adding the cumulative duration of all previous chunks. For example if chunk one is thirty seconds long then all timestamps in chunk two should be offset by thirty seconds. This ensures that the subtitles appear at the correct time in the final concatenated audio. Without this offset correction all subtitles would pile up at the beginning of the video which is exactly the bug we are testing for. Let us continue with more text to make sure this chunk is long enough. The quick brown fox jumps over the lazy dog. Testing one two three four five six seven eight nine ten. We are almost at the character limit for this chunk now. Just a few more sentences should do it. The weather today is perfect for testing video generation pipelines.",
        # Chunk 2: ~1600 chars
        "Now we are in the second chunk of the test video. If the subtitle system is working correctly you should see these words appearing after the first chunk has finished. The timestamps should flow naturally from where the first chunk ended. This is the critical test. If you see all the subtitles from both chunks appearing at the very beginning of the video then the timestamp offset is not working. But if the subtitles from this second chunk appear roughly halfway through the video then everything is working perfectly. Let us add some more content to make this chunk substantial enough for a proper test. Remember that the text to speech system processes each chunk in a separate subprocess to avoid memory issues. The subprocess loads the model processes the text and then exits freeing all memory. The parent process reads the generated audio and caption files from disk. It then adjusts the caption timestamps and concatenates all the audio chunks into one final file. This approach allows us to process very long scripts without running out of memory. Even a thirty minute video with over thirty thousand characters can be processed this way.",
        # Chunk 3: ~1600 chars (optional)
        "This is the third and final chunk of our test video. By now you should have seen subtitles flowing naturally through the video with no gaps or overlaps between chunks. The third chunk is an extra validation to make sure the cumulative offset works across multiple boundaries not just one. Some edge cases only appear with three or more chunks such as floating point precision issues or off by one errors in the timestamp calculations. If you have made it this far and the subtitles look correct then congratulations the chunked text to speech pipeline is working perfectly. The video should also loop the background smoothly without any freezing or stuttering at the loop point. And if rclone is configured the finished video should automatically upload to Google Drive and the local file should be deleted to save disk space. Thank you for running this test. The pipeline is healthy and ready to produce real content. End of test video generation. This has been a comprehensive validation of all major subsystems including text to speech subtitle generation video rendering and cloud storage upload."
    ]
    
    test_text = " ".join(chunk_texts[:num_chunks])
    
    # Use a default test person image if none provided
    if not person_image_url:
        # Try to find any character image from existing channels
        for vid_data in videos.values():
            d = vid_data.get("data", {})
            if d.get("person_image_url"):
                person_image_url = d["person_image_url"]
                break
    
    if not person_image_url:
        return JSONResponse(content={
            "error": "person_image_url required. Pass it in body or generate at least one video first."
        }, status_code=400)
    
    video_id, video_data, error = process_video_request(
        text=test_text,
        person_image_url=person_image_url,
        person_name="Test",
        bg_video_url=bg_video_url,
        voice=voice,
        version="v2",
        gdrive_folder_id=gdrive_folder_id
    )
    
    if error:
        return JSONResponse(content={"error": error}, status_code=400)
    
    videos[video_id] = video_data
    save_videos()
    video_queue.put(video_id)
    return {
        "video_id": video_id,
        "status": "queued",
        "test": True,
        "chunks_expected": num_chunks,
        "text_length": len(test_text),
        "note": f"~{num_chunks * 30}s video with {num_chunks} TTS chunks"
    }

### ImageFX (Google AI Image Generation) ###
# Based on: https://github.com/rohitaryal/imageFX-api
# Auth flow: cookie → labs.google/fx/api/auth/session → access_token → aisandbox API
IMAGEFX_API_URL = "https://aisandbox-pa.googleapis.com/v1:runImageFx"
IMAGEFX_SESSION_URL = "https://labs.google/fx/api/auth/session"
IMAGEFX_IMAGES_DIR = os.path.join(os.getcwd(), "imagefx_output")
os.makedirs(IMAGEFX_IMAGES_DIR, exist_ok=True)

# Supabase Storage for persistent image storage
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
IMAGEFX_BUCKET = "imagefx"

def ensure_supabase_bucket():
    """Create imagefx bucket in Supabase Storage if it doesn't exist."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False
    try:
        # Check if bucket exists
        resp = requests.get(
            f"{SUPABASE_URL}/storage/v1/bucket/{IMAGEFX_BUCKET}",
            headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
            timeout=10
        )
        if resp.status_code == 200:
            print(f"[ImageFX] Supabase bucket '{IMAGEFX_BUCKET}' exists")
            return True
        
        # Create bucket
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/bucket",
            headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}", "Content-Type": "application/json"},
            json={"id": IMAGEFX_BUCKET, "name": IMAGEFX_BUCKET, "public": True},
            timeout=10
        )
        if resp.status_code in (200, 201):
            print(f"[ImageFX] Created Supabase bucket '{IMAGEFX_BUCKET}' (public)")
            return True
        else:
            print(f"[ImageFX] Failed to create bucket: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"[ImageFX] Bucket check error: {e}")
        return False

# Try to create bucket on startup
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    ensure_supabase_bucket()
else:
    print("[ImageFX] Supabase Storage not configured — images will be local only")


def upload_to_supabase_storage(image_bytes: bytes, filename: str) -> str | None:
    """Upload image to Supabase Storage bucket. Returns public URL or None on failure."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[ImageFX] Supabase Storage not configured, skipping upload")
        return None
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{IMAGEFX_BUCKET}/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "image/png",
                "x-upsert": "true"
            },
            data=image_bytes,
            timeout=30
        )
        if resp.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{IMAGEFX_BUCKET}/{filename}"
            return public_url
        else:
            print(f"[ImageFX] Supabase upload failed: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[ImageFX] Supabase upload error: {e}")
        return None

IMAGEFX_DEFAULT_HEADERS = {
    "Origin": "https://labs.google",
    "Content-Type": "application/json",
    "Referer": "https://labs.google/fx/tools/image-fx",
}

ASPECT_RATIO_MAP = {
    "LANDSCAPE": "IMAGE_ASPECT_RATIO_LANDSCAPE",
    "PORTRAIT": "IMAGE_ASPECT_RATIO_PORTRAIT",
    "SQUARE": "IMAGE_ASPECT_RATIO_SQUARE",
    "LANDSCAPE_4_3": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
}


def imagefx_get_token(cookie: str) -> dict:
    """Exchange session cookie for access token via labs.google.
    
    Returns: { "access_token": "ya29...", "expires": "...", "user": {...} }
    Raises: Exception on failure
    """
    # Sanitize cookie: remove newlines, collapse spaces, trim
    cookie = cookie.replace("\r", "").replace("\n", " ").strip()
    cookie = re.sub(r'\s+', ' ', cookie)
    
    # Don't send Content-Type on GET — some servers reject it
    headers = {
        "Origin": "https://labs.google",
        "Referer": "https://labs.google/fx/tools/image-fx",
        "Cookie": cookie,
    }
    
    print(f"[ImageFX] Token exchange — cookie length: {len(cookie)} chars, first 80: {cookie[:80]}...")
    resp = requests.get(IMAGEFX_SESSION_URL, headers=headers, timeout=15)
    
    if not resp.ok:
        print(f"[ImageFX] Session failed: HTTP {resp.status_code} — {resp.text[:300]}")
        raise Exception(f"Session auth failed (HTTP {resp.status_code}): {resp.text[:300]}")
    
    data = resp.json()
    if not data.get("access_token") or not data.get("expires"):
        raise Exception(f"Session response missing access_token/expires. Keys: {list(data.keys())}")
    
    return data


@app.post("/api/generate-image")
def generate_imagefx(req: dict):
    """Generate an image using Google ImageFX API.
    
    Body: { "cookie": "<session cookie header string>", "prompt": "...", "aspect_ratio": "LANDSCAPE" }
    The "cookie" field should be the full cookie header string from labs.google (via Cookie Editor → Export → Header String).
    """
    cookie = req.get("cookie", "").replace("\r", "").replace("\n", " ").strip()
    cookie = re.sub(r'\s+', ' ', cookie)
    prompt = req.get("prompt", "").strip()
    aspect_ratio = req.get("aspect_ratio", "PORTRAIT").upper()
    num_images = req.get("num_images", 4)
    seed = req.get("seed", random.randint(1, 2**31))
    
    if not cookie:
        return JSONResponse(content={"error": "Cookie de sessão é obrigatório"}, status_code=400)
    if not prompt:
        return JSONResponse(content={"error": "Prompt é obrigatório"}, status_code=400)
    
    ar_value = ASPECT_RATIO_MAP.get(aspect_ratio, "IMAGE_ASPECT_RATIO_LANDSCAPE")
    
    # Step 1: Exchange cookie for access token
    try:
        session_data = imagefx_get_token(cookie)
        access_token = session_data["access_token"]
        print(f"[ImageFX] Got access token: {access_token[:20]}...")
    except Exception as e:
        return JSONResponse(
            content={"error": f"Falha na autenticação: {str(e)}", "auth_expired": True},
            status_code=401
        )
    
    # Step 2: Generate image
    payload = {
        "userInput": {
            "candidatesCount": num_images,
            "prompts": [prompt],
            "seed": seed
        },
        "clientContext": {
            "sessionId": f";{int(time.time() * 1000)}",
            "tool": "IMAGE_FX"
        },
        "modelInput": {
            "modelNameType": "IMAGEN_3_5"
        },
        "aspectRatio": ar_value
    }
    
    headers = {
        **IMAGEFX_DEFAULT_HEADERS,
        "Cookie": cookie,
        "Authorization": f"Bearer {access_token}",
    }
    
    try:
        resp = requests.post(IMAGEFX_API_URL, json=payload, headers=headers, timeout=60)
        
        if resp.status_code == 401 or resp.status_code == 403:
            return JSONResponse(
                content={"error": f"Token expirado ou inválido (HTTP {resp.status_code})", "auth_expired": True},
                status_code=resp.status_code
            )
        
        if resp.status_code != 200:
            return JSONResponse(
                content={"error": f"ImageFX API error: HTTP {resp.status_code}", "detail": resp.text[:500]},
                status_code=resp.status_code
            )
        
        data = resp.json()
        
        # Extract images from response
        images = []
        image_panels = data.get("imagePanels", [])
        for panel in image_panels:
            generated = panel.get("generatedImages", [])
            for img in generated:
                encoded = img.get("encodedImage", "")
                if encoded:
                    images.append(encoded)
        
        if not images:
            return JSONResponse(
                content={"error": "Nenhuma imagem retornada pelo ImageFX", "raw_keys": list(data.keys())},
                status_code=500
            )
        
        # Save ALL images (Supabase Storage for persistence + local cache)
        base_id = str(uuid.uuid4())[:12]
        saved_images = []
        
        for idx, encoded in enumerate(images):
            img_id = f"{base_id}_{idx}"
            img_bytes = base64.b64decode(encoded)
            
            # Local cache
            img_path = os.path.join(IMAGEFX_IMAGES_DIR, f"{img_id}.png")
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            
            # Upload to Supabase Storage (persistent)
            public_url = upload_to_supabase_storage(img_bytes, f"{img_id}.png")
            
            saved_images.append({
                "image_id": img_id,
                "image_url": public_url or f"/api/imagefx/{img_id}",
                "size_bytes": len(img_bytes)
            })
        
        primary = saved_images[0]
        storage_type = "supabase" if saved_images[0]["image_url"].startswith("http") else "local"
        print(f"[ImageFX] Generated {len(images)} images, saved as {base_id}_* ({storage_type})")
        
        return {
            "success": True,
            "image_id": primary["image_id"],
            "image_url": primary["image_url"],
            "total_generated": len(saved_images),
            "size_bytes": primary["size_bytes"],
            "all_images": [img["image_url"] for img in saved_images]
        }
        
    except requests.exceptions.Timeout:
        return JSONResponse(content={"error": "ImageFX API timeout (60s)"}, status_code=504)
    except Exception as e:
        return JSONResponse(content={"error": f"ImageFX error: {str(e)}"}, status_code=500)


@app.get("/api/imagefx/{image_id}")
def get_imagefx_image(image_id: str):
    """Serve a generated ImageFX image."""
    image_path = os.path.join(IMAGEFX_IMAGES_DIR, f"{image_id}.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png")
    return JSONResponse(content={"error": "Image not found"}, status_code=404)


@app.post("/api/test-imagefx")
def test_imagefx_token(req: dict):
    """Test if an ImageFX session cookie is valid.
    
    Body: { "cookie": "<session cookie header string>" }
    Returns: { "valid": true/false, "message": "...", "user": "..." }
    """
    cookie = req.get("cookie", "").replace("\r", "").replace("\n", " ").strip()
    cookie = re.sub(r'\s+', ' ', cookie)
    if not cookie:
        return JSONResponse(content={"valid": False, "message": "Cookie vazio"}, status_code=400)
    
    # Step 1: Try to get an access token from the cookie
    try:
        session_data = imagefx_get_token(cookie)
        access_token = session_data["access_token"]
        user_name = session_data.get("user", {}).get("name", "Unknown")
        user_email = session_data.get("user", {}).get("email", "")
        expires = session_data.get("expires", "")
    except Exception as e:
        return {"valid": False, "message": f"Autenticação falhou: {str(e)}"}
    
    # Step 2: Test a minimal image generation
    seed = random.randint(1, 2**31)
    payload = {
        "userInput": {
            "candidatesCount": 1,
            "prompts": ["a simple red circle on white background"],
            "seed": seed
        },
        "clientContext": {
            "sessionId": f";{int(time.time() * 1000)}",
            "tool": "IMAGE_FX"
        },
        "modelInput": {
            "modelNameType": "IMAGEN_3_5"
        },
        "aspectRatio": "IMAGE_ASPECT_RATIO_SQUARE"
    }
    
    headers = {
        **IMAGEFX_DEFAULT_HEADERS,
        "Cookie": cookie,
        "Authorization": f"Bearer {access_token}",
    }
    
    try:
        resp = requests.post(IMAGEFX_API_URL, json=payload, headers=headers, timeout=30)
        
        if resp.status_code == 401 or resp.status_code == 403:
            return {"valid": False, "message": f"Token aceito mas API rejeitou (HTTP {resp.status_code})"}
        
        if resp.status_code == 200:
            data = resp.json()
            has_images = any(
                img.get("encodedImage")
                for panel in data.get("imagePanels", [])
                for img in panel.get("generatedImages", [])
            )
            if has_images:
                return {
                    "valid": True,
                    "message": f"Cookie válido — imagem gerada com sucesso (user: {user_name})",
                    "user": user_name,
                    "email": user_email,
                    "expires": expires
                }
            else:
                return {"valid": True, "message": f"Cookie aceito mas sem imagens retornadas (user: {user_name})"}
        
        return {"valid": False, "message": f"Erro inesperado: HTTP {resp.status_code}", "detail": resp.text[:300]}
        
    except requests.exceptions.Timeout:
        return {"valid": False, "message": "Timeout ao testar (30s) — mas autenticação OK"}
    except Exception as e:
        return {"valid": False, "message": f"Erro: {str(e)}"}


### MCP Server ###
@mcp.tool()
def list_languages_mcp() -> dict:
    """List available languages and their voices."""
    return LANGUAGE_VOICE_CONFIG

@mcp.tool()
def create_video_mcp(
    text: Annotated[str, Field(description="The text to be narrated in the video.")],
    person_image_url: Annotated[str, Field(description="URL of the person's image.")],
    bg_video_url: Annotated[str, Field(description="URL of the background video.")],
    person_name: Annotated[Optional[str], Field(description="Name displayed in video.")] = "Narrator",
    voice: Annotated[Optional[str], Field(description="Voice for narration. Default: af_heart.")] = "af_heart",
    overlay_bg_color: Annotated[Optional[tuple], Field(description="Overlay color (R,G,B).")] = (232, 14, 64),
    version: Annotated[Optional[str], Field(description="'v1' static or 'v2' karaoke.")] = "v1"
) -> dict:
    """Create a new narrated video with the provided content."""
    print(f"Creating video with text: {text[:100]}...")
    voice_str = voice if voice else "af_heart"
    bg_color = overlay_bg_color if overlay_bg_color else (232, 14, 64)
    name = person_name if person_name else "Narrator"
    ver = version if version else "v1"
    
    video_id, video_data, error = process_video_request(
        text=text, person_image_url=person_image_url, person_name=name,
        bg_video_url=bg_video_url, voice=voice_str, overlay_bg_color=bg_color, version=ver
    )
    if error:
        return {"error": error}
    videos[video_id] = video_data
    save_videos()
    video_queue.put(video_id)
    return {"video_id": video_id, "status": VideoStatus.QUEUED.value}

sse = SseServerTransport("/mcp/messages/")
app.router.routes.append(Mount("/mcp/messages", app=sse.handle_post_message))

@app.get("/mcp/sse", tags=["MCP"])
async def handle_sse(request: Request):
    active_connections.add(request)
    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream, mcp._mcp_server.create_initialization_options())
    print("SSE connection closed")

def process_video_request(
    text: str, person_image_url: str, person_name: str, bg_video_url: str,
    voice: str = "af_heart", overlay_bg_color: tuple = (232, 14, 64), version: str = "v1",
    gdrive_folder_id: str = ""
) -> tuple[str, dict, str]:
    """Process video creation request."""
    if not text:
        return None, None, "Missing required field: text"
    if not person_image_url:
        return None, None, "Missing required field: person_image_url"
    if not bg_video_url:
        return None, None, "Missing required field: bg_video_url"
    if not person_name:
        person_name = "Narrator"
    if not bg_video_url.startswith("http"):
        return None, None, "Invalid bg_video_url: should start with http"
    if not person_image_url.startswith("http"):
        return None, None, "Invalid person_image_url: should start with http"
    
    # Trusted domains — skip HEAD validation (Google Drive doesn't handle HEAD well)
    TRUSTED_DOMAINS = ["drive.google.com", "googleapis.com", "supabase.co", "cloudflare", "easypanel.host"]

    # Check background video
    if not any(d in bg_video_url for d in TRUSTED_DOMAINS):
        try:
            response = requests.head(bg_video_url, timeout=10, allow_redirects=True)
            if response.status_code not in [200, 302, 303]:
                return None, None, f"Background video not accessible: {response.status_code}"
            ext = os.path.splitext(bg_video_url)[1].lower().split('?')[0]
            if ext and ext not in [".mp4", ".mov", ".avi", ".webm"]:
                return None, None, "Invalid bg_video_url: should be a video file"
        except Exception as e:
            return None, None, f"Error checking bg_video_url: {str(e)}"
    else:
        print(f"[VALIDATE] Skipping HEAD check for trusted domain: {bg_video_url[:60]}...")

    # Check person image
    if not any(d in person_image_url for d in TRUSTED_DOMAINS):
        try:
            response = requests.head(person_image_url, timeout=10, allow_redirects=True)
            if response.status_code not in [200, 302, 303]:
                return None, None, f"Person image not accessible: {response.status_code}"
            ext = os.path.splitext(person_image_url)[1].lower().split('?')[0]
            if ext and ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                return None, None, "Invalid person_image_url: should be an image file"
        except Exception as e:
            return None, None, f"Error checking person_image_url: {str(e)}"
    else:
        print(f"[VALIDATE] Skipping HEAD check for trusted domain: {person_image_url[:60]}...")
    
    if voice not in LANGUAGE_VOICE_MAP:
        return None, None, f"Invalid voice: {voice}. Available: {list(LANGUAGE_VOICE_MAP.keys())}"
    
    video_id = str(uuid.uuid4())
    video_data = {
        "id": video_id,
        "status": VideoStatus.QUEUED,
        "data": {
            "text": text, "person_name": person_name, "voice": voice,
            "overlay_bg_color": overlay_bg_color, "person_image_url": person_image_url,
            "bg_video_url": bg_video_url, "version": version,
            "gdrive_folder_id": gdrive_folder_id,
        },
        "created_at": time.time()
    }
    return video_id, video_data, ""

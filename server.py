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
from concurrent.futures import ThreadPoolExecutor
import shelve
import os
import atexit
import sys
import base64
import json
import random
import re
import shutil
import subprocess
import torch
torch.set_num_interop_threads(1)  # must be set before any parallel work starts

def normalize_text_for_tts(text: str) -> str:
    """Normalize text for TTS: convert symbols, currencies, and special chars to spoken words.
    
    Without this, TTS models like Kokoro choke on $12,000, €500, 50%, etc.
    causing audio to freeze and desync all subtitles.
    """
    import re as _re
    
    # Number pattern: proper comma-separated groups + optional decimal (NO trailing commas)
    # Matches: 12,000  |  1,234,567  |  20  |  1.2  |  3,500.50
    # Does NOT match: 20, (trailing comma)
    _NUM = r'\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
    
    # ── Currency with magnitude: $1.2 million → "1.2 million dollars" ──
    _MAGNITUDES = r'(?:million|billion|trillion|thousand|hundred|mil|milhão|milhões|bilhão|bilhões)'
    
    def currency_magnitude(match):
        prefix = match.group(1) or ''
        symbol = match.group(2)
        number = match.group(3)
        magnitude = match.group(4)
        names = {'$': 'dollars' if not prefix else 'reais', '€': 'euros', '£': 'pounds', '¥': 'yen'}
        return f"{number} {magnitude} {names.get(symbol, 'dollars')}"
    
    # R$ magnitude first, then other currencies
    text = _re.sub(r'(R)([$])\s?(' + _NUM + r')\s+(' + _MAGNITUDES + r')', currency_magnitude, text, flags=_re.IGNORECASE)
    text = _re.sub(r'()([$€£¥])\s?(' + _NUM + r')\s+(' + _MAGNITUDES + r')', currency_magnitude, text, flags=_re.IGNORECASE)
    
    # ── Currency with K/M/B abbreviation: $50K → "50 thousand dollars" ──
    _ABBREVS = {'k': 'thousand', 'K': 'thousand', 'm': 'million', 'M': 'million', 'b': 'billion', 'B': 'billion'}
    def currency_abbrev(match):
        prefix = match.group(1) or ''
        symbol = match.group(2)
        number = match.group(3)
        abbrev = match.group(4)
        names = {'$': 'dollars' if not prefix else 'reais', '€': 'euros', '£': 'pounds', '¥': 'yen'}
        magnitude = _ABBREVS.get(abbrev, abbrev)
        return f"{number} {magnitude} {names.get(symbol, 'dollars')}"
    
    text = _re.sub(r'(R)([$])\s?(' + _NUM + r')([kKmMbB])\b', currency_abbrev, text)
    text = _re.sub(r'()([$€£¥])\s?(' + _NUM + r')([kKmMbB])\b', currency_abbrev, text)
    
    # ── Regular currency: $12,000 → "12,000 dollars" ──
    def currency_to_words(match):
        prefix = match.group(1) or ''
        symbol = match.group(2)
        number = match.group(3)
        names = {'$': 'dollars' if not prefix else 'reais', '€': 'euros', '£': 'pounds', '¥': 'yen'}
        currency = names.get(symbol, 'dollars')
        try:
            if float(number.replace(',', '')) == 1:
                singular = {'dollars': 'dollar', 'euros': 'euro', 'pounds': 'pound', 'yen': 'yen', 'reais': 'real'}
                currency = singular.get(currency, currency)
        except: pass
        return f"{number} {currency}"
    
    text = _re.sub(r'(R)([$])\s?(' + _NUM + r')', currency_to_words, text)
    text = _re.sub(r'()([$€£¥])\s?(' + _NUM + r')', currency_to_words, text)
    
    # Currency AFTER number: 100$ → "100 dollars"
    text = _re.sub(r'(' + _NUM + r')\s?[$]', r'\1 dollars', text)
    text = _re.sub(r'(' + _NUM + r')\s?[€]', r'\1 euros', text)
    text = _re.sub(r'(' + _NUM + r')\s?[£]', r'\1 pounds', text)
    
    # ── Clean up any remaining currency symbols not caught by patterns ──
    text = text.replace('$', ' dollars ')
    text = text.replace('€', ' euros ')
    text = text.replace('£', ' pounds ')
    text = text.replace('¥', ' yen ')
    
    # ── Percent ──
    text = _re.sub(r'(\d+\.?\d*)\s?%', r'\1 percent', text)
    
    # ── Common symbols ──
    text = text.replace(' & ', ' and ')
    text = text.replace('&', ' and ')
    text = _re.sub(r'#(\d+)', r'number \1', text)
    text = text.replace('#', '')
    text = text.replace('@', ' at ')
    text = text.replace('…', '...')
    text = text.replace('—', ', ')
    text = text.replace('–', ', ')
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    # ── Standalone symbols that confuse TTS ──
    text = text.replace('*', '')
    text = text.replace('~', '')
    text = text.replace('`', '')
    text = text.replace('^', '')
    
    # ── Collapse multiple spaces ──
    text = _re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_drive_url(url: str) -> str:
    """Convert Google Drive share/view links to direct download URLs.
    
    Google changed their download endpoint — the old drive.google.com/uc path
    returns 404 for many files now. The new endpoint is:
    https://drive.usercontent.google.com/download?id=FILE_ID&export=download&confirm=t
    """
    if not url:
        return url
    
    # Extract file ID from various Drive URL formats
    file_id = None
    
    # Format: drive.google.com/file/d/FILE_ID/view
    match = re.search(r'drive\.google\.com/file/d/([^/?]+)', url)
    if match:
        file_id = match.group(1)
    
    # Format: drive.google.com/uc?...id=FILE_ID
    if not file_id:
        match = re.search(r'drive\.google\.com/uc\?.*?id=([^&]+)', url)
        if match:
            file_id = match.group(1)
    
    # Format: drive.usercontent.google.com/download?id=FILE_ID
    if not file_id:
        match = re.search(r'drive\.usercontent\.google\.com/download\?.*?id=([^&]+)', url)
        if match:
            file_id = match.group(1)
    
    # Format: lh3.googleusercontent.com/d/FILE_ID
    if not file_id:
        match = re.search(r'lh3\.googleusercontent\.com/d/([^/?]+)', url)
        if match:
            file_id = match.group(1)
    
    if file_id:
        # New endpoint (2024+): drive.usercontent.google.com
        return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    
    return url


def extract_drive_file_id(url: str) -> str:
    """Extract Google Drive file ID from various URL formats."""
    if not url:
        return None
    for pattern in [
        r'drive\.google\.com/file/d/([^/?]+)',
        r'drive\.usercontent\.google\.com/download\?.*?id=([^&]+)',
        r'drive\.google\.com/uc\?.*?id=([^&]+)',
        r'id=([a-zA-Z0-9_-]{20,})',
        r'lh3\.googleusercontent\.com/d/([^/?]+)',
    ]:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_drive_file(url: str, output_path: str, timeout: int = 120) -> bool:
    """Download a file from Google Drive.
    
    Priority:
    1. rclone (already authenticated, most reliable)
    2. HTTP fallback with multiple endpoints
    
    Returns True on success, raises Exception on failure.
    """
    import subprocess as sp
    
    file_id = extract_drive_file_id(url)
    last_error = None
    
    # ── Method 1: rclone (preferred — already authenticated, no URL issues) ──
    if file_id and rclone_available():
        try:
            print(f"[DOWNLOAD] rclone copyto by file ID: {file_id}")
            # rclone backend copyid copies a file by its Drive ID
            result = sp.run([
                "rclone", "backend", "copyid",
                f"{RCLONE_REMOTE}:",
                file_id,
                output_path,
            ], capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100:
                size = os.path.getsize(output_path)
                print(f"[DOWNLOAD] rclone success: {size / 1024:.0f} KB")
                return True
            else:
                err = result.stderr[-200:] if result.stderr else "unknown"
                print(f"[DOWNLOAD] rclone copyid failed (rc={result.returncode}): {err}")
                last_error = f"rclone copyid: {err}"
                
                # Fallback: rclone copy with --drive-files-only flag
                # Create temp dir, copy into it, then move
                import tempfile
                tmp_dir = tempfile.mkdtemp()
                try:
                    result2 = sp.run([
                        "rclone", "copy",
                        f"{RCLONE_REMOTE}:{{{{ {file_id} }}}}",
                        tmp_dir,
                    ], capture_output=True, text=True, timeout=timeout)
                    
                    # Find any file in tmp_dir
                    for f in os.listdir(tmp_dir):
                        tmp_file = os.path.join(tmp_dir, f)
                        if os.path.getsize(tmp_file) > 100:
                            shutil.move(tmp_file, output_path)
                            size = os.path.getsize(output_path)
                            print(f"[DOWNLOAD] rclone copy fallback success: {size / 1024:.0f} KB")
                            return True
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                
        except Exception as e:
            last_error = str(e)
            print(f"[DOWNLOAD] rclone error: {e}")
    
    # ── Method 2: HTTP with multiple endpoint fallbacks ──
    urls_to_try = []
    if file_id:
        urls_to_try = [
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
            f"https://lh3.googleusercontent.com/d/{file_id}",
        ]
    elif url:
        urls_to_try = [url]
    
    for try_url in urls_to_try:
        try:
            print(f"[DOWNLOAD] HTTP trying: {try_url[:80]}...")
            response = requests.get(try_url, stream=True, timeout=timeout, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    first_chunk = next(response.iter_content(chunk_size=1024), b'')
                    if b'<!DOCTYPE' in first_chunk or b'<html' in first_chunk:
                        print(f"[DOWNLOAD] Got HTML page, skipping...")
                        last_error = "Got HTML instead of file"
                        continue
                    with open(output_path, 'wb') as f:
                        f.write(first_chunk)
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                
                size = os.path.getsize(output_path)
                if size < 100:
                    last_error = f"File too small: {size} bytes"
                    print(f"[DOWNLOAD] File too small ({size}B), skipping...")
                    continue
                
                print(f"[DOWNLOAD] HTTP success: {size / 1024:.0f} KB")
                return True
            else:
                last_error = f"HTTP {response.status_code}"
                print(f"[DOWNLOAD] HTTP {response.status_code}")
        except Exception as e:
            last_error = str(e)
            print(f"[DOWNLOAD] HTTP error: {e}")
    
    raise Exception(f"All download methods failed for {url[:80]} (id={file_id}) — {last_error}")
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount
from video_maker import (
    create_overlay,
    create_overlay_v3,
    create_overlay_v4,
    create_tts_international,
    create_tts_english,
    create_subtitle_segments_international,
    create_subtitle_segments_english,
    create_subtitle,
    create_subtitle_v2_karaoke,
    create_subtitle_v4_karaoke,
    render_video,
    render_video_v4,
    COLOR_PRESETS,
)
import shutil
import gc
import wave

# ═══════════════════════════════
# CHUNKED TTS — Process large texts in pieces to avoid OOM
# ═══════════════════════════════

TTS_CHUNK_CHARS = int(os.environ.get("TTS_CHUNK_CHARS", "1500"))  # Max chars per TTS chunk
FONT_PATH = os.environ.get("FONT_PATH", "assets/noto.ttf")

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

def create_tts_chunked(text, output_path, lang_code, voice, is_international=False, production_id=None):
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
            update_production_progress(production_id, {
                "stage": "tts",
                "chunk": i + 1,
                "total_chunks": total_chunks,
                "elapsed_s": round(cumulative_duration, 1),
            })
            
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
    print(f"[MEM] Torch threads: {num_threads}, interop: 1 (set at import)")

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

# ── Subscribe overlay cache ──
# Green-screen subscribe overlay videos are downloaded once and reused across renders
SUBSCRIBE_CACHE_DIR = os.path.join(WORK_DIR, "subscribe_cache")
os.makedirs(SUBSCRIBE_CACHE_DIR, exist_ok=True)

def get_cached_subscribe_overlay(url: str = None, drive_folder_id: str = None, drive_filename: str = None) -> str:
    """Download subscribe overlay video, cache locally. Returns local path or None.
    
    Supports two modes:
    - url: Direct HTTP/Drive URL download
    - drive_folder_id + drive_filename: Download via rclone (more reliable for Drive)
    """
    if not url and not (drive_folder_id and drive_filename):
        return None
    
    import hashlib
    import subprocess
    cache_key = url or f"{drive_folder_id}/{drive_filename}"
    url_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    ext = os.path.splitext(drive_filename or url or ".mp4")[1] or ".mp4"
    cached_path = os.path.join(SUBSCRIBE_CACHE_DIR, f"subscribe_{url_hash}{ext}")
    
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
        print(f"[SUB-OVERLAY] Using cached: {cached_path} ({os.path.getsize(cached_path) / 1024:.0f} KB)")
        return cached_path
    
    # Method 1: rclone (preferred for Drive folders)
    if drive_folder_id and drive_filename and rclone_available():
        try:
            print(f"[SUB-OVERLAY] Downloading via rclone: {drive_filename} from folder {drive_folder_id}")
            result = subprocess.run([
                "rclone", "copyto",
                f"{RCLONE_REMOTE}:{drive_filename}",
                cached_path,
                "--drive-root-folder-id", drive_folder_id,
            ], capture_output=True, text=True, timeout=120)
            
            if os.path.exists(cached_path) and os.path.getsize(cached_path) > 1000:
                print(f"[SUB-OVERLAY] Cached via rclone: {os.path.getsize(cached_path) / 1024:.0f} KB")
                return cached_path
            else:
                print(f"[SUB-OVERLAY] Rclone copyto failed: {result.stderr[-200:]}")
        except Exception as e:
            print(f"[SUB-OVERLAY] Rclone error: {e}")
    
    # Method 2: HTTP download with fallback URLs (for direct URLs)
    if url:
        try:
            download_drive_file(url, cached_path, timeout=60)
            return cached_path
        except Exception as e:
            print(f"[SUB-OVERLAY] HTTP error: {e}")
    
    return None

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
        with worker_lock:
            videos_copy = dict(videos)
        with shelve.open(SHELVE_FILE_PATH) as db:
            db['videos'] = videos_copy
            print(f"Saved {len(videos_copy)} videos to persistent storage")
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
                worker_start_time = time.time()
                video_id = video_queue.get()
                if video_id in videos:
                    with worker_lock:
                        videos[video_id]["status"] = VideoStatus.PROCESSING
                        data = videos[video_id]["data"]
                    save_videos()
                    video_dir = os.path.join(TMP_DIR, video_id)
                    os.makedirs(video_dir, exist_ok=True)
                    
                    try:
                        # Download background video — skip for v4 with folder_ids (clips selected later)
                        bg_video_path = os.path.join(video_dir, "background.mp4")
                        _is_v4_folders = (data.get("version") == "v4" and data.get("bg_video_folder_ids"))
                        if not _is_v4_folders:
                            print(f"Downloading background video for {video_id}")
                            download_drive_file(data["bg_video_url"], bg_video_path, timeout=120)
                        else:
                            print(f"[V4] Skipping static bg download for {video_id} — will use bg_video_folder_ids")

                        # Download person image
                        print(f"Downloading person image for {video_id}")
                        person_image_path = os.path.join(video_dir, "person.png")
                        download_drive_file(data["person_image_url"], person_image_path, timeout=60)
                    except Exception as download_error:
                        try:
                            shutil.rmtree(video_dir)
                        except:
                            pass
                        raise Exception(f"Download failed: {download_error}")
                    
                    overlay_path = os.path.join(video_dir, "overlay.png")
                    print("creating overlay")
                    font_path = FONT_PATH
                    if LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"] == "h":
                        font_path = "assets/noto_hindi.ttf"
                    
                    display_name = data.get("person_name") or "Narrator"
                    version = data.get("version", "v2")

                    if version == "v4":
                        # --- POOF background removal ---
                        if data.get("poof_remove_bg"):
                            poof_key = get_system_config("poof_api_key")
                            if not poof_key:
                                raise RuntimeError("poof_api_key not found in system_config")
                            person_image_path = remove_background_poof(person_image_path, poof_key)

                        # --- Character position ---
                        pos = data.get("character_position", "random")
                        if pos == "random":
                            pos = random.choice(["left", "center", "right"])
                        print(f"[V4] character_position={pos}")

                        create_overlay_v4(
                            person_image_path=person_image_path,
                            position=pos,
                            output_path=overlay_path,
                        )
                    elif version == "v3":
                        create_overlay_v3(
                            person_image_path=person_image_path,
                            output_path=overlay_path,
                            subtitle_background_color=(0, 0, 0, 200),
                        )
                    else:
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
                    
                    # Normalize text: $12,000 → "12,000 dollars", 50% → "50 percent", etc.
                    # Without this, TTS freezes on symbols and desyncs all subtitles
                    tts_text = normalize_text_for_tts(data["text"])
                    if tts_text != data["text"]:
                        diff = len(data["text"]) - len(tts_text)
                        print(f"[TTS] Text normalized ({diff:+d} chars)")
                    
                    text_len = len(tts_text)
                    use_chunked = text_len > TTS_CHUNK_CHARS
                    
                    if use_chunked:
                        print(f"[TTS] Large text ({text_len} chars), using chunked processing")
                        captions, audio_length = create_tts_chunked(
                            text=tts_text, output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                            is_international=is_international,
                            production_id=data.get("production_id", ""),
                        )
                    elif is_international:
                        captions, audio_length = create_tts_international(
                            text=tts_text, output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                        )
                    else:
                        captions, audio_length = create_tts_english(
                            text=tts_text, output_path=sound_path,
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
                    
                    subtitle_path = os.path.join(video_dir, "subtitle.ass")
                    print(f"Creating subtitle (version: {version})")
                    
                    if version == "v4":
                        # --- Color preset ---
                        preset_name = data.get("subtitle_color_preset", "random")
                        if preset_name == "random":
                            color_preset = random.choice(COLOR_PRESETS)
                        else:
                            color_preset = next(
                                (p for p in COLOR_PRESETS if p["name"] == preset_name),
                                COLOR_PRESETS[0],
                            )
                        print(f"[V4] subtitle_color_preset={color_preset['name']}")
                        create_subtitle_v4_karaoke(
                            word_captions=captions, output_path=subtitle_path,
                            color_preset=color_preset, font_size=80,
                        )
                    elif version in ("v2", "v3"):
                        print(f"Using {version} karaoke subtitle style")
                        create_subtitle_v2_karaoke(
                            word_captions=captions, font_size=80, output_path=subtitle_path,
                        )
                    else:
                        print("Using v1 static subtitle style")
                        create_subtitle(
                            segments=segments, font_size=80, output_path=subtitle_path,
                        )

                    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
                    print("rendering video")
                    _prod_id = data.get("production_id", "")
                    _render_start = time.time()
                    _last_progress_ts = [0.0]  # mutable cell for closure
                    def _render_progress(pct):
                        now = time.time()
                        if now - _last_progress_ts[0] < 5.0:
                            return
                        _last_progress_ts[0] = now
                        update_production_progress(_prod_id, {
                            "stage": "render",
                            "pct": pct,
                            "elapsed_s": round(now - _render_start, 1),
                        })

                    if version == "v4":
                        # --- Background clips ---
                        folder_ids = data.get("bg_video_folder_ids", [])
                        if folder_ids:
                            bg_paths = select_bg_videos(
                                folder_ids=folder_ids,
                                max_clips=data.get("max_bg_clips", 10),
                                video_dir=video_dir,
                            ) or ([bg_video_path] if os.path.exists(bg_video_path) else [])
                            if not bg_paths:
                                raise RuntimeError(
                                    f"[V4] No background clips available: select_bg_videos returned empty "
                                    f"and bg_video_path does not exist at {bg_video_path}. "
                                    f"Check that bg_video_folder_ids are valid Drive folders accessible via rclone."
                                )
                        else:
                            bg_paths = [bg_video_path]

                        # --- Effect overlays ---
                        effect_layers_config = get_effect_layers(data)
                        effect_layers_resolved = []
                        for layer in effect_layers_config:
                            resolved = resolve_effect_layer(layer, video_dir)
                            if resolved:
                                effect_layers_resolved.append(resolved)
                                print(f"[V4] Effect layer '{resolved['label']}' ready: "
                                      f"{resolved['blend_mode']} opacity={resolved['opacity']}")
                            else:
                                print(f"[V4] Warning: effect layer "
                                      f"'{layer.get('label', layer.get('id', '?'))}' could not be resolved")

                        # --- Intro videos (Veo 3.1 via Flow) ---
                        # 2026-05-20: download dos clips IA e passe pro render como intro.
                        # `intro_video_urls` deve ser lista de URLs públicas (geralmente do
                        # bucket flow-videos no Supabase Storage). JOB 4 envia esses URLs no
                        # payload quando o canal tem intro_video_enabled=true.
                        intro_video_paths = []
                        raw_intro_urls = data.get("intro_video_urls") or []
                        # Compat: também aceita campo singular `intro_video_url` (string)
                        if not raw_intro_urls and data.get("intro_video_url"):
                            raw_intro_urls = [data["intro_video_url"]]
                        for i, url in enumerate(raw_intro_urls):
                            if not url:
                                continue
                            try:
                                local_intro = os.path.join(video_dir, f"intro_{i}.mp4")
                                resp_dl = requests.get(url, timeout=120, stream=True)
                                if resp_dl.status_code == 200:
                                    with open(local_intro, "wb") as f_intro:
                                        for chunk in resp_dl.iter_content(chunk_size=64 * 1024):
                                            if chunk:
                                                f_intro.write(chunk)
                                    intro_video_paths.append(local_intro)
                                    print(f"[V4 Intro] downloaded intro {i}: {url[:80]} → {local_intro} "
                                          f"({os.path.getsize(local_intro)} bytes)")
                                else:
                                    print(f"[V4 Intro] WARN: download intro {i} HTTP {resp_dl.status_code} — skipping")
                            except Exception as e:
                                print(f"[V4 Intro] WARN: failed to download intro {i} ({url[:80]}): {e} — skipping")

                        overlay_during_intro = bool(data.get("overlay_during_intro", True))
                        subtitle_during_intro = bool(data.get("subtitle_during_intro", True))

                        success = render_video_v4(
                            bg_paths=bg_paths,
                            overlay_path=overlay_path,
                            sound_path=sound_path,
                            subtitle_path=subtitle_path,
                            output_path=video_path,
                            audio_length=audio_length,
                            effect_layers_resolved=effect_layers_resolved if effect_layers_resolved else None,
                            progress_callback=_render_progress,
                            intro_video_paths=intro_video_paths if intro_video_paths else None,
                            overlay_during_intro=overlay_during_intro,
                            subtitle_during_intro=subtitle_during_intro,
                        )
                        if not success:
                            raise RuntimeError("[RENDER-V4] render_video_v4 returned False")

                        # Clean up resolved effect temp files
                        for resolved in effect_layers_resolved:
                            try:
                                if os.path.exists(resolved["local_path"]):
                                    os.remove(resolved["local_path"])
                            except Exception:
                                pass
                    else:
                        # Background video loop is handled by concat demuxer in render_video
                        # Download subscribe overlay if provided (green-screen video)
                        subscribe_overlay_local = None
                        sub_overlay_url = data.get("subscribe_overlay_url")
                        sub_overlay_folder = data.get("subscribe_overlay_drive_folder")
                        sub_overlay_file = data.get("subscribe_overlay_filename", "overlay-subscribe-new.mp4")

                        if sub_overlay_url or sub_overlay_folder:
                            subscribe_overlay_local = get_cached_subscribe_overlay(
                                url=sub_overlay_url,
                                drive_folder_id=sub_overlay_folder,
                                drive_filename=sub_overlay_file,
                            )

                        success = render_video(
                            sound_path=sound_path, subtitle_path=subtitle_path,
                            overlay_path=overlay_path, audio_length=audio_length,
                            bg_video_path=bg_video_path, output_path=video_path,
                            subscribe_overlay_path=subscribe_overlay_local,
                            subscribe_first_at=data.get("subscribe_first_at", 30),
                            subscribe_interval=data.get("subscribe_interval", 180),
                            progress_callback=_render_progress,
                        )
                        if not success:
                            raise RuntimeError("[RENDER-V2] render_video returned False")
                    
                    try:
                        print(f"Cleaning up temporary files for video: {video_id}")
                        shutil.rmtree(video_dir)
                    except Exception as cleanup_error:
                        print(f"Warning: Failed to clean up: {cleanup_error}")
                    
                    # Upload to Google Drive via rclone if available
                    drive_url = None
                    folder_id = data.get("gdrive_folder_id") or RCLONE_FOLDER_ID
                    if rclone_available() and folder_id:
                        drive_url = rclone_upload_video(video_path, f"{video_id}.mp4", folder_id=folder_id)
                        if drive_url:
                            # Delete local file to save disk
                            try:
                                os.remove(video_path)
                                print(f"[DISK] Deleted local: {video_path}")
                            except:
                                pass

                    worker_duration = int(time.time() - worker_start_time)
                    with worker_lock:
                        if drive_url:
                            videos[video_id]["drive_url"] = drive_url
                        videos[video_id]["data"]["video_render_duration_seconds"] = worker_duration
                        videos[video_id]["data"]["video_editing_version"] = data.get("version", "v1")
                        videos[video_id]["data"]["video_duration_seconds"] = int(audio_length)
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
                with worker_lock:
                    videos[video_id]["status"] = VideoStatus.FAILED
                    videos[video_id]["error"] = str(e)
                save_videos()
                try:
                    if 'video_dir' in locals():
                        shutil.rmtree(video_dir)
                except:
                    pass
                # Best-effort: log error_message to Supabase productions table
                prod_id = locals().get('_prod_id') or (videos.get(video_id, {}).get('data', {}).get('production_id', ''))
                if prod_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
                    try:
                        requests.patch(
                            f"{SUPABASE_URL}/rest/v1/productions?id=eq.{prod_id}",
                            headers={
                                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                                "apikey": SUPABASE_SERVICE_KEY,
                                "Content-Type": "application/json",
                                "Prefer": "return=minimal",
                            },
                            json={"error_message": str(e)[:500]},
                            timeout=5,
                        )
                    except Exception:
                        pass

### REST API ###
@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.get("/api/languages")
def get_languages():
    return LANGUAGE_VOICE_CONFIG

@app.get("/api/videos")
def list_videos():
    with worker_lock:
        videos_copy = dict(videos)
    return [{"video_id": vid, "status": vd["status"]} for vid, vd in videos_copy.items()]

@app.post("/api/videos")
def create_video(video: dict):
    text_len = len(video.get("text", ""))
    print(f"[API] POST /api/videos received — text: {text_len} chars, version: {video.get('version', '?')}")
    version = video.get("version", "v1")
    voice = video.get("voice", "af_heart")
    overlay_bg_color = video.get("overlay_bg_color", (232, 14, 64))
    bg_video_url = video.get("bg_video_url", "")
    bg_video_folder_ids = video.get("bg_video_folder_ids", [])

    # bg_video_url required unless V4 with folder_ids
    if not bg_video_url and not bg_video_folder_ids:
        return {"error": "bg_video_url is required (or bg_video_folder_ids for v4)"}

    print(f"[{version}] Creating video — bg_url={bg_video_url[:60] if bg_video_url else 'none'}, folders={len(bg_video_folder_ids)}")

    video_id, video_data, error = process_video_request(
        text=video.get("text", ""),
        person_image_url=video.get("person_image_url", ""),
        person_name=video.get("person_name", ""),
        bg_video_url=bg_video_url,
        voice=voice,
        overlay_bg_color=overlay_bg_color,
        version=version,
        gdrive_folder_id=video.get("gdrive_folder_id", ""),
        subscribe_overlay_url=video.get("subscribe_overlay_url", ""),
        subscribe_overlay_drive_folder=video.get("subscribe_overlay_drive_folder", ""),
        subscribe_overlay_filename=video.get("subscribe_overlay_filename", "overlay-subscribe-new.mp4"),
        subscribe_first_at=int(video.get("subscribe_first_at", 30)),
        subscribe_interval=int(video.get("subscribe_interval", 180)),
        production_id=video.get("production_id", ""),
        # V4
        character_position=video.get("character_position", "random"),
        subtitle_color_preset=video.get("subtitle_color_preset", "random"),
        effect_overlay_ids=video.get("effect_overlay_ids", []),
        effect_layers=video.get("effect_layers", []),
        bg_video_folder_ids=bg_video_folder_ids,
        max_bg_clips=int(video.get("max_bg_clips", 10)),
        poof_remove_bg=bool(video.get("poof_remove_bg", False)),
    )

    if error:
        return {"error": error}
    
    with worker_lock:
        videos[video_id] = video_data
    save_videos()
    video_queue.put(video_id)
    return {"video_id": video_id, "status": VideoStatus.QUEUED}

@app.get("/api/videos/{video_id}/status")
def get_video(video_id: str):
    with worker_lock:
        vid_entry = videos.get(video_id)
    if vid_entry is not None:
        result = {"video_id": video_id, "status": vid_entry["status"]}
        vid_data = vid_entry.get("data", {})
        if "video_render_duration_seconds" in vid_data:
            result["video_render_duration_seconds"] = vid_data["video_render_duration_seconds"]
        if "video_editing_version" in vid_data:
            result["video_editing_version"] = vid_data["video_editing_version"]
        if "video_duration_seconds" in vid_data:
            result["video_duration_seconds"] = vid_data["video_duration_seconds"]

        if vid_entry["status"] == VideoStatus.COMPLETED:
            drive_url = vid_entry.get("drive_url")
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
    with worker_lock:
        vid_entry = videos.get(video_id)
    if vid_entry is not None and vid_entry["status"] == VideoStatus.COMPLETED:
        # If on Google Drive, redirect
        drive_url = vid_entry.get("drive_url")
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
    elif vid_entry is not None:
        if vid_entry["status"] == VideoStatus.FAILED:
            return JSONResponse(content={"video_id": video_id, "status": VideoStatus.FAILED}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        if vid_entry["status"] == VideoStatus.PROCESSING:
            return JSONResponse(content={"video_id": video_id, "status": VideoStatus.PROCESSING}, status_code=status.HTTP_202_ACCEPTED)
    return JSONResponse(content={"video_id": video_id, "status": VideoStatus.NOT_FOUND}, status_code=status.HTTP_404_NOT_FOUND)

@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    """Delete video file and metadata. Called by n8n after uploading to Google Drive."""
    with worker_lock:
        if video_id not in videos:
            return {"video_id": video_id, "status": VideoStatus.NOT_FOUND}
        del videos[video_id]
    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
    freed = 0
    if os.path.exists(video_path):
        freed = os.path.getsize(video_path)
        os.remove(video_path)
        print(f"[DISK] Deleted {video_id}: {freed / 1024 / 1024:.1f} MB freed")
    save_videos()
    return {"video_id": video_id, "status": VideoStatus.DELETED, "freed_mb": round(freed / 1024 / 1024, 1)}

@app.get("/api/disk")
def disk_status():
    """Show disk usage for video storage."""
    with worker_lock:
        videos_copy = dict(videos)
    entries = []
    total_size = 0
    drive_count = 0
    for vid, data in videos_copy.items():
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

    results = []
    targets = []

    with worker_lock:
        videos_copy = dict(videos)

    if video_id:
        if video_id in videos_copy:
            targets.append(video_id)
        else:
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
    else:
        # All local completed videos without drive_url
        targets = [vid for vid, data in videos_copy.items()
                   if data["status"] == VideoStatus.COMPLETED and not data.get("drive_url")]

    for vid in targets:
        video_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
        if not os.path.exists(video_path):
            results.append({"video_id": vid, "status": "file_missing"})
            continue

        size_mb = os.path.getsize(video_path) / 1024 / 1024
        with worker_lock:
            vid_entry = videos.get(vid)
            folder_id = vid_entry.get("data", {}).get("gdrive_folder_id") or RCLONE_FOLDER_ID if vid_entry else None
        if not folder_id:
            results.append({"video_id": vid, "status": "no_folder_id"})
            continue
        drive_url = rclone_upload_video(video_path, f"{vid}.mp4", folder_id=folder_id)
        if drive_url:
            with worker_lock:
                if vid in videos:
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

@app.post("/tts")
async def tts_preview(request: Request):
    """Generate a short TTS audio preview for a given voice.
    Body: { "text": "...", "voice": "af_heart", "speed": 1.0 }
    Returns: WAV audio blob (audio/wav).
    """
    body = await request.json()
    text  = str(body.get("text", "Hello, this is a voice preview sample.")).strip()
    voice = str(body.get("voice", "af_heart")).strip()

    if voice not in LANGUAGE_VOICE_MAP:
        return JSONResponse({"error": f"Unknown voice: {voice}"}, status_code=400)

    cfg = LANGUAGE_VOICE_MAP[voice]
    lang_code      = cfg["lang_code"]
    is_international = cfg["international"]

    text = normalize_text_for_tts(text)

    tmp_path = f"/tmp/tts_preview_{uuid.uuid4().hex}.wav"
    try:
        if is_international:
            create_tts_international(text, tmp_path, lang_code, voice)
        else:
            create_tts_english(text, tmp_path, lang_code, voice)

        return FileResponse(tmp_path, media_type="audio/wav", background=None)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Schedule cleanup after response is sent
        async def _cleanup():
            await asyncio.sleep(10)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        asyncio.create_task(_cleanup())


@app.get("/api/queue")
def get_queue_status():
    with worker_lock:
        videos_copy = dict(videos)
    return {
        "queue_size": video_queue.qsize(),
        "queued": len([v for v in videos_copy.values() if v["status"] == VideoStatus.QUEUED]),
        "processing": len([v for v in videos_copy.values() if v["status"] == VideoStatus.PROCESSING])
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
    with worker_lock:
        videos_copy = dict(videos)
    for v in videos_copy.values():
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
    version = params.get("version", "v2")
    character_position = params.get("character_position", "random")
    subtitle_color_preset = params.get("subtitle_color_preset", "random")
    poof_remove_bg = bool(params.get("poof_remove_bg", False))
    bg_video_folder_ids = params.get("bg_video_folder_ids", [])
    effect_overlay_ids = params.get("effect_overlay_ids", [])
    max_bg_clips = int(params.get("max_bg_clips", 10))

    if not bg_video_url and not bg_video_folder_ids:
        return JSONResponse(content={"error": "bg_video_url or bg_video_folder_ids is required"}, status_code=400)
    
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
        with worker_lock:
            videos_copy = dict(videos)
        for vid_data in videos_copy.values():
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
        version=version,
        gdrive_folder_id=gdrive_folder_id,
        subscribe_overlay_url=params.get("subscribe_overlay_url", ""),
        subscribe_overlay_drive_folder=params.get("subscribe_overlay_drive_folder", ""),
        subscribe_overlay_filename=params.get("subscribe_overlay_filename", "overlay-subscribe-new.mp4"),
        subscribe_first_at=int(params.get("subscribe_first_at", 30)),
        subscribe_interval=int(params.get("subscribe_interval", 180)),
        character_position=character_position,
        subtitle_color_preset=subtitle_color_preset,
        poof_remove_bg=poof_remove_bg,
        bg_video_folder_ids=bg_video_folder_ids,
        effect_overlay_ids=effect_overlay_ids,
        effect_layers=params.get("effect_layers", []),
        max_bg_clips=max_bg_clips,
    )

    if error:
        return JSONResponse(content={"error": error}, status_code=400)

    with worker_lock:
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

### Google Flow (NanoBanana 2 / Pro) Image Generation ###
# Replaces ImageFX (descontinuado). Auth flow:
#   cookie → labs.google/fx/api/auth/session → access_token (Bearer)
# Plus reCAPTCHA Enterprise token (fornecido pela flow-token-extension Chrome).
FLOW_API_URL_TEMPLATE = "https://aisandbox-pa.googleapis.com/v1/projects/{project_id}/flowMedia:batchGenerateImages"
FLOW_SESSION_URL = "https://labs.google/fx/api/auth/session"
FLOW_IMAGES_DIR = os.path.join(os.getcwd(), "flow_output")
os.makedirs(FLOW_IMAGES_DIR, exist_ok=True)
# Mantido para servir imagens legacy já cacheadas em disco
IMAGEFX_IMAGES_DIR = os.path.join(os.getcwd(), "imagefx_output")
os.makedirs(IMAGEFX_IMAGES_DIR, exist_ok=True)

# Supabase Storage for persistent image / video storage
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
IMAGEFX_BUCKET = "imagefx"  # mantido (URLs já em produção apontam para este bucket)
# 2026-05-20: bucket separado para MP4 gerados por Veo 3.1 via /api/generate-video.
# Mantido fora do `imagefx` para permitir lifecycle/retention diferente no futuro.
FLOW_VIDEOS_BUCKET = "flow-videos"
FLOW_VIDEOS_DIR = os.path.join(os.getcwd(), "flow_videos_output")
os.makedirs(FLOW_VIDEOS_DIR, exist_ok=True)

def update_production_progress(production_id: str, progress: dict):
    """Update processing_progress JSONB on the productions row in Supabase.

    Called from TTS (after each chunk) and render (after each 1% milestone).
    Silently ignores failures — progress display is best-effort.
    """
    if not production_id or not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    try:
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/productions?id=eq.{production_id}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={"processing_progress": progress},
            timeout=3,
        )
    except Exception:
        pass

def ensure_supabase_bucket(bucket_name: str = None):
    """Ensure a Supabase Storage bucket exists (create if missing, public).

    Args:
      bucket_name: nome do bucket; default = IMAGEFX_BUCKET (retro-compat).
    Returns:
      True se existir ou foi criado; False em qualquer erro.
    """
    if bucket_name is None:
        bucket_name = IMAGEFX_BUCKET
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return False
    log_prefix = f"[Storage:{bucket_name}]"
    try:
        # Check if bucket exists
        resp = requests.get(
            f"{SUPABASE_URL}/storage/v1/bucket/{bucket_name}",
            headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
            timeout=10
        )
        if resp.status_code == 200:
            print(f"{log_prefix} bucket exists")
            return True

        # Create bucket (public)
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/bucket",
            headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}", "Content-Type": "application/json"},
            json={"id": bucket_name, "name": bucket_name, "public": True},
            timeout=10
        )
        if resp.status_code in (200, 201):
            print(f"{log_prefix} created (public)")
            return True
        else:
            print(f"{log_prefix} failed to create: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"{log_prefix} check error: {e}")
        return False

# Try to create buckets on startup
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    ensure_supabase_bucket(IMAGEFX_BUCKET)
    ensure_supabase_bucket(FLOW_VIDEOS_BUCKET)
else:
    print("[Storage] Supabase Storage not configured — images/videos will be local only")


# ═══════════════════════════════
# V4 HELPERS
# ═══════════════════════════════

_SYSTEM_CONFIG_CACHE: dict = {}  # { key: (value, timestamp) }
_SYSTEM_CONFIG_TTL = 300  # 5 minutes

def get_system_config(key: str) -> str | None:
    """Fetch a value from the system_config table in Supabase (5-min in-memory cache)."""
    cached = _SYSTEM_CONFIG_CACHE.get(key)
    if cached and time.time() - cached[1] < _SYSTEM_CONFIG_TTL:
        return cached[0]
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/system_config?key=eq.{key}&select=value",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
            },
            timeout=5,
        )
        data = resp.json()
        if data and isinstance(data, list) and len(data) > 0:
            value = data[0]["value"]
            _SYSTEM_CONFIG_CACHE[key] = (value, time.time())
            return value
    except Exception as e:
        print(f"[V4] get_system_config({key}) failed: {e}")
    return None


def remove_background_poof(image_path: str, poof_api_key: str) -> str:
    """Remove background via POOF API. Returns path to cropped transparent PNG."""
    from PIL import Image
    import io

    print(f"[V4] POOF: removing background from {image_path}")
    with open(image_path, "rb") as f:
        resp = requests.post(
            "https://api.poof.bg/v1/remove",
            headers={"x-api-key": poof_api_key},
            files={"image_file": f},
            data={"format": "png", "size": "full"},
            timeout=60,
        )
    try:
        resp.raise_for_status()
    except Exception as http_err:
        print(f"[POOF] HTTP error: {http_err} | status={resp.status_code} | body={resp.text[:200]}")
        raise

    img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    out_path = os.path.splitext(image_path)[0] + "_poof.png"
    img.save(out_path)
    print(f"[V4] POOF: saved cropped PNG → {out_path} ({img.size})")
    return out_path


@app.post("/api/remove-bg")
async def remove_bg(request: Request):
    """Remove background via POOF; returns a cropped transparent PNG (image/png).

    Body: raw image bytes — send with:
        curl -s -X POST --data-binary @char.png http://<host>/api/remove-bg -o char_nobg.png
    Used by the Rede Z "Drone V2.2" editor to cut out the character before compositing.
    The poof_api_key stays server-side (read from system_config); callers never need it.
    """
    import tempfile
    poof_key = get_system_config("poof_api_key")
    if not poof_key:
        return JSONResponse({"error": "poof_api_key not found in system_config"}, status_code=500)
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body — send raw image bytes (--data-binary @file)"}, status_code=400)

    tmp_dir = tempfile.mkdtemp(prefix="removebg_")
    in_path = os.path.join(tmp_dir, "input")
    try:
        with open(in_path, "wb") as fh:
            fh.write(body)
        out_path = remove_background_poof(in_path, poof_key)
        return FileResponse(out_path, media_type="image/png",
                            filename="character_nobg.png", background=None)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[remove-bg] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)
    finally:
        # Cleanup the temp dir shortly after the response is streamed.
        async def _cleanup():
            await asyncio.sleep(10)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        asyncio.create_task(_cleanup())


def select_bg_videos(folder_ids: list, max_clips: int, video_dir: str) -> list:
    """Pick random folder, list .mp4s via rclone, download up to max_clips.

    Returns [] on any rclone failure so the caller can fall back to the static bg video.
    """
    import subprocess as _sp

    try:
        folder_id = random.choice(folder_ids)
        print(f"[V4] select_bg_videos: listing folder {folder_id}")

        result = _sp.run(
            [
                "rclone", "lsjson",
                f"{RCLONE_REMOTE}:",
                "--drive-root-folder-id", folder_id,
                "--include", "*.mp4",
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            print(f"[V4] rclone lsjson failed (code {result.returncode}): {result.stderr[:200]}")
            return []

        files = json.loads(result.stdout)
        random.shuffle(files)
        selected = files[:max_clips]
        print(f"[V4] select_bg_videos: {len(files)} found, downloading {len(selected)}")

        local_paths = []
        for f in selected:
            out = os.path.join(video_dir, f"bg_{f['ID']}.mp4")
            _sp.run(
                ["rclone", "backend", "copyid", f"{RCLONE_REMOTE}:", f["ID"], out],
                check=True, timeout=120,
            )
            local_paths.append(out)
            print(f"[V4] Downloaded bg clip: {f.get('Name', f['ID'])}")

        return local_paths
    except Exception as e:
        print(f"[V4] select_bg_videos error: {e}")
        return []


def get_effect_layers(data: dict) -> list:
    """Return effect layers from request data, with backward compat for effect_overlay_ids.

    New channels store effect_layers (list of EffectLayer dicts) in the request data.
    Old channels only have effect_overlay_ids (list of Drive file IDs).
    """
    layers = data.get("effect_layers")
    if layers:
        return [l for l in layers if l.get("enabled", True)]
    # Legacy fallback: synthesise a single colorkey_black layer from effect_overlay_ids
    old_ids = data.get("effect_overlay_ids", [])
    if old_ids:
        return [{
            "id": "legacy-effect",
            "label": "Efeito (legado)",
            "mode": "random_from_ids",
            "drive_file_ids": old_ids,
            "blend_mode": "colorkey_black",
            "opacity_min": 1.0,
            "opacity_max": 1.0,
            "enabled": True,
        }]
    return []


def resolve_effect_layer(layer: dict, video_dir: str) -> dict | None:
    """Download the effect file for a layer. Returns resolved dict or None on failure."""
    import json as _json
    if not layer.get("enabled", True):
        return None
    mode = layer.get("mode", "random_from_ids")
    layer_id = layer.get("id", "unknown")
    tmp_path = os.path.join(video_dir, f"effect_{layer_id}.mp4")

    if mode == "fixed":
        file_id = layer.get("drive_file_id")
        if not file_id:
            return None
        result = subprocess.run(
            ["rclone", "backend", "copyid", f"{RCLONE_REMOTE}:", file_id, tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"[EFFECT LAYER] rclone copyid failed for {file_id}: {result.stderr[:200]}")
            return None

    elif mode == "random_from_folder":
        folder_id = layer.get("drive_folder_id")
        if not folder_id:
            return None
        result = subprocess.run(
            ["rclone", "lsjson", f"{RCLONE_REMOTE}:",
             "--drive-root-folder-id", folder_id,
             "--no-modtime", "--include", "*.mp4"],
            capture_output=True, text=True, timeout=30,
        )
        try:
            files = _json.loads(result.stdout) if result.returncode == 0 and result.stdout.strip() else []
        except Exception:
            files = []
        if not files:
            print(f"[EFFECT LAYER] No mp4 files in folder {folder_id}")
            return None
        chosen = random.choice(files)
        chosen_id = chosen.get("ID") or chosen.get("id")
        if not chosen_id:
            return None
        res2 = subprocess.run(
            ["rclone", "backend", "copyid", f"{RCLONE_REMOTE}:", chosen_id, tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        if res2.returncode != 0:
            print(f"[EFFECT LAYER] rclone copyid failed for folder file {chosen_id}: {res2.stderr[:200]}")
            return None

    elif mode == "random_from_ids":
        file_ids = layer.get("drive_file_ids", [])
        if not file_ids:
            return None
        file_id = random.choice(file_ids)
        result = subprocess.run(
            ["rclone", "backend", "copyid", f"{RCLONE_REMOTE}:", file_id, tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"[EFFECT LAYER] rclone copyid failed for {file_id}: {result.stderr[:200]}")
            return None

    else:
        print(f"[EFFECT LAYER] Unknown mode: {mode}")
        return None

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1000:
        print(f"[EFFECT LAYER] File too small or missing: {tmp_path}")
        return None

    opacity_min = float(layer.get("opacity_min", 1.0))
    opacity_max = float(layer.get("opacity_max", 1.0))
    opacity = round(random.uniform(min(opacity_min, opacity_max), max(opacity_min, opacity_max)), 3)

    return {
        "local_path": tmp_path,
        "opacity": opacity,
        "blend_mode": layer.get("blend_mode", "colorkey_black"),
        "label": layer.get("label", ""),
    }


def upload_to_supabase_storage(
    image_bytes: bytes,
    filename: str,
    content_type: str = "image/png",
    bucket: str = None,
    timeout: int = 30,
) -> str | None:
    """Upload bytes to Supabase Storage bucket. Returns public URL or None on failure.

    Args:
      image_bytes: payload (image OR video bytes).
      filename: nome do arquivo dentro do bucket.
      content_type: MIME type (e.g., 'image/png', 'video/mp4').
      bucket: nome do bucket; default = IMAGEFX_BUCKET (retro-compat).
      timeout: timeout HTTP em segundos (vídeos podem ser maiores, default 30s mantém OK até ~30MB).
    """
    if bucket is None:
        bucket = IMAGEFX_BUCKET
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print(f"[Storage:{bucket}] Supabase Storage not configured, skipping upload")
        return None
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{bucket}/{filename}",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": content_type,
                "x-upsert": "true",
            },
            data=image_bytes,
            timeout=timeout,
        )
        if resp.status_code in (200, 201):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{filename}"
            return public_url
        else:
            print(f"[Storage:{bucket}] upload failed: {resp.status_code} {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[Storage:{bucket}] upload error: {e}")
        return None


def _detect_image_type(image_bytes: bytes) -> tuple[str, str] | None:
    """Detect image type from magic bytes.

    Returns (content_type, extension) or None if not a valid image.
    """
    if not image_bytes or len(image_bytes) < 12:
        return None
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ("image/png", "png")
    if image_bytes[:3] == b"\xff\xd8\xff":
        return ("image/jpeg", "jpg")
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return ("image/webp", "webp")
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return ("image/gif", "gif")
    return None

FLOW_DEFAULT_HEADERS = {
    "Content-Type": "text/plain;charset=UTF-8",  # importante: evita CORS preflight (body é JSON)
    "Referer": "https://labs.google/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    "sec-ch-ua": '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Origin": "https://labs.google",
    "Accept": "*/*",
}

# Aspect ratio: aceita "16:9"/"9:16"/etc + aliases legacy "PORTRAIT"/"LANDSCAPE"/"SQUARE"
ASPECT_RATIO_MAP = {
    "16:9": "IMAGE_ASPECT_RATIO_LANDSCAPE",
    "4:3": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
    "1:1": "IMAGE_ASPECT_RATIO_SQUARE",
    "3:4": "IMAGE_ASPECT_RATIO_PORTRAIT_THREE_FOUR",
    "9:16": "IMAGE_ASPECT_RATIO_PORTRAIT",
    # legacy
    "LANDSCAPE": "IMAGE_ASPECT_RATIO_LANDSCAPE",
    "PORTRAIT": "IMAGE_ASPECT_RATIO_PORTRAIT",
    "SQUARE": "IMAGE_ASPECT_RATIO_SQUARE",
    "LANDSCAPE_4_3": "IMAGE_ASPECT_RATIO_LANDSCAPE_FOUR_THREE",
}

# Modelo: nome interno do payload Flow
MODEL_MAP = {
    "nano_banana_2": "NARWHAL",
    "nano_banana_pro": "GEM_PIX_2",
}

# Pool de tokens reCAPTCHA Enterprise alimentado pela flow-token-extension Chrome.
# Cada token é tipicamente single-use no servidor e válido por ~120s.
#
# 2026-05-20: pool foi particionado por *action* do reCAPTCHA. A página do Flow gera
# tokens com action='IMAGE_GENERATION' para chamadas de imagem e action='VIDEO_GENERATION'
# para chamadas de vídeo (descoberto inspecionando o bundle do labs.google).
# Misturar actions dispara PUBLIC_ERROR_UNUSUAL_ACTIVITY no backend Google.
_RECAPTCHA_ACTION_DEFAULT = "IMAGE_GENERATION"
_RECAPTCHA_ACTION_VIDEO = "VIDEO_GENERATION"
_recaptcha_pools: dict[str, list[dict]] = {}  # action → list of {token, received_at, used}
_recaptcha_pool_lock = threading.Lock()
_RECAPTCHA_POOL_MAX = 12
_RECAPTCHA_TTL = 110.0  # margem antes da expiração real (~120s)


def _normalize_recaptcha_action(action: str | None) -> str:
    """Normaliza action; aceita 'video'/'image' como atalho da UI."""
    if not action:
        return _RECAPTCHA_ACTION_DEFAULT
    a = action.strip().upper()
    if a in ("VIDEO", "VIDEO_GENERATION"):
        return _RECAPTCHA_ACTION_VIDEO
    if a in ("IMAGE", "IMAGE_GENERATION"):
        return _RECAPTCHA_ACTION_DEFAULT
    return a  # action customizada (futura)


def _flow_get_project_id() -> str:
    """Buscar flow_project_id em system_config; fallback para constante."""
    DEFAULT = "9fc28b8b-d679-47db-8142-0befe8f2a15a"
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return DEFAULT
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/system_config?key=eq.flow_project_id&select=value",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
            },
            timeout=5,
        )
        if resp.ok:
            rows = resp.json()
            if rows and rows[0].get("value"):
                return rows[0]["value"]
    except Exception:
        pass
    return DEFAULT


def _consume_recaptcha_tokens(n: int, action: str = _RECAPTCHA_ACTION_DEFAULT) -> list[str]:
    """Tira até N tokens não-usados não-expirados do pool da action. Mais recentes primeiro."""
    action = _normalize_recaptcha_action(action)
    now = time.time()
    out: list[str] = []
    with _recaptcha_pool_lock:
        pool = _recaptcha_pools.get(action, [])
        for entry in reversed(pool):
            if entry["used"]:
                continue
            if now - entry["received_at"] >= _RECAPTCHA_TTL:
                continue
            entry["used"] = True
            out.append(entry["token"])
            if len(out) >= n:
                break
    return out


def _wait_for_recaptcha_tokens(n: int, timeout: float = 30.0, action: str = _RECAPTCHA_ACTION_DEFAULT) -> list[str]:
    """Pede N tokens da action; espera até timeout enquanto a extension entrega novos."""
    deadline = time.time() + timeout
    tokens = _consume_recaptcha_tokens(n, action=action)
    while len(tokens) < n and time.time() < deadline:
        time.sleep(1.0)
        more = _consume_recaptcha_tokens(n - len(tokens), action=action)
        tokens.extend(more)
    return tokens


@app.post("/api/flow-token-set")
def flow_token_set(req: dict):
    """Endpoint chamado pela flow-token-extension a cada token reCAPTCHA novo.

    Body:
      token: string (obrigatório) — token gerado por grecaptcha.enterprise.execute
      action: string (opcional) — 'IMAGE_GENERATION' (default) ou 'VIDEO_GENERATION'
              Extensões v1.1 e anteriores não enviam → cai no default = image (retro-compat).
      generated_at: int (opcional) — timestamp ms da geração (informativo)
    """
    token = (req.get("token") or "").strip()
    if not token:
        return JSONResponse(content={"error": "empty token"}, status_code=400)
    action = _normalize_recaptcha_action(req.get("action"))
    with _recaptcha_pool_lock:
        pool = _recaptcha_pools.setdefault(action, [])
        pool.append({"token": token, "received_at": time.time(), "used": False})
        # Trim pool: descartar usados e expirados, manter no máx _RECAPTCHA_POOL_MAX recentes
        now = time.time()
        _recaptcha_pools[action] = [
            e for e in pool
            if not e["used"] and now - e["received_at"] < _RECAPTCHA_TTL
        ][-_RECAPTCHA_POOL_MAX:]
        pool_size = len(_recaptcha_pools[action])
    return {"success": True, "pool_size": pool_size, "action": action}


@app.post("/api/flow-cookie-set")
def flow_cookie_set(req: dict):
    """Endpoint chamado pela flow-token-extension a cada ciclo de sincronização de
    cookies (a cada ~5 min, ou só quando o cookie muda — a extensão envia um hash
    diff para evitar spam).

    Atualiza system_config.flow_cookie via Supabase REST e invalida o cache local
    para que a próxima chamada de /api/generate-image use o cookie novo.

    Body:
      cookie: string Cookie: header completa (e.g., "SID=...; HSID=...; ...")
    """
    cookie = (req.get("cookie") or "").strip()
    if not cookie:
        return JSONResponse(content={"error": "empty cookie"}, status_code=400)
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return JSONResponse(content={"error": "supabase not configured on backend"}, status_code=500)
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/system_config?on_conflict=key",
            headers={
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "apikey": SUPABASE_SERVICE_KEY,
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            json={"key": "flow_cookie", "value": cookie},
            timeout=10,
        )
        # Bust the in-memory cache so the next generate-image picks up the new value.
        try:
            _SYSTEM_CONFIG_CACHE.pop("flow_cookie", None)
        except NameError:
            pass
        if not resp.ok:
            return JSONResponse(
                content={"error": f"supabase {resp.status_code}: {resp.text[:200]}"},
                status_code=500,
            )
        return {"success": True, "cookie_length": len(cookie)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/flow-token-status")
def flow_token_status(action: str | None = None):
    """UI/extension consulta para mostrar estado do pool.

    Query:
      action: 'image' (default) | 'video' — qual pool consultar
              Sem action: retorna estado do pool image (retro-compat) +
              campo 'pools' com todos os pools rastreados.
    """
    now = time.time()
    action_norm = _normalize_recaptcha_action(action)
    with _recaptcha_pool_lock:
        # Snapshot de todos os pools (para diagnóstico/UI dual)
        all_pools: dict[str, dict] = {}
        for act, pool in _recaptcha_pools.items():
            valid = [e for e in pool if not e["used"] and now - e["received_at"] < _RECAPTCHA_TTL]
            most_recent = max((e["received_at"] for e in pool), default=0)
            all_pools[act] = {
                "pool_size": len(valid),
                "total_tracked": len(pool),
                "most_recent_age_seconds": (now - most_recent) if most_recent else None,
                "has_fresh_tokens": len(valid) > 0,
            }
        target = all_pools.get(action_norm, {
            "pool_size": 0, "total_tracked": 0,
            "most_recent_age_seconds": None, "has_fresh_tokens": False,
        })
    return {
        **target,           # retro-compat: pool_size, has_fresh_tokens, etc. no top-level
        "action": action_norm,
        "pools": all_pools, # snapshot de todas as actions rastreadas
    }


@app.get("/api/flow-token-get")
def flow_token_get(action: str | None = None):
    """Pop a fresh reCAPTCHA token from the pool for testing.

    Query:
      action: 'image' (default) | 'video' — qual pool extrair.
    """
    action_norm = _normalize_recaptcha_action(action)
    tokens = _consume_recaptcha_tokens(1, action=action_norm)
    if not tokens:
        return JSONResponse(content={"error": f"no fresh tokens in pool '{action_norm}'"}, status_code=503)
    return {"token": tokens[0], "action": action_norm}



def flow_get_token(cookie: str) -> dict:
    """Trocar session cookie por access_token via labs.google.

    Igual ImageFX (mesmo endpoint). Retorna { access_token, expires, user{...} }.
    """
    cookie = cookie.replace("\r", "").replace("\n", " ").strip()
    cookie = re.sub(r"\s+", " ", cookie)

    headers = {
        "Origin": "https://labs.google",
        "Referer": "https://labs.google/",
        "Cookie": cookie,
    }

    print(f"[Flow] Token exchange — cookie length: {len(cookie)} chars, first 80: {cookie[:80]}...")
    resp = requests.get(FLOW_SESSION_URL, headers=headers, timeout=15)

    if not resp.ok:
        print(f"[Flow] Session failed: HTTP {resp.status_code} — {resp.text[:300]}")
        raise Exception(f"Session auth failed (HTTP {resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    if not data.get("access_token") or not data.get("expires"):
        raise Exception(f"Session response missing access_token/expires. Keys: {list(data.keys())}")

    return data


FLOW_UPLOAD_URL = "https://aisandbox-pa.googleapis.com/v1/flow/uploadImage"


def _flow_upload_reference(*, access_token: str, project_id: str, src: str) -> str | None:
    """Faz upload de uma imagem de referência (URL http(s) ou data URL) ao Flow e
    retorna o mediaId ('name') para uso em imageInputs.

    Schema REAL capturado do labs.google (POST text/plain com corpo JSON):
      POST https://aisandbox-pa.googleapis.com/v1/flow/uploadImage
      body = {"clientContext":{"projectId":..,"tool":"PINHOLE"},
              "imageBytes": <base64 SEM prefixo data:>,
              "isUserUploaded": true, "isHidden": false,
              "mimeType": "image/png", "fileName": "ref.png"}
      resp = {"media": {"name": "<mediaId>", ...}}  → media.name é o id p/ imageInputs.

    Resolve src (data URL ou http) → bytes; faz upload; devolve media.name.
    Em qualquer falha retorna None (caller pula a referência sem quebrar a geração).
    """
    try:
        # 1) resolver bytes + mimeType
        if src.startswith("data:"):
            head, _, b64 = src.partition(",")
            mime = "image/png"
            if head.startswith("data:") and ";" in head:
                mime = head[len("data:"):head.index(";")] or mime
            raw = base64.b64decode(b64)
        elif src.startswith("http"):
            r = requests.get(src, timeout=60)
            if r.status_code != 200 or not r.content:
                print(f"[Flow upload] download falhou ({r.status_code}): {src[:80]}")
                return None
            raw = r.content
            mime = r.headers.get("Content-Type", "image/png").split(";")[0].strip() or "image/png"
        else:
            return None

        if not raw:
            return None
        ext = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg", "image/webp": "webp"}.get(mime, "png")
        payload = {
            "clientContext": {"projectId": project_id, "tool": "PINHOLE"},
            "imageBytes": base64.b64encode(raw).decode("ascii"),
            "isUserUploaded": True,
            "isHidden": False,
            "mimeType": mime,
            "fileName": f"ref_reference.{ext}",
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "text/plain;charset=UTF-8",
        }
        # IMPORTANTE: enviar como text/plain (data=), não json= (que forçaria application/json)
        resp = requests.post(FLOW_UPLOAD_URL, data=json.dumps(payload), headers=headers, timeout=90)
        if resp.status_code != 200:
            print(f"[Flow upload] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        name = (resp.json().get("media") or {}).get("name")
        if name:
            print(f"[Flow upload] ref → mediaId {name}")
        return name or None
    except Exception as e:
        print(f"[Flow upload] erro: {e}")
        return None


def _flow_call_single(
    *,
    project_id: str,
    access_token: str,
    recaptcha_token: str,
    batch_id: str,
    session_id: str,
    model_name: str,
    aspect_ratio: str,
    prompt: str,
    seed: int,
    image_inputs: list | None = None,
) -> dict:
    """Faz uma chamada flowMedia:batchGenerateImages para gerar 1 imagem.

    Retorna {"ok": True, "fife_url": ..., "media_id": ..., "seed": ..., "dimensions": {...}}
    ou {"ok": False, "error": ..., "auth_expired": bool, "recaptcha_rejected": bool}.
    """
    url = FLOW_API_URL_TEMPLATE.format(project_id=project_id)
    client_context = {
        "recaptchaContext": {
            "token": recaptcha_token,
            "applicationType": "RECAPTCHA_APPLICATION_TYPE_WEB",
        },
        "projectId": project_id,
        "tool": "PINHOLE",
        "sessionId": session_id,
    }
    def _build_payload(inputs):
        return {
            "clientContext": client_context,
            "mediaGenerationContext": {"batchId": batch_id},
            "useNewMedia": True,
            "requests": [
                {
                    "clientContext": client_context,
                    "imageModelName": model_name,
                    "imageAspectRatio": aspect_ratio,
                    "structuredPrompt": {"parts": [{"text": prompt}]},
                    "seed": seed,
                    "imageInputs": inputs or [],
                }
            ],
        }
    headers = {
        **FLOW_DEFAULT_HEADERS,
        "Authorization": f"Bearer {access_token}",
    }
    try:
        resp = requests.post(url, json=_build_payload(image_inputs), headers=headers, timeout=90)
        # Se o schema do Flow rejeitar imageInputs (400) E o usuário NÃO tinha passado refs,
        # refaz sem refs (defensivo). Se o usuário PASSOU refs (intenção explícita), NÃO silencia
        # — devolve erro com a body do Flow p/ que a página mostre claramente.
        if resp.status_code == 400 and "image_inputs" in (resp.text or ""):
            if not image_inputs:
                resp = requests.post(url, json=_build_payload([]), headers=headers, timeout=90)
            else:
                return {"ok": False, "status_code": 400,
                        "error": "Flow rejeitou imageInputs (refs ativas). Body: " + (resp.text or "")[:400],
                        "image_inputs_rejected": True}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout (90s)"}
    except Exception as e:
        return {"ok": False, "error": f"request failed: {e}"}

    if resp.status_code in (401, 403):
        body_excerpt = resp.text[:300]
        recaptcha_rejected = False
        if resp.status_code == 403:
            try:
                data = resp.json()
                details = data.get("error", {}).get("details", [])
                reason = (details[0].get("reason", "") if details else "")
                recaptcha_rejected = "UNUSUAL_ACTIVITY" in reason or "RECAPTCHA" in reason.upper()
            except Exception:
                pass
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": f"HTTP {resp.status_code}: {body_excerpt}",
            "auth_expired": resp.status_code == 401,
            "recaptcha_rejected": recaptcha_rejected,
        }
    if resp.status_code != 200:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
        }
    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "error": "non-JSON response"}

    for entry in data.get("media", []):
        gen = entry.get("image", {}).get("generatedImage", {})
        fife_url = gen.get("fifeUrl")
        if fife_url:
            return {
                "ok": True,
                "fife_url": fife_url,
                "media_id": entry.get("name"),
                "seed": gen.get("seed"),
                "dimensions": entry.get("image", {}).get("dimensions", {}),
            }
    return {"ok": False, "error": "no images in response", "raw_keys": list(data.keys())}


def _download_fife_url(fife_url: str) -> tuple[bytes, str, str] | None:
    """Baixa imagem da URL signada do Flow (flow-content.google).

    Retorna (bytes, content_type, extension) ou None se falhar.
    Valida via magic bytes — rejeita HTML/JSON retornados com status 200.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
        ),
        "Referer": "https://labs.google/",
        "Accept": "image/avif,image/webp,image/png,image/*,*/*;q=0.8",
    }
    try:
        resp = requests.get(fife_url, headers=headers, timeout=60)
    except Exception as e:
        print(f"[Flow] download exception: {e}")
        return None

    if resp.status_code != 200:
        print(f"[Flow] download HTTP {resp.status_code}: {resp.text[:200]}")
        return None

    body = resp.content
    server_ct = resp.headers.get("Content-Type", "")
    if not body:
        print("[Flow] download returned empty body")
        return None

    detected = _detect_image_type(body)
    if not detected:
        # Não é imagem válida — provavelmente HTML/JSON de erro com status 200
        preview = body[:200].decode("utf-8", errors="replace")
        print(
            f"[Flow] download não é imagem válida — Content-Type={server_ct}, "
            f"size={len(body)}, preview={preview!r}"
        )
        return None

    content_type, ext = detected
    print(
        f"[Flow] download ok — server_ct={server_ct}, detected={content_type}, "
        f"size={len(body)} bytes"
    )
    return (body, content_type, ext)


@app.post("/api/generate-image")
def generate_flow(req: dict):
    """Generate images via Google Flow (NanoBanana 2 / Pro).

    Body:
      cookie: session cookie header string from labs.google
      prompt: text prompt
      aspect_ratio: "16:9" | "4:3" | "1:1" | "3:4" | "9:16" (legacy "PORTRAIT"/"LANDSCAPE" também aceitos)
      num_images: 1-8 (default 4) — disparados em paralelo
      model: "nano_banana_2" (default) | "nano_banana_pro"
      project_id: opcional override (senão lê system_config.flow_project_id)
    """
    cookie = (req.get("cookie") or "").strip()
    # Fallback: se body não trouxer cookie, lê system_config.flow_cookie (que a flow-token-extension
    # mantém sempre atualizado via /api/flow-cookie-set + cache invalidation). Isso permite que o
    # caller (n8n) NÃO precise capturar o cookie estaticamente — cada call usa o cookie mais novo.
    if not cookie:
        cookie = get_system_config("flow_cookie") or ""
    cookie = re.sub(r"\s+", " ", cookie.replace("\r", "").replace("\n", " ")).strip()
    prompt = (req.get("prompt") or "").strip()
    aspect_ratio_raw = (req.get("aspect_ratio") or "16:9").strip()
    num_images = int(req.get("num_images") or 4)
    model_raw = (req.get("model") or "nano_banana_2").strip().lower()
    project_id = (req.get("project_id") or "").strip() or _flow_get_project_id()

    if not cookie:
        return JSONResponse(content={"error": "Cookie de sessão indisponível (nem no body nem em system_config.flow_cookie). Garanta que a flow-token-extension está rodando."}, status_code=400)
    if not prompt:
        return JSONResponse(content={"error": "Prompt é obrigatório"}, status_code=400)

    ar_value = ASPECT_RATIO_MAP.get(aspect_ratio_raw) or ASPECT_RATIO_MAP.get(aspect_ratio_raw.upper())
    if not ar_value:
        return JSONResponse(content={"error": f"aspect_ratio inválido: {aspect_ratio_raw}"}, status_code=400)

    model_name = MODEL_MAP.get(model_raw)
    if not model_name:
        return JSONResponse(
            content={"error": f"model inválido: {model_raw}. Use nano_banana_2 ou nano_banana_pro"},
            status_code=400,
        )

    if num_images < 1 or num_images > 8:
        return JSONResponse(content={"error": "num_images deve estar entre 1 e 8"}, status_code=400)

    # Step 1: cookie → access_token
    try:
        session_data = flow_get_token(cookie)
        access_token = session_data["access_token"]
        print(f"[Flow] access_token obtido: {access_token[:20]}...")
    except Exception as e:
        return JSONResponse(
            content={"error": f"Falha na autenticação: {e}", "auth_expired": True},
            status_code=401,
        )

    # Step 2: pegar N tokens reCAPTCHA do pool (espera até 30s)
    recaptcha_tokens = _wait_for_recaptcha_tokens(num_images, timeout=30.0)
    if len(recaptcha_tokens) < num_images:
        return JSONResponse(
            content={
                "error": (
                    f"Pool de tokens reCAPTCHA insuficiente: {len(recaptcha_tokens)}/{num_images} "
                    "disponíveis. Verifique se a flow-token-extension está rodando com a aba do Flow "
                    "aberta no Chrome."
                ),
                "flow_token_missing": True,
                "available": len(recaptcha_tokens),
                "needed": num_images,
            },
            status_code=503,
        )

    batch_id = str(uuid.uuid4())
    session_id = f";{int(time.time() * 1000)}"

    # ---- Imagens de referência (image-to-image), opcional ----
    # req.reference_images: lista de strings base64 (data URL "data:...;base64,XXX" ou base64 cru).
    # Quando vazio, imageInputs fica [] e o comportamento é idêntico ao anterior (RA não muda).
    # ATENÇÃO: o schema do imageInputs do Flow NÃO está documentado aqui — esta estrutura
    # (`{"image": {"encodedImage": b64}}`) é um PALPITE e precisa ser validada num teste real.
    # Se o Flow exigir mediaId, será necessário primeiro fazer upload da imagem ao Flow e usar o id.
    # Schema REAL do Flow (capturado do labs.google):
    #   imageInputs[i] = {"imageInputType":"IMAGE_INPUT_TYPE_REFERENCE", "name": <mediaId>}
    # O <mediaId> ("name") NÃO é base64 — vem de fazer UPLOAD da imagem ao Flow primeiro.
    # modo: "all" (manda todas) | "random1" (sorteia 1) — para A/B testar.
    ref_list_raw = list(req.get("reference_images") or [])
    if (req.get("reference_mode") or "all").strip().lower() == "random1" and len(ref_list_raw) > 1:
        ref_list_raw = [random.choice(ref_list_raw)]
    image_inputs: list = []
    ref_failures: list = []  # [{src, error}]
    # Regex p/ extrair mediaId direto de URLs labs.google do Flow (form: .../edit/<UUID>)
    _LABS_EDIT_RE = re.compile(r"labs\.google/.*?/edit/([0-9a-fA-F-]{20,})")
    _UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    for ri in ref_list_raw:
        if isinstance(ri, dict):
            ri = ri.get("name") or ri.get("media_id") or ri.get("url") or ri.get("base64") or ""
        ri = (ri or "").strip()
        if not ri:
            continue
        # Atalho: URL labs.google do Flow → extrai mediaId, sem upload (evita 401 no upload)
        m = _LABS_EDIT_RE.search(ri)
        if m:
            media_name = m.group(1)
            print(f"[Flow] ref labs.google URL → mediaId {media_name} (sem upload)")
        elif _UUID_RE.match(ri):
            media_name = ri  # bare UUID = mediaId pronto
            print(f"[Flow] ref bare mediaId → {media_name}")
        elif ri.startswith("http") or ri.startswith("data:"):
            media_name = _flow_upload_reference(access_token=access_token, project_id=project_id, src=ri)
            if not media_name:
                ref_failures.append({"src": ri[:120], "error": "upload ao Flow falhou (cookie pode estar expirado / 4xx)"})
                print(f"[Flow] ref upload FALHOU, pulando: {ri[:80]}")
                continue
        else:
            media_name = ri  # qualquer outro: trata como mediaId/name do Flow
        image_inputs.append({"imageInputType": "IMAGE_INPUT_TYPE_REFERENCE", "name": media_name})
    references_requested = len([r for r in ref_list_raw if r])
    references_uploaded  = len(image_inputs)
    print(f"[Flow] refs requested={references_requested} uploaded={references_uploaded} failed={len(ref_failures)}")

    # 🔬 Importante (2026-06-05, capturado de curl real): imageInputs só funciona com GEM_PIX_2
    # (= nano_banana_pro). NARWHAL (= nano_banana_2) recebido com imageInputs devolve 500 INTERNAL
    # silencioso. Auto-promover quando há refs evita esse erro e mantém compat (sem refs continua
    # respeitando o model do request).
    if image_inputs and model_name == "NARWHAL":
        print(f"[Flow] refs presentes (n={len(image_inputs)}) → forçando model GEM_PIX_2 (nano_banana_pro). NARWHAL não suporta imageInputs.")
        model_name = "GEM_PIX_2"
        model_raw = "nano_banana_pro"

    # Step 3: disparar N chamadas paralelas (cada uma com seed e recaptcha próprios)
    def _one(i: int) -> dict:
        return _flow_call_single(
            project_id=project_id,
            access_token=access_token,
            recaptcha_token=recaptcha_tokens[i],
            batch_id=batch_id,
            session_id=session_id,
            model_name=model_name,
            aspect_ratio=ar_value,
            prompt=prompt,
            seed=random.randint(1, 2 ** 31),
            image_inputs=image_inputs,
        )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(num_images, 4)) as ex:
        futures = [ex.submit(_one, i) for i in range(num_images)]
        for fut in futures:
            try:
                results.append(fut.result(timeout=120))
            except Exception as e:
                results.append({"ok": False, "error": f"exception: {e}"})

    successes = [r for r in results if r.get("ok")]
    failures = [r for r in results if not r.get("ok")]

    if not successes:
        any_auth_expired = any(r.get("auth_expired") for r in failures)
        any_recaptcha = any(r.get("recaptcha_rejected") for r in failures)
        first_err = failures[0].get("error") if failures else "unknown"
        # 2026-05-14 — return 200 with success=false so reverse proxies (Easypanel/Traefik)
        # do NOT replace the JSON body with their generic "Not Found" HTML error page.
        # Callers (N8N JOB 3) must check the `success` field on the body, not just status_code.
        return JSONResponse(
            content={
                "success": False,
                "error": f"Todas as {num_images} gerações falharam. Primeiro erro: {first_err}",
                "auth_expired": any_auth_expired,
                "recaptcha_rejected": any_recaptcha,
                "failures": failures[:4],
            },
            status_code=200,
        )

    # Step 4: download cada fifeUrl + upload Supabase Storage + cache local
    base_id = str(uuid.uuid4())[:12]
    saved_images: list[dict] = []
    for idx, r in enumerate(successes):
        downloaded = _download_fife_url(r["fife_url"])
        if not downloaded:
            print(f"[Flow] failed to download image {idx} from {r['fife_url'][:80]}")
            continue
        img_bytes, content_type, ext = downloaded
        img_id = f"{base_id}_{idx}"
        filename = f"{img_id}.{ext}"
        try:
            img_path = os.path.join(FLOW_IMAGES_DIR, filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print(f"[Flow] local cache failed: {e}")
        public_url = upload_to_supabase_storage(img_bytes, filename, content_type=content_type)
        saved_images.append(
            {
                "image_id": img_id,
                "image_url": public_url or f"/api/flow/{filename}",
                "size_bytes": len(img_bytes),
                "content_type": content_type,
                # 2026-05-20: mediaId do Flow é preservado aqui para que callers possam
                # reaproveitar a imagem como referenceImages.mediaId em /api/generate-video.
                # Isso é o que garante "mesma personagem da thumb aparece no intro_video".
                "media_id": r.get("media_id"),
            }
        )

    if not saved_images:
        # 2026-05-14 — return 200 with success=false (see note on the other failure return above).
        return JSONResponse(
            content={
                "success": False,
                "error": "Imagens geradas pelo Flow mas todos os downloads falharam",
            },
            status_code=200,
        )

    primary = saved_images[0]
    storage_type = "supabase" if primary["image_url"].startswith("http") else "local"
    print(
        f"[Flow] {len(saved_images)} imagens geradas e salvas como {base_id}_* "
        f"({storage_type}); falhas: {len(failures)}"
    )

    return {
        "success": True,
        "image_id": primary["image_id"],
        "image_url": primary["image_url"],
        "total_generated": len(saved_images),
        "size_bytes": primary["size_bytes"],
        "all_images": [s["image_url"] for s in saved_images],
        "model": model_raw,
        "aspect_ratio": aspect_ratio_raw,
        "failures": len(failures),
        # 2026-05-20: novos campos para integração com /api/generate-video.
        # `media_id` é o mediaId do Flow da imagem principal — pode ser usado como
        # `reference_media_id` no submit de vídeo (mantém personagem consistente).
        "media_id": primary.get("media_id"),
        "all_media_ids": [s.get("media_id") for s in saved_images],
        # 2026-06-05: visibilidade do upload de referências p/ a UI saber se estilo foi aplicado.
        "references_requested": references_requested,
        "references_uploaded": references_uploaded,
        "references_failures": ref_failures,
    }


_IMG_EXT_MEDIA_TYPE = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


def _serve_image_from_dirs(name: str, dirs: tuple[str, ...]):
    """Procura `name` (com ou sem extensão) em `dirs`. Retorna FileResponse com media_type correto."""
    # Caso o caller passe já com extensão
    if "." in os.path.basename(name):
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                ext = name.rsplit(".", 1)[-1].lower()
                return FileResponse(p, media_type=_IMG_EXT_MEDIA_TYPE.get(ext, "application/octet-stream"))
    # Caso só image_id sem ext — testa cada extensão conhecida
    for d in dirs:
        for ext, media in _IMG_EXT_MEDIA_TYPE.items():
            p = os.path.join(d, f"{name}.{ext}")
            if os.path.exists(p):
                return FileResponse(p, media_type=media)
    return JSONResponse(content={"error": "Image not found"}, status_code=404)


@app.get("/api/flow/{image_id}")
def get_flow_image(image_id: str):
    """Serve a generated Flow image (local cache fallback)."""
    return _serve_image_from_dirs(image_id, (FLOW_IMAGES_DIR,))


@app.get("/api/imagefx/{image_id}")
def get_imagefx_image(image_id: str):
    """Serve a generated image (legacy ImageFX path; tenta flow_output e imagefx_output)."""
    return _serve_image_from_dirs(image_id, (FLOW_IMAGES_DIR, IMAGEFX_IMAGES_DIR))


@app.post("/api/test-flow")
@app.post("/api/test-imagefx")
def test_flow_token(req: dict):
    """Test if a Flow session cookie + reCAPTCHA pool + Flow API are working.

    Body: { "cookie": "<session cookie header string>" }
    Returns: { "valid": true/false, "message": "...", "user": "..." }
    """
    cookie = (req.get("cookie") or "").replace("\r", "").replace("\n", " ").strip()
    cookie = re.sub(r"\s+", " ", cookie)
    if not cookie:
        return JSONResponse(content={"valid": False, "message": "Cookie vazio"}, status_code=400)

    try:
        session_data = flow_get_token(cookie)
        access_token = session_data["access_token"]
        user_name = session_data.get("user", {}).get("name", "Unknown")
        user_email = session_data.get("user", {}).get("email", "")
        expires = session_data.get("expires", "")
    except Exception as e:
        return {"valid": False, "message": f"Autenticação falhou: {e}"}

    tokens = _consume_recaptcha_tokens(1)
    if not tokens:
        return {
            "valid": False,
            "message": (
                f"Cookie OK (user: {user_name}) mas pool de tokens reCAPTCHA está vazio. "
                "Abra o Flow no Chrome com a flow-token-extension instalada."
            ),
            "user": user_name,
            "email": user_email,
            "expires": expires,
            "flow_token_missing": True,
        }

    project_id = _flow_get_project_id()
    result = _flow_call_single(
        project_id=project_id,
        access_token=access_token,
        recaptcha_token=tokens[0],
        batch_id=str(uuid.uuid4()),
        session_id=f";{int(time.time() * 1000)}",
        model_name="NARWHAL",
        aspect_ratio="IMAGE_ASPECT_RATIO_SQUARE",
        prompt="a simple red circle on white background",
        seed=random.randint(1, 2 ** 31),
    )

    if result.get("ok"):
        return {
            "valid": True,
            "message": f"Cookie + reCAPTCHA + Flow API OK (user: {user_name})",
            "user": user_name,
            "email": user_email,
            "expires": expires,
            "fife_url": result.get("fife_url"),
        }

    return {
        "valid": False,
        "message": f"Cookie aceito mas Flow API rejeitou: {result.get('error', 'erro desconhecido')}",
        "user": user_name,
        "auth_expired": result.get("auth_expired", False),
        "recaptcha_rejected": result.get("recaptcha_rejected", False),
    }


### ═══════════════════════════════════════════════════════════════════════
### Google Flow Video Generation (Veo 3.1 — Lite/Fast/Quality)
### ═══════════════════════════════════════════════════════════════════════
### Pipeline: cookie → Bearer (existing flow_get_token) → reCAPTCHA fresh
###           (action=VIDEO_GENERATION) → submit batchAsyncGenerate... →
###           poll batchCheckAsync... (MEDIA_GENERATION_STATUS_SUCCESSFUL) →
###           GET /v1/media/{mediaId} (returns base64-encoded MP4 inline) →
###           decode + upload Supabase Storage bucket flow-videos.
### Validação E2E: 2026-05-20 — vídeo gerado e baixado em ~9s (Veo Lite).

# Aspect ratio map (vídeo aceita LANDSCAPE / PORTRAIT / SQUARE)
VIDEO_ASPECT_RATIO_MAP = {
    "16:9": "VIDEO_ASPECT_RATIO_LANDSCAPE",
    "9:16": "VIDEO_ASPECT_RATIO_PORTRAIT",
    "1:1":  "VIDEO_ASPECT_RATIO_SQUARE",
    # legacy
    "LANDSCAPE": "VIDEO_ASPECT_RATIO_LANDSCAPE",
    "PORTRAIT":  "VIDEO_ASPECT_RATIO_PORTRAIT",
    "SQUARE":    "VIDEO_ASPECT_RATIO_SQUARE",
}

# Modelos válidos do Veo 3.1 (custos em créditos por geração — plano AI Pro)
VIDEO_MODEL_KEYS = {
    "veo_3_1_r2v_lite":    {"credits": 10,  "label": "Veo 3.1 Lite"},
    "veo_3_1_r2v_fast":    {"credits": 80,  "label": "Veo 3.1 Fast"},
    "veo_3_1_r2v_quality": {"credits": 400, "label": "Veo 3.1 Quality"},
    # Omniflash (modelo experimental, custo conhecido: 100 créditos)
    "omniflash":           {"credits": 100, "label": "OmniFlash"},
}

VIDEO_DEFAULT_HEADERS = FLOW_DEFAULT_HEADERS  # mesmos headers do endpoint de imagem


def _flow_video_submit(
    *,
    access_token: str,
    recaptcha_token: str,
    project_id: str,
    prompt: str,
    model_key: str,
    aspect_ratio: str,
    seed: int,
    reference_media_id: str | None = None,
) -> dict:
    """Submete uma geração de vídeo (1 request) e retorna o operation/media handle.

    Retorna:
      { ok: True, media_id, workflow_id, remaining_credits }
      ou
      { ok: False, error, status_code?, auth_expired?, recaptcha_rejected? }
    """
    batch_id = str(uuid.uuid4())
    session_id = f";{int(time.time() * 1000)}"
    requests_arr = [{
        "aspectRatio": aspect_ratio,
        "textInput": {"structuredPrompt": {"parts": [{"text": prompt}]}},
        "videoModelKey": model_key,
        "seed": seed,
        "metadata": {},
    }]
    if reference_media_id:
        requests_arr[0]["referenceImages"] = [{
            "mediaId": reference_media_id,
            "imageUsageType": "IMAGE_USAGE_TYPE_ASSET",
        }]

    payload = {
        "mediaGenerationContext": {
            "batchId": batch_id,
            "audioFailurePreference": "BLOCK_SILENCED_VIDEOS",
        },
        "clientContext": {
            "projectId": project_id,
            "tool": "PINHOLE",
            "userPaygateTier": "PAYGATE_TIER_ONE",
            "sessionId": session_id,
            "recaptchaContext": {
                "token": recaptcha_token,
                "applicationType": "RECAPTCHA_APPLICATION_TYPE_WEB",
            },
        },
        "requests": requests_arr,
        "useV2ModelConfig": True,
    }
    headers = {**VIDEO_DEFAULT_HEADERS, "Authorization": f"Bearer {access_token}"}
    url = "https://aisandbox-pa.googleapis.com/v1/video:batchAsyncGenerateVideoReferenceImages"
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "submit timeout (60s)"}
    except Exception as e:
        return {"ok": False, "error": f"submit request failed: {e}"}

    if resp.status_code in (401, 403):
        recaptcha_rejected = False
        if resp.status_code == 403:
            try:
                details = resp.json().get("error", {}).get("details", [])
                reason = (details[0].get("reason", "") if details else "")
                recaptcha_rejected = "UNUSUAL_ACTIVITY" in reason or "RECAPTCHA" in reason.upper()
            except Exception:
                pass
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            "auth_expired": resp.status_code == 401,
            "recaptcha_rejected": recaptcha_rejected,
        }
    if resp.status_code != 200:
        return {
            "ok": False,
            "status_code": resp.status_code,
            "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
        }
    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "error": "submit non-JSON response"}

    media = data.get("media") or []
    if not media:
        return {"ok": False, "error": "submit response has no media", "raw_keys": list(data.keys())}
    return {
        "ok": True,
        "media_id": media[0].get("name"),
        "workflow_id": media[0].get("workflowId"),
        "project_id": media[0].get("projectId") or project_id,
        "remaining_credits": data.get("remainingCredits"),
    }


def _flow_video_poll(
    *,
    access_token: str,
    media_id: str,
    project_id: str,
    timeout_s: int = 180,
    interval_s: float = 5.0,
) -> dict:
    """Polling do status até `MEDIA_GENERATION_STATUS_SUCCESSFUL` ou erro/timeout.

    Veo Lite tipicamente fica pronto em 9-15s. Vídeos Fast/Quality podem demorar mais.
    """
    url = "https://aisandbox-pa.googleapis.com/v1/video:batchCheckAsyncVideoGenerationStatus"
    headers = {**VIDEO_DEFAULT_HEADERS, "Authorization": f"Bearer {access_token}"}
    payload = {"media": [{"name": media_id, "projectId": project_id}]}
    deadline = time.time() + timeout_s
    last_status = "?"
    polls = 0
    while time.time() < deadline:
        polls += 1
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=20)
        except Exception as e:
            print(f"[Flow Video] poll #{polls} request error: {e}")
            time.sleep(interval_s)
            continue
        if resp.status_code != 200:
            print(f"[Flow Video] poll #{polls} HTTP {resp.status_code}: {resp.text[:200]}")
            time.sleep(interval_s)
            continue
        try:
            data = resp.json()
            m = (data.get("media") or [{}])[0]
            last_status = (
                m.get("mediaMetadata", {}).get("mediaStatus", {}).get("mediaGenerationStatus", "?")
            )
        except Exception as e:
            print(f"[Flow Video] poll #{polls} parse error: {e}")
            time.sleep(interval_s)
            continue
        if last_status == "MEDIA_GENERATION_STATUS_SUCCESSFUL":
            return {"ok": True, "polls": polls, "status": last_status}
        if last_status in ("MEDIA_GENERATION_STATUS_FAILED", "MEDIA_GENERATION_STATUS_ERROR"):
            return {"ok": False, "status": last_status, "error": "generation failed", "polls": polls}
        time.sleep(interval_s)
    return {"ok": False, "error": f"timeout {timeout_s}s, last status: {last_status}", "polls": polls}


def _flow_video_fetch_mp4(*, access_token: str, media_id: str) -> dict:
    """GET v1/media/{mediaId} retorna JSON com video.encodedVideo (base64 do MP4).

    Retorna { ok, mp4_bytes, duration, model, seed } ou { ok: False, error }.
    """
    url = f"https://aisandbox-pa.googleapis.com/v1/media/{media_id}"
    headers = {**VIDEO_DEFAULT_HEADERS, "Authorization": f"Bearer {access_token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=120)
    except Exception as e:
        return {"ok": False, "error": f"fetch error: {e}"}
    if resp.status_code != 200:
        return {"ok": False, "error": f"fetch HTTP {resp.status_code}: {resp.text[:300]}"}
    try:
        data = resp.json()
    except Exception:
        return {"ok": False, "error": "fetch non-JSON response"}
    b64 = (data.get("video") or {}).get("encodedVideo")
    if not b64:
        return {"ok": False, "error": "no encodedVideo in response"}
    try:
        mp4 = base64.b64decode(b64)
    except Exception as e:
        return {"ok": False, "error": f"base64 decode failed: {e}"}
    # Validação básica: magic bytes do MP4 (ftypisom / ftyp...)
    if len(mp4) < 32 or b"ftyp" not in mp4[:32]:
        return {"ok": False, "error": "decoded bytes are not a valid MP4"}
    gv = (data.get("video") or {}).get("generatedVideo", {})
    return {
        "ok": True,
        "mp4_bytes": mp4,
        "model": gv.get("model"),
        "seed": gv.get("seed"),
        "aspect_ratio": gv.get("aspectRatio"),
    }


@app.post("/api/generate-video")
def generate_flow_video(req: dict):
    """Gera 1 vídeo de ~8s via Google Flow (Veo 3.1).

    Body:
      cookie: string (opcional — default lê system_config.flow_cookie)
      prompt: string (obrigatório)
      reference_media_id: string (opcional) — mediaId de uma imagem prévia
                          (vinda de /api/generate-image) para usar como ASSET.
                          Se ausente: text-to-video puro.
      model: "veo_3_1_r2v_lite" (default) | "veo_3_1_r2v_fast" | "veo_3_1_r2v_quality"
      aspect_ratio: "16:9" | "9:16" | "1:1" (default "16:9")
      project_id: string (opcional override do system_config.flow_project_id)
      poll_timeout_s: int (default 180)
      production_id: string (opcional — só pra logging/progress tracking)

    Retorna:
      { success: true, video_id, video_url, media_id, model, credits_used,
        remaining_credits, poll_seconds, size_bytes }
      ou
      { success: false, error, auth_expired?, recaptcha_rejected? } (status 200 — mesmo
      padrão do /api/generate-image para Easypanel não trocar o body por HTML).
    """
    # 1. Cookie: do body ou system_config
    cookie = (req.get("cookie") or "").strip()
    if not cookie:
        cookie = get_system_config("flow_cookie") or ""
    cookie = re.sub(r"\s+", " ", cookie.replace("\r", "").replace("\n", " ")).strip()
    if not cookie:
        return JSONResponse(content={"success": False, "error": "Cookie de sessão indisponível (nem no body nem em system_config.flow_cookie)"}, status_code=400)

    # 2. Validação dos campos
    prompt = (req.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse(content={"success": False, "error": "prompt é obrigatório"}, status_code=400)
    if len(prompt) > 4000:
        return JSONResponse(content={"success": False, "error": "prompt excede 4000 chars"}, status_code=400)

    model_key = (req.get("model") or "veo_3_1_r2v_lite").strip().lower()
    if model_key not in VIDEO_MODEL_KEYS:
        return JSONResponse(content={"success": False, "error": f"model inválido: {model_key}. Válidos: {list(VIDEO_MODEL_KEYS.keys())}"}, status_code=400)
    model_meta = VIDEO_MODEL_KEYS[model_key]

    aspect_raw = (req.get("aspect_ratio") or "16:9").strip()
    aspect = VIDEO_ASPECT_RATIO_MAP.get(aspect_raw) or VIDEO_ASPECT_RATIO_MAP.get(aspect_raw.upper())
    if not aspect:
        return JSONResponse(content={"success": False, "error": f"aspect_ratio inválido: {aspect_raw}"}, status_code=400)

    reference_media_id = (req.get("reference_media_id") or "").strip() or None
    project_id = (req.get("project_id") or "").strip() or _flow_get_project_id()
    poll_timeout_s = int(req.get("poll_timeout_s") or 180)
    production_id = (req.get("production_id") or "").strip() or None

    log_tag = f"[Flow Video {model_key}{' ref' if reference_media_id else ''}]"
    print(f"{log_tag} submitting — prompt_len={len(prompt)} aspect={aspect} project={project_id[:8]}")

    # 3. Cookie → access_token
    try:
        session_data = flow_get_token(cookie)
        access_token = session_data["access_token"]
    except Exception as e:
        return JSONResponse(
            content={"success": False, "error": f"Falha na autenticação: {e}", "auth_expired": True},
            status_code=200,
        )

    # 4. Pega 1 token reCAPTCHA da action VIDEO_GENERATION (espera até 30s se pool vazio)
    tokens = _wait_for_recaptcha_tokens(1, timeout=30.0, action=_RECAPTCHA_ACTION_VIDEO)
    if not tokens:
        return JSONResponse(
            content={
                "success": False,
                "error": (
                    "Pool de tokens reCAPTCHA (action=VIDEO_GENERATION) vazio. "
                    "Verifique se a flow-token-extension v1.2+ está rodando com a aba do Flow aberta."
                ),
                "flow_token_missing": True,
            },
            status_code=200,
        )

    # 5. Submit
    submit = _flow_video_submit(
        access_token=access_token,
        recaptcha_token=tokens[0],
        project_id=project_id,
        prompt=prompt,
        model_key=model_key,
        aspect_ratio=aspect,
        seed=random.randint(1, 2 ** 31),
        reference_media_id=reference_media_id,
    )
    if not submit.get("ok"):
        print(f"{log_tag} submit failed: {submit.get('error')}")
        return JSONResponse(content={"success": False, **submit}, status_code=200)
    media_id = submit["media_id"]
    print(f"{log_tag} submitted — mediaId={media_id} remainingCredits={submit.get('remaining_credits')}")

    if production_id:
        update_production_progress(production_id, {"stage": "flow_video", "step": "polling", "media_id": media_id})

    # 6. Poll
    t_poll_start = time.time()
    poll = _flow_video_poll(
        access_token=access_token,
        media_id=media_id,
        project_id=submit.get("project_id", project_id),
        timeout_s=poll_timeout_s,
    )
    if not poll.get("ok"):
        print(f"{log_tag} poll failed: {poll.get('error')}")
        return JSONResponse(content={"success": False, "media_id": media_id, **poll}, status_code=200)
    poll_seconds = round(time.time() - t_poll_start, 1)
    print(f"{log_tag} ready in {poll_seconds}s ({poll.get('polls')} polls)")

    # 7. Fetch MP4 (base64 inline)
    if production_id:
        update_production_progress(production_id, {"stage": "flow_video", "step": "downloading", "media_id": media_id})
    fetched = _flow_video_fetch_mp4(access_token=access_token, media_id=media_id)
    if not fetched.get("ok"):
        return JSONResponse(content={"success": False, "media_id": media_id, **fetched}, status_code=200)
    mp4_bytes = fetched["mp4_bytes"]
    size_bytes = len(mp4_bytes)

    # 8. Persist: local cache + Supabase Storage
    video_id = f"{media_id[:8]}_{int(time.time())}"
    filename = f"{video_id}.mp4"
    try:
        with open(os.path.join(FLOW_VIDEOS_DIR, filename), "wb") as f:
            f.write(mp4_bytes)
    except Exception as e:
        print(f"{log_tag} local cache failed: {e}")
    public_url = upload_to_supabase_storage(
        mp4_bytes, filename, content_type="video/mp4",
        bucket=FLOW_VIDEOS_BUCKET, timeout=60,
    )
    storage_type = "supabase" if public_url else "local"
    print(f"{log_tag} saved ({size_bytes/1024/1024:.2f} MB, {storage_type})")

    return {
        "success": True,
        "video_id": video_id,
        "video_url": public_url or f"/api/flow-videos/{filename}",
        "media_id": media_id,
        "model": model_key,
        "model_label": model_meta["label"],
        "credits_used": model_meta["credits"],
        "remaining_credits": submit.get("remaining_credits"),
        "size_bytes": size_bytes,
        "poll_seconds": poll_seconds,
        "aspect_ratio": aspect_raw,
        "reference_media_id": reference_media_id,
        "seed": fetched.get("seed"),
    }


@app.get("/api/flow-videos/{video_filename}")
def get_flow_video(video_filename: str):
    """Serve um MP4 gerado pelo /api/generate-video (fallback do cache local)."""
    # Aceita com ou sem extensão; assume .mp4 por padrão
    if not video_filename.endswith(".mp4"):
        video_filename = f"{video_filename}.mp4"
    path = os.path.join(FLOW_VIDEOS_DIR, video_filename)
    if not os.path.exists(path):
        return JSONResponse(content={"error": "Video not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/test-flow-video")
def test_flow_video(req: dict):
    """Smoke test: valida cookie + pool VIDEO_GENERATION + paygate tier + créditos.

    Body: { "cookie": "<opcional, default system_config.flow_cookie>" }
    Returns: { valid, message, user, email, credits_remaining, paygate_tier, ... }

    NÃO faz submit de geração — apenas troca cookie por Bearer e bate em
    /v1/credits. Não gasta créditos.

    NOTA (2026-05-20): rota nomeada explicitamente `/api/test-flow-video` para evitar
    colisão com `/api/test-video` (que é o endpoint legado de gerar vídeo de teste).
    """
    cookie = (req.get("cookie") or "").strip()
    if not cookie:
        cookie = get_system_config("flow_cookie") or ""
    cookie = re.sub(r"\s+", " ", cookie.replace("\r", "").replace("\n", " ")).strip()
    if not cookie:
        return JSONResponse(content={"valid": False, "message": "Cookie indisponível"}, status_code=400)

    try:
        session_data = flow_get_token(cookie)
        access_token = session_data["access_token"]
        user_name = session_data.get("user", {}).get("name", "Unknown")
        user_email = session_data.get("user", {}).get("email", "")
        expires = session_data.get("expires", "")
    except Exception as e:
        return {"valid": False, "message": f"Autenticação falhou: {e}"}

    # Pool check
    now = time.time()
    with _recaptcha_pool_lock:
        video_pool = _recaptcha_pools.get(_RECAPTCHA_ACTION_VIDEO, [])
        valid_tokens = [e for e in video_pool if not e["used"] and now - e["received_at"] < _RECAPTCHA_TTL]
    has_video_tokens = len(valid_tokens) > 0

    # Credits check (não gasta nada)
    credits_remaining = None
    paygate_tier = None
    try:
        resp = requests.get(
            "https://aisandbox-pa.googleapis.com/v1/credits?key=AIzaSyBtrm0o5ab1c-Ec8ZuLcGt3oJAA5VWt3pY",
            headers={**VIDEO_DEFAULT_HEADERS, "Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if resp.status_code == 200:
            jd = resp.json()
            credits_remaining = jd.get("credits")
            paygate_tier = jd.get("userPaygateTier")
    except Exception as e:
        print(f"[test-video] credits fetch error: {e}")

    return {
        "valid": bool(credits_remaining is not None and has_video_tokens),
        "message": (
            f"Cookie OK (user: {user_name}). "
            f"Pool video: {len(valid_tokens)} tokens. "
            f"Credits: {credits_remaining}. "
            f"Paygate: {paygate_tier}."
        ),
        "user": user_name,
        "email": user_email,
        "expires": expires,
        "credits_remaining": credits_remaining,
        "paygate_tier": paygate_tier,
        "video_pool_size": len(valid_tokens),
        "has_video_tokens": has_video_tokens,
    }


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
    with worker_lock:
        videos[video_id] = video_data
    save_videos()
    video_queue.put(video_id)
    return {"video_id": video_id, "status": VideoStatus.QUEUED.value}


# ─────────────────────────────────────────────────────────────────────
# THUMB MAKER — standalone manual thumbnail composer.
#
# Used by the Pipeline Manager "Thumb Maker" view. Fetches the chosen
# row from public.thumb_templates (html_template + global_css + canvas
# dims + shapes), builds the full HTML the same way the JOB 5 "Montar
# HTML HCTI" node does, then renders it to PNG with Playwright/Chromium
# running inside THIS container. No external service dependency, no
# Supabase round-trip — the PNG is streamed straight to the browser.
# ─────────────────────────────────────────────────────────────────────

# Singleton Chromium browser (lazy-initialized, reused across requests).
# Async-locked so concurrent requests don't race the launch.
_thumb_playwright_ctx = None
_thumb_browser = None
_thumb_browser_lock: asyncio.Lock | None = None


async def _get_thumb_browser():
    """Return a connected Chromium instance, launching it on first use."""
    from playwright.async_api import async_playwright

    global _thumb_playwright_ctx, _thumb_browser, _thumb_browser_lock
    if _thumb_browser_lock is None:
        _thumb_browser_lock = asyncio.Lock()
    async with _thumb_browser_lock:
        if _thumb_browser is not None and _thumb_browser.is_connected():
            return _thumb_browser
        if _thumb_playwright_ctx is None:
            _thumb_playwright_ctx = await async_playwright().start()
        _thumb_browser = await _thumb_playwright_ctx.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-first-run",
                "--no-zygote",
            ],
        )
        return _thumb_browser


async def _render_html_to_png(html: str, width: int = 1280, height: int = 720,
                              ms_delay: int = 500) -> bytes:
    """Render an HTML document to a PNG screenshot using the shared Chromium."""
    browser = await _get_thumb_browser()
    page = await browser.new_page(viewport={"width": width, "height": height})
    try:
        await page.set_content(html, wait_until="networkidle", timeout=30000)
        if ms_delay > 0:
            await asyncio.sleep(ms_delay / 1000.0)
        return await page.screenshot(
            type="png",
            clip={"x": 0, "y": 0, "width": width, "height": height},
        )
    finally:
        await page.close()


def _thumb_supabase_get(path: str, params: dict | None = None):
    """GET against the Supabase PostgREST API using env credentials.
    Returns the parsed JSON list/object. Raises on non-2xx."""
    base = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not base or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_KEY não configurados no env")
    r = requests.get(
        f"{base}/rest/v1/{path}",
        params=params or {},
        headers={
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Accept": "application/json",
        },
        timeout=15,
    )
    if not r.ok:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:300]}")
    return r.json()


def _thumb_fix_short_tags(html: str) -> str:
    """Mirror of the JOB 5 fixShortTags() — expand <s1>-<s5> → <span class="sN">."""
    return re.sub(
        r"<(/?)s([1-5])>",
        lambda m: "</span>" if m.group(1) else f'<span class="s{m.group(2)}">',
        html,
        flags=re.IGNORECASE,
    )


def _thumb_hex_to_rgba(hex_str: str, op: float = 1.0) -> str:
    """Mirror of JOB 5 hexToRgba()."""
    if not hex_str or len(hex_str) < 7:
        return f"rgba(0,0,0,{op})"
    r = int(hex_str[1:3], 16)
    g = int(hex_str[3:5], 16)
    b = int(hex_str[5:7], 16)
    return f"rgba({r},{g},{b},{op})"


def _thumb_serialize_shapes(shapes) -> str:
    """Mirror of JOB 5 serializeShapes() — turn a shapes[] JSON into absolutely
    positioned <div>s with gradient/border/rotation support."""
    if not shapes:
        return ""
    ordered = sorted(shapes, key=lambda s: s.get("zIndex", 0))
    out = []
    for s in ordered:
        is_line = s.get("type") == "line"
        is_circle = s.get("type") == "circle"
        # Background
        grad = s.get("gradient") or {}
        if grad.get("enabled") and isinstance(grad.get("stops"), list) and len(grad["stops"]) >= 2:
            stops = []
            for st in grad["stops"]:
                op = st.get("opacity", 1)
                pos = st.get("position", 0)
                stops.append(f"{_thumb_hex_to_rgba(st.get('color', '#000000'), op)} {pos}%")
            angle = grad.get("angle", 90)
            bg = f"linear-gradient({angle}deg, {', '.join(stops)})"
        else:
            bg = _thumb_hex_to_rgba(s.get("fillColor", "#000000"), s.get("fillOpacity", 1))
        style_parts = [
            "position:absolute",
            f"left:{s.get('x', 0)}%",
            f"top:{s.get('y', 0)}%",
            f"width:{s.get('width', 0)}%",
            (f"height:{s.get('borderWidth', 2)}px" if is_line else f"height:{s.get('height', 0)}%"),
            f"background:{bg}",
            f"border:{s.get('borderWidth', 0)}px solid {s.get('borderColor', 'transparent')}",
            (f"border-radius:{'50%' if is_circle else str(s.get('borderRadius', 0)) + 'px'}"),
            f"transform:rotate({s.get('rotation', 0)}deg)",
            f"z-index:{s.get('zIndex', 1)}",
            "pointer-events:none",
            "box-sizing:border-box",
        ]
        out.append(f'<div style="{";".join(style_parts)}"></div>')
    return "".join(out)


def _thumb_inject_layer_css(css: str, layers: list) -> str:
    """Templates salvos com geradores antigos podem não ter as regras
    `.layer_X { position/z-index/background-image/... }` no global_css.
    Sem essas regras, a imagem e shape-divs do html_template ficam invisíveis
    e o `shapes_html` duplicado (gerado de template.shapes) acaba sobrepondo
    com z-indices errados.

    Esta função regera as regras a partir de `template.layers`, espelhando
    a lógica de `tplConvertLayersToCode()` em pipeline-manager.html. Só anexa
    quando a regra `.layer_X { ... }` não existe no CSS (não destrói edits
    manuais)."""
    if not layers:
        return css
    import re
    out_rules = []
    for idx, layer in enumerate(layers):
        if not layer:
            continue
        raw_id = layer.get("id") or ""
        sid = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_id)
        if not sid:
            continue
        # Skip se a regra já existe (não sobrescreve)
        if re.search(r'\.' + re.escape(sid) + r'\s*\{', css):
            continue
        x = layer.get("x", 0)
        y = layer.get("y", 0)
        w = layer.get("w", 0)
        h = layer.get("h", 0)
        z = idx * 2
        layer_type = layer.get("type")
        slot = layer.get("slot")
        if layer_type == "shape":
            grad = layer.get("gradient") or {}
            if grad.get("enabled") and isinstance(grad.get("stops"), list) and len(grad["stops"]) >= 2:
                stops = []
                for st in grad["stops"]:
                    op = st.get("opacity", 1)
                    pos = st.get("position", 0)
                    stops.append(f"{_thumb_hex_to_rgba(st.get('color', '#000000'), op)} {pos}%")
                angle = grad.get("angle", 90)
                bg = f"linear-gradient({angle}deg, {', '.join(stops)})"
            else:
                bg = _thumb_hex_to_rgba(layer.get("fillColor", "#000000"), layer.get("fillOpacity", 1))
            shape_type = layer.get("shapeType", "rect")
            border_radius = ("50%" if shape_type == "circle" else f"{layer.get('borderRadius', 0)}px")
            rotation = layer.get("rotation", 0)
            transform = f" transform: rotate({rotation}deg);" if rotation else ""
            rule = (
                f".{sid} {{ position: absolute; z-index: {z}; "
                f"left: {x}px; top: {y}px; width: {w}px; height: {h}px; "
                f"background: {bg}; "
                f"border: {layer.get('borderWidth', 0)}px solid {layer.get('borderColor', 'transparent')}; "
                f"border-radius: {border_radius}; pointer-events: none; box-sizing: border-box;{transform} }}"
            )
        elif slot == "background_image" or layer_type == "image":
            # `{{background_image}}` permanece como placeholder — substituição é feita depois.
            rule = (
                f".{sid} {{ position: absolute; z-index: {z}; "
                f"left: {x}px; top: {y}px; width: {w}px; height: {h}px; "
                f"background-image: url('{{{{background_image}}}}'); "
                f"background-size: cover; background-position: 68% center; }}"
            )
        else:
            # Layer de texto (hook / formatted_context / label)
            va = layer.get("verticalAlign", "top")
            fj = "center" if va == "center" else "flex-end" if va == "bottom" else "flex-start"
            rule = (
                f".{sid} {{ position: absolute; z-index: {z}; "
                f"left: {x}px; top: {y}px; width: {w}px; height: {h}px; "
                f"display: flex; flex-direction: column; justify-content: {fj}; overflow: visible; }}"
            )
        out_rules.append(rule)
    if out_rules:
        css = css.rstrip() + "\n\n/* === Auto-injected layer CSS (template missing .layer_X rules) === */\n" + "\n".join(out_rules) + "\n"
    return css


def _thumb_build_full_html(template: dict, formatted_context: str,
                           hook: str, background_image_url: str) -> str:
    """Replicates exactly the JOB 5 "Montar HTML HCTI" node so manual renders
    look identical to pipeline renders. `background_image_url` may be any
    URL (https) or a data: URL (browser-side base64-inlined upload)."""
    html_template = template.get("html_template") or ""
    global_css = template.get("global_css") or ""
    shapes = template.get("shapes") or []

    html = _thumb_fix_short_tags(
        html_template
        .replace("{{formatted_context}}", formatted_context or "")
        .replace("{{hook}}", hook or "")
        .replace("{{background_image}}", background_image_url or "")
    )
    import re
    # Strip any old script tags from html template so they can be replaced by updated ones
    html = re.sub(r'<script\b[^>]*>([\s\S]*?)<\/script>', '', html)
    # Auto-injeta regras `.layer_X { position/z-index/background-image/... }` se faltarem.
    # Templates antigos podem ter sido salvos sem essas regras, e sem elas a imagem e os
    # shape-divs do html_template ficam invisíveis (apenas shapes_html aparece, com z-indices
    # errados → imagem aparenta cobrir shapes). Aplica-se ANTES do replace de placeholder.
    css = _thumb_inject_layer_css(global_css, template.get("layers") or [])
    css = css.replace("{{background_image}}", background_image_url or "")

    # Auto-inject Google Fonts if not imported (handles Oswald, Bebas Neue, Archivo Black, and Anton)
    if "@import" not in css and "fonts.googleapis.com" not in css:
        css = "@import url('https://fonts.googleapis.com/css2?family=Anton&family=Archivo+Black&family=Bebas+Neue&family=Oswald:wght@200;300;400;500;600;700;800;900&display=swap');\n" + css

    # Ensure font-family fallbacks and weights are correct for fixed-weight fonts
    import re
    # Add Google Font fallbacks to font-family declarations if not present
    css = css.replace('"Arial Black", Arial, sans-serif', '"Arial Black", "Archivo Black", Arial, sans-serif')
    css = css.replace('"Arial Black",Arial,sans-serif', '"Arial Black", "Archivo Black", Arial, sans-serif')
    css = css.replace("font-family: 'Arial Black'", "font-family: 'Arial Black', 'Archivo Black'")
    css = css.replace("font-family: ArialBlack", "font-family: 'Arial Black', 'Archivo Black'")
    
    css = css.replace('Impact, "Arial Narrow", sans-serif', 'Impact, Anton, "Arial Narrow", sans-serif')
    css = css.replace('Impact,\"Arial Narrow\",sans-serif', 'Impact, Anton, "Arial Narrow", sans-serif')
    css = css.replace("font-family: 'Impact'", "font-family: 'Impact', 'Anton'")

    # For selectors referencing fixed-weight fonts, force font-weight: normal to prevent Chromium fallback bugs
    for font_keyword in ['Arial Black', 'Archivo Black', 'Impact', 'Anton', 'Bebas Neue']:
        def font_weight_replacer(match):
            block_content = match.group(0)
            if font_keyword in block_content:
                block_content = re.sub(r'font-weight\s*:\s*[^;!]+(?:!\s*important)?\s*;?', 'font-weight: normal !important;', block_content)
            return block_content
        css = re.sub(r'[^{}]+\{[^{}]+\}', font_weight_replacer, css)

    # Auto-inject stroke fix for top layer overlay to prevent thin/eaten letters
    if "._text-top" in css and "-webkit-text-stroke: 0" not in css:
        css += "\n.text-block._text-top { -webkit-text-stroke: 0 transparent !important; paint-order: normal !important; }\n"
        css += "._text-top .s1, ._text-top .s2, ._text-top .s3, ._text-top .s4, ._text-top .s5, ._text-top .hook-inline { -webkit-text-stroke: 0 transparent !important; paint-order: normal !important; }\n"

    # Auto-inject hook height flex container CSS & HTML wrapper styling if it is a hook layer
    layers = template.get("layers") or []
    hook_layers = [l for l in layers if l.get("slot") == "hook"]
    for hl in hook_layers:
        import re
        sid = re.sub(r'[^a-zA-Z0-9_-]', '_', hl.get("id") or "")
        hook_text_selector = f".{sid} .hook-text"
        if hook_text_selector not in css:
            va = hl.get("verticalAlign") or "top"
            fj = "center" if va == "center" else "flex-end" if va == "bottom" else "flex-start"
            ta = hl.get("textAlign") or "left"
            ai = "center" if ta == "center" else "flex-end" if ta == "right" else "flex-start"
            css += f"\n.{sid} .hook-text {{\n  height: 100% !important;\n  display: flex !important;\n  flex-direction: column !important;\n  justify-content: {fj} !important;\n  align-items: {ai} !important;\n}}\n"

        target_str = f'class="{sid}"'
        idx = html.find(target_str)
        if idx != -1:
            dbl_idx = html.find('class="_dbl-wrap"', idx, idx + 300)
            if dbl_idx != -1 and 'style="height:100% !important;"' not in html[idx:dbl_idx+50]:
                html = html[:dbl_idx] + 'style="height:100% !important;" ' + html[dbl_idx:]

    shapes_html = _thumb_serialize_shapes(shapes)
    body_content = (
        f'<div style="position:relative;width:100%;height:100%;overflow:hidden;">{html}{shapes_html}</div>'
        if shapes_html else html
    )

    # MANTENHA SINCRONIZADO com _buildThumbScripts() em pipeline-manager.html e
    # const THUMB_SCRIPTS no Code node "Montar HTML HCTI" em n8n/JOB 5 - Monitor.json.
    # Os 3 scripts (autofit + tokenizer per-linha _hil + multisize) precisam ser idênticos
    # em todos os 4 pontos para que a thumb renderizada bata com o preview do editor.
    # Sempre injeta os 3 — idempotência cobre o caso "DOM não tem o seletor" → no-op silencioso.
    js_scripts = ""
    # Autofit (busca binária 8–400)
    if True:
        js_scripts += r"""
<script>
document.fonts.ready.then(function(){
  document.querySelectorAll("[data-autofit]").forEach(function(c){
    var layerCls=null;for(var i=0;i<c.classList.length;i++){if(c.classList[i].indexOf("layer_")===0){layerCls=c.classList[i];break;}}
    var siblings=layerCls?document.getElementsByClassName(layerCls):[c];
    var apply=function(fn){for(var k=0;k<siblings.length;k++){fn(siblings[k]);}};
    apply(function(el){el.querySelectorAll(".hook-inline").forEach(function(hi){hi.style.setProperty("font-size","inherit","important");hi.style.setProperty("line-height","inherit","important");});});
    var setFS=function(s){apply(function(el){el.querySelectorAll(".text-block").forEach(function(tb){tb.style.setProperty("font-size",s+"px","important");});});};
    var lo=8,hi=400,mid;while(lo<hi-1){mid=Math.floor((lo+hi)/2);setFS(mid);if(c.scrollHeight<=c.offsetHeight)lo=mid;else hi=mid;}setFS(lo);
  });
});
</script>
"""
    # Tokenizer per-linha (FIX-7 2026-05-21): tokeniza .hook-inline em palavras (_tmpw),
    # agrupa por offsetTop em linhas, envolve cada linha em <span class="_hil"> com <br>.
    # Idempotente via dataset.hilTok. Sem ele, o BG vermelho per-linha não renderiza.
    js_scripts += r"""
<script>
document.fonts.ready.then(function(){setTimeout(function(){document.querySelectorAll(".hook-inline").forEach(function(hi){if(hi.dataset.hilTok==="1")return;var t=hi.textContent;if(!t||!t.trim()){hi.dataset.hilTok="1";return;}var parts=t.split(/(\s+)/);hi.innerHTML=parts.map(function(p){if(!p)return "";if(/^\s+$/.test(p))return p;var esc=p.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");return "<span class='_tmpw'>"+esc+"</span>";}).join("");var ws=hi.querySelectorAll("._tmpw");if(ws.length===0){hi.dataset.hilTok="1";return;}var lines=[];var cur=null;for(var i=0;i<ws.length;i++){var top=ws[i].offsetTop;if(!cur||Math.abs(cur.top-top)>2){cur={top:top,words:[]};lines.push(cur);}cur.words.push(ws[i].textContent);}hi.innerHTML=lines.map(function(line,idx){var content=line.words.join(" ").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");return (idx>0?"<br>":"")+"<span class='_hil'>"+content+"</span>";}).join("");hi.dataset.hilTok="1";});},0);});
</script>
"""
    # Multisize (sempre injeta — no-op se DOM não tem [data-multisize])
    if True:
        js_scripts += r"""
<script>
document.fonts.ready.then(function(){
  var Q=String.fromCharCode(34);function buildLine(toks){var parts=[],cur=null;toks.forEach(function(t){if(!cur||cur.cls!==t.cls){if(cur)parts.push(cur);cur={cls:t.cls,ws:[t.w]};}else cur.ws.push(t.w);});if(cur)parts.push(cur);return parts.map(function(p){var txt=p.ws.join(" ");return p.cls?"<span class="+Q+p.cls+Q+">"+txt+"</span>":txt;}).join(" ");}document.querySelectorAll("[data-multisize]").forEach(function(container){var _lc=null;for(var _i=0;_i<container.classList.length;_i++){if(container.classList[_i].indexOf("layer_")===0){_lc=container.classList[_i];break;}}var _sibs=_lc?document.getElementsByClassName(_lc):[container];for(var _si=0;_si<_sibs.length;_si++){_sibs[_si].querySelectorAll(".hook-inline").forEach(function(hi){hi.style.setProperty("font-size","inherit","important");hi.style.setProperty("line-height","inherit","important");});}var cs=window.getComputedStyle(container);var cw=container.clientWidth-parseFloat(cs.paddingLeft||0)-parseFloat(cs.paddingRight||0);var containerH=container.clientHeight-parseFloat(cs.paddingTop||0)-parseFloat(cs.paddingBottom||0);if(cw<10||containerH<10)return;var nonTops=Array.from(container.querySelectorAll(".text-block:not(._text-top)"));nonTops.forEach(function(sp){var spCS=window.getComputedStyle(sp);var LH=parseFloat(spCS.lineHeight)/(parseFloat(spCS.fontSize)||36);if(!(LH>0.5&&LH<5))LH=1.17;if(!sp.dataset.msOrig)sp.dataset.msOrig=sp.innerHTML;var src=document.createElement("span");src.innerHTML=sp.dataset.msOrig;var tokens=[];(function walk(node,cls){if(node.nodeType===3){node.textContent.trim().split(/\s+/).filter(Boolean).forEach(function(w){tokens.push({w:w,cls:cls});});}else if(node.nodeType===1&&node.tagName!=="SCRIPT"){var c2=node.tagName==="SPAN"?(node.getAttribute("class")||""):cls;node.childNodes.forEach(function(ch){walk(ch,c2);});}})(src,"");if(!tokens.length)return;var tChar=tokens.reduce(function(a,t){return a+t.w.length;},0)+Math.max(0,tokens.length-1);var tgtLn=Math.max(8,Math.min(16,Math.round(tChar/21)));var probe=document.createElement("span");probe.className=sp.className.replace(/_text-top/g,"").trim();probe.style.cssText="position:absolute;visibility:hidden;white-space:nowrap;top:-9999px;left:0;text-shadow:none!important;filter:none!important;";document.body.appendChild(probe);probe.innerHTML="<span class="+Q+"hook-inline"+Q+">\u00a0</span>";var _hip0=probe.offsetWidth;probe.innerHTML="&nbsp;";var _hip=Math.max(0,_hip0-probe.offsetWidth);var blo=8,bhi=400;while(blo<bhi-1){var bm=(blo+bhi)>>1;probe.style.setProperty("font-size",bm+"px","important");probe.innerHTML="&nbsp;";var bsw=probe.offsetWidth;var bnL=0,bcW=0,bfst=true;tokens.forEach(function(t){probe.innerHTML=buildLine([t]);var btw=probe.offsetWidth-(t.cls==="hook-inline"?_hip:0);if(!bfst&&bcW+bsw+btw>cw){bnL++;bcW=btw;}else{bcW=bfst?btw:bcW+bsw+btw;bfst=false;}});bnL++;if(bnL*bm*LH<=containerH)blo=bm;else bhi=bm;}var countLn=function(fs){probe.style.setProperty("font-size",fs+"px","important");probe.innerHTML="&nbsp;";var _sw=probe.offsetWidth,_n=0,_cW=0,_ft=true;tokens.forEach(function(t){probe.innerHTML=buildLine([t]);var _tw=probe.offsetWidth-(t.cls==="hook-inline"?_hip:0);if(!_ft&&_cW+_sw+_tw>cw){_n++;_cW=_tw;}else{_cW=_ft?_tw:_cW+_sw+_tw;_ft=false;}});return _n+1;};var wrapFs=blo;if(countLn(blo)<tgtLn+1&&countLn(400)>=tgtLn+1){var wl=blo,wh=400;while(wl<wh-1){var wm=(wl+wh)>>1;if(countLn(wm)>=tgtLn+1)wh=wm;else wl=wm;}wrapFs=wh;}probe.style.setProperty("font-size",wrapFs+"px","important");probe.innerHTML="&nbsp;";var spW=probe.offsetWidth;var lines=[],curLine=[],lineW=0;tokens.forEach(function(t){probe.innerHTML=buildLine([t]);var tw=probe.offsetWidth-(t.cls==="hook-inline"?_hip:0);var needed=curLine.length?lineW+spW+tw:tw;if(curLine.length&&needed>cw){lines.push(curLine);curLine=[t];lineW=tw;}else{curLine.push(t);lineW=needed;}});if(curLine.length)lines.push(curLine);var mg=[],mDone=0;lines.forEach(function(l){if(l.length===1&&mg.length>0&&(lines.length-mDone-1)>=tgtLn-1){mg[mg.length-1]=mg[mg.length-1].concat(l);mDone++;}else{mg.push(l);}});lines=mg;if(!lines.length){document.body.removeChild(probe);return;}var fits=lines.map(function(lt){probe.innerHTML=buildLine(lt);var flo=8,fhi=600,fm;while(flo<fhi-1){fm=(flo+fhi)>>1;probe.style.setProperty("font-size",fm+"px","important");if(probe.offsetWidth<=cw)flo=fm;else fhi=fm;}return flo;});document.body.removeChild(probe);var charCounts=lines.map(function(lt){return lt.reduce(function(a,t){return a+t.w.length;},0)+Math.max(0,lt.length-1);});var minCC=Math.min.apply(null,charCounts);var maxCC=Math.max.apply(null,charCounts);var ccRange=Math.max(1,maxCC-minCC);var seed=lines.length*137+Math.floor(containerH*0.1);var pr=function(n){var x=Math.sin(n*9301+seed*49297+233)*93176;return x-Math.floor(x);};var ph=pr(0)*6.28;var span=(0.7+pr(1)*0.4)*6.28;var baseSize=containerH/(lines.length*LH);var gM=pr(20)<0.25?0:pr(20)<0.5?1:pr(20)<0.75?2:3;var gW=[0.65,0.65,0.50,0.60][gM];var factors=lines.map(function(lt,i){var norm=(charCounts[i]-minCC)/ccRange;var density=norm;var t=lines.length>1?i/(lines.length-1):0.5;var guitar;if(gM===0)guitar=(1.0+Math.cos(6.28*(t-0.22)))/2;else if(gM===1)guitar=(1.0+Math.cos(6.28*(t-0.68)))/2;else if(gM===2)guitar=(1.0+Math.cos(12.57*t))/2;else guitar=(1.0+Math.cos(6.28*(t-0.45)))/2;var combined=(1-gW)*density+gW*guitar;var base=0.740+combined*0.507;var sine=0.015*Math.sin(ph+i*span/Math.max(1,lines.length-1));return Math.min(1.247,Math.max(0.740,base+sine));});if(lines.length>5){var ndis=3;var disUsed=[];for(var di=0;di<ndis;di++){var idx=2+Math.floor(pr(7+di)*Math.max(1,lines.length-3));if(disUsed.indexOf(idx)<0&&idx<lines.length-1){disUsed.push(idx);factors[idx]=pr(10+di)<0.5?0.700+pr(11+di)*0.03:1.22+pr(11+di)*0.02;}}}if(lines.length>4){var minFIdx=2;for(var mi=3;mi<lines.length-1;mi++){if(factors[mi]<factors[minFIdx])minFIdx=mi;}factors[minFIdx]=0.680;}var sizes=lines.map(function(lt,i){return Math.min(fits[i],Math.max(14,Math.round(baseSize*factors[i])));});var totalH=sizes.reduce(function(a,s){return a+s*LH;},0);if(totalH<containerH*0.95){var up=containerH/totalH;sizes=sizes.map(function(s,i){return Math.min(fits[i],Math.max(14,Math.round(s*up)));});}var tHfinal=sizes.reduce(function(a,s){return a+s*LH;},0);if(tHfinal>containerH){var dn=containerH*0.98/tHfinal;sizes=sizes.map(function(s){return Math.max(14,Math.round(s*dn));});}var mfI=2,mfR=0;for(var fi=2;fi<sizes.length-1;fi++){var fr=sizes[fi]/fits[fi];if(fr>mfR){mfR=fr;mfI=fi;}}if(mfR<0.88){var t85=Math.min(fits[mfI],Math.floor(0.88*fits[mfI]));if(t85>sizes[mfI])sizes[mfI]=t85;}var pkS=sizes[mfI];for(var sd=1;sd<=3;sd++){var pMin=Math.round(pkS*(1-sd*0.17));if(mfI-sd>=0)sizes[mfI-sd]=Math.min(fits[mfI-sd],Math.max(sizes[mfI-sd],pMin));if(mfI+sd<lines.length)sizes[mfI+sd]=Math.min(fits[mfI+sd],Math.max(sizes[mfI+sd],pMin));}var tHslope=sizes.reduce(function(a,s){return a+s*LH;},0);if(tHslope>containerH*0.99){var dnSlope=containerH*0.98/tHslope;sizes=sizes.map(function(s){return Math.max(14,Math.round(s*dnSlope));});}else if(tHslope<containerH*0.90){var upSlope=containerH*0.96/tHslope;sizes=sizes.map(function(s,i){return Math.min(fits[i],Math.max(14,Math.round(s*upSlope)));});}var pkFill=sizes[mfI]/fits[mfI];if(pkFill<0.88){var pkTgt=Math.min(fits[mfI],Math.floor(Math.min(0.93*fits[mfI],containerH*0.80/LH)));var npH2=sizes.reduce(function(a,s,i){return i!==mfI?a+s*LH:a;},0);var npTgt2=containerH*0.97-pkTgt*LH;var npSc2=npH2>0?npTgt2/npH2:1;if(npSc2>=0.35&&npSc2<=1.0)sizes=sizes.map(function(s,i){return i===mfI?pkTgt:Math.max(14,Math.round(s*npSc2));});}sp.innerHTML=lines.map(function(lt,i){return"<span style="+Q+"display:block;position:relative;z-index:2;font-size:"+sizes[i]+"px;line-height:"+Math.round(sizes[i]*LH)+"px;white-space:nowrap;"+Q+">"+buildLine(lt)+"</span>";}).join("");});var tops=Array.from(container.querySelectorAll("._text-top"));nonTops.forEach(function(sp,i){if(tops[i])tops[i].innerHTML=sp.innerHTML;});});document.querySelectorAll("[data-multisize]._ghost").forEach(function(g){var lc=null;for(var i=0;i<g.classList.length;i++){if(g.classList[i].indexOf("layer_")===0){lc=g.classList[i];break;}}if(!lc)return;var sibs=document.getElementsByClassName(lc);var main=null;for(var j=0;j<sibs.length;j++){if(!sibs[j].classList.contains("_ghost")){main=sibs[j];break;}}if(!main)return;var mTBs=main.querySelectorAll(".text-block:not(._text-top)");var gTBs=g.querySelectorAll(".text-block:not(._text-top)");for(var k=0;k<gTBs.length&&k<mTBs.length;k++){gTBs[k].innerHTML=mTBs[k].innerHTML;}var gTops=g.querySelectorAll("._text-top");for(var _t=0;_t<gTops.length&&_t<mTBs.length;_t++){gTops[_t].innerHTML=mTBs[_t].innerHTML;}});
});
</script>
"""

    body_content += js_scripts
    return f'<html><head><style>{css}</style></head><body style="margin:0;padding:0">{body_content}</body></html>'


def _thumb_guess_image_mime(image_bytes: bytes) -> str:
    """Detect mime via magic bytes for the data URL prefix."""
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


@app.post("/api/thumb-make")
async def thumb_make(req: dict):
    """Render one of the templates from public.thumb_templates with a manually
    uploaded image and a hand-typed formatted_context. Returns the PNG.

    Body:
      template_id: uuid of the row in public.thumb_templates
      formatted_context: HTML-ish string with <s1>..<s5>...</sX> shorthand spans
      hook: optional, fills {{hook}} placeholder (legacy templates)
      image_base64: data URL ("data:image/png;base64,...") OR raw base64 string

    Returns: image/png stream — or JSON error.
    """
    template_id = (req.get("template_id") or "").strip()
    formatted_context = req.get("formatted_context") or ""
    hook = req.get("hook") or ""
    image_b64 = req.get("image_base64") or ""

    if not template_id:
        return JSONResponse(content={"error": "template_id é obrigatório"}, status_code=400)
    if not formatted_context.strip():
        return JSONResponse(content={"error": "formatted_context é obrigatório"}, status_code=400)
    if not image_b64:
        return JSONResponse(content={"error": "image_base64 é obrigatório"}, status_code=400)

    # Decode base64 (strip data: prefix if present)
    if image_b64.startswith("data:"):
        comma = image_b64.find(",")
        if comma >= 0:
            image_b64 = image_b64[comma + 1:]
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception as e:
        return JSONResponse(content={"error": f"image_base64 inválido: {e}"}, status_code=400)
    mime = _thumb_guess_image_mime(image_bytes)
    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"

    # Fetch template from Supabase
    try:
        rows = _thumb_supabase_get(
            "thumb_templates",
            params={"id": f"eq.{template_id}", "select": "*", "limit": "1"},
        )
        if not rows:
            return JSONResponse(content={"error": f"template_id não encontrado: {template_id}"}, status_code=404)
        template = rows[0]
    except Exception as e:
        return JSONResponse(content={"error": f"falha buscando template: {e}"}, status_code=502)

    full_html = _thumb_build_full_html(template, formatted_context, hook, data_url)

    # Render in-process with Playwright/Chromium (installed in the container
    # via `playwright install --with-deps chromium`). No external service,
    # no Supabase round-trip — just bytes back to the browser.
    try:
        png_bytes = await _render_html_to_png(
            full_html,
            width=int(template.get("canvas_width") or 1280),
            height=int(template.get("canvas_height") or 720),
            ms_delay=500,
        )
    except Exception as e:
        return JSONResponse(content={"error": f"render falhou: {e}"}, status_code=500)

    import io as _io
    return StreamingResponse(
        _io.BytesIO(png_bytes),
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="thumb.png"'},
    )


def safe_filename(name: str) -> str | None:
    """Sanitize the output filename to be safe for filesystem and url, ending with .png."""
    if not name or not isinstance(name, str):
        return None
    cleaned = re.sub(r'[^a-zA-Z0-9._-]', '_', name)[:200]
    return cleaned if cleaned.lower().endswith('.png') else cleaned + '.png'


def _thumb_supabase_upload(url: str, key: str, bucket: str, png_bytes: bytes, filename: str) -> str:
    """Uploads PNG bytes to Supabase storage and returns the public URL."""
    base = url.rstrip("/")
    target = f"{base}/storage/v1/object/{bucket}/{filename}"
    r = requests.post(
        target,
        headers={
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Content-Type": "image/png",
            "x-upsert": "true",
            "Cache-Control": "public, max-age=31536000, immutable",
        },
        data=png_bytes,
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Supabase upload failed {r.status_code}: {r.text[:300]}")
    return f"{base}/storage/v1/object/public/{bucket}/{filename}"


@app.post("/render")
@app.post("/api/render")
async def render_endpoint(req: dict):
    """Replicates Node.js thumb-renderer POST /render endpoint.
    Renders an HTML document to PNG using internal Playwright, uploads to Supabase,
    and returns the public URL.
    """
    html = req.get("html") or ""
    viewport_width = int(req.get("viewport_width") or 1280)
    viewport_height = int(req.get("viewport_height") or 720)
    ms_delay = int(req.get("ms_delay") or 500)
    filename = req.get("filename") or ""
    supabase_url = req.get("supabase_url") or ""
    supabase_key = req.get("supabase_key") or ""
    supabase_bucket = req.get("supabase_bucket") or "thumbnails"

    if not html:
        return JSONResponse(content={"error": "html (string) required"}, status_code=400)
    if not supabase_url or not supabase_key:
        return JSONResponse(content={"error": "supabase_url and supabase_key required in body"}, status_code=400)

    # Sanitize/resolve filename
    final_name = safe_filename(filename)
    if not final_name:
        import uuid
        import time
        final_name = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:12]}.png"

    # Render HTML to PNG
    try:
        png_bytes = await _render_html_to_png(
            html,
            width=viewport_width,
            height=viewport_height,
            ms_delay=ms_delay
        )
    except Exception as e:
        return JSONResponse(content={"error": f"render failed: {e}"}, status_code=500)

    # Upload to Supabase Storage
    try:
        public_url = _thumb_supabase_upload(
            supabase_url,
            supabase_key,
            supabase_bucket,
            png_bytes,
            final_name
        )
        return {"url": public_url, "filename": final_name}
    except Exception as e:
        return JSONResponse(content={"error": f"upload failed: {e}"}, status_code=502)


sse = SseServerTransport("/mcp/messages/")
app.router.routes.append(Mount("/mcp/messages", app=sse.handle_post_message))

@app.get("/mcp/sse", tags=["MCP"])
async def handle_sse(request: Request):
    active_connections.add(request)
    async with sse.connect_sse(request.scope, request.receive, request._send) as (read_stream, write_stream):
        await mcp._mcp_server.run(read_stream, write_stream, mcp._mcp_server.create_initialization_options())
    print("SSE connection closed")

def process_video_request(
    text: str, person_image_url: str, person_name: str, bg_video_url: str = "",
    voice: str = "af_heart", overlay_bg_color: tuple = (232, 14, 64), version: str = "v1",
    gdrive_folder_id: str = "",
    subscribe_overlay_url: str = "", subscribe_overlay_drive_folder: str = "",
    subscribe_overlay_filename: str = "overlay-subscribe-new.mp4",
    subscribe_first_at: int = 30, subscribe_interval: int = 180,
    production_id: str = "",
    # V4 fields
    character_position: str = "random",
    subtitle_color_preset: str = "random",
    effect_overlay_ids: list = None,
    effect_layers: list = None,
    bg_video_folder_ids: list = None,
    max_bg_clips: int = 10,
    poof_remove_bg: bool = False,
) -> tuple[str, dict, str]:
    """Process video creation request."""
    if not text:
        return None, None, "Missing required field: text"
    if not person_image_url:
        return None, None, "Missing required field: person_image_url"
    # bg_video_url is optional when bg_video_folder_ids is provided (V4)
    if not bg_video_url and not bg_video_folder_ids:
        return None, None, "Missing required field: bg_video_url (or bg_video_folder_ids for v4)"
    if not person_name:
        person_name = "Narrator"
    if bg_video_url and not bg_video_url.startswith("http"):
        return None, None, "Invalid bg_video_url: should start with http"
    if not person_image_url.startswith("http"):
        return None, None, "Invalid person_image_url: should start with http"
    
    # Trusted domains — skip HEAD validation (Google Drive doesn't handle HEAD well)
    TRUSTED_DOMAINS = ["drive.google.com", "googleapis.com", "supabase.co", "cloudflare", "easypanel.host"]

    # Check background video (skip entirely for v4 with folder_ids, even if bg_video_url is set)
    if bg_video_url and not bg_video_folder_ids and not any(d in bg_video_url for d in TRUSTED_DOMAINS):
        try:
            response = requests.head(bg_video_url, timeout=10, allow_redirects=True)
            if response.status_code not in [200, 302, 303]:
                return None, None, f"Background video not accessible: {response.status_code}"
            ext = os.path.splitext(bg_video_url)[1].lower().split('?')[0]
            if ext and ext not in [".mp4", ".mov", ".avi", ".webm"]:
                return None, None, "Invalid bg_video_url: should be a video file"
        except Exception as e:
            return None, None, f"Error checking bg_video_url: {str(e)}"
    elif bg_video_url and not bg_video_folder_ids:
        print(f"[VALIDATE] Skipping HEAD check for trusted domain: {bg_video_url[:60]}...")
    else:
        print(f"[VALIDATE] Skipping bg_video_url check — v4 folder mode (bg_video_folder_ids={len(bg_video_folder_ids or [])})")

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
            "subscribe_overlay_url": subscribe_overlay_url,
            "subscribe_overlay_drive_folder": subscribe_overlay_drive_folder,
            "subscribe_overlay_filename": subscribe_overlay_filename,
            "subscribe_first_at": subscribe_first_at,
            "subscribe_interval": subscribe_interval,
            "production_id": production_id,
            # V4
            "character_position": character_position,
            "subtitle_color_preset": subtitle_color_preset,
            "effect_overlay_ids": effect_overlay_ids or [],
            "effect_layers": effect_layers or [],
            "bg_video_folder_ids": bg_video_folder_ids or [],
            "max_bg_clips": max_bg_clips,
            "poof_remove_bg": poof_remove_bg,
        },
        "created_at": time.time()
    }
    return video_id, video_data, ""

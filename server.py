from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
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
import torch
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
    
    num_threads = os.environ.get("NUM_THREADS", num_cores * 1.5)
    torch.set_num_threads(int(num_threads))

WORK_DIR = os.environ.get('WORK_DIR', os.getcwd())
TMP_DIR = os.path.join(WORK_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)
VIDEOS_DIR = os.path.join(WORK_DIR, "videos")
os.makedirs(VIDEOS_DIR, exist_ok=True)
SHELVE_FILE_PATH = os.path.join(WORK_DIR, "videos_db")

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def iterfile(path: str):
    with open(path, mode="rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            yield chunk

LANGUAGE_CONFIG = {
    'en-us': {
        'lang_code': 'a',
        'international': False,
    },
    'en': {
        'lang_code': 'a',
        'international': False,
    },
    'en-gb': {
        'lang_code': 'b',
        'international': False,
    },
    'es': {
        'lang_code': 'e',
        'international': True
    },
    'fr': {
        'lang_code': 'f',
        'international': True
    },
    'hi': {
        'lang_code': 'h',
        'international': True
    },
    'it': {
        'lang_code': 'i',
        'international': True
    },
    'pt': {
        'lang_code': 'p',
        'international': True
    },
    'ja': {
        'lang_code': 'j',
        'international': True
    },
    'zh': {
        'lang_code': 'z',
        'international': True
    },
}
LANGUAGE_VOICE_CONFIG = {
    'en-us': [
        'af_heart',
        'af_alloy', 
        'af_aoede', 
        'af_bella', 
        'af_jessica',
        'af_kore', 
        'af_nicole', 
        'af_nova', 
        'af_river', 
        'af_sarah', 
        'af_sky',
        'am_adam',
        'am_echo',
        'am_eric',
        'am_fenrir',
        'am_liam',
        'am_michael',
        'am_onyx',
        'am_puck',
        'am_santa'
    ],
    'en-gb': [
        'bf_alice',
        'bf_emma',
        'bf_isabella',
        'bf_lily',
        'bm_daniel',
        'bm_fable',
        'bm_george',
        'bm_lewis'
    ],
    'zh': [
        'zf_xiaobei',
        'zf_xiaoni',
        'zf_xiaoxiao',
        'zf_xiaoyi',
        'zm_yunjian',
        'zm_yunxi',
        'zm_yunxia',
        'zm_yunyang'
    ],
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
                # Re-queue videos that were in QUEUED state
                for video_id, video_data in videos.items():
                    if video_data['status'] == VideoStatus.QUEUED:
                        video_queue.put(video_id)
                    # Reset videos that were in PROCESSING state (they were interrupted)
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

# worker thread for processing videos
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
                    # Set status to processing
                    videos[video_id]["status"] = VideoStatus.PROCESSING
                    save_videos()  # Save state change
                    
                    data = videos[video_id]["data"]
                    
                    # Create video directory
                    video_dir = os.path.join(TMP_DIR, video_id)
                    os.makedirs(video_dir, exist_ok=True)
                    
                    try:
                        # Download background video
                        print(f"Downloading background video for {video_id}")
                        bg_extension = os.path.splitext(data["bg_video_url"])[1]
                        bg_video_path = os.path.join(video_dir, f"background{bg_extension}")
                        response = requests.get(data["bg_video_url"], stream=True, timeout=60)
                        if response.status_code == 200:
                            with open(bg_video_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        else:
                            raise Exception(f"Failed to download background video: {response.status_code}")
                        
                        # Download person image
                        print(f"Downloading person image for {video_id}")
                        person_extension = os.path.splitext(data["person_image_url"])[1]
                        person_image_path = os.path.join(video_dir, f"person{person_extension}")
                        response = requests.get(data["person_image_url"], stream=True, timeout=60)
                        if response.status_code == 200:
                            with open(person_image_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        else:
                            raise Exception(f"Failed to download person image: {response.status_code}")
                        
                    except Exception as download_error:
                        # Clean up on download failure
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
                    create_overlay(
                        person_image_path=person_image_path,
                        volume_icon_path="assets/icon_volume.png",
                        display_name=data["person_name"],
                        output_path=overlay_path,
                        subtitle_background_color=(0, 0, 0, 200),
                        font_path=font_path,
                    )
                    
                    print("creating narration")
                    sound_path = os.path.join(video_dir, "sound.wav")
                    segments = []
                    if LANGUAGE_VOICE_MAP[data["voice"]]["international"]:
                        captions, audio_length = create_tts_international(
                            text=data["text"],
                            output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                        )
                        max_line_length = 30
                        if LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"] == "z":
                            max_line_length = 15
                        segments = create_subtitle_segments_international(
                            captions=captions,
                            max_length=max_line_length,
                            lines=2,
                        )
                    else:
                        captions, audio_length = create_tts_english(
                            text=data["text"],
                            output_path=sound_path,
                            lang_code=LANGUAGE_VOICE_MAP[data["voice"]]["lang_code"],
                            voice=data["voice"],
                        )
                        
                        segments = create_subtitle_segments_english(
                            captions=captions,
                            max_length=30,
                            lines=2
                        )
                    
                    # Check version to decide subtitle style
                    version = data.get("version", "v1")
                    subtitle_path = os.path.join(video_dir, "subtitle.ass")
                    
                    print(f"Creating subtitle (version: {version})")
                    
                    if version == "v2":
                        # v2: Use word-by-word karaoke effect
                        print("Using v2 karaoke subtitle style (word-by-word animation)")
                        create_subtitle_v2_karaoke(
                            word_captions=captions,  # Use raw captions (word-level)
                            font_size=80,
                            output_path=subtitle_path,
                        )
                    else:
                        # v1: Use traditional static subtitles
                        print("Using v1 static subtitle style")
                        create_subtitle(
                            segments=segments,
                            font_size=80,
                            output_path=subtitle_path,
                        )
                    video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
                    print("rendering video")
                    render_video(
                        sound_path=sound_path,
                        subtitle_path=subtitle_path,
                        overlay_path=overlay_path,
                        audio_length=audio_length,
                        bg_video_path=bg_video_path,
                        output_path=video_path,
                    )
                    
                    # Clean up temporary files
                    try:
                        print(f"Cleaning up temporary files for video: {video_id}")
                        shutil.rmtree(video_dir)
                        print(f"Successfully removed temporary directory: {video_dir}")
                    except Exception as cleanup_error:
                        print(f"Warning: Failed to clean up temporary files for {video_id}: {cleanup_error}")
                    
                    videos[video_id]["status"] = VideoStatus.COMPLETED
                    save_videos()  # Save state change
                    print(f"Completed video: {video_id}")
                
                video_queue.task_done()
            else:
                # Sleep briefly when the queue is empty
                time.sleep(0.5)
        except Exception as e:
            print(f"Error in worker thread: {e}")
            # If there was an error processing, mark as failed
            if 'video_id' in locals() and video_id in videos:
                videos[video_id]["status"] = VideoStatus.FAILED
                videos[video_id]["error"] = str(e)
                save_videos()  # Save failure state
                
                # Clean up temporary files even on failure
                try:
                    if 'video_dir' in locals():
                        print(f"Cleaning up temporary files after error for video: {video_id}")
                        shutil.rmtree(video_dir)
                except Exception as cleanup_error:
                    print(f"Warning: Failed to clean up temporary files after error for {video_id}: {cleanup_error}")

# load videos at startup
load_videos()

# start worker thread
worker_thread = threading.Thread(target=process_video_queue, daemon=True)

### REST API endpoints ###
@app.get("/health")
def read_root():
    return {"status": "ok"}

# get available languages (and their voices)
@app.get("/api/languages")
def get_languages():
    return LANGUAGE_VOICE_CONFIG

# list all videos and their status
@app.get("/api/videos")
def list_videos():
    return [{"video_id": video_id, "status": video_data["status"]} for video_id, video_data in videos.items()]

# create a new video
@app.post("/api/videos")
def create_video(video: dict):
    # Get version parameter (v1 or v2)
    # v1 = static subtitles (original)
    # v2 = word-by-word karaoke effect
    version = video.get("version", "v1")
    
    # Get optional parameters with defaults
    voice = video.get("voice", "af_heart")
    overlay_bg_color = video.get("overlay_bg_color", (232, 14, 64))
    
    # bg_video_url is always required (n8n will provide it)
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
        version=version
    )
    
    if error:
        return {"error": error}
    
    # Store video in the tracking dictionary
    videos[video_id] = video_data
    save_videos()  # Save to persistent storage
    
    # Add to processing queue
    video_queue.put(video_id)
    
    return {"video_id": video_id, "status": VideoStatus.QUEUED}

# get video status
@app.get("/api/videos/{video_id}/status")
def get_video(video_id: str):
    if video_id in videos:
        return {
            "video_id": video_id, 
            "status": videos[video_id]["status"]
        }
    else:
        return {"video_id": video_id, "status": "not_found"}

# download a video
@app.get("/api/videos/{video_id}")
def download_video(video_id: str, download: bool = False):
    if video_id in videos and videos[video_id]["status"] == VideoStatus.COMPLETED:
        video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            return StreamingResponse(
                iterfile(video_path),
                media_type="video/mp4",
                headers={
                    "Content-Disposition": f'attachment; filename="{video_id}.mp4"'
                }
            )
    elif video_id in videos:
        if videos[video_id]["status"] == VideoStatus.FAILED:
            return JSONResponse(
                content={"video_id": video_id, "status": VideoStatus.FAILED},
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        if videos[video_id]["status"] == VideoStatus.PROCESSING:
            return JSONResponse(
                content={"video_id": video_id, "status": VideoStatus.PROCESSING},
                status_code=status.HTTP_202_ACCEPTED,
            )

    return JSONResponse(
        content={"video_id": video_id, "status": VideoStatus.NOT_FOUND},
        status_code=status.HTTP_404_NOT_FOUND,
    )

# delete a video
@app.delete("/api/videos/{video_id}")
def delete_video(video_id: str):
    if video_id in videos:
        video_path = os.path.join(VIDEOS_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)
        del videos[video_id]
        save_videos()
        return {"video_id": video_id, "status": VideoStatus.DELETED}
    else:
        return {"video_id": video_id, "status": VideoStatus.NOT_FOUND}

# get queue status
@app.get("/api/queue")
def get_queue_status():
    queue_size = video_queue.qsize()
    queued_videos = [v for v in videos.values() if v["status"] == VideoStatus.QUEUED]
    processing_videos = [v for v in videos.values() if v["status"] == VideoStatus.PROCESSING]
    
    return {
        "queue_size": queue_size,
        "queued": len(queued_videos),
        "processing": len(processing_videos)
    }

### todo add the MCP server ###

## tools
@mcp.tool()
def list_languages_mcp() -> dict:
    """
    List available languages and their voices.
    """
    return LANGUAGE_VOICE_CONFIG

@mcp.tool()
def create_video_mcp(
    text: Annotated[str, Field(description="The text to be narrated in the video.")],
    person_image_url: Annotated[str, Field(description="URL of the person's image to be used in the video.")],
    person_name: Annotated[str, Field(description="Name of the person to be displayed in the video.")],
    bg_video_url: Annotated[str, Field(description="URL of the background video to be used.")],
    voice: Optional[AvailableVoices] = Field(description="Voice to be used for narration. Defaults to 'af_heart' if not provided.", default=None)) -> dict:
    overlay_bg_color: Optional[tuple] = Field(description="Background color for overlay. Defaults to (232, 14, 64) if not provided.", default=(232, 14, 64))
    """
    Create a new narrated video with the provided content.
    Args:
        text: The text to be narrated in the video.
        person_image_url: URL of the person's image to be used in the video.
        person_name: Name of the person to be displayed in the video.
        bg_video_url: URL of the background video to be used.
        voice: Voice to be used for narration. Defaults to None.
    Returns:
        A dictionary containing the video_id and status of the created video
    """

    print(f"Creating video with text: {text}")
    
    # Set default values if not provided
    voice_str = voice if voice else "af_heart"
    bg_color = overlay_bg_color if overlay_bg_color else (232, 14, 64)
    
    video_id, video_data, error = process_video_request(
        text=text,
        person_image_url=person_image_url,
        person_name=person_name,
        bg_video_url=bg_video_url,
        voice=voice_str,
        overlay_bg_color=bg_color
    )
    
    if error:
        return {"error": error}
    
    # Store video in the tracking dictionary
    videos[video_id] = video_data
    save_videos()  # Save to persistent storage
    
    # Add to processing queue
    video_queue.put(video_id)
    
    print(f"Creating video with text: {text}")
    return {"video_id": video_id, "status": VideoStatus.QUEUED.value}

## mount the MCP server
sse = SseServerTransport("/mcp/messages/")
app.router.routes.append(Mount("/mcp/messages", app=sse.handle_post_message))

@app.get("/mcp/sse", tags=["MCP"])
async def handle_sse(request: Request):
    """
    SSE endpoint that connects to the MCP server

    This endpoint establishes a Server-Sent Events connection with the client
    and forwards communication to the Model Context Protocol server.
    """
    
    active_connections.add(request)
    
    async with sse.connect_sse(request.scope, request.receive, request._send) as (
        read_stream,
        write_stream,
    ):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )
    
    print("SSE connection closed")
        
def process_video_request(
    text: str,
    person_image_url: str,
    person_name: str,
    bg_video_url: str,
    voice: str = "af_heart",
    overlay_bg_color: tuple = (232, 14, 64),
    version: str = "v1"
) -> tuple[str, dict, str]:
    """
    Process video creation request by validating inputs and checking resource availability.
    
    Args:
        text: The text to be narrated in the video
        person_image_url: URL of the person's image to use
        person_name: Name of the person to display
        bg_video_url: URL of the background video
        voice: Voice ID to use for narration (default: af_heart)
        overlay_bg_color: Background color for overlay (default: (232, 14, 64))
        version: Version of the API (v1 or v2, default: v1)
        
    Returns:
        tuple containing:
        - video_id: Unique ID for the video
        - video_data: Dictionary with video configuration
        - error: Error message if any, empty string if successful
    """
    # Validate required fields
    required_fields = ["text", "person_image_url", "person_name", "bg_video_url"]
    for field_name, field_value in [
        ("text", text),
        ("person_image_url", person_image_url),
        ("person_name", person_name),
        ("bg_video_url", bg_video_url)
    ]:
        if not field_value:
            return None, None, f"Missing required field: {field_name}"
    
    # Validate URLs and check if resources exist
    if not bg_video_url.startswith("http"):
        return None, None, "Invalid bg_video_url: should start with http"
    
    if not person_image_url.startswith("http"):
        return None, None, "Invalid person_image_url: should start with http"
    
    # Check if background video exists and validate extension
    try:
        response = requests.head(bg_video_url, timeout=10, allow_redirects=True)
        
        # Google Drive pode retornar 303 (See Other) ou 302 (Found) para redirects
        # Ambos são válidos e indicam que o arquivo existe
        if response.status_code not in [200, 302, 303]:
            return None, None, f"Background video not accessible: {response.status_code}"
        
        # Validate extension for standard URLs
        # Google Drive URLs don't have extensions in the URL, so we skip validation for them
        if "drive.google.com" not in bg_video_url:
            extension = os.path.splitext(bg_video_url)[1].lower()
            if extension not in [".mp4", ".mov", ".avi"]:
                return None, None, "Invalid bg_video_url: should be a video file (.mp4, .mov, .avi)"
    except Exception as e:
        return None, None, f"Error checking bg_video_url: {str(e)}"

    # Check if person image exists and validate extension
    try:
        response = requests.head(person_image_url, timeout=10)
        if response.status_code != 200:
            return None, None, f"Person image not accessible: {response.status_code}"
        
        extension = os.path.splitext(person_image_url)[1].lower()
        if extension not in [".jpg", ".jpeg", ".png"]:
            return None, None, "Invalid person_image_url: should be an image file (.jpg, .jpeg, .png)"
    except Exception as e:
        return None, None, f"Error checking person_image_url: {str(e)}"
    
    # Validate voice
    if voice not in LANGUAGE_VOICE_MAP:
        return None, None, f"Invalid voice: {voice}"
    
    video_id = str(uuid.uuid4())
            
    # Create video data structure (no downloading yet)
    video_data = {
        "id": video_id,
        "status": VideoStatus.QUEUED,
        "data": {
            "text": text,
            "person_name": person_name,
            "voice": voice,
            "overlay_bg_color": overlay_bg_color,
            "person_image_url": person_image_url,
            "bg_video_url": bg_video_url,
            "version": version,
        },
        "created_at": time.time()
    }
    
    return video_id, video_data, ""
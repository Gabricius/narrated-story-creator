#!/usr/bin/env python3
"""
YouTube Upload Script — runs on VPS host (not container)
Called via SSH from n8n Job 6.

Usage:
  python3 /root/youtube_upload.py --json '{...}'

Input JSON:
{
  "drive_file_id": "1BQThsKaWXMPsiL9iFUgMuwTPONAJr89S",
  "title": "Video Title",
  "description": "Video description",
  "tags": "stories,narration",
  "category_id": "24",
  "privacy_status": "private",
  "publish_at": "2026-02-19T16:00:00.000Z",
  "language": "pt-BR",
  "thumb_url": "https://cdn.renderform.io/...",
  "channel_credential": "nora-scott"
}

Output (JSON to stdout):
{
  "success": true,
  "youtube_video_id": "dQw4w9WgXcQ",
  "youtube_url": "https://youtu.be/dQw4w9WgXcQ"
}

Setup:
  pip3 install google-api-python-client google-auth-oauthlib google-auth-httplib2 requests

  Environment variables (in /root/.env or passed via SSH):
    SUPABASE_URL=https://xxx.supabase.co
    SUPABASE_ANON_KEY=eyJ...

  Credentials are stored in the youtube_credentials table in Supabase.
  Any VPS with these env vars can upload to any channel.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import requests

# Google API imports
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

# Load env from /root/.env if it exists (for standalone SSH execution)
_env_file = os.path.expanduser("~/.env")
if os.path.exists(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "") or os.environ.get("SUPABASE_SERVICE_KEY", "")
TEMP_DIR = "/tmp/youtube-uploads"
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]

# ═══════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════

def log(msg):
    """Log to stderr (stdout is reserved for JSON output)"""
    print(f"[YT-Upload] {msg}", file=sys.stderr)


def get_credentials(channel_credential):
    """Load OAuth2 credentials from Supabase youtube_credentials table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_ANON_KEY must be set. "
            "Add them to /root/.env or pass via environment."
        )
    
    log(f"Fetching credentials for '{channel_credential}' from Supabase...")
    
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/youtube_credentials"
        f"?credential_name=eq.{channel_credential}&select=client_id,client_secret,refresh_token",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        timeout=15,
    )
    
    if resp.status_code != 200:
        raise RuntimeError(f"Supabase query failed: HTTP {resp.status_code} — {resp.text[:200]}")
    
    rows = resp.json()
    if not rows:
        raise FileNotFoundError(
            f"Credential '{channel_credential}' not found in Supabase. "
            f"Add it via Pipeline Manager → Editar Canal → YouTube Credential → + Nova."
        )
    
    cred_data = rows[0]
    
    credentials = Credentials(
        token=None,
        refresh_token=cred_data["refresh_token"],
        client_id=cred_data["client_id"],
        client_secret=cred_data["client_secret"],
        token_uri="https://oauth2.googleapis.com/token"
    )
    
    # Force refresh — uses whatever scopes the token was originally granted
    from google.auth.transport.requests import Request
    credentials.refresh(Request())
    log(f"Token refreshed, expires: {credentials.expiry}")
    
    return credentials


def download_from_drive(drive_file_id, folder_id, credentials=None):
    """Download file from Google Drive via rclone (already configured on host)"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Clean any leftover temp files from previous failed runs
    import shutil
    for item in os.listdir(TEMP_DIR):
        item_path = os.path.join(TEMP_DIR, item)
        if item.startswith("dl_"):
            shutil.rmtree(item_path, ignore_errors=True)
        elif item.endswith(".partial"):
            os.remove(item_path)
    
    if not folder_id:
        raise ValueError("gdrive_folder_id not provided")
    
    log(f"Looking for file {drive_file_id} in Drive folder...")
    
    # Step 1: List files in Drive folder, find our file by ID
    lsj = subprocess.run(
        ["rclone", "lsjson", "gdrive:", "--drive-root-folder-id", folder_id],
        capture_output=True, text=True, timeout=60
    )
    
    if lsj.returncode != 0:
        raise RuntimeError(f"rclone lsjson failed: {lsj.stderr[:300]}")
    
    filename = None
    try:
        entries = json.loads(lsj.stdout)
        for entry in entries:
            if entry.get("ID") == drive_file_id:
                filename = entry["Name"]
                size_mb = entry.get("Size", 0) / (1024 * 1024)
                log(f"Found: {filename} ({size_mb:.0f} MB)")
                break
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse lsjson: {lsj.stdout[:200]}")
    
    if not filename:
        raise FileNotFoundError(f"File ID {drive_file_id} not found in Drive folder {folder_id}")
    
    # Step 2: Copy file by name
    local_path = os.path.join(TEMP_DIR, filename)
    
    # Remove if exists from previous attempt
    if os.path.exists(local_path):
        os.remove(local_path)
    
    log(f"Downloading {filename}...")
    start = time.time()
    
    result = subprocess.run(
        ["rclone", "copyto", 
         f"gdrive:{filename}", local_path,
         "--drive-root-folder-id", folder_id,
         "--drive-acknowledge-abuse"],
        capture_output=True, text=True, timeout=1800  # 30 min max
    )
    
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise FileNotFoundError(
            f"rclone copyto failed for {filename}.\n"
            f"stderr: {result.stderr[:500]}"
        )
    
    elapsed = time.time() - start
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    log(f"Done: {size_mb:.1f} MB in {elapsed:.0f}s")
    
    return local_path


def find_local_video(drive_file_id):
    """Check if video exists locally in server's video dir"""
    video_dirs = ["/app/videos", "/root/videos", TEMP_DIR]
    for d in video_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if drive_file_id in f or f.endswith('.mp4'):
                path = os.path.join(d, f)
                if os.path.getsize(path) > 0:
                    return path
    return None


def upload_to_youtube(credentials, file_path, metadata):
    """Upload video to YouTube using resumable upload"""
    youtube = build("youtube", "v3", credentials=credentials)
    
    tags = metadata.get("tags", "")
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    
    body = {
        "snippet": {
            "title": metadata["title"][:100].replace("{", "").replace("}", ""),
            "description": metadata.get("description", "")[:5000].replace("{", "").replace("}", ""),
            "tags": tags,
            "categoryId": metadata.get("category_id", "24"),
            "defaultLanguage": metadata.get("language", "pt-BR"),
            "defaultAudioLanguage": metadata.get("language", "pt-BR"),
        },
        "status": {
            "privacyStatus": metadata.get("privacy_status", "private"),
            "selfDeclaredMadeForKids": False,
        }
    }
    
    # Add publishAt for scheduling
    publish_at = metadata.get("publish_at", "")
    if publish_at and metadata.get("privacy_status") == "private":
        body["status"]["publishAt"] = publish_at
    
    file_size = os.path.getsize(file_path)
    log(f"Uploading to YouTube: {metadata['title'][:50]}... ({file_size / 1024 / 1024:.0f} MB)")
    
    # Use resumable upload with 10MB chunks
    media = MediaFileUpload(
        file_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024  # 10MB chunks
    )
    
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    
    response = None
    retries = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                log(f"Upload progress: {pct}%")
        except HttpError as e:
            if e.resp.status in [500, 502, 503, 504] and retries < 5:
                retries += 1
                wait = 2 ** retries
                log(f"Server error, retrying in {wait}s... (attempt {retries})")
                time.sleep(wait)
            else:
                raise
    
    video_id = response["id"]
    log(f"Upload complete! Video ID: {video_id}")
    
    return video_id


def set_thumbnail(credentials, video_id, thumb_url):
    """Download thumbnail and set it on the YouTube video"""
    if not thumb_url:
        return
    
    log(f"Setting thumbnail from: {thumb_url[:60]}...")
    
    try:
        # Download thumbnail
        resp = requests.get(thumb_url, timeout=30)
        resp.raise_for_status()
        
        thumb_path = os.path.join(TEMP_DIR, f"thumb_{video_id}.jpg")
        with open(thumb_path, 'wb') as f:
            f.write(resp.content)
        
        # Upload to YouTube
        youtube = build("youtube", "v3", credentials=credentials)
        media = MediaFileUpload(thumb_path, mimetype="image/jpeg")
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        
        os.remove(thumb_path)
        log("Thumbnail set successfully")
        
    except Exception as e:
        log(f"Thumbnail failed (non-critical): {e}")


def cleanup(file_path):
    """Remove temporary file"""
    try:
        if file_path and os.path.exists(file_path) and TEMP_DIR in file_path:
            os.remove(file_path)
            log(f"Cleaned up: {file_path}")
    except Exception as e:
        log(f"Cleanup error: {e}")


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Upload video to YouTube")
    parser.add_argument("--json", required=False, help="JSON input with video metadata")
    parser.add_argument("--update", action="store_true", help="Self-update from GitHub")
    parser.add_argument("--test", action="store_true", help="Test Supabase connection")
    args = parser.parse_args()
    
    # ── Self-update mode ──
    if args.update:
        url = os.environ.get("UPLOAD_SCRIPT_URL", "")
        if not url:
            # Try to read from .env
            env_file = os.path.expanduser("~/.env")
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        if line.strip().startswith("UPLOAD_SCRIPT_URL="):
                            url = line.strip().split("=", 1)[1].strip().strip('"')
            if not url:
                print("Set UPLOAD_SCRIPT_URL in /root/.env or environment")
                sys.exit(1)
        
        log(f"Updating from {url}")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(__file__, 'w') as f:
                f.write(resp.text)
            log("Updated successfully")
        else:
            log(f"Update failed: HTTP {resp.status_code}")
            sys.exit(1)
        sys.exit(0)
    
    # ── Test mode ──
    if args.test:
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/youtube_credentials?select=credential_name,channel_name",
                headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"},
                timeout=10,
            )
            creds = resp.json()
            print(f"OK — {len(creds)} credencial(is):")
            for c in creds:
                print(f"  {c['credential_name']} ({c['channel_name']})")
        except Exception as e:
            print(f"ERRO: {e}")
            sys.exit(1)
        sys.exit(0)
    
    # ── Upload mode ──
    if not args.json:
        parser.print_help()
        sys.exit(1)
    
    try:
        params = json.loads(args.json)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    
    local_path = None
    
    try:
        # 1. Get credentials
        channel_cred = params.get("channel_credential", "default")
        log(f"Channel credential: {channel_cred}")
        credentials = get_credentials(channel_cred)
        
        # 2. Get video file
        drive_file_id = params.get("drive_file_id", "")
        folder_id = params.get("gdrive_folder_id", "")
        
        if drive_file_id:
            local_path = find_local_video(drive_file_id)
            if local_path:
                log(f"Found local file: {local_path}")
            else:
                local_path = download_from_drive(drive_file_id, folder_id)
        else:
            raise ValueError("drive_file_id is required")
        
        # 3. Upload to YouTube
        video_id = upload_to_youtube(credentials, local_path, params)
        
        # 4. Set thumbnail
        set_thumbnail(credentials, video_id, params.get("thumb_url", ""))
        
        # 5. Output result (JSON to stdout)
        result = {
            "success": True,
            "youtube_video_id": video_id,
            "youtube_url": f"https://youtu.be/{video_id}"
        }
        print(json.dumps(result))
        
    except Exception as e:
        log(f"ERROR: {e}")
        result = {"success": False, "error": str(e)}
        print(json.dumps(result))
        sys.exit(1)
    
    finally:
        cleanup(local_path)


if __name__ == "__main__":
    main()

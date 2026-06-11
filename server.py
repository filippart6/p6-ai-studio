#!/usr/bin/env python3
import os
import io
import json
import sqlite3
import uuid
import requests
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import (Flask, request, jsonify, send_from_directory,
                   Response, session, redirect, url_for)

# ── Google Drive ───────────────────────────────────────────────────────────────
DRIVE_FOLDER_ID = '14brOnkWE8JIjmcl8Y1j4j4HNqtGH-BIz'

def get_drive_service():
    creds_json = os.environ.get('GOOGLE_SERVICE_ACCOUNT_JSON')
    if not creds_json:
        return None
    try:
        from googleapiclient.discovery import build
        from google.oauth2.service_account import Credentials
        info = json.loads(creds_json)
        creds = Credentials.from_service_account_info(
            info, scopes=['https://www.googleapis.com/auth/drive'])
        return build('drive', 'v3', credentials=creds, cache_discovery=False)
    except Exception as e:
        print(f'[drive] init error: {e}', flush=True)
        return None

def drive_upload(file_bytes, filename, mime_type='image/jpeg'):
    """Upload bytes to Drive, make public, return (file_id, direct_url)."""
    service = get_drive_service()
    if not service:
        return None, None
    try:
        from googleapiclient.http import MediaIoBaseUpload
        meta = {'name': filename, 'parents': [DRIVE_FOLDER_ID]}
        media = MediaIoBaseUpload(io.BytesIO(file_bytes), mimetype=mime_type)
        f = service.files().create(body=meta, media_body=media,
                                   fields='id').execute()
        fid = f['id']
        service.permissions().create(
            fileId=fid, body={'type': 'anyone', 'role': 'reader'}).execute()
        url = f'https://drive.google.com/uc?export=view&id={fid}'
        print(f'[drive] uploaded {filename} → {url}', flush=True)
        return fid, url
    except Exception as e:
        print(f'[drive] upload error: {e}', flush=True)
        return None, None

def drive_list_images():
    """List image files in the Drive folder, newest first."""
    service = get_drive_service()
    if not service:
        return []
    try:
        res = service.files().list(
            q=f"'{DRIVE_FOLDER_ID}' in parents and trashed=false and mimeType contains 'image/'",
            fields='files(id,name,thumbnailLink,createdTime)',
            orderBy='createdTime desc',
            pageSize=50
        ).execute()
        return [{
            'id':        f['id'],
            'name':      f.get('name', ''),
            'thumbnail': f.get('thumbnailLink', '').replace('=s220', '=s400'),
            'url':       f'https://drive.google.com/uc?export=view&id={f["id"]}',
            'created':   f.get('createdTime', ''),
        } for f in res.get('files', [])]
    except Exception as e:
        print(f'[drive] list error: {e}', flush=True)
        return []

app = Flask(__name__, static_folder='public')

ALLOWED_EXTENSIONS = {'.wav', '.mp3'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / 'history.db'


def load_env():
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()
app.secret_key = os.environ.get('SECRET_KEY', 'change-me-in-production')


# ── Database ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS image_history (
                id              TEXT PRIMARY KEY,
                user            TEXT NOT NULL,
                prompt          TEXT NOT NULL,
                neg_prompt      TEXT,
                size            TEXT,
                model           TEXT,
                filename        TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                guidance_scale  REAL DEFAULT 7.5,
                steps           INTEGER DEFAULT 30
            )
        ''')
        # Migrate existing DBs that lack the new columns
        for col, default in [('guidance_scale', '7.5'), ('steps', '30')]:
            try:
                conn.execute(f'ALTER TABLE image_history ADD COLUMN {col} REAL DEFAULT {default}')
            except Exception:
                pass
        # Migrate: add media_type + video_meta columns to existing image_history
        for col, default in [('media_type', "'image'"), ('video_meta', 'NULL')]:
            try:
                conn.execute(f'ALTER TABLE image_history ADD COLUMN {col} TEXT DEFAULT {default}')
            except Exception:
                pass
        conn.execute('''
            CREATE TABLE IF NOT EXISTS elements (
                id          TEXT PRIMARY KEY,
                user        TEXT NOT NULL,
                name        TEXT NOT NULL,
                label       TEXT NOT NULL,
                type        TEXT NOT NULL,
                images      TEXT NOT NULL,
                description TEXT,
                created_at  TEXT NOT NULL
            )
        ''')


init_db()


# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def get_api_key():
    return os.environ.get('ELEVENLABS_API_KEY', '').strip() or None


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if username == os.environ.get('LOGIN_USER', '') and \
           password == os.environ.get('LOGIN_PASSWORD', ''):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        return redirect(url_for('login') + '?error=1')
    return send_from_directory(app.static_folder, 'login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ── Static uploads ────────────────────────────────────────────────────────────

@app.route('/uploads/<filename>')
def serve_upload(filename):
    # Reference images must be publicly accessible so Seedream can fetch them
    return send_from_directory(UPLOADS_DIR, filename)


# ── App routes ────────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/upload-reference', methods=['POST'])
@login_required
def upload_reference():
    """Save a reference image and return its public URL."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    f = request.files['image']
    ext = Path(f.filename).suffix.lower() if f.filename else '.jpg'
    if ext not in {'.jpg', '.jpeg', '.png', '.webp'}:
        ext = '.jpg'
    filename = f'ref_{uuid.uuid4().hex}{ext}'
    (UPLOADS_DIR / filename).write_bytes(f.read())
    # Use Railway's public domain if available, otherwise fall back to request host
    railway_domain = os.environ.get('RAILWAY_PUBLIC_DOMAIN')
    if railway_domain:
        url = f"https://{railway_domain}/uploads/{filename}"
    else:
        url = f"http://localhost:{os.environ.get('PORT', 3000)}/uploads/{filename}"
    return jsonify({'url': url, 'filename': filename})


@app.route('/api/clone-voice', methods=['POST'])
@login_required
def clone_voice():
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'ElevenLabs API key not configured'}), 500

    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    voice_name = request.form.get('voiceName', '').strip()
    description = request.form.get('description', '').strip()
    noise_reduction = request.form.get('noiseReduction', 'false').lower() == 'true'

    if not voice_name:
        return jsonify({'error': 'Voice name is required'}), 400

    ext = Path(audio_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Only .wav and .mp3 files are supported'}), 400

    audio_bytes = audio_file.read()
    if len(audio_bytes) > MAX_FILE_SIZE:
        return jsonify({'error': 'File size must be under 50 MB'}), 400

    content_type = 'audio/wav' if ext == '.wav' else 'audio/mpeg'
    files = [('files', (audio_file.filename, audio_bytes, content_type))]
    data = {'name': voice_name, 'remove_background_noise': 'true' if noise_reduction else 'false'}
    if description:
        data['description'] = description

    try:
        resp = requests.post(
            'https://api.elevenlabs.io/v1/voices/add',
            headers={'xi-api-key': api_key},
            files=files, data=data, timeout=120
        )
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 502

    try:
        resp_data = resp.json()
    except Exception:
        resp_data = {}

    if not resp.ok:
        detail = resp_data.get('detail', {})
        msg = detail.get('message', str(resp_data)) if isinstance(detail, dict) else str(detail)
        return jsonify({'error': msg}), resp.status_code

    return jsonify({'success': True, 'voiceId': resp_data.get('voice_id'), 'voiceName': voice_name})


@app.route('/api/voices')
@login_required
def list_voices():
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'API key not configured'}), 500
    try:
        resp = requests.get('https://api.elevenlabs.io/v1/voices',
                            headers={'xi-api-key': api_key}, timeout=30)
        data = resp.json()
        cloned = [v for v in data.get('voices', []) if v.get('category') == 'cloned']
        return jsonify({'voices': cloned})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts/<voice_id>', methods=['POST'])
@login_required
def text_to_speech(voice_id):
    api_key = get_api_key()
    if not api_key:
        return jsonify({'error': 'ElevenLabs API key not configured'}), 500

    body = request.get_json()
    if not body or not body.get('text', '').strip():
        return jsonify({'error': 'Text is required'}), 400

    text = body['text'].strip()
    if len(text) > 5000:
        return jsonify({'error': 'Text must be under 5000 characters'}), 400

    payload = {
        'text': text,
        'model_id': body.get('model_id', 'eleven_multilingual_v2'),
        'voice_settings': {
            'stability': float(body.get('voice_settings', {}).get('stability', 0.5)),
            'similarity_boost': float(body.get('voice_settings', {}).get('similarity_boost', 0.75)),
            'style': float(body.get('voice_settings', {}).get('style', 0.0)),
            'use_speaker_boost': bool(body.get('voice_settings', {}).get('use_speaker_boost', True))
        }
    }

    try:
        resp = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}',
            headers={'xi-api-key': api_key, 'Content-Type': 'application/json', 'Accept': 'audio/mpeg'},
            json=payload, timeout=60
        )
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 502

    if not resp.ok:
        try:
            err = resp.json()
            detail = err.get('detail', {})
            msg = detail.get('message', str(err)) if isinstance(detail, dict) else str(detail)
        except Exception:
            msg = f'ElevenLabs error {resp.status_code}'
        return jsonify({'error': msg}), resp.status_code

    return Response(resp.content, status=200, mimetype='audio/mpeg',
                    headers={'Content-Disposition': 'inline; filename="speech.mp3"'})


def _aspect_ratio_from_size(w, h):
    """Pick the closest standard aspect-ratio string for the given pixel size."""
    if not w or not h:
        return '1:1'
    candidates = {'1:1': 1.0, '16:9': 16/9, '9:16': 9/16, '4:3': 4/3,
                  '3:4': 3/4, '21:9': 21/9, '3:2': 3/2, '2:3': 2/3}
    target = w / h
    return min(candidates.items(), key=lambda kv: abs(kv[1] - target))[0]


def _luma_generate(prompt, aspect_ratio, ref_url=None):
    """Submit a Luma generation, poll until done, return image bytes + final state.

    Returns (img_bytes, model_used, error) — img_bytes is None on error."""
    api_key = os.environ.get('LUMA_AGENTS_API_KEY', '').strip()
    if not api_key:
        return None, None, 'LUMA_AGENTS_API_KEY env var not set'

    # Luma supports "uni-1" (base) and "uni-1-max" (higher quality). Allow override via env var.
    model_name = os.environ.get('LUMA_MODEL_ID', 'uni-1-max').strip() or 'uni-1-max'

    payload = {'model': model_name, 'prompt': prompt}
    if aspect_ratio:
        payload['aspect_ratio'] = aspect_ratio
    if ref_url and ref_url.startswith('http'):
        payload['image_ref'] = [{'url': ref_url}]

    print(f'[luma] submitting payload keys={list(payload.keys())} model={model_name}', flush=True)
    try:
        resp = requests.post(
            'https://agents.lumalabs.ai/v1/generations',
            headers={'Authorization': f'Bearer {api_key}',
                     'Content-Type': 'application/json'},
            json=payload, timeout=30
        )
    except requests.exceptions.RequestException as e:
        return None, model_name, f'Luma network error: {e}'

    try:
        data = resp.json()
    except Exception:
        return None, model_name, f'Luma invalid response (status {resp.status_code})'

    if not resp.ok:
        msg = (data.get('error') if isinstance(data.get('error'), str)
               else (data.get('error') or {}).get('message')) or f'Luma API error {resp.status_code}: {data}'
        return None, model_name, msg

    gen_id = data.get('id') or data.get('generation_id')
    if not gen_id:
        return None, model_name, f'Luma did not return generation id: {data}'

    # Poll up to ~90s
    import time
    deadline = time.time() + 90
    final = None
    while time.time() < deadline:
        time.sleep(2)
        try:
            poll = requests.get(
                f'https://agents.lumalabs.ai/v1/generations/{gen_id}',
                headers={'Authorization': f'Bearer {api_key}'}, timeout=15
            )
            pdata = poll.json()
        except Exception as e:
            print(f'[luma] poll error: {e}', flush=True)
            continue
        state = pdata.get('state') or pdata.get('status')
        print(f'[luma] {gen_id} → state={state}', flush=True)
        if state in ('completed', 'succeeded'):
            final = pdata
            break
        if state == 'failed':
            return None, model_name, f'Luma generation failed: {pdata.get("failure_reason") or pdata}'

    if not final:
        return None, model_name, 'Luma generation timed out after 90s'

    # Extract URL — try a few shapes
    url = None
    out = final.get('output')
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, dict):
            url = first.get('url') or first.get('image_url')
        elif isinstance(first, str):
            url = first
    elif isinstance(out, dict):
        url = out.get('url') or out.get('image_url')
    if not url:
        # Try assets.image
        assets = final.get('assets') or {}
        url = assets.get('image') or assets.get('url')
    if not url:
        return None, model_name, f'Luma succeeded but no URL found: {final}'

    try:
        img_resp = requests.get(url, timeout=60)
        img_resp.raise_for_status()
        return img_resp.content, model_name, None
    except Exception as e:
        return None, model_name, f'Failed to download Luma image: {e}'


@app.route('/api/generate-image', methods=['POST'])
@login_required
def generate_image():
    body = request.get_json()
    if not body or not body.get('prompt', '').strip():
        return jsonify({'error': 'Prompt is required'}), 400

    requested_model = (body.get('model') or '').strip().lower()
    prompt_text = body['prompt'].strip()
    width  = int(body.get('width',  1920))
    height = int(body.get('height', 1920))

    # ── Dispatch: Luma Uni ───────────────────────────────────────────────
    if requested_model.startswith('uni'):
        aspect = _aspect_ratio_from_size(width, height)
        # Prefer Drive URL ref; ignore base64 (Luma needs HTTPS URL)
        ref_url = body.get('reference_image_url', '') or None
        img_bytes, model_used, err = _luma_generate(prompt_text, aspect, ref_url)
        if err:
            print(f'[luma] error: {err}', flush=True)
            return jsonify({'error': err}), 502

        filename = f"{uuid.uuid4().hex}.jpg"
        _, drive_url = drive_upload(img_bytes, filename)
        if drive_url:
            image_url = drive_url
        else:
            (UPLOADS_DIR / filename).write_bytes(img_bytes)
            image_url = f'/uploads/{filename}'

        entry_id = uuid.uuid4().hex
        user = session.get('username', 'admin')
        size_str = f'{width}x{height}'
        with get_db() as conn:
            conn.execute(
                '''INSERT INTO image_history
                   (id,user,prompt,neg_prompt,size,model,filename,created_at,guidance_scale,steps)
                   VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (entry_id, user, prompt_text,
                 body.get('negative_prompt', '') or '',
                 size_str, model_used, image_url,
                 datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                 0.0, 0)
            )
        return jsonify({'url': image_url, 'id': entry_id})

    # ── Default: Seedream ────────────────────────────────────────────────
    api_key = os.environ.get('SEEDREAM_API_KEY', '').strip()
    if not api_key:
        return jsonify({'error': 'Seedream API key not configured'}), 500

    size_str = f"{width}x{height}"
    guidance = float(body.get('guidance_scale', 7.5))
    steps    = int(body.get('steps', 30))

    payload = {
        'model': os.environ.get('SEEDREAM_MODEL_ID', 'ep-20251203184030-sffv6'),
        'prompt': prompt_text,
        'size': size_str,
        'n': 1,
        'watermark': False,
    }
    if body.get('negative_prompt'):
        payload['negative_prompt'] = body['negative_prompt']

    ref_url = body.get('reference_image_url', '')
    ref_b64 = body.get('reference_image_b64', '')
    if ref_url:
        # Drive URL — real HTTPS, Seedream can fetch it
        payload['image'] = ref_url
        print(f'[ref-image] sending Drive URL: {ref_url}', flush=True)
    elif ref_b64:
        if not ref_b64.startswith('data:'):
            ref_b64 = f'data:image/jpeg;base64,{ref_b64}'
        payload['image'] = ref_b64
        print(f'[ref-image] sending as data URI, length={len(ref_b64)}', flush=True)

    print(f'[generate] payload keys: {list(payload.keys())}', flush=True)

    try:
        resp = requests.post(
            'https://ark.ap-southeast.bytepluses.com/api/v3/images/generations',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload, timeout=120
        )
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 502

    try:
        data = resp.json()
    except Exception:
        return jsonify({'error': f'Invalid response (status {resp.status_code})'}), 502

    print(f'[generate] API status={resp.status_code} response keys={list(data.keys())}', flush=True)
    if 'usage' in data:
        print(f'[generate] usage={data["usage"]}', flush=True)

    if not resp.ok:
        msg = data.get('error', {}).get('message') or f'API error {resp.status_code}'
        print(f'[generate] API error: {msg}', flush=True)
        return jsonify({'error': msg}), resp.status_code

    try:
        remote_url = data['data'][0]['url']
    except (KeyError, IndexError):
        return jsonify({'error': 'No image returned from API'}), 502

    # Download image from Seedream
    try:
        img_resp = requests.get(remote_url, timeout=60)
        img_resp.raise_for_status()
        img_bytes = img_resp.content
    except Exception as e:
        return jsonify({'error': f'Failed to download image: {str(e)}'}), 502

    filename = f"{uuid.uuid4().hex}.jpg"

    # Try uploading to Google Drive first
    _, drive_url = drive_upload(img_bytes, filename)

    if drive_url:
        image_url = drive_url
    else:
        # Fallback: save locally
        (UPLOADS_DIR / filename).write_bytes(img_bytes)
        image_url = f'/uploads/{filename}'

    # Save to history DB
    entry_id = uuid.uuid4().hex
    user = session.get('username', 'admin')
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO image_history
               (id,user,prompt,neg_prompt,size,model,filename,created_at,guidance_scale,steps)
               VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (entry_id, user,
             prompt_text,
             body.get('negative_prompt', '') or '',
             size_str, payload['model'],
             image_url,
             datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
             guidance, steps)
        )

    return jsonify({'url': image_url, 'id': entry_id})


# ── Drive routes ──────────────────────────────────────────────────────────────

@app.route('/api/drive/files')
@login_required
def drive_files():
    return jsonify(drive_list_images())

@app.route('/api/drive/upload-reference', methods=['POST'])
@login_required
def drive_upload_reference():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    f = request.files['image']
    ext = Path(f.filename).suffix.lower() if f.filename else '.jpg'
    mime = 'image/png' if ext == '.png' else 'image/webp' if ext == '.webp' else 'image/jpeg'
    filename = f'ref_{uuid.uuid4().hex}{ext}'
    _, url = drive_upload(f.read(), filename, mime)
    if not url:
        return jsonify({'error': 'Drive upload failed'}), 502
    return jsonify({'url': url})


# ── History routes ────────────────────────────────────────────────────────────

@app.route('/api/history')
@login_required
def get_history():
    user = session.get('username', 'admin')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    offset = (page - 1) * per_page

    with get_db() as conn:
        total = conn.execute(
            'SELECT COUNT(*) FROM image_history WHERE user=?', (user,)
        ).fetchone()[0]
        rows = conn.execute(
            'SELECT * FROM image_history WHERE user=? ORDER BY created_at DESC LIMIT ? OFFSET ?',
            (user, per_page, offset)
        ).fetchall()

    items = [dict(r) for r in rows]
    for item in items:
        fn = item['filename']
        if fn.startswith('http') or fn.startswith('/uploads/'):
            item['image_url'] = fn
        else:
            item['image_url'] = f"/uploads/{fn}"
        # Parse video metadata if present
        if item.get('video_meta'):
            try:
                item['video_meta'] = json.loads(item['video_meta'])
            except Exception:
                item['video_meta'] = {}
        item.setdefault('media_type', 'image')

    return jsonify({'items': items, 'total': total, 'page': page, 'per_page': per_page})


@app.route('/api/history/<entry_id>', methods=['DELETE'])
@login_required
def delete_history(entry_id):
    user = session.get('username', 'admin')
    with get_db() as conn:
        row = conn.execute(
            'SELECT filename FROM image_history WHERE id=? AND user=?', (entry_id, user)
        ).fetchone()
        if not row:
            return jsonify({'error': 'Not found'}), 404
        # Delete file
        img_path = UPLOADS_DIR / row['filename']
        if img_path.exists():
            img_path.unlink()
        conn.execute('DELETE FROM image_history WHERE id=?', (entry_id,))
    return jsonify({'success': True})


# ── Save video to history ─────────────────────────────────────────────────────

@app.route('/api/save-video', methods=['POST'])
@login_required
def save_video():
    body = request.get_json()
    if not body or not body.get('video_url'):
        return jsonify({'error': 'video_url required'}), 400
    user = session.get('username', 'admin')
    entry_id = uuid.uuid4().hex
    meta = {
        'ratio':      body.get('ratio', 'Auto'),
        'resolution': body.get('resolution', '1080p'),
        'duration':   body.get('duration', 5),
        'seed':       body.get('seed', -1),
        'with_audio': body.get('with_audio', True),
    }
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO image_history
               (id,user,prompt,neg_prompt,size,model,filename,created_at,
                guidance_scale,steps,media_type,video_meta)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
            (entry_id, user,
             body.get('prompt', ''),
             '',
             f"{meta['ratio']} · {meta['resolution']}",
             body.get('model', ''),
             body['video_url'],
             datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
             0, 0,
             'video',
             json.dumps(meta))
        )
    return jsonify({'id': entry_id})


# ── Elements routes ───────────────────────────────────────────────────────────

import re as _re

def _slugify(text):
    """Convert display label to a safe @mention slug."""
    text = text.lower().strip()
    text = _re.sub(r'[^a-z0-9\s-]', '', text)
    text = _re.sub(r'[\s-]+', '-', text)
    return text.strip('-')

@app.route('/api/elements')
@login_required
def list_elements():
    user = session.get('username', 'admin')
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM elements WHERE user=? ORDER BY created_at DESC', (user,)
        ).fetchall()
    items = []
    for r in rows:
        d = dict(r)
        try:
            d['images'] = json.loads(d['images'])
        except Exception:
            d['images'] = []
        items.append(d)
    return jsonify({'elements': items})

@app.route('/api/elements', methods=['POST'])
@login_required
def create_element():
    user = session.get('username', 'admin')
    body = request.get_json()
    if not body:
        return jsonify({'error': 'No data'}), 400
    label = body.get('label', '').strip()
    if not label:
        return jsonify({'error': 'Label is required'}), 400
    etype = body.get('type', 'character')
    if etype not in ('character', 'location'):
        return jsonify({'error': 'Type must be character or location'}), 400
    images = body.get('images', [])
    if not images or len(images) > 4:
        return jsonify({'error': '1–4 images required'}), 400
    description = body.get('description', '').strip()
    name = _slugify(label)
    if not name:
        return jsonify({'error': 'Invalid label'}), 400
    entry_id = uuid.uuid4().hex
    created_at = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    with get_db() as conn:
        existing = conn.execute(
            'SELECT id FROM elements WHERE user=? AND name=?', (user, name)
        ).fetchone()
        if existing:
            return jsonify({'error': f'An element named @{name} already exists'}), 409
        conn.execute(
            '''INSERT INTO elements (id,user,name,label,type,images,description,created_at)
               VALUES (?,?,?,?,?,?,?,?)''',
            (entry_id, user, name, label, etype, json.dumps(images), description, created_at)
        )
    return jsonify({'id': entry_id, 'name': name, 'label': label, 'type': etype,
                    'images': images, 'description': description, 'created_at': created_at})

@app.route('/api/elements/<element_id>', methods=['DELETE'])
@login_required
def delete_element(element_id):
    user = session.get('username', 'admin')
    with get_db() as conn:
        row = conn.execute(
            'SELECT id FROM elements WHERE id=? AND user=?', (element_id, user)
        ).fetchone()
        if not row:
            return jsonify({'error': 'Not found'}), 404
        conn.execute('DELETE FROM elements WHERE id=?', (element_id,))
    return jsonify({'success': True})

@app.route('/api/elements/upload-image', methods=['POST'])
@login_required
def element_upload_image():
    """Upload an element image to Drive, return URL."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    f = request.files['image']
    ext = Path(f.filename).suffix.lower() if f.filename else '.jpg'
    mime = 'image/png' if ext == '.png' else 'image/webp' if ext == '.webp' else 'image/jpeg'
    filename = f'elem_{uuid.uuid4().hex}{ext}'
    file_bytes = f.read()
    _, url = drive_upload(file_bytes, filename, mime)
    if not url:
        # Fallback: save locally and return local URL
        (UPLOADS_DIR / filename).write_bytes(file_bytes)
        railway_domain = os.environ.get('RAILWAY_PUBLIC_DOMAIN')
        if railway_domain:
            url = f'https://{railway_domain}/uploads/{filename}'
        else:
            url = f'http://localhost:{os.environ.get("PORT", 3000)}/uploads/{filename}'
    return jsonify({'url': url})


# ── Video generation routes ───────────────────────────────────────────────────

def _resolve_video_model(requested_model):
    """Pick the right model id + API key based on the model the user selected.
    Seedance 2.0 is the only active model right now.
    Returns (model_id, api_key, error)."""
    # All active video models go through the ARK platform with the Seedance key.
    model_id = (os.environ.get('SEEDANCE2_MODEL_ID', '').strip() or
                os.environ.get('SEEDANCE_MODEL_ID', '').strip())
    api_key  = (os.environ.get('SEEDANCE2_API_KEY', '') or
                os.environ.get('SEEDANCE_API_KEY', '') or
                os.environ.get('SEEDREAM_API_KEY', '')).strip()
    if not model_id:
        return None, None, 'SEEDANCE2_MODEL_ID env var not set — add the Seedance 2.0 endpoint ID in Railway Variables'
    if not api_key:
        return None, None, 'Seedance API key not configured'
    return model_id, api_key, None


@app.route('/api/generate-video', methods=['POST'])
@login_required
def generate_video():
    body = request.get_json()
    if not body:
        return jsonify({'error': 'No data'}), 400

    start_frame_url = body.get('start_frame_url', '').strip()
    if not start_frame_url:
        return jsonify({'error': 'Start frame image is required'}), 400

    requested_model = (body.get('model') or 'seedance-1-5-pro').strip().lower()
    model_id, api_key, err = _resolve_video_model(requested_model)
    if err:
        return jsonify({'error': err}), 500

    prompt = body.get('prompt', '').strip()
    duration = int(body.get('duration', 5))
    duration = max(4, min(12, duration))

    # Build inline flags — Seedance takes params embedded in the text prompt
    flags = []
    flags.append(f'--duration {duration}')
    ratio = body.get('ratio', '')
    if ratio and ratio != 'Auto':
        flags.append(f'--ratio {ratio}')
    resolution = body.get('resolution', '')
    if resolution:
        flags.append(f'--resolution {resolution}')
    seed = body.get('seed', -1)
    if seed is not None and int(seed) >= 0:
        flags.append(f'--seed {int(seed)}')
    if not body.get('with_audio', True):
        flags.append('--audio false')
    if body.get('draft_mode'):
        flags.append('--draft true')
    camera_fixed = body.get('fixed_lens', False)
    flags.append(f'--camerafixed {str(camera_fixed).lower()}')

    text_with_flags = (prompt or 'animate this scene naturally') + '  ' + '  '.join(flags)

    content = [
        {'type': 'text', 'text': text_with_flags},
        {'type': 'image_url', 'image_url': {'url': start_frame_url}},
    ]
    end_frame_url = body.get('end_frame_url', '').strip()
    if end_frame_url:
        content.append({'type': 'image_url', 'image_url': {'url': end_frame_url}})

    payload = {
        'model': model_id,
        'content': content,
    }

    n = int(body.get('n', 1))
    if n > 1:
        payload['n'] = n

    print(f'[video] model={model_id} text="{text_with_flags}"', flush=True)

    try:
        resp = requests.post(
            'https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks',
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            json=payload,
            timeout=90
        )
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Network error: {str(e)}'}), 502

    try:
        data = resp.json()
    except Exception:
        return jsonify({'error': f'Invalid response (status {resp.status_code})'}), 502

    print(f'[video] task response: {data}', flush=True)

    if not resp.ok:
        msg = (data.get('error', {}) or {}).get('message') or f'API error {resp.status_code}: {data}'
        return jsonify({'error': msg}), resp.status_code

    task_id = data.get('id') or data.get('task_id')
    if not task_id:
        return jsonify({'error': 'No task_id returned', 'raw': data}), 502

    return jsonify({'task_id': task_id, 'status': data.get('status', 'pending')})


@app.route('/api/video-status/<task_id>')
@login_required
def video_status(task_id):
    # Try keys in priority order — Seedance 2, Seedance 1, Seedream — to support
    # tasks created by either video model.
    keys = [k for k in [
        os.environ.get('SEEDANCE2_API_KEY', '').strip(),
        os.environ.get('SEEDANCE_API_KEY', '').strip(),
        os.environ.get('SEEDREAM_API_KEY', '').strip(),
    ] if k]
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]
    if not keys:
        return jsonify({'error': 'API key not configured'}), 500

    resp = None
    last_err = None
    for api_key in keys:
        try:
            resp = requests.get(
                f'https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{task_id}',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=15
            )
            if resp.status_code != 401:
                break  # this key worked (or got a non-auth error)
        except requests.exceptions.RequestException as e:
            last_err = str(e)
    if resp is None:
        return jsonify({'error': last_err or 'Network error'}), 502

    try:
        data = resp.json()
    except Exception:
        return jsonify({'error': 'Invalid response'}), 502

    print(f'[video-status] {task_id} → {data}', flush=True)

    status = data.get('status', 'unknown')
    video_url = None

    if status == 'succeeded':
        content = data.get('content')
        # Actual response: content is a dict with a direct video_url string
        if isinstance(content, dict):
            video_url = content.get('video_url')
        # Fallback: content is a list of typed items
        if not video_url and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'video_url':
                    video_url = (item.get('video_url') or {}).get('url')
                    if video_url: break
        # Fallback: choices structure
        if not video_url:
            for choice in (data.get('choices') or []):
                msg = choice.get('message') or {}
                for item in (msg.get('content') or []):
                    if isinstance(item, dict) and item.get('type') == 'video_url':
                        video_url = (item.get('video_url') or {}).get('url')
                        if video_url: break
        print(f'[video-status] succeeded, video_url={video_url}', flush=True)

    return jsonify({'status': status, 'video_url': video_url, 'raw': data})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    print(f'P6 AI Studio server running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)

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


@app.route('/api/generate-image', methods=['POST'])
@login_required
def generate_image():
    api_key = os.environ.get('SEEDREAM_API_KEY', '').strip()
    if not api_key:
        return jsonify({'error': 'Seedream API key not configured'}), 500

    body = request.get_json()
    if not body or not body.get('prompt', '').strip():
        return jsonify({'error': 'Prompt is required'}), 400

    size_str = f"{body.get('width', 1920)}x{body.get('height', 1920)}"
    guidance = float(body.get('guidance_scale', 7.5))
    steps    = int(body.get('steps', 30))

    payload = {
        'model': os.environ.get('SEEDREAM_MODEL_ID', 'ep-20251203184030-sffv6'),
        'prompt': body['prompt'].strip(),
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
             body['prompt'].strip(),
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
        # filename now stores either a full URL (Drive) or just a filename (local)
        if fn.startswith('http') or fn.startswith('/uploads/'):
            item['image_url'] = fn
        else:
            item['image_url'] = f"/uploads/{fn}"

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

@app.route('/api/generate-video', methods=['POST'])
@login_required
def generate_video():
    api_key = (os.environ.get('SEEDANCE_API_KEY', '') or
               os.environ.get('SEEDREAM_API_KEY', '')).strip()
    if not api_key:
        return jsonify({'error': 'Seedance API key not configured'}), 500

    body = request.get_json()
    if not body:
        return jsonify({'error': 'No data'}), 400

    start_frame_url = body.get('start_frame_url', '').strip()
    if not start_frame_url:
        return jsonify({'error': 'Start frame image is required'}), 400

    prompt = body.get('prompt', '').strip()
    model_id = os.environ.get('SEEDANCE_MODEL_ID', '').strip()
    if not model_id:
        return jsonify({'error': 'SEEDANCE_MODEL_ID env var not set — add your Seedance endpoint ID (e.g. ep-XXXX) in Railway Variables'}), 500
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
    api_key = (os.environ.get('SEEDANCE_API_KEY', '') or
               os.environ.get('SEEDREAM_API_KEY', '')).strip()
    if not api_key:
        return jsonify({'error': 'API key not configured'}), 500

    try:
        resp = requests.get(
            f'https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{task_id}',
            headers={'Authorization': f'Bearer {api_key}'},
            timeout=15
        )
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)}), 502

    try:
        data = resp.json()
    except Exception:
        return jsonify({'error': 'Invalid response'}), 502

    print(f'[video-status] {task_id} → {data}', flush=True)

    status = data.get('status', 'unknown')
    video_url = None

    if status == 'succeeded':
        # Try every known response structure
        # 1. Top-level content array
        for item in (data.get('content') or []):
            if isinstance(item, dict) and item.get('type') == 'video_url':
                video_url = (item.get('video_url') or {}).get('url')
                if video_url: break
        # 2. choices[0].message.content array (chat-style)
        if not video_url:
            for choice in (data.get('choices') or []):
                msg = choice.get('message') or {}
                for item in (msg.get('content') or []):
                    if isinstance(item, dict) and item.get('type') == 'video_url':
                        video_url = (item.get('video_url') or {}).get('url')
                        if video_url: break
        # 3. Flat top-level url fields
        if not video_url:
            video_url = (data.get('video_url') or {}).get('url') or data.get('url') or data.get('video')
        # 4. output field
        if not video_url:
            output = data.get('output') or {}
            video_url = output.get('url') or output.get('video_url')

        print(f'[video-status] succeeded, video_url={video_url}, keys={list(data.keys())}', flush=True)

    return jsonify({'status': status, 'video_url': video_url, 'raw': data})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    print(f'Voice Clone server running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)

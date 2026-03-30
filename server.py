#!/usr/bin/env python3
import os
import sqlite3
import uuid
import requests
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import (Flask, request, jsonify, send_from_directory,
                   Response, session, redirect, url_for)

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
@login_required
def serve_upload(filename):
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
    # Build absolute URL so Seedream can fetch it
    url = request.host_url.rstrip('/') + f'/uploads/{filename}'
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
    if ref_url:
        payload['image'] = ref_url
        print(f'[ref-image] sending image URL: {ref_url}', flush=True)

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

    # Download and store image locally so history persists
    try:
        img_resp = requests.get(remote_url, timeout=60)
        img_resp.raise_for_status()
        filename = f"{uuid.uuid4().hex}.jpg"
        (UPLOADS_DIR / filename).write_bytes(img_resp.content)
    except Exception as e:
        return jsonify({'error': f'Failed to save image: {str(e)}'}), 502

    # Save to history DB
    entry_id = uuid.uuid4().hex
    user = session.get('username', 'admin')
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO image_history
               (id,user,prompt,neg_prompt,size,model,filename,created_at,guidance_scale,steps)
               VALUES (?,?,?,?,?,?,?,?,?,?)''',
            (
                entry_id, user,
                body['prompt'].strip(),
                body.get('negative_prompt', '') or '',
                size_str,
                payload['model'],
                filename,
                datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
                guidance,
                steps
            )
        )

    local_url = f'/uploads/{filename}'
    return jsonify({'url': local_url, 'id': entry_id})


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
        item['image_url'] = f"/uploads/{item['filename']}"

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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    print(f'Voice Clone server running at http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, debug=False)

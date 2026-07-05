# 🛌 BOBOK — Sistem Deteksi Kantuk Real-Time

<div align="center">

![Version](https://img.shields.io/badge/version-3.4-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Flask](https://img.shields.io/badge/flask-3.x-lightgrey)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

**B**antuan **O**bjektif **B**eristirahat & **O**bservasi **K**elelahan

AI-powered drowsiness detection untuk keselamatan kerja & berkendara

[🚀 Quick Start](#-quick-start) •
[📖 Arsitektur](#-arsitektur) •
[🔧 API](#-api-reference) •
[📱 Mobile](#-mobile--testing)

</div>

---

## 🎯 Fitur

- 🔴 **Deteksi Microsleep** — mata tertutup >3 detik → ALARM
- 🟡 **Deteksi Kelelahan** — mata sayu, menguap, head nod → PERINGATAN
- 🟢 **Fokus / Sadar** — kondisi normal
- 📊 **Live Dashboard** — EAR, MAR, Microsleep counter, buffer status
- 📈 **Analytics** — riwayat event kantuk per user
- 🖥️ **Admin Panel** — monitor real-time semua pekerja
- 📱 **Mobile Support** — kamera HP via browser (HTTPS)

## 🏗️ Arsitektur

```
┌─ Browser ─────────────────────┐     ┌─ Server (Flask) ──────────┐
│                               │     │                           │
│  getUserMedia() → Kamera      │     │  POST /api/detection/frame│
│         ↓                     │     │         ↓                 │
│  Canvas → JPEG base64         │───→│  MediaPipe FaceLandmarker │
│         ↓                     │     │         ↓                 │
│  JSON {frame_base64, ...}     │     │  14 Fitur Engineering     │
│                               │     │         ↓                 │
│  ← JSON {status, ear, mar}    │←───│  CatBoost Model (.cbm)    │
│         ↓                     │     │         ↓                 │
│  updateHUD() → UI Dashboard   │     │  SQLite (event logging)   │
└───────────────────────────────┘     └───────────────────────────┘
```

| Komponen | Lokasi | Teknologi |
|----------|--------|-----------|
| Kamera | Browser | `getUserMedia()` |
| Face Landmarking | Server | MediaPipe Python `FaceLandmarker` |
| ML Inference | Server | CatBoost Classifier (`.cbm`) |
| Database | Server | SQLite (`database.db`) |
| Auth | Server | Flask-Login (session) |
| API | Server | Flask HTTP (port 5000) |

## 🚀 Quick Start

### Prasyarat
- Python 3.10+
- pip
- Browser yang support `getUserMedia()` (Chrome/Firefox/Safari)

### Install & Run

```bash
# Clone repo
git clone https://github.com/Noyelz/Machine-Learning-Bobok.app
cd Machine-Learning-Bobok.app

# Install dependencies
pip install -r requirements.txt

# Jalankan server
python app.py
```

Buka `http://127.0.0.1:5000` → Register → Login → Klik **MULAI** → Izinkan kamera

### 📱 Mobile & Testing via HTTPS

Kamera hanya berfungsi di **localhost** atau **HTTPS**. Untuk akses dari HP atau jaringan:

```bash
# Cloudflare Tunnel (gratis, auto-HTTPS)
cloudflared tunnel --url http://127.0.0.1:5000
# Output: https://bobok-xxxx.trycloudflare.com ← buka di HP

# SSH Tunnel (lokal)
ssh -L 5000:localhost:5000 user@server-ip
# Lalu buka http://localhost:5000
```

### 📦 Deploy ke LXC

```bash
# System dependencies
apt-get install -y libgl1-mesa-glx libgles2-mesa libegl1-mesa libglib2.0-0

# Pull & run
git pull
pip install -r requirements.txt
python3 app.py
```

## 🔧 API Reference

### Detection Endpoints

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| `POST` | `/api/detection/start` | `{}` | `{"ok": true}` |
| `POST` | `/api/detection/stop` | `{}` | `{"ok": true}` |
| `POST` | `/api/detection/frame` | `{"frame_base64": "...", "frame_width": 640, "frame_height": 480}` | `{"status": "...", "ear": 0.12, "mar": 0.05, "microsleep_counter": 0, "trigger_alarm": false, "buffer_size": 15}` |

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"SADAR (FOKUS)"` / `"PERINGATAN: ..."` / `"BAHAYA: MICROSLEEP!"` |
| `ear` | float | Eye Aspect Ratio (0-1, >0.35 = microsleep) |
| `mar` | float | Mouth Aspect Ratio (0-1, >0.30 = menguap) |
| `microsleep_counter` | int | Frame mata tertutup beruntun (alarm ≥180) |
| `trigger_alarm` | bool | Alarm aktif |
| `buffer_size` | int | Sliding window terisi (max 30) |

### Admin Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/admin/live_data` | Status real-time semua pekerja |
| `GET` | `/api/analytics?user_id=X` | Riwayat event kantuk |

## 🎛️ Tuning Sensitivitas

Edit `ai/deteksi_bobok.py`:

```python
# Threshold — turunkan untuk lebih sensitif
MICROSLEEP_THRESH      = 0.35  # (0-1) eye closed threshold
MICROSLEEP_ALARM_FRAME = 180   # (~3 detik @15fps) trigger alarm
MENGUAP_MAR_THRESH     = 0.30  # mouth open threshold
LELAH_COUNTER_LIMIT    = 90    # fatigue counter limit
KALIBRASI_DETIK        = 5     # calibration duration
```

## 📁 Struktur Project

```
├── app.py                  # Flask server + routes
├── requirements.txt        # Python dependencies
├── ai/
│   ├── deteksi_bobok.py    # Detection engine (MediaPipe + CatBoost)
│   ├── model_kantuk_v1.cbm # ML model
│   └── face_landmarker.task# MediaPipe model (auto-download)
├── static/
│   ├── js/monitor.js       # Client-side camera + dashboard
│   ├── css/style.css       # Design system (HUD theme)
│   └── alert.mp3           # Alarm sound
├── templates/
│   ├── monitor.html        # Worker dashboard
│   ├── admin_live.html     # Admin live monitor
│   ├── admin_analytics.html
│   ├── login.html          # Login page
│   └── register.html       # Registration page
└── database.db             # SQLite (auto-created)
```

## 🔒 Security Notes

- `SECRET_KEY` di `app.py` — ubah via environment variable di production
- Gunakan **HTTPS** untuk production (kamera wajib secure context)
- SQLite cukup untuk single-server, migrasi ke PostgreSQL untuk multi-node

## 📝 License

MIT — [Noyelz](https://github.com/Noyelz)

---

<div align="center">
Made with ❤️ for safety
</div>
"""
deteksi_bobok.py
================
Sistem deteksi kantuk real-time BOBOK — OPSI C (Hybrid)
- Browser: kamera + MediaPipe JS (ekstrak blendshapes & landmark)
- Server: terima data via WebSocket, lakukan inference CatBoost

Perubahan dari versi sebelumnya:
  - cv2.VideoCapture dihapus (kamera di browser)
  - MediaPipe Python dihapus (pindah ke JS di browser)
  - start_detection() diganti dengan DetectionSession class (per-user state)
  - Tetap pakai CatBoost model (.cbm) untuk inference
  - cv2.solvePnP tetap dipakai untuk head pose (pure numerik)
"""

import cv2
import numpy as np
from catboost import CatBoostClassifier
from collections import deque
import os
import time
import threading
import logging

logger = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model_kantuk_v1.cbm")

WINDOW_SIZE      = 30     # sliding window untuk fitur engineering
BLINK_THRESH     = 0.30   # threshold mata tertutup
LONG_BLINK_FRAME = 8      # threshold blink panjang
YAWN_THRESH      = 0.30   # threshold mulut terbuka (menguap)
YAWN_MIN_FRAME   = 10     # minimal frame untuk dihitung menguap
NOD_THRESH       = 10     # threshold head nod (pitch)
MICROSLEEP_THRESH = 0.55  # threshold microsleep (mata tertutup beruntun)

KALIBRASI_DETIK  = 10     # durasi fase kalibrasi (detik)

LABEL_NAMES = {0: "FOKUS", 1: "MULAI MENGANTUK", 2: "MENGANTUK"}
WARNA       = {
    0: (50, 205, 50),    # Hijau
    1: (0, 165, 255),    # Oranye
    2: (0, 0, 255),      # Merah
}

# ============================================================
# FACE 3D MODEL untuk head pose (sama seperti sebelumnya)
# ============================================================

FACE_3D  = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0  ],
    [-225.0, 170.0, -135.0 ],
    [225.0,  170.0, -135.0 ],
    [-150.0,-150.0, -125.0 ],
    [150.0, -150.0, -125.0 ],
], dtype=np.float64)

# Indeks landmark MediaPipe yang digunakan untuk head pose
# Browser HARUS mengirim 6 titik 2D dengan urutan sesuai indeks ini
POSE_IDX = [1, 152, 33, 263, 78, 308]


# ============================================================
# HELPER FUNCTIONS (pure functions, tidak berubah)
# ============================================================

def hitung_head_pose(landmarks_2d, fw, fh):
    """
    Hitung pitch, yaw, roll dari 6 titik landmark 2D.
    
    Args:
        landmarks_2d: list of [x, y] dalam pixel (sudah dikali fw/fh oleh browser)
        fw: frame width
        fh: frame height
    
    Returns:
        (pitch, yaw, roll) dalam derajat, atau (None, None, None) jika gagal
    """
    try:
        face_2d = np.array(landmarks_2d, dtype=np.float64)
        if face_2d.shape != (6, 2):
            return None, None, None
        
        cam  = np.array([[fw, 0, fw/2], [0, fw, fh/2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rv, _ = cv2.solvePnP(FACE_3D, face_2d, cam, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        rm, _ = cv2.Rodrigues(rv)
        ang, _, _, _, _, _ = cv2.RQDecomp3x3(rm)
        p = float(ang[0])
        y = float(ang[1])
        r = float(ang[2])
        p = p - 180 if p > 90 else (p + 180 if p < -90 else p)
        y = y - 180 if y > 90 else (y + 180 if y < -90 else y)
        r = r - 180 if r > 90 else (r + 180 if r < -90 else r)
        return round(p, 4), round(y, 4), round(r, 4)
    except Exception:
        return None, None, None


def deteksi_blink_events(eye_list):
    """
    Deteksi event blink (mata tertutup beruntun) dari buffer eye blendshape.
    """
    events   = []
    in_blink = False
    dur      = 0
    for val in eye_list:
        if val >= BLINK_THRESH:
            in_blink = True
            dur     += 1
        else:
            if in_blink and dur > 0:
                events.append(dur)
            in_blink = False
            dur      = 0
    if in_blink and dur > 0:
        events.append(dur)
    return events


def hitung_fitur_dari_window(eye_list, jaw_list, pitch_list, yaw_list):
    """
    Hitung 14 fitur dari buffer sliding window.
    Harus identik dengan feature engineering saat training model.
    """
    eye   = np.array(eye_list)
    jaw   = np.array(jaw_list)
    pitch = np.array(pitch_list)
    yaw_v = np.array(yaw_list)

    # Eye blendshape
    blink_mean = float(np.mean(eye))
    blink_max  = float(np.max(eye))
    blink_std  = float(np.std(eye))

    # Blink events
    events           = deteksi_blink_events(eye_list)
    blink_rate       = len(events)
    blink_dur_mean   = float(np.mean(events)) if events else 0.0
    blink_dur_max    = float(np.max(events))  if events else 0.0
    long_n           = sum(1 for d in events if d > LONG_BLINK_FRAME)
    long_blink_ratio = long_n / len(events) if events else 0.0

    # Mouth
    jaw_mean   = float(np.mean(jaw))
    jaw_max    = float(np.max(jaw))
    yawn_count = 0
    in_yawn    = False
    yawn_dur   = 0
    for val in jaw_list:
        if val >= YAWN_THRESH:
            in_yawn  = True
            yawn_dur += 1
        else:
            if in_yawn and yawn_dur >= YAWN_MIN_FRAME:
                yawn_count += 1
            in_yawn  = False
            yawn_dur = 0

    # Head pose
    pitch_mean     = float(np.mean(pitch))
    pitch_std      = float(np.std(pitch))
    yaw_std        = float(np.std(yaw_v))
    head_nod_count = int(np.sum(pitch > NOD_THRESH))

    return np.array([[
        blink_mean, blink_max, blink_std,
        blink_rate, blink_dur_mean, blink_dur_max, long_blink_ratio,
        jaw_mean, jaw_max, yawn_count,
        pitch_mean, pitch_std, yaw_std, head_nod_count
    ]])


# ============================================================
# MODEL LOADER (di-load sekali saat startup)
# ============================================================

_model = None
_model_lock = threading.Lock()

def get_model():
    """Load model CatBoost sekali (singleton), thread-safe."""
    global _model
    if _model is not None:
        return _model
    
    with _model_lock:
        if _model is not None:
            return _model
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        
        logger.info("Memuat model CatBoost dari %s ...", MODEL_PATH)
        _model = CatBoostClassifier()
        _model.load_model(MODEL_PATH)
        logger.info("Model CatBoost siap.")
        return _model


# ============================================================
# DETECTION SESSION — per-user state untuk WebSocket
# ============================================================

class DetectionSession:
    """
    Satu instance per user yang sedang melakukan deteksi.
    State-fully menerima frame data dari browser, menjalankan inference,
    dan mengembalikan hasil deteksi.

    Dipanggil dari WebSocket handler di app.py:
        session = DetectionSession(user_id)
        result = session.process_frame(frame_data)
    """

    def __init__(self, user_id):
        self.user_id = user_id
        self.model = get_model()

        # Sliding window buffers (30 frame)
        self.buf_eye   = deque(maxlen=WINDOW_SIZE)
        self.buf_jaw   = deque(maxlen=WINDOW_SIZE)
        self.buf_pitch = deque(maxlen=WINDOW_SIZE)
        self.buf_yaw   = deque(maxlen=WINDOW_SIZE)

        # State
        self.frame_idx        = 0
        self.status           = 'Mencari Wajah...'
        self.label_pred       = 0
        self.warna_status     = WARNA[0]
        self.microsleep_counter = 0
        self.lelah_counter    = 0
        self.menguap_counter  = 0
        self.lelah_logged     = False
        self.microsleep_logged = False
        self.microsleep_event_count = 0
        self.deteksi_history  = deque(maxlen=90)

        # Kalibrasi
        self.kalibrasi_aktif  = True
        self.kalibrasi_mulai  = time.time()

        # Pending events (untuk dicatat ke DB oleh Flask)
        self.pending_events   = []

    def process_frame(self, data):
        """
        Proses satu frame data dari browser.

        Args:
            data (dict): Data dari browser dengan struktur:
                {
                    "eyeBlinkLeft": float,
                    "eyeBlinkRight": float,
                    "jawOpen": float,
                    "mouthFunnel": float,
                    "landmarks_2d": [[x,y]*6],  # 6 titik POSE_IDX dalam pixel
                    "frame_width": int,
                    "frame_height": int,
                    "has_face": bool
                }

        Returns:
            dict: Hasil deteksi untuk dikirim kembali ke browser:
                {
                    "status": str,
                    "label_pred": int,
                    "ear": float or None,
                    "mar": float or None,
                    "microsleep_counter": int,
                    "microsleep_event_count": int,
                    "trigger_alarm": bool,
                    "buffer_size": int,
                    "ear_mean": float or None,
                    "mar_mean": float or None,
                    "has_new_events": list  # event baru untuk dicatat di DB
                }
        """
        self.frame_idx += 1

        has_face = data.get('has_face', False)
        ear_val = None
        mar_val = None

        if has_face:
            eye_blink_left  = data.get('eyeBlinkLeft', 0)
            eye_blink_right = data.get('eyeBlinkRight', 0)
            jaw_open        = data.get('jawOpen', 0)
            landmarks_2d    = data.get('landmarks_2d', [])
            fw              = data.get('frame_width', 640)
            fh              = data.get('frame_height', 480)

            # Hitung head pose dari 6 titik landmark
            p, y, r = hitung_head_pose(landmarks_2d, fw, fh)

            if p is not None and len(landmarks_2d) == 6:
                eye_avg = (eye_blink_left + eye_blink_right) / 2.0
                ear_val = float(eye_avg)
                mar_val = float(jaw_open)

                self.buf_eye.append(eye_avg)
                self.buf_jaw.append(jaw_open)
                self.buf_pitch.append(p)
                self.buf_yaw.append(y)

        # Tracking deteksi
        self.deteksi_history.append(0 if not has_face else 1)

        # Cek kalibrasi
        sisa_kalibrasi = KALIBRASI_DETIK - (time.time() - self.kalibrasi_mulai)
        if self.kalibrasi_aktif and sisa_kalibrasi <= 0:
            self.kalibrasi_aktif = False
            logger.info("User %s: Kalibrasi selesai.", self.user_id)

        trigger_alarm = False

        if not has_face:
            self.status = 'Mencari Wajah...'
            self.microsleep_counter = 0
        elif self.kalibrasi_aktif:
            self.status = f'Kalibrasi... ({max(0, sisa_kalibrasi):.0f}s)'
        elif len(self.buf_eye) == WINDOW_SIZE:
            # Inference
            fitur = hitung_fitur_dari_window(
                list(self.buf_eye), list(self.buf_jaw),
                list(self.buf_pitch), list(self.buf_yaw)
            )
            pred = int(self.model.predict(fitur).flatten()[0])
            self.label_pred = pred
            self.warna_status = WARNA[pred]

            # Microsleep counter
            if ear_val is not None and ear_val >= MICROSLEEP_THRESH:
                self.microsleep_counter += 1
            else:
                self.microsleep_counter = 0
                self.microsleep_logged = False

            # Lelah counter
            if pred in [1, 2]:
                self.lelah_counter += 1
            else:
                self.lelah_counter = 0

            # Menguap counter
            if mar_val is not None and mar_val >= 0.4:
                self.menguap_counter += 1
            else:
                self.menguap_counter = 0

            is_fatigued = (self.lelah_counter >= 360) or (self.menguap_counter >= 60)

            if not is_fatigued:
                self.lelah_logged = False

            # Catat event ke pending queue
            if is_fatigued and not self.lelah_logged:
                self.pending_events.append('menguap_lelah')
                self.lelah_logged = True

            if self.microsleep_counter >= 720 and not self.microsleep_logged:
                self.pending_events.append('microsleep')
                self.microsleep_logged = True

            # Alarm
            trigger_alarm = (self.microsleep_counter >= 720) or is_fatigued

            # Status string
            if self.microsleep_counter >= 720:
                self.status = 'BAHAYA: MICROSLEEP!'
                self.warna_status = (0, 0, 255)
            elif is_fatigued:
                if self.menguap_counter >= 60 or pred == 2:
                    self.status = 'PERINGATAN: MENGUAP / LELAH BERAT'
                else:
                    self.status = 'PERINGATAN: MATA MULAI LELAH / SAYU'
                self.warna_status = (0, 165, 255)
            else:
                self.status = 'SADAR (FOKUS)'
                self.warna_status = (50, 205, 50)

        # Ambil pending events
        new_events = []
        if self.pending_events:
            new_events = list(self.pending_events)
            self.pending_events.clear()

        # Hasil untuk dikirim via WebSocket
        det_rate = sum(self.deteksi_history) / len(self.deteksi_history) * 100 if self.deteksi_history else 0

        return {
            'status': self.status,
            'label_pred': self.label_pred,
            'ear': ear_val,
            'mar': mar_val,
            'microsleep_counter': self.microsleep_counter,
            'microsleep_event_count': self.microsleep_event_count,
            'trigger_alarm': trigger_alarm,
            'buffer_size': len(self.buf_eye),
            'ear_mean': float(np.mean(self.buf_eye)) if self.buf_eye else None,
            'mar_mean': float(np.mean(self.buf_jaw)) if self.buf_jaw else None,
            'deteksi_rate': round(det_rate, 1),
            'kalibrasi_aktif': self.kalibrasi_aktif,
            'kalibrasi_sisa': max(0, sisa_kalibrasi) if 'sisa_kalibrasi' in dir() else 0,
            'has_new_events': new_events,
        }

    def is_calibrating(self):
        return self.kalibrasi_aktif

    def reset(self):
        """Reset session (tidak digunakan lagi)."""
        self.buf_eye.clear()
        self.buf_jaw.clear()
        self.buf_pitch.clear()
        self.buf_yaw.clear()
        self.frame_idx = 0
        self.status = 'Mencari Wajah...'
        self.label_pred = 0
        self.microsleep_counter = 0
        self.lelah_counter = 0
        self.menguap_counter = 0
        self.pending_events.clear()
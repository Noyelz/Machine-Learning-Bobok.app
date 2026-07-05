"""
deteksi_bobok.py
================
Sistem deteksi kantuk real-time BOBOK — v3.2 (Frame Upload)
- Browser: kamera, kirim JPEG via POST
- Server: MediaPipe Python + CatBoost inference
"""

import cv2
import numpy as np
from catboost import CatBoostClassifier
from collections import deque
import os
import time
import threading
import logging

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request

logger = logging.getLogger(__name__)

# ============================================================
# KONFIGURASI
# ============================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model_kantuk_v1.cbm")
LANDMARKER_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
LANDMARKER_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/1/face_landmarker.task")

WINDOW_SIZE      = 30
BLINK_THRESH     = 0.30
LONG_BLINK_FRAME = 8
YAWN_THRESH      = 0.30
YAWN_MIN_FRAME   = 10
NOD_THRESH       = 10
MICROSLEEP_THRESH = 0.55
KALIBRASI_DETIK  = 10

LABEL_NAMES = {0: "FOKUS", 1: "MULAI MENGANTUK", 2: "MENGANTUK"}

# ============================================================
# FACE 3D MODEL
# ============================================================

FACE_3D = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0  ],
    [-225.0, 170.0, -135.0 ],
    [225.0,  170.0, -135.0 ],
    [-150.0,-150.0, -125.0 ],
    [150.0, -150.0, -125.0 ],
], dtype=np.float64)

POSE_IDX = [1, 152, 33, 263, 78, 308]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def download_landmarker():
    if os.path.exists(LANDMARKER_PATH):
        return
    logger.info("Mengunduh face_landmarker.task...")
    os.makedirs(os.path.dirname(LANDMARKER_PATH), exist_ok=True)
    urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)
    logger.info("Download selesai.")


def hitung_head_pose(landmarks_2d, fw, fh):
    try:
        face_2d = np.array(landmarks_2d, dtype=np.float64)
        if face_2d.shape != (6, 2):
            return None, None, None
        cam  = np.array([[fw, 0, fw/2], [0, fw, fh/2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        ok, rv, _ = cv2.solvePnP(FACE_3D, face_2d, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None, None
        rm, _ = cv2.Rodrigues(rv)
        ang, _, _, _, _, _ = cv2.RQDecomp3x3(rm)
        p, y, r = float(ang[0]), float(ang[1]), float(ang[2])
        p = p - 180 if p > 90 else (p + 180 if p < -90 else p)
        y = y - 180 if y > 90 else (y + 180 if y < -90 else y)
        r = r - 180 if r > 90 else (r + 180 if r < -90 else r)
        return round(p,4), round(y,4), round(r,4)
    except:
        return None, None, None


def deteksi_blink_events(eye_list):
    events, in_blink, dur = [], False, 0
    for val in eye_list:
        if val >= BLINK_THRESH:
            in_blink = True; dur += 1
        else:
            if in_blink and dur > 0: events.append(dur)
            in_blink = False; dur = 0
    if in_blink and dur > 0: events.append(dur)
    return events


def hitung_fitur_dari_window(eye_list, jaw_list, pitch_list, yaw_list):
    eye   = np.array(eye_list)
    jaw   = np.array(jaw_list)
    pitch = np.array(pitch_list)
    yaw_v = np.array(yaw_list)

    blink_mean = float(np.mean(eye))
    blink_max  = float(np.max(eye))
    blink_std  = float(np.std(eye))

    events = deteksi_blink_events(eye_list)
    blink_rate = len(events)
    blink_dur_mean = float(np.mean(events)) if events else 0.0
    blink_dur_max  = float(np.max(events)) if events else 0.0
    long_n = sum(1 for d in events if d > LONG_BLINK_FRAME)
    long_blink_ratio = long_n / len(events) if events else 0.0

    jaw_mean = float(np.mean(jaw))
    jaw_max  = float(np.max(jaw))
    yawn_count, in_yawn, yawn_dur = 0, False, 0
    for val in jaw_list:
        if val >= YAWN_THRESH:
            in_yawn = True; yawn_dur += 1
        else:
            if in_yawn and yawn_dur >= YAWN_MIN_FRAME: yawn_count += 1
            in_yawn = False; yawn_dur = 0

    pitch_mean = float(np.mean(pitch))
    pitch_std  = float(np.std(pitch))
    yaw_std    = float(np.std(yaw_v))
    head_nod_count = int(np.sum(pitch > NOD_THRESH))

    return np.array([[
        blink_mean, blink_max, blink_std,
        blink_rate, blink_dur_mean, blink_dur_max, long_blink_ratio,
        jaw_mean, jaw_max, yawn_count,
        pitch_mean, pitch_std, yaw_std, head_nod_count
    ]])


# ============================================================
# MODEL SINGLETON
# ============================================================

_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
        logger.info("Memuat model CatBoost...")
        _model = CatBoostClassifier()
        _model.load_model(MODEL_PATH)
        logger.info("Model CatBoost siap.")
        return _model


# ============================================================
# DETECTOR POOL (single MediaPipe instance)
# ============================================================

_detector = None
_detector_lock = threading.Lock()

def get_detector():
    global _detector
    if _detector is not None:
        return _detector
    with _detector_lock:
        if _detector is not None:
            return _detector
        download_landmarker()
        logger.info("Menyiapkan FaceLandmarker...")
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        _detector = mp_vision.FaceLandmarker.create_from_options(opts)
        logger.info("FaceLandmarker siap.")
        return _detector


def process_image(frame_bgr):
    """
    Proses satu frame BGR (numpy array) dengan MediaPipe.
    Return: (has_face, blendshapes_dict, landmarks_list)
    """
    detector = get_detector()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    has_face = (result.face_landmarks and len(result.face_landmarks) > 0)
    blendshapes_dict = {}
    landmarks_list = None

    if has_face:
        landmarks_list = result.face_landmarks[0]
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for bs in result.face_blendshapes[0]:
                if bs.category_name in {'eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthFunnel'}:
                    blendshapes_dict[bs.category_name] = bs.score

    return has_face, blendshapes_dict, landmarks_list


def ambil_blendshapes(blendshapes):
    target = {'eyeBlinkLeft', 'eyeBlinkRight', 'jawOpen', 'mouthFunnel'}
    return {bs.category_name: bs.score for bs in blendshapes[0] if bs.category_name in target} if blendshapes else None


# ============================================================
# DETECTION SESSION
# ============================================================

class DetectionSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.model = get_model()

        self.buf_eye   = deque(maxlen=WINDOW_SIZE)
        self.buf_jaw   = deque(maxlen=WINDOW_SIZE)
        self.buf_pitch = deque(maxlen=WINDOW_SIZE)
        self.buf_yaw   = deque(maxlen=WINDOW_SIZE)

        self.frame_idx = 0
        self.status = 'Mencari Wajah...'
        self.label_pred = 0
        self.microsleep_counter = 0
        self.lelah_counter = 0
        self.menguap_counter = 0
        self.lelah_logged = False
        self.microsleep_logged = False
        self.microsleep_event_count = 0
        self.deteksi_history = deque(maxlen=90)

        self.kalibrasi_aktif = True
        self.kalibrasi_mulai = time.time()
        self.pending_events = []

    def process_frame_data(self, blendshapes_dict, landmarks_list, fw, fh, frame_bgr):
        """
        Versi hybrid dengan blendshapes + landmarks dari proses_image().
        """
        self.frame_idx += 1
        has_face = landmarks_list is not None and len(blendshapes_dict) >= 4
        ear_val, mar_val = None, None

        if has_face:
            # Head pose dari 6 landmark
            lm2d = []
            for idx in POSE_IDX:
                lm = landmarks_list[idx]
                lm2d.append([lm.x * fw, lm.y * fh])
            p, y, r = hitung_head_pose(lm2d, fw, fh)

            if p is not None:
                eye_avg = (blendshapes_dict.get('eyeBlinkLeft', 0) + blendshapes_dict.get('eyeBlinkRight', 0)) / 2.0
                ear_val = float(eye_avg)
                mar_val = float(blendshapes_dict.get('jawOpen', 0))
                self.buf_eye.append(eye_avg)
                self.buf_jaw.append(blendshapes_dict.get('jawOpen', 0))
                self.buf_pitch.append(p)
                self.buf_yaw.append(y)
            else:
                has_face = False

        self.deteksi_history.append(0 if not has_face else 1)

        sisa_kalibrasi = KALIBRASI_DETIK - (time.time() - self.kalibrasi_mulai)
        if self.kalibrasi_aktif and sisa_kalibrasi <= 0:
            self.kalibrasi_aktif = False

        trigger_alarm = False

        if not has_face:
            self.status = 'Mencari Wajah...'
            self.microsleep_counter = 0
        elif self.kalibrasi_aktif:
            self.status = f'Kalibrasi... ({max(0, sisa_kalibrasi):.0f}s)'
        elif len(self.buf_eye) == WINDOW_SIZE:
            fitur = hitung_fitur_dari_window(
                list(self.buf_eye), list(self.buf_jaw),
                list(self.buf_pitch), list(self.buf_yaw))
            pred = int(self.model.predict(fitur).flatten()[0])
            self.label_pred = pred

            if ear_val is not None and ear_val >= MICROSLEEP_THRESH:
                self.microsleep_counter += 1
            else:
                self.microsleep_counter = 0
                self.microsleep_logged = False

            if pred in [1, 2]: self.lelah_counter += 1
            else: self.lelah_counter = 0

            if mar_val is not None and mar_val >= 0.4: self.menguap_counter += 1
            else: self.menguap_counter = 0

            is_fatigued = (self.lelah_counter >= 360) or (self.menguap_counter >= 60)
            if not is_fatigued: self.lelah_logged = False

            if is_fatigued and not self.lelah_logged:
                self.pending_events.append('menguap_lelah')
                self.lelah_logged = True

            if self.microsleep_counter >= 720 and not self.microsleep_logged:
                self.pending_events.append('microsleep')
                self.microsleep_logged = True

            trigger_alarm = (self.microsleep_counter >= 720) or is_fatigued

            if self.microsleep_counter >= 720:
                self.status = 'BAHAYA: MICROSLEEP!'
            elif is_fatigued:
                if self.menguap_counter >= 60 or pred == 2:
                    self.status = 'PERINGATAN: MENGUAP / LELAH BERAT'
                else:
                    self.status = 'PERINGATAN: MATA MULAI LELAH / SAYU'
            else:
                self.status = 'SADAR (FOKUS)'

        new_events = list(self.pending_events)
        self.pending_events.clear()

        det_rate = sum(self.deteksi_history) / len(self.deteksi_history) * 100 if self.deteksi_history else 0

        return {
            'status': self.status,
            'label_pred': self.label_pred,
            'ear': ear_val,
            'mar': mar_val,
            'microsleep_counter': self.microsleep_counter,
            'trigger_alarm': trigger_alarm,
            'buffer_size': len(self.buf_eye),
            'ear_mean': float(np.mean(self.buf_eye)) if self.buf_eye else None,
            'mar_mean': float(np.mean(self.buf_jaw)) if self.buf_jaw else None,
            'has_new_events': new_events,
        }
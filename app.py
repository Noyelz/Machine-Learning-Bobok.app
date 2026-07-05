import os
import logging
from datetime import datetime

from flask import Flask, jsonify, render_template, redirect, url_for, request, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from ai.deteksi_bobok import DetectionSession, get_model

# ============================================================
# APP INIT
# ============================================================

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'bobok-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Dictionary untuk menyimpan DetectionSession per user_id
user_sessions = {}


# ============================================================
# MODELS
# ============================================================

class User(db.Model, UserMixin):
    id            = db.Column(db.Integer, primary_key=True)
    nama          = db.Column(db.String(100), nullable=False)
    username      = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role          = db.Column(db.String(20), nullable=False)
    events        = db.relationship('EventKantuk', backref='user', lazy=True)
    live_status   = db.relationship('LiveStatus', backref='user', uselist=False)


class EventKantuk(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    jenis     = db.Column(db.String(50), nullable=False)


class LiveStatus(db.Model):
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    status      = db.Column(db.String(100), default='OFFLINE')
    ear         = db.Column(db.Float, default=0.0)
    mar         = db.Column(db.Float, default=0.0)
    last_update = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================================
# AUTH ROUTES
# ============================================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_live'))
        return redirect(url_for('monitor'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_live'))
        return redirect(url_for('monitor'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_live'))
            return redirect(url_for('monitor'))
        flash('Username atau password salah.')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nama     = request.form.get('nama')
        username = request.form.get('username')
        password = request.form.get('password')
        role     = 'pekerja'
        if User.query.filter_by(username=username).first():
            flash('Username sudah terdaftar.')
            return redirect(url_for('register'))
        user = User(
            nama=nama,
            username=username,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.session.add(user)
        db.session.commit()
        flash('Registrasi berhasil, silakan login.')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    if current_user.id in user_sessions:
        del user_sessions[current_user.id]

    live = LiveStatus.query.get(current_user.id)
    if live:
        live.status = 'OFFLINE'
        db.session.commit()
    logout_user()
    return redirect(url_for('login'))


# ============================================================
# PEKERJA ROUTES
# ============================================================

@app.route('/monitor')
@login_required
def monitor():
    if current_user.role != 'pekerja':
        return redirect(url_for('admin_live'))
    return render_template('monitor.html', user=current_user)


@app.route('/analytics')
@login_required
def analytics():
    if current_user.role != 'pekerja':
        return redirect(url_for('admin_analytics'))
    return render_template('analytics_pekerja.html', user=current_user)


# ============================================================
# ADMIN ROUTES
# ============================================================

@app.route('/admin/live')
@login_required
def admin_live():
    if current_user.role != 'admin':
        return redirect(url_for('monitor'))
    return render_template('admin_live.html', user=current_user)


@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if current_user.role != 'admin':
        return redirect(url_for('monitor'))
    return render_template('admin_analytics.html', user=current_user)


# ============================================================
# API ROUTES — ADMIN & ANALYTICS
# ============================================================

@app.route('/admin/live_data')
@login_required
def admin_live_data():
    if current_user.role != 'admin':
        return jsonify([]), 403
    statuses = LiveStatus.query.join(User).filter(User.role == 'pekerja').all()
    hasil = []
    for s in statuses:
        hasil.append({
            'user_id':     s.user_id,
            'nama':        s.user.nama,
            'status':      s.status,
            'ear':         s.ear,
            'mar':         s.mar,
            'last_update': s.last_update.isoformat() if s.last_update else None
        })
    return jsonify(hasil)


@app.route('/api/analytics')
@login_required
def api_analytics():
    if current_user.role == 'pekerja':
        target_id = current_user.id
    else:
        target_id = request.args.get('user_id', type=int)

    query = EventKantuk.query
    if target_id:
        query = query.filter_by(user_id=target_id)

    events = query.order_by(EventKantuk.timestamp.desc()).all()
    hasil = []
    for e in events:
        hasil.append({
            'id':        e.id,
            'user_id':   e.user_id,
            'nama':      e.user.nama,
            'timestamp': e.timestamp.isoformat(),
            'jenis':     e.jenis
        })
    return jsonify(hasil)


@app.route('/api/pekerja_list')
@login_required
def pekerja_list():
    if current_user.role != 'admin':
        return jsonify([]), 403
    pekerja = User.query.filter_by(role='pekerja').all()
    return jsonify([{'id': p.id, 'nama': p.nama} for p in pekerja])


# ============================================================
# DETECTION API — HTTP polling (ganti WebSocket)
# ============================================================

@app.route('/api/detection/start', methods=['POST'])
@login_required
def api_start_detection():
    """Memulai sesi deteksi untuk user yang sedang login."""
    user_id = current_user.id

    if user_id in user_sessions:
        logger.info("User %s: Mereset session deteksi.", user_id)
    else:
        logger.info("User %s: Membuat session deteksi baru.", user_id)

    user_sessions[user_id] = DetectionSession(user_id)

    # Update DB
    live = LiveStatus.query.get(user_id)
    if not live:
        live = LiveStatus(user_id=user_id, status='AKTIF')
        db.session.add(live)
    else:
        live.status = 'AKTIF'
        live.last_update = datetime.utcnow()
    db.session.commit()

    return jsonify({'ok': True, 'message': 'Detection started'})


@app.route('/api/detection/stop', methods=['POST'])
@login_required
def api_stop_detection():
    """Menghentikan sesi deteksi."""
    user_id = current_user.id

    if user_id in user_sessions:
        del user_sessions[user_id]
        logger.info("User %s: Session deteksi dihentikan.", user_id)

    live = LiveStatus.query.get(user_id)
    if live:
        live.status = 'OFFLINE'
        db.session.commit()

    return jsonify({'ok': True, 'message': 'Detection stopped'})


@app.route('/api/detection/frame', methods=['POST'])
@login_required
def api_process_frame():
    """
    Terima data frame dari browser (blendshapes + landmarks),
    jalankan inference, balas hasil deteksi.
    """
    user_id = current_user.id
    data = request.get_json(force=True)

    if not data:
        return jsonify({'error': 'No data'}), 400

    # Auto-start session jika belum ada
    if user_id not in user_sessions:
        user_sessions[user_id] = DetectionSession(user_id)

    session = user_sessions[user_id]

    # Jalankan inference
    try:
        result = session.process_frame(data)
    except Exception as e:
        logger.error("Error process_frame user %s: %s", user_id, e)
        return jsonify({'error': str(e)}), 500

    # Catat event baru ke DB
    new_events = result.get('has_new_events', [])
    if new_events:
        for ev_type in new_events:
            ev = EventKantuk(user_id=user_id, jenis=ev_type)
            db.session.add(ev)
        db.session.commit()

    # Update LiveStatus
    live = LiveStatus.query.get(user_id)
    if live:
        live.status      = result.get('status', 'UNKNOWN')
        live.ear         = result.get('ear') or 0.0
        live.mar         = result.get('mar') or 0.0
        live.last_update = datetime.utcnow()
        db.session.commit()

    return jsonify(result)


# ============================================================
# INIT & RUN
# ============================================================

with app.app_context():
    db.create_all()
    # Pre-load model saat startup
    try:
        get_model()
        logger.info("Model CatBoost berhasil dimuat saat startup.")
    except Exception as e:
        logger.warning("Model CatBoost belum bisa dimuat: %s", e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

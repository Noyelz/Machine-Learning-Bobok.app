/**
 * BOBOK — monitor.js (v3.2 — HTTP Frame Upload)
 * ==============================================
 * Browser: kamera + render ke canvas
 * Setiap frame: kirim JPEG ke server via POST /api/detection/frame
 * Server: MediaPipe Python + CatBoost inference
 */

// ── CONFIG ──────────────────────────────────────────────────────
const LOG_MAX       = 60;
const DETECTION_FPS = 15;

// ── STATE ────────────────────────────────────────────────────────
let isConnected      = false;
let sessionActive    = false;
let sessionStart     = null;
let sessionTimer     = null;
let lastStatus       = '';
let lastLogStatus    = '';
let dangerFlashCooldown = false;

// Kamera
let videoElement     = null;
let canvasElement    = null;
let canvasCtx        = null;
let mediaStream      = null;
let detectionLoopId  = null;
let lastFrameTime    = 0;
let frameErrorCount  = 0;

const stats = { bahaya: 0, peringatan: 0, sadar: 0 };

// ── DOM REFS ─────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const sysClock        = $('sysClock');
const connDot         = $('connDot');
const connLabel       = $('connLabel');
const btnConnect      = $('btnConnect');
const overlayCanvas   = $('overlayCanvas');
const camOffline      = $('camOffline');
const faceBox         = $('faceBox');
const ringProgress    = $('ringProgress');
const ringLabel       = $('ringLabel');
const alertCode       = $('alertCode');
const alertMsg        = $('alertMsg');
const alertLog        = $('alertLog');
const earFill         = $('earFill');
const earVal          = $('earVal');
const earStatus       = $('earStatus');
const marFill         = $('marFill');
const marVal          = $('marVal');
const marStatus       = $('marStatus');
const microFrames     = $('microFrames');
const microVal        = $('microVal');
const microStatus     = $('microStatus');
const bufferFill      = $('bufferFill');
const bufferVal       = $('bufferVal');
const bufferStatus    = $('bufferStatus');
const statSesi        = $('statSesi');
const statBahaya      = $('statBahaya');
const statPeringatan  = $('statPeringatan');
const statSadar       = $('statSadar');
const dangerFlash     = $('dangerFlash');

// ── INIT ─────────────────────────────────────────────────────────
buildMicroFrames();
startClock();
updateConnectionUI(false);

function buildMicroFrames() {
  microFrames.innerHTML = '';
  for (let i = 0; i < 15; i++) {
    const b = document.createElement('div');
    b.className = 'micro-frame-block';
    if (i === 9) b.classList.add('threshold');
    microFrames.appendChild(b);
  }
}

function startClock() {
  function tick() {
    const now = new Date();
    sysClock.textContent = [String(now.getHours()).padStart(2,'0'), String(now.getMinutes()).padStart(2,'0'), String(now.getSeconds()).padStart(2,'0')].join(':');
  }
  tick();
  setInterval(tick, 1000);
}

function updateConnectionUI(connected) {
  isConnected = connected;
  connDot.classList.toggle('active', connected);
  connLabel.textContent = connected ? 'ONLINE' : 'OFFLINE';
}

// ── CONNECTION TOGGLE ─────────────────────────────────────────────
async function toggleConnect() {
  if (!sessionActive) {
    await startSession();
  } else {
    stopSession();
  }
}

async function startSession() {
  // 1. Start detection di server
  appendLog('[SISTEM] Menghubungi server...', 'log-sys');
  try {
    const res = await fetch('/api/detection/start', { method: 'POST' });
    const data = await res.json();
    if (!data.ok) throw new Error(data.message || 'Failed');
    appendLog('[SISTEM] Server siap.', 'log-sys');
  } catch (err) {
    appendLog('[ERROR] Gagal start server: ' + err.message, 'log-danger');
    return;
  }

  // 2. Kamera
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    appendLog('[ERROR] Kamera tidak tersedia. Gunakan HTTPS atau localhost.', 'log-danger');
    await fetch('/api/detection/stop', { method: 'POST' });
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
  } catch (err) {
    appendLog('[ERROR] Kamera: ' + err.message, 'log-danger');
    await fetch('/api/detection/stop', { method: 'POST' });
    return;
  }

  // 3. Video + Canvas
  videoElement = $('videoElement');
  videoElement.srcObject = mediaStream;
  await videoElement.play();

  canvasElement = overlayCanvas;
  canvasCtx = canvasElement.getContext('2d');

  // 4. UI
  sessionActive = true;
  sessionStart = Date.now();
  sessionTimer = setInterval(updateSessionTimer, 1000);
  frameErrorCount = 0;

  updateConnectionUI(true);
  btnConnect.innerHTML = '<span>■ STOP</span>';
  btnConnect.classList.add('active');
  showCamera(true);
  resizeCanvas();

  appendLog('[SISTEM] Sesi dimulai. Kamera aktif, deteksi berjalan...', 'log-sys');

  // 5. Mulai loop
  lastFrameTime = 0;
  detectionLoopId = requestAnimationFrame(detectionLoop);
}

function stopSession() {
  if (detectionLoopId) { cancelAnimationFrame(detectionLoopId); detectionLoopId = null; }
  fetch('/api/detection/stop', { method: 'POST' }).catch(() => {});
  stopCamera();

  sessionActive = false;
  clearInterval(sessionTimer);
  sessionTimer = null;
  sessionStart = null;

  btnConnect.innerHTML = '<span>▶ MULAI</span>';
  btnConnect.classList.remove('active');
  updateConnectionUI(false);
  showCamera(false);
  if (canvasCtx) canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);

  setStatus('standby');
  ringLabel.textContent = 'STANDBY';
  alertCode.textContent = 'SYS.STOP';
  alertMsg.textContent = 'Sesi dihentikan.';
  setRingProgress(0);
  resetMetrics();
  appendLog('[SISTEM] Sesi dihentikan.', 'log-sys');
}

function stopCamera() {
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  if (videoElement) videoElement.srcObject = null;
}

function resizeCanvas() {
  const wrapper = document.querySelector('.cam-wrapper');
  if (!wrapper || !canvasElement) return;
  const rect = wrapper.getBoundingClientRect();
  canvasElement.width = rect.width;
  canvasElement.height = rect.height;
}

// ── DETECTION LOOP ───────────────────────────────────────────────
function detectionLoop(timestamp) {
  if (!sessionActive) return;
  detectionLoopId = requestAnimationFrame(detectionLoop);

  // Throttle
  const frameInterval = 1000 / DETECTION_FPS;
  if (timestamp - lastFrameTime < frameInterval) {
    // Hanya render canvas, jangan kirim
    if (videoElement && videoElement.readyState >= 2) renderCanvasOnly();
    return;
  }
  lastFrameTime = timestamp;

  if (!videoElement || videoElement.readyState < 2) return;
  if (!canvasCtx || !canvasElement) return;

  const videoWidth  = videoElement.videoWidth;
  const videoHeight = videoElement.videoHeight;
  if (videoWidth === 0 || videoHeight === 0) return;

  // Resize canvas
  const wrapper = document.querySelector('.cam-wrapper');
  if (wrapper) {
    const rect = wrapper.getBoundingClientRect();
    if (canvasElement.width !== rect.width || canvasElement.height !== rect.height) {
      canvasElement.width = rect.width;
      canvasElement.height = rect.height;
    }
  }

  // Resize canvas max 640px width untuk hemat bandwidth
  if (canvasElement.width > 640) {
    canvasElement.height = Math.round(canvasElement.height * (640 / canvasElement.width));
    canvasElement.width = 640;
  }

  // Draw video ke canvas (flip horizontal)
  const cw = canvasElement.width, ch = canvasElement.height;
  canvasCtx.save();
  canvasCtx.translate(cw, 0);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(videoElement, 0, 0, cw, ch);
  canvasCtx.restore();

  showCamera(true);

  // Export JPEG dan kirim ke server
  canvasElement.toBlob(async (blob) => {
    if (!blob) {
      console.error('[Frame] toBlob returned null');
      return;
    }

    const formData = new FormData();
    formData.append('frame', blob, 'frame.jpg');
    formData.append('frame_width', videoWidth);
    formData.append('frame_height', videoHeight);

    try {
      const res = await fetch('/api/detection/frame', {
        method: 'POST',
        body: formData
      });
      if (!res.ok) {
        let errorText = '';
        try { errorText = await res.text(); } catch (e) {}
        console.error('[Frame] Server error:', res.status, errorText);
        throw new Error('HTTP ' + res.status + (errorText ? ': ' + errorText.substring(0,100) : ''));
      }
      const result = await res.json();
      frameErrorCount = 0;
      updateHUD(result);
    } catch (err) {
      frameErrorCount++;
      console.error('[Frame] Send error:', err.message);
      if (frameErrorCount > 10) {
        appendLog('[ERROR] Server: ' + err.message, 'log-danger');
        frameErrorCount = 0;
      }
    }
  }, 'image/jpeg', 0.5);
}

function renderCanvasOnly() {
  if (!canvasCtx || !canvasElement || !videoElement || videoElement.readyState < 2) return;
  const cw = canvasElement.width, ch = canvasElement.height;
  canvasCtx.save();
  canvasCtx.translate(cw, 0);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(videoElement, 0, 0, cw, ch);
  canvasCtx.restore();
}

// ── HUD UPDATE ──────────────────────────────────────────────────
function updateHUD(data) {
  const {
    status = 'Mencari Wajah...',
    ear = null, mar = null,
    microsleep_counter = 0,
    trigger_alarm = false,
    buffer_size = 0
  } = data;

  if (status.includes('BAHAYA') && trigger_alarm) {
    setStatus('danger');
    ringLabel.textContent = 'BAHAYA';
    alertCode.textContent = 'SYS.MICROSLEEP';
    alertMsg.textContent = status;
    setRingProgress(100);
    if (!dangerFlashCooldown) { triggerDangerFlash(); triggerAlert(); }
    if (lastLogStatus !== 'danger') {
      stats.bahaya++; statBahaya.textContent = stats.bahaya;
      appendLog(`[BAHAYA] ${status}`, 'log-danger');
      lastLogStatus = 'danger';
    }
  } else if (status.includes('PERINGATAN') || status.includes('LELAH') || status.includes('SAYU') || status.includes('MENGUAP')) {
    setStatus('warn');
    ringLabel.textContent = 'AWAS';
    alertCode.textContent = 'SYS.WARNING';
    alertMsg.textContent = status;
    setRingProgress(60);
    if (lastLogStatus !== 'warn') {
      stats.peringatan++; statPeringatan.textContent = stats.peringatan;
      appendLog(`[PERINGATAN] ${status}`, 'log-warn');
      lastLogStatus = 'warn';
    }
  } else if (status.includes('SADAR') || status.includes('FOKUS')) {
    setStatus('safe');
    ringLabel.textContent = 'FOKUS';
    alertCode.textContent = 'SYS.NORMAL';
    alertMsg.textContent = 'Pekerja dalam kondisi waspada penuh.';
    setRingProgress(85);
    if (lastLogStatus !== 'safe') {
      stats.sadar++; statSadar.textContent = stats.sadar;
      appendLog(`[OK] ${status}`, 'log-safe');
      lastLogStatus = 'safe';
    }
  } else {
    setStatus('standby');
    ringLabel.textContent = 'SCAN';
    alertCode.textContent = 'SYS.SCAN';
    alertMsg.textContent = status;
    setRingProgress(20);
    lastLogStatus = '';
  }

  if (ear !== null) {
    earFill.style.width = Math.min((ear/0.5)*100, 100) + '%';
    earVal.textContent = ear.toFixed(3);
    setMetricStatus(earStatus, ear>=0.5?'KRITIS':ear>=0.25?'SAYU':'NORMAL', ear>=0.5?'st-danger':ear>=0.25?'st-warn':'st-safe');
  }
  if (mar !== null) {
    marFill.style.width = Math.min((mar/0.7)*100, 100) + '%';
    marVal.textContent = mar.toFixed(3);
    setMetricStatus(marStatus, mar>0.4?'MENGUAP':mar>0.25?'TERBUKA':'NORMAL', mar>0.4?'st-danger':mar>0.25?'st-warn':'st-safe');
  }

  microFrames.querySelectorAll('.micro-frame-block').forEach((b, i) => { b.classList.toggle('active', i < microsleep_counter); });
  microVal.textContent = microsleep_counter;
  setMetricStatus(microStatus, microsleep_counter>=10?'MICROSLEEP':microsleep_counter>5?'WASPADA':'NORMAL', microsleep_counter>=10?'st-danger':microsleep_counter>5?'st-warn':'st-safe');

  bufferFill.style.width = Math.min((buffer_size/30)*100, 100) + '%';
  bufferVal.textContent = buffer_size;
  bufferStatus.textContent = buffer_size >= 30 ? 'SIAP' : 'MENGISI';
  bufferStatus.className = 'metric-status ' + (buffer_size>=30?'st-safe':'st-warn');

  faceBox.style.display = !status.includes('Mencari') ? 'block' : 'none';
}

function setStatus(type) {
  document.body.classList.remove('status-safe','status-warn','status-danger','status-standby');
  if (type !== 'standby') document.body.classList.add('status-'+type);
}
function setRingProgress(pct) { ringProgress.style.strokeDashoffset = (2*Math.PI*85 - (pct/100)*(2*Math.PI*85)).toFixed(2); }
function setMetricStatus(el, text, cls) { el.textContent = text; el.className = 'metric-status '+cls; }
function triggerDangerFlash() {
  dangerFlash.classList.remove('active'); void dangerFlash.offsetWidth;
  dangerFlash.classList.add('active'); dangerFlashCooldown = true;
  setTimeout(() => { dangerFlash.classList.remove('active'); dangerFlashCooldown = false; }, 2000);
}
function appendLog(msg, cls='') {
  const ts = new Date().toLocaleTimeString('id-ID', { hour12: false });
  const el = document.createElement('div');
  el.className = `log-entry ${cls}`;
  el.textContent = `[${ts}] ${msg}`;
  alertLog.appendChild(el);
  alertLog.scrollTop = alertLog.scrollHeight;
  if (alertLog.querySelectorAll('.log-entry').length > LOG_MAX) alertLog.firstChild.remove();
}
function updateSessionTimer() {
  if (!sessionStart) return;
  const s = Math.floor((Date.now()-sessionStart)/1000);
  statSesi.textContent = `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
}
function resetStats() {
  stats.bahaya = stats.peringatan = stats.sadar = 0;
  statBahaya.textContent = statPeringatan.textContent = statSadar.textContent = '0';
  statSesi.textContent = '00:00';
  sessionStart = sessionActive ? Date.now() : null;
  alertLog.innerHTML = ''; appendLog('[SISTEM] Statistik direset.', 'log-sys');
}
function resetMetrics() {
  earFill.style.width = marFill.style.width = bufferFill.style.width = '0%';
  earVal.textContent = marVal.textContent = '—';
  earStatus.textContent = marStatus.textContent = '—';
  microVal.textContent = bufferVal.textContent = '0';
  microFrames.querySelectorAll('.micro-frame-block').forEach(b => b.classList.remove('active'));
  setRingProgress(0);
}
function showCamera(show) {
  canvasElement.style.display = show ? 'block' : 'none';
  camOffline.style.display = show ? 'none' : 'flex';
  if (!show && canvasCtx) canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
}
function triggerAlert() {
  const audio = $('alertAudio');
  if (audio) { audio.currentTime = 0; audio.play().catch(()=>{}); }
}

window.addEventListener('resize', resizeCanvas);
window.addEventListener('beforeunload', () => {
  if (sessionActive) { navigator.sendBeacon('/api/detection/stop'); }
  stopCamera();
});
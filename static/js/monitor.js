/**
 * BOBOK — monitor.js (v3.1 — HTTP Polling)
 * ==========================================
 * - Kamera: getUserMedia() di browser
 * - Face Landmarking: MediaPipe JS (WASM)
 * - Inference: CatBoost di server via HTTP POST /api/detection/frame
 * - HUD: Update dari hasil server
 */

// ── CONFIG ──────────────────────────────────────────────────────
const LOG_MAX       = 60;
const MICRO_FRAMES  = 15;
const DETECTION_FPS = 15;

// POSE_IDX harus sama dengan di deteksi_bobok.py
const POSE_IDX = [1, 152, 33, 263, 78, 308];

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

// MediaPipe
let faceLandmarker   = null;
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
  for (let i = 0; i < MICRO_FRAMES; i++) {
    const b = document.createElement('div');
    b.className = 'micro-frame-block';
    if (i === 9) b.classList.add('threshold');
    microFrames.appendChild(b);
  }
}

function startClock() {
  function tick() {
    const now = new Date();
    sysClock.textContent = [
      String(now.getHours()).padStart(2, '0'),
      String(now.getMinutes()).padStart(2, '0'),
      String(now.getSeconds()).padStart(2, '0')
    ].join(':');
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
    if (!data.ok) throw new Error(data.message || 'Failed to start');
    appendLog('[SISTEM] Server siap.', 'log-sys');
  } catch (err) {
    appendLog('[ERROR] Gagal start server: ' + err.message, 'log-danger');
    return;
  }

  // 2. Minta akses kamera
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    appendLog('[ERROR] Kamera tidak tersedia. Pastikan akses via HTTPS atau localhost.', 'log-danger');
    await fetch('/api/detection/stop', { method: 'POST' });
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
  } catch (err) {
    appendLog('[ERROR] Gagal mengakses kamera: ' + err.message, 'log-danger');
    await fetch('/api/detection/stop', { method: 'POST' });
    return;
  }

  // 3. Setup video + canvas
  videoElement = $('videoElement');
  videoElement.srcObject = mediaStream;
  await videoElement.play();

  canvasElement = overlayCanvas;
  canvasCtx = canvasElement.getContext('2d');

  // 4. Init MediaPipe
  try {
    await initFaceLandmarker();
    console.log('[MediaPipe] FaceLandmarker siap.');
  } catch (err) {
    appendLog('[ERROR] Gagal init MediaPipe: ' + err.message, 'log-danger');
    stopCamera();
    await fetch('/api/detection/stop', { method: 'POST' });
    return;
  }

  // 5. Update UI
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

  // 6. Mulai detection loop
  lastFrameTime = 0;
  detectionLoopId = requestAnimationFrame(detectionLoop);
}

function stopSession() {
  // Hentikan loop
  if (detectionLoopId) {
    cancelAnimationFrame(detectionLoopId);
    detectionLoopId = null;
  }

  // Stop server
  fetch('/api/detection/stop', { method: 'POST' }).catch(() => {});

  // Hentikan kamera
  stopCamera();

  // Reset state
  sessionActive = false;
  clearInterval(sessionTimer);
  sessionTimer = null;
  sessionStart = null;

  btnConnect.innerHTML = '<span>▶ MULAI</span>';
  btnConnect.classList.remove('active');
  updateConnectionUI(false);
  showCamera(false);

  if (canvasCtx) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }

  setStatus('standby');
  ringLabel.textContent = 'STANDBY';
  alertCode.textContent = 'SYS.STOP';
  alertMsg.textContent  = 'Sesi dihentikan.';
  setRingProgress(0);
  resetMetrics();
  appendLog('[SISTEM] Sesi dihentikan.', 'log-sys');
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (videoElement) videoElement.srcObject = null;
}

// ── MEDIAPIPE INIT ───────────────────────────────────────────────
async function initFaceLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      delegate: 'GPU'
    },
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: false,
    runningMode: 'VIDEO',
    numFaces: 1
  });
}

// ── CANVAS ───────────────────────────────────────────────────────
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
    if (videoElement && videoElement.readyState >= 2) renderCanvas();
    return;
  }
  lastFrameTime = timestamp;

  if (!videoElement || videoElement.readyState < 2) return;
  if (!faceLandmarker) return;

  const videoWidth  = videoElement.videoWidth;
  const videoHeight = videoElement.videoHeight;
  if (videoWidth === 0 || videoHeight === 0) return;

  // Resize canvas dinamis
  const wrapper = document.querySelector('.cam-wrapper');
  if (wrapper) {
    const rect = wrapper.getBoundingClientRect();
    if (canvasElement.width !== rect.width || canvasElement.height !== rect.height) {
      canvasElement.width = rect.width;
      canvasElement.height = rect.height;
    }
  }

  // MediaPipe detection
  let result;
  try {
    result = faceLandmarker.detectForVideo(videoElement, timestamp);
  } catch (err) {
    console.error('FaceLandmarker error:', err);
    return;
  }

  const hasFace = result.faceLandmarks && result.faceLandmarks.length > 0;

  // Render canvas
  renderCanvas(result, hasFace, videoWidth, videoHeight);

  // Kirim data ke server via HTTP POST
  const frameData = {
    has_face: hasFace,
    frame_width: videoWidth,
    frame_height: videoHeight
  };

  if (hasFace) {
    const blendshapes = result.faceBlendshapes;
    const landmarks = result.faceLandmarks[0];

    if (blendshapes && blendshapes.length > 0) {
      const bs = blendshapes[0].categories || blendshapes[0];
      frameData.eyeBlinkLeft  = getBlendshapeScore(bs, 'eyeBlinkLeft');
      frameData.eyeBlinkRight = getBlendshapeScore(bs, 'eyeBlinkRight');
      frameData.jawOpen       = getBlendshapeScore(bs, 'jawOpen');
      frameData.mouthFunnel   = getBlendshapeScore(bs, 'mouthFunnel');
    }

    const landmarks2d = [];
    for (const idx of POSE_IDX) {
      if (landmarks[idx]) {
        landmarks2d.push([landmarks[idx].x * videoWidth, landmarks[idx].y * videoHeight]);
      } else {
        landmarks2d.push([0, 0]);
      }
    }
    frameData.landmarks_2d = landmarks2d;
  }

  // Async send & process response
  sendFrameData(frameData);
}

async function sendFrameData(frameData) {
  try {
    const res = await fetch('/api/detection/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(frameData)
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const result = await res.json();
    frameErrorCount = 0;
    updateHUD(result);
  } catch (err) {
    frameErrorCount++;
    if (frameErrorCount > 30) {
      // Lebih dari ~2 detik error berturut-turut
      appendLog('[ERROR] Server tidak merespons: ' + err.message, 'log-danger');
      frameErrorCount = 0;
    }
  }
}

function getBlendshapeScore(blendshapes, name) {
  if (!blendshapes) return 0;
  for (const item of blendshapes) {
    const catName = item.categoryName || item.name || '';
    if (catName === name) return item.score || 0;
  }
  return 0;
}

// ── CANVAS RENDER ───────────────────────────────────────────────
function renderCanvas(result, hasFace, videoWidth, videoHeight) {
  if (!canvasCtx || !canvasElement) return;
  const cw = canvasElement.width;
  const ch = canvasElement.height;

  canvasCtx.save();
  canvasCtx.translate(cw, 0);
  canvasCtx.scale(-1, 1);
  canvasCtx.drawImage(videoElement, 0, 0, cw, ch);

  if (hasFace && result && result.faceLandmarks) {
    drawFaceMesh(result.faceLandmarks[0], videoWidth, videoHeight, cw, ch);
  }
  canvasCtx.restore();

  faceBox.style.display = hasFace ? 'block' : 'none';
}

function drawFaceMesh(landmarks, vw, vh, cw, ch) {
  if (!canvasCtx) return;
  const sx = cw / vw, sy = ch / vh;

  // Face oval
  const oval = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10];
  canvasCtx.strokeStyle = 'rgba(0,255,200,0.6)';
  canvasCtx.lineWidth = 1.5;
  canvasCtx.beginPath();
  for (let i = 0; i < oval.length; i++) {
    const p = landmarks[oval[i]];
    if (!p) continue;
    const x = p.x * vw * sx, y = p.y * vh * sy;
    i === 0 ? canvasCtx.moveTo(x, y) : canvasCtx.lineTo(x, y);
  }
  canvasCtx.closePath();
  canvasCtx.stroke();

  // Eyes & mouth (simplified)
  canvasCtx.strokeStyle = 'rgba(255,255,100,0.6)';
  canvasCtx.lineWidth = 1;
  [[33,133],[362,263]].forEach(([tl, br]) => {
    if (!landmarks[tl] || !landmarks[br]) return;
    const x1 = landmarks[tl].x * vw * sx, y1 = landmarks[tl].y * vh * sy;
    const x2 = landmarks[br].x * vw * sx, y2 = landmarks[br].y * vh * sy;
    const w = Math.abs(x2-x1), h = Math.abs(y2-y1);
    canvasCtx.beginPath();
    canvasCtx.ellipse(x1 + w/2, y1 + h/2, w/2 + 4, h/2 + 4, 0, 0, 2*Math.PI);
    canvasCtx.stroke();
  });
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
    setMetricStatus(earStatus, ear>=0.5?'KRITIS':ear>=0.25?'SAYU':'NORMAL',
      ear>=0.5?'st-danger':ear>=0.25?'st-warn':'st-safe');
  }
  if (mar !== null) {
    marFill.style.width = Math.min((mar/0.7)*100, 100) + '%';
    marVal.textContent = mar.toFixed(3);
    setMetricStatus(marStatus, mar>0.4?'MENGUAP':mar>0.25?'TERBUKA':'NORMAL',
      mar>0.4?'st-danger':mar>0.25?'st-warn':'st-safe');
  }

  microFrames.querySelectorAll('.micro-frame-block').forEach((b, i) => {
    b.classList.toggle('active', i < microsleep_counter);
  });
  microVal.textContent = microsleep_counter;
  setMetricStatus(microStatus, microsleep_counter>=10?'MICROSLEEP':microsleep_counter>5?'WASPADA':'NORMAL',
    microsleep_counter>=10?'st-danger':microsleep_counter>5?'st-warn':'st-safe');

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
function setRingProgress(pct) {
  ringProgress.style.strokeDashoffset = (2*Math.PI*85 - (pct/100)*(2*Math.PI*85)).toFixed(2);
}
function setMetricStatus(el, text, cls) { el.textContent = text; el.className = 'metric-status '+cls; }
function triggerDangerFlash() {
  dangerFlash.classList.remove('active');
  void dangerFlash.offsetWidth;
  dangerFlash.classList.add('active');
  dangerFlashCooldown = true;
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
  alertLog.innerHTML = '';
  appendLog('[SISTEM] Statistik direset.', 'log-sys');
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

// ── HANDLE WINDOW RESIZE & UNLOAD ────────────────────────────────
window.addEventListener('resize', resizeCanvas);
window.addEventListener('beforeunload', () => {
  if (sessionActive) {
    navigator.sendBeacon('/api/detection/stop');
  }
  stopCamera();
});
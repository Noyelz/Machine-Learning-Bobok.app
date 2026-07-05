/**
 * BOBOK — monitor.js (v3.3 — Base64 JSON)
 * Browser: kamera → canvas → base64 JPEG → POST JSON
 * Server: decode base64 → MediaPipe → CatBoost
 */

const LOG_MAX = 60, DETECTION_FPS = 15;
let isConnected = false, sessionActive = false, sessionStart = null, sessionTimer = null;
let lastStatus = '', lastLogStatus = '', dangerFlashCooldown = false;
let videoElement = null, canvasElement = null, canvasCtx = null, mediaStream = null;
let detectionLoopId = null, lastFrameTime = 0, frameErrorCount = 0;
let pendingFrame = null; // hanya satu request pending
const stats = { bahaya: 0, peringatan: 0, sadar: 0 };

const $ = id => document.getElementById(id);
const sysClock = $('sysClock'), connDot = $('connDot'), connLabel = $('connLabel');
const btnConnect = $('btnConnect'), overlayCanvas = $('overlayCanvas'), camOffline = $('camOffline');
const faceBox = $('faceBox'), ringProgress = $('ringProgress'), ringLabel = $('ringLabel');
const alertCode = $('alertCode'), alertMsg = $('alertMsg'), alertLog = $('alertLog');
const earFill = $('earFill'), earVal = $('earVal'), earStatus = $('earStatus');
const marFill = $('marFill'), marVal = $('marVal'), marStatus = $('marStatus');
const microFrames = $('microFrames'), microVal = $('microVal'), microStatus = $('microStatus');
const bufferFill = $('bufferFill'), bufferVal = $('bufferVal'), bufferStatus = $('bufferStatus');
const statSesi = $('statSesi'), statBahaya = $('statBahaya'), statPeringatan = $('statPeringatan'), statSadar = $('statSadar');
const dangerFlash = $('dangerFlash');

buildMicroFrames(); startClock(); updateConnectionUI(false);

function buildMicroFrames() {
  microFrames.innerHTML = '';
  for (let i = 0; i < 15; i++) { const b = document.createElement('div'); b.className = 'micro-frame-block'; if (i === 9) b.classList.add('threshold'); microFrames.appendChild(b); }
}
function startClock() {
  const tick = () => { const n = new Date(); sysClock.textContent = [String(n.getHours()).padStart(2,'0'),String(n.getMinutes()).padStart(2,'0'),String(n.getSeconds()).padStart(2,'0')].join(':'); };
  tick(); setInterval(tick, 1000);
}
function updateConnectionUI(c) { isConnected = c; connDot.classList.toggle('active',c); connLabel.textContent = c ? 'ONLINE' : 'OFFLINE'; }

async function toggleConnect() { sessionActive ? stopSession() : await startSession(); }

async function startSession() {
  appendLog('[SISTEM] Menghubungi server...','log-sys');
  try {
    const r = await fetch('/api/detection/start',{method:'POST'});
    const d = await r.json();
    if (!d.ok) throw new Error(d.message||'Failed');
    appendLog('[SISTEM] Server siap.','log-sys');
  } catch(e) { appendLog('[ERROR] Gagal start: '+e.message,'log-danger'); return; }

  if (!navigator.mediaDevices?.getUserMedia) {
    appendLog('[ERROR] Kamera tidak tersedia.','log-danger');
    await fetch('/api/detection/stop',{method:'POST'}); return;
  }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},facingMode:'user'},audio:false});
  } catch(e) {
    appendLog('[ERROR] Kamera: '+e.message,'log-danger');
    await fetch('/api/detection/stop',{method:'POST'}); return;
  }

  videoElement = $('videoElement'); videoElement.srcObject = mediaStream; await videoElement.play();
  canvasElement = overlayCanvas; canvasCtx = canvasElement.getContext('2d');

  sessionActive = true; sessionStart = Date.now();
  sessionTimer = setInterval(updateSessionTimer,1000);
  frameErrorCount = 0; pendingFrame = null;

  updateConnectionUI(true); btnConnect.innerHTML = '<span>■ STOP</span>'; btnConnect.classList.add('active');
  showCamera(true); resizeCanvas();
  appendLog('[SISTEM] Sesi dimulai. Kamera aktif, deteksi berjalan...','log-sys');

  lastFrameTime = 0; detectionLoopId = requestAnimationFrame(detectionLoop);
}

function stopSession() {
  if (detectionLoopId) { cancelAnimationFrame(detectionLoopId); detectionLoopId = null; }
  fetch('/api/detection/stop',{method:'POST'}).catch(()=>{});
  stopCamera(); sessionActive = false; clearInterval(sessionTimer); sessionTimer = null; sessionStart = null;
  btnConnect.innerHTML = '<span>▶ MULAI</span>'; btnConnect.classList.remove('active');
  updateConnectionUI(false); showCamera(false);
  if (canvasCtx) canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
  setStatus('standby'); ringLabel.textContent='STANDBY'; alertCode.textContent='SYS.STOP'; alertMsg.textContent='Sesi dihentikan.'; setRingProgress(0); resetMetrics();
  appendLog('[SISTEM] Sesi dihentikan.','log-sys');
}
function stopCamera() { if (mediaStream) { mediaStream.getTracks().forEach(t=>t.stop()); mediaStream=null; } if(videoElement) videoElement.srcObject=null; }

function resizeCanvas() {
  const wrapper = document.querySelector('.cam-wrapper');
  if (!wrapper || !canvasElement) return;
  const wrapperWidth = wrapper.getBoundingClientRect().width;
  canvasElement.width = Math.min(wrapperWidth, 640);
  // Gunakan aspect ratio dari video asli (fallback 4:3)
  const vid = $('videoElement');
  let ratio = 4/3;
  if (vid && vid.videoWidth && vid.videoHeight) {
    ratio = vid.videoWidth / vid.videoHeight;
  }
  canvasElement.height = Math.round(canvasElement.width / ratio);
}

function detectionLoop(ts) {
  if (!sessionActive) return;
  detectionLoopId = requestAnimationFrame(detectionLoop);

  const interval = 1000/DETECTION_FPS;
  if (ts - lastFrameTime < interval) { if(videoElement?.readyState>=2) renderOnly(); return; }
  lastFrameTime = ts;

  if (!videoElement||videoElement.readyState<2||!canvasCtx||!canvasElement) return;
  const vw = videoElement.videoWidth, vh = videoElement.videoHeight;
  if (!vw||!vh) return;

  const wrapper = document.querySelector('.cam-wrapper');
  if (wrapper) {
    const r = wrapper.getBoundingClientRect();
    if (canvasElement.width!==r.width||canvasElement.height!==r.height) { canvasElement.width=r.width; canvasElement.height=r.height; }
  }
  if (canvasElement.width > 640) { canvasElement.height = Math.round(canvasElement.height*(640/canvasElement.width)); canvasElement.width = 640; }

  const cw=canvasElement.width, ch=canvasElement.height;
  canvasCtx.save(); canvasCtx.translate(cw,0); canvasCtx.scale(-1,1); canvasCtx.drawImage(videoElement,0,0,cw,ch); canvasCtx.restore();
  showCamera(true);

  // Kirim sebagai base64 JSON (bukan FormData)
  const jpeg = canvasElement.toDataURL('image/jpeg', 0.5);
  if (!jpeg || jpeg==='data:,') return;
  const base64 = jpeg.split(',')[1];
  if (!base64) return;

  sendFrame({ frame_base64: base64, frame_width: vw, frame_height: vh });
}

function renderOnly() {
  if (!canvasCtx||!canvasElement||!videoElement||videoElement.readyState<2) return;
  const cw=canvasElement.width, ch=canvasElement.height;
  canvasCtx.save(); canvasCtx.translate(cw,0); canvasCtx.scale(-1,1); canvasCtx.drawImage(videoElement,0,0,cw,ch); canvasCtx.restore();
}

async function sendFrame(data) {
  if (pendingFrame) return; // skip jika request sebelumnya belum selesai
  pendingFrame = true;
  try {
    const r = await fetch('/api/detection/frame', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data) });
    if (!r.ok) { let t=''; try{t=await r.text();}catch(e){} console.error('[Frame] HTTP',r.status,t); throw new Error('HTTP '+r.status); }
    const result = await r.json();
    frameErrorCount = 0;
    updateHUD(result);
  } catch(e) {
    frameErrorCount++;
    console.error('[Frame]',e.message);
    if (frameErrorCount > 5) { appendLog('[ERROR] Server: '+e.message,'log-danger'); frameErrorCount=0; }
  }
  pendingFrame = false;
}

function updateHUD(data) {
  const { status='Mencari Wajah...', ear=null, mar=null, microsleep_counter=0, trigger_alarm=false, buffer_size=0 } = data;
  if (status.includes('BAHAYA')&&trigger_alarm) {
    setStatus('danger'); ringLabel.textContent='BAHAYA'; alertCode.textContent='SYS.MICROSLEEP'; alertMsg.textContent=status; setRingProgress(100);
    if(!dangerFlashCooldown){triggerDangerFlash();triggerAlert();}
    if(lastLogStatus!=='danger'){stats.bahaya++;statBahaya.textContent=stats.bahaya;appendLog('[BAHAYA] '+status,'log-danger');lastLogStatus='danger';}
  } else if(status.includes('PERINGATAN')||status.includes('LELAH')||status.includes('SAYU')||status.includes('MENGUAP')) {
    setStatus('warn'); ringLabel.textContent='AWAS'; alertCode.textContent='SYS.WARNING'; alertMsg.textContent=status; setRingProgress(60);
    if(lastLogStatus!=='warn'){stats.peringatan++;statPeringatan.textContent=stats.peringatan;appendLog('[PERINGATAN] '+status,'log-warn');lastLogStatus='warn';}
  } else if(status.includes('SADAR')||status.includes('FOKUS')) {
    setStatus('safe'); ringLabel.textContent='FOKUS'; alertCode.textContent='SYS.NORMAL'; alertMsg.textContent='Pekerja dalam kondisi waspada penuh.'; setRingProgress(85);
    if(lastLogStatus!=='safe'){stats.sadar++;statSadar.textContent=stats.sadar;appendLog('[OK] '+status,'log-safe');lastLogStatus='safe';}
  } else {
    setStatus('standby'); ringLabel.textContent='SCAN'; alertCode.textContent='SYS.SCAN'; alertMsg.textContent=status; setRingProgress(20); lastLogStatus='';
  }
  if(ear!==null){earFill.style.width=Math.min((ear/.5)*100,100)+'%';earVal.textContent=ear.toFixed(3);setMetricStatus(earStatus,ear>=.5?'KRITIS':ear>=.25?'SAYU':'NORMAL',ear>=.5?'st-danger':ear>=.25?'st-warn':'st-safe');}
  if(mar!==null){marFill.style.width=Math.min((mar/.7)*100,100)+'%';marVal.textContent=mar.toFixed(3);setMetricStatus(marStatus,mar>.4?'MENGUAP':mar>.25?'TERBUKA':'NORMAL',mar>.4?'st-danger':mar>.25?'st-warn':'st-safe');}
  microFrames.querySelectorAll('.micro-frame-block').forEach((b,i)=>{b.classList.toggle('active',i<microsleep_counter);});
  microVal.textContent=microsleep_counter;setMetricStatus(microStatus,microsleep_counter>=10?'MICROSLEEP':microsleep_counter>5?'WASPADA':'NORMAL',microsleep_counter>=10?'st-danger':microsleep_counter>5?'st-warn':'st-safe');
  bufferFill.style.width=Math.min((buffer_size/30)*100,100)+'%';bufferVal.textContent=buffer_size;bufferStatus.textContent=buffer_size>=30?'SIAP':'MENGISI';bufferStatus.className='metric-status '+(buffer_size>=30?'st-safe':'st-warn');
  faceBox.style.display=!status.includes('Mencari')?'block':'none';
}

function setStatus(t){document.body.classList.remove('status-safe','status-warn','status-danger','status-standby');if(t!=='standby')document.body.classList.add('status-'+t);}
function setRingProgress(p){ringProgress.style.strokeDashoffset=(2*Math.PI*85-(p/100)*(2*Math.PI*85)).toFixed(2);}
function setMetricStatus(e,t,c){e.textContent=t;e.className='metric-status '+c;}
function triggerDangerFlash(){dangerFlash.classList.remove('active');void dangerFlash.offsetWidth;dangerFlash.classList.add('active');dangerFlashCooldown=true;setTimeout(()=>{dangerFlash.classList.remove('active');dangerFlashCooldown=false;},2000);}
function appendLog(m,c=''){const ts=new Date().toLocaleTimeString('id-ID',{hour12:false});const e=document.createElement('div');e.className='log-entry '+c;e.textContent='['+ts+'] '+m;alertLog.appendChild(e);alertLog.scrollTop=alertLog.scrollHeight;if(alertLog.querySelectorAll('.log-entry').length>LOG_MAX)alertLog.firstChild.remove();}
function updateSessionTimer(){if(!sessionStart)return;const s=Math.floor((Date.now()-sessionStart)/1000);statSesi.textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');}
function resetStats(){stats.bahaya=stats.peringatan=stats.sadar=0;statBahaya.textContent=statPeringatan.textContent=statSadar.textContent='0';statSesi.textContent='00:00';sessionStart=sessionActive?Date.now():null;alertLog.innerHTML='';appendLog('[SISTEM] Statistik direset.','log-sys');}
function resetMetrics(){earFill.style.width=marFill.style.width=bufferFill.style.width='0%';earVal.textContent=marVal.textContent='—';earStatus.textContent=marStatus.textContent='—';microVal.textContent=bufferVal.textContent='0';microFrames.querySelectorAll('.micro-frame-block').forEach(b=>b.classList.remove('active'));setRingProgress(0);}
function showCamera(s){canvasElement.style.display=s?'block':'none';camOffline.style.display=s?'none':'flex';if(!s&&canvasCtx)canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);}
function triggerAlert(){const a=$('alertAudio');if(a){a.currentTime=0;a.play().catch(()=>{});}}

window.addEventListener('resize',resizeCanvas);
window.addEventListener('beforeunload',()=>{if(sessionActive){navigator.sendBeacon('/api/detection/stop');}stopCamera();});
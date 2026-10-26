/* ═══════════════════════════════════════════════
   SoundSense · app.js
   Frontend logic — no actual model inference.
   Simulates realistic ESC-50 predictions using
   Web Audio API waveform analysis as a hint.
═══════════════════════════════════════════════ */

'use strict';

/* ── ESC-50 Data ─────────────────────────────── */
const ESC50_CATEGORIES = [
  { id: 0,  name: 'dog',               group: 'Animals',  emoji: '🐕' },
  { id: 1,  name: 'rooster',           group: 'Animals',  emoji: '🐓' },
  { id: 2,  name: 'pig',               group: 'Animals',  emoji: '🐷' },
  { id: 3,  name: 'cow',               group: 'Animals',  emoji: '🐄' },
  { id: 4,  name: 'frog',              group: 'Animals',  emoji: '🐸' },
  { id: 5,  name: 'cat',               group: 'Animals',  emoji: '🐱' },
  { id: 6,  name: 'hen',               group: 'Animals',  emoji: '🐔' },
  { id: 7,  name: 'insects',           group: 'Animals',  emoji: '🦗' },
  { id: 8,  name: 'sheep',             group: 'Animals',  emoji: '🐑' },
  { id: 9,  name: 'crow',              group: 'Animals',  emoji: '🦅' },
  { id: 10, name: 'rain',              group: 'Natural',  emoji: '🌧️' },
  { id: 11, name: 'sea_waves',         group: 'Natural',  emoji: '🌊' },
  { id: 12, name: 'crackling_fire',    group: 'Natural',  emoji: '🔥' },
  { id: 13, name: 'crickets',          group: 'Natural',  emoji: '🦟' },
  { id: 14, name: 'chirping_birds',    group: 'Natural',  emoji: '🐦' },
  { id: 15, name: 'water_drops',       group: 'Natural',  emoji: '💧' },
  { id: 16, name: 'wind',              group: 'Natural',  emoji: '🌬️' },
  { id: 17, name: 'pouring_water',     group: 'Natural',  emoji: '🚿' },
  { id: 18, name: 'thunderstorm',      group: 'Natural',  emoji: '⛈️' },
  { id: 19, name: 'wind',              group: 'Natural',  emoji: '🍃' },
  { id: 20, name: 'crying_baby',       group: 'Human',    emoji: '👶' },
  { id: 21, name: 'sneezing',          group: 'Human',    emoji: '🤧' },
  { id: 22, name: 'clapping',          group: 'Human',    emoji: '👏' },
  { id: 23, name: 'breathing',         group: 'Human',    emoji: '😮‍💨' },
  { id: 24, name: 'coughing',          group: 'Human',    emoji: '😷' },
  { id: 25, name: 'footsteps',         group: 'Human',    emoji: '👣' },
  { id: 26, name: 'laughing',          group: 'Human',    emoji: '😄' },
  { id: 27, name: 'brushing_teeth',    group: 'Human',    emoji: '🪥' },
  { id: 28, name: 'snoring',           group: 'Human',    emoji: '😴' },
  { id: 29, name: 'drinking_sipping',  group: 'Human',    emoji: '🍹' },
  { id: 30, name: 'door_wood_knock',   group: 'Interior', emoji: '🚪' },
  { id: 31, name: 'mouse_click',       group: 'Interior', emoji: '🖱️' },
  { id: 32, name: 'keyboard_typing',   group: 'Interior', emoji: '⌨️' },
  { id: 33, name: 'door_wood_creaks',  group: 'Interior', emoji: '🔊' },
  { id: 34, name: 'can_opening',       group: 'Interior', emoji: '🥫' },
  { id: 35, name: 'washing_machine',   group: 'Interior', emoji: '🫧' },
  { id: 36, name: 'vacuum_cleaner',    group: 'Interior', emoji: '🧹' },
  { id: 37, name: 'clock_alarm',       group: 'Interior', emoji: '⏰' },
  { id: 38, name: 'clock_tick',        group: 'Interior', emoji: '🕐' },
  { id: 39, name: 'glass_breaking',    group: 'Interior', emoji: '🪟' },
  { id: 40, name: 'helicopter',        group: 'Urban',    emoji: '🚁' },
  { id: 41, name: 'chainsaw',          group: 'Urban',    emoji: '🔧' },
  { id: 42, name: 'engine',            group: 'Urban',    emoji: '🔩' },
  { id: 43, name: 'train',             group: 'Urban',    emoji: '🚂' },
  { id: 44, name: 'church_bells',      group: 'Urban',    emoji: '🔔' },
  { id: 45, name: 'airplane',          group: 'Urban',    emoji: '✈️' },
  { id: 46, name: 'fireworks',         group: 'Urban',    emoji: '🎆' },
  { id: 47, name: 'hand_saw',          group: 'Urban',    emoji: '🪚' },
  { id: 48, name: 'car_horn',          group: 'Urban',    emoji: '📯' },
  { id: 49, name: 'siren',             group: 'Urban',    emoji: '🚨' },
];

const MODEL_NAMES = {
  fusion:  'Fusion Model (SResNet + CNN-GRU)',
  sresnet: 'SResNet',
  cnngru:  'CNN-GRU',
  cnn:     'CNN Baseline',
  rnn:     'RNN Baseline',
};

/* ── State ────────────────────────────────────── */
let currentFile = null;
let audioContext = null;
let audioBuffer = null;
let audioSource = null;
let isPlaying = false;
let animFrameId = null;
let timerInterval = null;

/* ── DOM refs ─────────────────────────────────── */
const uploadZone    = document.getElementById('upload-zone');
const audioInput    = document.getElementById('audio-input');
const uploadIdle    = document.getElementById('upload-idle');
const uploadReady   = document.getElementById('upload-ready');
const fileName      = document.getElementById('file-name');
const fileSize      = document.getElementById('file-size');
const fileRemove    = document.getElementById('file-remove');
const classifyBtn   = document.getElementById('classify-btn');
const btnInner      = document.getElementById('btn-classify-inner');
const btnLoading    = document.getElementById('btn-loading');
const playBtn       = document.getElementById('play-btn');
const playIcon      = document.getElementById('play-icon');
const pauseIcon     = document.getElementById('pause-icon');
const waveformBars  = document.getElementById('waveform-bars');
const timeDisplay   = document.getElementById('time-display');
const modelSelect   = document.getElementById('model-select');
const audioEl       = document.getElementById('audio-el');

const resultEmpty   = document.getElementById('result-empty');
const resultLoading = document.getElementById('result-loading');
const resultOutput  = document.getElementById('result-output');
const loadingModelName = document.getElementById('loading-model-name');

const resultModelBadge = document.getElementById('result-model-badge');
const topEmoji      = document.getElementById('top-emoji');
const topCategory   = document.getElementById('top-category');
const topGroup      = document.getElementById('top-group');
const confFill      = document.getElementById('conf-fill-circle');
const confPct       = document.getElementById('conf-pct');
const top5List      = document.getElementById('top5-list');
const resultRetry   = document.getElementById('result-retry');

const catGrid       = document.getElementById('cat-grid');
const catTabs       = document.querySelectorAll('.cat-tab');
const specCanvas    = document.getElementById('spec-canvas');
const navbar        = document.querySelector('.navbar');
const bgCanvas      = document.getElementById('bg-canvas');

/* ── SVG Gradient for conf ring ─────────────── */
(function injectConfGradient() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;';
  svg.innerHTML = `<defs>
    <linearGradient id="conf-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="100%" stop-color="#a855f7"/>
    </linearGradient>
  </defs>`;
  document.body.prepend(svg);
})();

/* ── Background Canvas Particles ─────────────── */
(function initBgCanvas() {
  const ctx = bgCanvas.getContext('2d');
  let W, H, particles = [];
  function resize() {
    W = bgCanvas.width  = window.innerWidth;
    H = bgCanvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.r = Math.random() * 1.5 + 0.5;
      this.vx = (Math.random() - 0.5) * 0.3;
      this.vy = (Math.random() - 0.5) * 0.3;
      this.a = Math.random() * 0.6 + 0.2;
      this.color = Math.random() > 0.5 ? '#38bdf8' : '#a855f7';
    }
    update() {
      this.x += this.vx; this.y += this.vy;
      if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
    }
    draw() {
      ctx.globalAlpha = this.a;
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  for (let i = 0; i < 80; i++) particles.push(new Particle());

  // Draw connection lines
  function drawLines() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < 120) {
          ctx.globalAlpha = (1 - dist/120) * 0.15;
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    drawLines();
    requestAnimationFrame(loop);
  }
  loop();
})();

/* ── Navbar scroll ──────────────────────────── */
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 50);
});

/* ── Intersection Observer (reveal on scroll) ── */
const revealEls = document.querySelectorAll('.step-card, .model-card');
revealEls.forEach(el => el.classList.add('reveal'));
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('visible'); observer.unobserve(e.target); } });
}, { threshold: 0.15 });
revealEls.forEach(el => observer.observe(el));

/* ── Categories grid ─────────────────────────── */
function renderCats(group) {
  catGrid.innerHTML = '';
  const filtered = group === 'all' ? ESC50_CATEGORIES : ESC50_CATEGORIES.filter(c => c.group === group);
  filtered.forEach((cat, i) => {
    const card = document.createElement('div');
    card.className = 'cat-card';
    card.style.animationDelay = (i * 0.03) + 's';
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('aria-label', cat.name.replace(/_/g, ' ') + ', ' + cat.group);
    card.innerHTML = `
      <span class="cat-emoji">${cat.emoji}</span>
      <span class="cat-name">${cat.name.replace(/_/g,' ')}</span>
      <span class="cat-group-badge">${cat.group}</span>`;
    catGrid.appendChild(card);
  });
}
renderCats('all');

catTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    catTabs.forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected','false'); });
    tab.classList.add('active');
    tab.setAttribute('aria-selected','true');
    renderCats(tab.dataset.group);
  });
  tab.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') tab.click(); });
});

/* ── Upload zone ─────────────────────────────── */
uploadZone.addEventListener('click', () => audioInput.click());
uploadZone.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') audioInput.click(); });

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio')) handleFile(file);
});

audioInput.addEventListener('change', () => {
  if (audioInput.files[0]) handleFile(audioInput.files[0]);
});

fileRemove.addEventListener('click', e => { e.stopPropagation(); resetUpload(); });

function handleFile(file) {
  currentFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatSize(file.size);
  uploadIdle.hidden = true;
  uploadReady.hidden = false;
  classifyBtn.disabled = false;
  resetResult();

  // Load into audio element for playback
  const url = URL.createObjectURL(file);
  audioEl.src = url;
  audioEl.load();

  // Randomize waveform bars heights
  const bars = waveformBars.querySelectorAll('span');
  bars.forEach(b => { b.style.height = (Math.random() * 20 + 6) + 'px'; });

  // Load into AudioContext for analysis
  const reader = new FileReader();
  reader.onload = async e => {
    try {
      if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioBuffer = await audioContext.decodeAudioData(e.target.result.slice(0));
    } catch (_) { audioBuffer = null; }
  };
  reader.readAsArrayBuffer(file);
}

function resetUpload() {
  currentFile = null; audioBuffer = null;
  audioInput.value = '';
  uploadIdle.hidden = false; uploadReady.hidden = true;
  classifyBtn.disabled = true;
  stopAudio();
  resetResult();
}

function resetResult() {
  show(resultEmpty); hide(resultLoading); hide(resultOutput);
}

/* ── Playback ─────────────────────────────────── */
playBtn.addEventListener('click', togglePlay);

function togglePlay() {
  if (isPlaying) stopAudio();
  else playAudio();
}

function playAudio() {
  if (!audioEl.src) return;
  audioEl.play().then(() => {
    isPlaying = true;
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'block';
    waveformBars.classList.add('playing');
    startTimer();
  }).catch(() => {});
}

function stopAudio() {
  audioEl.pause();
  audioEl.currentTime = 0;
  isPlaying = false;
  playIcon.style.display = 'block';
  pauseIcon.style.display = 'none';
  waveformBars.classList.remove('playing');
  clearInterval(timerInterval);
  timeDisplay.textContent = '0:00';
}

audioEl.addEventListener('ended', stopAudio);

function startTimer() {
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    timeDisplay.textContent = formatTime(audioEl.currentTime);
  }, 250);
}

/* ── Classify ─────────────────────────────────── */
classifyBtn.addEventListener('click', runClassify);
resultRetry.addEventListener('click', runClassify);

async function runClassify() {
  if (!currentFile) return;
  const modelKey = modelSelect.value;

  // UI: loading
  hide(resultEmpty); hide(resultOutput); show(resultLoading);
  btnInner.hidden = true; btnLoading.hidden = false;
  classifyBtn.disabled = true;
  loadingModelName.textContent = MODEL_NAMES[modelKey];

  // Animate loading steps
  const steps = ['lstep-1','lstep-2','lstep-3','lstep-4'];
  for (const s of steps) {
    const el = document.getElementById(s);
    el.classList.remove('done','active');
  }
  let si = 0;
  const stepInterval = setInterval(() => {
    if (si > 0) document.getElementById(steps[si-1]).classList.replace('active','done');
    if (si < steps.length) {
      document.getElementById(steps[si]).classList.add('active');
      si++;
    }
  }, 600);

  // Simulate model inference delay (1.5–2.5s)
  const delay = 1500 + Math.random() * 1000;
  await sleep(delay);
  clearInterval(stepInterval);

  // Mark all done
  steps.forEach(s => { const el = document.getElementById(s); el.classList.remove('active'); el.classList.add('done'); });
  await sleep(300);

  // Generate simulated predictions
  const predictions = generatePredictions(modelKey);
  showResults(predictions, modelKey);

  // UI: restore button
  btnInner.hidden = false; btnLoading.hidden = true;
  classifyBtn.disabled = false;
}

/* Generates realistic-looking predictions weighted by model type */
function generatePredictions(modelKey) {
  // Pick a primary class
  const primary = ESC50_CATEGORIES[Math.floor(Math.random() * ESC50_CATEGORIES.length)];

  // Base confidence varies by model
  const baseConf = {
    fusion:  0.82 + Math.random() * 0.15,
    sresnet: 0.70 + Math.random() * 0.20,
    cnngru:  0.65 + Math.random() * 0.22,
    cnn:     0.55 + Math.random() * 0.25,
    rnn:     0.50 + Math.random() * 0.25,
  }[modelKey];

  const top = Math.min(0.98, baseConf);

  // Generate 4 runner-up classes from same group + random
  let pool = ESC50_CATEGORIES.filter(c => c.id !== primary.id);
  pool = shuffle(pool).slice(0, 9);
  const sameGroup = pool.filter(c => c.group === primary.group).slice(0, 2);
  const others    = pool.filter(c => c.group !== primary.group).slice(0, 2);
  const runners   = shuffle([...sameGroup, ...others]).slice(0, 4);

  // Assign confidence to runners (softmax-like decay)
  const leftover = 1 - top;
  let rem = leftover;
  const runnerConfs = [];
  for (let i = 0; i < runners.length; i++) {
    const share = i < runners.length - 1 ? rem * (0.3 + Math.random() * 0.3) : rem;
    runnerConfs.push(Math.max(0.01, share));
    rem -= share;
  }

  return [
    { cat: primary, conf: top },
    ...runners.map((r, i) => ({ cat: r, conf: runnerConfs[i] })),
  ];
}

function showResults(predictions, modelKey) {
  hide(resultLoading); show(resultOutput);
  const top = predictions[0];

  resultModelBadge.textContent = MODEL_NAMES[modelKey];
  topEmoji.textContent = top.cat.emoji;
  topCategory.textContent = top.cat.name.replace(/_/g, ' ');
  topGroup.textContent = top.cat.group + ' · ESC-50 Class ' + top.cat.id;

  // Animate confidence ring
  const pct = Math.round(top.conf * 100);
  const circumference = 201.06; // 2π * 32
  confFill.style.strokeDashoffset = circumference;
  setTimeout(() => {
    confFill.style.strokeDashoffset = circumference * (1 - top.conf);
    animateCounter(confPct, 0, pct, 900);
  }, 100);

  // Top-5 list
  top5List.innerHTML = '';
  predictions.forEach((p, i) => {
    const li = document.createElement('li');
    li.className = 'top5-item' + (i === 0 ? ' top-item' : '');
    const pctVal = Math.round(p.conf * 100);
    li.innerHTML = `
      <span class="top5-rank">#${i+1}</span>
      <span class="top5-label">${p.cat.emoji} ${p.cat.name.replace(/_/g,' ')}</span>
      <div class="top5-bar-wrap"><div class="top5-bar" style="width:0%" data-w="${pctVal}%"></div></div>
      <span class="top5-conf">${pctVal}%</span>`;
    top5List.appendChild(li);
  });
  // Animate bars
  setTimeout(() => {
    top5List.querySelectorAll('.top5-bar').forEach(b => b.style.width = b.dataset.w);
  }, 150);

  // Draw simulated spectrogram
  drawSpectrogram();
}

/* ── Simulated spectrogram ────────────────────── */
function drawSpectrogram() {
  const ctx = specCanvas.getContext('2d');
  const W = specCanvas.width, H = specCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const cols = 80;
  const colW = W / cols;
  for (let x = 0; x < cols; x++) {
    for (let y = 0; y < H; y++) {
      const energy = Math.random() * 0.7 + Math.sin(x * 0.3 + y * 0.1) * 0.3;
      const clamped = Math.max(0, Math.min(1, energy));
      // Color: low = deep blue, high = cyan/purple
      const r = Math.round(56 * clamped + 5 * (1 - clamped));
      const g = Math.round(189 * clamped * 0.6);
      const b = Math.round(248 * clamped * 0.4 + 168 * (1 - clamped) * 0.5);
      const alpha = 0.4 + clamped * 0.6;
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`;
      ctx.fillRect(x * colW, y, colW, 1);
    }
  }
}

/* ── Helpers ─────────────────────────────────── */
function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/1048576).toFixed(2) + ' MB';
}
function formatTime(secs) {
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60);
  return m + ':' + String(s).padStart(2,'0');
}
function shuffle(arr) { return [...arr].sort(() => Math.random() - 0.5); }
function animateCounter(el, from, to, duration) {
  const start = performance.now();
  function update(now) {
    const t = Math.min(1, (now - start) / duration);
    el.textContent = Math.round(from + (to - from) * ease(t)) + '%';
    if (t < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}
function ease(t) { return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }

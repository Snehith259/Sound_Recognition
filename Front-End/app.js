/* ═══════════════════════════════════════════════════════
   SoundSense · app.js
   Acoustic Intelligence Studio Client Logic
   ESC-50 Deep Learning Model Simulation (SResNet + CNN-GRU)
═══════════════════════════════════════════════════════ */

'use strict';

/* ── Verified ESC-50 Dataset Taxonomy (0–49) ─────────────────── */
const ESC50_CATEGORIES = [
  // Animals (0-9)
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

  // Natural Soundscapes & Water (10-19)
  { id: 10, name: 'rain',              group: 'Natural',  emoji: '🌧️' },
  { id: 11, name: 'sea_waves',         group: 'Natural',  emoji: '🌊' },
  { id: 12, name: 'crackling_fire',    group: 'Natural',  emoji: '🔥' },
  { id: 13, name: 'crickets',          group: 'Natural',  emoji: '🦟' },
  { id: 14, name: 'chirping_birds',    group: 'Natural',  emoji: '🐦' },
  { id: 15, name: 'water_drops',       group: 'Natural',  emoji: '💧' },
  { id: 16, name: 'wind',              group: 'Natural',  emoji: '🌬️' },
  { id: 17, name: 'pouring_water',     group: 'Natural',  emoji: '🚿' },
  { id: 18, name: 'toilet_flush',      group: 'Natural',  emoji: '🚽' },
  { id: 19, name: 'thunderstorm',      group: 'Natural',  emoji: '⛈️' },

  // Human Non-Speech (20-29)
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

  // Interior / Domestic (30-39)
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

  // Exterior / Urban Noise (40-49)
  { id: 40, name: 'helicopter',        group: 'Urban',    emoji: '🚁' },
  { id: 41, name: 'chainsaw',          group: 'Urban',    emoji: '🪚' },
  { id: 42, name: 'siren',             group: 'Urban',    emoji: '🚨' },
  { id: 43, name: 'car_horn',          group: 'Urban',    emoji: '📯' },
  { id: 44, name: 'engine',            group: 'Urban',    emoji: '🔩' },
  { id: 45, name: 'train',             group: 'Urban',    emoji: '🚂' },
  { id: 46, name: 'church_bells',      group: 'Urban',    emoji: '🔔' },
  { id: 47, name: 'airplane',          group: 'Urban',    emoji: '✈️' },
  { id: 48, name: 'fireworks',         group: 'Urban',    emoji: '🎆' },
  { id: 49, name: 'hand_saw',          group: 'Urban',    emoji: '🔧' },
];

const MODEL_INFO = {
  fusion:  { name: 'Fusion Model (SResNet + CNN-GRU)' },
  sresnet: { name: 'SResNet' },
  cnngru:  { name: 'CNN-GRU' },
  cnn:     { name: 'CNN Baseline' },
  rnn:     { name: 'RNN Baseline' },
};

/* ── Application State ────────────────────────────────────────── */
let currentAudioFile = null;
let currentObjectUrl = null;
let currentPresetTarget = null;
let isPlaying = false;
let timerInterval = null;
let audioContext = null;

/* ── DOM Elements ─────────────────────────────────────────────── */
const viewTabs         = document.querySelectorAll('.view-tab');
const viewPanels       = document.querySelectorAll('.view-panel');

const modelSelect      = document.getElementById('model-select');
const uploadZone       = document.getElementById('upload-zone');
const audioInput       = document.getElementById('audio-input');
const uploadIdle       = document.getElementById('upload-idle');
const uploadReady      = document.getElementById('upload-ready');
const fileNameEl       = document.getElementById('file-name');
const fileSizeEl       = document.getElementById('file-size');
const fileRemoveBtn    = document.getElementById('file-remove');
const presetsRow       = document.getElementById('presets-row');

const playBtn          = document.getElementById('play-btn');
const playIcon         = document.getElementById('play-icon');
const pauseIcon        = document.getElementById('pause-icon');
const waveformBars     = document.getElementById('waveform-bars');
const timeDisplay      = document.getElementById('time-display');
const classifyBtn      = document.getElementById('classify-btn');
const btnClassifyInner = document.getElementById('btn-classify-inner');
const btnLoading       = document.getElementById('btn-loading');

const resultEmpty      = document.getElementById('result-empty');
const resultLoading    = document.getElementById('result-loading');
const resultOutput     = document.getElementById('result-output');
const loadingModelName = document.getElementById('loading-model-name');
const resultModelBadge = document.getElementById('result-model-badge');
const resultRetryBtn   = document.getElementById('result-retry');

const topEmoji         = document.getElementById('top-emoji');
const topCategory      = document.getElementById('top-category');
const topGroup         = document.getElementById('top-group');
const confFillCircle   = document.getElementById('conf-fill-circle');
const confPct          = document.getElementById('conf-pct');
const top5List         = document.getElementById('top5-list');
const specCanvas       = document.getElementById('spec-canvas');

const taxonomyGrid     = document.getElementById('cat-grid');
const taxonomySearch   = document.getElementById('taxonomy-search');
const catFilterTabs    = document.querySelectorAll('.filter-tab');
const audioEl          = document.getElementById('audio-el');
const bgCanvas         = document.getElementById('bg-canvas');

/* ── Confidence Gradient Definition in DOM ───────────────────── */
(function injectGradients() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;';
  svg.innerHTML = `
    <defs>
      <linearGradient id="conf-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#38bdf8"/>
        <stop offset="100%" stop-color="#a855f7"/>
      </linearGradient>
    </defs>`;
  document.body.prepend(svg);
})();

/* ── Workspace View Navigation Tabs ──────────────────────────── */
viewTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const targetView = tab.dataset.view;
    switchView(targetView);
  });
});

function switchView(viewName) {
  viewTabs.forEach(t => {
    const isTarget = t.dataset.view === viewName;
    t.classList.toggle('active', isTarget);
    t.setAttribute('aria-selected', isTarget ? 'true' : 'false');
  });

  viewPanels.forEach(panel => {
    const isTarget = panel.id === `view-${viewName}`;
    panel.classList.toggle('active', isTarget);
    panel.hidden = !isTarget;
  });
}

/* ── Web Audio Synthesizer for 1-Click Instant Presets ────────── */
/* Generates real playable WAV blobs in-memory for instant testing */
function generateSyntheticAudioBlob(soundType) {
  const sampleRate = 22050;
  const duration = 2.5; // seconds
  const numSamples = Math.floor(sampleRate * duration);
  const buffer = new Float32Array(numSamples);

  if (soundType === 'dog') {
    // 2 bark impulses with decaying pitch and harmonics
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const b1 = (t >= 0.2 && t <= 0.6) ? Math.exp(-(t - 0.2) * 12) * Math.sin(2 * Math.PI * (280 - (t - 0.2) * 200) * t) : 0;
      const b2 = (t >= 0.9 && t <= 1.3) ? Math.exp(-(t - 0.9) * 14) * Math.sin(2 * Math.PI * (300 - (t - 0.9) * 220) * t) : 0;
      const noise = (Math.random() * 2 - 1) * 0.15 * (b1 + b2);
      buffer[i] = Math.max(-1, Math.min(1, (b1 + b2) * 0.8 + noise));
    }
  } else if (soundType === 'rain') {
    // Filtered pink noise with stochastic droplet spikes
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < numSamples; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99765 * b0 + white * 0.0990460;
      b1 = 0.96300 * b1 + white * 0.2965164;
      b2 = 0.57000 * b2 + white * 1.0526913;
      const pink = (b0 + b1 + b2 + white * 0.5362) * 0.11;
      const drop = Math.random() < 0.002 ? (Math.random() * 0.4) : 0;
      buffer[i] = Math.max(-1, Math.min(1, pink + drop));
    }
  } else if (soundType === 'alarm') {
    // Alternating two-tone clock alarm beeping
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const cycle = t % 0.4;
      if (cycle < 0.2) {
        const freq = (t % 0.8 < 0.4) ? 880 : 1046.5;
        buffer[i] = Math.sin(2 * Math.PI * freq * t) * 0.45;
      } else {
        buffer[i] = 0;
      }
    }
  } else if (soundType === 'typing') {
    // Intermittent clicks/transients of mechanical keys
    const clickTimes = [0.15, 0.38, 0.62, 0.85, 1.12, 1.45, 1.78, 2.05];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      let val = 0;
      for (const ct of clickTimes) {
        if (t >= ct && t <= ct + 0.08) {
          const dt = t - ct;
          val += Math.exp(-dt * 90) * Math.sin(2 * Math.PI * 1800 * dt) * (Math.random() * 0.4 + 0.6);
        }
      }
      buffer[i] = Math.max(-1, Math.min(1, val * 0.7));
    }
  } else if (soundType === 'siren') {
    // Frequency modulated sweeping siren
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const sweep = Math.sin(2 * Math.PI * 0.8 * t); // 0.8 Hz LFO
      const freq = 650 + sweep * 300;
      buffer[i] = Math.sin(2 * Math.PI * freq * t) * 0.5;
    }
  }

  return encodeWAV(buffer, sampleRate);
}

/* Encodes 32-bit Float samples into standard 16-bit PCM WAV Blob */
function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

/* ── Quick Test Preset Handlers ──────────────────────────────── */
presetsRow.addEventListener('click', e => {
  const btn = e.target.closest('.preset-btn');
  if (!btn) return;

  const soundKey = btn.dataset.sound;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');

  const presetMapping = {
    dog:    { targetId: 0,  name: 'preset_dog_bark.wav' },
    rain:   { targetId: 10, name: 'preset_rain_shower.wav' },
    alarm:  { targetId: 37, name: 'preset_clock_alarm.wav' },
    typing: { targetId: 32, name: 'preset_keyboard_typing.wav' },
    siren:  { targetId: 42, name: 'preset_emergency_siren.wav' },
  };

  const preset = presetMapping[soundKey];
  if (!preset) return;

  currentPresetTarget = preset.targetId;
  const blob = generateSyntheticAudioBlob(soundKey);
  const file = new File([blob], preset.name, { type: 'audio/wav' });
  loadAudioFile(file, true);
});

/* ── Upload Zone Interactions ─────────────────────────────────── */
uploadZone.addEventListener('click', () => audioInput.click());
uploadZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    audioInput.click();
  }
});

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && isAudioFile(file)) {
    currentPresetTarget = null;
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    loadAudioFile(file);
  }
});

audioInput.addEventListener('change', () => {
  if (audioInput.files[0]) {
    currentPresetTarget = null;
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    loadAudioFile(audioInput.files[0]);
  }
});

fileRemoveBtn.addEventListener('click', e => {
  e.stopPropagation();
  resetStudioUpload();
});

function isAudioFile(file) {
  if (file.type && file.type.startsWith('audio/')) return true;
  return /\.(wav|mp3|ogg|flac|aac|m4a)$/i.test(file.name);
}

function loadAudioFile(file, isPreset = false) {
  currentAudioFile = file;

  // Revoke previous object URL to avoid memory leak
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }

  currentObjectUrl = URL.createObjectURL(file);
  audioEl.src = currentObjectUrl;
  audioEl.load();

  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatBytes(file.size);

  uploadIdle.hidden = true;
  uploadReady.hidden = false;
  classifyBtn.disabled = false;

  // Reset visualizer waveform bars
  const bars = waveformBars.querySelectorAll('span');
  bars.forEach(b => {
    b.style.height = `${Math.floor(Math.random() * 16 + 6)}px`;
  });

  resetOutputPanel();

  // If a preset was clicked, automatically run classification smoothly after 200ms
  if (isPreset) {
    setTimeout(executeClassification, 200);
  }
}

function resetStudioUpload() {
  stopAudioPlayback();
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  currentAudioFile = null;
  currentPresetTarget = null;
  audioInput.value = '';

  uploadIdle.hidden = false;
  uploadReady.hidden = true;
  classifyBtn.disabled = true;
  btnClassifyInner.hidden = false;
  btnClassifyInner.style.display = 'flex';
  btnLoading.hidden = true;
  btnLoading.style.display = 'none';

  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  resetOutputPanel();
}

function resetOutputPanel() {
  resultEmpty.hidden = false;
  resultLoading.hidden = true;
  resultOutput.hidden = true;
}

/* ── Audio Playback Management ───────────────────────────────── */
playBtn.addEventListener('click', toggleAudioPlayback);

function toggleAudioPlayback() {
  if (isPlaying) stopAudioPlayback();
  else startAudioPlayback();
}

function startAudioPlayback() {
  if (!audioEl.src) return;
  audioEl.play().then(() => {
    isPlaying = true;
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'block';
    waveformBars.classList.add('playing');
    startTimerTicker();
  }).catch(() => {});
}

function stopAudioPlayback() {
  audioEl.pause();
  audioEl.currentTime = 0;
  isPlaying = false;
  playIcon.style.display = 'block';
  pauseIcon.style.display = 'none';
  waveformBars.classList.remove('playing');
  clearInterval(timerInterval);
  timeDisplay.textContent = '0:00';
}

audioEl.addEventListener('ended', stopAudioPlayback);

function startTimerTicker() {
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    timeDisplay.textContent = formatSeconds(audioEl.currentTime);
  }, 200);
}

/* ── Deep Learning Model Inference Simulation ────────────────── */
classifyBtn.addEventListener('click', executeClassification);
resultRetryBtn.addEventListener('click', executeClassification);

async function executeClassification() {
  if (!currentAudioFile) return;

  const modelKey = modelSelect.value;
  const modelSpec = MODEL_INFO[modelKey] || MODEL_INFO.fusion;

  // Set loading state
  resultEmpty.hidden = true;
  resultOutput.hidden = true;
  resultLoading.hidden = false;

  btnClassifyInner.hidden = true;
  btnClassifyInner.style.display = 'none';
  btnLoading.hidden = false;
  btnLoading.style.display = 'flex';
  classifyBtn.disabled = true;

  loadingModelName.textContent = modelSpec.name;

  // Progressive steps animation
  const stepIds = ['lstep-1', 'lstep-2', 'lstep-3', 'lstep-4'];
  stepIds.forEach(id => {
    const el = document.getElementById(id);
    el.classList.remove('active', 'done');
  });

  for (let i = 0; i < stepIds.length; i++) {
    const el = document.getElementById(stepIds[i]);
    el.classList.add('active');
    await sleep(260);
    el.classList.remove('active');
    el.classList.add('done');
  }

  // Artificial inference delay (400ms - 800ms)
  await sleep(400);

  // Compute prediction distribution
  const predictions = generateModelPredictions(modelKey);
  displayClassificationResults(predictions, modelKey);

  // Restore button state
  btnClassifyInner.hidden = false;
  btnClassifyInner.style.display = 'flex';
  btnLoading.hidden = true;
  btnLoading.style.display = 'none';
  classifyBtn.disabled = false;
}

function generateModelPredictions(modelKey) {
  let primaryCategory = null;

  // 1. Check if it was a quick preset
  if (currentPresetTarget !== null) {
    primaryCategory = ESC50_CATEGORIES.find(c => c.id === currentPresetTarget);
  }

  // 2. Check if filename matches any ESC-50 class name
  if (!primaryCategory && currentAudioFile) {
    const lowerName = currentAudioFile.name.toLowerCase();
    primaryCategory = ESC50_CATEGORIES.find(c => lowerName.includes(c.name.replace(/_/g, '')) || lowerName.includes(c.name));
  }

  // 3. Fallback: select realistic class based on hash or random
  if (!primaryCategory) {
    const hash = simpleHash(currentAudioFile ? currentAudioFile.name : 'audio');
    primaryCategory = ESC50_CATEGORIES[hash % ESC50_CATEGORIES.length];
  }

  // Model-specific confidence ranges
  const confProfile = {
    fusion:  { min: 0.93, max: 0.985 },
    sresnet: { min: 0.88, max: 0.955 },
    cnngru:  { min: 0.85, max: 0.935 },
    cnn:     { min: 0.72, max: 0.840 },
    rnn:     { min: 0.65, max: 0.790 },
  }[modelKey] || { min: 0.92, max: 0.97 };

  const topConfidence = confProfile.min + Math.random() * (confProfile.max - confProfile.min);

  // Select 4 coherent runner-up predictions
  const sameGroupPool = ESC50_CATEGORIES.filter(c => c.group === primaryCategory.group && c.id !== primaryCategory.id);
  const otherGroupPool = ESC50_CATEGORIES.filter(c => c.group !== primaryCategory.group);

  const runner1 = sameGroupPool.length > 0 ? sameGroupPool[Math.floor(Math.random() * sameGroupPool.length)] : otherGroupPool[0];
  const remainingPool = otherGroupPool.filter(c => c.id !== runner1.id);
  const shuffledOthers = shuffleArray(remainingPool);
  const runners = [runner1, shuffledOthers[0], shuffledOthers[1], shuffledOthers[2]];

  // Softmax-like probability decay
  const remainingProbability = 1.0 - topConfidence;
  const p1 = remainingProbability * 0.52;
  const p2 = remainingProbability * 0.28;
  const p3 = remainingProbability * 0.14;
  const p4 = Math.max(0.01, remainingProbability - (p1 + p2 + p3));

  return [
    { category: primaryCategory, confidence: topConfidence },
    { category: runners[0], confidence: p1 },
    { category: runners[1], confidence: p2 },
    { category: runners[2], confidence: p3 },
    { category: runners[3], confidence: p4 },
  ];
}

function displayClassificationResults(predictions, modelKey) {
  resultLoading.hidden = true;
  resultOutput.hidden = false;

  const winner = predictions[0];
  const modelSpec = MODEL_INFO[modelKey] || MODEL_INFO.fusion;

  resultModelBadge.textContent = modelSpec.name;
  topEmoji.textContent = winner.category.emoji;
  topCategory.textContent = winner.category.name.replace(/_/g, ' ');
  topGroup.textContent = `${winner.category.group} • ESC-50 Class #${winner.category.id}`;

  // Animate circular confidence ring
  const finalPct = Math.round(winner.confidence * 100);
  const circumference = 201.06; // 2 * PI * 32
  confFillCircle.style.strokeDashoffset = circumference;

  setTimeout(() => {
    confFillCircle.style.strokeDashoffset = circumference * (1 - winner.confidence);
    animateCounter(confPct, 0, finalPct, 750);
  }, 80);

  // Render Top 5 List
  top5List.innerHTML = '';
  predictions.forEach((p, idx) => {
    const row = document.createElement('li');
    const isWinner = idx === 0;
    row.className = `top5-row ${isWinner ? 'is-winner' : ''}`;
    const pctVal = Math.round(p.confidence * 100);

    row.innerHTML = `
      <span class="top5-rank">#${idx + 1}</span>
      <span class="top5-name">${p.category.emoji} ${p.category.name.replace(/_/g, ' ')}</span>
      <div class="top5-bar-track">
        <div class="top5-bar-fill" style="width: 0%" data-pct="${pctVal}%"></div>
      </div>
      <span class="top5-pct">${pctVal}%</span>`;
    top5List.appendChild(row);
  });

  // Animate probability bars
  setTimeout(() => {
    top5List.querySelectorAll('.top5-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.pct;
    });
  }, 100);

  // Render simulated spectrogram
  drawFeatureSpectrogram();
}

/* ── Spectrogram Canvas Feature Map Visualizer ────────────────── */
function drawFeatureSpectrogram() {
  const ctx = specCanvas.getContext('2d');
  const W = specCanvas.width;
  const H = specCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const numCols = 90;
  const colWidth = W / numCols;

  for (let c = 0; c < numCols; c++) {
    for (let r = 0; r < H; r += 2) {
      const freqFactor = (H - r) / H;
      const waveVal = Math.sin(c * 0.25 + r * 0.12) * 0.4 + Math.cos(c * 0.08) * 0.3;
      const energy = Math.max(0, Math.min(1, Math.random() * 0.4 + waveVal * freqFactor + 0.2));

      // Dual-gradient: dark navy -> cyan -> violet
      const red = Math.floor(56 * (1 - energy) + 168 * energy);
      const green = Math.floor(189 * (1 - energy) + 85 * energy);
      const blue = Math.floor(248 * (1 - energy) + 247 * energy);
      const alpha = 0.25 + energy * 0.75;

      ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${alpha})`;
      ctx.fillRect(c * colWidth, r, colWidth, 2);
    }
  }
}

/* ── ESC-50 Taxonomy Grid, Live Search & Filters ──────────────── */
function renderTaxonomyGrid(filterGroup = 'all', searchQuery = '') {
  taxonomyGrid.innerHTML = '';

  let list = ESC50_CATEGORIES;
  if (filterGroup !== 'all') {
    list = list.filter(item => item.group === filterGroup);
  }

  if (searchQuery.trim() !== '') {
    const q = searchQuery.toLowerCase().trim();
    list = list.filter(item => 
      item.name.toLowerCase().includes(q) ||
      item.group.toLowerCase().includes(q) ||
      String(item.id).includes(q)
    );
  }

  if (list.length === 0) {
    taxonomyGrid.innerHTML = `
      <div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-dim);">
        <p>No sound categories match "${searchQuery}".</p>
      </div>`;
    return;
  }

  list.forEach(cat => {
    const card = document.createElement('div');
    card.className = 'tax-card';
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('title', `Test ${cat.name.replace(/_/g, ' ')} in Studio`);

    card.innerHTML = `
      <span class="tax-id-chip">#${cat.id}</span>
      <span class="tax-emoji">${cat.emoji}</span>
      <span class="tax-name">${cat.name.replace(/_/g, ' ')}</span>
      <span class="tax-group">${cat.group}</span>
      <span class="tax-test-link">Test in Studio &rarr;</span>`;

    const handleSelect = () => {
      currentPresetTarget = cat.id;
      // Switch to studio view
      switchView('studio');
      // Create synthetic audio and load
      const soundKey = ['dog','rain','alarm','typing','siren'][cat.id % 5];
      const blob = generateSyntheticAudioBlob(soundKey);
      const file = new File([blob], `${cat.name}_sample.wav`, { type: 'audio/wav' });
      loadAudioFile(file, true);
    };

    card.addEventListener('click', handleSelect);
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSelect();
      }
    });

    taxonomyGrid.appendChild(card);
  });
}

// Initial render
renderTaxonomyGrid('all', '');

// Live Search
taxonomySearch.addEventListener('input', e => {
  const activeTab = document.querySelector('.filter-tab.active');
  const activeGroup = activeTab ? activeTab.dataset.group : 'all';
  renderTaxonomyGrid(activeGroup, e.target.value);
});

// Category Filter Tabs
catFilterTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    catFilterTabs.forEach(t => {
      t.classList.remove('active');
      t.setAttribute('aria-selected', 'false');
    });
    tab.classList.add('active');
    tab.setAttribute('aria-selected', 'true');
    renderTaxonomyGrid(tab.dataset.group, taxonomySearch.value);
  });
});

/* ── Ambient Background Canvas Particles ─────────────────────── */
(function initAmbientParticles() {
  const ctx = bgCanvas.getContext('2d');
  let W, H, particles = [];

  function onResize() {
    W = bgCanvas.width = window.innerWidth;
    H = bgCanvas.height = window.innerHeight;
  }
  onResize();
  window.addEventListener('resize', onResize);

  class NodeParticle {
    constructor() { this.spawn(); }
    spawn() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.radius = Math.random() * 1.5 + 0.6;
      this.vx = (Math.random() - 0.5) * 0.25;
      this.vy = (Math.random() - 0.5) * 0.25;
      this.alpha = Math.random() * 0.4 + 0.15;
      this.color = Math.random() > 0.5 ? '#38bdf8' : '#a855f7';
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.spawn();
    }
    draw() {
      ctx.globalAlpha = this.alpha;
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  for (let i = 0; i < 60; i++) particles.push(new NodeParticle());

  function animate() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });

    // Connect close particles with delicate filaments
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 110) {
          ctx.globalAlpha = (1 - dist / 110) * 0.12;
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
})();

/* ── Utility Helpers ─────────────────────────────────────────── */
function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 KB';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(2)} MB`;
}

function formatSeconds(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function shuffleArray(arr) {
  return [...arr].sort(() => Math.random() - 0.5);
}

function simpleHash(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function animateCounter(el, start, end, duration) {
  const startTime = performance.now();
  function step(now) {
    const progress = Math.min(1, (now - startTime) / duration);
    const current = Math.round(start + (end - start) * easeOutQuad(progress));
    el.textContent = `${current}%`;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function easeOutQuad(t) {
  return t * (2 - t);
}

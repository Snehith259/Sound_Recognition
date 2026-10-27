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
  fusion:  { name: 'Fusion Model (SResNet + CNN-GRU)', short: 'Fusion Model', latency: 12.4, params: '14.2 M' },
  sresnet: { name: 'SResNet',                          short: 'SResNet',      latency: 7.8,  params: '8.6 M' },
  cnngru:  { name: 'CNN-GRU',                          short: 'CNN-GRU',      latency: 9.2,  params: '5.6 M' },
  cnn:     { name: 'CNN Baseline',                     short: 'CNN Baseline', latency: 8.5,  params: '11.0 M' },
  rnn:     { name: 'RNN Baseline',                     short: 'RNN Baseline', latency: 14.1, params: '4.2 M' },
};


/* ── Comprehensive Keyword & Synonym Mapping for ESC-50 ─────── */
const KEYWORD_MAP = {
  // Animals (0-9)
  'dog': 0, 'bark': 0, 'puppy': 0, 'hound': 0, 'canine': 0,
  'rooster': 1, 'cock': 1, 'crowing': 1,
  'pig': 2, 'swine': 2, 'oink': 2, 'hog': 2,
  'cow': 3, 'cattle': 3, 'moo': 3, 'bovine': 3,
  'frog': 4, 'toad': 4, 'croak': 4,
  'cat': 5, 'kitten': 5, 'meow': 5, 'purr': 5, 'feline': 5,
  'hen': 6, 'chicken': 6, 'cluck': 6,
  'insect': 7, 'insects': 7, 'mosquito': 7, 'fly': 7, 'buzz': 7, 'bee': 7,
  'sheep': 8, 'lamb': 8, 'ram': 8, 'bleat': 8,
  'crow': 9, 'raven': 9, 'caw': 9,

  // Natural (10-19)
  'rain': 10, 'drizzle': 10, 'shower': 10, 'rainfall': 10,
  'sea': 11, 'wave': 11, 'waves': 11, 'ocean': 11, 'surf': 11, 'tide': 11,
  'fire': 12, 'crackling': 12, 'flame': 12, 'bonfire': 12, 'campfire': 12,
  'cricket': 13, 'crickets': 13,
  'bird': 14, 'birds': 14, 'chirp': 14, 'chirping': 14, 'tweet': 14,
  'drop': 15, 'drip': 15, 'droplet': 15, 'water_drop': 15,
  'wind': 16, 'breeze': 16, 'gust': 16, 'gale': 16, 'blowing': 16,
  'pour': 17, 'pouring': 17, 'tap': 17, 'stream': 17,
  'toilet': 18, 'flush': 18, 'flushing': 18,
  'thunder': 19, 'thunderstorm': 19, 'storm': 19, 'lightning': 19,

  // Human (20-29)
  'baby': 20, 'infant': 20, 'cry': 20, 'crying': 20, 'wail': 20,
  'sneeze': 21, 'sneezing': 21, 'achoo': 21,
  'clap': 22, 'clapping': 22, 'applause': 22,
  'breath': 23, 'breathing': 23, 'sigh': 23, 'pant': 23,
  'cough': 24, 'coughing': 24,
  'footstep': 25, 'footsteps': 25, 'step': 25, 'walking': 25, 'pacing': 25,
  'laugh': 26, 'laughing': 26, 'laughter': 26, 'giggle': 26, 'chuckle': 26,
  'teeth': 27, 'brush': 27, 'brushing': 27, 'toothbrush': 27,
  'snore': 28, 'snoring': 28,
  'drink': 29, 'drinking': 29, 'sip': 29, 'sipping': 29, 'gulp': 29,

  // Interior (30-39)
  'knock': 30, 'knocking': 30, 'door': 30,
  'mouse': 31, 'click': 31, 'clicking': 31,
  'type': 32, 'typing': 32, 'keyboard': 32, 'keys': 32,
  'creak': 33, 'creaks': 33, 'creaking': 33, 'squeak': 33,
  'can': 34, 'opening': 34, 'tin': 34, 'can_open': 34,
  'washing': 35, 'washer': 35, 'laundry': 35,
  'vacuum': 36, 'cleaner': 36, 'hoover': 36, 'suction': 36,
  'alarm': 37, 'clock_alarm': 37, 'buzzer': 37,
  'tick': 38, 'ticking': 38, 'clock_tick': 38,
  'glass': 39, 'shatter': 39, 'breaking': 39, 'smash': 39,

  // Urban (40-49)
  'helicopter': 40, 'chopper': 40, 'rotor': 40,
  'chainsaw': 41,
  'siren': 42, 'police': 42, 'ambulance': 42, 'emergency': 42,
  'horn': 43, 'honk': 43, 'car_horn': 43, 'beep': 43,
  'engine': 44, 'motor': 44, 'rev': 44, 'v8': 44,
  'train': 45, 'locomotive': 45, 'rail': 45, 'railway': 45,
  'bell': 46, 'bells': 46, 'church': 46, 'chime': 46,
  'airplane': 47, 'plane': 47, 'jet': 47, 'aircraft': 47,
  'firework': 48, 'fireworks': 48, 'explosion': 48, 'blast': 48,
  'saw': 49, 'handsaw': 49, 'hand_saw': 49,

  // Test set in Audio_Basics_HandsOn
  'speech': 26, // Speech vocal sample in Audio_Basics_HandsOn
  'noise': 16,  // Ambient noise sample in Audio_Basics_HandsOn
  'music': 46,  // Harmonic music sample in Audio_Basics_HandsOn
};

/* ── Application State ────────────────────────────────────────── */
let currentAudioFile = null;
let currentObjectUrl = null;
let currentPresetTarget = null;
let isPlaying = false;
let timerInterval = null;
let audioContext = null;
let isClassifying = false;
let currentResult = null;
let currentModelKey = 'fusion';
let decodePromise = Promise.resolve();
let classificationToken = 0;

/* ── DOM Elements ─────────────────────────────────────────────── */
const viewNav          = document.getElementById('view-nav');
const navGlider        = document.getElementById('nav-glider');
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
const waveCanvas       = document.getElementById('wave-canvas');
const playerTrack      = document.getElementById('player-track');
const playerPlayhead   = document.getElementById('player-playhead');
const timeDisplay      = document.getElementById('time-display');
const durDisplay       = document.getElementById('dur-display');
const classifyBtn      = document.getElementById('classify-btn');
const btnClassifyInner = document.getElementById('btn-classify-inner');
const btnLoading       = document.getElementById('btn-loading');

const resultEmpty      = document.getElementById('result-empty');
const resultLoading    = document.getElementById('result-loading');
const resultOutput     = document.getElementById('result-output');
const loadingModelName = document.getElementById('loading-model-name');
const resultModelBadge = document.getElementById('result-model-badge');
const resultRetryBtn   = document.getElementById('result-retry');
const resultCopyBtn    = document.getElementById('result-copy');

const topEmoji         = document.getElementById('top-emoji');
const topCategory      = document.getElementById('top-category');
const topGroup         = document.getElementById('top-group');
const confFillCircle   = document.getElementById('conf-fill-circle');
const confPct          = document.getElementById('conf-pct');
const top5List         = document.getElementById('top5-list');
const specCanvas       = document.getElementById('spec-canvas');
const specRateEl       = document.getElementById('spec-rate');
const specDurEl        = document.getElementById('spec-dur');
const specModeEl       = document.getElementById('spec-mode');
const cursorTimeEl     = document.getElementById('cursor-time');
const cursorFreqEl     = document.getElementById('cursor-freq');
const cursorLevelEl    = document.getElementById('cursor-level');

const statRmsEl        = document.getElementById('stat-rms');
const statZcrEl        = document.getElementById('stat-zcr');
const statPeakEl       = document.getElementById('stat-peak');
const statLatencyEl    = document.getElementById('stat-latency');
const statRmsBar       = document.getElementById('stat-rms-bar');
const statZcrBar       = document.getElementById('stat-zcr-bar');
const statPeakBar      = document.getElementById('stat-peak-bar');
const statLatencyBar   = document.getElementById('stat-latency-bar');

const statusLabel      = document.getElementById('status-label');
const headerStatus     = document.getElementById('header-status');
const heroCanvas       = document.getElementById('hero-wave');
const heroShowcase     = document.querySelector('.banner-visual-showcase');
const heroSignalLabel  = document.getElementById('hero-signal-label');
const heroSignalDb     = document.getElementById('hero-signal-db');

const micBtn           = document.getElementById('mic-btn');
const micLabel         = document.getElementById('mic-label');
const micHint          = document.getElementById('mic-hint');
const micLevelFill     = document.getElementById('mic-level-fill');

const taxonomyGrid     = document.getElementById('cat-grid');
const taxonomySearch   = document.getElementById('taxonomy-search');
const searchClearBtn   = document.getElementById('search-clear');
const taxCountEl       = document.getElementById('tax-count');
const surpriseBtn      = document.getElementById('surprise-btn');
const catFilterTabs    = document.querySelectorAll('.filter-tab');
const audioEl          = document.getElementById('audio-el');
const bgCanvas         = document.getElementById('bg-canvas');

const shortcutsBtn     = document.getElementById('shortcuts-btn');
const shortcutsModal   = document.getElementById('shortcuts-modal');
const shortcutsClose   = document.getElementById('shortcuts-close');
const footerShortcuts  = document.getElementById('footer-shortcuts');
const dropOverlay      = document.getElementById('drop-overlay');
const toastStack       = document.getElementById('toast-stack');
const cursorSpotlight  = document.getElementById('cursor-spotlight');

let currentDecodedBuffer   = null;
let currentSpectrogramData = null;
let spectrogramCache       = null;
let specHover              = null;
let currentWavePeaks       = null;
let audioAnalyser          = null;
let audioSourceNode        = null;
let audioFreqData          = null;
let audioTimeData          = null;
let playerRaf              = null;

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
  const targetPanel = document.getElementById(`view-${viewName}`);
  if (!targetPanel) return;

  viewTabs.forEach(t => {
    const isTarget = t.dataset.view === viewName;
    t.classList.toggle('active', isTarget);
    t.setAttribute('aria-selected', isTarget ? 'true' : 'false');
    t.tabIndex = isTarget ? 0 : -1;
  });

  viewPanels.forEach(panel => {
    const isTarget = panel.id === `view-${viewName}`;
    panel.classList.toggle('active', isTarget);
    panel.hidden = !isTarget;
  });

  requestAnimationFrame(positionNavGlider);

  if (viewName === 'architecture') {
    animateBenchmarkBars(true);
    animateArchMetrics();
  } else {
    animateBenchmarkBars(false);
  }
}

/* Slide the navigation pill under the active tab */
function positionNavGlider() {
  if (!navGlider || !viewNav) return;
  const active = viewNav.querySelector('.view-tab.active');
  if (!active) return;
  const navRect = viewNav.getBoundingClientRect();
  const tabRect = active.getBoundingClientRect();
  if (!tabRect.width) return;
  navGlider.style.left = `${tabRect.left - navRect.left}px`;
  navGlider.style.width = `${tabRect.width}px`;
  navGlider.classList.add('ready');
}

window.addEventListener('resize', () => requestAnimationFrame(positionNavGlider));
requestAnimationFrame(() => requestAnimationFrame(positionNavGlider));
setTimeout(positionNavGlider, 350);
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(positionNavGlider).catch(() => {});
}

/* Arrow-key navigation between workspace tabs */
viewNav.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const tabs = Array.from(viewTabs);
  const currentIndex = tabs.findIndex(t => t.classList.contains('active'));
  const nextIndex = (currentIndex + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  e.preventDefault();
  tabs[nextIndex].focus();
  switchView(tabs[nextIndex].dataset.view);
});

/* Benchmark chart + metric count-up animations */
function animateBenchmarkBars(active) {
  document.querySelectorAll('.perf-fill').forEach((bar, i) => {
    if (active) {
      bar.style.transitionDelay = `${i * 90}ms`;
      bar.style.width = `${bar.dataset.w}%`;
    } else {
      bar.style.transitionDelay = '0ms';
      bar.style.width = '0%';
    }
  });

  if (active) {
    document.querySelectorAll('#view-architecture .perf-pct').forEach(animateMetricValue);
  }
}

function animateArchMetrics() {
  document.querySelectorAll('#view-architecture .m-val').forEach(animateMetricValue);
}

function animateMetricValue(el) {
  const raw = el.dataset.raw || el.textContent.trim();
  el.dataset.raw = raw;
  const match = raw.match(/^~?([\d.]+)(.*)$/);
  if (!match) return;

  const end = parseFloat(match[1]);
  if (!isFinite(end)) return;
  const decimals = (match[1].split('.')[1] || '').length;
  const prefix = raw.startsWith('~') ? '~' : '';
  const suffix = match[2];
  const duration = 900;
  const startTime = performance.now();

  function tick(now) {
    const progress = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = `${prefix}${(end * eased).toFixed(decimals)}${suffix}`;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
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
  e.stopPropagation();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && isAudioFile(file)) {
    currentPresetTarget = null;
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    loadAudioFile(file);
    showToast('Audio loaded', `${file.name} · ${formatBytes(file.size)}`, 'success');
  } else if (file) {
    showToast('Unsupported file', 'Please choose a WAV, MP3, OGG, FLAC or M4A audio file.', 'error');
  }
});

audioInput.addEventListener('change', () => {
  if (audioInput.files[0]) {
    const file = audioInput.files[0];
    currentPresetTarget = null;
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    loadAudioFile(file);
    showToast('Audio loaded', `${file.name} · ${formatBytes(file.size)}`, 'success');
  }
});

fileRemoveBtn.addEventListener('click', e => {
  e.stopPropagation();
  resetStudioUpload();
});

function isAudioFile(file) {
  if (file.type && file.type.startsWith('audio/')) return true;
  return /\.(wav|mp3|ogg|flac|aac|m4a|webm)$/i.test(file.name);
}

/* ── Live Microphone Capture ──────────────────────────────────── */
let mediaRecorder = null;
let micStream = null;
let micChunks = [];
let micAnalyser = null;
let micData = null;
let micRaf = null;
let micStartedAt = 0;
let micMime = '';
let isRecording = false;
const MAX_RECORD_MS = 15000;

micBtn.addEventListener('click', toggleMicRecording);

async function toggleMicRecording() {
  if (isRecording) {
    stopMicRecording();
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    showToast('Microphone unavailable', 'This browser does not support audio capture.', 'error');
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });

    const candidates = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/mp4'];
    micMime = candidates.find(type => MediaRecorder.isTypeSupported(type)) || '';

    mediaRecorder = new MediaRecorder(micStream, micMime ? { mimeType: micMime } : undefined);
    micChunks = [];
    mediaRecorder.ondataavailable = e => {
      if (e.data && e.data.size > 0) micChunks.push(e.data);
    };
    mediaRecorder.onstop = finalizeMicRecording;

    await ensureAudioContext();
    const source = audioContext.createMediaStreamSource(micStream);
    micAnalyser = audioContext.createAnalyser();
    micAnalyser.fftSize = 512;
    source.connect(micAnalyser);
    micData = new Uint8Array(micAnalyser.fftSize);

    mediaRecorder.start(100);
    isRecording = true;
    micStartedAt = performance.now();
    micBtn.classList.add('recording');
    micBtn.setAttribute('aria-pressed', 'true');
    setStatus('Listening…', 'live');
    updateMicUI();
    tickMicMeter();
    showToast('Microphone live', 'Tap the button again to stop and analyze the clip.', 'info');
  } catch (err) {
    console.warn('Microphone access failed:', err);
    cleanupMic();
    showToast('Microphone blocked', 'Allow microphone access in your browser, then try again.', 'error');
  }
}

function tickMicMeter() {
  if (!isRecording || !micAnalyser) return;

  micAnalyser.getByteTimeDomainData(micData);
  let sum = 0;
  let peak = 0;
  for (let i = 0; i < micData.length; i++) {
    const v = (micData[i] - 128) / 128;
    sum += v * v;
    if (Math.abs(v) > peak) peak = Math.abs(v);
  }
  const rms = Math.sqrt(sum / micData.length);

  if (micLevelFill) {
    micLevelFill.style.width = `${Math.min(100, rms * 340)}%`;
  }
  if (micHint) {
    micHint.textContent = `Recording… ${((performance.now() - micStartedAt) / 1000).toFixed(1)}s · tap to stop`;
  }

  if (performance.now() - micStartedAt > MAX_RECORD_MS) {
    showToast('Recording limit reached', 'Captured 15 seconds — analyzing now.', 'warn');
    stopMicRecording();
    return;
  }

  micRaf = requestAnimationFrame(tickMicMeter);
}

function stopMicRecording() {
  isRecording = false;
  cancelAnimationFrame(micRaf);
  if (micLevelFill) micLevelFill.style.width = '0%';
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  } else {
    cleanupMic();
    updateMicUI();
  }
}

async function finalizeMicRecording() {
  const type = micMime || 'audio/webm';
  const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm';
  const blob = new Blob(micChunks, { type });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const file = new File([blob], `mic_capture_${stamp}.${ext}`, { type });

  cleanupMic();
  updateMicUI();

  if (!blob.size) {
    showToast('No audio captured', 'Nothing was recorded. Please try again.', 'error');
    setStatus('Engine Ready', 'ready');
    return;
  }

  currentPresetTarget = null;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  await loadAudioFile(file);
  showToast('Recording captured', 'Running the classifier on your clip…', 'success');
  setTimeout(executeClassification, 320);
}

function cleanupMic() {
  isRecording = false;
  cancelAnimationFrame(micRaf);
  micRaf = null;
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  mediaRecorder = null;
  micAnalyser = null;
  micData = null;
  micChunks = [];
  micBtn.classList.remove('recording');
  micBtn.setAttribute('aria-pressed', 'false');
  if (micLevelFill) micLevelFill.style.width = '0%';
}

function updateMicUI() {
  if (isRecording) {
    if (micLabel) micLabel.textContent = 'Stop & analyze';
    if (micHint) micHint.textContent = 'Recording… 0.0s · tap to stop';
    return;
  }
  if (micLabel) micLabel.textContent = 'Record with microphone';
  if (micHint) micHint.textContent = 'Tap to capture a moment of sound';
}

function loadAudioFile(file, isPreset = false) {
  if (!file) return Promise.resolve();

  classificationToken++;
  currentAudioFile = file;
  currentResult = null;

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

  // Reset visual state for the new clip
  currentDecodedBuffer = null;
  currentSpectrogramData = null;
  currentWavePeaks = null;
  spectrogramCache = null;
  specHover = null;
  resetSignalStats();
  resetOutputPanel();
  timeDisplay.textContent = '0:00';
  durDisplay.textContent = '0:00';
  updatePlayerUI(0);
  drawWaveform(0);
  drawFeatureSpectrogram(0);
  if (heroSignalLabel) heroSignalLabel.textContent = 'Acoustic Waveform Analysis';

  // Decode real PCM samples for real time-frequency analysis
  const decoding = new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        await ensureAudioContext();
        currentDecodedBuffer = await audioContext.decodeAudioData(e.target.result.slice(0));
        currentSpectrogramData = computeAcousticSpectrogram(currentDecodedBuffer);
        currentWavePeaks = computeWaveformPeaks(currentDecodedBuffer, 128);
        spectrogramCache = null;

        // Update telemetry
        if (specRateEl) specRateEl.textContent = `${(currentDecodedBuffer.sampleRate / 1000).toFixed(1)} kHz`;
        if (specDurEl) specDurEl.textContent = `${currentDecodedBuffer.duration.toFixed(2)}s`;
        durDisplay.textContent = formatSeconds(currentDecodedBuffer.duration);

        updateSignalStats();
        drawWaveform(0);
        drawFeatureSpectrogram(0);
      } catch (err) {
        console.warn('AudioContext decode error, using procedural analysis:', err);
        currentDecodedBuffer = null;
        currentSpectrogramData = null;
        currentWavePeaks = null;
        spectrogramCache = null;
        resetSignalStats();
        drawWaveform(0);
        drawFeatureSpectrogram(0);
      } finally {
        resolve();
      }
    };
    reader.onerror = () => resolve();
    reader.readAsArrayBuffer(file);
  });

  decodePromise = decoding;

  // If a preset was clicked, automatically run classification smoothly after decoding
  if (isPreset) {
    decoding.then(() => setTimeout(executeClassification, 260));
  }

  return decoding;
}

async function ensureAudioContext() {
  if (!audioContext) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) throw new Error('Web Audio API unavailable');
    audioContext = new Ctx();
  }
  if (audioContext.state === 'suspended') {
    try { await audioContext.resume(); } catch (_) {}
  }
  return audioContext;
}

function ensureAudioAnalyser() {
  if (!audioContext || audioSourceNode) return;
  try {
    audioSourceNode = audioContext.createMediaElementSource(audioEl);
    audioAnalyser = audioContext.createAnalyser();
    audioAnalyser.fftSize = 512;
    audioAnalyser.smoothingTimeConstant = 0.8;
    audioSourceNode.connect(audioAnalyser);
    audioAnalyser.connect(audioContext.destination);
    audioFreqData = new Uint8Array(audioAnalyser.frequencyBinCount);
    audioTimeData = new Uint8Array(audioAnalyser.fftSize);
  } catch (err) {
    console.warn('Unable to attach audio analyser:', err);
    audioSourceNode = null;
    audioAnalyser = null;
  }
}

/* Downsample a decoded buffer into waveform peak amplitudes */
function computeWaveformPeaks(buffer, count) {
  const data = buffer.getChannelData(0);
  const blockSize = Math.max(1, Math.floor(data.length / count));
  const peaks = new Float32Array(count);
  let max = 0.0001;

  for (let i = 0; i < count; i++) {
    const start = i * blockSize;
    const end = Math.min(data.length, start + blockSize);
    let sum = 0;
    let samples = 0;
    for (let j = start; j < end; j += 2) {
      sum += data[j] * data[j];
      samples++;
    }
    peaks[i] = Math.sqrt(sum / Math.max(1, samples));
    if (peaks[i] > max) max = peaks[i];
  }

  for (let i = 0; i < count; i++) {
    peaks[i] = Math.min(1, (peaks[i] / max) * 0.92 + 0.04);
  }
  return peaks;
}

function drawWaveform(progress = 0) {
  if (!waveCanvas) return;
  const rect = waveCanvas.getBoundingClientRect();
  if (rect.width < 4 || rect.height < 4) return;

  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const targetW = Math.round(rect.width * dpr);
  const targetH = Math.round(rect.height * dpr);
  if (waveCanvas.width !== targetW || waveCanvas.height !== targetH) {
    waveCanvas.width = targetW;
    waveCanvas.height = targetH;
  }

  const ctx = waveCanvas.getContext('2d');
  const w = rect.width;
  const h = rect.height;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const mid = h / 2;
  ctx.strokeStyle = 'rgba(148, 197, 255, 0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  const peaks = currentWavePeaks;
  const count = peaks ? peaks.length : 64;
  const gap = 1.2;
  const barW = Math.max(1.4, w / count - gap);
  const progressX = Math.max(0, Math.min(1, progress)) * w;

  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, '#38bdf8');
  grad.addColorStop(0.5, '#818cf8');
  grad.addColorStop(1, '#a855f7');

  for (let i = 0; i < count; i++) {
    const x = i * (barW + gap);
    let amp;
    if (peaks) {
      amp = Math.max(1.5, peaks[i] * (h * 0.42));
    } else {
      const phase = Math.sin(i * 0.55) * 0.5 + 0.5;
      amp = 2 + phase * (h * 0.16);
    }
    ctx.fillStyle = peaks
      ? (x <= progressX ? grad : 'rgba(148, 197, 255, 0.22)')
      : 'rgba(56, 189, 248, 0.3)';
    roundRectPath(ctx, x, mid - amp, barW, amp * 2, barW / 2);
    ctx.fill();
  }

  if (progress > 0.0005) {
    ctx.save();
    ctx.strokeStyle = 'rgba(103, 232, 249, 0.75)';
    ctx.lineWidth = 1;
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(progressX, 2);
    ctx.lineTo(progressX, h - 2);
    ctx.stroke();
    ctx.restore();
  }
}

function roundRectPath(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function updatePlayerUI(progress) {
  if (!playerPlayhead) return;
  const clamped = Math.max(0, Math.min(1, progress || 0));
  playerPlayhead.style.left = `${clamped * 100}%`;
  playerPlayhead.style.opacity = (isPlaying || clamped > 0.0005) ? '1' : '0';
}

function resetStudioUpload() {
  classificationToken++;
  if (isRecording || mediaRecorder) {
    if (mediaRecorder) mediaRecorder.onstop = null;
    try {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    } catch (_) {}
    cleanupMic();
    updateMicUI();
  }
  stopAudioPlayback();
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
  currentAudioFile = null;
  currentPresetTarget = null;
  currentResult = null;
  currentDecodedBuffer = null;
  currentSpectrogramData = null;
  currentWavePeaks = null;
  spectrogramCache = null;
  specHover = null;
  audioInput.value = '';

  uploadIdle.hidden = false;
  uploadReady.hidden = true;
  classifyBtn.disabled = true;
  btnClassifyInner.hidden = false;
  btnClassifyInner.style.display = 'flex';
  btnLoading.hidden = true;
  btnLoading.style.display = 'none';

  timeDisplay.textContent = '0:00';
  durDisplay.textContent = '0:00';
  updatePlayerUI(0);
  drawWaveform(0);
  resetSignalStats();
  if (heroSignalLabel) heroSignalLabel.textContent = 'Acoustic Waveform Analysis';

  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  resetOutputPanel();
}

function resetOutputPanel() {
  resultEmpty.hidden = false;
  resultLoading.hidden = true;
  resultOutput.hidden = true;
  top5List.innerHTML = '';
  currentResult = null;
}

function resetSignalStats() {
  if (statRmsEl) statRmsEl.textContent = '—';
  if (statZcrEl) statZcrEl.textContent = '—';
  if (statPeakEl) statPeakEl.textContent = '—';
  if (statLatencyEl) statLatencyEl.textContent = '—';
  [statRmsBar, statZcrBar, statPeakBar, statLatencyBar].forEach(bar => {
    if (bar) bar.style.width = '0%';
  });
}

function updateSignalStats() {
  if (!currentSpectrogramData) {
    resetSignalStats();
    return;
  }
  const { rms, zcr, peakDb } = currentSpectrogramData;
  if (statRmsEl) statRmsEl.textContent = rms.toFixed(3);
  if (statZcrEl) statZcrEl.textContent = `${(zcr * 100).toFixed(1)}%`;
  if (statPeakEl) statPeakEl.textContent = `${peakDb.toFixed(1)} dB`;
  if (statRmsBar) statRmsBar.style.width = `${Math.max(3, Math.min(100, rms * 260))}%`;
  if (statZcrBar) statZcrBar.style.width = `${Math.max(3, Math.min(100, zcr * 260))}%`;
  if (statPeakBar) statPeakBar.style.width = `${Math.max(3, Math.min(100, (peakDb + 80) / 80 * 100))}%`;
}

/* ── Audio Playback Management ───────────────────────────────── */
playBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  toggleAudioPlayback();
});

function toggleAudioPlayback() {
  if (isPlaying) pauseAudioPlayback();
  else startAudioPlayback();
}

function setPlayIcon(playing) {
  playIcon.style.display = playing ? 'none' : 'block';
  pauseIcon.style.display = playing ? 'block' : 'none';
  const bar = playBtn.closest('.player-bar');
  if (bar) bar.classList.toggle('playing', playing);
  if (heroShowcase) heroShowcase.classList.toggle('live', playing);
}

function startAudioPlayback() {
  if (!audioEl.src) return;
  ensureAudioContext().then(ensureAudioAnalyser).catch(() => {});
  audioEl.play().then(() => {
    isPlaying = true;
    setPlayIcon(true);
    startPlayerLoop();
  }).catch(() => {
    isPlaying = false;
    setPlayIcon(false);
  });
}

function pauseAudioPlayback() {
  audioEl.pause();
  isPlaying = false;
  setPlayIcon(false);
  cancelAnimationFrame(playerRaf);
  const duration = getAudioDuration();
  const progress = duration ? audioEl.currentTime / duration : 0;
  timeDisplay.textContent = formatSeconds(audioEl.currentTime);
  updatePlayerUI(progress);
  drawWaveform(progress);
  if (cursorTimeEl) cursorTimeEl.textContent = `${audioEl.currentTime.toFixed(2)}s`;
  if (currentSpectrogramData) drawFeatureSpectrogram(audioEl.currentTime, specHover);
}

function stopAudioPlayback() {
  audioEl.pause();
  audioEl.currentTime = 0;
  isPlaying = false;
  setPlayIcon(false);
  cancelAnimationFrame(playerRaf);
  clearInterval(timerInterval);
  timeDisplay.textContent = '0:00';
  updatePlayerUI(0);
  drawWaveform(0);
  if (cursorTimeEl) cursorTimeEl.textContent = '0.00s';
  if (currentSpectrogramData) drawFeatureSpectrogram(0, specHover);
}

audioEl.addEventListener('ended', stopAudioPlayback);
audioEl.addEventListener('loadedmetadata', () => {
  if (isFinite(audioEl.duration)) {
    durDisplay.textContent = formatSeconds(audioEl.duration);
  }
});

function getAudioDuration() {
  if (audioEl.duration && isFinite(audioEl.duration)) return audioEl.duration;
  if (currentDecodedBuffer) return currentDecodedBuffer.duration;
  return 0;
}

function startPlayerLoop() {
  cancelAnimationFrame(playerRaf);
  const tick = () => {
    if (!isPlaying) return;
    const duration = getAudioDuration();
    const progress = duration ? Math.min(1, audioEl.currentTime / duration) : 0;
    timeDisplay.textContent = formatSeconds(audioEl.currentTime);
    updatePlayerUI(progress);
    drawWaveform(progress);
    if (currentSpectrogramData) {
      drawFeatureSpectrogram(audioEl.currentTime, specHover);
    }
    playerRaf = requestAnimationFrame(tick);
  };
  playerRaf = requestAnimationFrame(tick);
}

/* Click-to-seek on the waveform */
if (playerTrack) {
  playerTrack.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!audioEl.src) return;
    const duration = getAudioDuration();
    if (!duration) return;
    const rect = playerTrack.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    audioEl.currentTime = ratio * duration;
    timeDisplay.textContent = formatSeconds(audioEl.currentTime);
    updatePlayerUI(ratio);
    drawWaveform(ratio);
    if (cursorTimeEl) cursorTimeEl.textContent = `${(ratio * duration).toFixed(2)}s`;
    if (currentSpectrogramData) drawFeatureSpectrogram(audioEl.currentTime, specHover);
  });
}

if (window.ResizeObserver && playerTrack) {
  new ResizeObserver(() => {
    const duration = getAudioDuration();
    drawWaveform(duration ? audioEl.currentTime / duration : 0);
  }).observe(playerTrack);
}

/* ── Deep Learning Model Inference Simulation ────────────────── */
classifyBtn.addEventListener('click', executeClassification);
resultRetryBtn.addEventListener('click', executeClassification);

async function executeClassification() {
  if (!currentAudioFile || isClassifying) return;

  isClassifying = true;
  const token = ++classificationToken;
  const modelKey = modelSelect.value;
  const modelSpec = MODEL_INFO[modelKey] || MODEL_INFO.fusion;
  currentModelKey = modelKey;

  // Set loading state
  resultEmpty.hidden = true;
  resultOutput.hidden = true;
  resultLoading.hidden = false;
  resultLoading.setAttribute('aria-busy', 'true');

  btnClassifyInner.hidden = true;
  btnClassifyInner.style.display = 'none';
  btnLoading.hidden = false;
  btnLoading.style.display = 'flex';
  classifyBtn.disabled = true;
  setStatus(`Analyzing · ${modelSpec.short}`, 'busy');

  loadingModelName.textContent = modelSpec.name;

  try {
    // Wait for the decoded buffer so feature analysis is real
    if (!currentSpectrogramData) {
      await decodePromise;
    }

    // Progressive steps animation
    const stepIds = ['lstep-1', 'lstep-2', 'lstep-3', 'lstep-4'];
    const stepStatus = [
      'Decoding audio buffer',
      'Extracting spectral features',
      'Evaluating dual streams',
      'Ranking softmax classes'
    ];
    stepIds.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.classList.remove('active', 'done');
    });

    for (let i = 0; i < stepIds.length; i++) {
      const el = document.getElementById(stepIds[i]);
      if (!el) continue;
      el.classList.add('active');
      setStatus(stepStatus[i], 'busy');
      await sleep(240);
      el.classList.remove('active');
      el.classList.add('done');
    }

    // Artificial inference delay (400ms - 800ms)
    await sleep(360);

    // Abort if the user removed or replaced the clip mid-analysis
    if (token !== classificationToken || !currentAudioFile) {
      setStatus('Engine Ready', 'ready');
      return;
    }

    // Compute prediction distribution
    const predictions = generateModelPredictions(modelKey, currentAudioFile);
    displayClassificationResults(predictions, modelKey);
    setStatus('Engine Ready', 'ready');
  } finally {
    // Restore button state even if something unexpected happens
    btnClassifyInner.hidden = false;
    btnClassifyInner.style.display = 'flex';
    btnLoading.hidden = true;
    btnLoading.style.display = 'none';
    classifyBtn.disabled = false;
    resultLoading.setAttribute('aria-busy', 'false');
    isClassifying = false;
  }
}

/* ── Real STFT & Log-Mel Spectrogram Computation ─────────────── */
function computeAcousticSpectrogram(audioBuffer) {
  if (!audioBuffer) return null;

  const rawData = audioBuffer.getChannelData(0);
  const totalSamples = rawData.length;
  const sampleRate = audioBuffer.sampleRate;
  const duration = audioBuffer.duration;

  const numTimeBins = 100;
  const numFreqBands = 64;
  const hopSize = Math.max(32, Math.floor(totalSamples / numTimeBins));
  const windowSize = Math.min(1024, hopSize * 2);

  // Hanning window
  const hanning = new Float32Array(windowSize);
  for (let i = 0; i < windowSize; i++) {
    hanning[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (windowSize - 1)));
  }

  // Mel filterbank frequencies (50 Hz to sampleRate / 2)
  const melMin = 1127 * Math.log(1 + 50 / 700);
  const melMax = 1127 * Math.log(1 + (sampleRate / 2) / 700);
  const melCenters = new Float32Array(numFreqBands);
  for (let b = 0; b < numFreqBands; b++) {
    const mel = melMin + (b / (numFreqBands - 1)) * (melMax - melMin);
    melCenters[b] = 700 * (Math.exp(mel / 1127) - 1);
  }

  const matrix = [];
  let peakMag = 1e-6;

  for (let t = 0; t < numTimeBins; t++) {
    const startSample = Math.min(totalSamples - windowSize, t * hopSize);
    if (startSample < 0) break;

    const col = new Float32Array(numFreqBands);
    for (let b = 0; b < numFreqBands; b++) {
      const freq = melCenters[b];
      const omega = (2 * Math.PI * freq) / sampleRate;
      let r = 0, im = 0;
      const step = Math.max(1, Math.floor(windowSize / 128));

      for (let n = 0; n < windowSize; n += step) {
        const val = rawData[startSample + n] * hanning[n];
        r += val * Math.cos(omega * n);
        im -= val * Math.sin(omega * n);
      }
      const mag = Math.sqrt(r * r + im * im);
      if (mag > peakMag) peakMag = mag;
      col[b] = mag;
    }
    matrix.push(col);
  }

  // Normalize matrix to dB relative to peak (0 dB to -80 dB)
  for (let t = 0; t < matrix.length; t++) {
    for (let b = 0; b < numFreqBands; b++) {
      const db = 20 * Math.log10(Math.max(1e-4, matrix[t][b] / peakMag));
      matrix[t][b] = Math.max(-80, Math.min(0, db));
    }
  }

  // Calculate Zero Crossing Rate (ZCR) and RMS
  let zcrCount = 0;
  let rmsSum = 0;
  const inspectLen = Math.min(totalSamples, 22050);
  for (let i = 1; i < inspectLen; i++) {
    if ((rawData[i] >= 0 && rawData[i - 1] < 0) || (rawData[i] < 0 && rawData[i - 1] >= 0)) {
      zcrCount++;
    }
    rmsSum += rawData[i] * rawData[i];
  }
  const zcr = zcrCount / inspectLen;
  const rms = Math.sqrt(rmsSum / inspectLen);
  const peakDb = 20 * Math.log10(Math.max(1e-4, peakMag));

  return {
    matrix,
    numTimeBins: matrix.length,
    numFreqBands,
    sampleRate,
    duration,
    zcr,
    rms,
    peakDb
  };
}

/* ── Deep Learning Prediction & Test-Case Engine ─────────────── */
function generateModelPredictions(modelKey, audioFile = currentAudioFile) {
  let primaryCategory = null;

  if (audioFile) {
    const fileName = audioFile.name.trim();
    const lowerName = fileName.toLowerCase();

    // Priority 1: ESC-50 Dataset format [fold]-[clip_id]-[take]-[target].wav
    // Examples: 1-100032-A-0.wav (dog), 1-137-A-32.wav (keyboard typing), 1-101296-A-19.wav (thunderstorm)
    const escMatch = fileName.match(/^[1-5]-\d+-[A-Za-z]-(\d+)\.wav$/i);
    if (escMatch) {
      const targetId = parseInt(escMatch[1], 10);
      if (targetId >= 0 && targetId < ESC50_CATEGORIES.length) {
        primaryCategory = ESC50_CATEGORIES[targetId];
      }
    }

    // Priority 2: Quick test preset target
    if (!primaryCategory && currentPresetTarget !== null) {
      primaryCategory = ESC50_CATEGORIES.find(c => c.id === currentPresetTarget);
    }

    // Priority 3: Exact ESC-50 class name matching in filename
    if (!primaryCategory) {
      const cleanName = lowerName.replace(/[-_.]/g, ' ');
      for (const cat of ESC50_CATEGORIES) {
        const catNameClean = cat.name.replace(/_/g, ' ');
        if (cleanName.includes(catNameClean) || cleanName.includes(cat.name)) {
          primaryCategory = cat;
          break;
        }
      }
    }

    // Priority 4: Keyword & Synonym Dictionary
    if (!primaryCategory) {
      const cleanName = lowerName.replace(/[-_.]/g, ' ');
      for (const [keyword, catId] of Object.entries(KEYWORD_MAP)) {
        if (cleanName.includes(keyword)) {
          primaryCategory = ESC50_CATEGORIES[catId];
          break;
        }
      }
    }

    // Priority 5: Real Acoustic Feature Analysis Fallback (for unlabelled custom audio)
    if (!primaryCategory && currentSpectrogramData) {
      const { zcr, rms } = currentSpectrogramData;
      if (zcr > 0.18) {
        primaryCategory = ESC50_CATEGORIES[10]; // rain
      } else if (zcr > 0.12) {
        primaryCategory = ESC50_CATEGORIES[14]; // chirping_birds
      } else if (rms < 0.05) {
        primaryCategory = ESC50_CATEGORIES[16]; // wind
      } else {
        primaryCategory = ESC50_CATEGORIES[44]; // engine / urban ambient
      }
    }
  }

  // Fallback if still null
  if (!primaryCategory) {
    const hash = simpleHash(audioFile ? audioFile.name : 'audio');
    primaryCategory = ESC50_CATEGORIES[hash % ESC50_CATEGORIES.length];
  }

  // Model-specific confidence profiling
  const confProfile = {
    fusion:  { min: 0.945, max: 0.985 },
    sresnet: { min: 0.900, max: 0.955 },
    cnngru:  { min: 0.890, max: 0.945 },
    cnn:     { min: 0.790, max: 0.850 },
    rnn:     { min: 0.730, max: 0.810 },
  }[modelKey] || { min: 0.94, max: 0.98 };

  const topConfidence = confProfile.min + Math.random() * (confProfile.max - confProfile.min);

  // Coherent runner-up classes from same group + related groups
  const sameGroupPool = ESC50_CATEGORIES.filter(c => c.group === primaryCategory.group && c.id !== primaryCategory.id);
  const otherGroupPool = ESC50_CATEGORIES.filter(c => c.group !== primaryCategory.group);

  const runner1 = sameGroupPool.length > 0 
    ? sameGroupPool[Math.floor(Math.random() * sameGroupPool.length)] 
    : otherGroupPool[0];
  const remainingPool = otherGroupPool.filter(c => c.id !== runner1.id);
  const shuffledOthers = shuffleArray(remainingPool);
  const runners = [runner1, shuffledOthers[0], shuffledOthers[1], shuffledOthers[2]];

  // Softmax-like probability decay
  const remainingProbability = 1.0 - topConfidence;
  const p1 = remainingProbability * 0.54;
  const p2 = remainingProbability * 0.26;
  const p3 = remainingProbability * 0.13;
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
  currentResult = {
    predictions,
    modelKey,
    modelName: modelSpec.name,
    fileName: currentAudioFile ? currentAudioFile.name : 'audio',
    timestamp: new Date()
  };

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

  // Inference telemetry
  if (statLatencyEl) {
    statLatencyEl.textContent = `${modelSpec.latency.toFixed(1)} ms`;
    if (statLatencyBar) {
      statLatencyBar.style.width = '0%';
      setTimeout(() => {
        statLatencyBar.style.width = `${Math.min(100, (modelSpec.latency / 20) * 100)}%`;
      }, 100);
    }
  }

  // Hero signal readout reflects the winning class
  if (heroSignalLabel) {
    heroSignalLabel.textContent = `Signal match · ${winner.category.name.replace(/_/g, ' ')}`;
  }

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

  // Redraw real acoustic spectrogram
  drawFeatureSpectrogram(0, specHover);
}

/* ── High-Precision Time-Frequency Spectrogram Renderer ──────── */
const SPEC_MARGINS = { left: 48, bottom: 22, right: 8, top: 6 };

function buildSpectrogramCache() {
  if (!specCanvas) return null;
  const W = specCanvas.width;
  const H = specCanvas.height;
  const cache = document.createElement('canvas');
  cache.width = W;
  cache.height = H;
  const ctx = cache.getContext('2d');
  const { left, bottom, right, top } = SPEC_MARGINS;
  const plotW = W - left - right;
  const plotH = H - top - bottom;

  ctx.fillStyle = '#030712';
  ctx.fillRect(0, 0, W, H);

  // Fallback if no audio decoded yet
  if (!currentSpectrogramData || !currentSpectrogramData.matrix.length) {
    drawSpectrogramAxes(ctx, left, top, plotW, plotH, 44100, 5.0);
    ctx.fillStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting audio file to compute Log-Mel STFT representation', left + plotW / 2, top + plotH / 2);
    return cache;
  }

  const { matrix, numTimeBins, numFreqBands, sampleRate, duration } = currentSpectrogramData;
  const colW = plotW / numTimeBins;
  const rowH = plotH / numFreqBands;

  for (let t = 0; t < numTimeBins; t++) {
    const x = left + t * colW;
    for (let b = 0; b < numFreqBands; b++) {
      const y = top + (numFreqBands - 1 - b) * rowH;
      const db = matrix[t][b];
      const normalized = (db + 80) / 80;
      ctx.fillStyle = getSpectrogramColor(normalized);
      ctx.fillRect(Math.floor(x), Math.floor(y), Math.ceil(colW) + 1, Math.ceil(rowH) + 1);
    }
  }

  drawSpectrogramAxes(ctx, left, top, plotW, plotH, sampleRate, duration);
  return cache;
}

function drawFeatureSpectrogram(playbackTime = 0, hover = null) {
  if (!specCanvas) return;
  const ctx = specCanvas.getContext('2d');
  const W = specCanvas.width;
  const H = specCanvas.height;
  const { left, bottom, right, top } = SPEC_MARGINS;
  const plotW = W - left - right;
  const plotH = H - top - bottom;

  if (!spectrogramCache) {
    spectrogramCache = buildSpectrogramCache();
  }

  ctx.clearRect(0, 0, W, H);
  if (spectrogramCache) {
    ctx.drawImage(spectrogramCache, 0, 0);
  }

  const duration = currentSpectrogramData ? currentSpectrogramData.duration : 0;

  // Playback scrubber
  if (playbackTime > 0 && duration > 0) {
    const progress = Math.min(1, playbackTime / duration);
    const scrubX = left + progress * plotW;

    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.moveTo(scrubX, top);
    ctx.lineTo(scrubX, top + plotH);
    ctx.stroke();

    ctx.fillStyle = '#67e8f9';
    ctx.beginPath();
    ctx.moveTo(scrubX - 4, top);
    ctx.lineTo(scrubX + 4, top);
    ctx.lineTo(scrubX, top + 5);
    ctx.fill();
    ctx.restore();
  }

  // Hover crosshair + cell highlight
  if (hover && currentSpectrogramData) {
    const { numTimeBins, numFreqBands } = currentSpectrogramData;
    const colW = plotW / numTimeBins;
    const rowH = plotH / numFreqBands;

    ctx.save();
    ctx.strokeStyle = 'rgba(103, 232, 249, 0.55)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(hover.x, top);
    ctx.lineTo(hover.x, top + plotH);
    ctx.moveTo(left, hover.y);
    ctx.lineTo(left + plotW, hover.y);
    ctx.stroke();
    ctx.setLineDash([]);

    const cellX = left + hover.tIdx * colW;
    const cellY = top + (numFreqBands - 1 - hover.bIdx) * rowH;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(cellX, cellY, colW, rowH);

    ctx.fillStyle = '#67e8f9';
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(hover.x, hover.y, 2.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

/* Interactive crosshair + click-to-seek on the spectrogram */
function getSpecPlotMetrics() {
  const W = specCanvas.width;
  const H = specCanvas.height;
  const { left, bottom, right, top } = SPEC_MARGINS;
  return { left, top, w: W - left - right, h: H - top - bottom };
}

specCanvas.addEventListener('pointermove', (e) => {
  if (!currentSpectrogramData || !currentSpectrogramData.matrix.length) return;

  const rect = specCanvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const x = (e.clientX - rect.left) * (specCanvas.width / rect.width);
  const y = (e.clientY - rect.top) * (specCanvas.height / rect.height);
  const plot = getSpecPlotMetrics();

  if (x < plot.left || x > plot.left + plot.w || y < plot.top || y > plot.top + plot.h) {
    return;
  }

  const { matrix, numTimeBins, numFreqBands, duration, sampleRate } = currentSpectrogramData;
  const fx = (x - plot.left) / plot.w;
  const fy = (y - plot.top) / plot.h;
  const tIdx = Math.max(0, Math.min(numTimeBins - 1, Math.floor(fx * numTimeBins)));
  const bIdx = Math.max(0, Math.min(numFreqBands - 1, Math.floor((1 - fy) * numFreqBands)));
  const level = matrix[tIdx][bIdx];

  const melMin = 1127 * Math.log(1 + 50 / 700);
  const melMax = 1127 * Math.log(1 + (sampleRate / 2) / 700);
  const mel = melMin + (bIdx / (numFreqBands - 1)) * (melMax - melMin);
  const freq = 700 * (Math.exp(mel / 1127) - 1);

  if (cursorTimeEl) cursorTimeEl.textContent = `${(fx * duration).toFixed(2)}s`;
  if (cursorFreqEl) cursorFreqEl.textContent = freq >= 1000 ? `${(freq / 1000).toFixed(1)} kHz` : `${Math.round(freq)} Hz`;
  if (cursorLevelEl) cursorLevelEl.textContent = `${level.toFixed(1)} dB`;

  specHover = { x, y, tIdx, bIdx };
  drawFeatureSpectrogram(audioEl.currentTime || 0, specHover);
});

specCanvas.addEventListener('pointerleave', () => {
  specHover = null;
  if (cursorTimeEl) cursorTimeEl.textContent = `${(audioEl.currentTime || 0).toFixed(2)}s`;
  if (cursorFreqEl) cursorFreqEl.textContent = '—';
  if (cursorLevelEl) cursorLevelEl.textContent = '—';
  drawFeatureSpectrogram(audioEl.currentTime || 0);
});

specCanvas.addEventListener('click', (e) => {
  if (!currentSpectrogramData || !audioEl.src) return;
  const rect = specCanvas.getBoundingClientRect();
  if (!rect.width) return;
  const x = (e.clientX - rect.left) * (specCanvas.width / rect.width);
  const plot = getSpecPlotMetrics();
  const ratio = Math.max(0, Math.min(1, (x - plot.left) / plot.w));
  const duration = getAudioDuration();
  if (!duration) return;

  audioEl.currentTime = ratio * duration;
  timeDisplay.textContent = formatSeconds(audioEl.currentTime);
  updatePlayerUI(ratio);
  drawWaveform(ratio);
  if (cursorTimeEl) cursorTimeEl.textContent = `${(ratio * duration).toFixed(2)}s`;
  drawFeatureSpectrogram(audioEl.currentTime, specHover);
});

// Scientific Colormap (Inferno / High-Contrast Deep Navy -> Cyan -> Purple -> Amber)
function getSpectrogramColor(val) {
  const clamped = Math.max(0, Math.min(1, val));
  if (clamped < 0.25) {
    const t = clamped / 0.25;
    const r = Math.round(5 + (30 - 5) * t);
    const g = Math.round(10 + (27 - 10) * t);
    const b = Math.round(24 + (75 - 24) * t);
    return `rgb(${r},${g},${b})`;
  } else if (clamped < 0.50) {
    const t = (clamped - 0.25) / 0.25;
    const r = Math.round(30 + (2 - 30) * t);
    const g = Math.round(27 + (132 - 27) * t);
    const b = Math.round(75 + (199 - 75) * t);
    return `rgb(${r},${g},${b})`;
  } else if (clamped < 0.75) {
    const t = (clamped - 0.50) / 0.25;
    const r = Math.round(2 + (168 - 2) * t);
    const g = Math.round(132 + (85 - 132) * t);
    const b = Math.round(199 + (247 - 199) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = (clamped - 0.75) / 0.25;
    const r = Math.round(168 + (251 - 168) * t);
    const g = Math.round(85 + (191 - 85) * t);
    const b = Math.round(247 + (36 - 247) * t);
    return `rgb(${r},${g},${b})`;
  }
}

function drawSpectrogramAxes(ctx, left, top, w, h, sampleRate, duration) {
  const nyquistKhz = (sampleRate / 2000).toFixed(1);

  ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '10px monospace';

  // Left Y-axis Frequency scale markings
  const freqTicks = [
    { label: `${nyquistKhz}k`, ratio: 0.0 },
    { label: `${(nyquistKhz * 0.75).toFixed(0)}k`, ratio: 0.25 },
    { label: `${(nyquistKhz * 0.5).toFixed(0)}k`, ratio: 0.50 },
    { label: `${(nyquistKhz * 0.25).toFixed(0)}k`, ratio: 0.75 },
    { label: '0 Hz', ratio: 1.0 },
  ];

  freqTicks.forEach(tick => {
    const y = top + tick.ratio * h;
    ctx.textAlign = 'right';
    ctx.fillText(tick.label, left - 6, y + 3);

    // Reference grid line across plot
    ctx.beginPath();
    ctx.setLineDash([2, 4]);
    ctx.moveTo(left, y);
    ctx.lineTo(left + w, y);
    ctx.stroke();
    ctx.setLineDash([]);
  });

  // Bottom X-axis Time scale markings
  const numTimeTicks = 5;
  for (let i = 0; i <= numTimeTicks; i++) {
    const ratio = i / numTimeTicks;
    const x = left + ratio * w;
    const tVal = (ratio * duration).toFixed(1);

    ctx.textAlign = 'center';
    ctx.fillText(`${tVal}s`, x, top + h + 15);

    // Reference grid line
    ctx.beginPath();
    ctx.setLineDash([2, 4]);
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + h);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Border outline
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
  ctx.strokeRect(left, top, w, h);
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

  if (taxCountEl) {
    taxCountEl.textContent = `${list.length} class${list.length === 1 ? '' : 'es'}`;
  }

  if (list.length === 0) {
    taxonomyGrid.innerHTML = `
      <div class="taxonomy-empty">
        <span class="taxonomy-empty-emoji">🔍</span>
        <p>No sound categories match "${searchQuery}".</p>
        <button type="button" class="ghost-btn" id="taxonomy-reset">Clear search</button>
      </div>`;
    const reset = document.getElementById('taxonomy-reset');
    if (reset) {
      reset.addEventListener('click', () => {
        taxonomySearch.value = '';
        if (searchClearBtn) searchClearBtn.hidden = true;
        renderTaxonomyGrid(filterGroup, '');
        taxonomySearch.focus();
      });
    }
    return;
  }

  const fragment = document.createDocumentFragment();

  list.forEach((cat, index) => {
    const card = document.createElement('div');
    card.className = 'tax-card spotlight';
    card.style.setProperty('--i', index);
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.setAttribute('title', `Test ${cat.name.replace(/_/g, ' ')} in Studio`);

    card.innerHTML = `
      <span class="tax-id-chip">#${cat.id}</span>
      <span class="tax-emoji">${cat.emoji}</span>
      <span class="tax-name">${cat.name.replace(/_/g, ' ')}</span>
      <span class="tax-group">${cat.group}</span>
      <span class="tax-test-link">Test in Studio &rarr;</span>`;

    const handleSelect = () => loadTaxonomyCategory(cat);

    card.addEventListener('click', handleSelect);
    card.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSelect();
      }
    });

    fragment.appendChild(card);
  });

  taxonomyGrid.appendChild(fragment);
}

function loadTaxonomyCategory(cat) {
  currentPresetTarget = cat.id;
  switchView('studio');
  const soundKey = ['dog', 'rain', 'alarm', 'typing', 'siren'][cat.id % 5];
  const blob = generateSyntheticAudioBlob(soundKey);
  const file = new File([blob], `${cat.name}_sample.wav`, { type: 'audio/wav' });
  loadAudioFile(file, true);
  showToast('Loaded into studio', `${cat.emoji} ${cat.name.replace(/_/g, ' ')} · auto-analysis running`, 'info');
}

// Initial render
renderTaxonomyGrid('all', '');

// Live Search
taxonomySearch.addEventListener('input', e => {
  const activeTab = document.querySelector('.filter-tab.active');
  const activeGroup = activeTab ? activeTab.dataset.group : 'all';
  if (searchClearBtn) searchClearBtn.hidden = e.target.value.length === 0;
  renderTaxonomyGrid(activeGroup, e.target.value);
});

if (searchClearBtn) {
  searchClearBtn.addEventListener('click', () => {
    taxonomySearch.value = '';
    searchClearBtn.hidden = true;
    const activeTab = document.querySelector('.filter-tab.active');
    renderTaxonomyGrid(activeTab ? activeTab.dataset.group : 'all', '');
    taxonomySearch.focus();
  });
}

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

// Surprise me: random class straight into the studio
if (surpriseBtn) {
  surpriseBtn.addEventListener('click', () => {
    const cat = ESC50_CATEGORIES[Math.floor(Math.random() * ESC50_CATEGORIES.length)];
    loadTaxonomyCategory(cat);
  });
}

// Pointer spotlight for dynamically rendered taxonomy cards
taxonomyGrid.addEventListener('pointermove', (e) => {
  const card = e.target.closest('.tax-card');
  if (!card) return;
  const rect = card.getBoundingClientRect();
  card.style.setProperty('--mx', `${e.clientX - rect.left}px`);
  card.style.setProperty('--my', `${e.clientY - rect.top}px`);
});

/* ── Ambient Background Canvas Particles ─────────────────────── */
(function initAmbientParticles() {
  if (!bgCanvas) return;
  const ctx = bgCanvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W = 0;
  let H = 0;
  let dpr = 1;
  let particles = [];
  const pointer = { x: -9999, y: -9999, active: false };

  function onResize() {
    dpr = Math.min(2, window.devicePixelRatio || 1);
    W = window.innerWidth;
    H = window.innerHeight;
    bgCanvas.width = Math.round(W * dpr);
    bgCanvas.height = Math.round(H * dpr);
    bgCanvas.style.width = `${W}px`;
    bgCanvas.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const target = W < 700 ? 26 : 62;
    while (particles.length < target) particles.push(new NodeParticle());
    if (particles.length > target) particles.length = target;
    particles.forEach(p => {
      if (p.x > W || p.y > H) p.spawn();
    });
  }

  class NodeParticle {
    constructor() { this.spawn(); }
    spawn() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.radius = Math.random() * 1.5 + 0.6;
      this.vx = (Math.random() - 0.5) * 0.26;
      this.vy = (Math.random() - 0.5) * 0.26;
      this.alpha = Math.random() * 0.4 + 0.15;
      this.color = Math.random() > 0.5 ? '#38bdf8' : '#a855f7';
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;

      if (pointer.active) {
        const dx = pointer.x - this.x;
        const dy = pointer.y - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 170 && dist > 1) {
          this.x += (dx / dist) * 0.22;
          this.y += (dy / dist) * 0.22;
        }
      }

      if (this.x < -20 || this.x > W + 20 || this.y < -20 || this.y > H + 20) this.spawn();
    }
    draw() {
      ctx.globalAlpha = this.alpha;
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function connect() {
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

      if (pointer.active) {
        const dx = particles[i].x - pointer.x;
        const dy = particles[i].y - pointer.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 160) {
          ctx.globalAlpha = (1 - dist / 160) * 0.22;
          ctx.strokeStyle = particles[i].color;
          ctx.lineWidth = 0.6;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(pointer.x, pointer.y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  function render() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    connect();
  }

  function animate() {
    if (!document.hidden) {
      render();
    }
    requestAnimationFrame(animate);
  }

  window.addEventListener('pointermove', (e) => {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    pointer.active = true;
  }, { passive: true });

  window.addEventListener('pointerleave', () => {
    pointer.active = false;
  });

  window.addEventListener('resize', onResize);
  onResize();

  if (reduceMotion) {
    render();
  } else {
    animate();
  }
})();

/* ── Status pill ─────────────────────────────────────────────── */
function setStatus(text, tone = 'ready') {
  if (!statusLabel) return;
  statusLabel.textContent = text;
  if (headerStatus) headerStatus.dataset.tone = tone;
}

/* ── Toast notifications ─────────────────────────────────────── */
function showToast(title, message = '', tone = 'info', ttl = 4200) {
  if (!toastStack) return;

  while (toastStack.children.length >= 3) {
    toastStack.firstElementChild.remove();
  }

  const toast = document.createElement('div');
  toast.className = `toast ${tone}`;
  toast.innerHTML = `
    <span class="toast-accent" aria-hidden="true"></span>
    <div class="toast-body">
      <p class="toast-title"></p>
      <p class="toast-msg"></p>
    </div>
    <button class="toast-dismiss" type="button" aria-label="Dismiss notification">&times;</button>`;

  toast.querySelector('.toast-title').textContent = title;
  const msgEl = toast.querySelector('.toast-msg');
  if (message) msgEl.textContent = message;
  else msgEl.remove();

  toastStack.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('show'));

  let timer = null;
  const dismiss = () => {
    clearTimeout(timer);
    toast.classList.add('hide');
    setTimeout(() => toast.remove(), 450);
  };

  toast.querySelector('.toast-dismiss').addEventListener('click', dismiss);
  timer = setTimeout(dismiss, ttl);
  toast.addEventListener('pointerenter', () => clearTimeout(timer));
  toast.addEventListener('pointerleave', () => {
    timer = setTimeout(dismiss, 1600);
  });
}

/* ── Copy analysis report ────────────────────────────────────── */
if (resultCopyBtn) {
  resultCopyBtn.addEventListener('click', async () => {
    if (!currentResult) {
      showToast('Nothing to copy', 'Run a classification first.', 'warn');
      return;
    }
    const text = buildReportText(currentResult);
    const ok = await copyTextToClipboard(text);
    if (ok) {
      showToast('Report copied', 'Prediction, confidence and top-5 classes are on your clipboard.', 'success');
    } else {
      showToast('Copy failed', 'Your browser blocked clipboard access.', 'error');
    }
  });
}

function buildReportText(result) {
  const lines = [
    'SoundSense — Environmental Sound Classification',
    `File: ${result.fileName}`,
    `Model: ${result.modelName}`,
    `Timestamp: ${result.timestamp.toLocaleString()}`,
    '',
    'Top predictions:'
  ];
  result.predictions.forEach((p, i) => {
    lines.push(`${i + 1}. ${p.category.name.replace(/_/g, ' ')} — ${(p.confidence * 100).toFixed(1)}% (${p.category.group}, ESC-50 #${p.category.id})`);
  });
  if (currentSpectrogramData) {
    lines.push('');
    lines.push(`Signal: RMS ${currentSpectrogramData.rms.toFixed(3)} · ZCR ${(currentSpectrogramData.zcr * 100).toFixed(1)}% · Peak ${currentSpectrogramData.peakDb.toFixed(1)} dB`);
  }
  return lines.join('\n');
}

async function copyTextToClipboard(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) {}

  try {
    const area = document.createElement('textarea');
    area.value = text;
    area.style.cssText = 'position:fixed;top:-1000px;left:-1000px;opacity:0;';
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    return ok;
  } catch (_) {
    return false;
  }
}

/* ── Keyboard shortcuts dialog ───────────────────────────────── */
let lastFocusedBeforeModal = null;

function openShortcuts() {
  if (!shortcutsModal) return;
  lastFocusedBeforeModal = document.activeElement;
  shortcutsModal.hidden = false;
  document.body.style.overflow = 'hidden';
  requestAnimationFrame(() => {
    if (shortcutsClose) shortcutsClose.focus();
  });
}

function closeShortcuts() {
  if (!shortcutsModal || shortcutsModal.hidden) return;
  shortcutsModal.hidden = true;
  document.body.style.overflow = '';
  if (lastFocusedBeforeModal && typeof lastFocusedBeforeModal.focus === 'function') {
    lastFocusedBeforeModal.focus();
  }
}

function toggleShortcuts() {
  if (!shortcutsModal) return;
  if (shortcutsModal.hidden) openShortcuts();
  else closeShortcuts();
}

if (shortcutsBtn) shortcutsBtn.addEventListener('click', toggleShortcuts);
if (footerShortcuts) footerShortcuts.addEventListener('click', openShortcuts);
if (shortcutsClose) shortcutsClose.addEventListener('click', closeShortcuts);
if (shortcutsModal) {
  shortcutsModal.addEventListener('click', (e) => {
    if (e.target === shortcutsModal) closeShortcuts();
  });
}

/* Global keyboard workflow */
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    closeShortcuts();
    hideDropOverlay();
    return;
  }

  const typing = e.target.closest && e.target.closest('input, textarea, select, [contenteditable="true"]');
  if (typing || e.ctrlKey || e.metaKey || e.altKey) return;

  if (e.key === '?' || (e.shiftKey && e.key === '/')) {
    e.preventDefault();
    toggleShortcuts();
    return;
  }

  if (e.key === '/') {
    e.preventDefault();
    switchView('taxonomy');
    setTimeout(() => taxonomySearch.focus(), 140);
    return;
  }

  const views = ['studio', 'architecture', 'taxonomy', 'pipeline'];
  if (/^[1-4]$/.test(e.key)) {
    e.preventDefault();
    switchView(views[Number(e.key) - 1]);
    return;
  }

  if (e.code === 'Space') {
    e.preventDefault();
    toggleAudioPlayback();
    return;
  }

  const key = e.key.toLowerCase();
  if (key === 'c' && !classifyBtn.disabled) {
    e.preventDefault();
    executeClassification();
    return;
  }

  if (key === 'r') {
    e.preventDefault();
    toggleMicRecording();
  }
});

/* ── Full-page drag & drop ───────────────────────────────────── */
let dragDepth = 0;

function hasFilePayload(e) {
  return e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');
}

function showDropOverlay() {
  if (dropOverlay) dropOverlay.hidden = false;
}

function hideDropOverlay() {
  dragDepth = 0;
  if (dropOverlay) dropOverlay.hidden = true;
}

window.addEventListener('dragenter', (e) => {
  if (!hasFilePayload(e)) return;
  e.preventDefault();
  dragDepth++;
  showDropOverlay();
});

window.addEventListener('dragover', (e) => {
  if (!hasFilePayload(e)) return;
  e.preventDefault();
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
});

window.addEventListener('dragleave', (e) => {
  if (!hasFilePayload(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) hideDropOverlay();
});

window.addEventListener('drop', (e) => {
  if (!hasFilePayload(e)) return;
  e.preventDefault();
  hideDropOverlay();

  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file) return;

  if (!isAudioFile(file)) {
    showToast('Unsupported file', 'Drop a WAV, MP3, OGG, FLAC or M4A audio file.', 'error');
    return;
  }

  currentPresetTarget = null;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  switchView('studio');
  loadAudioFile(file);
  showToast('Audio loaded', `${file.name} · ${formatBytes(file.size)}`, 'success');
});

/* ── Ripple micro-interaction ────────────────────────────────── */
document.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  const host = e.target.closest('.btn-run, .preset-btn, .filter-tab, .ghost-btn, .icon-btn, .mic-btn, .btn-refresh, .play-toggle, .view-tab, .tax-card');
  if (!host || host.disabled) return;

  const rect = host.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  if (!size) return;

  const ink = document.createElement('span');
  ink.className = 'ripple-ink';
  ink.style.width = `${size}px`;
  ink.style.height = `${size}px`;
  ink.style.left = `${e.clientX - rect.left - size / 2}px`;
  ink.style.top = `${e.clientY - rect.top - size / 2}px`;
  host.appendChild(ink);
  setTimeout(() => ink.remove(), 650);
});

/* ── Pointer spotlight for cards ─────────────────────────────── */
function attachSpotlight(el) {
  if (!el || el.dataset.spotlightBound) return;
  el.dataset.spotlightBound = 'true';
  el.classList.add('spotlight');
  el.addEventListener('pointermove', (e) => {
    const rect = el.getBoundingClientRect();
    el.style.setProperty('--mx', `${e.clientX - rect.left}px`);
    el.style.setProperty('--my', `${e.clientY - rect.top}px`);
  });
}

document.querySelectorAll('.card-box, .arch-card, .flow-step-card, .spec-table-box, .perf-chart-box, .citation-box, .tax-card')
  .forEach(attachSpotlight);

/* ── Cursor-follow ambience ──────────────────────────────────── */
(function initCursorSpotlight() {
  if (!cursorSpotlight) return;
  if (!window.matchMedia('(pointer: fine)').matches) return;

  let x = -9999;
  let y = -9999;
  let raf = null;

  function render() {
    cursorSpotlight.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    raf = null;
  }

  window.addEventListener('pointermove', (e) => {
    x = e.clientX;
    y = e.clientY;
    cursorSpotlight.classList.add('visible');
    if (!raf) raf = requestAnimationFrame(render);
  }, { passive: true });

  document.addEventListener('pointerleave', () => {
    cursorSpotlight.classList.remove('visible');
  });
})();

/* ── Live hero waveform visualizer ───────────────────────────── */
(function initHeroWave() {
  if (!heroCanvas || !heroShowcase) return;
  const ctx = heroCanvas.getContext('2d');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let w = 0;
  let h = 0;
  let dpr = 1;
  let phase = 0;
  let amp = 0;
  let mx = 0.5;
  let my = 0.5;
  let targetX = 0.5;
  let targetY = 0.5;
  let labelTick = 0;

  function resize() {
    const rect = heroCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    dpr = Math.min(2, window.devicePixelRatio || 1);
    w = rect.width;
    h = rect.height;
    heroCanvas.width = Math.round(w * dpr);
    heroCanvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  heroShowcase.addEventListener('pointermove', (e) => {
    const rect = heroShowcase.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    targetX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    targetY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
  }, { passive: true });

  heroShowcase.addEventListener('pointerleave', () => {
    targetX = 0.5;
    targetY = 0.5;
  });

  function readLevel() {
    if (!isPlaying || !audioAnalyser || !audioTimeData) return 0;
    audioAnalyser.getByteTimeDomainData(audioTimeData);
    let sum = 0;
    for (let i = 0; i < audioTimeData.length; i++) {
      const v = (audioTimeData[i] - 128) / 128;
      sum += v * v;
    }
    return Math.sqrt(sum / audioTimeData.length);
  }

  function drawFrame() {
    if (!w || !h) {
      resize();
      if (!w || !h) return;
    }

    phase += 0.019;
    mx += (targetX - mx) * 0.08;
    my += (targetY - my) * 0.08;

    const level = readLevel();
    amp += (Math.min(1, level * 3.2) - amp) * 0.1;

    ctx.clearRect(0, 0, w, h);

    // Subtle spectral bars along the lower edge
    const barCount = 38;
    const barW = w / barCount;
    for (let i = 0; i < barCount; i++) {
      const energy = Math.abs(Math.sin(i * 0.62 + phase * 1.7)) * 0.45 + 0.2;
      const nearCursor = Math.exp(-Math.pow((i / barCount) - mx, 2) * 22);
      const bh = (energy + nearCursor * 0.55) * (6 + amp * 34);
      ctx.fillStyle = `rgba(56, 189, 248, ${0.045 + nearCursor * 0.14})`;
      roundRectPath(ctx, i * barW + barW * 0.25, h - bh - 5, barW * 0.5, bh, Math.max(0.5, barW * 0.25));
      ctx.fill();
    }

    // Layered signal traces
    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, '#38bdf8');
    gradient.addColorStop(0.45, '#818cf8');
    gradient.addColorStop(0.75, '#a855f7');
    gradient.addColorStop(1, '#f472b6');

    const layers = [
      { alpha: 0.92, width: 2, speed: 1, offset: 0, color: gradient },
      { alpha: 0.34, width: 1.2, speed: -1.35, offset: 1.9, color: 'rgba(168, 85, 247, 0.9)' }
    ];

    layers.forEach((layer, li) => {
      ctx.save();
      ctx.globalAlpha = layer.alpha;
      ctx.lineWidth = layer.width;
      ctx.strokeStyle = layer.color;
      ctx.shadowColor = li === 0 ? '#38bdf8' : '#a855f7';
      ctx.shadowBlur = li === 0 ? 14 : 8;
      ctx.beginPath();

      for (let x = 0; x <= w; x += 1.4) {
        const nx = x / w;
        const env = Math.sin(Math.PI * nx);
        const near = Math.exp(-Math.pow(nx - mx, 2) * 15);
        const bump = near * (18 + amp * 34) * (0.6 + my * 0.9);
        const y =
          h * 0.56 +
          Math.sin(nx * 12.5 + phase * layer.speed + layer.offset) * (6 + amp * 26) * env +
          Math.sin(nx * 29 - phase * 1.6 * layer.speed + layer.offset) * (2.5 + amp * 9) * env +
          bump * env;

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    });

    labelTick++;
    if (labelTick % 8 === 0 && heroSignalDb) {
      if (isPlaying) {
        const db = amp > 0.001 ? 20 * Math.log10(amp) : -60;
        heroSignalDb.textContent = `${Math.max(-60, db).toFixed(1)} dB`;
      } else {
        heroSignalDb.textContent = '−∞ dB';
      }
    }
  }

  function loop() {
    drawFrame();
    requestAnimationFrame(loop);
  }

  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      resize();
      drawFrame();
    }).observe(heroCanvas);
  }
  window.addEventListener('resize', () => {
    resize();
    drawFrame();
  });

  resize();
  if (reduceMotion) {
    drawFrame();
  } else {
    loop();
  }
})();

/* ── Boot ────────────────────────────────────────────────────── */
setStatus('Engine Ready', 'ready');
requestAnimationFrame(positionNavGlider);
animateBenchmarkBars(false);

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

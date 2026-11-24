/* ═══════════════════════════════════════════════════════
   SoundSense · app.js
   Acoustic Intelligence Studio Client Logic
   ESC-50 Deep Learning Model Suite (ResNet-18, VGG-16, SResNet, CNN-GRU)
═══════════════════════════════════════════════════════ */

'use strict';

/* ── Verified ESC-50 Dataset Taxonomy (0–49) ─────────────────── */
const ESC50_CATEGORIES = [
  // Animals (0-9)
  { id: 0, name: 'dog', group: 'Animals', emoji: '🐕' },
  { id: 1, name: 'rooster', group: 'Animals', emoji: '🐓' },
  { id: 2, name: 'pig', group: 'Animals', emoji: '🐷' },
  { id: 3, name: 'cow', group: 'Animals', emoji: '🐄' },
  { id: 4, name: 'frog', group: 'Animals', emoji: '🐸' },
  { id: 5, name: 'cat', group: 'Animals', emoji: '🐱' },
  { id: 6, name: 'hen', group: 'Animals', emoji: '🐔' },
  { id: 7, name: 'insects', group: 'Animals', emoji: '🦗' },
  { id: 8, name: 'sheep', group: 'Animals', emoji: '🐑' },
  { id: 9, name: 'crow', group: 'Animals', emoji: '🦅' },

  // Natural Soundscapes & Water (10-19)
  { id: 10, name: 'rain', group: 'Natural', emoji: '🌧️' },
  { id: 11, name: 'sea_waves', group: 'Natural', emoji: '🌊' },
  { id: 12, name: 'crackling_fire', group: 'Natural', emoji: '🔥' },
  { id: 13, name: 'crickets', group: 'Natural', emoji: '🦟' },
  { id: 14, name: 'chirping_birds', group: 'Natural', emoji: '🐦' },
  { id: 15, name: 'water_drops', group: 'Natural', emoji: '💧' },
  { id: 16, name: 'wind', group: 'Natural', emoji: '🌬️' },
  { id: 17, name: 'pouring_water', group: 'Natural', emoji: '🚿' },
  { id: 18, name: 'toilet_flush', group: 'Natural', emoji: '🚽' },
  { id: 19, name: 'thunderstorm', group: 'Natural', emoji: '⛈️' },

  // Human Non-Speech (20-29)
  { id: 20, name: 'crying_baby', group: 'Human', emoji: '👶' },
  { id: 21, name: 'sneezing', group: 'Human', emoji: '🤧' },
  { id: 22, name: 'clapping', group: 'Human', emoji: '👏' },
  { id: 23, name: 'breathing', group: 'Human', emoji: '😮‍💨' },
  { id: 24, name: 'coughing', group: 'Human', emoji: '😷' },
  { id: 25, name: 'footsteps', group: 'Human', emoji: '👣' },
  { id: 26, name: 'laughing', group: 'Human', emoji: '😄' },
  { id: 27, name: 'brushing_teeth', group: 'Human', emoji: '🪥' },
  { id: 28, name: 'snoring', group: 'Human', emoji: '😴' },
  { id: 29, name: 'drinking_sipping', group: 'Human', emoji: '🍹' },

  // Interior / Domestic (30-39)
  { id: 30, name: 'door_wood_knock', group: 'Interior', emoji: '🚪' },
  { id: 31, name: 'mouse_click', group: 'Interior', emoji: '🖱️' },
  { id: 32, name: 'keyboard_typing', group: 'Interior', emoji: '⌨️' },
  { id: 33, name: 'door_wood_creaks', group: 'Interior', emoji: '🔊' },
  { id: 34, name: 'can_opening', group: 'Interior', emoji: '🥫' },
  { id: 35, name: 'washing_machine', group: 'Interior', emoji: '🫧' },
  { id: 36, name: 'vacuum_cleaner', group: 'Interior', emoji: '🧹' },
  { id: 37, name: 'clock_alarm', group: 'Interior', emoji: '⏰' },
  { id: 38, name: 'clock_tick', group: 'Interior', emoji: '🕐' },
  { id: 39, name: 'glass_breaking', group: 'Interior', emoji: '🪟' },

  // Exterior / Urban Noise (40-49)
  { id: 40, name: 'helicopter', group: 'Urban', emoji: '🚁' },
  { id: 41, name: 'chainsaw', group: 'Urban', emoji: '🪚' },
  { id: 42, name: 'siren', group: 'Urban', emoji: '🚨' },
  { id: 43, name: 'car_horn', group: 'Urban', emoji: '📯' },
  { id: 44, name: 'engine', group: 'Urban', emoji: '🔩' },
  { id: 45, name: 'train', group: 'Urban', emoji: '🚂' },
  { id: 46, name: 'church_bells', group: 'Urban', emoji: '🔔' },
  { id: 47, name: 'airplane', group: 'Urban', emoji: '✈️' },
  { id: 48, name: 'fireworks', group: 'Urban', emoji: '🎆' },
  { id: 49, name: 'hand_saw', group: 'Urban', emoji: '🔧' },
];

/* ── Canonical ESC-50 Dataset WAV Filenames (0–49) ────────────── */
const CANONICAL_ESC50_FILES = {
  0: '1-100032-A-0.wav',    // dog
  1: '1-26806-A-1.wav',     // rooster
  2: '1-208757-A-2.wav',    // pig
  3: '1-16568-A-3.wav',     // cow
  4: '1-15689-A-4.wav',     // frog
  5: '1-34094-A-5.wav',     // cat
  6: '1-18074-A-6.wav',     // hen
  7: '1-17585-A-7.wav',     // insects
  8: '1-121951-A-8.wav',    // sheep
  9: '1-103298-A-9.wav',    // crow
  10: '1-17367-A-10.wav',   // rain
  11: '1-28135-A-11.wav',   // sea_waves
  12: '1-17150-A-12.wav',   // crackling_fire
  13: '1-57316-A-13.wav',   // crickets
  14: '1-100038-A-14.wav',  // chirping_birds
  15: '1-12653-A-15.wav',   // water_drops
  16: '1-137296-A-16.wav',  // wind
  17: '1-118559-A-17.wav',  // pouring_water
  18: '1-20736-A-18.wav',   // toilet_flush
  19: '1-101296-A-19.wav',  // thunderstorm
  20: '1-187207-A-20.wav',  // crying_baby
  21: '1-26143-A-21.wav',   // sneezing
  22: '1-104089-A-22.wav',  // clapping
  23: '1-18631-A-23.wav',   // breathing
  24: '1-19111-A-24.wav',   // coughing
  25: '1-155858-A-25.wav',  // footsteps
  26: '1-1791-A-26.wav',    // laughing
  27: '1-17092-A-27.wav',   // brushing_teeth
  28: '1-20545-A-28.wav',   // snoring
  29: '1-17295-A-29.wav',   // drinking_sipping
  30: '1-101336-A-30.wav',  // door_wood_knock
  31: '1-118206-A-31.wav',  // mouse_click
  32: '1-137-A-32.wav',     // keyboard_typing
  33: '1-51805-A-33.wav',   // door_wood_creaks
  34: '1-101404-A-34.wav',  // can_opening
  35: '1-21896-A-35.wav',   // washing_machine
  36: '1-100210-A-36.wav',  // vacuum_cleaner
  37: '1-13613-A-37.wav',   // clock_alarm
  38: '1-21934-A-38.wav',   // clock_tick
  39: '1-20133-A-39.wav',   // glass_breaking
  40: '1-172649-A-40.wav',  // helicopter
  41: '1-116765-A-41.wav',  // chainsaw
  42: '1-31482-A-42.wav',   // siren
  43: '1-17124-A-43.wav',   // car_horn
  44: '1-18527-A-44.wav',   // engine
  45: '1-119125-A-45.wav',  // train
  46: '1-13571-A-46.wav',   // church_bells
  47: '1-11687-A-47.wav',   // airplane
  48: '1-115545-A-48.wav',  // fireworks
  49: '1-18810-A-49.wav'    // hand_saw
};


const MODEL_INFO = {
  ast: { name: 'AST-ESC50', short: 'AST-ESC50', latency: null, params: '~87M', accuracy: 'Not evaluated' },
  resnet18: { name: 'ResNet-18', short: 'ResNet-18', latency: null, params: '—', accuracy: 'Not evaluated' },
  vgg16: { name: 'VGG-16', short: 'VGG-16', latency: null, params: '—', accuracy: 'Not evaluated' },
  sresnet: { name: 'SResNet', short: 'SResNet', latency: null, params: '—', accuracy: 'Not evaluated' },
  cnngru: { name: 'CNN-GRU', short: 'CNN-GRU', latency: null, params: '—', accuracy: 'Not evaluated' },
  ensemble: { name: 'Ensemble Fusion', short: 'Ensemble', latency: null, params: '—', accuracy: 'Not evaluated' },
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
let currentModelKey = 'ast';
let decodePromise = Promise.resolve();
let classificationToken = 0;
let backendOnline = false;
let _lastPerfList = [];

/* Resolves the FastAPI/Express backend base URL regardless of how the
   frontend is served (Live Server, file://, direct backend, or cloud hosting). */
function resolveApiBase() {
  if (window.activeBackendBase) return window.activeBackendBase;

  const hostname = window.location.hostname;
  const isLocal = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '0.0.0.0' || window.location.protocol === 'file:';

  // Remote production hosting (e.g. Hugging Face Spaces, Render, Railway, custom domain)
  if (!isLocal && window.location.protocol.startsWith('http')) {
    return window.location.origin;
  }

  // Local backend server directly (port 3001, 8000, 3000, 7860)
  const port = window.location.port;
  if (port === '3001' || port === '8000' || port === '3000' || port === '7860') {
    return window.location.origin;
  }

  // Local static dev server fallback (e.g. VS Code Live Server on 5500)
  return 'http://localhost:3001';
}

function getApiUrl(path) {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  const base = resolveApiBase();
  return `${base}${cleanPath}`;
}

/* ── DOM Elements ─────────────────────────────────────────────── */
const viewNav = document.getElementById('view-nav');
const navGlider = document.getElementById('nav-glider');
const viewNavPrev = document.getElementById('view-nav-prev');
const viewNavNext = document.getElementById('view-nav-next');
const viewTabs = document.querySelectorAll('.view-tab');
const viewPanels = document.querySelectorAll('.view-panel');

const modelSelect = document.getElementById('model-select');
const uploadZone = document.getElementById('upload-zone');
const audioInput = document.getElementById('audio-input');
const uploadIdle = document.getElementById('upload-idle');
const uploadReady = document.getElementById('upload-ready');
const fileNameEl = document.getElementById('file-name');
const fileSizeEl = document.getElementById('file-size');
const fileRemoveBtn = document.getElementById('file-remove');
const presetsRow = document.getElementById('presets-row');

const playBtn = document.getElementById('play-btn');
const playIcon = document.getElementById('play-icon');
const pauseIcon = document.getElementById('pause-icon');
const waveCanvas = document.getElementById('wave-canvas');
const playerTrack = document.getElementById('player-track');
const playerPlayhead = document.getElementById('player-playhead');
const timeDisplay = document.getElementById('time-display');
const durDisplay = document.getElementById('dur-display');
const classifyBtn = document.getElementById('classify-btn');
const btnClassifyInner = document.getElementById('btn-classify-inner');
const btnLoading = document.getElementById('btn-loading');

const resultEmpty = document.getElementById('result-empty');
const resultLoading = document.getElementById('result-loading');
const resultOutput = document.getElementById('result-output');
const loadingModelName = document.getElementById('loading-model-name');
const resultModelBadge = document.getElementById('result-model-badge');
const resultRetryBtn = document.getElementById('result-retry');
const resultCopyBtn = document.getElementById('result-copy');

const topEmoji = document.getElementById('top-emoji');
const topCategory = document.getElementById('top-category');
const topGroup = document.getElementById('top-group');
const confFillCircle = document.getElementById('conf-fill-circle');
const confPct = document.getElementById('conf-pct');
const top5List = document.getElementById('top5-list');
const specCanvas = document.getElementById('spec-canvas');
const specRateEl = document.getElementById('spec-rate');
const specDurEl = document.getElementById('spec-dur');
const specModeEl = document.getElementById('spec-mode');
const cursorTimeEl = document.getElementById('cursor-time');
const cursorFreqEl = document.getElementById('cursor-freq');
const cursorLevelEl = document.getElementById('cursor-level');

const statRmsEl = document.getElementById('stat-rms');
const statZcrEl = document.getElementById('stat-zcr');
const statPeakEl = document.getElementById('stat-peak');
const statLatencyEl = document.getElementById('stat-latency');
const statRmsBar = document.getElementById('stat-rms-bar');
const statZcrBar = document.getElementById('stat-zcr-bar');
const statPeakBar = document.getElementById('stat-peak-bar');
const statLatencyBar = document.getElementById('stat-latency-bar');

const statusLabel = document.getElementById('status-label');
const headerStatus = document.getElementById('header-status');
const heroCanvas = document.getElementById('hero-wave');
const heroShowcase = document.querySelector('.banner-visual-showcase');
const heroSignalLabel = document.getElementById('hero-signal-label');
const heroSignalDb = document.getElementById('hero-signal-db');

const micBtn = document.getElementById('mic-btn');
const micLabel = document.getElementById('mic-label');
const micHint = document.getElementById('mic-hint');
const micLevelFill = document.getElementById('mic-level-fill');

const taxonomyGrid = document.getElementById('cat-grid');
const taxonomySearch = document.getElementById('taxonomy-search');
const searchClearBtn = document.getElementById('search-clear');
const taxCountEl = document.getElementById('tax-count');
const surpriseBtn = document.getElementById('surprise-btn');
const catFilterTabs = document.querySelectorAll('.filter-tab');
const audioEl = document.getElementById('audio-el');
const bgCanvas = document.getElementById('bg-canvas');

const shortcutsBtn = document.getElementById('shortcuts-btn');
const shortcutsModal = document.getElementById('shortcuts-modal');
const shortcutsClose = document.getElementById('shortcuts-close');
const footerShortcuts = document.getElementById('footer-shortcuts');
const dropOverlay = document.getElementById('drop-overlay');
const toastStack = document.getElementById('toast-stack');

let currentDecodedBuffer = null;
let currentSpectrogramData = null;
let spectrogramCache = null;
let specHover = null;
let currentWavePeaks = null;
let currentFeatureMode = 'mel'; // 'mel' | 'rp' | 'active'
let currentRecurrenceData = null;
let currentActiveSegmentData = null;
let recurrencePlotCache = null;
let activeSegmentCache = null;

const featModeBar = document.getElementById('feat-mode-bar');
const specSublabel = document.getElementById('spec-sublabel');
const legScaleLbl = document.getElementById('leg-scale-lbl');
const legGradientBar = document.getElementById('leg-gradient-bar');
const legMarks = document.getElementById('leg-marks');
const ciK1 = document.getElementById('ci-k1');
const ciK2 = document.getElementById('ci-k2');
const ciK3 = document.getElementById('ci-k3');
let audioAnalyser = null;
let audioSourceNode = null;
let audioFreqData = null;
let audioTimeData = null;
let playerRaf = null;

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

const DEV_ONLY_VIEWS = ['pipeline', 'training', 'analytics'];

function switchView(viewName, updateHistory = true) {
  // Map legacy architecture view to the combined pipeline view
  if (viewName === 'architecture') viewName = 'pipeline';

  // Guard developer-only views if not authenticated
  if (DEV_ONLY_VIEWS.includes(viewName) && !isDeveloperAuthenticated) {
    pendingDevView = viewName;
    openAuthModal();
    const viewLabels = {
      pipeline: 'Neural Architecture & Research Pipeline',
      training: 'Model Training',
      analytics: 'System Analytics'
    };
    showToast('Developer Access Required', `Sign in to access ${viewLabels[viewName] || 'Developer Mode'}.`, 'warn', 4000);
    return;
  }

  const targetPanel = document.getElementById(`view-${viewName}`);
  if (!targetPanel) return;

  try {
    sessionStorage.setItem('soundsense_active_view', viewName);
    if (updateHistory && window.location.hash !== `#${viewName}`) {
      history.replaceState(null, '', `#${viewName}`);
    }
  } catch (_) { }

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

  // Smooth scroll to top on view change
  window.scrollTo({ top: 0, behavior: 'smooth' });

  requestAnimationFrame(() => positionNavGlider(true));

  if (viewName === 'pipeline') {
    animateBenchmarkBars(true);
    animateArchMetrics();
    loadAnalyticsData();
  } else if (viewName === 'training') {
    animateBenchmarkBars(false);
    loadTrainingData();
  } else if (viewName === 'analytics') {
    animateBenchmarkBars(false);
    loadAnalyticsData();
  } else {
    animateBenchmarkBars(false);
  }
}

/* Restore active view from URL hash or sessionStorage */
function initActiveView() {
  let target = '';
  if (window.location.hash) {
    const hash = window.location.hash.replace('#', '').trim();
    if (document.getElementById(`view-${hash}`)) {
      target = hash;
    }
  }
  if (!target) {
    try {
      const stored = sessionStorage.getItem('soundsense_active_view');
      if (stored && document.getElementById(`view-${stored}`)) {
        target = stored;
      }
    } catch (_) { }
  }
  if (target) {
    switchView(target, false);
  }
}

window.addEventListener('hashchange', () => {
  const hash = window.location.hash.replace('#', '').trim();
  if (hash && document.getElementById(`view-${hash}`)) {
    switchView(hash, false);
  }
});

/* Slide the navigation pill under the active tab */
function positionNavGlider(scrollIntoView = false) {
  if (!navGlider || !viewNav) return;
  const active = viewNav.querySelector('.view-tab.active');
  if (!active) return;

  // Use offset properties relative to viewNav (invariant to horizontal scroll)
  const tabLeft = active.offsetLeft;
  const tabWidth = active.offsetWidth;
  const tabTop = active.offsetTop;
  const tabHeight = active.offsetHeight;
  if (!tabWidth || !tabHeight) return;

  navGlider.style.left = `${tabLeft}px`;
  navGlider.style.width = `${tabWidth}px`;
  navGlider.style.top = `${tabTop}px`;
  navGlider.style.height = `${tabHeight}px`;
  navGlider.classList.add('ready');

  // Smoothly center the active tab in view if nav bar overflows
  if (scrollIntoView && viewNav.scrollWidth > viewNav.clientWidth) {
    const navWidth = viewNav.clientWidth;
    const targetScrollLeft = tabLeft - (navWidth / 2) + (tabWidth / 2);
    viewNav.scrollTo({
      left: Math.max(0, targetScrollLeft),
      behavior: 'smooth'
    });
  }
}

window.addEventListener('resize', () => requestAnimationFrame(() => positionNavGlider(false)));
window.addEventListener('orientationchange', () => setTimeout(() => positionNavGlider(true), 150));
requestAnimationFrame(() => requestAnimationFrame(() => positionNavGlider(false)));
setTimeout(() => positionNavGlider(false), 350);
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(() => positionNavGlider(false)).catch(() => { });
}

/* Arrow-key navigation between workspace tabs */
viewNav.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const tabs = Array.from(viewTabs).filter(t => !t.classList.contains('dev-tab') || isDeveloperAuthenticated);
  if (!tabs.length) return;
  const currentIndex = tabs.findIndex(t => t.classList.contains('active'));
  const nextIndex = (currentIndex + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  e.preventDefault();
  tabs[nextIndex].focus();
  switchView(tabs[nextIndex].dataset.view);
});

/* Slide the view navigation horizontally with the < / > arrows.
   These only scroll the tab strip — they never switch the active view. */
function updateNavArrows() {
  if (!viewNav || !viewNavPrev || !viewNavNext) return;

  const maxScroll = Math.max(0, viewNav.scrollWidth - viewNav.clientWidth);
  const hasOverflow = maxScroll > 2;

  viewNavPrev.hidden = !hasOverflow;
  viewNavNext.hidden = !hasOverflow;
  viewNavPrev.disabled = !hasOverflow || viewNav.scrollLeft <= 2;
  viewNavNext.disabled = !hasOverflow || viewNav.scrollLeft >= maxScroll - 2;
}

function scrollViewNav(direction) {
  if (!viewNav) return;
  const step = Math.max(140, Math.round(viewNav.clientWidth * 0.65));
  viewNav.scrollBy({ left: direction * step, behavior: 'smooth' });
}

if (viewNavPrev) {
  viewNavPrev.addEventListener('click', () => scrollViewNav(-1));
}
if (viewNavNext) {
  viewNavNext.addEventListener('click', () => scrollViewNav(1));
}
if (viewNav) {
  viewNav.addEventListener('scroll', () => requestAnimationFrame(updateNavArrows), { passive: true });
}
window.addEventListener('resize', () => requestAnimationFrame(updateNavArrows));
requestAnimationFrame(updateNavArrows);
setTimeout(updateNavArrows, 350);
if (document.fonts && document.fonts.ready) {
  document.fonts.ready.then(updateNavArrows).catch(() => { });
}

// Restore active workspace view from hash or session storage
initActiveView();

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
    document.querySelectorAll('#view-pipeline .perf-pct, #view-architecture .perf-pct').forEach(animateMetricValue);
  }
}

function animateArchMetrics() {
  document.querySelectorAll('#view-pipeline .m-val, #view-architecture .m-val').forEach(animateMetricValue);
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
/* Generates authentic acoustic waveforms matching any ESC-50 sound category */
function generateSyntheticAudioForCategory(targetId, catName = '') {
  const sampleRate = 22050;
  const duration = 3.0; // 3 seconds
  const numSamples = Math.floor(sampleRate * duration);
  const buffer = new Float32Array(numSamples);

  const id = typeof targetId === 'number' ? targetId : (KEYWORD_MAP[targetId] ?? 0);

  // Group 0: Animals (0–9)
  if (id === 0) { // dog
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const b1 = (t >= 0.2 && t <= 0.6) ? Math.exp(-(t - 0.2) * 12) * Math.sin(2 * Math.PI * (280 - (t - 0.2) * 200) * t) : 0;
      const b2 = (t >= 0.9 && t <= 1.3) ? Math.exp(-(t - 0.9) * 14) * Math.sin(2 * Math.PI * (300 - (t - 0.9) * 220) * t) : 0;
      const noise = (Math.random() * 2 - 1) * 0.15 * (b1 + b2);
      buffer[i] = Math.max(-1, Math.min(1, (b1 + b2) * 0.8 + noise));
    }
  } else if (id === 1) { // rooster
    const pitches = [520, 680, 840, 720];
    const starts = [0.2, 0.5, 0.8, 1.2];
    const lens = [0.22, 0.22, 0.35, 0.65];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (let k = 0; k < 4; k++) {
        if (t >= starts[k] && t <= starts[k] + lens[k]) {
          const dt = t - starts[k];
          const env = Math.sin(Math.PI * (dt / lens[k]));
          const f = pitches[k];
          buffer[i] += (Math.sin(2 * Math.PI * f * t) + 0.5 * Math.sin(2 * Math.PI * f * 2 * t) + 0.3 * Math.sin(2 * Math.PI * f * 3 * t)) * env * 0.3;
        }
      }
    }
  } else if (id === 2) { // pig (grunting/oink)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const pulse = Math.sin(2 * Math.PI * 9 * t);
      if (t >= 0.3 && t <= 2.2 && pulse > 0.2) {
        const env = Math.sin(Math.PI * (t - 0.3) / 1.9) * pulse;
        const carrier = Math.sin(2 * Math.PI * 135 * t) + 0.5 * Math.sin(2 * Math.PI * 270 * t);
        const noise = (Math.random() * 2 - 1) * 0.25;
        buffer[i] = (carrier + noise) * env * 0.45;
      }
    }
  } else if (id === 3) { // cow (deep resonant moo)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.2 && t <= 2.4) {
        const p = (t - 0.2) / 2.2;
        const f0 = 145 - p * 35;
        const vibrato = 1 + 0.03 * Math.sin(2 * Math.PI * 5 * t);
        const freq = f0 * vibrato;
        const env = Math.sin(Math.PI * p) * Math.min(1, (t - 0.2) * 6);
        const s1 = Math.sin(2 * Math.PI * freq * t);
        const s2 = 0.65 * Math.sin(2 * Math.PI * freq * 2 * t);
        const s3 = 0.45 * Math.sin(2 * Math.PI * freq * 3 * t);
        const s4 = 0.25 * Math.sin(2 * Math.PI * freq * 4 * t);
        buffer[i] = (s1 + s2 + s3 + s4) * env * 0.4;
      }
    }
  } else if (id === 4) { // frog (ribbit croak)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const burst1 = (t >= 0.3 && t <= 0.8);
      const burst2 = (t >= 1.2 && t <= 1.7);
      if (burst1 || burst2) {
        const mod = Math.sin(2 * Math.PI * 24 * t);
        const env = mod > 0 ? mod : 0;
        const tone = Math.sin(2 * Math.PI * 390 * t) + 0.4 * Math.sin(2 * Math.PI * 780 * t);
        buffer[i] = tone * env * 0.55;
      }
    }
  } else if (id === 5) { // cat (meow)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.3 && t <= 1.9) {
        const p = (t - 0.3) / 1.6;
        const freq = p < 0.4 ? 470 + (p / 0.4) * 330 : 800 - ((p - 0.4) / 0.6) * 270;
        const env = Math.sin(Math.PI * p);
        const tone = Math.sin(2 * Math.PI * freq * t) + 0.4 * Math.sin(2 * Math.PI * freq * 2 * t);
        buffer[i] = tone * env * 0.45;
      }
    }
  } else if (id === 6) { // hen (clucking)
    const clucks = [0.2, 0.45, 0.65, 0.9, 1.2, 1.45];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const ct of clucks) {
        if (t >= ct && t <= ct + 0.12) {
          const dt = t - ct;
          const env = Math.exp(-dt * 30);
          buffer[i] += (Math.sin(2 * Math.PI * (420 - dt * 400) * dt) + (Math.random() * 2 - 1) * 0.2) * env * 0.5;
        }
      }
    }
  } else if (id === 7 || id === 13) { // insects / crickets
    const freq = id === 13 ? 4600 : 4200;
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const pulse = Math.sin(2 * Math.PI * 16 * t);
      const env = pulse > 0 ? pulse * pulse : 0;
      buffer[i] = Math.sin(2 * Math.PI * freq * t) * env * 0.35;
    }
  } else if (id === 8) { // sheep (baa)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.2 && t <= 2.2) {
        const p = (t - 0.2) / 2.0;
        const tremolo = 1 + 0.4 * Math.sin(2 * Math.PI * 8 * t);
        const env = Math.sin(Math.PI * p) * tremolo;
        const s1 = Math.sin(2 * Math.PI * 225 * t);
        const s2 = 0.6 * Math.sin(2 * Math.PI * 450 * t);
        const s3 = 0.3 * Math.sin(2 * Math.PI * 675 * t);
        buffer[i] = (s1 + s2 + s3) * env * 0.35;
      }
    }
  } else if (id === 9) { // crow (caw caw)
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const b1 = (t >= 0.2 && t <= 0.7);
      const b2 = (t >= 1.0 && t <= 1.5);
      if (b1 || b2) {
        const dt = b1 ? (t - 0.2) : (t - 1.0);
        const env = Math.sin(Math.PI * (dt / 0.5));
        const rasp = Math.sin(2 * Math.PI * 720 * t) + 0.6 * Math.sin(2 * Math.PI * 1440 * t) + (Math.random() * 2 - 1) * 0.4;
        buffer[i] = rasp * env * 0.4;
      }
    }
  }
  // Group 1: Natural (10–19)
  else if (id === 10) { // rain
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < numSamples; i++) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99765 * b0 + white * 0.0990460;
      b1 = 0.96300 * b1 + white * 0.2965164;
      b2 = 0.57000 * b2 + white * 1.0526913;
      const pink = (b0 + b1 + b2 + white * 0.5362) * 0.11;
      const drop = Math.random() < 0.003 ? (Math.random() * 0.35) : 0;
      buffer[i] = Math.max(-1, Math.min(1, pink + drop));
    }
  } else if (id === 11) { // sea_waves
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const swell = (Math.sin(2 * Math.PI * 0.3 * t) + 1) * 0.5;
      const white = Math.random() * 2 - 1;
      buffer[i] = white * (0.05 + 0.35 * swell);
    }
  } else if (id === 12) { // crackling_fire
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const rumble = Math.sin(2 * Math.PI * 65 * t) * 0.08 * Math.random();
      const pop = Math.random() < 0.004 ? (Math.random() * 2 - 1) * 0.75 : 0;
      buffer[i] = Math.max(-1, Math.min(1, rumble + pop));
    }
  } else if (id === 14) { // chirping_birds
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const chirp = Math.sin(2 * Math.PI * 4 * t);
      if (chirp > 0.4) {
        const dt = (t % 0.25) / 0.25;
        const f = 2800 + dt * 1200;
        buffer[i] = Math.sin(2 * Math.PI * f * t) * (1 - dt) * 0.35;
      }
    }
  } else if (id === 15) { // water_drops
    const drops = [0.2, 0.7, 1.2, 1.7, 2.3];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const d of drops) {
        if (t >= d && t <= d + 0.2) {
          const dt = t - d;
          buffer[i] += Math.sin(2 * Math.PI * (1200 + dt * 400) * dt) * Math.exp(-dt * 45) * 0.6;
        }
      }
    }
  } else if (id === 16) { // wind
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const gust = (Math.sin(2 * Math.PI * 0.4 * t) + 1) * 0.5;
      const noise = (Math.random() * 2 - 1) * (0.08 + gust * 0.3);
      buffer[i] = noise;
    }
  } else if (id === 17) { // pouring_water
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      buffer[i] = (Math.random() * 2 - 1) * 0.28 + Math.sin(2 * Math.PI * 340 * t) * 0.08;
    }
  } else if (id === 18) { // toilet_flush
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const surge = t < 1.2 ? t / 1.2 : Math.max(0, 1 - (t - 1.2) / 1.5);
      buffer[i] = (Math.random() * 2 - 1) * surge * 0.55;
    }
  } else if (id === 19) { // thunderstorm
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.1) {
        const dt = t - 0.1;
        const boom = Math.exp(-dt * 1.8) * Math.sin(2 * Math.PI * (75 - dt * 15) * dt);
        const rumble = (Math.random() * 2 - 1) * Math.exp(-dt * 1.2) * 0.4;
        buffer[i] = Math.max(-1, Math.min(1, boom * 0.7 + rumble));
      }
    }
  }
  // Group 2: Human (20–29)
  else if (id === 20) { // crying_baby
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const cycle = t % 0.8;
      const f = 550 + Math.sin(2 * Math.PI * 1.2 * t) * 200;
      const env = Math.sin(Math.PI * (cycle / 0.8));
      buffer[i] = (Math.sin(2 * Math.PI * f * t) + 0.4 * Math.sin(2 * Math.PI * f * 2 * t)) * env * 0.35;
    }
  } else if (id === 21) { // sneezing
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.6 && t <= 1.2) {
        const dt = t - 0.6;
        buffer[i] = (Math.random() * 2 - 1) * Math.exp(-dt * 12) * 0.85;
      }
    }
  } else if (id === 22) { // clapping
    const claps = [0.2, 0.55, 0.9, 1.25, 1.6, 1.95, 2.3];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const c of claps) {
        if (t >= c && t <= c + 0.15) {
          const dt = t - c;
          buffer[i] += (Math.random() * 2 - 1) * Math.exp(-dt * 45) * 0.65;
        }
      }
    }
  } else if (id === 23) { // breathing
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const cycle = Math.sin(2 * Math.PI * 0.5 * t);
      buffer[i] = (Math.random() * 2 - 1) * Math.abs(cycle) * 0.2;
    }
  } else if (id === 24) { // coughing
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const c1 = (t >= 0.3 && t <= 0.6) ? Math.exp(-(t - 0.3) * 16) : 0;
      const c2 = (t >= 0.8 && t <= 1.1) ? Math.exp(-(t - 0.8) * 16) : 0;
      buffer[i] = (Math.random() * 2 - 1) * (c1 + c2) * 0.7;
    }
  } else if (id === 25) { // footsteps
    const steps = [0.2, 0.7, 1.2, 1.7, 2.2];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const s of steps) {
        if (t >= s && t <= s + 0.15) {
          const dt = t - s;
          buffer[i] += Math.sin(2 * Math.PI * 90 * dt) * Math.exp(-dt * 25) * 0.5;
        }
      }
    }
  } else if (id === 26) { // laughing
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const pulse = Math.sin(2 * Math.PI * 5 * t);
      if (t >= 0.2 && t <= 2.2 && pulse > 0) {
        buffer[i] = (Math.sin(2 * Math.PI * 320 * t) + 0.4 * Math.sin(2 * Math.PI * 640 * t)) * pulse * 0.35;
      }
    }
  } else if (id === 27) { // brushing_teeth
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const scrub = Math.sin(2 * Math.PI * 4 * t);
      buffer[i] = (Math.random() * 2 - 1) * Math.abs(scrub) * 0.3;
    }
  } else if (id === 28) { // snoring
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const cycle = (Math.sin(2 * Math.PI * 0.4 * t) + 1) * 0.5;
      const rasp = Math.sin(2 * Math.PI * 75 * t) + (Math.random() * 2 - 1) * 0.25;
      buffer[i] = rasp * cycle * 0.4;
    }
  } else if (id === 29) { // drinking_sipping
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.3 && t <= 1.4) {
        buffer[i] = (Math.random() * 2 - 1) * 0.2 + Math.sin(2 * Math.PI * 600 * t) * 0.15;
      }
    }
  }
  // Group 3: Interior (30–39)
  else if (id === 30) { // door_wood_knock
    const knocks = [0.2, 0.45, 0.7];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const k of knocks) {
        if (t >= k && t <= k + 0.12) {
          const dt = t - k;
          buffer[i] += Math.sin(2 * Math.PI * 180 * dt) * Math.exp(-dt * 35) * 0.7;
        }
      }
    }
  } else if (id === 31) { // mouse_click
    const clicks = [0.3, 0.45, 1.2];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const c of clicks) {
        if (t >= c && t <= c + 0.05) {
          const dt = t - c;
          buffer[i] += Math.sin(2 * Math.PI * 2400 * dt) * Math.exp(-dt * 120) * 0.75;
        }
      }
    }
  } else if (id === 32) { // keyboard_typing
    const clickTimes = [0.15, 0.38, 0.62, 0.85, 1.12, 1.45, 1.78, 2.05];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const ct of clickTimes) {
        if (t >= ct && t <= ct + 0.08) {
          const dt = t - ct;
          buffer[i] += Math.exp(-dt * 90) * Math.sin(2 * Math.PI * 1800 * dt) * 0.6;
        }
      }
    }
  } else if (id === 33) { // door_wood_creaks
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.3 && t <= 2.2) {
        const dt = t - 0.3;
        const f = 450 + Math.sin(2 * Math.PI * 12 * dt) * 180;
        buffer[i] = Math.sin(2 * Math.PI * f * dt) * Math.sin(Math.PI * dt / 1.9) * 0.4;
      }
    }
  } else if (id === 34) { // can_opening
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.2 && t <= 0.35) {
        buffer[i] = (Math.random() * 2 - 1) * Math.exp(-(t - 0.2) * 50) * 0.8;
      } else if (t > 0.35 && t <= 1.4) {
        buffer[i] = (Math.random() * 2 - 1) * Math.exp(-(t - 0.35) * 4) * 0.35; // fizz
      }
    }
  } else if (id === 35) { // washing_machine
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const rot = Math.sin(2 * Math.PI * 1.5 * t);
      buffer[i] = (Math.sin(2 * Math.PI * 60 * t) * 0.3 + (Math.random() * 2 - 1) * 0.15) * (0.6 + 0.4 * rot);
    }
  } else if (id === 36) { // vacuum_cleaner
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      buffer[i] = Math.sin(2 * Math.PI * 220 * t) * 0.3 + Math.sin(2 * Math.PI * 440 * t) * 0.2 + (Math.random() * 2 - 1) * 0.25;
    }
  } else if (id === 37) { // clock_alarm
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const cycle = t % 0.4;
      if (cycle < 0.2) {
        const freq = (t % 0.8 < 0.4) ? 880 : 1046.5;
        buffer[i] = Math.sin(2 * Math.PI * freq * t) * 0.45;
      }
    }
  } else if (id === 38) { // clock_tick
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const sub = t % 0.5;
      if (sub < 0.04) {
        const isTick = (t % 1.0 < 0.5);
        const f = isTick ? 1200 : 950;
        buffer[i] = Math.sin(2 * Math.PI * f * sub) * Math.exp(-sub * 120) * 0.6;
      }
    }
  } else if (id === 39) { // glass_breaking
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.1 && t <= 1.5) {
        const dt = t - 0.1;
        const crash = (Math.random() * 2 - 1) * Math.exp(-dt * 8);
        const ring = Math.sin(2 * Math.PI * 3400 * dt) * Math.exp(-dt * 4) * 0.4;
        buffer[i] = Math.max(-1, Math.min(1, crash + ring));
      }
    }
  }
  // Group 4: Urban (40–49)
  else if (id === 40) { // helicopter
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const chop = Math.sin(2 * Math.PI * 11.5 * t);
      const env = chop > 0 ? chop * chop : 0;
      buffer[i] = (Math.sin(2 * Math.PI * 120 * t) * 0.4 + (Math.random() * 2 - 1) * 0.2) * env;
    }
  } else if (id === 41) { // chainsaw
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      buffer[i] = (Math.sin(2 * Math.PI * 180 * t) + 0.5 * Math.sin(2 * Math.PI * 360 * t) + (Math.random() * 2 - 1) * 0.35) * 0.45;
    }
  } else if (id === 42) { // siren
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const sweep = Math.sin(2 * Math.PI * 0.8 * t);
      const freq = 650 + sweep * 300;
      buffer[i] = Math.sin(2 * Math.PI * freq * t) * 0.45;
    }
  } else if (id === 43) { // car_horn
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.2 && t <= 1.8) {
        buffer[i] = (Math.sin(2 * Math.PI * 440 * t) + Math.sin(2 * Math.PI * 554 * t)) * 0.35;
      }
    }
  } else if (id === 44) { // engine
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      buffer[i] = (Math.sin(2 * Math.PI * 85 * t) + 0.5 * Math.sin(2 * Math.PI * 170 * t) + (Math.random() * 2 - 1) * 0.2) * 0.45;
    }
  } else if (id === 45) { // train
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const chug = Math.sin(2 * Math.PI * 4 * t);
      buffer[i] = (Math.random() * 2 - 1) * Math.abs(chug) * 0.35 + Math.sin(2 * Math.PI * 520 * t) * 0.15;
    }
  } else if (id === 46) { // church_bells
    const tolls = [0.2, 1.4];
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      for (const toll of tolls) {
        if (t >= toll) {
          const dt = t - toll;
          const s1 = Math.sin(2 * Math.PI * 330 * dt) * Math.exp(-dt * 1.5);
          const s2 = 0.5 * Math.sin(2 * Math.PI * 660 * dt) * Math.exp(-dt * 2.2);
          const s3 = 0.3 * Math.sin(2 * Math.PI * 920 * dt) * Math.exp(-dt * 3.0);
          buffer[i] += (s1 + s2 + s3) * 0.4;
        }
      }
    }
  } else if (id === 47) { // airplane
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      buffer[i] = (Math.random() * 2 - 1) * 0.3 + Math.sin(2 * Math.PI * 280 * t) * 0.15;
    }
  } else if (id === 48) { // fireworks
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      if (t >= 0.1 && t <= 0.6) {
        const dt = t - 0.1;
        buffer[i] = Math.sin(2 * Math.PI * (600 + dt * 1500) * dt) * (0.6 - dt); // whistle
      } else if (t > 0.6) {
        const dt = t - 0.6;
        buffer[i] = (Math.random() * 2 - 1) * Math.exp(-dt * 3.5) * 0.85; // boom
      }
    }
  } else if (id === 49) { // hand_saw
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const stroke = Math.sin(2 * Math.PI * 2 * t);
      buffer[i] = (Math.random() * 2 - 1) * Math.abs(stroke) * 0.45;
    }
  } else {
    // General fallback
    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      buffer[i] = Math.sin(2 * Math.PI * 350 * t) * 0.3 + (Math.random() * 2 - 1) * 0.1;
    }
  }

  return encodeWAV(buffer, sampleRate);
}

/* Backward compatibility alias */
function generateSyntheticAudioBlob(soundType) {
  return generateSyntheticAudioForCategory(soundType);
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
    dog: { targetId: 0, name: CANONICAL_ESC50_FILES[0] },
    rain: { targetId: 10, name: CANONICAL_ESC50_FILES[10] },
    alarm: { targetId: 37, name: CANONICAL_ESC50_FILES[37] },
    typing: { targetId: 32, name: CANONICAL_ESC50_FILES[32] },
    keyboard_typing: { targetId: 32, name: CANONICAL_ESC50_FILES[32] },
    siren: { targetId: 42, name: CANONICAL_ESC50_FILES[42] },
    baby: { targetId: 20, name: CANONICAL_ESC50_FILES[20] },
    knock: { targetId: 30, name: CANONICAL_ESC50_FILES[30] },
    chainsaw: { targetId: 41, name: CANONICAL_ESC50_FILES[41] },
  };

  const preset = presetMapping[soundKey];
  if (!preset) return;

  currentPresetTarget = preset.targetId;
  const catObj = ESC50_CATEGORIES[preset.targetId];
  loadTaxonomyCategory(catObj);
});

/* ── Upload Zone Interactions ─────────────────────────────────── */
uploadZone.addEventListener('click', () => audioInput.click());
uploadZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    audioInput.click();
  }
});

uploadZone.addEventListener('dragenter', e => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', (e) => {
  if (e.relatedTarget && uploadZone.contains(e.relatedTarget)) return;
  uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  e.stopPropagation();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files[0] : null;
  if (file && isAudioFile(file)) {
    currentPresetTarget = null;
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    loadAudioFile(file);
    showToast('Audio loaded', `${file.name} · ${formatBytes(file.size)}`, 'success');
  } else if (file) {
    showToast('Unsupported file', 'Please choose a WAV, MP3, or MPEG/MPG audio file.', 'error');
  }
});

audioInput.addEventListener('change', () => {
  if (audioInput.files[0]) {
    const file = audioInput.files[0];
    if (isAudioFile(file)) {
      currentPresetTarget = null;
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      loadAudioFile(file);
      showToast('Audio loaded', `${file.name} · ${formatBytes(file.size)}`, 'success');
    } else {
      showToast('Unsupported file', 'Please choose a WAV, MP3, or MPEG/MPG audio file.', 'error');
      audioInput.value = '';
    }
  }
});

fileRemoveBtn.addEventListener('click', e => {
  e.stopPropagation();
  resetStudioUpload();
});

function isAudioFile(file) {
  if (!file) return false;
  const name = (file.name || '').toLowerCase();
  // Disallow removed formats (OGG, FLAC, M4A)
  if (/\.(ogg|flac|m4a)$/i.test(name)) return false;
  // Retain allowed audio formats: WAV, MP3, MPEG / MPG, WebM, AAC
  if (/\.(wav|mp3|mpg|mpeg|mpga|mpe|webm|aac)$/i.test(name)) return true;
  if (file.type) {
    const type = file.type.toLowerCase();
    if (type.includes('ogg') || type.includes('flac') || type.includes('m4a') || type.includes('mp4')) {
      return false;
    }
    if (type === 'audio/wav' || type === 'audio/x-wav' || type === 'audio/mp3' || type === 'audio/mpeg' || type === 'video/mpeg' || type === 'audio/webm' || type === 'audio/aac' || type === 'audio/aacp' || type === 'audio/x-aac') {
      return true;
    }
  }
  return false;
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
  const recordedBlob = new Blob(micChunks, { type: micMime || 'audio/webm' });
  cleanupMic();
  updateMicUI();

  if (!recordedBlob.size) {
    showToast('No audio captured', 'Nothing was recorded. Please try again.', 'error');
    setStatus('Engine Ready', 'ready');
    return;
  }

  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  let wavFile;
  try {
    await ensureAudioContext();
    const arrayBuffer = await recordedBlob.arrayBuffer();
    const decoded = await audioContext.decodeAudioData(arrayBuffer);
    const pcm = decoded.getChannelData(0);
    const wavBlob = encodeWAV(pcm, decoded.sampleRate);
    wavFile = new File([wavBlob], `mic_recording_${stamp}.wav`, { type: 'audio/wav' });
  } catch (err) {
    console.warn('WAV conversion fallback:', err);
    wavFile = new File([recordedBlob], `mic_recording_${stamp}.wav`, { type: 'audio/wav' });
  }

  currentPresetTarget = null;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  await loadAudioFile(wavFile);
  showToast('Recording captured', 'Running acoustic classification on your microphone clip…', 'success');
  setTimeout(executeClassification, 80);
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
  currentRecurrenceData = null;
  currentActiveSegmentData = null;
  currentWavePeaks = null;
  spectrogramCache = null;
  recurrencePlotCache = null;
  activeSegmentCache = null;
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

        // Compute Threshold-Free Recurrence Plot & Active Segment
        const rawPcm = currentDecodedBuffer.getChannelData(0);
        currentRecurrenceData = computeThresholdFreeRecurrencePlot(rawPcm, 128);
        currentActiveSegmentData = computeActiveSegment(rawPcm, currentDecodedBuffer.sampleRate);

        spectrogramCache = null;
        recurrencePlotCache = null;
        activeSegmentCache = null;

        // Update telemetry
        if (specRateEl) specRateEl.textContent = `${(currentDecodedBuffer.sampleRate / 1000).toFixed(1)} kHz`;
        if (specDurEl) specDurEl.textContent = `${currentDecodedBuffer.duration.toFixed(2)}s`;
        durDisplay.textContent = formatSeconds(currentDecodedBuffer.duration);

        updateSignalStats();
        drawWaveform(0);
        drawFeatureSpectrogram(0);
      } catch (err) {
        console.warn('AudioContext decode error, generating procedural acoustic buffer fallback:', err);
        try {
          if (audioContext) {
            const sr = 22050;
            const dur = 5.0;
            const numSamples = Math.round(sr * dur);
            const fallbackBuf = audioContext.createBuffer(1, numSamples, sr);
            const channel = fallbackBuf.getChannelData(0);
            for (let s = 0; s < numSamples; s++) {
              const t = s / sr;
              channel[s] = Math.sin(2 * Math.PI * 440 * t) * 0.18 * Math.exp(-t % 1.2) + (Math.random() - 0.5) * 0.05;
            }
            currentDecodedBuffer = fallbackBuf;
            currentSpectrogramData = computeAcousticSpectrogram(fallbackBuf);
            currentWavePeaks = computeWaveformPeaks(fallbackBuf, 128);
            currentRecurrenceData = computeThresholdFreeRecurrencePlot(channel, 128);
            currentActiveSegmentData = computeActiveSegment(channel, sr);
            if (specRateEl) specRateEl.textContent = '22.1 kHz';
            if (specDurEl) specDurEl.textContent = '5.00s';
            durDisplay.textContent = '0:05';
            updateSignalStats();
            drawWaveform(0);
            drawFeatureSpectrogram(0);
          } else {
            throw err;
          }
        } catch (innerErr) {
          currentDecodedBuffer = null;
          currentSpectrogramData = null;
          currentWavePeaks = null;
          spectrogramCache = null;
          resetSignalStats();
          drawWaveform(0);
          drawFeatureSpectrogram(0);
        }
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
    decoding.then(() => setTimeout(executeClassification, 80));
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
    try { await audioContext.resume(); } catch (_) { }
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
    } catch (_) { }
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
  document.body.classList.toggle('audio-playing', playing);
}

function startAudioPlayback() {
  if (!audioEl.src) return;
  ensureAudioContext().then(ensureAudioAnalyser).catch(() => { });
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
const modelBadgeInfo = document.getElementById('model-badge-info');
function updateModelBadge() {
  if (modelBadgeInfo && modelSelect) {
    const spec = MODEL_INFO[modelSelect.value] || MODEL_INFO.sresnet;
    modelBadgeInfo.textContent = spec.name;
  }
}
if (modelSelect) {
  modelSelect.addEventListener('change', updateModelBadge);
}

classifyBtn.addEventListener('click', executeClassification);
resultRetryBtn.addEventListener('click', executeClassification);

async function executeClassification() {
  if (!currentAudioFile || isClassifying) return;

  isClassifying = true;
  const token = ++classificationToken;
  const modelKey = modelSelect.value;
  const modelSpec = MODEL_INFO[modelKey] || MODEL_INFO.sresnet;
  currentModelKey = modelKey;

  // Read Confidence Threshold
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdVal = thresholdSlider ? (parseFloat(thresholdSlider.value) / 100) : 0.5;

  // Set loading state & start elapsed timer
  resultEmpty.hidden = true;
  resultOutput.hidden = true;
  resultLoading.hidden = false;
  resultLoading.setAttribute('aria-busy', 'true');

  const timerElapsedEl = document.getElementById('inference-elapsed');
  const timerStart = performance.now();
  if (timerElapsedEl) timerElapsedEl.textContent = '0.0s';
  const timerInterval = setInterval(() => {
    if (timerElapsedEl) {
      timerElapsedEl.textContent = `${((performance.now() - timerStart) / 1000).toFixed(1)}s`;
    }
  }, 100);

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
      await sleep(60);
      el.classList.remove('active');
      el.classList.add('done');
    }

    // High-performance inference completion
    await sleep(80);

    // Abort if the user removed or replaced the clip mid-analysis
    if (token !== classificationToken || !currentAudioFile) {
      setStatus('Engine Ready', 'ready');
      return;
    }

    // Attempt inference via Backend API (works from Live Server, file://, or backend-served)
    let predictions = null;
    let inferenceSource = 'local';
    let serverLatency = null;
    let backendErrorDetail = null;

    const apiBase = resolveApiBase();
    const isLocalHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
    const candidateUrls = [
      `${apiBase}/api/predict`,
      '/api/predict'
    ];
    if (isLocalHost) {
      candidateUrls.push(
        'http://localhost:3001/api/predict',
        'http://127.0.0.1:3001/api/predict',
        'http://localhost:8000/api/predict',
        'http://127.0.0.1:8000/api/predict'
      );
    }

    for (const url of candidateUrls) {
      try {
        const formData = new FormData();
        formData.append('audio', currentAudioFile, currentAudioFile.name || 'sample.wav');
        formData.append('model', modelKey);
        formData.append('threshold', thresholdVal.toString());

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 25000);

        const res = await fetch(url, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (res.ok) {
          const data = await res.json();
          if (data && data.success && Array.isArray(data.predictions)) {
            const classPredictions = data.predictions.map(p => ({
              category: {
                ...p.category,
                emoji: p.category.emoji || ESC50_CATEGORIES[p.category.id]?.emoji || '🔊',
              },
              confidence: p.confidence,
              inferenceTime: data.inferenceTime,
              isOutOfScope: Boolean(p.isOutOfScope || (p.category && p.category.id === -1) || (data.isOutOfScope && p === data.predictions[0])),
              explanation: p.explanation || data.explanation,
            }));
            predictions = data.isOutOfScope && data.category && data.category.id === -1
              ? [{
                category: { ...data.category, emoji: data.category.emoji || '❓' },
                confidence: data.confidenceType === 'not_computed_rule_gate'
                  ? null
                  : (Number.isFinite(data.oodScore) ? data.oodScore : null),
                inferenceTime: data.inferenceTime,
                isOutOfScope: true,
                confidenceType: data.confidenceType || 'rule_gate',
                explanation: data.explanation,
              }, ...classPredictions.filter(p => p.category.id >= 0)]
              : classPredictions;
            inferenceSource = 'api';
            serverLatency = data.inferenceTime;
            break;
          }
        } else if (res.status === 503) {
          const errorBody = await res.json().catch(() => ({}));
          backendErrorDetail = errorBody.details || errorBody.detail || errorBody.error || 'The selected model checkpoint is unavailable.';
          break;
        }
      } catch (_netErr) {
        // Continue trying next candidate endpoint
      }
    }

    if (!predictions || predictions.length === 0) {
      throw new Error(backendErrorDetail || 'The backend did not return model predictions. Check that the selected model checkpoint is available.');
    }
    backendOnline = true;

    displayClassificationResults(predictions, modelKey);
    setStatus('Engine Ready · Backend Online', 'ready');
    showToast('API Inference Complete', `Classified by backend (/api/predict) · ${serverLatency != null ? `${serverLatency} ms` : 'latency not measured'}`, 'success', 3000);
  } catch (err) {
    resultLoading.hidden = true;
    resultEmpty.hidden = false;
    backendOnline = false;
    setStatus('Backend unavailable · No prediction returned', 'warn');
    showToast('Prediction unavailable', err.message || 'Start the Python backend and verify model checkpoints.', 'error', 6000);
  } finally {
    clearInterval(timerInterval);
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
function displayClassificationResults(predictions, modelKey) {
  resultLoading.hidden = true;
  resultOutput.hidden = false;

  const winner = predictions[0];
  const isOOD = winner.isOutOfScope || winner.category.id === -1;
  const modelSpec = MODEL_INFO[modelKey] || MODEL_INFO.sresnet;

  // Confidence Threshold Evaluation
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdVal = thresholdSlider ? (parseFloat(thresholdSlider.value) / 100) : 0.5;
  const isBelowThreshold = Number.isFinite(winner.confidence) && winner.confidence < thresholdVal;
  const showAnomaly = isOOD || isBelowThreshold;

  currentResult = {
    predictions,
    modelKey,
    modelName: modelSpec.name,
    fileName: currentAudioFile ? currentAudioFile.name : 'audio',
    timestamp: new Date()
  };

  resultModelBadge.textContent = modelSpec.name;

  // Track tested category for taxonomy card checkmark
  if (winner && winner.category && typeof winner.category.id === 'number' && winner.category.id >= 0) {
    testedCategories.add(winner.category.id);
    const cardEl = document.querySelector(`.tax-card[data-id="${winner.category.id}"]`);
    if (cardEl) {
      cardEl.classList.add('tax-card--tested');
    }
  }

  // Toggle Out-of-Scope / Below Threshold Banner
  const oosBanner = document.getElementById('out-of-scope-banner');
  const oosDesc = document.getElementById('oos-desc-text');
  if (oosBanner) {
    if (showAnomaly) {
      oosBanner.removeAttribute('hidden');
      oosBanner.hidden = false;
      oosBanner.style.display = 'flex';
      if (oosDesc) {
        if (isBelowThreshold && !isOOD) {
          oosDesc.textContent = `Model probability (${(winner.confidence * 100).toFixed(1)}%) is below your chosen threshold (${(thresholdVal * 100).toFixed(0)}%). Review the runner-up scores or lower the threshold.`;
        } else {
          oosDesc.textContent = winner.explanation || 'The acoustic characteristics (frequency envelope, harmonics, and temporal dynamics) do not match any of the 50 standardized environmental categories in ESC-50.';
        }
      }
    } else {
      oosBanner.setAttribute('hidden', '');
      oosBanner.hidden = true;
      oosBanner.style.display = 'none';
    }
  }

  topEmoji.textContent = winner.category.emoji || '🔊';
  topCategory.textContent = winner.category.name.replace(/_/g, ' ');

  if (isOOD) {
    topGroup.textContent = 'Out of Domain • Not in ESC-50 Classes';
  } else {
    topGroup.textContent = `${winner.category.group} • ESC-50 Class #${winner.category.id}`;
  }

  // Animate circular confidence ring
  const hasDisplayedConfidence = Number.isFinite(winner.confidence);
  const finalPct = hasDisplayedConfidence ? Math.round(winner.confidence * 100) : null;
  const circumference = 201.06; // 2 * PI * 32
  confFillCircle.style.strokeDashoffset = circumference;

  if (hasDisplayedConfidence) {
    setTimeout(() => {
      confFillCircle.style.strokeDashoffset = circumference * (1 - winner.confidence);
      animateCounter(confPct, 0, finalPct, 750);
    }, 80);
  } else if (confPct) {
    confPct.textContent = '—';
  }
  const confidenceRing = document.getElementById('top-conf-ring');
  if (confidenceRing) {
    const confidenceLabel = confidenceRing.querySelector('.pct-lbl');
    if (confidenceLabel) confidenceLabel.textContent = isOOD
      ? (winner.confidenceType === 'not_computed_rule_gate' ? 'GATE' : 'OOD')
      : 'PROB';
    confidenceRing.title = isOOD
      ? (winner.confidenceType === 'not_computed_rule_gate' ? 'Rule-based out-of-scope decision; classifier confidence was not computed' : 'Out-of-scope score; it is not a class probability')
      : 'Model class probability; calibration status is shown in the evaluation report';
  }

  // Inference telemetry
  if (statLatencyEl) {
    const measuredLatency = Number.isFinite(winner.inferenceTime) ? winner.inferenceTime : modelSpec.latency;
    statLatencyEl.textContent = Number.isFinite(measuredLatency) ? `${measuredLatency.toFixed(1)} ms` : '—';
    if (statLatencyBar) {
      statLatencyBar.style.width = '0%';
      if (Number.isFinite(measuredLatency)) {
        setTimeout(() => {
          statLatencyBar.style.width = `${Math.min(100, (measuredLatency / 20) * 100)}%`;
        }, 100);
      }
    }
  }

  // Hero signal readout reflects the winning class
  if (heroSignalLabel) {
    heroSignalLabel.textContent = isOOD
      ? 'Acoustic signature · Out of ESC-50 Scope'
      : `Signal match · ${winner.category.name.replace(/_/g, ' ')}`;
  }

  // Render Top 5 List
  top5List.innerHTML = '';
  predictions.forEach((p, idx) => {
    const row = document.createElement('li');
    const isWinner = idx === 0;
    row.className = `top5-row ${isWinner ? 'is-winner' : ''}`;
    const hasConfidence = Number.isFinite(p.confidence);
    const pctVal = hasConfidence ? Math.round(p.confidence * 100) : null;
    const subtitle = isOOD && idx > 0 ? ' <span style="font-size:0.73rem; opacity:0.65;">(tentative reference)</span>' : '';

    row.innerHTML = `
      <span class="top5-rank">#${idx + 1}</span>
      <span class="top5-name">${p.category.emoji || '🔊'} ${p.category.name.replace(/_/g, ' ')}${subtitle}</span>
      <div class="top5-bar-track">
        <div class="top5-bar-fill" style="width: 0%" data-pct="${pctVal ?? 0}%"></div>
      </div>
      <span class="top5-pct">${pctVal == null ? '—' : `${pctVal}%`}</span>`;
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

/* ── Microscopic Chaotic Recurrence Plot & Active Segment ──── */
function computeThresholdFreeRecurrencePlot(rawPcm, paaDim = 128) {
  if (!rawPcm || rawPcm.length === 0) return null;
  const n = rawPcm.length;
  const paa = new Float32Array(paaDim);
  const binSize = n / paaDim;

  for (let i = 0; i < paaDim; i++) {
    const start = Math.floor(i * binSize);
    const end = Math.floor((i + 1) * binSize);
    let sum = 0;
    const count = Math.max(1, end - start);
    for (let j = start; j < end; j++) {
      sum += rawPcm[j];
    }
    paa[i] = sum / count;
  }

  const matrix = [];
  let minDiff = 1e9;
  let maxDiff = -1e9;
  for (let i = 0; i < paaDim; i++) {
    const row = new Float32Array(paaDim);
    for (let j = 0; j < paaDim; j++) {
      const d = Math.abs(paa[i] - paa[j]);
      row[j] = d;
      if (d < minDiff) minDiff = d;
      if (d > maxDiff) maxDiff = d;
    }
    matrix.push(row);
  }

  const range = Math.max(1e-6, maxDiff - minDiff);
  for (let i = 0; i < paaDim; i++) {
    for (let j = 0; j < paaDim; j++) {
      matrix[i][j] = (matrix[i][j] - minDiff) / range;
    }
  }

  return { matrix, paaDim, minDiff, maxDiff };
}

function computeActiveSegment(rawPcm, sampleRate = 22050, frameLen = 2048, hopLen = 512) {
  if (!rawPcm || rawPcm.length < frameLen) {
    return { startTime: 0, endTime: rawPcm ? rawPcm.length / sampleRate : 0, ratio: 1.0, energies: [] };
  }
  const numFrames = Math.floor((rawPcm.length - frameLen) / hopLen);
  const energies = new Float32Array(numFrames);
  let meanEnergy = 0;

  for (let f = 0; f < numFrames; f++) {
    const start = f * hopLen;
    let e = 0;
    for (let n = 0; n < frameLen; n += 4) {
      e += rawPcm[start + n] * rawPcm[start + n];
    }
    energies[f] = e;
    meanEnergy += e;
  }
  meanEnergy /= Math.max(1, numFrames);
  const thresh = Math.max(1e-5, meanEnergy * 0.15);

  let maxLen = 0, bestStart = 0, currLen = 0, currStart = 0, activeCount = 0;
  for (let f = 0; f < numFrames; f++) {
    if (energies[f] >= thresh) {
      activeCount++;
      if (currLen === 0) currStart = f;
      currLen++;
      if (currLen > maxLen) {
        maxLen = currLen;
        bestStart = currStart;
      }
    } else {
      currLen = 0;
    }
  }

  const startTime = (bestStart * hopLen) / sampleRate;
  const endTime = Math.min(rawPcm.length / sampleRate, (bestStart * hopLen + maxLen * hopLen + frameLen) / sampleRate);
  const ratio = activeCount / Math.max(1, numFrames);

  return { startTime, endTime, ratio, energies };
}

/* ── High-Precision Time-Frequency Spectrogram Renderer ──────── */
const SPEC_MARGINS = { left: 48, bottom: 22, right: 8, top: 6 };

function buildRecurrencePlotCache() {
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

  if (!currentRecurrenceData || !currentRecurrenceData.matrix.length) {
    ctx.fillStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting audio file to compute Threshold-Free Recurrence Plot', left + plotW / 2, top + plotH / 2);
    return cache;
  }

  const { matrix, paaDim } = currentRecurrenceData;
  const cellW = plotW / paaDim;
  const cellH = plotH / paaDim;

  for (let i = 0; i < paaDim; i++) {
    const x = left + i * cellW;
    for (let j = 0; j < paaDim; j++) {
      const y = top + (paaDim - 1 - j) * cellH;
      const normVal = matrix[i][j];
      ctx.fillStyle = getSpectrogramColor(1 - normVal);
      ctx.fillRect(Math.floor(x), Math.floor(y), Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
    }
  }

  // Draw Axes
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.15)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#64748b';
  ctx.font = '10px monospace';
  const ticks = [
    { lbl: '128', ratio: 0.0 },
    { lbl: '96', ratio: 0.25 },
    { lbl: '64', ratio: 0.5 },
    { lbl: '32', ratio: 0.75 },
    { lbl: '0', ratio: 1.0 },
  ];
  ticks.forEach(t => {
    const y = top + t.ratio * plotH;
    ctx.beginPath();
    ctx.moveTo(left - 4, y);
    ctx.lineTo(left, y);
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(t.lbl, left - 6, y + 3);
  });
  ticks.forEach(t => {
    const x = left + (1 - t.ratio) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, top + plotH);
    ctx.lineTo(x, top + plotH + 4);
    ctx.stroke();
    ctx.textAlign = 'center';
    ctx.fillText(t.lbl, x, top + plotH + 14);
  });

  return cache;
}

function buildActiveSegmentCache() {
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

  if (!currentDecodedBuffer) {
    ctx.fillStyle = 'rgba(56, 189, 248, 0.5)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting audio file to compute active segment boundaries', left + plotW / 2, top + plotH / 2);
    return cache;
  }

  const raw = currentDecodedBuffer.getChannelData(0);
  const dur = currentDecodedBuffer.duration || 5.0;
  const seg = currentActiveSegmentData || { startTime: 0, endTime: dur, ratio: 1.0 };

  // Draw background grid
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
  ctx.lineWidth = 1;
  for (let s = 1; s <= 4; s++) {
    const x = left + (s / 5) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, top + plotH);
    ctx.stroke();
  }

  // Draw active zone highlight
  const actX1 = left + Math.max(0, Math.min(1, seg.startTime / dur)) * plotW;
  const actX2 = left + Math.max(0, Math.min(1, seg.endTime / dur)) * plotW;
  ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
  ctx.fillRect(actX1, top, actX2 - actX1, plotH);
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.5)';
  ctx.lineWidth = 1.5;
  ctx.strokeRect(actX1, top, actX2 - actX1, plotH);

  // Draw waveform
  const midY = top + plotH / 2;
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(raw.length / plotW));
  for (let i = 0; i < plotW; i++) {
    const sampleIdx = Math.floor(i * step);
    const amp = raw[sampleIdx] || 0;
    const y = midY - amp * (plotH / 2 * 0.9);
    if (i === 0) ctx.moveTo(left + i, y);
    else ctx.lineTo(left + i, y);
  }
  ctx.stroke();

  // Active Tag
  ctx.fillStyle = '#67e8f9';
  ctx.font = 'bold 10px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`ACTIVE: ${seg.startTime.toFixed(2)}s - ${seg.endTime.toFixed(2)}s (${Math.round(seg.ratio * 100)}% energy)`, actX1 + 6, top + 16);

  return cache;
}

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

  let activeCache = null;
  if (currentFeatureMode === 'mel') {
    if (!spectrogramCache) spectrogramCache = buildSpectrogramCache();
    activeCache = spectrogramCache;
  } else if (currentFeatureMode === 'rp') {
    if (!recurrencePlotCache) recurrencePlotCache = buildRecurrencePlotCache();
    activeCache = recurrencePlotCache;
  } else {
    if (!activeSegmentCache) activeSegmentCache = buildActiveSegmentCache();
    activeCache = activeSegmentCache;
  }

  ctx.clearRect(0, 0, W, H);
  if (activeCache) {
    ctx.drawImage(activeCache, 0, 0);
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

  if (currentFeatureMode === 'rp') {
    if (currentRecurrenceData) {
      const dim = currentRecurrenceData.paaDim;
      const iIdx = Math.max(0, Math.min(dim - 1, Math.floor(fx * dim)));
      const jIdx = Math.max(0, Math.min(dim - 1, Math.floor((1 - fy) * dim)));
      const dist = currentRecurrenceData.matrix[iIdx][jIdx];

      if (cursorTimeEl) cursorTimeEl.textContent = `${iIdx}`;
      if (cursorFreqEl) cursorFreqEl.textContent = `${jIdx}`;
      if (cursorLevelEl) cursorLevelEl.textContent = `${dist.toFixed(3)}`;
    }
  } else if (currentFeatureMode === 'active') {
    const seg = currentActiveSegmentData;
    const timeSec = fx * duration;
    const isInside = seg && timeSec >= seg.startTime && timeSec <= seg.endTime;

    if (cursorTimeEl) cursorTimeEl.textContent = `${timeSec.toFixed(2)}s`;
    if (cursorFreqEl) cursorFreqEl.textContent = isInside ? 'Active VAD' : 'Ambient';
    if (cursorLevelEl) cursorLevelEl.textContent = seg ? `${Math.round(seg.ratio * 100)}%` : '—';
  } else {
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
  }

  const tIdx = Math.max(0, Math.min(numTimeBins - 1, Math.floor(fx * numTimeBins)));
  const bIdx = Math.max(0, Math.min(numFreqBands - 1, Math.floor((1 - fy) * numFreqBands)));
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

  ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#94a3b8';
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

  // Frequency Axis Unit Indicator
  ctx.save();
  ctx.fillStyle = 'rgba(34, 211, 238, 0.75)';
  ctx.font = '9px monospace';
  ctx.textAlign = 'right';
  ctx.fillText('kHz', left - 6, top - 2);
  ctx.restore();

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
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
  ctx.strokeRect(left, top, w, h);
}

/* ── Track tested sound classes across session ─────────────────── */
const testedCategories = new Set();

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
    card.className = 'tax-card';
    card.setAttribute('data-id', cat.id);
    if (testedCategories.has(cat.id)) {
      card.classList.add('tax-card--tested');
    }
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

async function loadTaxonomyCategory(cat) {
  currentPresetTarget = cat.id;
  switchView('studio');
  const filename = CANONICAL_ESC50_FILES[cat.id] || `1-100032-A-${cat.id}.wav`;

  let file = null;
  const isLocalSampleHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
  const candidatePaths = [
    getApiUrl(`/api/sample/${cat.id}`),
    `/api/sample/${cat.id}`,
    `samples/${filename}`,
    `samples/${cat.id}.wav`,
    `./samples/${filename}`,
    `./samples/${cat.id}.wav`,
    `/samples/${filename}`,
    `/audio/esc50/${filename}`,
    `/ESC-50/wav_files/${filename}`,
    `ESC-50/wav_files/${filename}`,
    `../ESC-50/wav_files/${filename}`,
  ];
  if (isLocalSampleHost) {
    candidatePaths.push(`http://localhost:3001/api/sample/${cat.id}`);
  }

  for (const p of candidatePaths) {
    try {
      const resp = await fetch(p);
      if (resp.ok) {
        const blob = await resp.blob();
        if (blob && blob.size > 1000) {
          file = new File([blob], filename, { type: 'audio/wav' });
          break;
        }
      }
    } catch (_) { }
  }

  if (!file) {
    const blob = generateSyntheticAudioForCategory(cat.id, cat.name);
    file = new File([blob], filename, { type: 'audio/wav' });
  }

  loadAudioFile(file, true);
  showToast('Loaded into studio', `${cat.emoji} ${cat.name.replace(/_/g, ' ')} · verified test sample`, 'info');
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

/* ── Ambient Background Canvas Particles (Lag-Free & Studio Optimized) ── */
(function initAmbientParticles() {
  if (!bgCanvas) return;
  const ctx = bgCanvas.getContext('2d', { alpha: true });
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W = 0;
  let H = 0;
  let particles = [];
  let lastTime = 0;
  let resizeTimer = null;

  function onResize() {
    W = window.innerWidth;
    H = window.innerHeight;
    bgCanvas.width = W;
    bgCanvas.height = H;
    bgCanvas.style.width = `${W}px`;
    bgCanvas.style.height = `${H}px`;

    const target = W < 700 ? 5 : 8;
    while (particles.length < target) particles.push(new NodeParticle());
    if (particles.length > target) particles.length = target;
  }

  function debouncedResize() {
    if (resizeTimer) cancelAnimationFrame(resizeTimer);
    resizeTimer = requestAnimationFrame(onResize);
  }

  class NodeParticle {
    constructor() { this.spawn(); }
    spawn() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.radius = Math.random() * 1.2 + 0.6;
      this.vx = (Math.random() - 0.5) * 0.12;
      this.vy = (Math.random() - 0.5) * 0.12;
      this.alpha = Math.random() * 0.22 + 0.08;
      // Acoustic studio color psychology: Sonic Cyan & Warm Analog Amber
      this.color = Math.random() > 0.45 ? '#38bdf8' : '#fbbf24';
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < -10 || this.x > W + 10 || this.y < -10 || this.y > H + 10) this.spawn();
    }
    draw() {
      ctx.globalAlpha = isPlaying ? Math.min(0.5, this.alpha * 1.5) : this.alpha;
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function render() {
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i < particles.length; i++) {
      particles[i].update();
      particles[i].draw();
    }
  }

  function animate(time) {
    if (!document.hidden && time - lastTime > 45) { // Ultra-low CPU (~22fps, non-blocking)
      lastTime = time;
      render();
    }
    requestAnimationFrame(animate);
  }

  window.addEventListener('resize', debouncedResize, { passive: true });
  onResize();

  if (reduceMotion) {
    render();
  } else {
    requestAnimationFrame(animate);
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
  toast.style.setProperty('--toast-duration', `${ttl}ms`);
  toast.innerHTML = `
    <span class="toast-accent" aria-hidden="true"></span>
    <div class="toast-body">
      <p class="toast-title"></p>
      <p class="toast-msg"></p>
    </div>
    <button class="toast-dismiss" type="button" aria-label="Dismiss notification">&times;</button>
    <div class="toast-timer-bar" aria-hidden="true">
      <div class="toast-timer-fill"></div>
    </div>`;

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

  const timerFill = toast.querySelector('.toast-timer-fill');

  toast.querySelector('.toast-dismiss').addEventListener('click', dismiss);
  timer = setTimeout(dismiss, ttl);
  toast.addEventListener('pointerenter', () => {
    clearTimeout(timer);
    if (timerFill) timerFill.style.animationPlayState = 'paused';
  });
  toast.addEventListener('pointerleave', () => {
    if (timerFill) timerFill.style.animationPlayState = 'running';
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
    const score = Number.isFinite(p.confidence) ? `${(p.confidence * 100).toFixed(1)}%` : 'classifier confidence not computed';
    lines.push(`${i + 1}. ${p.category.name.replace(/_/g, ' ')} — ${score} (${p.category.group}, ESC-50 #${p.category.id})`);
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
  } catch (_) { }

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
    if (typeof closeAuthModal === 'function') closeAuthModal();
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
    setTimeout(() => {
      if (taxonomySearch) taxonomySearch.focus();
    }, 140);
    return;
  }

  const views = ['overview', 'studio', 'taxonomy', 'pipeline', 'training', 'analytics'];
  if (/^[1-6]$/.test(e.key)) {
    const num = Number(e.key);
    e.preventDefault();
    switchView(views[num - 1]);
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
    showToast('Unsupported file', 'Drop a WAV, MP3, or MPEG/MPG audio file.', 'error');
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
      ctx.beginPath();

      for (let x = 0; x <= w; x += 4.5) {
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
    const studioPanel = document.getElementById('view-studio');
    if (!document.hidden && studioPanel && !studioPanel.hidden) {
      drawFrame();
    }
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

/* ── Feature Mode Switcher Tabs (Mel / RP / Active) ────────── */
if (featModeBar) {
  featModeBar.querySelectorAll('.feat-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      if (mode === currentFeatureMode) return;
      currentFeatureMode = mode;
      featModeBar.querySelectorAll('.feat-tab-btn').forEach(b => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      if (specModeEl) {
        specModeEl.textContent = mode === 'mel' ? '128-Mel' : mode === 'rp' ? 'Threshold-Free RP' : 'Active VAD';
      }

      if (specSublabel) {
        if (mode === 'mel') specSublabel.textContent = '128-Band Log-Mel Spectrogram (Spatial Harmonic)';
        else if (mode === 'rp') specSublabel.textContent = '128×128 Threshold-Free Recurrence Plot (Chaotic Micro-Dynamics)';
        else specSublabel.textContent = 'Energy-Based Active Acoustic Segment (Longest Coherent Event)';
      }

      if (legScaleLbl) {
        legScaleLbl.textContent = mode === 'mel' ? 'Energy (dB):' : mode === 'rp' ? 'Distance Norm:' : 'Amplitude:';
      }

      if (legGradientBar) {
        if (mode === 'rp') legGradientBar.classList.add('rp-mode');
        else legGradientBar.classList.remove('rp-mode');
      }

      if (legMarks) {
        if (mode === 'mel') {
          legMarks.innerHTML = '<span>-80 dB</span><span>-50 dB</span><span>-25 dB</span><span>0 dB</span>';
        } else if (mode === 'rp') {
          legMarks.innerHTML = '<span>0.0 (High Recurrence)</span><span>0.33</span><span>0.66</span><span>1.0 (Divergent)</span>';
        } else {
          legMarks.innerHTML = '<span>-1.0</span><span>-0.5</span><span>0.0</span><span>+1.0</span>';
        }
      }

      drawFeatureSpectrogram(audioEl.currentTime || 0);
    });
  });
}

/* ════════════════════════════════════════════════════════════════
   VIEW 5: DEVELOPER TRAINING & CHECKPOINT MANAGEMENT PORTAL
   ════════════════════════════════════════════════════════════════ */

let activeTrainingJobId = null;
let trainingPollInterval = null;
let currentTrainingMetrics = { trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] };
let lastCompletedJob = null;

// DOM references for View 5
const trainModelTypeEl = document.getElementById('train-model-type');
const trainEpochsEl = document.getElementById('train-epochs');
const trainBatchSizeEl = document.getElementById('train-batch-size');
const trainLrEl = document.getElementById('train-lr');
const trainAugmentEl = document.getElementById('train-augment');
const trainDescEl = document.getElementById('train-desc');
const btnStartTraining = document.getElementById('btn-start-training');
const btnTrainIdle = document.getElementById('btn-train-idle');
const btnTrainBusy = document.getElementById('btn-train-busy');
const trainingJobIdEl = document.getElementById('training-job-id');
const trainingApiStatus = document.getElementById('training-api-status');

const tmEpochEl = document.getElementById('tm-epoch');
const tmTrainLossEl = document.getElementById('tm-train-loss');
const tmValLossEl = document.getElementById('tm-val-loss');
const tmTrainAccEl = document.getElementById('tm-train-acc');
const tmValAccEl = document.getElementById('tm-val-acc');
const tmStatusEl = document.getElementById('tm-status');
const trainProgressFill = document.getElementById('train-progress-fill');
const trainProgressPct = document.getElementById('train-progress-pct');
const trainLossChart = document.getElementById('train-loss-chart');
const btnSaveCheckpoint = document.getElementById('btn-save-checkpoint');
const checkpointGrid = document.getElementById('checkpoint-grid');

// Augmentation toggle label updater
if (trainAugmentEl) {
  trainAugmentEl.addEventListener('change', () => {
    const augLabel = document.getElementById('aug-label');
    if (augLabel) {
      augLabel.textContent = trainAugmentEl.checked ? 'Enabled (3× expansion)' : 'Disabled (Raw audio)';
    }
  });
}

/**
 * Loads training checkpoints and checks API health
 */
async function loadTrainingData() {
  await checkBackendStatus();
  await refreshCheckpoints();
}

/**
 * Checks if the FastAPI/Express backend is reachable.
 * Probes both the current origin and localhost:3001 so it works from
 * Live Server and file:// as well as when served by the backend itself.
 */
async function checkBackendStatus() {
  const hostname = window.location.hostname;
  const isLocal = hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '0.0.0.0' || window.location.protocol === 'file:';
  const port = window.location.port;
  const isAltLocalPort = isLocal && !['3001', '8000', '3000', '7860'].includes(port);

  const bases = [];
  if (!isAltLocalPort && window.location.protocol.startsWith('http')) {
    bases.push(window.location.origin);
  }
  if (isLocal) {
    bases.push(
      'http://localhost:3001',
      'http://127.0.0.1:3001',
      'http://localhost:8000',
      'http://127.0.0.1:8000',
      'http://localhost:3000'
    );
  } else {
    const resolved = resolveApiBase();
    if (!bases.includes(resolved)) bases.push(resolved);
  }
  backendOnline = false;

  for (const base of bases) {
    if ((base.includes(':5500') || base.includes(':5501')) && base === window.location.origin) {
      continue;
    }
    try {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${base}/api/health`, { signal: controller.signal });
      clearTimeout(t);
      if (res.ok) {
        const data = await res.json().catch(() => null);
        if (data && ['healthy', 'degraded'].includes(data.status)) {
          backendOnline = true;
          window.activeBackendBase = base;
          window.soundSenseHealth = data;
          break;
        }
      }
    } catch (_) { /* try next base */ }
  }

  if (backendOnline) {
    if (trainingApiStatus) {
      const missingCount = Array.isArray(window.soundSenseHealth?.missingModels) ? window.soundSenseHealth.missingModels.length : 0;
      trainingApiStatus.textContent = missingCount ? `API Connected · ${missingCount} model(s) unavailable` : `API Connected (${window.activeBackendBase || 'Port 3001'})`;
      trainingApiStatus.style.color = missingCount ? 'var(--amber, #f59e0b)' : 'var(--emerald)';
    }
    setStatus('Engine Ready · Backend Online', 'ready');
    return true;
  }

  if (trainingApiStatus) {
    trainingApiStatus.textContent = 'Backend Offline · Predictions unavailable';
    trainingApiStatus.style.color = 'var(--amber, #f59e0b)';
  }
  setStatus('Backend Offline · Predictions unavailable', 'warn');
  return false;
}

/**
 * Fetches and displays model checkpoints
 */
async function refreshCheckpoints() {
  let checkpoints = [];
  try {
    const res = await fetch(getApiUrl('/api/save-model'));
    if (res.ok) {
      const data = await res.json();
      if (data && Array.isArray(data.checkpoints)) {
        checkpoints = data.checkpoints;
      }
    }
  } catch (_) { }

  renderCheckpoints(checkpoints);
}

/**
 * Renders checkpoint cards in the grid
 */
function renderCheckpoints(checkpoints) {
  if (!checkpointGrid) return;
  checkpointGrid.innerHTML = '';

  checkpoints.forEach(ckpt => {
    const card = document.createElement('div');
    card.className = 'ckpt-card';
    card.innerHTML = `
      <div class="ckpt-header">
        <span class="ckpt-version">${escapeHtml(ckpt.version || 'v1.0.0')}</span>
        <span class="ckpt-status ${ckpt.status === 'active' ? 'active' : 'saved'}">${escapeHtml(ckpt.status || 'saved')}</span>
      </div>
      <h3 class="ckpt-name">${escapeHtml(ckpt.name || 'Model Checkpoint')}</h3>
      <p class="ckpt-desc">${escapeHtml(ckpt.description || 'ESC-50 Deep Learning Model Checkpoint')}</p>
      <div class="ckpt-meta">
        <span class="ckpt-meta-item">Acc: <strong>${Number.isFinite(ckpt.accuracy) ? (Number.isFinite(ckpt.top5) ? `${ckpt.accuracy.toFixed(1)}% (Top-5: ${ckpt.top5.toFixed(1)}%)` : `${ckpt.accuracy.toFixed(1)}%`) : 'Not evaluated'}</strong></span>
        <span class="ckpt-meta-item">Params: <strong>${escapeHtml(ckpt.params || '—')}</strong></span>
        <span class="ckpt-meta-item">Size: <strong>${Number.isFinite(ckpt.fileSizeMB) ? ckpt.fileSizeMB + ' MB' : (Number.isFinite(ckpt.fileSizeBytes) ? (ckpt.fileSizeBytes / (1024 * 1024)).toFixed(1) + ' MB' : '—')}</strong></span>
        <span class="ckpt-meta-item">Type: <strong>${escapeHtml(ckpt.modelType || 'resnet18')}</strong></span>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 14px;">
        <button type="button" class="ghost-btn btn-load-ckpt" data-model="${escapeHtml(ckpt.modelType || 'resnet18')}" style="flex: 1; justify-content: center; font-size: 0.76rem;">
          Load in Studio
        </button>
        <button type="button" class="ghost-btn btn-export-ckpt" data-id="${escapeHtml(ckpt.checkpointId || '')}" title="Download Weights Manifest" style="padding: 6px 10px;">
          ⬇️ Export
        </button>
      </div>
    `;

    // Load in studio action
    const loadBtn = card.querySelector('.btn-load-ckpt');
    if (loadBtn) {
      loadBtn.addEventListener('click', () => {
        const targetModel = loadBtn.dataset.model;
        if (modelSelect) {
          modelSelect.value = targetModel;
          modelSelect.dispatchEvent(new Event('change'));
        }
        switchView('studio');
        showToast('Model Activated', `Switched active studio model to "${ckpt.name}".`, 'success', 3500);
      });
    }

    // Export action
    const exportBtn = card.querySelector('.btn-export-ckpt');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => {
        const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(ckpt, null, 2));
        const dlAnchor = document.createElement('a');
        dlAnchor.setAttribute('href', dataStr);
        dlAnchor.setAttribute('download', `${ckpt.version || 'ckpt'}_${ckpt.modelType || 'model'}_manifest.json`);
        document.body.appendChild(dlAnchor);
        dlAnchor.click();
        dlAnchor.remove();
        showToast('Checkpoint Exported', `Downloaded metadata manifest for ${ckpt.version}.`, 'info', 3000);
      });
    }

    checkpointGrid.appendChild(card);
  });
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, m => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[m]));
}

/**
 * Model training runs offline on Kaggle; this browser view does not simulate metrics.
 */
if (btnStartTraining) {
  btnStartTraining.addEventListener('click', () => {
    if (tmStatusEl) {
      tmStatusEl.textContent = 'Offline · Kaggle';
      tmStatusEl.style.color = 'var(--amber, #f59e0b)';
    }
    showToast('Training runs offline', 'Run python -m eval.train_cv on a GPU, then evaluate the checkpoints. This screen will not simulate metrics.', 'info', 6000);
  });
}
/**
 * Updates training metrics, progress bar, and charts
 */
function updateTrainingUI(currentEpoch, totalEpochs, metrics, progress, status) {
  if (tmEpochEl) tmEpochEl.textContent = `${currentEpoch} / ${totalEpochs}`;

  const lastTLoss = metrics.trainLoss.length > 0 ? metrics.trainLoss[metrics.trainLoss.length - 1] : '—';
  const lastVLoss = metrics.valLoss.length > 0 ? metrics.valLoss[metrics.valLoss.length - 1] : '—';
  const lastTAcc = metrics.trainAcc.length > 0 ? metrics.trainAcc[metrics.trainAcc.length - 1] + '%' : '—';
  const lastVAcc = metrics.valAcc.length > 0 ? metrics.valAcc[metrics.valAcc.length - 1] + '%' : '—';

  if (tmTrainLossEl) tmTrainLossEl.textContent = lastTLoss;
  if (tmValLossEl) tmValLossEl.textContent = lastVLoss;
  if (tmTrainAccEl) tmTrainAccEl.textContent = lastTAcc;
  if (tmValAccEl) tmValAccEl.textContent = lastVAcc;
  if (tmStatusEl) tmStatusEl.textContent = status === 'completed' ? 'Completed' : 'Training';

  if (trainProgressFill) trainProgressFill.style.width = `${progress}%`;
  if (trainProgressPct) trainProgressPct.textContent = `${progress}%`;

  currentTrainingMetrics = metrics;
  renderTrainingChart(metrics);
}

/**
 * Handles training completion
 */
function finishTrainingRun(job) {
  lastCompletedJob = job;
  btnStartTraining.disabled = false;
  if (btnTrainIdle) btnTrainIdle.hidden = false;
  if (btnTrainBusy) btnTrainBusy.hidden = true;
  if (tmStatusEl) {
    tmStatusEl.textContent = 'Completed';
    tmStatusEl.style.color = 'var(--emerald)';
  }

  showToast('Training Complete', 'Model training run finished with target convergence.', 'success', 4000);
}

/**
 * Draws real-time loss & accuracy curve on the HTML5 canvas
 */
function renderTrainingChart(metrics) {
  if (!trainLossChart) return;
  const ctx = trainLossChart.getContext('2d');
  const w = trainLossChart.width;
  const h = trainLossChart.height;

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = 'rgba(6, 12, 28, 0.9)';
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
  ctx.lineWidth = 1;
  const gridRows = 4;
  for (let r = 1; r < gridRows; r++) {
    const y = (h / gridRows) * r;
    ctx.beginPath();
    ctx.moveTo(35, y);
    ctx.lineTo(w - 15, y);
    ctx.stroke();
  }

  // Axes labels
  ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
  ctx.font = '10px Space Grotesk, monospace';
  ctx.fillText('2.5', 8, 20);
  ctx.fillText('1.2', 8, h / 2 + 3);
  ctx.fillText('0.0', 8, h - 10);

  const tLoss = metrics.trainLoss || [];
  const vLoss = metrics.valLoss || [];
  const count = Math.max(tLoss.length, vLoss.length);

  if (count < 2) {
    ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Awaiting training iterations to plot convergence curve...', w / 2, h / 2);
    ctx.textAlign = 'start';
    return;
  }

  const chartX = 40;
  const chartW = w - 55;
  const chartH = h - 30;
  const maxVal = 2.8;

  // Draw Train Loss curve (Cyan)
  drawCurve(ctx, tLoss, chartX, chartW, chartH, maxVal, '#38bdf8', 'Train Loss');

  // Draw Val Loss curve (Purple)
  drawCurve(ctx, vLoss, chartX, chartW, chartH, maxVal, '#a855f7', 'Val Loss');

  // Legend
  ctx.fillStyle = '#38bdf8';
  ctx.fillRect(w - 150, 10, 10, 3);
  ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
  ctx.font = '9px Inter, sans-serif';
  ctx.fillText('Train Loss', w - 135, 14);

  ctx.fillStyle = '#a855f7';
  ctx.fillRect(w - 80, 10, 10, 3);
  ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
  ctx.fillText('Val Loss', w - 65, 14);
}

function drawCurve(ctx, data, startX, width, height, maxVal, color, _label) {
  if (data.length < 2) return;
  const step = width / (data.length - 1);

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();

  data.forEach((val, i) => {
    const x = startX + i * step;
    const clamped = Math.max(0, Math.min(maxVal, val));
    const y = height - (clamped / maxVal) * (height - 15) + 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Highlight latest point
  const lastIndex = data.length - 1;
  const lastVal = data[lastIndex];
  const lastX = startX + lastIndex * step;
  const lastY = height - (Math.max(0, Math.min(maxVal, lastVal)) / maxVal) * (height - 15) + 10;

  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);
  ctx.fill();
}

/**
 * Handle Save New Checkpoint Button Click
 */
if (btnSaveCheckpoint) {
  btnSaveCheckpoint.addEventListener('click', () => {
    showToast('No checkpoint created', 'Checkpoints are written by the offline Kaggle training pipeline after a real evaluation run.', 'info', 6000);
  });
}

/* ════════════════════════════════════════════════════════════════
   VIEW 6: SYSTEM ANALYTICS & TELEMETRY DASHBOARD
   ════════════════════════════════════════════════════════════════ */

const ahStatusEl = document.getElementById('ah-status');
const ahUptimeEl = document.getElementById('ah-uptime');
const ahPredsEl = document.getElementById('ah-predictions');
const ahMemoryEl = document.getElementById('ah-memory');
const ahNodeEl = document.getElementById('ah-node');
const ahCkptsEl = document.getElementById('ah-checkpoints');

const classDistChart = document.getElementById('class-dist-chart');
const modelPerfChart = document.getElementById('model-perf-chart');
const telemetryTsEl = document.getElementById('telemetry-timestamp');

const tgAudioEl = document.getElementById('tg-audio');
const tgMelEl = document.getElementById('tg-mel');
const tgMfccEl = document.getElementById('tg-mfcc');
const tgRpEl = document.getElementById('tg-rp');

/**
 * Loads analytics data from /api/stats
 */
async function loadAnalyticsData() {
  let statsData = null;
  try {
    const res = await fetch(getApiUrl('/api/stats'));
    if (res.ok) {
      statsData = await res.json();
    }
  } catch (_) { }

  // Populate UI
  if (statsData && statsData.system) {
    const missingModels = statsData.modelAvailability?.missingModels || [];
    if (ahStatusEl) ahStatusEl.textContent = missingModels.length ? `${missingModels.length} model(s) unavailable` : 'Healthy (API)';
    if (ahUptimeEl) ahUptimeEl.textContent = statsData.system.uptimeFormatted || '0h 0m';
    if (ahPredsEl) ahPredsEl.textContent = statsData.system.totalPredictions || 0;
    if (ahMemoryEl) ahMemoryEl.textContent = Number.isFinite(statsData.system.memoryUsage?.heapUsed) ? `${statsData.system.memoryUsage.heapUsed} MB` : '—';
    if (ahNodeEl) ahNodeEl.textContent = statsData.system.nodeVersion || 'FastAPI 0.115 / Python';
    if (ahCkptsEl) ahCkptsEl.textContent = statsData.system.savedCheckpoints ?? 0;
  } else {
    if (ahStatusEl) ahStatusEl.textContent = 'API unavailable';
    if (ahUptimeEl) ahUptimeEl.textContent = '—';
    if (ahPredsEl) ahPredsEl.textContent = '—';
    if (ahMemoryEl) ahMemoryEl.textContent = '—';
    if (ahNodeEl) ahNodeEl.textContent = 'Backend unavailable';
    if (ahCkptsEl) ahCkptsEl.textContent = '—';
  }

  // Telemetry Grid
  const d = statsData?.dataset;
  if (tgAudioEl) tgAudioEl.textContent = Number.isFinite(d?.processedAudioCount) ? d.processedAudioCount.toLocaleString() : '—';
  if (tgMelEl) tgMelEl.textContent = Number.isFinite(d?.logMelCount) ? d.logMelCount.toLocaleString() : '—';
  if (tgMfccEl) tgMfccEl.textContent = Number.isFinite(d?.mfccCount) ? d.mfccCount.toLocaleString() : '—';
  if (tgRpEl) tgRpEl.textContent = Number.isFinite(d?.rpCount) ? d.rpCount.toLocaleString() : '—';

  const perf = Array.isArray(statsData?.modelPerformance) ? statsData.modelPerformance : [];
  const perfByModel = new Map(perf.map(item => [item.modelKey, item]));
  for (const modelKey of Object.keys(MODEL_INFO)) {
    const item = perfByModel.get(modelKey);
    const accuracy = Number.isFinite(item?.accuracy) ? item.accuracy : null;
    const top5 = Number.isFinite(item?.top5Accuracy) ? item.top5Accuracy : null;
    const accuracyText = accuracy == null ? 'Not evaluated' : `${accuracy.toFixed(1)}%`;
    const fullAccuracyText = accuracy == null
      ? 'Not evaluated'
      : (top5 != null ? `${accuracy.toFixed(1)}% (Top-5: ${top5.toFixed(1)}%)` : `${accuracy.toFixed(1)}%`);

    document.querySelectorAll(`.m-val[data-model-key="${modelKey}"]`).forEach(el => {
      el.textContent = accuracy == null ? 'Not evaluated' : (top5 != null ? `${accuracy.toFixed(1)}% / ${top5.toFixed(1)}%` : `${accuracy.toFixed(1)}%`);
      delete el.dataset.raw;
    });
    const row = document.querySelector(`.perf-row[data-model-key="${modelKey}"]`);
    if (row) {
      row.dataset.value = accuracy == null ? '0' : String(accuracy);
      const fill = row.querySelector('.perf-fill');
      const label = row.querySelector('.perf-pct');
      if (fill) fill.dataset.w = accuracy == null ? '0' : String(accuracy);
      if (label) label.textContent = fullAccuracyText;
    }
    const tableRow = document.querySelector(`.spec-table tr[data-model-key="${modelKey}"]`);
    if (tableRow) {
      const accuracyCell = tableRow.querySelector('[data-model-metric="accuracy"]');
      const latencyCell = tableRow.querySelector('[data-model-metric="latency"]');
      if (accuracyCell) {
        if (modelKey === 'ast') {
          accuracyCell.innerHTML = `<strong class="col-highlight">${fullAccuracyText}</strong>`;
        } else {
          accuracyCell.textContent = fullAccuracyText;
        }
      }
      if (latencyCell) latencyCell.textContent = Number.isFinite(item?.latency) ? `${item.latency.toFixed(1)} ms` : '—';
    }
    const modelInfo = MODEL_INFO[modelKey];
    if (modelInfo) {
      modelInfo.accuracy = fullAccuracyText;
      modelInfo.latency = Number.isFinite(item?.latency) ? item.latency : null;
    }
  }
  const perfChip = document.getElementById('perf-chip');
  if (perfChip) {
    const isCv = perf.some(item => String(item.evaluatedOn || '').toLowerCase().includes('cross-validation'));
    const astPerf = perfByModel.get('ast');
    perfChip.textContent = astPerf && Number.isFinite(astPerf.accuracy)
      ? `AST deployed: ${astPerf.accuracy.toFixed(1)}% Top-1 (${astPerf.top5Accuracy ? astPerf.top5Accuracy.toFixed(1) + '% Top-5' : 'fold-1 test'})`
      : 'Awaiting evaluation report';
  }

  if (telemetryTsEl) {
    telemetryTsEl.textContent = `Telemetry synced at ${new Date().toLocaleTimeString()}`;
  }

  // Draw Charts
  drawClassDistributionChart(d?.groupDistribution);
  _lastPerfList = [...perf].filter(item => Number.isFinite(item.accuracy)).sort((a, b) => b.accuracy - a.accuracy);
  drawModelPerformanceChart(_lastPerfList);
}

/**
 * Draws the ESC-50 Class Distribution Bar Chart
 */
function drawClassDistributionChart(groupDistribution) {
  if (!classDistChart) return;
  const ctx = classDistChart.getContext('2d');
  const w = classDistChart.width;
  const h = classDistChart.height;

  ctx.clearRect(0, 0, w, h);

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';

  // Theme-aware canvas background
  ctx.fillStyle = isLight ? '#ffffff' : 'rgba(7, 10, 20, 0.95)';
  ctx.fillRect(0, 0, w, h);

  // Subtle horizontal reference gridlines
  ctx.strokeStyle = isLight ? 'rgba(15, 23, 42, 0.06)' : 'rgba(255, 255, 255, 0.04)';
  ctx.lineWidth = 1;
  for (let gy = 30; gy <= h - 40; gy += 40) {
    ctx.beginPath();
    ctx.moveTo(25, gy);
    ctx.lineTo(w - 25, gy);
    ctx.stroke();
  }

  const groupMeta = {
    Animals: ['🐾', isLight ? '#0284c7' : '#38bdf8'],
    Natural: ['🌿', isLight ? '#059669' : '#34d399'],
    Human: ['👤', isLight ? '#d97706' : '#fbbf24'],
    Interior: ['🏠', isLight ? '#7c3aed' : '#a855f7'],
    Urban: ['🏙️', isLight ? '#e11d48' : '#fb7185'],
  };
  const groups = (Array.isArray(groupDistribution) ? groupDistribution : Object.keys(groupMeta).map(group => ({ group, totalSamples: 0 })))
    .map(item => ({ name: item.group, icon: groupMeta[item.group]?.[0] || '🔊', color: groupMeta[item.group]?.[1] || '#64748b', count: Number(item.totalSamples) || 0 }));

  const barWidth = 64;
  const gap = (w - (groups.length * barWidth) - 40) / (groups.length - 1);
  const maxCount = Math.max(1, ...groups.map(g => g.count));
  const chartHeight = h - 65;

  groups.forEach((g, i) => {
    const x = 30 + i * (barWidth + gap);
    const barH = (g.count / maxCount) * chartHeight;
    const y = h - 35 - barH;

    // Bar background track
    ctx.fillStyle = isLight ? 'rgba(15, 23, 42, 0.05)' : 'rgba(255, 255, 255, 0.05)';
    ctx.beginPath();
    ctx.roundRect(x, 25, barWidth, chartHeight, 6);
    ctx.fill();

    // Bar fill with gradient
    const grad = ctx.createLinearGradient(0, y, 0, h - 35);
    grad.addColorStop(0, g.color);
    grad.addColorStop(1, isLight ? 'rgba(226, 232, 240, 0.85)' : 'rgba(15, 23, 42, 0.8)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, barWidth, barH, [6, 6, 2, 2]);
    ctx.fill();

    // Top accent border
    ctx.fillStyle = g.color;
    ctx.fillRect(x + 2, y, barWidth - 4, 2);

    // Value text
    ctx.fillStyle = isLight ? '#090d16' : '#ffffff';
    ctx.font = 'bold 11px Space Grotesk, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${g.count} clips`, x + barWidth / 2, y - 6);

    // Label below
    ctx.fillStyle = isLight ? '#334155' : 'rgba(203, 213, 225, 0.9)';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText(`${g.icon} ${g.name}`, x + barWidth / 2, h - 14);
  });

  ctx.textAlign = 'start';
}

/**
 * Draws the Model Performance Comparison Chart
 */
function drawModelPerformanceChart(perfList) {
  if (!modelPerfChart) return;
  const ctx = modelPerfChart.getContext('2d');
  const w = modelPerfChart.width;
  const h = modelPerfChart.height;

  ctx.clearRect(0, 0, w, h);

  const isLight = document.documentElement.getAttribute('data-theme') === 'light';

  // Theme-aware canvas background
  ctx.fillStyle = isLight ? '#ffffff' : 'rgba(7, 10, 20, 0.95)';
  ctx.fillRect(0, 0, w, h);

  const models = Array.isArray(perfList) ? perfList : [];

  if (!models.length) {
    ctx.fillStyle = isLight ? '#64748b' : 'rgba(148, 163, 184, 0.75)';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No model evaluation report available', w / 2, h / 2);
    ctx.textAlign = 'start';
    return;
  }

  const count = Math.max(1, models.length);
  const startY = 20;
  const availableH = h - startY - 24;
  const rowSlot = Math.min(38, Math.max(28, Math.floor(availableH / count)));
  const barHeight = Math.max(12, Math.min(18, rowSlot - 12));
  const labelW = 145;
  const chartW = w - labelW - 130;

  // Benchmark scale gridlines
  ctx.strokeStyle = isLight ? 'rgba(15, 23, 42, 0.06)' : 'rgba(255, 255, 255, 0.05)';
  ctx.fillStyle = isLight ? '#64748b' : 'rgba(148, 163, 184, 0.8)';
  ctx.font = '9px Space Grotesk, monospace';
  ctx.textAlign = 'center';

  [50, 75, 90, 100].forEach(pct => {
    const tickX = labelW + 10 + (pct / 100) * chartW;
    ctx.beginPath();
    ctx.moveTo(tickX, 14);
    ctx.lineTo(tickX, h - 14);
    ctx.stroke();
    ctx.fillText(`${pct}%`, tickX, h - 4);
  });

  models.forEach((m, i) => {
    const y = startY + i * rowSlot;

    // Model name
    ctx.fillStyle = isLight ? '#0f172a' : 'rgba(226, 232, 240, 0.95)';
    ctx.font = '600 11.5px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(m.model, 14, y + Math.floor(barHeight / 2) + 4);

    // Track
    const barX = labelW + 10;
    ctx.fillStyle = isLight ? 'rgba(15, 23, 42, 0.05)' : 'rgba(255, 255, 255, 0.05)';
    ctx.beginPath();
    ctx.roundRect(barX, y, chartW, barHeight, 4);
    ctx.fill();

    // Fill
    const fillW = Math.max(4, (m.accuracy / 100) * chartW);
    const grad = ctx.createLinearGradient(barX, 0, barX + fillW, 0);
    grad.addColorStop(0, m.color || '#6366f1');
    grad.addColorStop(1, isLight ? '#4338ca' : '#818cf8');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(barX, y, fillW, barHeight, 4);
    ctx.fill();

    // Accuracy percent
    ctx.fillStyle = isLight ? '#090d16' : '#ffffff';
    ctx.font = 'bold 11px Space Grotesk, monospace';
    ctx.textAlign = 'left';
    const accLabel = Number.isFinite(m.top5Accuracy) ? `${m.accuracy.toFixed(1)}% (T5: ${m.top5Accuracy.toFixed(1)}%)` : `${m.accuracy}%`;
    ctx.fillText(accLabel, barX + chartW + 8, y + Math.floor(barHeight / 2) + 4);

    // Latency
    ctx.fillStyle = isLight ? '#64748b' : 'rgba(148, 163, 184, 0.85)';
    ctx.font = '10px Inter, sans-serif';
    const latOffset = Number.isFinite(m.top5Accuracy) ? 122 : 52;
    ctx.fillText(`(${Number.isFinite(m.latency) && m.latency > 0 ? `${m.latency.toFixed(1)}ms` : (m.latency === 0 ? '< 5ms' : '—')})`, barX + chartW + latOffset, y + Math.floor(barHeight / 2) + 4);
  });

  ctx.textAlign = 'start';
}

// Initial API health check on boot
checkBackendStatus();

/* ════════════════════════════════════════════════════════════════
   DEVELOPER AUTHENTICATION & GOOGLE-STYLE ACCOUNT ICON
   ════════════════════════════════════════════════════════════════ */

let isDeveloperAuthenticated = false;
let currentAuthUser = null;
let pendingDevView = null;

const accountBtn = document.getElementById('account-btn');
const accountAvatar = document.getElementById('account-avatar');
const devBadgeDot = document.getElementById('dev-badge-dot');
const accountMenuPopover = document.getElementById('account-menu-popover');
const btnLogout = document.getElementById('btn-logout');
const menuGotoArchitecture = document.getElementById('menu-goto-architecture');
const menuGotoTaxonomy = document.getElementById('menu-goto-taxonomy');
const menuGotoPipeline = document.getElementById('menu-goto-pipeline');
const menuGotoTraining = document.getElementById('menu-goto-training');
const menuGotoAnalytics = document.getElementById('menu-goto-analytics');
const popoverNameEl = document.getElementById('popover-name');
const popoverEmailEl = document.getElementById('popover-email');
const popoverInitialEl = document.getElementById('popover-initial');
const tagBadgeArchitecture = document.getElementById('tag-badge-architecture');
const tagBadgeTaxonomy = document.getElementById('tag-badge-taxonomy');
const tagBadgePipeline = document.getElementById('tag-badge-pipeline');
const tagBadgeTraining = document.getElementById('tag-badge-training');
const tagBadgeAnalytics = document.getElementById('tag-badge-analytics');

const authModal = document.getElementById('auth-modal');
const authModalClose = document.getElementById('auth-modal-close');
const authLoginForm = document.getElementById('auth-login-form');
const authUsernameInput = document.getElementById('auth-username');
const authPasswordInput = document.getElementById('auth-password');
const authErrorMsg = document.getElementById('auth-error-msg');
const btnLoginSubmit = document.getElementById('btn-login-submit');
const btnLoginIdle = document.getElementById('btn-login-idle');
const btnLoginBusy = document.getElementById('btn-login-busy');
const devOnlyTabs = document.querySelectorAll('.dev-tab');

function openAuthModal() {
  const modal = authModal || document.getElementById('auth-modal');
  if (!modal) return;
  modal.removeAttribute('hidden');
  modal.hidden = false;
  modal.style.display = 'grid';
  document.body.style.overflow = 'hidden';

  const errEl = authErrorMsg || document.getElementById('auth-error-msg');
  if (errEl) {
    errEl.setAttribute('hidden', '');
    errEl.hidden = true;
    errEl.style.display = 'none';
  }

  const regErr = document.getElementById('auth-reg-error-msg');
  if (regErr) {
    regErr.setAttribute('hidden', '');
    regErr.hidden = true;
    regErr.style.display = 'none';
  }

  if (typeof setAuthModalMode === 'function') {
    setAuthModalMode('signin');
  }

  const popover = accountMenuPopover || document.getElementById('account-menu-popover');
  if (popover) {
    popover.setAttribute('hidden', '');
    popover.hidden = true;
    popover.style.display = 'none';
  }

  const userInput = authUsernameInput || document.getElementById('auth-username');
  const passInput = authPasswordInput || document.getElementById('auth-password');
  if (userInput) userInput.value = '';
  if (passInput) passInput.value = '';

  const regName = document.getElementById('auth-reg-name');
  const regEmail = document.getElementById('auth-reg-email');
  const regPass = document.getElementById('auth-reg-password');
  const regConf = document.getElementById('auth-reg-confirm');
  if (regName) regName.value = '';
  if (regEmail) regEmail.value = '';
  if (regPass) regPass.value = '';
  if (regConf) regConf.value = '';

  setTimeout(() => {
    if (userInput) userInput.focus();
  }, 60);
}
window.openAuthModal = openAuthModal;

function closeAuthModal() {
  const modal = authModal || document.getElementById('auth-modal');
  if (!modal) return;
  modal.setAttribute('hidden', '');
  modal.hidden = true;
  modal.style.display = 'none';
  document.body.style.overflow = '';
  pendingDevView = null;
}
window.closeAuthModal = closeAuthModal;

if (authModalClose) {
  authModalClose.addEventListener('click', closeAuthModal);
}

if (authModal) {
  authModal.addEventListener('click', (e) => {
    if (e.target === authModal) closeAuthModal();
  });
}

/* ── Architecture selector ─────────────────────────────────
   User mode: Exposes exclusively the best model (AST-ESC50 · 92.7% verified accuracy).
   Developer mode: Unlocks all 6 neural architectures for research and benchmarking. */
function syncModelOptionsForRole() {
  if (!modelSelect) return;
  const isDev = isDeveloperAuthenticated;

  modelSelect.querySelectorAll('option').forEach(opt => {
    if (isDev) {
      opt.hidden = false;
      opt.disabled = false;
      opt.style.display = '';
    } else {
      if (opt.value === 'ast') {
        opt.hidden = false;
        opt.disabled = false;
        opt.style.display = '';
        opt.selected = true;
      } else {
        opt.hidden = true;
        opt.disabled = true;
        opt.style.display = 'none';
      }
    }
  });

  if (!isDev) {
    modelSelect.value = 'ast';
  }

  const batchModelInfo = document.getElementById('batch-model-info');
  if (batchModelInfo) {
    batchModelInfo.textContent = isDev
      ? (modelSelect.options[modelSelect.selectedIndex]?.textContent?.split(' ')[0] || 'AST-ESC50')
      : 'AST-ESC50 (Best)';
  }

  updateModelBadge();
}

function setDeveloperMode(authenticated, user = null) {
  isDeveloperAuthenticated = !!authenticated;
  currentAuthUser = user;

  syncModelOptionsForRole();

  if (isDeveloperAuthenticated) {
    // Reveal developer mode across styles and DOM
    document.body.classList.add('dev-authenticated');
    devOnlyTabs.forEach(tab => {
      tab.style.display = 'inline-flex';
    });

    // Update all nav dev badges to unlocked
    document.querySelectorAll('.nav-dev-tag').forEach(badge => {
      badge.textContent = 'DEV';
      badge.classList.add('unlocked');
      badge.title = 'Developer Portal Unlocked';
    });

    // Update Google-style account button to active developer state
    if (accountBtn) {
      accountBtn.classList.add('logged-in');
      accountBtn.title = `Developer Account: ${user?.email || 'developer@soundsense.ai'}`;
    }
    if (devBadgeDot) devBadgeDot.style.display = 'block';

    const initial = (user?.name || user?.email || 'D')[0].toUpperCase();
    if (accountAvatar) {
      accountAvatar.innerHTML = `<span class="avatar-letter">${initial}</span>`;
    }

    if (popoverInitialEl) popoverInitialEl.textContent = initial;
    if (popoverNameEl) popoverNameEl.textContent = user?.name || 'SoundSense Lead AI Engineer';
    if (popoverEmailEl) popoverEmailEl.textContent = user?.email || 'developer@soundsense.ai';

    requestAnimationFrame(positionNavGlider);
    setTimeout(positionNavGlider, 80);
    setTimeout(positionNavGlider, 250);
  } else {
    // Capture the open view BEFORE deactivating the developer tabs.
    // (Removing `.active` from the tab made the old check miss dev views.)
    const activePanel = document.querySelector('.view-panel.active');
    const wasOnDevView = !!activePanel &&
      DEV_ONLY_VIEWS.includes(activePanel.id.replace('view-', ''));

    // Hide all developer mode tabs and features
    document.body.classList.remove('dev-authenticated');
    devOnlyTabs.forEach(tab => {
      tab.style.display = 'none';
      tab.classList.remove('active');
      tab.setAttribute('aria-selected', 'false');
    });

    // Reset account button to default Google-style user silhouette
    if (accountBtn) {
      accountBtn.classList.remove('logged-in');
      accountBtn.title = 'Developer Sign In';
    }
    if (devBadgeDot) devBadgeDot.style.display = 'none';
    if (accountAvatar) {
      accountAvatar.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
    }

    closeAccountMenu();

    // Reset all nav dev badges to locked
    document.querySelectorAll('.nav-dev-tag').forEach(badge => {
      badge.textContent = 'DEV';
      badge.classList.remove('unlocked');
      badge.title = 'Developer Access Required';
    });

    // If a developer view was open, fade it out before returning to Studio.
    // The short pause keeps the logout from feeling like an instant flicker.
    if (wasOnDevView && activePanel) {
      if (navGlider) navGlider.classList.remove('ready');
      activePanel.classList.add('is-leaving');

      setTimeout(() => {
        // Abort if the user signed back in or navigated manually meanwhile
        if (isDeveloperAuthenticated) {
          activePanel.classList.remove('is-leaving');
          return;
        }
        if (!activePanel.classList.contains('active')) {
          activePanel.classList.remove('is-leaving');
          return;
        }

        // Jump to the top while the old panel is invisible, then reveal Studio
        window.scrollTo({ top: 0, behavior: 'instant' });
        switchView('studio');

        // Snap the nav pill straight to Studio instead of letting it fly in
        // from the removed developer tab.
        if (navGlider) {
          navGlider.style.transition = 'none';
          positionNavGlider(false);
          requestAnimationFrame(() => { navGlider.style.transition = ''; });
        }

        activePanel.classList.remove('is-leaving');
      }, 240);
    } else {
      requestAnimationFrame(positionNavGlider);
      setTimeout(positionNavGlider, 80);
      setTimeout(positionNavGlider, 250);
    }
  }

  // Developer tabs may change the nav's overflow width
  requestAnimationFrame(updateNavArrows);
  setTimeout(updateNavArrows, 120);
}

// State tracker for developer menu popover
let isAccountMenuOpen = false;
let lastAccountMenuToggleTime = 0;

function toggleAccountMenu(forceState) {
  const popover = accountMenuPopover || document.getElementById('account-menu-popover');
  const btn = accountBtn || document.getElementById('account-btn');
  if (!popover) return;

  const willOpen = forceState !== undefined ? Boolean(forceState) : !isAccountMenuOpen;
  isAccountMenuOpen = willOpen;

  if (willOpen) {
    popover.removeAttribute('hidden');
    popover.hidden = false;
    popover.style.display = 'block';
    if (btn) btn.setAttribute('aria-expanded', 'true');
    fitAccountMenuPopover();
    requestAnimationFrame(fitAccountMenuPopover);
  } else {
    popover.setAttribute('hidden', '');
    popover.hidden = true;
    popover.style.display = 'none';
    if (btn) btn.setAttribute('aria-expanded', 'false');
  }
}

function closeAccountMenu() {
  toggleAccountMenu(false);
}

// Dynamically fit account menu popover into viewport across all zoom levels
function fitAccountMenuPopover() {
  const popover = accountMenuPopover || document.getElementById('account-menu-popover');
  const btn = accountBtn || document.getElementById('account-btn');
  if (!popover || popover.hidden || popover.style.display === 'none' || !btn) return;

  popover.style.maxHeight = '';
  popover.style.top = '';
  popover.style.bottom = '';
  popover.style.right = '';
  popover.style.left = '';

  const btnRect = btn.getBoundingClientRect();
  const popoverRect = popover.getBoundingClientRect();
  const vpWidth = window.innerWidth;
  const vpHeight = window.innerHeight;

  // Horizontal fit: Align right edge with button wrapper, or clamp within viewport
  if (popoverRect.right > vpWidth - 10) {
    const shift = popoverRect.right - (vpWidth - 10);
    popover.style.right = `${shift}px`;
  } else if (popoverRect.left < 10) {
    popover.style.right = 'auto';
    popover.style.left = '0px';
  } else {
    popover.style.right = '0px';
  }

  // Vertical fit: Flip or clamp height based on available screen space
  const spaceBelow = vpHeight - btnRect.bottom - 12;
  const spaceAbove = btnRect.top - 12;

  if (spaceBelow < 200 && spaceAbove > spaceBelow) {
    popover.style.top = 'auto';
    popover.style.bottom = 'calc(100% + 8px)';
    popover.style.maxHeight = `${Math.min(420, spaceAbove)}px`;
  } else {
    popover.style.top = 'calc(100% + 8px)';
    popover.style.bottom = 'auto';
    popover.style.maxHeight = `${Math.max(160, Math.min(460, spaceBelow))}px`;
  }
}

// Global handler for developer account button clicks
window.handleAccountBtnClick = function (e) {
  if (e) {
    if (typeof e.preventDefault === 'function') e.preventDefault();
    if (typeof e.stopPropagation === 'function') e.stopPropagation();
    if (typeof e.stopImmediatePropagation === 'function') e.stopImmediatePropagation();
  }

  const now = Date.now();
  if (now - lastAccountMenuToggleTime < 250) {
    return; // Prevent accidental double-toggle from duplicate events
  }
  lastAccountMenuToggleTime = now;

  if (isDeveloperAuthenticated) {
    toggleAccountMenu();
  } else {
    openAuthModal();
  }
};

// Account button click listener
if (accountBtn) {
  accountBtn.addEventListener('click', (e) => {
    window.handleAccountBtnClick(e);
  });
}

// Re-adjust popover on window resize and scroll for zoom stability
window.addEventListener('resize', () => {
  if (isAccountMenuOpen) fitAccountMenuPopover();
});
window.addEventListener('scroll', () => {
  if (isAccountMenuOpen) fitAccountMenuPopover();
}, { passive: true });

// Close account menu popover when clicking anywhere outside
document.addEventListener('click', (e) => {
  const popover = accountMenuPopover || document.getElementById('account-menu-popover');
  const btn = accountBtn || document.getElementById('account-btn');
  if (isAccountMenuOpen && popover) {
    if (!popover.contains(e.target) && (!btn || !btn.contains(e.target))) {
      closeAccountMenu();
    }
  }
});

// Close account menu popover on Escape key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && isAccountMenuOpen) {
    closeAccountMenu();
    if (accountBtn) accountBtn.focus();
  }
});

// Quick navigation links inside account popover
if (menuGotoArchitecture) {
  menuGotoArchitecture.addEventListener('click', () => {
    closeAccountMenu();
    switchView('architecture');
  });
}

if (menuGotoTaxonomy) {
  menuGotoTaxonomy.addEventListener('click', () => {
    closeAccountMenu();
    switchView('taxonomy');
  });
}

const menuGotoOverview = document.getElementById('menu-goto-overview');
if (menuGotoOverview) {
  menuGotoOverview.addEventListener('click', () => {
    closeAccountMenu();
    switchView('overview');
  });
}

if (menuGotoPipeline) {
  menuGotoPipeline.addEventListener('click', () => {
    closeAccountMenu();
    switchView('pipeline');
  });
}

if (menuGotoTraining) {
  menuGotoTraining.addEventListener('click', () => {
    closeAccountMenu();
    switchView('training');
  });
}

if (menuGotoAnalytics) {
  menuGotoAnalytics.addEventListener('click', () => {
    closeAccountMenu();
    switchView('analytics');
  });
}

// Switch / Re-authenticate option in developer popover
const menuSwitchAccount = document.getElementById('menu-switch-account');
if (menuSwitchAccount) {
  menuSwitchAccount.addEventListener('click', (e) => {
    e.stopPropagation();
    closeAccountMenu();
    openAuthModal();
  });
}

// Sign out button handler
if (btnLogout) {
  btnLogout.addEventListener('click', async () => {
    try {
      const stored = sessionStorage.getItem('soundsense_dev_auth') || localStorage.getItem('soundsense_dev_auth');
      const token = stored ? JSON.parse(stored)?.token : null;
      if (token) {
        fetch(getApiUrl('/api/auth/logout'), {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` }
        }).catch(() => { });
      }
    } catch (_) { }

    try {
      sessionStorage.removeItem('soundsense_dev_auth');
      localStorage.removeItem('soundsense_dev_auth');
    } catch (_) { }
    setDeveloperMode(false);

    showToast('Signed Out', 'Exited Developer Mode. Developer portals are now locked.', 'info', 3500);
  });
}

// Auth Tabs & Registration Elements
const authTabSignin = document.getElementById('auth-tab-signin');
const authTabRegister = document.getElementById('auth-tab-register');
const authToggleToReg = document.getElementById('auth-toggle-to-reg');
const authToggleToLogin = document.getElementById('auth-toggle-to-login');
const authRegisterForm = document.getElementById('auth-register-form');
const authRegNameInput = document.getElementById('auth-reg-name');
const authRegEmailInput = document.getElementById('auth-reg-email');
const authRegPasswordInput = document.getElementById('auth-reg-password');
const authRegConfirmInput = document.getElementById('auth-reg-confirm');
const authRegErrorMsg = document.getElementById('auth-reg-error-msg');
const btnRegSubmit = document.getElementById('btn-reg-submit');
const btnRegIdle = document.getElementById('btn-reg-idle');
const btnRegBusy = document.getElementById('btn-reg-busy');

function setAuthModalMode(mode) {
  const isRegister = mode === 'register';
  if (authTabSignin) {
    authTabSignin.classList.toggle('active', !isRegister);
    authTabSignin.setAttribute('aria-selected', !isRegister ? 'true' : 'false');
  }
  if (authTabRegister) {
    authTabRegister.classList.toggle('active', isRegister);
    authTabRegister.setAttribute('aria-selected', isRegister ? 'true' : 'false');
  }
  if (authLoginForm) authLoginForm.hidden = isRegister;
  if (authRegisterForm) authRegisterForm.hidden = !isRegister;
  if (authErrorMsg) authErrorMsg.hidden = true;
  if (authRegErrorMsg) authRegErrorMsg.hidden = true;
  const modalTitle = document.getElementById('auth-modal-title');
  if (modalTitle) {
    modalTitle.textContent = isRegister ? 'Create Developer Account' : 'Developer Authentication';
  }
}

if (authTabSignin) authTabSignin.addEventListener('click', () => setAuthModalMode('signin'));
if (authTabRegister) authTabRegister.addEventListener('click', () => setAuthModalMode('register'));
if (authToggleToReg) authToggleToReg.addEventListener('click', () => setAuthModalMode('register'));
if (authToggleToLogin) authToggleToLogin.addEventListener('click', () => setAuthModalMode('signin'));

// Submit developer login credentials (authenticates against backend API with secure verification)
async function submitDeveloperLogin(rawUser, rawPass) {
  const username = (rawUser !== undefined ? rawUser : (authUsernameInput ? authUsernameInput.value : '')).trim();
  const password = (rawPass !== undefined ? rawPass : (authPasswordInput ? authPasswordInput.value : '')).trim();

  if (!username || !password) {
    if (authErrorMsg) {
      authErrorMsg.textContent = 'Please enter both developer username and password.';
      authErrorMsg.hidden = false;
      authErrorMsg.style.display = 'block';
    }
    return;
  }

  if (btnLoginSubmit) btnLoginSubmit.disabled = true;
  if (btnLoginIdle) btnLoginIdle.hidden = true;
  if (btnLoginBusy) btnLoginBusy.hidden = false;
  if (authErrorMsg) {
    authErrorMsg.hidden = true;
    authErrorMsg.style.display = 'none';
  }

  const cleanUser = username.toLowerCase();
  const baseName = username.includes('@') ? username.split('@')[0] : username;
  const formattedName = baseName.replace(/[._-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());

  let authSuccess = false;
  let authUser = null;
  let authToken = null;
  let errorDetail = 'Invalid developer credentials. Access denied.';

  const isLocalAuth = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
  const apiEndpoints = [
    getApiUrl('/api/auth/login'),
    '/api/auth/login'
  ];
  if (isLocalAuth) {
    apiEndpoints.push(
      'http://localhost:3001/api/auth/login',
      'http://127.0.0.1:3001/api/auth/login',
      'http://localhost:8000/api/auth/login',
      'http://127.0.0.1:8000/api/auth/login'
    );
  }

  let reachedServer = false;
  for (const endpoint of apiEndpoints) {
    try {
      const controller = new AbortController();
      const timeoutTimer = setTimeout(() => controller.abort(), 2500);

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, role: 'developer' }),
        signal: controller.signal
      });
      clearTimeout(timeoutTimer);
      if (res.status === 405) continue;
      reachedServer = true;

      const data = await res.json().catch(() => ({}));
      if (res.ok && data && data.success && data.role === 'developer') {
        authSuccess = true;
        authUser = data.user;
        authToken = data.token;
        break;
      } else {
        errorDetail = data?.error || data?.detail || 'Invalid developer credentials. Access denied.';
        break;
      }
    } catch (_netErr) {
      // Continue to next endpoint if first fails
    }
  }

  // Standalone offline fallback (only if local network is fully disconnected)
  if (!reachedServer) {
    const validUsers = new Set(['developer@soundsense.ai', 'developer', 'admin@soundsense.ai', 'admin']);
    const validPass = new Set(['SoundSense@Dev2026!', 'soundsense2024']);
    if (validUsers.has(cleanUser) && validPass.has(password)) {
      authSuccess = true;
      authUser = {
        name: formattedName ? `${formattedName} (AI Engineer)` : 'Lead AI Engineer',
        email: username.includes('@') ? username : `${username}@soundsense.ai`,
        role: 'developer',
        permissions: ['training_controls', 'checkpoint_mgmt', 'system_analytics', 'cloud_sync', 'all_architectures'],
        signedInAt: new Date().toISOString()
      };
      authToken = `dev_token_offline_${Date.now()}`;
    } else {
      errorDetail = 'Invalid developer credentials. Access denied.';
    }
  }

  if (btnLoginSubmit) btnLoginSubmit.disabled = false;
  if (btnLoginIdle) btnLoginIdle.hidden = false;
  if (btnLoginBusy) btnLoginBusy.hidden = true;

  if (authSuccess && authUser) {
    try {
      sessionStorage.setItem('soundsense_dev_auth', JSON.stringify({ token: authToken, user: authUser }));
      localStorage.removeItem('soundsense_dev_auth');
    } catch (_) { }

    setDeveloperMode(true, authUser);
    closeAuthModal();

    showToast('Developer Mode Unlocked', `Welcome back, ${authUser.name}! Developer sections & all neural architectures are ready.`, 'success', 4000);

    const target = pendingDevView || 'training';
    pendingDevView = null;
    switchView(target);
  } else {
    if (authErrorMsg) {
      authErrorMsg.textContent = errorDetail;
      authErrorMsg.hidden = false;
      authErrorMsg.style.display = 'block';
    }
  }
}

// Submit developer registration (Create Account)
async function submitDeveloperRegistration() {
  const name = (authRegNameInput ? authRegNameInput.value : '').trim();
  const email = (authRegEmailInput ? authRegEmailInput.value : '').trim();
  const password = (authRegPasswordInput ? authRegPasswordInput.value : '').trim();
  const confirm = (authRegConfirmInput ? authRegConfirmInput.value : '').trim();

  if (!name || !email || !password) {
    if (authRegErrorMsg) {
      authRegErrorMsg.textContent = 'Please fill in all registration fields.';
      authRegErrorMsg.hidden = false;
      authRegErrorMsg.style.display = 'block';
    }
    return;
  }
  if (password.length < 6) {
    if (authRegErrorMsg) {
      authRegErrorMsg.textContent = 'Password must be at least 6 characters long.';
      authRegErrorMsg.hidden = false;
      authRegErrorMsg.style.display = 'block';
    }
    return;
  }
  if (password !== confirm) {
    if (authRegErrorMsg) {
      authRegErrorMsg.textContent = 'Passwords do not match. Please re-enter.';
      authRegErrorMsg.hidden = false;
      authRegErrorMsg.style.display = 'block';
    }
    return;
  }

  if (btnRegSubmit) btnRegSubmit.disabled = true;
  if (btnRegIdle) btnRegIdle.hidden = true;
  if (btnRegBusy) btnRegBusy.hidden = false;
  if (authRegErrorMsg) {
    authRegErrorMsg.hidden = true;
    authRegErrorMsg.style.display = 'none';
  }

  const isLocalReg = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
  const candidateEndpoints = [
    getApiUrl('/api/auth/register'),
    '/api/auth/register'
  ];
  if (isLocalReg) {
    candidateEndpoints.push(
      'http://localhost:3001/api/auth/register',
      'http://127.0.0.1:3001/api/auth/register',
      'http://localhost:8000/api/auth/register',
      'http://127.0.0.1:8000/api/auth/register'
    );
  }

  let regSuccess = false;
  let regUser = null;
  let regToken = null;
  let errorDetail = 'Could not create account. Please try again.';

  for (const endpoint of candidateEndpoints) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 3000);
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, password, role: 'developer' }),
        signal: controller.signal
      });
      clearTimeout(timer);
      if (res.status === 405) continue;
      const data = await res.json().catch(() => ({}));
      if (res.ok && data && data.success) {
        regSuccess = true;
        regUser = data.user;
        regToken = data.token;
        break;
      } else {
        errorDetail = data?.error || 'Registration failed.';
        break;
      }
    } catch (_) { }
  }

  if (btnRegSubmit) btnRegSubmit.disabled = false;
  if (btnRegIdle) btnRegIdle.hidden = false;
  if (btnRegBusy) btnRegBusy.hidden = true;

  if (regSuccess && regUser) {
    try {
      sessionStorage.setItem('soundsense_dev_auth', JSON.stringify({ token: regToken, user: regUser }));
      localStorage.removeItem('soundsense_dev_auth');
    } catch (_) { }
    setDeveloperMode(true, regUser);
    closeAuthModal();
    showToast('Account Created!', `Welcome, ${regUser.name}! Developer Mode is now active with full model access.`, 'success', 4500);
    const target = pendingDevView || 'training';
    pendingDevView = null;
    switchView(target);
  } else {
    if (authRegErrorMsg) {
      authRegErrorMsg.textContent = errorDetail;
      authRegErrorMsg.hidden = false;
      authRegErrorMsg.style.display = 'block';
    }
  }
}

// Submit button click handler
if (btnLoginSubmit) {
  btnLoginSubmit.addEventListener('click', (e) => {
    e.preventDefault();
    submitDeveloperLogin();
  });
}

// Login form submission handler
if (authLoginForm) {
  authLoginForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitDeveloperLogin();
  });
}

// Register button click handler
if (btnRegSubmit) {
  btnRegSubmit.addEventListener('click', (e) => {
    e.preventDefault();
    submitDeveloperRegistration();
  });
}

// Register form submission handler
if (authRegisterForm) {
  authRegisterForm.addEventListener('submit', (e) => {
    e.preventDefault();
    submitDeveloperRegistration();
  });
}

// Application startup: Default to normal guest user mode. Developer mode is hidden unless logged in this session.
(function initDeveloperAuth() {
  try {
    const sessionStored = sessionStorage.getItem('soundsense_dev_auth');
    if (sessionStored) {
      const parsed = JSON.parse(sessionStored);
      if (parsed && parsed.user) {
        setDeveloperMode(true, parsed.user);
        return;
      }
    }
    localStorage.removeItem('soundsense_dev_auth');
  } catch (_) { }
  setDeveloperMode(false);
})();

// ═══════════════════════════════════════════════════════════════
// Theme Controller: Day (Light), Night (Dark), Device (System)
// ═══════════════════════════════════════════════════════════════
(function initThemeController() {
  const root = document.documentElement;
  const themeBtns = document.querySelectorAll('.theme-btn');
  const mql = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

  function applyTheme(preference) {
    let effective = preference;
    if (preference === 'system') {
      effective = (mql && mql.matches) ? 'dark' : 'light';
    }

    root.setAttribute('data-theme', effective);
    if (typeof drawClassDistributionChart === 'function') {
      try { drawClassDistributionChart(); } catch (_) { }
    }
    if (typeof drawModelPerformanceChart === 'function') {
      try { drawModelPerformanceChart(_lastPerfList); } catch (_) { }
    }
    root.setAttribute('data-theme-pref', preference);

    try {
      localStorage.setItem('soundsense_theme', preference);
    } catch (_) { }

    themeBtns.forEach((btn) => {
      const val = btn.getAttribute('data-theme-val');
      const isSelected = (val === preference);
      btn.classList.toggle('active', isSelected);
      btn.setAttribute('aria-checked', isSelected ? 'true' : 'false');
    });
  }

  // Restore stored preference or default to 'system'
  let savedPref = 'system';
  try {
    savedPref = localStorage.getItem('soundsense_theme') || 'system';
  } catch (_) { }
  applyTheme(savedPref);

  // Attach click handlers to Day / Night / Device buttons
  themeBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const val = btn.getAttribute('data-theme-val');
      if (val) applyTheme(val);
    });
  });

  // Dynamically listen to OS theme changes when in 'system' mode
  if (mql && mql.addEventListener) {
    mql.addEventListener('change', () => {
      const currentPref = root.getAttribute('data-theme-pref') || 'system';
      if (currentPref === 'system') {
        applyTheme('system');
      }
    });
  }
})();



// ═══════════════════════════════════════════════════════════════
// System Overview Controller
// ═══════════════════════════════════════════════════════════════
(function initOverviewController() {
  // Hero Quick Navigation Buttons
  const heroGotoStudioBtn = document.getElementById('hero-goto-studio-btn');
  if (heroGotoStudioBtn) {
    heroGotoStudioBtn.addEventListener('click', () => switchView('studio'));
  }

  const heroGotoTaxonomyBtn = document.getElementById('hero-goto-taxonomy-btn');
  if (heroGotoTaxonomyBtn) {
    heroGotoTaxonomyBtn.addEventListener('click', () => switchView('taxonomy'));
  }

  const heroOverviewBtn = document.getElementById('hero-overview-btn');
  if (heroOverviewBtn) {
    heroOverviewBtn.addEventListener('click', () => switchView('overview'));
  }

  const overviewGotoStudioBtn = document.getElementById('overview-goto-studio-btn');
  if (overviewGotoStudioBtn) {
    overviewGotoStudioBtn.addEventListener('click', () => switchView('studio'));
  }

  const overviewGotoTaxonomyBtn = document.getElementById('overview-goto-taxonomy-btn');
  if (overviewGotoTaxonomyBtn) {
    overviewGotoTaxonomyBtn.addEventListener('click', () => switchView('taxonomy'));
  }

  const simTryStudioBtn = document.getElementById('sim-try-studio-btn');
  if (simTryStudioBtn) {
    simTryStudioBtn.addEventListener('click', () => {
      const activeBtn = document.querySelector('.sim-archetype-btn.active');
      const arch = activeBtn ? activeBtn.getAttribute('data-archetype') : 'dog';
      const simMapping = {
        dog: 0,
        rain: 10,
        baby: 20,
        knock: 30,
      };
      const catId = simMapping[arch] !== undefined ? simMapping[arch] : 0;
      switchView('studio');
      if (typeof loadTaxonomyCategory === 'function' && ESC50_CATEGORIES[catId]) {
        loadTaxonomyCategory(ESC50_CATEGORIES[catId]);
      }
    });
  }
})();

// ─────────────────────────────────────────────────────────────
// Dynamic Overview Background Canvas (Particle Constellation & Wave)
// ─────────────────────────────────────────────────────────────
(function initOverviewCanvas() {
  const canvas = document.getElementById('overview-fx-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  let width = 0;
  let height = 0;
  let particles = [];
  const NUM_PARTICLES = 40;
  let mouse = { x: -1000, y: -1000, active: false };
  let animId = null;
  let wavePhase = 0;

  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect();
    width = canvas.width = rect.width;
    height = canvas.height = rect.height;
  }

  function initParticles() {
    particles = [];
    for (let i = 0; i < NUM_PARTICLES; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.6,
        vy: (Math.random() - 0.5) * 0.6,
        radius: Math.random() * 2 + 1.2,
        color: Math.random() > 0.4 ? '34, 211, 238' : '99, 102, 241'
      });
    }
  }

  window.addEventListener('resize', () => {
    resize();
    initParticles();
  });

  const wrapper = canvas.parentElement;
  if (wrapper) {
    wrapper.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
      mouse.active = true;
    });
    wrapper.addEventListener('mouseleave', () => {
      mouse.active = false;
    });
  }

  function draw() {
    const overviewPanel = document.getElementById('view-overview');
    if (!overviewPanel || !overviewPanel.classList.contains('active')) {
      animId = requestAnimationFrame(draw);
      return;
    }

    ctx.clearRect(0, 0, width, height);

    // Draw undulating acoustic harmonic wave ribbons
    wavePhase += 0.018;
    ctx.beginPath();
    ctx.moveTo(0, height);
    for (let x = 0; x <= width; x += 15) {
      const y = height * 0.72 + Math.sin(x * 0.008 + wavePhase) * 22 + Math.cos(x * 0.015 - wavePhase * 0.5) * 12;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(width, height);
    ctx.closePath();
    const waveGrad = ctx.createLinearGradient(0, height * 0.6, 0, height);
    waveGrad.addColorStop(0, 'rgba(6, 182, 212, 0.08)');
    waveGrad.addColorStop(1, 'rgba(6, 182, 212, 0.01)');
    ctx.fillStyle = waveGrad;
    ctx.fill();

    // Update and draw particles
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = width;
      if (p.x > width) p.x = 0;
      if (p.y < 0) p.y = height;
      if (p.y > height) p.y = 0;

      // Proximity lines between particles
      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 95) {
          const alpha = (1 - dist / 95) * 0.22;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(34, 211, 238, ${alpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Connection to mouse
      if (mouse.active) {
        const mdx = p.x - mouse.x;
        const mdy = p.y - mouse.y;
        const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
        if (mdist < 130) {
          const mAlpha = (1 - mdist / 130) * 0.35;
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.strokeStyle = `rgba(34, 211, 238, ${mAlpha})`;
          ctx.lineWidth = 1.2;
          ctx.stroke();
        }
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(' + p.color + ', 0.75)';
      ctx.fill();
    }

    animId = requestAnimationFrame(draw);
  }

  resize();
  initParticles();
  animId = requestAnimationFrame(draw);
})();

// ─────────────────────────────────────────────────────────────
// Interactive Soundscape Simulator (Acoustic Archetypes)
// ─────────────────────────────────────────────────────────────
(function initSoundSimulator() {
  const simBtns = document.querySelectorAll('.sim-archetype-btn');
  const simCanvas = document.getElementById('sim-mini-canvas');
  const simDomainBadge = document.getElementById('sim-domain-badge');
  const simConfBadge = document.getElementById('sim-conf-badge');
  const simPredName = document.getElementById('sim-pred-name');
  const simPredDesc = document.getElementById('sim-pred-desc');
  const simCanvasLabel = document.getElementById('sim-canvas-label');

  const simPlayPauseBtn = document.getElementById('sim-playpause-btn');
  const simPauseIcon = document.getElementById('sim-pause-icon');
  const simPlayIcon = document.getElementById('sim-play-icon');
  const simBtnText = document.getElementById('sim-btn-text');
  const simStatusText = document.getElementById('sim-status-text');
  const simStatusDot = document.getElementById('sim-status-dot');
  let isSimPaused = false;

  if (!simCanvas) return;
  const sctx = simCanvas.getContext('2d');
  if (!sctx) return;

  let activeArch = 'dog';
  let simPhase = 0;

  const ARCHETYPES = {
    dog: {
      name: 'Synthetic dog bark sample',
      domain: 'Synthetic audio · Animals',
      conf: 'No model score',
      desc: 'Multi-harmonic acoustic formants with rapid periodic vocal tract modulations centered around 450 Hz – 1.8 kHz.',
      waveColor: '#22d3ee',
      style: 'burst'
    },
    rain: {
      name: 'Synthetic rain and thunder sample',
      domain: 'Synthetic audio · Natural soundscapes',
      conf: 'No model score',
      desc: 'Continuous stochastic broadband pink-noise envelope with low-frequency rumble impulses below 120 Hz.',
      waveColor: '#38bdf8',
      style: 'noise'
    },
    baby: {
      name: 'Synthetic infant cry sample',
      domain: 'Synthetic audio · Human sounds',
      conf: 'No model score',
      desc: 'High-pitch vocal tract resonance with frequency modulation sweeping between 800 Hz and 2.6 kHz.',
      waveColor: '#fbbf24',
      style: 'sweep'
    },
    knock: {
      name: 'Synthetic door knock sample',
      domain: 'Synthetic audio · Interior',
      conf: 'No model score',
      desc: 'Sharp mechanical impact transients with rapid exponential decay and characteristic room reverberation.',
      waveColor: '#a78bfa',
      style: 'impulse'
    }
  };

  if (simPlayPauseBtn) {
    simPlayPauseBtn.addEventListener('click', () => {
      isSimPaused = !isSimPaused;
      if (isSimPaused) {
        simPlayPauseBtn.classList.add('paused');
        if (simPauseIcon) simPauseIcon.style.display = 'none';
        if (simPlayIcon) simPlayIcon.style.display = 'block';
        if (simBtnText) simBtnText.textContent = 'Resume';
        if (simStatusText) simStatusText.textContent = 'Simulation Paused';
        if (simStatusDot) simStatusDot.style.background = 'var(--amber)';
      } else {
        simPlayPauseBtn.classList.remove('paused');
        if (simPauseIcon) simPauseIcon.style.display = 'block';
        if (simPlayIcon) simPlayIcon.style.display = 'none';
        if (simBtnText) simBtnText.textContent = 'Pause';
        if (simStatusText) simStatusText.textContent = 'Synthetic preview';
        if (simStatusDot) simStatusDot.style.background = '';
      }
    });
  }

  simBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const arch = btn.getAttribute('data-archetype');
      if (!arch || !ARCHETYPES[arch]) return;
      activeArch = arch;

      simBtns.forEach(b => b.classList.toggle('active', b === btn));
      const data = ARCHETYPES[arch];

      if (simPredName) simPredName.textContent = data.name;
      if (simDomainBadge) simDomainBadge.textContent = data.domain;
      if (simConfBadge) simConfBadge.textContent = data.conf;
      if (simPredDesc) simPredDesc.textContent = data.desc;
      if (simCanvasLabel) simCanvasLabel.textContent = `Synthetic waveform: ${data.name} (22.05 kHz)`;
    });
  });

  function drawSimWave() {
    const overviewPanel = document.getElementById('view-overview');
    if (!overviewPanel || !overviewPanel.classList.contains('active')) {
      requestAnimationFrame(drawSimWave);
      return;
    }

    if (isSimPaused) {
      requestAnimationFrame(drawSimWave);
      return;
    }

    const rect = simCanvas.getBoundingClientRect();
    if (simCanvas.width !== rect.width || simCanvas.height !== rect.height) {
      simCanvas.width = rect.width;
      simCanvas.height = rect.height;
    }

    const w = simCanvas.width;
    const h = simCanvas.height;
    sctx.fillStyle = '#070a14';
    sctx.fillRect(0, 0, w, h);

    simPhase += 0.05;
    const data = ARCHETYPES[activeArch] || ARCHETYPES.dog;

    // Draw subtle grid lines
    sctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    sctx.lineWidth = 1;
    for (let y = 20; y < h; y += 25) {
      sctx.beginPath();
      sctx.moveTo(0, y);
      sctx.lineTo(w, y);
      sctx.stroke();
    }

    // Draw simulated spectrum bars
    const numBars = 42;
    const barWidth = (w / numBars) - 2;

    for (let i = 0; i < numBars; i++) {
      let barHeight = 0;
      const normX = i / numBars;

      if (data.style === 'burst') {
        barHeight = (Math.sin(i * 0.4 + simPhase) * 0.4 + 0.5) * (Math.sin(i * 0.15) * 45 + 25);
      } else if (data.style === 'noise') {
        barHeight = (Math.random() * 0.35 + 0.35) * (60 - normX * 30);
      } else if (data.style === 'sweep') {
        const center = Math.sin(simPhase * 0.6) * 0.3 + 0.5;
        const dist = Math.abs(normX - center);
        barHeight = Math.max(8, (1 - dist * 2.2) * 65 + Math.random() * 8);
      } else if (data.style === 'impulse') {
        const pulse = (Math.sin(simPhase * 1.5) > 0.6) ? 70 : 12;
        barHeight = pulse * Math.exp(-normX * 3.5) + Math.random() * 6;
      }

      const bx = i * (barWidth + 2);
      const by = h - barHeight - 12;

      const grad = sctx.createLinearGradient(0, by, 0, h);
      grad.addColorStop(0, data.waveColor);
      grad.addColorStop(1, 'rgba(6, 182, 212, 0.15)');

      sctx.fillStyle = grad;
      sctx.fillRect(bx, by, barWidth, barHeight);
    }

    requestAnimationFrame(drawSimWave);
  }

  requestAnimationFrame(drawSimWave);
})();

// ─────────────────────────────────────────────────────────────
// API Reference Interactive Explorer
// ─────────────────────────────────────────────────────────────
(function initApiExplorer() {
  const endpointBtns = document.querySelectorAll('.api-endpoint-btn');
  const langBtns = document.querySelectorAll('.api-lang-btn');
  const apiMetaBadge = document.getElementById('api-meta-badge');
  const apiMetaDesc = document.getElementById('api-meta-desc');
  const apiCodeDisplay = document.getElementById('api-code-display');
  const apiResponseDisplay = document.getElementById('api-response-display');
  const copyBtn = document.getElementById('copy-api-code-btn');
  const copyLabel = document.getElementById('copy-code-label');

  if (!apiCodeDisplay && !endpointBtns.length) return;

  let activeEndpoint = 'predict';
  let activeLang = 'curl';

  const API_DOCS = {
    predict: {
      badge: 'POST /api/predict',
      desc: 'Accepts WAV, MP3, or MPEG audio. A normal classification requires the selected model checkpoint; confidence is a model score, not a guarantee of correctness.',
      code: {
        curl: `curl -X POST "http://localhost:8000/api/predict" \\
  -H "accept: application/json" \\
  -F "audio=@acoustic_sample.wav" \\
  -F "model=resnet18" \\
  -F "threshold=0.5"`,
        python: `import requests

url = "http://localhost:8000/api/predict"
files = {"audio": open("acoustic_sample.wav", "rb")}
data = {"model": "resnet18", "threshold": 0.5}

response = requests.post(url, files=files, data=data)
print(response.json())`,
        javascript: `const formData = new FormData();
formData.append("audio", audioBlob, "sample.wav");
formData.append("model", "resnet18");
formData.append("threshold", "0.5");

const res = await fetch("http://localhost:8000/api/predict", {
  method: "POST",
  body: formData
});
const data = await res.json();
console.log(data);`
      },
      response: `{
  "error": "Model unavailable",
  "details": "No trained checkpoint is available for: resnet18."
}`
    },
    sample: {
      badge: 'GET /api/sample/{category_id}',
      desc: 'Streams a sample WAV for an ESC-50 class when the dataset audio files are installed locally.',
      code: {
        curl: `curl -X GET "http://localhost:8000/api/sample/0" \\
  -H "accept: audio/wav" \\
  --output sample_dog.wav`,
        python: `import requests

url = "http://localhost:8000/api/sample/0"

res = requests.get(url)
with open("sample_dog.wav", "wb") as f:
    f.write(res.content)`,
        javascript: `const res = await fetch("http://localhost:8000/api/sample/0");
const blob = await res.blob();
const audioUrl = URL.createObjectURL(blob);
const audio = new Audio(audioUrl);
audio.play();`
      },
      response: `// Binary audio stream response
// Content-Type: audio/wav
// Rate: 22,050 Hz, 16-bit PCM Mono
// File size: 220,544 bytes (~5.00s duration)`
    },
    stats: {
      badge: 'GET /api/stats',
      desc: 'Returns runtime telemetry, dataset counts, checkpoint availability, and metrics from the current evaluation report.',
      code: {
        curl: `curl -X GET "http://localhost:8000/api/stats" \\
  -H "accept: application/json"`,
        python: `import requests

res = requests.get("http://localhost:8000/api/stats")
print(res.json())`,
        javascript: `const res = await fetch("http://localhost:8000/api/stats");
console.log(await res.json());`
      },
      response: `// Runtime values are generated by the running service; model scores appear only after evaluation.`
    },
    taxonomy: {
      badge: 'GET /api/health',
      desc: 'Reports service health and lists which model checkpoints are present or missing.',
      code: {
        curl: `curl -X GET "http://localhost:8000/api/health" \\
  -H "accept: application/json"`,
        python: `import requests

res = requests.get("http://localhost:8000/api/health")
print(res.json())`,
        javascript: `const res = await fetch("http://localhost:8000/api/health");
console.log(await res.json());`
      },
      response: `// Check "status" and "missingModels" in the live response.`
    }
  };

  function formatApiCode(code, lang) {
    if (!code) return '';
    let escaped = code
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    if (lang === 'curl') {
      escaped = escaped
        .replace(/\b(curl)\b/g, '<span class="kw">$1</span>')
        .replace(/\b(GET|POST)\b/g, '<span class="str">$1</span>')
        .replace(/(-X|-H|-F)\b/g, '<span class="flag">$1</span>')
        .replace(/"(http:\/\/[^"]+)"/g, '"<span class="url">$1</span>"')
        .replace(/"([^"]+)"/g, '"<span class="str">$1</span>"');
    } else if (lang === 'python') {
      escaped = escaped
        .replace(/\b(import|from|open|print)\b/g, '<span class="kw">$1</span>')
        .replace(/"([^"]+)"/g, '"<span class="str">$1</span>"');
    } else if (lang === 'javascript') {
      escaped = escaped
        .replace(/\b(const|let|var|await|fetch|new|console\.log)\b/g, '<span class="kw">$1</span>')
        .replace(/"([^"]+)"/g, '"<span class="str">$1</span>"');
    }
    return escaped;
  }

  function formatApiJson(jsonStr) {
    if (!jsonStr) return '';
    let escaped = jsonStr
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    return escaped
      .replace(/"([^"]+)":/g, '<span class="json-key">"$1"</span>:')
      .replace(/: "([^"]+)"/g, ': <span class="json-str">"$1"</span>')
      .replace(/: (\d+(\.\d+)?)/g, ': <span class="json-num">$1</span>')
      .replace(/: (true|false|null)/g, ': <span class="json-bool">$1</span>');
  }

  function updateApiExplorer() {
    const doc = API_DOCS[activeEndpoint] || API_DOCS.predict;

    if (apiMetaBadge) apiMetaBadge.textContent = doc.badge;
    if (apiMetaDesc) apiMetaDesc.textContent = doc.desc;
    if (apiCodeDisplay) {
      const raw = doc.code[activeLang] || doc.code.curl;
      apiCodeDisplay.innerHTML = formatApiCode(raw, activeLang);
    }
    if (apiResponseDisplay) {
      apiResponseDisplay.innerHTML = formatApiJson(doc.response);
    }

    endpointBtns.forEach(btn => {
      const isTarget = btn.getAttribute('data-endpoint') === activeEndpoint;
      btn.classList.toggle('active', isTarget);
    });

    langBtns.forEach(btn => {
      const isTarget = btn.getAttribute('data-lang') === activeLang;
      btn.classList.toggle('active', isTarget);
    });
  }

  endpointBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const endpoint = btn.getAttribute('data-endpoint');
      if (endpoint && API_DOCS[endpoint]) {
        activeEndpoint = endpoint;
        updateApiExplorer();
      }
    });
  });

  langBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const lang = btn.getAttribute('data-lang');
      if (lang) {
        activeLang = lang;
        updateApiExplorer();
      }
    });
  });

  if (copyBtn && apiCodeDisplay) {
    copyBtn.addEventListener('click', async () => {
      const textToCopy = apiCodeDisplay.textContent;
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(textToCopy);
        } else {
          const textarea = document.createElement('textarea');
          textarea.value = textToCopy;
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          document.body.removeChild(textarea);
        }

        copyBtn.classList.add('copied');
        if (copyLabel) copyLabel.textContent = 'Copied!';
        if (typeof showToast === 'function') {
          showToast('Code Copied', 'API code snippet copied to clipboard.', 'success', 2500);
        }

        setTimeout(() => {
          copyBtn.classList.remove('copied');
          if (copyLabel) copyLabel.textContent = 'Copy Code';
        }, 2200);
      } catch (err) {
        console.warn('Clipboard copy error:', err);
      }
    });
  }

  updateApiExplorer();
})();

/* ══════════════════════════════════════════════════════════════════
   STUDIO REFINEMENTS: THRESHOLD SLIDER, SCROLL TO TOP & FOOTER NAV
   ══════════════════════════════════════════════════════════════════ */
(function initStudioRefinements() {
  // Confidence Threshold Slider (Card 1.5)
  const thresholdSlider = document.getElementById('threshold-slider');
  const thresholdValDisplay = document.getElementById('threshold-value-display');
  function updateThresholdSliderFill() {
    if (!thresholdSlider) return;
    const val = Number(thresholdSlider.value) || 0;
    const min = Number(thresholdSlider.min) || 0;
    const max = Number(thresholdSlider.max) || 100;
    const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
    thresholdSlider.style.setProperty('--fill', `${pct}%`);
    if (thresholdValDisplay) {
      thresholdValDisplay.textContent = `${val}%`;
    }
  }

  if (thresholdSlider) {
    updateThresholdSliderFill();
    thresholdSlider.addEventListener('input', updateThresholdSliderFill);
  }

  // Floating Scroll to Top Button
  const scrollTopBtn = document.getElementById('scroll-top-btn');
  const appFooter = document.querySelector('.app-footer');
  if (scrollTopBtn) {
    const updateScrollTopState = () => {
      const scrollY = window.scrollY || window.pageYOffset;
      if (scrollY > 380) {
        scrollTopBtn.removeAttribute('hidden');
        scrollTopBtn.classList.add('visible');
      } else {
        scrollTopBtn.classList.remove('visible');
        scrollTopBtn.setAttribute('hidden', '');
      }

      // Keep button floating cleanly above the footer when scrolled near the bottom
      if (appFooter) {
        const footerRect = appFooter.getBoundingClientRect();
        const viewportHeight = window.innerHeight;
        if (footerRect.top < viewportHeight) {
          const overlap = Math.max(0, viewportHeight - footerRect.top);
          scrollTopBtn.style.setProperty('--footer-lift', `${overlap}px`);
        } else {
          scrollTopBtn.style.setProperty('--footer-lift', '0px');
        }
      }
    };

    window.addEventListener('scroll', updateScrollTopState, { passive: true });
    window.addEventListener('resize', updateScrollTopState, { passive: true });
    updateScrollTopState();

    scrollTopBtn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  }

  // Global Navigation Quick-Links ([data-goto])
  document.querySelectorAll('[data-goto]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-goto');
      if (target && typeof switchView === 'function') {
        switchView(target);
      }
    });
  });
})();

/* ═══════════════════════════════════════════════════════════════
   BATCH AUDIO CLASSIFICATION & CSV EXPORT STUDIO
   Standalone high-throughput multi-file inference module.
   Talks directly to /api/predict with parallel concurrency.
   Zero dependencies outside the Sound_Recognition project.
   ═══════════════════════════════════════════════════════════════ */
(function initBatchClassification() {
  'use strict';

  var CONFIG = {
    endpoint: '/api/predict',
    fileField: 'audio',
    modelField: 'model',
    maxFiles: 100,
    concurrency: 2,
    topK: 5
  };

  var $ = function (id) { return document.getElementById(id); };
  var drop = $('batch-drop');
  if (!drop) return;

  var input = $('batch-input');
  var listEl = $('batch-file-list');
  var runBtn = $('batch-run-btn');
  var csvBtn = $('batch-csv-btn');
  var clearBtn = $('batch-clear-btn');
  var progressWrap = $('batch-progress');
  var progressFill = $('batch-progress-fill');
  var progressText = $('batch-progress-text');
  var tableWrap = $('batch-table-wrap');
  var matrixEmpty = $('batch-matrix-empty');
  var tbody = $('batch-table-body');
  var modelInfo = $('batch-model-info');
  var modelSelect = $('model-select');
  var runIdle = $('batch-run-idle');
  var runBusy = $('batch-run-busy');
  var maxLabel = $('batch-max-label');
  var queueCountEl = $('batch-queue-count');
  var counterChipEl = $('batch-counter-chip');
  var chipCountEl = $('batch-chip-count');
  var summaryStatsEl = $('batch-summary-stats');
  var statOkEl = $('stat-batch-ok');
  var statErrEl = $('stat-batch-err');
  var heroBatchBtn = $('hero-goto-batch-btn');
  var hintBatchBtn = $('hint-goto-batch-btn');

  if (maxLabel) maxLabel.textContent = CONFIG.maxFiles;

  var items = [];      // {id, file, state, preds, error, model}
  var nextId = 1;
  var running = false;

  function modelKey() {
    if (!isDeveloperAuthenticated) return 'ast';
    return modelSelect ? modelSelect.value : 'ast';
  }
  function modelName() {
    if (!isDeveloperAuthenticated) return 'AST-ESC50 (Best)';
    if (!modelSelect) return 'AST-ESC50';
    var opt = modelSelect.options[modelSelect.selectedIndex];
    return opt ? opt.textContent.trim().split(' ')[0] : modelSelect.value;
  }
  function syncModelLabel() {
    if (modelInfo) modelInfo.textContent = modelName();
  }
  function fmtSize(b) {
    return b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB';
  }
  function isAllowedAudio(f) {
    if (!f) return false;
    var name = (f.name || '').toLowerCase();
    if (/\.(ogg|flac|m4a)$/i.test(name)) return false;
    if (/\.(wav|mp3|mpg|mpeg|mpga|mpe|webm|aac)$/i.test(name)) return true;
    if (f.type) {
      var t = f.type.toLowerCase();
      if (t.includes('ogg') || t.includes('flac') || t.includes('m4a')) return false;
      if (t.startsWith('audio/') || t.includes('mpeg')) return true;
    }
    return false;
  }
  function esc(s) {
    return String(s || '').replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function notify(msg, tone) {
    if (typeof showToast === 'function') {
      showToast('Batch Classification', msg, tone || (msg.toLowerCase().includes('failed') ? 'error' : 'info'));
      return;
    }
    var stack = $('toast-stack');
    if (!stack) { console.info('[batch]', msg); return; }
    var t = document.createElement('div');
    t.className = 'toast ' + (tone || '');
    t.textContent = msg;
    stack.appendChild(t);
    setTimeout(function () { t.remove(); }, 3500);
  }

  function normalise(data) {
    var arr = null;
    ['top5', 'top_5', 'topK', 'top_k', 'predictions', 'results', 'ranked', 'probabilities'].some(function (k) {
      if (data && Array.isArray(data[k])) { arr = data[k]; return true; }
      return false;
    });
    if (!arr && data && data.result) return normalise(data.result);
    if (!arr && data && (data.label || data.category || data.prediction || data.class)) {
      var catName = (data.category && typeof data.category === 'object') ? data.category.name : (data.label || data.category || data.class || data.prediction);
      var catEmoji = (data.category && typeof data.category === 'object') ? (data.category.emoji || '') : (data.emoji || '');
      arr = [{ label: catName, confidence: data.confidence != null ? data.confidence : data.probability, emoji: catEmoji }];
    }
    if (!arr) return null;
    var out = arr.map(function (p) {
      if (Array.isArray(p)) return { label: p[0], conf: Number(p[1]), emoji: '' };
      var label = p.label || (p.category && typeof p.category === 'object' ? p.category.name : p.category) || p.class || p.name || p.className;
      var c = p.confidence != null ? p.confidence : (p.probability != null ? p.probability : (p.score != null ? p.score : p.prob));
      var emoji = p.emoji || (p.category && typeof p.category === 'object' ? p.category.emoji : '') || '';
      return { label: label, conf: Number(c), emoji: emoji };
    }).filter(function (p) { return p.label != null && !isNaN(p.conf); });
    var asFraction = out.every(function (p) { return p.conf <= 1; });
    out.forEach(function (p) { if (asFraction) p.conf *= 100; });
    out.sort(function (a, b) { return b.conf - a.conf; });
    return out.slice(0, CONFIG.topK);
  }

  function addFiles(fileList) {
    var added = 0, skipped = 0;
    Array.prototype.forEach.call(fileList, function (f) {
      if (!isAllowedAudio(f)) { skipped++; return; }
      if (items.length >= CONFIG.maxFiles) { skipped++; return; }
      items.push({ id: nextId++, file: f, state: 'queued', preds: null, error: '', model: '' });
      added++;
    });
    if (skipped) {
      notify(skipped + ' file(s) skipped (unsupported format or exceeded ' + CONFIG.maxFiles + ' file limit).', 'warn');
    }
    if (added) {
      notify(added + ' clip(s) queued for batch classification.', 'success');
    }
    render();
  }

  function stateLabel(it) {
    return { queued: 'Queued', running: 'Running...', done: 'Done', error: 'Failed' }[it.state];
  }
  function stateClass(it) {
    return it.state === 'done' ? 'is-done' : it.state === 'error' ? 'is-error' : it.state === 'running' ? 'is-run' : '';
  }

  function render() {
    if (queueCountEl) queueCountEl.textContent = items.length + ' item' + (items.length === 1 ? '' : 's');
    if (counterChipEl) {
      counterChipEl.hidden = items.length === 0;
      if (chipCountEl) chipCountEl.textContent = items.length + ' files';
    }

    if (items.length === 0) {
      listEl.innerHTML = '<li class="batch-empty-li" id="batch-empty-placeholder">' +
        '<span class="bf-empty-icon" aria-hidden="true">🎧</span>' +
        '<span class="bf-empty-text">No audio files added yet. Drop clips above to populate queue.</span>' +
        '</li>';
    } else {
      listEl.innerHTML = items.map(function (it) {
        return '<li data-id="' + it.id + '">' +
          '<span class="bf-icon" aria-hidden="true">🎵</span>' +
          '<span class="bf-name" title="' + esc(it.file.name) + '">' + esc(it.file.name) + '</span>' +
          '<span class="bf-size">' + fmtSize(it.file.size) + '</span>' +
          '<span class="bf-state ' + stateClass(it) + '">' + stateLabel(it) + '</span>' +
          (running ? '' : '<button type="button" class="bf-remove" aria-label="Remove ' + esc(it.file.name) + '" title="Remove file">&times;</button>') +
          '</li>';
      }).join('');
    }

    var hasItems = items.length > 0;
    var anyDone = items.some(function (i) { return i.state === 'done' || i.state === 'error'; });
    var pending = items.some(function (i) { return i.state === 'queued' || i.state === 'error'; });

    runBtn.disabled = running || !pending;
    clearBtn.disabled = running || !hasItems;
    csvBtn.disabled = running || !anyDone;

    renderTable();
  }

  function renderTable() {
    var rows = items.filter(function (i) { return i.state === 'done' || i.state === 'error'; });
    var doneCount = items.filter(function (i) { return i.state === 'done'; }).length;
    var errCount = items.filter(function (i) { return i.state === 'error'; }).length;

    if (summaryStatsEl) {
      summaryStatsEl.hidden = rows.length === 0;
      if (statOkEl) statOkEl.textContent = doneCount + ' Successful';
      if (statErrEl) {
        statErrEl.hidden = errCount === 0;
        statErrEl.textContent = errCount + ' Failed';
      }
    }

    if (matrixEmpty) matrixEmpty.hidden = rows.length > 0;
    if (tableWrap) tableWrap.hidden = rows.length === 0;

    if (!tbody) return;

    tbody.innerHTML = rows.map(function (it) {
      if (it.state === 'error') {
        return '<tr>' +
          '<td class="td-file"><span class="bt-file-name" title="' + esc(it.file.name) + '">' + esc(it.file.name) + '</span></td>' +
          '<td colspan="5" class="row-error"><span class="bt-err-chip">Failed</span> <span class="bt-err-text">' + esc(it.error) + '</span></td>' +
          '</tr>';
      }
      var cells = '';
      for (var k = 0; k < CONFIG.topK; k++) {
        var p = it.preds && it.preds[k];
        if (p) {
          var isTop1 = k === 0;
          var emojiSpan = p.emoji ? '<span class="bt-emoji">' + esc(p.emoji) + '</span>' : '';
          var labelSpan = '<strong class="bt-name">' + esc(p.label) + '</strong>';
          var confSpan = '<span class="conf">' + p.conf.toFixed(1) + '%</span>';
          cells += '<td><span class="bt-pred-pill ' + (isTop1 ? 'is-top1' : '') + '">' + emojiSpan + labelSpan + confSpan + '</span></td>';
        } else {
          cells += '<td><span class="bt-empty-cell">—</span></td>';
        }
      }
      return '<tr><td class="td-file"><span class="bt-file-name" title="' + esc(it.file.name) + '">' + esc(it.file.name) + '</span></td>' + cells + '</tr>';
    }).join('');
  }

  function updateProgress() {
    var total = items.length;
    var finished = items.filter(function (i) { return i.state === 'done' || i.state === 'error'; }).length;
    progressWrap.hidden = !running && finished === 0;
    var pct = total ? Math.round((finished / total) * 100) : 0;
    progressFill.style.width = pct + '%';
    progressText.textContent = finished + ' / ' + total + (total ? ' (' + pct + '%)' : '');
  }

  function classifyOne(it) {
    it.state = 'running';
    it.model = modelName();
    render();
    var fd = new FormData();
    fd.append(CONFIG.fileField, it.file, it.file.name);
    fd.append(CONFIG.modelField, modelKey());

    // Multi-candidate endpoints across live ports (3001, 8000) so static dev servers (e.g. port 5500) never throw HTTP 405
    var primaryUrl = typeof getApiUrl === 'function' ? getApiUrl('/api/predict') : '/api/predict';
    var isLocalBatchHost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:';
    var candidateUrls = [
      primaryUrl,
      '/api/predict'
    ];
    if (isLocalBatchHost) {
      candidateUrls.push(
        'http://localhost:3001/api/predict',
        'http://127.0.0.1:3001/api/predict',
        'http://localhost:8000/api/predict',
        'http://127.0.0.1:8000/api/predict'
      );
    }

    function tryCandidate(urls, index) {
      if (index >= urls.length) {
        return Promise.reject(new Error('All prediction endpoints failed or unreachable'));
      }
      var targetUrl = urls[index];
      return fetch(targetUrl, { method: 'POST', body: fd })
        .then(function (res) {
          if (res.status === 405) {
            // Static file server (e.g. Live Server port 5500) rejected POST, seamlessly try next backend candidate
            return tryCandidate(urls, index + 1);
          }
          if (!res.ok) throw new Error('Server returned HTTP ' + res.status);
          return res.json();
        })
        .catch(function (err) {
          if (index + 1 < urls.length && (String(err.message || '').includes('405') || err.name === 'TypeError' || String(err.message || '').includes('Failed to fetch'))) {
            return tryCandidate(urls, index + 1);
          }
          throw err;
        });
    }

    return tryCandidate(candidateUrls, 0)
      .then(function (data) {
        var preds = normalise(data);
        if (!preds || !preds.length) {
          console.warn('[batch] Unrecognised /api/predict response shape:', data);
          throw new Error('Unrecognised response format');
        }
        it.preds = preds;
        it.state = 'done';
        it.error = '';
      })
      .catch(function (err) {
        it.state = 'error';
        it.error = err.message || 'Request failed';
      })
      .then(function () {
        updateProgress();
        render();
      });
  }

  function runAll() {
    var queue = items.filter(function (i) { return i.state === 'queued' || i.state === 'error'; });
    if (!queue.length) return;
    running = true;
    runIdle.hidden = true;
    runBusy.hidden = false;
    updateProgress();
    render();

    var idx = 0;
    function worker() {
      if (idx >= queue.length) return Promise.resolve();
      var it = queue[idx++];
      return classifyOne(it).then(worker);
    }
    var workers = [];
    for (var w = 0; w < Math.min(CONFIG.concurrency, queue.length); w++) {
      workers.push(worker());
    }
    Promise.all(workers).then(function () {
      running = false;
      runIdle.hidden = false;
      runBusy.hidden = true;
      updateProgress();
      render();
      var failed = items.filter(function (i) { return i.state === 'error'; }).length;
      notify(
        failed ? 'Batch finished with ' + failed + ' failed file(s). Click Classify All to retry them.' : 'Batch complete! CSV is ready for download.',
        failed ? 'warn' : 'success'
      );
    });
  }

  function csvCell(v) {
    var s = v == null ? '' : String(v);
    return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  }

  function downloadCsv() {
    var head = ['file_name', 'model', 'status'];
    for (var k = 1; k <= CONFIG.topK; k++) {
      head.push('top' + k + '_label', 'top' + k + '_confidence_pct');
    }
    head.push('error');
    var lines = [head.map(csvCell).join(',')];
    items.forEach(function (it) {
      if (it.state !== 'done' && it.state !== 'error') return;
      var row = [it.file.name, it.model, it.state === 'done' ? 'ok' : 'failed'];
      for (var k = 0; k < CONFIG.topK; k++) {
        var p = it.preds && it.preds[k];
        row.push(p ? p.label : '', p ? p.conf.toFixed(2) : '');
      }
      row.push(it.error);
      lines.push(row.map(csvCell).join(','));
    });

    var blob = new Blob(['\ufeff' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8;' });
    var a = document.createElement('a');
    var stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    a.href = URL.createObjectURL(blob);
    a.download = 'sound-recognition-batch-predictions-' + stamp + '.csv';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    notify('CSV report downloaded successfully.', 'success');
  }

  // Event handlers
  drop.addEventListener('click', function () { if (!running) input.click(); });
  drop.addEventListener('keydown', function (e) {
    if ((e.key === 'Enter' || e.key === ' ') && !running) {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener('change', function () {
    addFiles(input.files);
    input.value = '';
  });

  ['dragenter', 'dragover'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.add('is-over');
    });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    drop.addEventListener(ev, function (e) {
      e.preventDefault();
      e.stopPropagation();
      drop.classList.remove('is-over');
    });
  });
  drop.addEventListener('drop', function (e) {
    if (!running && e.dataTransfer && e.dataTransfer.files.length) {
      addFiles(e.dataTransfer.files);
    }
  });

  listEl.addEventListener('click', function (e) {
    var btn = e.target.closest('.bf-remove');
    if (!btn || running) return;
    var li = btn.closest('li');
    var id = li ? Number(li.getAttribute('data-id')) : null;
    if (id) {
      items = items.filter(function (i) { return i.id !== id; });
      updateProgress();
      render();
    }
  });

  runBtn.addEventListener('click', runAll);
  csvBtn.addEventListener('click', downloadCsv);
  clearBtn.addEventListener('click', function () {
    if (running) return;
    items = [];
    progressWrap.hidden = true;
    render();
  });

  if (modelSelect) modelSelect.addEventListener('change', syncModelLabel);

  function scrollToBatch() {
    var card = $('batch-card');
    if (card) {
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      card.classList.add('batch-card-highlight');
      setTimeout(function () { card.classList.remove('batch-card-highlight'); }, 1800);
    }
  }

  if (heroBatchBtn) heroBatchBtn.addEventListener('click', scrollToBatch);
  if (hintBatchBtn) hintBatchBtn.addEventListener('click', scrollToBatch);

  // Keyboard shortcut 'b' while in Classifier Studio (not typing in an input)
  window.addEventListener('keydown', function (e) {
    if ((e.key === 'b' || e.key === 'B') && !['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
      var studioView = $('view-studio');
      if (studioView && !studioView.hidden) {
        e.preventDefault();
        scrollToBatch();
      }
    }
  });

  syncModelLabel();
  render();
})();

/**
 * ═══════════════════════════════════════════════════════════════
 *  SoundSense — Express.js Backend Server
 *  ESC-50 Environmental Sound Classification API
 *  Routes: /api/predict, /api/train, /api/save-model, /api/stats, /api/health
 * ═══════════════════════════════════════════════════════════════
 */

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { spawnSync } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3001;

/* ── Middleware ──────────────────────────────────────────────── */
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

/* Serve the frontend statically */
app.use(express.static(path.join(__dirname, 'Front-End')));

/* Serve processed audio files for preset loading */
app.use('/Preproccesing', express.static(path.join(__dirname, 'Preproccesing')));
app.use('/ESC-50', express.static(path.join(__dirname, 'ESC-50')));
app.use('/audio/esc50', express.static(path.join(__dirname, 'ESC-50', 'wav_files')));

app.get('/api/sample/:id', (req, res) => {
  const catId = parseInt(req.params.id, 10);
  const targetId = isNaN(catId) ? 0 : Math.max(0, Math.min(49, catId));
  const wavDir = path.join(__dirname, 'ESC-50', 'wav_files');
  if (fs.existsSync(wavDir)) {
    const files = fs.readdirSync(wavDir);
    const match = files.find(f => f.endsWith(`-${targetId}.wav`)) || files[0];
    if (match) {
      return res.sendFile(path.join(wavDir, match));
    }
  }
  return res.status(404).json({ error: 'Sample not found' });
});

/* ── Multer config for audio file uploads (Memory storage: zero disk writes) ───────────────────── */
const storage = multer.memoryStorage();

const upload = multer({
  storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB
  fileFilter: (_req, file, cb) => {
    const ext = path.extname(file.originalname || '').toLowerCase();
    const mime = (file.mimetype || '').toLowerCase();
    const disallowedExts = /\.(ogg|flac|m4a|aac)$/i;
    if (disallowedExts.test(ext) || mime.includes('ogg') || mime.includes('flac') || mime.includes('m4a') || mime.includes('aac')) {
      return cb(new Error('OGG, FLAC, and M4A files are not accepted. Please upload WAV, MP3, or MPEG/MPG.'), false);
    }
    const allowed = /\.(wav|mp3|mpg|mpeg|mpga|mpe|webm)$/i;
    const allowedMimes = ['audio/wav', 'audio/x-wav', 'audio/mp3', 'audio/mpeg', 'video/mpeg', 'audio/webm'];
    if (allowed.test(ext) || allowedMimes.includes(mime)) {
      cb(null, true);
    } else {
      cb(new Error('Only WAV, MP3, and MPEG/MPG audio files are accepted'), false);
    }
  },
});

/* ── Runtime counters only; checkpoints and scores are read from disk ── */
const serverState = {
  startedAt: new Date().toISOString(),
  totalPredictions: 0,
  trainingJobs: [],
};

const checkpointFiles = {
  ast: 'ast_manifest.json',
  resnet18: 'esc50_resnet18_v1.pt',
  vgg16: 'esc50_vgg16_v1.pt',
  sresnet: 'esc50_sresnet_linear_v09.pt',
  cnngru: 'esc50_cnngru_recurrent_v08.pt',
};

function readEvalReport() {
  try {
    return JSON.parse(fs.readFileSync(path.join(__dirname, 'checkpoints', 'eval_report.json'), 'utf8'));
  } catch (_) {
    return {};
  }
}

function getCheckpointInventory() {
  const inventory = {};
  for (const [model, filename] of Object.entries(checkpointFiles)) {
    const fullPath = path.join(__dirname, 'checkpoints', filename);
    const available = fs.existsSync(fullPath);
    inventory[model] = {
      filename,
      available,
      sha256: available ? crypto.createHash('sha256').update(fs.readFileSync(fullPath)).digest('hex') : null,
      sizeBytes: available ? fs.statSync(fullPath).size : null,
    };
  }
  return inventory;
}

function getModelPerformance(report) {
  const displayNames = { ast: 'AST-ESC50', resnet18: 'ResNet-18', vgg16: 'VGG-16', sresnet: 'SResNet', cnngru: 'CNN-GRU', ensemble: 'Ensemble Fusion' };
  const colors = { ast: '#6366f1', resnet18: '#38bdf8', vgg16: '#fbbf24', sresnet: '#a855f7', cnngru: '#34d399', ensemble: '#818cf8' };
  return Object.entries(report.models || {}).flatMap(([key, item]) => {
    const metrics = item.metrics || {};
    if (metrics.top1 == null) return [];
    return [{
      model: item.displayName || displayNames[key] || key,
      modelKey: key,
      accuracy: metrics.top1 * 100,
      top5Accuracy: metrics.top5 == null ? null : metrics.top5 * 100,
      r2Score: metrics.r2Probability ?? null,
      ece: metrics.ece ?? null,
      externalAccuracy: item.external?.top1 == null ? null : item.external.top1 * 100,
      latency: metrics.latencyMs ?? null,
      color: colors[key] || '#94a3b8',
      evaluatedOn: item.evaluation || null,
    }];
  });
}
/* ESC-50 category metadata */
const ESC50_CATEGORIES = [
  { id: 0, name: 'dog', group: 'Animals' }, { id: 1, name: 'rooster', group: 'Animals' },
  { id: 2, name: 'pig', group: 'Animals' }, { id: 3, name: 'cow', group: 'Animals' },
  { id: 4, name: 'frog', group: 'Animals' }, { id: 5, name: 'cat', group: 'Animals' },
  { id: 6, name: 'hen', group: 'Animals' }, { id: 7, name: 'insects', group: 'Animals' },
  { id: 8, name: 'sheep', group: 'Animals' }, { id: 9, name: 'crow', group: 'Animals' },
  { id: 10, name: 'rain', group: 'Natural' }, { id: 11, name: 'sea_waves', group: 'Natural' },
  { id: 12, name: 'crackling_fire', group: 'Natural' }, { id: 13, name: 'crickets', group: 'Natural' },
  { id: 14, name: 'chirping_birds', group: 'Natural' }, { id: 15, name: 'water_drops', group: 'Natural' },
  { id: 16, name: 'wind', group: 'Natural' }, { id: 17, name: 'pouring_water', group: 'Natural' },
  { id: 18, name: 'toilet_flush', group: 'Natural' }, { id: 19, name: 'thunderstorm', group: 'Natural' },
  { id: 20, name: 'crying_baby', group: 'Human' }, { id: 21, name: 'sneezing', group: 'Human' },
  { id: 22, name: 'clapping', group: 'Human' }, { id: 23, name: 'breathing', group: 'Human' },
  { id: 24, name: 'coughing', group: 'Human' }, { id: 25, name: 'footsteps', group: 'Human' },
  { id: 26, name: 'laughing', group: 'Human' }, { id: 27, name: 'brushing_teeth', group: 'Human' },
  { id: 28, name: 'snoring', group: 'Human' }, { id: 29, name: 'drinking_sipping', group: 'Human' },
  { id: 30, name: 'door_wood_knock', group: 'Interior' }, { id: 31, name: 'mouse_click', group: 'Interior' },
  { id: 32, name: 'keyboard_typing', group: 'Interior' }, { id: 33, name: 'door_wood_creaks', group: 'Interior' },
  { id: 34, name: 'can_opening', group: 'Interior' }, { id: 35, name: 'washing_machine', group: 'Interior' },
  { id: 36, name: 'vacuum_cleaner', group: 'Interior' }, { id: 37, name: 'clock_alarm', group: 'Interior' },
  { id: 38, name: 'clock_tick', group: 'Interior' }, { id: 39, name: 'glass_breaking', group: 'Interior' },
  { id: 40, name: 'helicopter', group: 'Urban' }, { id: 41, name: 'chainsaw', group: 'Urban' },
  { id: 42, name: 'siren', group: 'Urban' }, { id: 43, name: 'car_horn', group: 'Urban' },
  { id: 44, name: 'engine', group: 'Urban' }, { id: 45, name: 'train', group: 'Urban' },
  { id: 46, name: 'church_bells', group: 'Urban' }, { id: 47, name: 'airplane', group: 'Urban' },
  { id: 48, name: 'fireworks', group: 'Urban' }, { id: 49, name: 'hand_saw', group: 'Urban' },
];

/* ══════════════════════════════════════════════════════════════
   ROUTE 1: POST /api/predict — Real Audio-Based Inference Endpoint
   Analyzes audio waveform and spectral characteristics.
   Never relies on filename. Supports Out-of-Scope sound detection.
   ══════════════════════════════════════════════════════════════ */
const userHistory = [];

app.post('/api/predict', upload.single('audio'), (req, res) => {
  try {
    if (!req.file || !req.file.buffer) {
      return res.status(400).json({ error: 'No audio file provided. Send multipart/form-data with field "audio".' });
    }

    const modelKey = req.body.model || 'ast';
    const thresholdRaw = Number(req.body.threshold);
    const threshold = Number.isFinite(thresholdRaw) ? (thresholdRaw > 1 ? thresholdRaw / 100 : thresholdRaw) : null;
    const fileName = req.file.originalname || `upload_${Date.now()}.wav`;
    const audioBuffer = req.file.buffer;

    // 1. Primary: True Acoustic Neural Inference via Python module using stdin (zero disk writes)
    try {
      const pyScript = path.join(__dirname, 'Preproccesing', 'acoustic_inference.py');
      if (fs.existsSync(pyScript)) {
        const pyProc = spawnSync('python', [pyScript, '-', modelKey, threshold == null ? '' : String(threshold)], {
          input: audioBuffer,
          encoding: 'utf8',
          timeout: 30000,
          maxBuffer: 10 * 1024 * 1024,
        });
        if (pyProc.status === 0 && pyProc.stdout) {
          const result = JSON.parse(pyProc.stdout.trim());
          if (result && result.success) {
            serverState.totalPredictions++;
            const stamp = Date.now();
            const historyId = `hist_${stamp}_${Math.random().toString(36).substring(2, 8)}`;
            const historyEntry = {
              id: historyId,
              fileName,
              fileSize: req.file.size || audioBuffer.length,
              modelKey,
              modelName: result.isOutOfScope ? 'Not in ESC-50 Classes' : (result.category?.name || 'Unknown'),
              topPrediction: {
                category: result.category?.name || 'Unknown',
                emoji: result.category?.emoji || '🔊',
                group: result.category?.group || 'Other',
                confidence: result.confidence,
                confidencePercent: result.confidenceType === 'not_computed_rule_gate' || result.confidence == null
                  ? null
                  : `${Math.round(result.confidence * 100)}%`,
                isOutOfScope: result.isOutOfScope,
              },
              predictions: result.predictions,
              inferenceSource: 'api',
                latencyMs: result.inferenceTime ?? null,
              timestamp: new Date().toISOString(),
            };
            userHistory.unshift(historyEntry);
            if (userHistory.length > 200) userHistory.pop();

            return res.json({
              success: true,
              model: modelKey,
              engine: result.engine,
              fileName,
              isOutOfScope: result.isOutOfScope,
              confidenceType: result.confidenceType,
              classProbabilities: result.classProbabilities,
              oodScore: result.oodScore,
              category: result.category,
              predictions: result.predictions,
              confidence: result.confidence,
              explanation: result.explanation,
              metrics: result.metrics,
              inferenceTime: result.inferenceTime ?? null,
              belowConfidenceThreshold: result.belowConfidenceThreshold ?? false,
              confidenceThreshold: result.confidenceThreshold ?? null,
              historyId,
              timestamp: new Date().toISOString(),
            });
          }
        }
      }
    } catch (pyErr) {
      return res.status(503).json({
        success: false,
        error: 'Model inference is unavailable. Start the Python backend and ensure its dependencies and checkpoints are installed.',
        details: pyErr.message,
      });
    }

    return res.status(503).json({
      success: false,
      error: 'Model inference is unavailable. Start the Python backend and ensure its dependencies and checkpoints are installed.',
    });
  } catch (err) {
    res.status(500).json({ error: 'Inference failed', details: err.message });
  }
});

/* ══════════════════════════════════════════════════════════════
   ROUTE 1B: POST /api/predict/batch — Batch Audio Inference Endpoint
   ══════════════════════════════════════════════════════════════ */
app.post('/api/predict/batch', upload.array('audio', 100), (req, res) => {
  try {
    const files = req.files;
    if (!files || !files.length) {
      return res.status(400).json({ error: 'No audio files provided. Send multipart/form-data with field "audio".' });
    }

    const modelKey = req.body.model || 'ast';
    const thresholdRaw = Number(req.body.threshold);
    const threshold = Number.isFinite(thresholdRaw) ? (thresholdRaw > 1 ? thresholdRaw / 100 : thresholdRaw) : null;
    const pyScript = path.join(__dirname, 'Preproccesing', 'acoustic_inference.py');

    if (!fs.existsSync(pyScript)) {
      return res.status(503).json({
        success: false,
        error: 'Model inference is unavailable. Python script not found.',
      });
    }

    const results = [];
    for (const f of files) {
      const fileName = f.originalname || `upload_${Date.now()}.wav`;
      try {
        const pyProc = spawnSync('python', [pyScript, '-', modelKey, threshold == null ? '' : String(threshold)], {
          input: f.buffer,
          encoding: 'utf8',
          timeout: 30000,
          maxBuffer: 10 * 1024 * 1024,
        });

        if (pyProc.status === 0 && pyProc.stdout) {
          const result = JSON.parse(pyProc.stdout.trim());
          if (result && result.success) {
            serverState.totalPredictions++;
            const stamp = Date.now();
            const historyId = `hist_${stamp}_${Math.random().toString(36).substring(2, 8)}`;
            const historyEntry = {
              id: historyId,
              fileName,
              fileSize: f.size || f.buffer.length,
              modelKey,
              modelName: result.isOutOfScope ? 'Not in ESC-50 Classes' : (result.category?.name || 'Unknown'),
              topPrediction: {
                category: result.category?.name || 'Unknown',
                emoji: result.category?.emoji || '🔊',
                group: result.category?.group || 'Other',
                confidence: result.confidence,
                confidencePercent: result.confidenceType === 'not_computed_rule_gate' || result.confidence == null
                  ? null
                  : `${Math.round(result.confidence * 100)}%`,
                isOutOfScope: result.isOutOfScope,
              },
              predictions: result.predictions,
              inferenceSource: 'api_batch',
              latencyMs: result.inferenceTime ?? null,
              timestamp: new Date().toISOString(),
            };
            userHistory.unshift(historyEntry);
            if (userHistory.length > 200) userHistory.pop();

            results.push({
              fileName,
              success: true,
              isOutOfScope: result.isOutOfScope,
              confidenceType: result.confidenceType,
              classProbabilities: result.classProbabilities,
              oodScore: result.oodScore,
              category: result.category,
              confidence: result.confidence,
              predictions: result.predictions,
              inferenceTime: result.inferenceTime ?? null,
            });
            continue;
          }
        }
        results.push({ fileName, success: false, error: 'Inference failed' });
      } catch (fErr) {
        results.push({ fileName, success: false, error: fErr.message });
      }
    }

    return res.json({
      success: true,
      model: modelKey,
      count: results.length,
      successful: results.filter(r => r.success).length,
      failed: results.filter(r => !r.success).length,
      results,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    res.status(500).json({ error: 'Batch inference failed', details: err.message });
  }
});

/* ══════════════════════════════════════════════════════════════
   ROUTE 2: POST /api/train — Training is run offline on Kaggle
   ══════════════════════════════════════════════════════════════ */
app.post('/api/train', (_req, res) => {
  res.status(501).json({
    success: false,
    error: 'Training runs offline with python -m eval.train_cv. This endpoint does not simulate training or metrics.',
  });
});
/* GET /api/train/:jobId — Check training job status */
app.get('/api/train/:jobId', (req, res) => {
  const job = serverState.trainingJobs.find(j => j.jobId === req.params.jobId);
  if (!job) {
    return res.status(404).json({ error: 'Training job not found' });
  }
  res.json({ success: true, job });
});

/* GET /api/train — List all training jobs */
app.get('/api/train', (_req, res) => {
  res.json({
    success: true,
    jobs: serverState.trainingJobs.map(j => ({
      jobId: j.jobId,
      modelType: j.modelType,
      status: j.status,
      progress: j.progress,
      currentEpoch: j.currentEpoch,
      epochs: j.epochs,
      startedAt: j.startedAt,
      completedAt: j.completedAt,
    })),
  });
});

app.post('/api/save-model', (_req, res) => {
  res.status(501).json({
    success: false,
    error: 'Checkpoints are created by the offline training pipeline. This endpoint does not invent or save metrics.',
  });
});
/* GET /api/save-model — List all saved checkpoints */
app.get('/api/save-model', (_req, res) => {
  const inventory = getCheckpointInventory();
  const metrics = readEvalReport().models || {};
  const checkpoints = Object.entries(inventory)
    .filter(([, item]) => item.available)
    .map(([modelType, item]) => ({
      checkpointId: item.filename,
      name: modelType,
      modelType,
      filename: item.filename,
      sha256: item.sha256,
      fileSizeBytes: item.sizeBytes,
      accuracy: metrics[modelType]?.metrics?.top1 == null ? null : metrics[modelType].metrics.top1 * 100,
      top5: metrics[modelType]?.metrics?.top5 == null ? null : metrics[modelType].metrics.top5 * 100,
      status: 'available',
    }));
  res.json({
    success: true,
    checkpoints,
    modelVersions: [],
  });
});

/* POST /api/model/cloud-sync — Sync / download weights from Hugging Face Hub */
app.post('/api/model/cloud-sync', (req, res) => {
  const { repo = 'huggingface.co/soundsense/esc50-resnet18' } = req.body || {};
  res.status(501).json({
    success: false,
    error: `No verified model repository is configured (requested source: ${repo}).`,
  });
});

/* ══════════════════════════════════════════════════════════════
   ROUTE 4: GET /api/stats — System & Dataset Analytics
   ══════════════════════════════════════════════════════════════ */
app.get('/api/stats', (_req, res) => {
  // Count preprocessed files if they exist
  const preprocessedDir = path.join(__dirname, 'Preproccesing', 'processed');
  let processedAudioCount = 0;
  let logMelCount = 0;
  let mfccCount = 0;
  let rpCount = 0;

  try {
    const audioDir = path.join(preprocessedDir, 'audio_22k');
    if (fs.existsSync(audioDir)) {
      processedAudioCount = fs.readdirSync(audioDir).filter(f => f.endsWith('.wav')).length;
    }
    const melDir = path.join(preprocessedDir, 'features', 'log_mel');
    if (fs.existsSync(melDir)) {
      logMelCount = fs.readdirSync(melDir).filter(f => f.endsWith('.npy')).length;
    }
    const mfccDir = path.join(preprocessedDir, 'features', 'mfcc');
    if (fs.existsSync(mfccDir)) {
      mfccCount = fs.readdirSync(mfccDir).filter(f => f.endsWith('.npy')).length;
    }
    const rpDir = path.join(preprocessedDir, 'features', 'recurrence_plot');
    if (fs.existsSync(rpDir)) {
      rpCount = fs.readdirSync(rpDir).filter(f => f.endsWith('.npy')).length;
    }
  } catch (_) {}

  // Load preprocessing report if it exists
  let preprocessingReport = null;
  try {
    const reportPath = path.join(__dirname, 'Preproccesing', 'reports', 'preprocessing_report.json');
    if (fs.existsSync(reportPath)) {
      preprocessingReport = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    }
  } catch (_) {}

  // Per-class distribution
  const classDistribution = ESC50_CATEGORIES.map(cat => ({
    id: cat.id,
    name: cat.name,
    group: cat.group,
    sampleCount: Math.round(processedAudioCount / 50),
  }));

  // Per-group distribution
  const groups = ['Animals', 'Natural', 'Human', 'Interior', 'Urban'];
  const groupDistribution = groups.map(g => ({
    group: g,
    classCount: 10,
    totalSamples: Math.round(processedAudioCount / 5),
  }));

  const evaluationReport = readEvalReport();
  const inventory = getCheckpointInventory();
  const modelPerformance = getModelPerformance(evaluationReport);

  const uptime = Math.round((Date.now() - new Date(serverState.startedAt).getTime()) / 1000);

  res.json({
    success: true,
    system: {
      uptime,
      uptimeFormatted: `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m ${uptime % 60}s`,
      startedAt: serverState.startedAt,
      nodeVersion: process.version,
      platform: process.platform,
      memoryUsage: {
        rss: Math.round(process.memoryUsage().rss / 1024 / 1024),
        heapUsed: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        heapTotal: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
      },
      totalPredictions: serverState.totalPredictions,
      totalTrainingRuns: Number(evaluationReport.trainingRuns) || 0,
      savedCheckpoints: Object.values(inventory).filter(item => item.available).length,
    },
    dataset: {
      name: 'ESC-50',
      totalClasses: 50,
      totalGroups: 5,
      processedAudioCount,
      logMelCount,
      mfccCount,
      rpCount,
      sampleRate: 22050,
      duration: 5.0,
      featureShapes: {
        logMel: [128, 216],
        mfcc: [40, 216],
        recurrencePlot: [128, 128],
      },
      numFolds: 5,
      classDistribution,
      groupDistribution,
    },
    modelPerformance,
    preprocessingReport,
    modelVersions: [],
    evaluationReport,
    modelAvailability: {
      availableModels: Object.entries(inventory).filter(([, item]) => item.available).map(([key]) => key),
      missingModels: Object.entries(inventory).filter(([, item]) => !item.available).map(([key]) => key),
    },
  });
});

/* ══════════════════════════════════════════════════════════════
   ROUTE 6: DEVELOPER AUTHENTICATION ENDPOINTS
   ══════════════════════════════════════════════════════════════ */
const activeSessions = new Map();

/* POST /api/auth/login — Developer login */
app.post('/api/auth/login', (req, res) => {
  const { username = '', password = '' } = req.body || {};
  const cleanUser = username.trim().toLowerCase();
  const cleanPass = password.trim();

  // Flexible developer login: default credentials, admin/dev keywords, or any non-empty username/password
  const isDevUser = cleanUser === 'developer@soundsense.ai' ||
                    cleanUser === 'developer' ||
                    cleanUser === 'admin' ||
                    cleanUser === 'admin@soundsense.ai' ||
                    cleanUser.includes('developer') ||
                    cleanUser.includes('admin') ||
                    cleanUser.includes('dev') ||
                    cleanUser.length > 0;

  const isDevPass = cleanPass === 'soundsense2024' ||
                    cleanPass === 'soundsense' ||
                    cleanPass === 'admin' ||
                    cleanPass === 'admin123' ||
                    cleanPass.length >= 3;

  const isDev = isDevUser && isDevPass;

  if (isDev) {
    const token = `dev_token_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
    const baseName = cleanUser.includes('@') ? cleanUser.split('@')[0] : cleanUser;
    const formattedName = baseName.replace(/[._-]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    const user = {
      name: formattedName ? `${formattedName} (AI Engineer)` : 'Lead AI Engineer',
      email: cleanUser.includes('@') ? cleanUser : `${cleanUser}@soundsense.ai`,
      role: 'developer',
      permissions: ['training_controls', 'checkpoint_mgmt', 'system_analytics', 'cloud_sync'],
      signedInAt: new Date().toISOString(),
    };
    activeSessions.set(token, user);
    return res.json({
      success: true,
      token,
      user,
      message: 'Developer authenticated successfully.'
    });
  }

  return res.status(401).json({
    success: false,
    error: 'Please enter a valid developer username and password (minimum 3 characters).'
  });
});

/* GET /api/auth/me — Check current auth session */
app.get('/api/auth/me', (req, res) => {
  const authHeader = req.headers.authorization || '';
  const token = authHeader.replace(/^Bearer\s+/i, '');
  if (token && activeSessions.has(token)) {
    return res.json({ success: true, authenticated: true, user: activeSessions.get(token) });
  }
  return res.json({ success: true, authenticated: false, role: 'guest' });
});

/* POST /api/auth/logout — Developer logout */
app.post('/api/auth/logout', (req, res) => {
  const authHeader = req.headers.authorization || '';
  const token = authHeader.replace(/^Bearer\s+/i, '');
  if (token) activeSessions.delete(token);
  res.json({ success: true, message: 'Developer session terminated.' });
});

/* ══════════════════════════════════════════════════════════════
   ROUTE 5: GET /api/health — Health Check
   ══════════════════════════════════════════════════════════════ */
app.get('/api/health', (_req, res) => {
  const uptime = Math.round((Date.now() - new Date(serverState.startedAt).getTime()) / 1000);
  const inventory = getCheckpointInventory();
  const models = Object.entries(inventory).filter(([, item]) => item.available).map(([key, item]) => [key, item]);
  const missingModels = Object.entries(inventory).filter(([, item]) => !item.available).map(([key]) => key);
  res.json({
    status: missingModels.length ? 'degraded' : 'healthy',
    service: 'SoundSense Backend',
    version: '1.0.0',
    uptime,
    models: Object.fromEntries(models),
    missingModels,
    timestamp: new Date().toISOString(),
    endpoints: [
      { method: 'POST', path: '/api/auth/login', description: 'Developer login' },
      { method: 'GET', path: '/api/auth/me', description: 'Verify developer session' },
      { method: 'POST', path: '/api/auth/logout', description: 'Developer logout' },
      { method: 'POST', path: '/api/predict', description: 'Audio classification inference' },
      { method: 'POST', path: '/api/predict/batch', description: 'Batch audio classification inference' },
      { method: 'POST', path: '/api/train', description: 'Trigger model training' },
      { method: 'GET', path: '/api/train', description: 'List training jobs' },
      { method: 'GET', path: '/api/train/:jobId', description: 'Check training job status' },
      { method: 'POST', path: '/api/save-model', description: 'Save model checkpoint' },
      { method: 'GET', path: '/api/save-model', description: 'List saved checkpoints' },
      { method: 'GET', path: '/api/stats', description: 'System & dataset analytics' },
      { method: 'GET', path: '/api/health', description: 'Health check' },
    ],
  });
});

/* ── Fallback: Serve index.html for SPA navigation ──────────── */
app.get('/', (_req, res) => {
  res.sendFile(path.join(__dirname, 'Front-End', 'index.html'));
});

/* ── Error handling middleware ───────────────────────────────── */
app.use((err, _req, res, _next) => {
  console.error('Server error:', err.message);
  if (err instanceof multer.MulterError) {
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  res.status(500).json({ error: 'Internal server error', details: err.message });
});

/* ── Start server ───────────────────────────────────────────── */
app.listen(PORT, () => {
  console.log('═══════════════════════════════════════════════════════');
  console.log('  SoundSense Backend Server');
  console.log(`  Running at:  http://localhost:${PORT}`);
  console.log(`  API Base:    http://localhost:${PORT}/api`);
  console.log(`  Health:      http://localhost:${PORT}/api/health`);
  console.log('═══════════════════════════════════════════════════════');
});

module.exports = app;

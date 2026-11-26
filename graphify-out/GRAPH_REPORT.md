# Graph Report - Sound_Recognition  (2026-11-26)

## Corpus Check
- 88 files · ~990,996 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6020 file(s) not represented in the graph (top: .npy 5994, .csv 19, (none) 3)

## Summary
- 1222 nodes · 2448 edges · 80 communities (52 shown, 28 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4ddc2e9d`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- acoustic_inference.py
- Audio Processing & Spectrogram Synthesis
- models
- Node.js Server & Package Dependencies
- get_model
- UI Animation & Status Management
- main.py
- SoundSense — Environmental Sound Classification Studio
- MN20GRUAudioV4
- pathlib
- SoundSense — Truthful High-Accuracy Recovery Plan
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- _require_weights
- renderTrainingChart
- Interactive Particle Visualizer
- verify_and_report.py
- train_fold
- showToast
- fx.js
- build_external_mixed_set.py
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- updateApiExplorer
- render
- CNNGRUFast
- eval_esc50_cv_features.py
- Preproccesing/__init__.py
- Stratified 5-Fold Split Generator
- eval_esc50_cv.py
- copilot-instructions.md
- download_ast_weights
- acoustic_inference.py
- PannsCnn10Backbone
- log_mel_to_imagenet_3ch
- run_acoustic_inference
- .__init__
- post
- _PannsAudioBaseV4
- drawFrame
- onResize
- eval_external.py
- numpy
- train_cv.py
- audioset_backbones.py
- get
- Tensor
- generate_augmented_variants
- CNNGRUV3
- _PannsResNetLayers
- VGG16V3
- _ConcurrentSEBlock
- run
- SoundSense evaluation report
- test_batch_prediction.py
- lifespan
- report.py
- test_system_integration.py
- 4. Plan A — deploy AST as the production model
- 5. Plan B — train the four architecture checkpoints (optional, GPU)
- Module
- 2. Evidence and current state
- 7. Calibration, confidence, and OOD policy
- VGG16Transfer
- trigger_training

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 39 edges
2. `load_model_checkpoint()` - 28 edges
3. `classification_metrics()` - 19 edges
4. `loadAudioFile()` - 17 edges
5. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
6. `train_fold()` - 15 edges
7. `switchView()` - 14 edges
8. `load_audio_waveform()` - 14 edges
9. `_train_v4_model()` - 14 edges
10. `build()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A7. Verification tests` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A3. Add a clean AST evaluator` --references--> `classification_metrics()`  [INFERRED]
  PLAN.md → eval/metrics.py
- `10. Ordered execution checklist` --references--> `model_status()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A2. Add the AST engine to `Preproccesing/acoustic_inference.py`` --references--> `model_status()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py

## Import Cycles
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_splits.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/feature_extractor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/feature_extractor.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/feature_extractor.py -> Preproccesing/__init__.py`
- 4-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/feature_extractor.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`

## Hyperedges (group relationships)
- **Dual-Modal Acoustic Processing Flow** — preproccesing_readme_dmff_architecture, preproccesing_readme_recurrence_plot, preproccesing_readme_paa_dimension, preproccesing_readme_esc50_dataset [EXTRACTED 0.95]

## Communities (80 total, 28 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (176): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+168 more)

### Community 1 - "acoustic_inference.py"
Cohesion: 0.11
Nodes (10): load_pretrained_panns(), train_and_export(), calibrate_temperature(), compute_metrics(), load_split_features(), ResNet18Transfer, SResNetTransfer, train_and_export_all() (+2 more)

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.13
Nodes (27): buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks(), drawFeatureSpectrogram() (+19 more)

### Community 3 - "models"
Cohesion: 0.07
Nodes (28): available, filename, sha256, sizeBytes, available, filename, sha256, sizeBytes (+20 more)

### Community 4 - "Node.js Server & Package Dependencies"
Cohesion: 0.05
Nodes (34): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+26 more)

### Community 5 - "get_model"
Cohesion: 0.09
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.09
Nodes (32): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), applyTheme(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+24 more)

### Community 8 - "main.py"
Cohesion: 0.10
Nodes (4): clear_history(), int32_hash(), _is_soundsense_instance(), _port_is_open()

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.20
Nodes (9): API Endpoints, Deployed Model: AST-ESC50, Evaluated Architectures (from `checkpoints/eval_report.json`), Independently Measured Metrics (from evaluation harness), Key Features, Reproducing Evaluation Metrics, Running the Application, SoundSense — Environmental Sound Classification Studio (+1 more)

### Community 10 - "MN20GRUAudioV4"
Cohesion: 0.11
Nodes (6): load_pretrained_efficientat(), MelSTFT32k, _AudioHeadV4, DyMNAudioV4, MN20GRUAudioV4, _MN20GRUHead

### Community 11 - "pathlib"
Cohesion: 0.13
Nodes (7): evaluate_corpus(), _transcode_bytes(), run(), _signals(), _gate(), run(), binary_ood_metrics()

### Community 12 - "SoundSense — Truthful High-Accuracy Recovery Plan"
Cohesion: 0.10
Nodes (19): 0. Non-negotiable rules, 11. Command cheat sheet, 12. Definition of done, 1.1 Root cause of the current low/absent scores, 1.2 The best real model already available, 1.3 Recommended plan, 1. Executive summary, 3. Acceptance gates (+11 more)

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.18
Nodes (10): key, value, services, buildCommand, env, envVars, name, plan (+2 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.40
Nodes (5): Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 16 - "_require_weights"
Cohesion: 0.11
Nodes (7): CNNGRUV3, DualPoolHead, _require_weights(), ResNet18V3, SResNetV3, _StemTailBackboneV3, VGG16V3

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 19 - "verify_and_report.py"
Cohesion: 0.20
Nodes (3): ESC50MultiModalDataset, get_dataloaders(), test_pytorch_dataloader()

### Community 20 - "train_fold"
Cohesion: 0.08
Nodes (13): _amp_autocast(), _collate_fn(), ESC50WaveformDataset, _file_sha256(), find_esc50_audio_path(), main(), _make_grad_scaler(), _mixup_labels() (+5 more)

### Community 21 - "showToast"
Cohesion: 0.14
Nodes (19): animateCounter(), step(), checkBackendStatus(), cleanupMic(), displayClassificationResults(), easeOutQuad(), encodeWAV(), ensureAudioContext() (+11 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "build_external_mixed_set.py"
Cohesion: 0.10
Nodes (21): build(), _build_cross_source_mixtures(), _worker(), _download(), _fetch_audioset_candidates(), _fetch_audioset_strong_candidates(), _fetch_urbansound_candidates(), _fetch_wikimedia_candidates() (+13 more)

### Community 24 - "Vercel Edge Deployment Config"
Cohesion: 0.50
Nodes (3): builds, routes, version

### Community 25 - "Keyboard Shortcuts Modal"
Cohesion: 0.67
Nodes (3): closeShortcuts(), openShortcuts(), toggleShortcuts()

### Community 26 - "updateApiExplorer"
Cohesion: 0.67
Nodes (3): formatApiCode(), formatApiJson(), updateApiExplorer()

### Community 27 - "render"
Cohesion: 0.12
Nodes (20): addFiles(), animate(), classifyOne(), tryCandidate(), csvCell(), downloadCsv(), esc(), fmtSize() (+12 more)

### Community 30 - "eval_esc50_cv_features.py"
Cohesion: 0.13
Nodes (15): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval(), _ast_canonical_remap(), _ast_fold_logits(), _batched_logits() (+7 more)

### Community 31 - "Preproccesing/__init__.py"
Cohesion: 0.12
Nodes (7): detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio(), process_single_audio(), generate_5fold_splits()

### Community 33 - "eval_esc50_cv.py"
Cohesion: 0.20
Nodes (6): evaluate(), _load_wav(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 35 - "download_ast_weights"
Cohesion: 0.27
Nodes (3): download_ast_weights(), retrieve_cloud_weights(), _sha256()

### Community 37 - "PannsCnn10Backbone"
Cohesion: 0.15
Nodes (4): PannsCnn10Backbone, PannsCnn14Backbone, PannsConvBlock, PannsSTFT

### Community 38 - "log_mel_to_imagenet_3ch"
Cohesion: 0.10
Nodes (6): CNNGRUV2, log_mel_to_imagenet_3ch(), ResNet18V2, _SEConvBlock, SResNetV2, VGG16V2

### Community 40 - "run_acoustic_inference"
Cohesion: 0.16
Nodes (12): 10. Ordered execution checklist, A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, _ast_probabilities(), _build_silence_result(), _build_tone_result(), _load_ast_engine(), load_audio_waveform(), load_model_checkpoint() (+4 more)

### Community 41 - ".__init__"
Cohesion: 0.19
Nodes (5): _ContextGen, _CoordAtt, _DynamicConv, _DynamicWrapper, _DyReLUB

### Community 44 - "post"
Cohesion: 0.16
Nodes (14): add_history(), cloud_sync(), CloudSyncRequest, load_developer_accounts(), login(), LoginRequest, logout(), predict() (+6 more)

### Community 45 - "_PannsAudioBaseV4"
Cohesion: 0.17
Nodes (4): Cnn10AudioV4, Cnn14AudioV4, _PannsAudioBaseV4, ResNet22AudioV4

### Community 46 - "drawFrame"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 49 - "eval_external.py"
Cohesion: 0.23
Nodes (8): _encode(), _eval_model(), run(), _summarize(), run(), _snr_noise(), _variants(), _wav_bytes()

### Community 51 - "train_cv.py"
Cohesion: 0.07
Nodes (35): _active_crop_32k(), _build_optimizer(), _checkpoint_path(), _extract_stem_features(), _extract_v4_clean_cache(), _extract_v4_fold_mixes(), _fit_model_stages(), _fit_v4_head() (+27 more)

### Community 52 - "audioset_backbones.py"
Cohesion: 0.19
Nodes (3): AudioMobileNetBackbone, _cnn_out_size(), _InvertedResidual

### Community 54 - "get"
Cohesion: 0.17
Nodes (12): auth_me(), current_checkpoint_inventory(), get_history(), get_sample_for_category(), get_stats(), get_training_job(), get_uptime_seconds(), health() (+4 more)

### Community 55 - "Tensor"
Cohesion: 0.24
Nodes (3): AudioDyMNBackbone, audioset_ood_probability(), _DYBlock

### Community 57 - "generate_augmented_variants"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 64 - "_ConcurrentSEBlock"
Cohesion: 0.20
Nodes (4): _ConcurrentSEBlock, _InvertedResidualConfig, _make_divisible(), _SqueezeExciteDim

### Community 65 - "run"
Cohesion: 0.16
Nodes (7): run(), AcousticClassifierNet, compute_signal_metrics(), extract_acoustic_feature_vector(), _ffmpeg_decode_audio(), get_acoustic_classifier(), _mlp_probabilities()

### Community 66 - "SoundSense evaluation report"
Cohesion: 0.40
Nodes (4): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status

### Community 71 - "report.py"
Cohesion: 0.30
Nodes (7): build_report(), complete_external(), _read(), _write_markdown(), write_report(), A4. Update `eval/report.py`, model_status()

### Community 72 - "test_system_integration.py"
Cohesion: 0.39
Nodes (7): run_tests(), test_ast_checkpoint_available(), test_ast_real_audio_inference(), test_filename_invariance(), test_invalid_model_returns_400(), test_model_status_report(), test_other_checkpoints_trained_and_working()

### Community 73 - "4. Plan A — deploy AST as the production model"
Cohesion: 0.29
Nodes (7): modelKey(), 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A3. Add a clean AST evaluator, A5. Backend defaults and parity, A7. Verification tests

### Community 74 - "5. Plan B — train the four architecture checkpoints (optional, GPU)"
Cohesion: 0.33
Nodes (6): 5. Plan B — train the four architecture checkpoints (optional, GPU), B1. Train, B2. Five-fold evaluation, B3. External dev evaluation, B4. Publication gate for the four models, B5. Optional distillation (stretch, not required)

### Community 76 - "2. Evidence and current state"
Cohesion: 0.40
Nodes (5): 2.1 Verified measurements, 2.2 Confidence behavior is real and useful, 2.3 Failure cases to acknowledge, 2.4 What “R²” means in this repository, 2. Evidence and current state

### Community 77 - "7. Calibration, confidence, and OOD policy"
Cohesion: 0.40
Nodes (5): 7.1 Calibration, 7.2 Selective prediction, 7.3 OOD, 7.4 R² display rule, 7. Calibration, confidence, and OOD policy

## Knowledge Gaps
- **4 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Dual-Modal Verification Diagnostic Plot`, `5-Fold Stratified Split Generator`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 576 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **28 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A5. Backend defaults and parity` connect `4. Plan A — deploy AST as the production model` to `test_system_integration.py`?**
  _High betweenness centrality (0.274) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Dual-Modal Verification Diagnostic Plot` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.010416666666666666 - nodes in this community are weakly interconnected._
- **Why does `modelKey()` connect `4. Plan A — deploy AST as the production model` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.274) - this node is a cross-community bridge._
- **Should `acoustic_inference.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1140819964349376 - nodes in this community are weakly interconnected._
- **Why does `test_ast_checkpoint_available()` connect `test_system_integration.py` to `run_acoustic_inference`, `4. Plan A — deploy AST as the production model`?**
  _High betweenness centrality (0.241) - this node is a cross-community bridge._
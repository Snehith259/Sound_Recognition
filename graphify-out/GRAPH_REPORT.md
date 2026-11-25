# Graph Report - Sound_Recognition  (2026-11-25)

## Corpus Check
- 78 files · ~281,959 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6019 file(s) not represented in the graph (top: .npy 5994, .csv 18, (none) 3)

## Summary
- 1005 nodes · 1900 edges · 57 communities (37 shown, 20 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 63 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `71525af9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- train_sresnet_option_a.py
- Audio Processing & Spectrogram Synthesis
- models
- Node.js Server & Package Dependencies
- get_model
- UI Animation & Status Management
- main.py
- SoundSense — Environmental Sound Classification Studio
- eval_ast.py
- eval_legacy_mlp_external.py
- SoundSense — Truthful High-Accuracy Recovery Plan
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- Tensor
- renderTrainingChart
- Interactive Particle Visualizer
- eval_external.py
- train_fold
- showToast
- fx.js
- build_external_set.py
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- updateApiExplorer
- render
- CNNGRUFast
- eval_external.py
- numpy
- Stratified 5-Fold Split Generator
- eval_esc50_cv.py
- copilot-instructions.md
- download_ast_weights
- pandas
- train_ast_cv.py
- acoustic_inference.py
- run_acoustic_inference
- 4. Plan A — deploy AST as the production model
- time
- report.py
- drawFrame
- onResize
- eval_esc50_cv_features.py
- pathlib

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 35 edges
2. `classification_metrics()` - 19 edges
3. `loadAudioFile()` - 17 edges
4. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
5. `train_fold()` - 15 edges
6. `switchView()` - 14 edges
7. `load_audio_waveform()` - 13 edges
8. `model_status()` - 12 edges
9. `showToast()` - 12 edges
10. `train_model()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A7. Verification tests` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A3. Add a clean AST evaluator` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A3. Add a clean AST evaluator` --references--> `classification_metrics()`  [INFERRED]
  PLAN.md → eval/metrics.py
- `A5. Backend defaults and parity` --references--> `test_ast_checkpoint_available()`  [INFERRED]
  PLAN.md → tests/test_system_integration.py

## Import Cycles
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_splits.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/feature_extractor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/feature_extractor.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/feature_extractor.py -> Preproccesing/__init__.py`
- 4-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/feature_extractor.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`

## Hyperedges (group relationships)
- **Dual-Modal Acoustic Processing Flow** — preproccesing_readme_dmff_architecture, preproccesing_readme_recurrence_plot, preproccesing_readme_paa_dimension, preproccesing_readme_esc50_dataset [EXTRACTED 0.95]

## Communities (57 total, 20 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (176): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+168 more)

### Community 1 - "train_sresnet_option_a.py"
Cohesion: 0.14
Nodes (8): train_and_export(), calibrate_temperature(), compute_metrics(), extract_features_batched(), load_split_features(), ResNet18Transfer, train_and_export_all(), VGG16Transfer

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
Cohesion: 0.10
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.09
Nodes (32): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), applyTheme(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+24 more)

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (33): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, current_checkpoint_inventory(), get_history(), get_sample_for_category() (+25 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Evaluated Architectures (from `checkpoints/eval_report.json`), Independently Measured Metrics (from evaluation harness) (+5 more)

### Community 10 - "eval_ast.py"
Cohesion: 0.23
Nodes (5): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval()

### Community 11 - "eval_legacy_mlp_external.py"
Cohesion: 0.16
Nodes (7): run(), _gate(), run(), compute_signal_metrics(), extract_acoustic_feature_vector(), load_audio_waveform(), _mlp_probabilities()

### Community 12 - "SoundSense — Truthful High-Accuracy Recovery Plan"
Cohesion: 0.06
Nodes (35): 0. Non-negotiable rules, 11. Command cheat sheet, 12. Definition of done, 1.1 Root cause of the current low/absent scores, 1.2 The best real model already available, 1.3 Recommended plan, 1. Executive summary, 2.1 Verified measurements (+27 more)

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.18
Nodes (10): key, value, services, buildCommand, env, envVars, name, plan (+2 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.40
Nodes (5): Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 16 - "Tensor"
Cohesion: 0.11
Nodes (7): CNNGRUV2, log_mel_to_imagenet_3ch(), _require_weights(), ResNet18V2, _SEConvBlock, SResNetV2, VGG16V2

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 19 - "eval_external.py"
Cohesion: 0.36
Nodes (4): run(), _snr_noise(), _variants(), _wav_bytes()

### Community 20 - "train_fold"
Cohesion: 0.08
Nodes (13): _amp_autocast(), _collate_fn(), ESC50WaveformDataset, _file_sha256(), find_esc50_audio_path(), main(), _make_grad_scaler(), _mixup_labels() (+5 more)

### Community 21 - "showToast"
Cohesion: 0.14
Nodes (19): animateCounter(), step(), checkBackendStatus(), cleanupMic(), displayClassificationResults(), easeOutQuad(), encodeWAV(), ensureAudioContext() (+11 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "build_external_set.py"
Cohesion: 0.25
Nodes (7): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio()

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

### Community 30 - "eval_external.py"
Cohesion: 0.21
Nodes (6): _encode(), _eval_model(), run(), _summarize(), binary_ood_metrics(), classification_metrics()

### Community 31 - "numpy"
Cohesion: 0.06
Nodes (20): detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio(), process_single_audio(), add_gaussian_noise(), cyclic_time_shift() (+12 more)

### Community 35 - "download_ast_weights"
Cohesion: 0.31
Nodes (3): download_ast_weights(), retrieve_cloud_weights(), _sha256()

### Community 37 - "train_ast_cv.py"
Cohesion: 0.16
Nodes (3): SResNetTransfer, extract_features_batched(), main()

### Community 38 - "acoustic_inference.py"
Cohesion: 0.19
Nodes (4): run(), _signals(), AcousticClassifierNet, get_acoustic_classifier()

### Community 40 - "run_acoustic_inference"
Cohesion: 0.13
Nodes (19): 10. Ordered execution checklist, A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, _ast_probabilities(), _build_silence_result(), _build_tone_result(), get_engine_status(), _load_ast_engine(), load_model_checkpoint() (+11 more)

### Community 41 - "4. Plan A — deploy AST as the production model"
Cohesion: 0.29
Nodes (7): modelKey(), 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A3. Add a clean AST evaluator, A5. Backend defaults and parity, A7. Verification tests

### Community 45 - "report.py"
Cohesion: 0.38
Nodes (6): build_report(), complete_external(), _read(), _write_markdown(), write_report(), A4. Update `eval/report.py`

### Community 46 - "drawFrame"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 51 - "eval_esc50_cv_features.py"
Cohesion: 0.10
Nodes (25): _ast_canonical_remap(), _ast_fold_logits(), _batched_logits(), _checkpoint_arch_version(), evaluate(), evaluate_ast(), evaluate_ensemble(), _has_all_ast_folds() (+17 more)

### Community 67 - "pathlib"
Cohesion: 0.16
Nodes (3): encode_audio(), test_multiformat_ambient_and_pure(), run_tests()

## Knowledge Gaps
- **4 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Dual-Modal Verification Diagnostic Plot`, `5-Fold Stratified Split Generator`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 515 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **20 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A5. Backend defaults and parity` connect `4. Plan A — deploy AST as the production model` to `run_acoustic_inference`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Dual-Modal Verification Diagnostic Plot` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.010416666666666666 - nodes in this community are weakly interconnected._
- **Why does `modelKey()` connect `4. Plan A — deploy AST as the production model` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.277) - this node is a cross-community bridge._
- **Should `train_sresnet_option_a.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13768115942028986 - nodes in this community are weakly interconnected._
- **Why does `test_ast_checkpoint_available()` connect `run_acoustic_inference` to `4. Plan A — deploy AST as the production model`?**
  _High betweenness centrality (0.231) - this node is a cross-community bridge._
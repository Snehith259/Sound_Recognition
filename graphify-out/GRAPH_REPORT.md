# Graph Report - Sound_Recognition  (2026-11-25)

## Corpus Check
- 77 files · ~279,178 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6019 file(s) not represented in the graph (top: .npy 5994, .csv 18, (none) 3)

## Summary
- 964 nodes · 1796 edges · 59 communities (38 shown, 21 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 60 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1eab6050`
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
- eval_signal_gates.py
- SoundSense — Truthful High-Accuracy Recovery Plan
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- Tensor
- renderTrainingChart
- Interactive Particle Visualizer
- eval_external.py
- ESC50MultiModalDataset
- showToast
- fx.js
- build_external_set.py
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- updateApiExplorer
- render
- run
- CNNGRUFast
- train_cv.py
- data_augmentation.py
- Stratified 5-Fold Split Generator
- eval_esc50_cv.py
- copilot-instructions.md
- download_ast_weights
- batch_preprocessor.py
- SResNetTransfer
- eval_ood.py
- run_acoustic_inference
- train_models.py
- displayClassificationResults
- encodeWAV
- eval_esc50_cv_features.py
- pathlib

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 34 edges
2. `classification_metrics()` - 18 edges
3. `loadAudioFile()` - 17 edges
4. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
5. `switchView()` - 14 edges
6. `model_status()` - 12 edges
7. `train_model()` - 12 edges
8. `showToast()` - 12 edges
9. `load_audio_waveform()` - 11 edges
10. `extract_all_features()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A7. Verification tests` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A3. Add a clean AST evaluator` --references--> `classification_metrics()`  [INFERRED]
  PLAN.md → eval/metrics.py
- `10. Ordered execution checklist` --references--> `_load_ast_engine()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `10. Ordered execution checklist` --references--> `_ast_probabilities()`  [INFERRED]
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

## Communities (59 total, 21 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (176): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+168 more)

### Community 1 - "train_sresnet_option_a.py"
Cohesion: 0.18
Nodes (6): train_and_export(), calibrate_temperature(), extract_features_batched(), load_split_features(), ResNet18Transfer, train_and_export_all()

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.10
Nodes (32): buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks(), drawFeatureSpectrogram() (+24 more)

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
Cohesion: 0.10
Nodes (26): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), applyTheme(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+18 more)

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (33): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, current_checkpoint_inventory(), get_history(), get_sample_for_category() (+25 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Evaluated Architectures (from `checkpoints/eval_report.json`), Independently Measured Metrics (from evaluation harness) (+5 more)

### Community 10 - "eval_ast.py"
Cohesion: 0.15
Nodes (7): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval(), binary_ood_metrics(), classification_metrics()

### Community 11 - "eval_signal_gates.py"
Cohesion: 0.31
Nodes (3): _gate(), run(), load_audio_waveform()

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
Cohesion: 0.22
Nodes (3): CNNGRUV2, log_mel_to_imagenet_3ch(), _SEConvBlock

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "eval_external.py"
Cohesion: 0.23
Nodes (8): _encode(), _eval_model(), run(), _summarize(), run(), _snr_noise(), _variants(), _wav_bytes()

### Community 20 - "ESC50MultiModalDataset"
Cohesion: 0.18
Nodes (3): ESC50MultiModalDataset, get_dataloaders(), test_pytorch_dataloader()

### Community 21 - "showToast"
Cohesion: 0.19
Nodes (18): checkBackendStatus(), cleanupMic(), ensureAudioContext(), escapeHtml(), executeClassification(), finalizeMicRecording(), finishTrainingRun(), loadTrainingData() (+10 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "build_external_set.py"
Cohesion: 0.17
Nodes (9): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio(), encode_audio() (+1 more)

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

### Community 28 - "run"
Cohesion: 0.19
Nodes (6): run(), AcousticClassifierNet, compute_signal_metrics(), extract_acoustic_feature_vector(), get_acoustic_classifier(), _mlp_probabilities()

### Community 30 - "train_cv.py"
Cohesion: 0.18
Nodes (4): _require_weights(), ResNet18V2, SResNetV2, VGG16V2

### Community 31 - "data_augmentation.py"
Cohesion: 0.14
Nodes (7): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch(), generate_feature_visualization(), verify_feature_integrity()

### Community 33 - "eval_esc50_cv.py"
Cohesion: 0.17
Nodes (7): evaluate(), _load_wav(), piecewise_aggregate_approximation(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 35 - "download_ast_weights"
Cohesion: 0.27
Nodes (3): download_ast_weights(), retrieve_cloud_weights(), _sha256()

### Community 36 - "batch_preprocessor.py"
Cohesion: 0.11
Nodes (7): detect_active_segment(), load_and_resample(), peak_normalize(), save_audio(), process_single_audio(), run_batch_preprocessing(), generate_5fold_splits()

### Community 37 - "SResNetTransfer"
Cohesion: 0.25
Nodes (3): SResNetTransfer, extract_features_batched(), main()

### Community 40 - "run_acoustic_inference"
Cohesion: 0.08
Nodes (32): build_report(), complete_external(), _read(), _write_markdown(), write_report(), modelKey(), 10. Ordered execution checklist, 4. Plan A — deploy AST as the production model (+24 more)

### Community 44 - "displayClassificationResults"
Cohesion: 0.50
Nodes (4): animateCounter(), step(), displayClassificationResults(), easeOutQuad()

### Community 48 - "encodeWAV"
Cohesion: 0.50
Nodes (3): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory()

### Community 51 - "eval_esc50_cv_features.py"
Cohesion: 0.11
Nodes (21): _batched_logits(), _checkpoint_arch_version(), evaluate(), evaluate_ensemble(), main(), _checkpoint_path(), _forward(), _load_split() (+13 more)

## Knowledge Gaps
- **4 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Dual-Modal Verification Diagnostic Plot`, `5-Fold Stratified Split Generator`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 508 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `modelKey()` connect `run_acoustic_inference` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.281) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Dual-Modal Verification Diagnostic Plot` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.010416666666666666 - nodes in this community are weakly interconnected._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.09879032258064516 - nodes in this community are weakly interconnected._
- **Should `models` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
- **Should `Node.js Server & Package Dependencies` be split into smaller, more focused modules?**
  _Cohesion score 0.053156146179401995 - nodes in this community are weakly interconnected._
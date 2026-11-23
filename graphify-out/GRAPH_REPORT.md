# Graph Report - Sound_Recognition  (2026-11-23)

## Corpus Check
- 92 files · ~128,560 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6019 file(s) not represented in the graph (top: .npy 5994, .csv 18, (none) 3)

## Summary
- 961 nodes · 1795 edges · 55 communities (40 shown, 15 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1cedd476`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- data_augmentation.py
- Audio Processing & Spectrogram Synthesis
- models
- Node.js Server & Package Dependencies
- PyTorch Neural Architectures & Models
- UI Animation & Status Management
- numpy
- main.py
- SoundSense — Environmental Sound Classification Studio
- pathlib
- Preproccesing/__init__.py
- SoundSense — Truthful High-Accuracy Recovery Plan
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- eval_ast.py
- renderTrainingChart
- Interactive Particle Visualizer
- train_cv.py
- SoundSense Architecture & System Specs
- showToast
- fx.js
- json
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- updateApiExplorer
- render
- Constant-Q Transform Visualization
- STFT Time-Frequency Visualization
- Audio Types Comparison Plot
- Waveform Amplitude Plot
- Stratified 5-Fold Split Generator
- drawFrame
- copilot-instructions.md
- json
- run
- report.py
- dataset_loader.py
- audio_processor.py
- loadTaxonomyCategory
- run_acoustic_inference
- evaluate_external
- load_model_checkpoint
- eval_robustness.py
- loadAnalyticsData
- displayClassificationResults
- eval_signal_gates.py
- 4. Plan A — deploy AST as the production model
- run
- onResize

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 37 edges
2. `classification_metrics()` - 17 edges
3. `loadAudioFile()` - 17 edges
4. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
5. `load_model_checkpoint()` - 16 edges
6. `switchView()` - 13 edges
7. `train_model()` - 12 edges
8. `model_status()` - 12 edges
9. `load_audio_waveform()` - 11 edges
10. `showToast()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `A7. Verification tests` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A5. Backend defaults and parity` --references--> `test_ast_checkpoint_available()`  [INFERRED]
  PLAN.md → tests/test_system_integration.py
- `A3. Add a clean AST evaluator` --references--> `classification_metrics()`  [INFERRED]
  PLAN.md → eval/metrics.py
- `A3. Add a clean AST evaluator` --references--> `run_acoustic_inference()`  [INFERRED]
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

## Communities (55 total, 15 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (164): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+156 more)

### Community 1 - "data_augmentation.py"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.13
Nodes (27): buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks(), drawFeatureSpectrogram() (+19 more)

### Community 3 - "models"
Cohesion: 0.07
Nodes (28): available, filename, sha256, sizeBytes, available, filename, sha256, sizeBytes (+20 more)

### Community 4 - "Node.js Server & Package Dependencies"
Cohesion: 0.06
Nodes (31): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+23 more)

### Community 5 - "PyTorch Neural Architectures & Models"
Cohesion: 0.10
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.14
Nodes (16): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), closeAccountMenu(), closeAuthModal(), fitAccountMenuPopover(), initActiveView(), openAuthModal() (+8 more)

### Community 7 - "numpy"
Cohesion: 0.05
Nodes (16): extract_mel_feature_vector(), cross_correlation_similarity(), extract_mel_sequence(), explore_audio(), get_hubert_embedding(), load_wav_fallback(), check_audio_durations(), get_audio_duration() (+8 more)

### Community 8 - "main.py"
Cohesion: 0.06
Nodes (30): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, current_checkpoint_inventory(), get_history(), get_sample_for_category() (+22 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Independently Measured Metrics (from evaluation harness), Key Features (+5 more)

### Community 11 - "Preproccesing/__init__.py"
Cohesion: 0.20
Nodes (6): evaluate(), _load_wav(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 12 - "SoundSense — Truthful High-Accuracy Recovery Plan"
Cohesion: 0.06
Nodes (35): 0. Non-negotiable rules, 11. Command cheat sheet, 12. Definition of done, 1.1 Root cause of the current low/absent scores, 1.2 The best real model already available, 1.3 Recommended plan, 1. Executive summary, 2.1 Verified measurements (+27 more)

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.20
Nodes (9): key, value, services, dockerfilePath, env, envVars, name, plan (+1 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.29
Nodes (7): Mel Spectrogram Acoustic Heatmap, MFCC Feature Representation Plot, Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 16 - "eval_ast.py"
Cohesion: 0.17
Nodes (6): _encode(), _eval_model(), run(), _summarize(), binary_ood_metrics(), classification_metrics()

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 19 - "train_cv.py"
Cohesion: 0.06
Nodes (24): evaluate(), main(), _checkpoint_path(), _forward(), _load_split(), _loader(), main(), _new_model() (+16 more)

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 21 - "showToast"
Cohesion: 0.18
Nodes (18): checkBackendStatus(), cleanupMic(), ensureAudioContext(), escapeHtml(), executeClassification(), finalizeMicRecording(), finishTrainingRun(), loadTrainingData() (+10 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "json"
Cohesion: 0.08
Nodes (14): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio(), download_ast_weights() (+6 more)

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
Cohesion: 0.13
Nodes (19): addFiles(), animate(), classifyOne(), csvCell(), downloadCsv(), esc(), fmtSize(), isAllowedAudio() (+11 more)

### Community 33 - "drawFrame"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 35 - "json"
Cohesion: 0.20
Nodes (3): run_batch_preprocessing(), generate_5fold_splits(), run_tests()

### Community 36 - "run"
Cohesion: 0.19
Nodes (6): run(), AcousticClassifierNet, compute_signal_metrics(), extract_acoustic_feature_vector(), get_acoustic_classifier(), _mlp_probabilities()

### Community 37 - "report.py"
Cohesion: 0.35
Nodes (6): build_report(), complete_external(), _read(), _write_markdown(), write_report(), A4. Update `eval/report.py`

### Community 38 - "dataset_loader.py"
Cohesion: 0.15
Nodes (3): ESC50MultiModalDataset, get_dataloaders(), test_pytorch_dataloader()

### Community 39 - "audio_processor.py"
Cohesion: 0.16
Nodes (6): detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio(), process_single_audio()

### Community 40 - "loadTaxonomyCategory"
Cohesion: 0.33
Nodes (5): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory(), loadTaxonomyCategory(), renderTaxonomyGrid()

### Community 41 - "run_acoustic_inference"
Cohesion: 0.18
Nodes (11): 10. Ordered execution checklist, A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, _ast_probabilities(), _build_silence_result(), _build_tone_result(), _load_ast_engine(), load_audio_waveform(), load_model_checkpoint() (+3 more)

### Community 42 - "evaluate_external"
Cohesion: 0.24
Nodes (5): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval()

### Community 43 - "load_model_checkpoint"
Cohesion: 0.39
Nodes (7): run_tests(), test_ast_checkpoint_available(), test_ast_real_audio_inference(), test_filename_invariance(), test_invalid_model_returns_400(), test_model_status_report(), test_other_checkpoints_trained_and_working()

### Community 44 - "eval_robustness.py"
Cohesion: 0.43
Nodes (4): run(), _snr_noise(), _variants(), _wav_bytes()

### Community 45 - "loadAnalyticsData"
Cohesion: 0.50
Nodes (5): applyTheme(), drawClassDistributionChart(), drawModelPerformanceChart(), loadAnalyticsData(), A6. Minimal front-end integration

### Community 46 - "displayClassificationResults"
Cohesion: 0.50
Nodes (4): animateCounter(), step(), displayClassificationResults(), easeOutQuad()

### Community 50 - "4. Plan A — deploy AST as the production model"
Cohesion: 0.29
Nodes (7): modelKey(), 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A3. Add a clean AST evaluator, A5. Backend defaults and parity, A7. Verification tests

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 495 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **15 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A5. Backend defaults and parity` connect `4. Plan A — deploy AST as the production model` to `load_model_checkpoint`?**
  _High betweenness centrality (0.278) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.011111111111111112 - nodes in this community are weakly interconnected._
- **Why does `modelKey()` connect `4. Plan A — deploy AST as the production model` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.278) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.12535612535612536 - nodes in this community are weakly interconnected._
- **Why does `test_ast_checkpoint_available()` connect `load_model_checkpoint` to `run_acoustic_inference`, `4. Plan A — deploy AST as the production model`?**
  _High betweenness centrality (0.231) - this node is a cross-community bridge._
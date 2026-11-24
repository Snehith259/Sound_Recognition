# Graph Report - Sound_Recognition  (2026-11-24)

## Corpus Check
- 108 files · ~201,337 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6020 file(s) not represented in the graph (top: .npy 5994, .csv 18, (none) 3)

## Summary
- 1039 nodes · 2025 edges · 67 communities (43 shown, 24 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `52bb5545`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- data_augmentation.py
- Audio Processing & Spectrogram Synthesis
- models
- Node.js Server & Package Dependencies
- get_model
- UI Animation & Status Management
- soundfile
- main.py
- SoundSense — Environmental Sound Classification Studio
- json
- eval_legacy_mlp_external.py
- SoundSense — Truthful High-Accuracy Recovery Plan
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- acoustic_inference.py
- renderTrainingChart
- Interactive Particle Visualizer
- train_cv.py
- SoundSense Architecture & System Specs
- showToast
- fx.js
- build_external_set.py
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
- acoustic_inference.py
- os
- dataset_loader.py
- batch_preprocessor.py
- 4. Plan A — deploy AST as the production model
- split_long_audio.py
- benchmark_candidates.py
- Preproccesing/__init__.py
- test_system_integration.py
- displayClassificationResults
- eval_esc50_cv.py
- eval_ast.py
- numpy
- SResNetBackbone
- test_mixup.py
- fast_test_sresnet.py
- list_audio_files
- eval_robustness.py
- Hubert_t-Sne_visu.py
- time
- get_audio_duration
- ResNet18_ESC50
- verify_and_report.py

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 43 edges
2. `loadAudioFile()` - 17 edges
3. `classification_metrics()` - 17 edges
4. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
5. `switchView()` - 14 edges
6. `extract_log_mel_spectrogram()` - 13 edges
7. `showToast()` - 12 edges
8. `load_model_checkpoint()` - 12 edges
9. `train_model()` - 12 edges
10. `model_status()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A7. Verification tests` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A3. Add a clean AST evaluator` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
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

## Communities (67 total, 24 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (176): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+168 more)

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
Cohesion: 0.05
Nodes (34): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+26 more)

### Community 5 - "get_model"
Cohesion: 0.18
Nodes (3): CNNGRU, get_model(), VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.09
Nodes (30): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), applyTheme(), checkBackendStatus(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart() (+22 more)

### Community 7 - "soundfile"
Cohesion: 0.15
Nodes (3): explore_audio(), list_audio_files(), resample_file()

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (41): build_report(), complete_external(), _read(), _write_markdown(), write_report(), add_history(), auth_me(), clear_history() (+33 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Independently Measured Metrics (from evaluation harness), Key Features (+5 more)

### Community 10 - "json"
Cohesion: 0.18
Nodes (4): run(), _signals(), run_batch_preprocessing(), generate_5fold_splits()

### Community 11 - "eval_legacy_mlp_external.py"
Cohesion: 0.18
Nodes (6): run(), AcousticClassifierNet, compute_signal_metrics(), extract_acoustic_feature_vector(), get_acoustic_classifier(), _mlp_probabilities()

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

### Community 16 - "acoustic_inference.py"
Cohesion: 0.14
Nodes (13): 10. Ordered execution checklist, A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, _ast_probabilities(), _build_silence_result(), _build_tone_result(), _load_ast_engine(), load_audio_waveform(), load_model_checkpoint() (+5 more)

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "train_cv.py"
Cohesion: 0.15
Nodes (15): evaluate(), main(), _checkpoint_path(), _forward(), _load_split(), _loader(), main(), _new_model() (+7 more)

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 21 - "showToast"
Cohesion: 0.18
Nodes (17): cleanupMic(), encodeWAV(), ensureAudioContext(), executeClassification(), finalizeMicRecording(), finishTrainingRun(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory() (+9 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "build_external_set.py"
Cohesion: 0.07
Nodes (13): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio(), download_ast_weights() (+5 more)

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

### Community 33 - "drawFrame"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 35 - "json"
Cohesion: 0.18
Nodes (7): _encode(), _eval_model(), run(), _summarize(), _gate(), run(), evaluate()

### Community 36 - "acoustic_inference.py"
Cohesion: 0.14
Nodes (8): calibrate_temperature(), compute_metrics(), extract_features_batched(), load_split_features(), SResNetTransfer, train_and_export_all(), extract_features_batched(), main()

### Community 37 - "os"
Cohesion: 0.19
Nodes (3): extract_mel_feature_vector(), cross_correlation_similarity(), extract_mel_sequence()

### Community 39 - "batch_preprocessor.py"
Cohesion: 0.14
Nodes (5): detect_active_segment(), load_and_resample(), peak_normalize(), save_audio(), process_single_audio()

### Community 40 - "4. Plan A — deploy AST as the production model"
Cohesion: 0.29
Nodes (7): modelKey(), 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A3. Add a clean AST evaluator, A5. Backend defaults and parity, A7. Verification tests

### Community 42 - "benchmark_candidates.py"
Cohesion: 0.24
Nodes (4): extract_features(), SqueezeNetAudio, SResNetAudio, test_model()

### Community 43 - "Preproccesing/__init__.py"
Cohesion: 0.25
Nodes (5): piecewise_aggregate_approximation(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 44 - "test_system_integration.py"
Cohesion: 0.39
Nodes (7): run_tests(), test_ast_checkpoint_available(), test_ast_real_audio_inference(), test_filename_invariance(), test_invalid_model_returns_400(), test_model_status_report(), test_other_checkpoints_trained_and_working()

### Community 46 - "displayClassificationResults"
Cohesion: 0.50
Nodes (4): animateCounter(), step(), displayClassificationResults(), easeOutQuad()

### Community 49 - "eval_esc50_cv.py"
Cohesion: 0.12
Nodes (5): evaluate(), _load_wav(), CNNGRUFast, ResNet18Transfer, VGG16Transfer

### Community 51 - "eval_ast.py"
Cohesion: 0.16
Nodes (7): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval(), binary_ood_metrics(), classification_metrics()

### Community 52 - "numpy"
Cohesion: 0.27
Nodes (4): load_wav_fallback(), ensure_test_corpus(), run_full_evaluation(), simulate_microphone_capture()

### Community 60 - "eval_robustness.py"
Cohesion: 0.36
Nodes (4): run(), _snr_noise(), _variants(), _wav_bytes()

### Community 67 - "verify_and_report.py"
Cohesion: 0.22
Nodes (3): generate_feature_visualization(), test_pytorch_dataloader(), verify_feature_integrity()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 539 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A5. Backend defaults and parity` connect `4. Plan A — deploy AST as the production model` to `test_system_integration.py`?**
  _High betweenness centrality (0.295) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.010416666666666666 - nodes in this community are weakly interconnected._
- **Why does `modelKey()` connect `4. Plan A — deploy AST as the production model` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.294) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.12535612535612536 - nodes in this community are weakly interconnected._
- **Why does `test_ast_checkpoint_available()` connect `test_system_integration.py` to `4. Plan A — deploy AST as the production model`, `acoustic_inference.py`?**
  _High betweenness centrality (0.247) - this node is a cross-community bridge._
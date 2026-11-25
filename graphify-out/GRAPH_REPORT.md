# Graph Report - Sound_Recognition  (2026-11-25)

## Corpus Check
- 135 files · ~279,408 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6047 file(s) not represented in the graph (top: .npy 5994, .pt 21, .csv 18)

## Summary
- 1079 nodes · 2175 edges · 80 communities (47 shown, 33 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 61 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cc15ae84`
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
- soundfile
- main.py
- SoundSense — Environmental Sound Classification Studio
- eval_ast.py
- json
- SoundSense — Truthful High-Accuracy Recovery Plan
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- test_system_integration.py
- renderTrainingChart
- Interactive Particle Visualizer
- eval_external.py
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
- eval_legacy_mlp_external.py
- copilot-instructions.md
- download_ast_weights
- batch_preprocessor.py
- 4. Plan A — deploy AST as the production model
- evaluate
- drawFrame
- test_all_models_large_scale.py
- train_improved_models.py
- benchmark_candidates.py
- batch_preprocessor.py
- run_acoustic_inference
- numpy
- ResNet18_ESC50
- data_augmentation.py
- train_cv.py
- get_audio_duration
- test_mixup.py
- verify_and_report.py
- SResNetBackbone
- onResize
- 5. Plan B — train the four architecture checkpoints (optional, GPU)
- time
- list_audio_files
- _load_ast_engine
- dataset_splits.py
- 2. Evidence and current state
- acoustic_inference.py
- 6. Expand the external and ambient evidence
- 7. Calibration, confidence, and OOD policy
- loading_torchaudio.py
- displayClassificationResults
- 1. Executive summary
- split_audio
- get_hubert_embedding

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 47 edges
2. `classification_metrics()` - 19 edges
3. `load_model_checkpoint()` - 17 edges
4. `loadAudioFile()` - 17 edges
5. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
6. `switchView()` - 14 edges
7. `extract_log_mel_spectrogram()` - 13 edges
8. `calibrate_temperature()` - 12 edges
9. `compute_metrics()` - 12 edges
10. `model_status()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A7. Verification tests` --references--> `run_acoustic_inference()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A4. Update `eval/report.py`` --references--> `_write_markdown()`  [INFERRED]
  PLAN.md → eval/report.py
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

## Communities (80 total, 33 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (176): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+168 more)

### Community 1 - "acoustic_inference.py"
Cohesion: 0.09
Nodes (12): train_and_export(), calibrate_temperature(), CNNGRUFast, compute_metrics(), extract_features_batched(), load_split_features(), ResNet18Transfer, SResNetTransfer (+4 more)

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
Cohesion: 0.10
Nodes (27): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), applyTheme(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+19 more)

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (34): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, current_checkpoint_inventory(), get_history(), get_sample_for_category() (+26 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Independently Measured Metrics (from evaluation harness), Key Features (+5 more)

### Community 10 - "eval_ast.py"
Cohesion: 0.14
Nodes (8): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval(), binary_ood_metrics(), classification_metrics(), A3. Add a clean AST evaluator

### Community 11 - "json"
Cohesion: 0.12
Nodes (10): run(), _signals(), _gate(), run(), build_report(), _read(), _write_markdown(), write_report() (+2 more)

### Community 12 - "SoundSense — Truthful High-Accuracy Recovery Plan"
Cohesion: 0.18
Nodes (10): 0. Non-negotiable rules, 11. Command cheat sheet, 12. Definition of done, 3. Acceptance gates, 8. Final holdout protocol, 9. What not to do, Appendix A — Why the AST numbers are credible, Appendix B — Report file map (+2 more)

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.18
Nodes (10): key, value, services, buildCommand, env, envVars, name, plan (+2 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.29
Nodes (7): Mel Spectrogram Acoustic Heatmap, MFCC Feature Representation Plot, Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 16 - "test_system_integration.py"
Cohesion: 0.31
Nodes (8): model_status(), run_tests(), test_ast_checkpoint_available(), test_ast_real_audio_inference(), test_filename_invariance(), test_invalid_model_returns_400(), test_model_status_report(), test_other_checkpoints_trained_and_working()

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 19 - "eval_external.py"
Cohesion: 0.25
Nodes (8): _encode(), _eval_model(), run(), _summarize(), run(), _snr_noise(), _variants(), _wav_bytes()

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 21 - "showToast"
Cohesion: 0.16
Nodes (19): checkBackendStatus(), cleanupMic(), encodeWAV(), ensureAudioContext(), executeClassification(), finalizeMicRecording(), finishTrainingRun(), generateSyntheticAudioBlob() (+11 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "build_external_set.py"
Cohesion: 0.11
Nodes (11): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio(), ensure_test_corpus() (+3 more)

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

### Community 33 - "eval_legacy_mlp_external.py"
Cohesion: 0.18
Nodes (6): run(), AcousticClassifierNet, compute_signal_metrics(), extract_acoustic_feature_vector(), get_acoustic_classifier(), _mlp_probabilities()

### Community 35 - "download_ast_weights"
Cohesion: 0.36
Nodes (3): download_ast_weights(), retrieve_cloud_weights(), _sha256()

### Community 36 - "batch_preprocessor.py"
Cohesion: 0.18
Nodes (5): detect_active_segment(), load_and_resample(), peak_normalize(), save_audio(), process_single_audio()

### Community 37 - "4. Plan A — deploy AST as the production model"
Cohesion: 0.22
Nodes (9): complete_external(), modelKey(), 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A4. Update `eval/report.py`, A5. Backend defaults and parity, A6. Minimal front-end integration (+1 more)

### Community 39 - "drawFrame"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 40 - "test_all_models_large_scale.py"
Cohesion: 0.26
Nodes (8): A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, load_model_checkpoint(), warm_up_engine(), create_mixed_sounds_suite(), evaluate_esc50_test(), evaluate_mixed_sounds(), load_fold1_test_manifest(), main()

### Community 41 - "train_improved_models.py"
Cohesion: 0.24
Nodes (7): load_data(), main(), spec_augment(), train_improved_cnngru(), __init__(), train_improved_resnet18(), __init__()

### Community 42 - "benchmark_candidates.py"
Cohesion: 0.22
Nodes (4): extract_features(), SqueezeNetAudio, SResNetAudio, test_model()

### Community 43 - "batch_preprocessor.py"
Cohesion: 0.19
Nodes (5): piecewise_aggregate_approximation(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 44 - "run_acoustic_inference"
Cohesion: 0.21
Nodes (8): _build_silence_result(), _build_tone_result(), load_audio_waveform(), run_acoustic_inference(), build_test_suite(), run_multi_file_benchmark(), test_anti_faking_guarantee(), test_multiformat_ambient_and_pure()

### Community 48 - "numpy"
Cohesion: 0.12
Nodes (4): extract_mel_feature_vector(), cross_correlation_similarity(), extract_mel_sequence(), explore_audio()

### Community 50 - "data_augmentation.py"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 51 - "train_cv.py"
Cohesion: 0.14
Nodes (17): evaluate(), main(), _checkpoint_path(), _forward(), _load_split(), _loader(), main(), _new_model() (+9 more)

### Community 54 - "verify_and_report.py"
Cohesion: 0.11
Nodes (5): ESC50MultiModalDataset, get_dataloaders(), generate_feature_visualization(), test_pytorch_dataloader(), verify_feature_integrity()

### Community 57 - "5. Plan B — train the four architecture checkpoints (optional, GPU)"
Cohesion: 0.33
Nodes (6): 5. Plan B — train the four architecture checkpoints (optional, GPU), B1. Train, B2. Five-fold evaluation, B3. External dev evaluation, B4. Publication gate for the four models, B5. Optional distillation (stretch, not required)

### Community 60 - "_load_ast_engine"
Cohesion: 0.50
Nodes (3): 10. Ordered execution checklist, _ast_probabilities(), _load_ast_engine()

### Community 62 - "2. Evidence and current state"
Cohesion: 0.40
Nodes (5): 2.1 Verified measurements, 2.2 Confidence behavior is real and useful, 2.3 Failure cases to acknowledge, 2.4 What “R²” means in this repository, 2. Evidence and current state

### Community 73 - "6. Expand the external and ambient evidence"
Cohesion: 0.40
Nodes (5): 6.1 Current composition, 6.2 Required additions for a credible ambient claim, 6.3 Metric rules, 6.4 Rebuild commands, 6. Expand the external and ambient evidence

### Community 74 - "7. Calibration, confidence, and OOD policy"
Cohesion: 0.40
Nodes (5): 7.1 Calibration, 7.2 Selective prediction, 7.3 OOD, 7.4 R² display rule, 7. Calibration, confidence, and OOD policy

### Community 76 - "displayClassificationResults"
Cohesion: 0.50
Nodes (4): animateCounter(), step(), displayClassificationResults(), easeOutQuad()

### Community 77 - "1. Executive summary"
Cohesion: 0.50
Nodes (4): 1.1 Root cause of the current low/absent scores, 1.2 The best real model already available, 1.3 Recommended plan, 1. Executive summary

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 555 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **33 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A5. Backend defaults and parity` connect `4. Plan A — deploy AST as the production model` to `test_system_integration.py`?**
  _High betweenness centrality (0.292) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.010416666666666666 - nodes in this community are weakly interconnected._
- **Why does `modelKey()` connect `4. Plan A — deploy AST as the production model` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.291) - this node is a cross-community bridge._
- **Should `acoustic_inference.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09435707678075855 - nodes in this community are weakly interconnected._
- **Why does `test_ast_checkpoint_available()` connect `test_system_integration.py` to `test_all_models_large_scale.py`, `4. Plan A — deploy AST as the production model`?**
  _High betweenness centrality (0.249) - this node is a cross-community bridge._
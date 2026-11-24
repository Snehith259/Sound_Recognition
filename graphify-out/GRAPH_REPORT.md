# Graph Report - Sound_Recognition  (2026-11-24)

## Corpus Check
- 109 files · ~201,738 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6020 file(s) not represented in the graph (top: .npy 5994, .csv 18, (none) 3)

## Summary
- 1040 nodes · 2038 edges · 57 communities (38 shown, 19 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 65 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `700fcfe0`
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
- eval_ast.py
- compute_signal_metrics
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
- Any
- copilot-instructions.md
- acoustic_inference.py
- Resample_files_SR.py
- run
- displayClassificationResults
- eval_esc50_cv.py
- encodeWAV
- test_upgraded_audio.py
- Preproccesing/__init__.py
- run_acoustic_inference
- os
- SqueezeNetAudio
- train_cv.py
- Hubert_t-Sne_visu.py
- SResNetBackbone
- test_mixup.py
- fast_test_sresnet.py
- list_audio_files
- numpy
- time

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 43 edges
2. `loadAudioFile()` - 17 edges
3. `classification_metrics()` - 17 edges
4. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
5. `load_model_checkpoint()` - 16 edges
6. `switchView()` - 14 edges
7. `extract_log_mel_spectrogram()` - 13 edges
8. `showToast()` - 12 edges
9. `model_status()` - 12 edges
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
- `Mel Spectrogram Acoustic Heatmap` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mel_spectrogram.png → Preproccesing/README.md

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

## Communities (57 total, 19 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (176): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+168 more)

### Community 1 - "data_augmentation.py"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

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
Cohesion: 0.13
Nodes (4): CNNGRU, get_model(), ResNet18_ESC50, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.10
Nodes (26): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), applyTheme(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+18 more)

### Community 7 - "soundfile"
Cohesion: 0.14
Nodes (4): explore_audio(), check_audio_durations(), get_audio_duration(), split_audio()

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (33): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, current_checkpoint_inventory(), get_history(), get_sample_for_category() (+25 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Independently Measured Metrics (from evaluation harness), Key Features (+5 more)

### Community 10 - "eval_ast.py"
Cohesion: 0.10
Nodes (11): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval(), _eval_model(), run(), _summarize() (+3 more)

### Community 11 - "compute_signal_metrics"
Cohesion: 0.22
Nodes (5): AcousticClassifierNet, compute_signal_metrics(), extract_acoustic_feature_vector(), get_acoustic_classifier(), _mlp_probabilities()

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
Cohesion: 0.29
Nodes (7): Mel Spectrogram Acoustic Heatmap, MFCC Feature Representation Plot, Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 16 - "test_system_integration.py"
Cohesion: 0.17
Nodes (14): modelKey(), 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A3. Add a clean AST evaluator, A5. Backend defaults and parity, A7. Verification tests, run_tests() (+6 more)

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "eval_external.py"
Cohesion: 0.39
Nodes (5): _encode(), run(), _snr_noise(), _variants(), _wav_bytes()

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 21 - "showToast"
Cohesion: 0.19
Nodes (18): checkBackendStatus(), cleanupMic(), ensureAudioContext(), escapeHtml(), executeClassification(), finalizeMicRecording(), finishTrainingRun(), loadTrainingData() (+10 more)

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "build_external_set.py"
Cohesion: 0.09
Nodes (15): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio(), download_ast_weights() (+7 more)

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

### Community 33 - "Any"
Cohesion: 0.18
Nodes (10): _gate(), run(), build_report(), complete_external(), _read(), _write_markdown(), write_report(), A4. Update `eval/report.py` (+2 more)

### Community 36 - "acoustic_inference.py"
Cohesion: 0.16
Nodes (8): calibrate_temperature(), compute_metrics(), extract_features_batched(), load_split_features(), SResNetTransfer, train_and_export_all(), extract_features_batched(), main()

### Community 39 - "displayClassificationResults"
Cohesion: 0.50
Nodes (4): animateCounter(), step(), displayClassificationResults(), easeOutQuad()

### Community 40 - "eval_esc50_cv.py"
Cohesion: 0.12
Nodes (5): evaluate(), _load_wav(), CNNGRUFast, ResNet18Transfer, VGG16Transfer

### Community 41 - "encodeWAV"
Cohesion: 0.50
Nodes (3): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory()

### Community 42 - "test_upgraded_audio.py"
Cohesion: 0.33
Nodes (3): extract_features(), SResNetAudio, test_model()

### Community 43 - "Preproccesing/__init__.py"
Cohesion: 0.07
Nodes (14): detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio(), process_single_audio(), ESC50MultiModalDataset, get_dataloaders() (+6 more)

### Community 44 - "run_acoustic_inference"
Cohesion: 0.13
Nodes (15): 10. Ordered execution checklist, A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, _ast_probabilities(), _build_silence_result(), _build_tone_result(), get_engine_status(), _load_ast_engine(), load_audio_waveform() (+7 more)

### Community 51 - "train_cv.py"
Cohesion: 0.15
Nodes (15): evaluate(), main(), _checkpoint_path(), _forward(), _load_split(), _loader(), main(), _new_model() (+7 more)

### Community 61 - "numpy"
Cohesion: 0.20
Nodes (4): extract_mel_feature_vector(), load_wav_fallback(), generate_feature_visualization(), verify_feature_integrity()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 538 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `modelKey()` connect `test_system_integration.py` to `Front-End UI & State Controller`, `render`?**
  _High betweenness centrality (0.304) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.010416666666666666 - nodes in this community are weakly interconnected._
- **Why does `test_ast_checkpoint_available()` connect `test_system_integration.py` to `run_acoustic_inference`?**
  _High betweenness centrality (0.259) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.09879032258064516 - nodes in this community are weakly interconnected._
- **Should `models` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._
# Graph Report - Sound_Recognition  (2026-11-23)

## Corpus Check
- 91 files · ~128,531 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6019 file(s) not represented in the graph (top: .npy 5994, .csv 18, (none) 3)

## Summary
- 938 nodes · 1772 edges · 49 communities (35 shown, 14 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 66 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `40843f12`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- data_augmentation.py
- Audio Processing & Spectrogram Synthesis
- ast
- Node.js Server & Package Dependencies
- PyTorch Neural Architectures & Models
- UI Animation & Status Management
- soundfile
- main.py
- SoundSense — Environmental Sound Classification Studio
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
- trigger_training
- torch
- os
- numpy
- __init__.py
- loadTaxonomyCategory
- run_acoustic_inference
- lifespan
- load_model_checkpoint
- split_long_audio.py
- loadAnalyticsData
- displayClassificationResults

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 36 edges
2. `loadAudioFile()` - 17 edges
3. `classification_metrics()` - 17 edges
4. `SoundSense — Truthful High-Accuracy Recovery Plan` - 17 edges
5. `load_model_checkpoint()` - 16 edges
6. `switchView()` - 13 edges
7. `model_status()` - 12 edges
8. `train_model()` - 12 edges
9. `showToast()` - 11 edges
10. `load_audio_waveform()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `B4. Publication gate for the four models` --references--> `load_model_checkpoint()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A5. Backend defaults and parity` --references--> `modelKey()`  [INFERRED]
  PLAN.md → Front-End/app.js
- `10. Ordered execution checklist` --references--> `model_status()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A2. Add the AST engine to `Preproccesing/acoustic_inference.py`` --references--> `model_status()`  [INFERRED]
  PLAN.md → Preproccesing/acoustic_inference.py
- `A2. Add the AST engine to `Preproccesing/acoustic_inference.py`` --references--> `load_model_checkpoint()`  [INFERRED]
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

## Communities (49 total, 14 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (164): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+156 more)

### Community 1 - "data_augmentation.py"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.13
Nodes (27): buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks(), drawFeatureSpectrogram() (+19 more)

### Community 3 - "ast"
Cohesion: 0.22
Nodes (8): available, filename, sha256, sizeBytes, generatedAt, models, ast, schemaVersion

### Community 4 - "Node.js Server & Package Dependencies"
Cohesion: 0.06
Nodes (31): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+23 more)

### Community 5 - "PyTorch Neural Architectures & Models"
Cohesion: 0.09
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.14
Nodes (16): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), closeAccountMenu(), closeAuthModal(), fitAccountMenuPopover(), initActiveView(), openAuthModal() (+8 more)

### Community 7 - "soundfile"
Cohesion: 0.17
Nodes (5): explore_audio(), check_audio_durations(), get_audio_duration(), list_audio_files(), resample_file()

### Community 8 - "main.py"
Cohesion: 0.08
Nodes (17): add_history(), auth_me(), clear_history(), current_checkpoint_inventory(), get_history(), get_sample_for_category(), get_stats(), get_training_job() (+9 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.13
Nodes (13): Deployed Model (AST-ESC50), Local diagnostics, SoundSense evaluation report, Status, API Endpoints, Deployed Model: AST-ESC50, Independently Measured Metrics (from evaluation harness), Key Features (+5 more)

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
Cohesion: 0.06
Nodes (26): check_format_parity(), evaluate_external(), evaluate_fold1(), get_git_commit(), wilson_interval(), evaluate(), _load_wav(), _encode() (+18 more)

### Community 17 - "renderTrainingChart"
Cohesion: 0.67
Nodes (3): drawCurve(), renderTrainingChart(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

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
Cohesion: 0.06
Nodes (25): _assign_splits(), build(), _github_tree(), _request(), _safe_id(), _transcode(), _write_audio(), build_report() (+17 more)

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
Nodes (20): addFiles(), animate(), classifyOne(), csvCell(), downloadCsv(), esc(), fmtSize(), isAllowedAudio() (+12 more)

### Community 33 - "drawFrame"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 35 - "trigger_training"
Cohesion: 0.17
Nodes (11): cloud_sync(), CloudSyncRequest, login(), LoginRequest, logout(), predict(), predict_batch(), save_model() (+3 more)

### Community 38 - "numpy"
Cohesion: 0.15
Nodes (5): extract_mel_feature_vector(), cross_correlation_similarity(), load_wav_fallback(), generate_feature_visualization(), verify_feature_integrity()

### Community 39 - "__init__.py"
Cohesion: 0.06
Nodes (18): list_audio_files(), run(), _signals(), detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio() (+10 more)

### Community 40 - "loadTaxonomyCategory"
Cohesion: 0.33
Nodes (5): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory(), loadTaxonomyCategory(), renderTaxonomyGrid()

### Community 41 - "run_acoustic_inference"
Cohesion: 0.15
Nodes (13): 10. Ordered execution checklist, 4. Plan A — deploy AST as the production model, A0. Preserve the current working tree, A1. Add a reproducible AST weight bootstrap, A2. Add the AST engine to `Preproccesing/acoustic_inference.py`, A3. Add a clean AST evaluator, A7. Verification tests, _ast_probabilities() (+5 more)

### Community 43 - "load_model_checkpoint"
Cohesion: 0.30
Nodes (9): A5. Backend defaults and parity, load_model_checkpoint(), run_tests(), test_ast_checkpoint_available(), test_ast_real_audio_inference(), test_filename_invariance(), test_missing_model_returns_503(), test_model_status_report() (+1 more)

### Community 45 - "loadAnalyticsData"
Cohesion: 0.50
Nodes (5): applyTheme(), drawClassDistributionChart(), drawModelPerformanceChart(), loadAnalyticsData(), A6. Minimal front-end integration

### Community 46 - "displayClassificationResults"
Cohesion: 0.50
Nodes (4): animateCounter(), step(), displayClassificationResults(), easeOutQuad()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 477 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `A5. Backend defaults and parity` connect `load_model_checkpoint` to `run_acoustic_inference`, `render`?**
  _High betweenness centrality (0.317) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.011111111111111112 - nodes in this community are weakly interconnected._
- **Why does `modelKey()` connect `render` to `Front-End UI & State Controller`, `load_model_checkpoint`?**
  _High betweenness centrality (0.317) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.12535612535612536 - nodes in this community are weakly interconnected._
- **Why does `test_other_checkpoints_optional()` connect `load_model_checkpoint` to `run_acoustic_inference`?**
  _High betweenness centrality (0.209) - this node is a cross-community bridge._
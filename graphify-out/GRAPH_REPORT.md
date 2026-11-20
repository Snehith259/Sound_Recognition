# Graph Report - Sound_Recognition  (2026-11-20)

## Corpus Check
- 63 files · ~73,308 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6029 file(s) not represented in the graph (top: .npy 5994, .csv 17, .mpg 11)

## Summary
- 779 nodes · 1320 edges · 47 communities (31 shown, 16 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 43 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `006f78f5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- data_augmentation.py
- Audio Processing & Spectrogram Synthesis
- Checkpoint Manifest & Metadata
- Node.js Server & Package Dependencies
- PyTorch Neural Architectures & Models
- UI Animation & Status Management
- soundfile
- main.py
- SoundSense — Environmental Sound Classification Studio
- sys
- os
- batch_preprocessor.py
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- acoustic_inference.py
- Model Training & Progress Monitoring
- Interactive Particle Visualizer
- train_models.py
- SoundSense Architecture & System Specs
- Batch File Processing
- fx.js
- benchmark_all_models_multiformat.py
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
- Hubert_t-Sne_visu.py
- load_and_resample
- ESC50MultiModalDataset
- __init__.py
- librosa
- get_audio_duration
- download_checkpoints.py
- loadAnalyticsData

## God Nodes (most connected - your core abstractions)
1. `run_acoustic_inference()` - 19 edges
2. `loadAudioFile()` - 17 edges
3. `switchView()` - 13 edges
4. `showToast()` - 11 edges
5. `load_and_resample()` - 11 edges
6. `extract_all_features()` - 11 edges
7. `render()` - 10 edges
8. `process_single_audio()` - 10 edges
9. `finalizeMicRecording()` - 9 edges
10. `drawFeatureSpectrogram()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Mel Spectrogram Acoustic Heatmap` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mel_spectrogram.png → Preproccesing/README.md
- `MFCC Feature Representation Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mfcc_features.png → Preproccesing/README.md
- `predict()` --indirect_call--> `run_acoustic_inference()`  [INFERRED]
  main.py → Preproccesing/acoustic_inference.py
- `predict_batch()` --indirect_call--> `run_acoustic_inference()`  [INFERRED]
  main.py → Preproccesing/acoustic_inference.py
- `Dual-Modal Verification Diagnostic Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Preproccesing/reports/feature_visualization.png → Preproccesing/README.md

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

## Communities (47 total, 16 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (164): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+156 more)

### Community 1 - "data_augmentation.py"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.07
Nodes (48): animateCounter(), step(), buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), cleanupMic(), computeAcousticSpectrogram(), computeActiveSegment() (+40 more)

### Community 3 - "Checkpoint Manifest & Metadata"
Cohesion: 0.07
Nodes (27): accuracy, description, filename, size_mb, top5, models, cnngru_v08, resnet18_v1 (+19 more)

### Community 4 - "Node.js Server & Package Dependencies"
Cohesion: 0.07
Nodes (29): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+21 more)

### Community 5 - "PyTorch Neural Architectures & Models"
Cohesion: 0.09
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.11
Nodes (21): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), checkBackendStatus(), closeAccountMenu(), closeAuthModal(), escapeHtml(), fitAccountMenuPopover() (+13 more)

### Community 7 - "soundfile"
Cohesion: 0.16
Nodes (4): explore_audio(), list_audio_files(), resample_file(), split_audio()

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (27): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, get_history(), get_sample_for_category(), get_stats() (+19 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.33
Nodes (5): API Endpoints, Architecture, Deployed Inference Engine, Key Features, SoundSense — Environmental Sound Classification Studio

### Community 10 - "sys"
Cohesion: 0.13
Nodes (4): build_test_suite(), run_multi_file_benchmark(), test_api_predict_concurrency(), run_tests()

### Community 11 - "os"
Cohesion: 0.19
Nodes (3): extract_mel_feature_vector(), cross_correlation_similarity(), extract_mel_sequence()

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
Cohesion: 0.08
Nodes (19): _warm(), AcousticClassifierNet, _ast_probabilities(), _build_silence_result(), compute_signal_metrics(), extract_acoustic_feature_vector(), _find_local_ast_snapshot(), get_acoustic_classifier() (+11 more)

### Community 17 - "Model Training & Progress Monitoring"
Cohesion: 0.40
Nodes (6): drawCurve(), finishTrainingRun(), pollTrainingProgress(), renderTrainingChart(), simulateClientTraining(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "train_models.py"
Cohesion: 0.09
Nodes (9): calibrate_temperature(), CNNGRUFast, compute_metrics(), extract_features_batched(), load_split_features(), ResNet18Transfer, SResNetFast, train_and_export_all() (+1 more)

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "benchmark_all_models_multiformat.py"
Cohesion: 0.36
Nodes (3): ensure_test_corpus(), run_full_evaluation(), simulate_microphone_capture()

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

### Community 37 - "load_and_resample"
Cohesion: 0.18
Nodes (5): detect_active_segment(), load_and_resample(), peak_normalize(), save_audio(), process_single_audio()

### Community 38 - "ESC50MultiModalDataset"
Cohesion: 0.14
Nodes (3): ESC50MultiModalDataset, get_dataloaders(), test_pytorch_dataloader()

### Community 39 - "__init__.py"
Cohesion: 0.25
Nodes (5): piecewise_aggregate_approximation(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 45 - "loadAnalyticsData"
Cohesion: 0.67
Nodes (4): applyTheme(), drawClassDistributionChart(), drawModelPerformanceChart(), loadAnalyticsData()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 421 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **16 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_acoustic_inference()` connect `acoustic_inference.py` to `main.py`, `sys`, `benchmark_all_models_multiformat.py`, `__init__.py`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `run_acoustic_inference()` (e.g. with `predict()` and `predict_batch()`) actually correct?**
  _`run_acoustic_inference()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.011049723756906077 - nodes in this community are weakly interconnected._
- **Why does `extract_log_mel_spectrogram()` connect `__init__.py` to `acoustic_inference.py`, `librosa`, `soundfile`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.07142857142857142 - nodes in this community are weakly interconnected._
- **Why does `ESC50MultiModalDataset` connect `ESC50MultiModalDataset` to `__init__.py`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
# Graph Report - Sound_Recognition  (2026-11-20)

## Corpus Check
- 55 files · ~60,725 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6018 file(s) not represented in the graph (top: .npy 5994, .csv 17, (none) 3)

## Summary
- 734 nodes · 1206 edges · 55 communities (33 shown, 22 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e50aa97a`
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
- post
- SoundSense — Environmental Sound Classification Studio
- loadAnalyticsData
- numpy
- acoustic_inference.py
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- json
- Model Training & Progress Monitoring
- Interactive Particle Visualizer
- train_models.py
- SoundSense Architecture & System Specs
- Batch File Processing
- fx.js
- WAV Encoding & Audio Synthesis
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- updateApiExplorer
- Graphics Animation Loop
- Constant-Q Transform Visualization
- STFT Time-Frequency Visualization
- Audio Types Comparison Plot
- Waveform Amplitude Plot
- Stratified 5-Fold Split Generator
- __init__.py
- copilot-instructions.md
- Hubert_t-Sne_visu.py
- __init__.py
- ESC50MultiModalDataset
- run_acoustic_inference
- showToast
- get
- _load_ast_engine
- get_audio_duration
- warm_up_engine
- VGG16Transfer
- SResNetFast
- clear_history
- int32_hash
- _is_soundsense_instance
- _port_is_open

## God Nodes (most connected - your core abstractions)
1. `loadAudioFile()` - 17 edges
2. `run_acoustic_inference()` - 16 edges
3. `switchView()` - 13 edges
4. `load_and_resample()` - 11 edges
5. `extract_all_features()` - 11 edges
6. `showToast()` - 10 edges
7. `load_model_checkpoint()` - 10 edges
8. `process_single_audio()` - 10 edges
9. `executeClassification()` - 9 edges
10. `drawFeatureSpectrogram()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Mel Spectrogram Acoustic Heatmap` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mel_spectrogram.png → Preproccesing/README.md
- `MFCC Feature Representation Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mfcc_features.png → Preproccesing/README.md
- `predict()` --indirect_call--> `run_acoustic_inference()`  [INFERRED]
  main.py → Preproccesing/acoustic_inference.py
- `Dual-Modal Verification Diagnostic Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Preproccesing/reports/feature_visualization.png → Preproccesing/README.md
- `_warm()` --calls--> `warm_up_engine()`  [EXTRACTED]
  main.py → Preproccesing/acoustic_inference.py

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

## Communities (55 total, 22 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (164): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+156 more)

### Community 1 - "data_augmentation.py"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.10
Nodes (33): buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks(), drawFeatureSpectrogram() (+25 more)

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
Cohesion: 0.14
Nodes (5): extract_mel_sequence(), explore_audio(), list_audio_files(), resample_file(), split_audio()

### Community 8 - "post"
Cohesion: 0.14
Nodes (12): add_history(), cloud_sync(), CloudSyncRequest, login(), LoginRequest, logout(), predict(), save_model() (+4 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.33
Nodes (5): API Endpoints, Architecture, Deployed Inference Engine, Key Features, SoundSense — Environmental Sound Classification Studio

### Community 10 - "loadAnalyticsData"
Cohesion: 0.67
Nodes (4): applyTheme(), drawClassDistributionChart(), drawModelPerformanceChart(), loadAnalyticsData()

### Community 11 - "numpy"
Cohesion: 0.14
Nodes (4): extract_mel_feature_vector(), cross_correlation_similarity(), generate_feature_visualization(), verify_feature_integrity()

### Community 12 - "acoustic_inference.py"
Cohesion: 0.20
Nodes (6): AcousticClassifierNet, _ast_probabilities(), compute_signal_metrics(), extract_acoustic_feature_vector(), get_acoustic_classifier(), _mlp_probabilities()

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.20
Nodes (9): key, value, services, dockerfilePath, env, envVars, name, plan (+1 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.29
Nodes (7): Mel Spectrogram Acoustic Heatmap, MFCC Feature Representation Plot, Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 17 - "Model Training & Progress Monitoring"
Cohesion: 0.40
Nodes (6): drawCurve(), finishTrainingRun(), pollTrainingProgress(), renderTrainingChart(), simulateClientTraining(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "train_models.py"
Cohesion: 0.13
Nodes (7): calibrate_temperature(), CNNGRUFast, compute_metrics(), extract_features_batched(), load_split_features(), ResNet18Transfer, train_and_export_all()

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "WAV Encoding & Audio Synthesis"
Cohesion: 0.33
Nodes (5): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory(), loadTaxonomyCategory(), renderTaxonomyGrid()

### Community 24 - "Vercel Edge Deployment Config"
Cohesion: 0.50
Nodes (3): builds, routes, version

### Community 25 - "Keyboard Shortcuts Modal"
Cohesion: 0.67
Nodes (3): closeShortcuts(), openShortcuts(), toggleShortcuts()

### Community 26 - "updateApiExplorer"
Cohesion: 0.67
Nodes (3): formatApiCode(), formatApiJson(), updateApiExplorer()

### Community 33 - "__init__.py"
Cohesion: 0.11
Nodes (7): detect_active_segment(), load_and_resample(), peak_normalize(), save_audio(), process_single_audio(), run_batch_preprocessing(), generate_5fold_splits()

### Community 37 - "__init__.py"
Cohesion: 0.19
Nodes (5): piecewise_aggregate_approximation(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 38 - "ESC50MultiModalDataset"
Cohesion: 0.18
Nodes (3): ESC50MultiModalDataset, get_dataloaders(), test_pytorch_dataloader()

### Community 39 - "run_acoustic_inference"
Cohesion: 0.20
Nodes (7): _build_silence_result(), get_engine_status(), load_audio_waveform(), load_model_checkpoint(), model_status(), run_acoustic_inference(), run_tests()

### Community 43 - "showToast"
Cohesion: 0.20
Nodes (16): animateCounter(), step(), cleanupMic(), displayClassificationResults(), easeOutQuad(), executeClassification(), finalizeMicRecording(), generateModelPredictions() (+8 more)

### Community 44 - "get"
Cohesion: 0.22
Nodes (9): auth_me(), get_history(), get_sample_for_category(), get_stats(), get_training_job(), get_uptime_seconds(), health(), list_checkpoints() (+1 more)

### Community 45 - "_load_ast_engine"
Cohesion: 0.29
Nodes (3): _find_local_ast_snapshot(), _load_ast_engine(), _normalise_label()

### Community 47 - "warm_up_engine"
Cohesion: 0.33
Nodes (3): lifespan(), _warm(), warm_up_engine()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 411 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_acoustic_inference()` connect `run_acoustic_inference` to `__init__.py`, `post`, `main.py`, `acoustic_inference.py`, `_load_ast_engine`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.011235955056179775 - nodes in this community are weakly interconnected._
- **Why does `extract_log_mel_spectrogram()` connect `__init__.py` to `numpy`, `acoustic_inference.py`, `soundfile`, `run_acoustic_inference`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.09659090909090909 - nodes in this community are weakly interconnected._
- **Why does `ESC50MultiModalDataset` connect `ESC50MultiModalDataset` to `__init__.py`, `__init__.py`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `Checkpoint Manifest & Metadata` be split into smaller, more focused modules?**
  _Cohesion score 0.07142857142857142 - nodes in this community are weakly interconnected._
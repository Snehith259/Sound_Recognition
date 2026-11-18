# Graph Report - Sound_Recognition  (2026-11-18)

## Corpus Check
- 51 files · ~56,994 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6017 file(s) not represented in the graph (top: .npy 5994, .csv 17, (none) 3)

## Summary
- 662 nodes · 1054 edges · 38 communities (25 shown, 13 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 34 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c720db30`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- verify_and_report.py
- Audio Processing & Spectrogram Synthesis
- Checkpoint Manifest & Metadata
- Node.js Server & Package Dependencies
- PyTorch Neural Architectures & Models
- UI Animation & Status Management
- acoustic_inference.py
- main.py
- SoundSense — Environmental Sound Classification Studio
- loadAnalyticsData
- numpy
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- batch_preprocessor.py
- Model Training & Progress Monitoring
- Interactive Particle Visualizer
- Canvas Rendering & Audio Level Meter
- SoundSense Architecture & System Specs
- Batch File Processing
- fx.js
- WAV Encoding & Audio Synthesis
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- __init__.py
- Graphics Animation Loop
- Constant-Q Transform Visualization
- STFT Time-Frequency Visualization
- Audio Types Comparison Plot
- Waveform Amplitude Plot
- Stratified 5-Fold Split Generator
- copilot-instructions.md
- json
- get_audio_duration

## God Nodes (most connected - your core abstractions)
1. `loadAudioFile()` - 17 edges
2. `switchView()` - 12 edges
3. `showToast()` - 11 edges
4. `load_and_resample()` - 11 edges
5. `extract_all_features()` - 11 edges
6. `process_single_audio()` - 10 edges
7. `drawFeatureSpectrogram()` - 9 edges
8. `peak_normalize()` - 9 edges
9. `generate_5fold_splits()` - 9 edges
10. `run_acoustic_inference()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Mel Spectrogram Acoustic Heatmap` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mel_spectrogram.png → Preproccesing/README.md
- `MFCC Feature Representation Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mfcc_features.png → Preproccesing/README.md
- `Dual-Modal Verification Diagnostic Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Preproccesing/reports/feature_visualization.png → Preproccesing/README.md
- `predict()` --calls--> `run_acoustic_inference()`  [EXTRACTED]
  main.py → Preproccesing/acoustic_inference.py
- `process_single_audio()` --calls--> `extract_all_features()`  [EXTRACTED]
  Preproccesing/batch_preprocessor.py → Preproccesing/feature_extractor.py

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

## Communities (38 total, 13 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (161): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+153 more)

### Community 1 - "verify_and_report.py"
Cohesion: 0.16
Nodes (3): ESC50MultiModalDataset, get_dataloaders(), test_pytorch_dataloader()

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.08
Nodes (45): animateCounter(), step(), buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), cleanupMic(), computeAcousticSpectrogram(), computeActiveSegment() (+37 more)

### Community 3 - "Checkpoint Manifest & Metadata"
Cohesion: 0.06
Nodes (32): accuracy, description, filename, repo_id, sha256, size_mb, hub, models (+24 more)

### Community 4 - "Node.js Server & Package Dependencies"
Cohesion: 0.07
Nodes (30): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+22 more)

### Community 5 - "PyTorch Neural Architectures & Models"
Cohesion: 0.09
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.13
Nodes (18): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), checkBackendStatus(), closeAccountMenu(), closeAuthModal(), escapeHtml(), fitAccountMenuPopover() (+10 more)

### Community 8 - "main.py"
Cohesion: 0.05
Nodes (28): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, get_history(), get_sample_for_category(), get_stats() (+20 more)

### Community 9 - "SoundSense — Environmental Sound Classification Studio"
Cohesion: 0.40
Nodes (4): API Endpoints, Architecture, Key Features, SoundSense — Environmental Sound Classification Studio

### Community 10 - "loadAnalyticsData"
Cohesion: 0.67
Nodes (4): applyTheme(), drawClassDistributionChart(), drawModelPerformanceChart(), loadAnalyticsData()

### Community 11 - "numpy"
Cohesion: 0.06
Nodes (14): extract_mel_feature_vector(), cross_correlation_similarity(), extract_mel_sequence(), explore_audio(), get_hubert_embedding(), load_wav_fallback(), split_audio(), add_gaussian_noise() (+6 more)

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.20
Nodes (9): key, value, services, dockerfilePath, env, envVars, name, plan (+1 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.29
Nodes (7): Mel Spectrogram Acoustic Heatmap, MFCC Feature Representation Plot, Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 16 - "batch_preprocessor.py"
Cohesion: 0.14
Nodes (6): detect_active_segment(), load_and_resample(), peak_normalize(), save_audio(), process_single_audio(), run_batch_preprocessing()

### Community 17 - "Model Training & Progress Monitoring"
Cohesion: 0.40
Nodes (6): drawCurve(), finishTrainingRun(), pollTrainingProgress(), renderTrainingChart(), simulateClientTraining(), updateTrainingUI()

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "Canvas Rendering & Audio Level Meter"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 22 - "fx.js"
Cohesion: 0.16
Nodes (13): addTask(), attach(), burst(), clamp(), countUp(), el(), ensure(), frame() (+5 more)

### Community 23 - "WAV Encoding & Audio Synthesis"
Cohesion: 0.50
Nodes (3): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory()

### Community 24 - "Vercel Edge Deployment Config"
Cohesion: 0.50
Nodes (3): builds, routes, version

### Community 25 - "Keyboard Shortcuts Modal"
Cohesion: 0.67
Nodes (3): closeShortcuts(), openShortcuts(), toggleShortcuts()

### Community 26 - "__init__.py"
Cohesion: 0.25
Nodes (5): piecewise_aggregate_approximation(), extract_all_features(), extract_log_mel_spectrogram(), extract_mfcc(), extract_threshold_free_recurrence_plot()

### Community 35 - "json"
Cohesion: 0.18
Nodes (3): generate_5fold_splits(), ensure_checkpoint_cache(), retrieve_cloud_weights()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 383 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_acoustic_inference()` connect `main.py` to `acoustic_inference.py`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.011363636363636364 - nodes in this community are weakly interconnected._
- **Why does `NodeParticle` connect `Interactive Particle Visualizer` to `Front-End UI & State Controller`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.07878787878787878 - nodes in this community are weakly interconnected._
- **Why does `ESC50MultiModalDataset` connect `verify_and_report.py` to `__init__.py`, `dataset_loader.py`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `Checkpoint Manifest & Metadata` be split into smaller, more focused modules?**
  _Cohesion score 0.06060606060606061 - nodes in this community are weakly interconnected._
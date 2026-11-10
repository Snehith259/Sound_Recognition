# Graph Report - Sound_Recognition  (2026-11-04)

## Corpus Check
- Corpus is ~38,947 words - fits in a single context window. You may not need a graph.

## Summary
- 551 nodes · 882 edges · 33 communities (21 shown, 12 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Front-End UI & State Controller
- Dataset Ingestion & Batch Loading
- Audio Processing & Spectrogram Synthesis
- Checkpoint Manifest & Metadata
- Node.js Server & Package Dependencies
- PyTorch Neural Architectures & Models
- UI Animation & Status Management
- Feature Vector & Similarity Metrics
- Audio Distance & Sequence Comparison
- Audio Properties & Resampling
- Audio Data Augmentation
- Sample Rates & HuBERT Embeddings
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- Audio Duration & Sample Validation
- Model Training & Progress Monitoring
- Interactive Particle Visualizer
- Canvas Rendering & Audio Level Meter
- SoundSense Architecture & System Specs
- Batch File Processing
- Spectrogram & Recurrence Plot Cache
- WAV Encoding & Audio Synthesis
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
- Synthetic Model Inference
- Graphics Animation Loop
- Constant-Q Transform Visualization
- STFT Time-Frequency Visualization
- Audio Types Comparison Plot
- Waveform Amplitude Plot
- Stratified 5-Fold Split Generator

## God Nodes (most connected - your core abstractions)
1. `loadAudioFile()` - 17 edges
2. `switchView()` - 12 edges
3. `load_and_resample()` - 11 edges
4. `extract_all_features()` - 11 edges
5. `showToast()` - 10 edges
6. `process_single_audio()` - 10 edges
7. `drawFeatureSpectrogram()` - 9 edges
8. `peak_normalize()` - 9 edges
9. `generate_5fold_splits()` - 9 edges
10. `toggleMicRecording()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `Mel Spectrogram Acoustic Heatmap` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mel_spectrogram.png → Preproccesing/README.md
- `MFCC Feature Representation Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mfcc_features.png → Preproccesing/README.md
- `Dual-Modal Verification Diagnostic Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Preproccesing/reports/feature_visualization.png → Preproccesing/README.md
- `process_single_audio()` --calls--> `load_and_resample()`  [EXTRACTED]
  Preproccesing/batch_preprocessor.py → Preproccesing/audio_processor.py
- `process_single_audio()` --calls--> `peak_normalize()`  [EXTRACTED]
  Preproccesing/batch_preprocessor.py → Preproccesing/audio_processor.py

## Import Cycles
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_splits.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 2-file cycle: `Preproccesing/__init__.py -> Preproccesing/feature_extractor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/feature_extractor.py -> Preproccesing/__init__.py`
- 3-file cycle: `Preproccesing/__init__.py -> Preproccesing/feature_extractor.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`
- 4-file cycle: `Preproccesing/__init__.py -> Preproccesing/dataset_loader.py -> Preproccesing/feature_extractor.py -> Preproccesing/audio_processor.py -> Preproccesing/__init__.py`

## Hyperedges (group relationships)
- **Dual-Modal Acoustic Processing Flow** — preproccesing_readme_dmff_architecture, preproccesing_readme_recurrence_plot, preproccesing_readme_paa_dimension, preproccesing_readme_esc50_dataset [EXTRACTED 0.95]

## Communities (33 total, 12 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (152): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+144 more)

### Community 1 - "Dataset Ingestion & Batch Loading"
Cohesion: 0.06
Nodes (17): detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio(), process_single_audio(), run_batch_preprocessing(), ESC50MultiModalDataset (+9 more)

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.09
Nodes (40): animateCounter(), step(), buildActiveSegmentCache(), cleanupMic(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks() (+32 more)

### Community 3 - "Checkpoint Manifest & Metadata"
Cohesion: 0.06
Nodes (32): accuracy, description, filename, repo_id, sha256, size_mb, hub, models (+24 more)

### Community 4 - "Node.js Server & Package Dependencies"
Cohesion: 0.08
Nodes (26): dependencies, cors@^2.8.5, express@^4.21.0, multer@^1.4.5-lts.1, description, main, name, scripts (+18 more)

### Community 5 - "PyTorch Neural Architectures & Models"
Cohesion: 0.09
Nodes (6): BasicBlock, CNNGRU, get_model(), ResNet18_ESC50, SResNet, VGG16_ESC50

### Community 6 - "UI Animation & Status Management"
Cohesion: 0.11
Nodes (20): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), checkBackendStatus(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+12 more)

### Community 7 - "Feature Vector & Similarity Metrics"
Cohesion: 0.16
Nodes (5): extract_mel_feature_vector(), cross_correlation_similarity(), load_wav_fallback(), generate_feature_visualization(), verify_feature_integrity()

### Community 9 - "Audio Properties & Resampling"
Cohesion: 0.18
Nodes (4): explore_audio(), list_audio_files(), resample_file(), split_audio()

### Community 10 - "Audio Data Augmentation"
Cohesion: 0.24
Nodes (5): add_gaussian_noise(), cyclic_time_shift(), generate_augmented_variants(), pitch_shift(), time_stretch()

### Community 13 - "Render Cloud Deployment Config"
Cohesion: 0.18
Nodes (10): key, value, services, buildCommand, env, envVars, name, plan (+2 more)

### Community 14 - "Docker Compose Service Orchestration"
Cohesion: 0.20
Nodes (9): services, soundsense-app, build, container_name, environment, ports, restart, volumes (+1 more)

### Community 15 - "DMFF & Acoustic Feature Verification"
Cohesion: 0.29
Nodes (7): Mel Spectrogram Acoustic Heatmap, MFCC Feature Representation Plot, Dual-Modal Feature Fusion (DMFF), ESC-50 Acoustic Dataset, Piecewise Aggregate Approximation (PAA), Threshold-Free Recurrence Plot (RP), Dual-Modal Verification Diagnostic Plot

### Community 17 - "Model Training & Progress Monitoring"
Cohesion: 0.40
Nodes (6): drawCurve(), finishTrainingRun(), pollTrainingProgress(), renderTrainingChart(), simulateClientTraining(), updateTrainingUI()

### Community 19 - "Canvas Rendering & Audio Level Meter"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 22 - "Spectrogram & Recurrence Plot Cache"
Cohesion: 0.50
Nodes (4): buildRecurrencePlotCache(), buildSpectrogramCache(), drawSpectrogramAxes(), getSpectrogramColor()

### Community 23 - "WAV Encoding & Audio Synthesis"
Cohesion: 0.50
Nodes (3): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory()

### Community 24 - "Vercel Edge Deployment Config"
Cohesion: 0.50
Nodes (3): builds, routes, version

### Community 25 - "Keyboard Shortcuts Modal"
Cohesion: 0.67
Nodes (3): closeShortcuts(), openShortcuts(), toggleShortcuts()

### Community 26 - "Synthetic Model Inference"
Cohesion: 0.67
Nodes (3): generateModelPredictions(), shuffleArray(), simpleHash()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `5-Fold Stratified Split Generator`, `ESC-50 Acoustic Dataset`, `Classifier Studio Workspace`, `Developer Training & Checkpoint Portal` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 326 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `NodeParticle` connect `Interactive Particle Visualizer` to `Front-End UI & State Controller`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `5-Fold Stratified Split Generator`, `ESC-50 Acoustic Dataset` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.012422360248447204 - nodes in this community are weakly interconnected._
- **Why does `load_and_resample()` connect `Dataset Ingestion & Batch Loading` to `Spectral Transformations & CQT`, `Feature Vector & Similarity Metrics`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `Dataset Ingestion & Batch Loading` be split into smaller, more focused modules?**
  _Cohesion score 0.05543859649122807 - nodes in this community are weakly interconnected._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.09230769230769231 - nodes in this community are weakly interconnected._
- **Should `Checkpoint Manifest & Metadata` be split into smaller, more focused modules?**
  _Cohesion score 0.06060606060606061 - nodes in this community are weakly interconnected._
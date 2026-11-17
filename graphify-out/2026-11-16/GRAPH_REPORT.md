# Graph Report - Sound_Recognition  (2026-11-12)

## Corpus Check
- 49 files · ~48,162 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6016 file(s) not represented in the graph (top: .npy 5994, .csv 17, (none) 2)

## Summary
- 620 nodes · 996 edges · 30 communities (21 shown, 9 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 21 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `df8832e5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Front-End UI & State Controller
- numpy
- Audio Processing & Spectrogram Synthesis
- Checkpoint Manifest & Metadata
- Node.js Server & Package Dependencies
- PyTorch Neural Architectures & Models
- UI Animation & Status Management
- showToast
- main.py
- librosa
- run_acoustic_inference
- download_checkpoints.py
- Render Cloud Deployment Config
- Docker Compose Service Orchestration
- DMFF & Acoustic Feature Verification
- Model Training & Progress Monitoring
- Interactive Particle Visualizer
- Canvas Rendering & Audio Level Meter
- SoundSense Architecture & System Specs
- Batch File Processing
- WAV Encoding & Audio Synthesis
- Vercel Edge Deployment Config
- Keyboard Shortcuts Modal
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
8. `run_acoustic_inference()` - 9 edges
9. `peak_normalize()` - 9 edges
10. `generate_5fold_splits()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Mel Spectrogram Acoustic Heatmap` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mel_spectrogram.png → Preproccesing/README.md
- `MFCC Feature Representation Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Audio_Basics_HandsOn/mfcc_features.png → Preproccesing/README.md
- `Dual-Modal Verification Diagnostic Plot` --conceptually_related_to--> `Dual-Modal Feature Fusion (DMFF)`  [INFERRED]
  Preproccesing/reports/feature_visualization.png → Preproccesing/README.md
- `predict()` --calls--> `run_acoustic_inference()`  [EXTRACTED]
  main.py → Preproccesing/acoustic_inference.py
- `process_single_audio()` --calls--> `detect_active_segment()`  [EXTRACTED]
  Preproccesing/batch_preprocessor.py → Preproccesing/audio_processor.py

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

## Communities (30 total, 9 thin omitted)

### Community 0 - "Front-End UI & State Controller"
Cohesion: 0.01
Nodes (153): accountAvatar, accountBtn, accountMenuPopover, ahCkptsEl, ahMemoryEl, ahNodeEl, ahPredsEl, ahStatusEl (+145 more)

### Community 1 - "numpy"
Cohesion: 0.06
Nodes (16): detect_active_segment(), load_and_resample(), peak_normalize(), piecewise_aggregate_approximation(), save_audio(), process_single_audio(), run_batch_preprocessing(), ESC50MultiModalDataset (+8 more)

### Community 2 - "Audio Processing & Spectrogram Synthesis"
Cohesion: 0.12
Nodes (28): buildActiveSegmentCache(), buildRecurrencePlotCache(), buildSpectrogramCache(), computeAcousticSpectrogram(), computeActiveSegment(), computeThresholdFreeRecurrencePlot(), computeWaveformPeaks(), drawFeatureSpectrogram() (+20 more)

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
Cohesion: 0.11
Nodes (20): animateArchMetrics(), animateBenchmarkBars(), animateMetricValue(), checkBackendStatus(), closeAccountMenu(), closeAuthModal(), drawClassDistributionChart(), drawModelPerformanceChart() (+12 more)

### Community 7 - "showToast"
Cohesion: 0.22
Nodes (15): animateCounter(), step(), cleanupMic(), displayClassificationResults(), easeOutQuad(), executeClassification(), finalizeMicRecording(), generateModelPredictions() (+7 more)

### Community 8 - "main.py"
Cohesion: 0.07
Nodes (23): add_history(), auth_me(), clear_history(), cloud_sync(), CloudSyncRequest, get_history(), get_sample_for_category(), get_stats() (+15 more)

### Community 9 - "librosa"
Cohesion: 0.05
Nodes (17): extract_mel_feature_vector(), cross_correlation_similarity(), extract_mel_sequence(), explore_audio(), get_hubert_embedding(), load_wav_fallback(), check_audio_durations(), get_audio_duration() (+9 more)

### Community 10 - "run_acoustic_inference"
Cohesion: 0.19
Nodes (5): AcousticClassifierNet, extract_acoustic_feature_vector(), get_acoustic_classifier(), load_audio_waveform(), run_acoustic_inference()

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

### Community 18 - "Interactive Particle Visualizer"
Cohesion: 0.38
Nodes (3): debouncedResize(), NodeParticle, onResize()

### Community 19 - "Canvas Rendering & Audio Level Meter"
Cohesion: 0.40
Nodes (5): drawFrame(), loop(), readLevel(), resize(), roundRectPath()

### Community 20 - "SoundSense Architecture & System Specs"
Cohesion: 0.40
Nodes (5): Classifier Studio Workspace, Developer Training & Checkpoint Portal, Comparative Model Benchmark Suite, Express.js REST API Layer, SoundSense Audio Recognition System

### Community 23 - "WAV Encoding & Audio Synthesis"
Cohesion: 0.33
Nodes (5): encodeWAV(), generateSyntheticAudioBlob(), generateSyntheticAudioForCategory(), loadTaxonomyCategory(), renderTaxonomyGrid()

### Community 24 - "Vercel Edge Deployment Config"
Cohesion: 0.50
Nodes (3): builds, routes, version

### Community 25 - "Keyboard Shortcuts Modal"
Cohesion: 0.67
Nodes (3): closeShortcuts(), openShortcuts(), toggleShortcuts()

## Knowledge Gaps
- **14 isolated node(s):** `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap`, `MFCC Feature Representation Plot`, `Dual-Modal Verification Diagnostic Plot` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 358 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_acoustic_inference()` connect `run_acoustic_inference` to `main.py`, `librosa`?**
  _High betweenness centrality (0.006) - this node is a cross-community bridge._
- **What connects `Piecewise Aggregate Approximation (PAA)`, `ESC-50 Acoustic Dataset`, `Mel Spectrogram Acoustic Heatmap` to the rest of the system?**
  _14 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Front-End UI & State Controller` be split into smaller, more focused modules?**
  _Cohesion score 0.012195121951219513 - nodes in this community are weakly interconnected._
- **Why does `NodeParticle` connect `Interactive Particle Visualizer` to `Front-End UI & State Controller`?**
  _High betweenness centrality (0.005) - this node is a cross-community bridge._
- **Should `numpy` be split into smaller, more focused modules?**
  _Cohesion score 0.0579476861167002 - nodes in this community are weakly interconnected._
- **Should `Audio Processing & Spectrogram Synthesis` be split into smaller, more focused modules?**
  _Cohesion score 0.12169312169312169 - nodes in this community are weakly interconnected._
- **Should `Checkpoint Manifest & Metadata` be split into smaller, more focused modules?**
  _Cohesion score 0.06060606060606061 - nodes in this community are weakly interconnected._
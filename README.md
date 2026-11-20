---
title: SoundSense ESC-50 Audio Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SoundSense — Environmental Sound Classification Studio

**SoundSense** is a real-time environmental sound recognition and audio classification platform trained on the **ESC-50** dataset (50 distinct sound categories across Animals, Natural Soundscapes, Human Sounds, Interior Audio, and Urban Environments).

## Key Features

- **Pure Neural Acoustic Inference**: Zero filename guesswork. Analyzes real audio waveform, 128-band Log-Mel Spectrograms, and 40-coefficient MFCC spectral dynamics.
- **Deep Acoustic Models**:
  - **SResNet**: Compact Squeezed Residual Network for high efficiency.
  - **ResNet-18**: 18-layer Deep Residual Network on Log-Mel representations.
  - **VGG-16**: High-capacity Deep Convolutional Network.
  - **CNN-GRU**: Hybrid Convolutional Recurrent Network for temporal sound dynamics.
- **Out-of-Distribution (OOD) Sound Detection**: Identifies silence, non-environmental sounds, and noise beyond the standard 50 categories.
- **Interactive Studio HUD**: Real-time oscilloscope, frequency spectrum visualizer, 3D interactive controls, and confidence distribution charts.
- **Full REST API**: Endpoints for audio classification (`/api/predict`), sample playback (`/api/sample/{id}`), system metrics (`/api/stats`), and health checks (`/api/health`).

## API Endpoints

- `POST /api/predict` — Upload an audio file (`multipart/form-data`) and select a model (`resnet18`, `vgg16`, `sresnet`, `cnngru`).
- `GET /api/sample/{category_id}` — Stream sample audio for any of the 50 ESC-50 categories (0–49).
- `GET /api/stats` — Real-time telemetry, model benchmarks, and hardware statistics.
- `GET /api/health` — Service health check.

## Architecture

- **Backend**: FastAPI (Python 3.11) with Uvicorn, PyTorch, Librosa, and SoundFile.
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas Web Audio API, Modern CSS3 Studio HUD.
- **Container**: Docker containerized, UID 1000 compliant, optimized for Hugging Face Spaces and cloud container runtimes.

## Deployed Inference Engine

`POST /api/predict` loads the architecture selected in the `model` form field and runs its matching
checkpoint from `checkpoints/`. The ensemble option averages the available architecture probabilities.
Inputs are decoded as mono audio, resampled to 22.05 kHz, aligned to five seconds, and peak-normalized
to match the training pipeline. The response includes the actual checkpoint engine used.

The bundled validation report records top-1 accuracy between 35.68% (SResNet) and 64.07% (CNN-GRU)
on its ESC-50 split. Softmax confidence is a model score, not a guarantee that a prediction is correct.
If a selected checkpoint cannot be loaded, the API returns an inference error instead of silently
substituting another model or generating a heuristic prediction. Set `SOUNDSENSE_TORCH_THREADS` to
override the default single-thread CPU inference setting.

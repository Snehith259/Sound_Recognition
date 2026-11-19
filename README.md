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

`POST /api/predict` runs a real pretrained **AST (Audio Spectrogram Transformer) fine-tuned on ESC-50**
(`Adam-ousse/ast-esc50-finetuned-fold1`, 93.5% top-1 on the ESC-50 fold-1 test split).

- Weights are downloaded once from the Hugging Face Hub and cached in `checkpoints/.cache/huggingface`.
- The model is warmed up in a background thread at server startup, so the first upload is fast.
- The `model` form field (`resnet18`, `vgg16`, `sresnet`, `cnngru`) is kept for UI/API compatibility;
  every selection is served by the deployed AST engine, and each response reports the real `engine`.
- If the weights cannot be loaded (e.g. offline), the backend falls back to the bundled
  `checkpoints/acoustic_classifier.pt` MLP so predictions keep working.
- Override the checkpoint with `SOUNDSENSE_HF_MODEL=<hf-repo-id>` and the OOD confidence
  threshold with `SOUNDSENSE_OOD_THRESHOLD=<0..1>`.

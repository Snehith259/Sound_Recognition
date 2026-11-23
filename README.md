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

**SoundSense** is an environmental sound recognition demo built around the 50 classes in **ESC-50**. This checkout does not contain the four neural-network checkpoints used by the selectable ResNet-18, VGG-16, SResNet, and CNN-GRU options. Until trained checkpoints are supplied, those prediction requests return an explicit unavailable-model error; the app does not substitute the legacy MLP or invent a prediction.

## Key Features

- **Acoustic Inference**: Uses decoded audio only. A requested neural architecture requires its matching checkpoint.
- **Deep Acoustic Models**:
  - **SResNet**: Compact convolutional classifier on MFCC features (the current implementation has no residual blocks).
  - **ResNet-18**: 18-layer Deep Residual Network on Log-Mel representations.
  - **VGG-16**: High-capacity Deep Convolutional Network.
  - **CNN-GRU**: Hybrid Convolutional Recurrent Network for temporal sound dynamics.
- **Rule-Based Sound Gates**: Detects digital silence, very quiet audio, and stable single-frequency tones. General OOD detection has not been evaluated.
- **Interactive Studio HUD**: Real-time oscilloscope, frequency spectrum visualizer, 3D interactive controls, and confidence distribution charts.
- **Full REST API**: Endpoints for audio classification (`/api/predict`), sample playback (`/api/sample/{id}`), system metrics (`/api/stats`), and health checks (`/api/health`).

## API Endpoints

- `POST /api/predict` — Upload an audio file (`multipart/form-data`) and select a model (`resnet18`, `vgg16`, `sresnet`, `cnngru`).
- `GET /api/sample/{category_id}` — Stream a sample audio file when ESC-50 audio is installed locally.
- `GET /api/stats` — Runtime telemetry, model availability, and evaluation metrics when present.
- `GET /api/health` — Service health check.

## Architecture

- **Backend**: FastAPI (Python 3.11) with Uvicorn, PyTorch, Librosa, and SoundFile.
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas Web Audio API, Modern CSS3 Studio HUD.
- **Container**: Docker containerized, UID 1000 compliant, optimized for Hugging Face Spaces and cloud container runtimes.

## Deployed Inference Engine

`POST /api/predict` loads only the architecture selected in the `model` field. The ensemble option
requires all four architecture checkpoints. This checkout currently has no such checkpoints, so its
accuracy, R², and external prediction quality are **not evaluated**. The older `acoustic_classifier.pt`
file is retained for a standalone diagnostic and is never used by the app's model choices.

Run `python -m eval.build_external_set` to prepare a grouped, source-separated external set,
`python -m eval.eval_legacy_mlp_external` to inspect the legacy diagnostic model, and the
`eval/eval_external.py` / `eval/eval_esc50_cv.py` harnesses after trained checkpoints are available.
The external manifest keeps a final holdout separate from the dev set. See [eval/REPORT.md](eval/REPORT.md)
for measured results and current checkpoint status. Softmax output is a model
score, not a guarantee of correctness or a measured probability of correctness. The current rule
gates do not provide a general OOD detector, and they report no classifier confidence. Set
`SOUNDSENSE_TORCH_THREADS` to override the default single-thread CPU inference setting.

## Training and verification

The repository includes precomputed ESC-50 features and official five-fold split manifests. For a full run, use a Kaggle GPU session, turn Internet on so torchvision can fetch the official ImageNet weights for ResNet-18 and VGG-16, then run:

```text
python -m eval.train_cv --model all --device cuda --include-final --epochs 40
python -m eval.eval_esc50_cv_features --model all --device cuda
python -m eval.eval_external --split dev --models resnet18 vgg16 sresnet cnngru ensemble --formats wav mp3 webm
python -m eval.report
```

The trainer writes one checkpoint per held-out fold under `checkpoints/cv/` and a final deployment checkpoint for each model under `checkpoints/`. It uses validation folds for early stopping and temperature scaling, plus feature-space SpecAugment; it does not train on the external audio. The external harness runs the production inference function on the downloaded clips and checks audio decoding across formats. Only after the dev results and thresholds are reviewed should the held-out set be run once with `--split final`.

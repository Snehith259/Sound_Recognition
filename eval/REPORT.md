# SoundSense evaluation report

Generated: 2026-11-21T15:50:59.839438+00:00

Scores in this table are included only when produced by the evaluation harness.

| Model | Checkpoint | ESC-50 CV top-1 | External top-1 | ECE |
|---|---|---:|---:|---:|
| ResNet-18 | missing | — | — | — |
| VGG-16 | missing | — | — | — |
| SResNet | missing | — | — | — |
| CNN-GRU | missing | — | — | — |
| Ensemble Fusion | missing | — | — | — |

## Status

- **ResNet-18:** Checkpoint file is absent.
- **VGG-16:** Checkpoint file is absent.
- **SResNet:** Checkpoint file is absent.
- **CNN-GRU:** Checkpoint file is absent.
- **Ensemble Fusion:** Requires all four selected architecture checkpoints.
- External audio set: 69 dev clips; 105 final holdout clips (not evaluated).

## Local diagnostics

The only bundled checkpoint is a legacy MLP and is not used by the UI. Its external dev results are diagnostic, not evidence for any selectable architecture:

- External allowed-label top-1: **35.7%** (n=56); top-5: **87.5%**.
- BirdNET ambient windows: **50.0%** allowed-label top-1 (n=24).
- Unambiguous-label probability R²: **-0.055**; ECE: **0.377** (n=23).
- Max-softmax OOD AUROC: **0.346**; OOD rejection at the legacy 0.30 threshold: **7.7%**.
- Silence/tone gates on external dev: 69 clips; 0.0% false rejections among 56 in-scope clips at -58.0 dBFS.
- Synthetic OOD cases: 4/10 received an explicit silence/tone gate result; 6 require a classifier checkpoint, which is missing.

R² is reported only as one-hot-target versus predicted-probability R² (`r2Probability`). It is not a replacement for classification accuracy.

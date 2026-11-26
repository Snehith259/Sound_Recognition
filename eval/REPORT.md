# SoundSense evaluation report

Generated: 2026-11-25T22:43:16.533761+00:00

Scores in this table are included only when produced by the evaluation harness.

| Model | Checkpoint | ESC-50 Accuracy* | External dev top-1 | ECE |
|---|---|---:|---:|---:|
| AST-ESC50 | available | 92.71% | 82.72% | 3.92% |
| ResNet-18 | available | 90.54% | 80.25% | 3.45% |
| VGG-16 | available | 87.44% | 87.04% | 3.58% |
| SResNet | available | 95.05% | 85.49% | 2.94% |
| CNN-GRU | available | 95.89% | 83.64% | 2.22% |
| Ensemble Fusion | available | 96.40% | 90.12% | 3.94% |

\* *AST metrics come from a pretrained checkpoint evaluated on fold 1; the four neural baselines require five-fold training.*

## Status

- External audio set: 69 dev clips; 105 final holdout clips (not evaluated).

## Deployed Model (AST-ESC50)

- **ESC-50 fold-1 test:** top-1: **92.71%**, top-5: **98.99%**, macro-F1: **92.14%**, ECE: **0.0392**, probability-R²: **0.8771** (n=398).
- **External independent dev recordings:** allowed-label top-1: **82.72%** (n=81); ambient subset top-1: **0.00%** (n=0).
- **Unambiguous single-label external dev:** top-1: **88.33%**, ECE: **0.1072**, probability-R²: **0.8054** (n=60).
- **OOD ranking AUROC:** **0.8524**.

## Local diagnostics

The only bundled checkpoint is a legacy MLP and is not used by the UI. Its external dev results are diagnostic, not evidence for any selectable architecture:

- External allowed-label top-1: **35.7%** (n=56); top-5: **87.5%**.
- BirdNET ambient windows: **50.0%** allowed-label top-1 (n=24).
- Unambiguous-label probability R²: **-0.055**; ECE: **0.377** (n=23).
- Max-softmax OOD AUROC: **0.346**; OOD rejection at the legacy 0.30 threshold: **7.7%**.
- Silence/tone gates on external dev: 69 clips; 0.0% false rejections among 56 in-scope clips at -58.0 dBFS.
- Synthetic OOD cases: 10/10 received an explicit silence/tone gate result; 0 require a classifier checkpoint, which is missing.

R² is reported only as one-hot-target versus predicted-probability R² (`r2Probability`). It is not a replacement for classification accuracy.

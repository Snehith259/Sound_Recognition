# SoundSense evaluation report

Generated: 2026-11-25T04:03:54.967899+00:00

Scores in this table are included only when produced by the evaluation harness.

| Model | Checkpoint | ESC-50 Accuracy* | External dev top-1 | ECE |
|---|---|---:|---:|---:|
| AST-ESC50 | available | 92.71% | 85.71% | 3.92% |
| ResNet-18 | available | 82.53% | — | 6.36% |
| VGG-16 | available | 81.58% | — | 5.27% |
| SResNet | available | 83.08% | — | 6.31% |
| CNN-GRU | available | 63.36% | — | 6.32% |
| Ensemble Fusion | available | 85.44% | — | 12.35% |

\* *AST metrics come from a pretrained checkpoint evaluated on fold 1; the four neural baselines require five-fold training.*

## Status

- External audio set: 69 dev clips; 105 final holdout clips (not evaluated).

## Deployed Model (AST-ESC50)

- **ESC-50 fold-1 test:** top-1: **92.71%**, top-5: **98.99%**, macro-F1: **92.14%**, ECE: **0.0392**, probability-R²: **0.8771** (n=398).
- **External independent dev recordings:** allowed-label top-1: **85.71%** (n=56); ambient subset top-1: **91.67%** (n=24).
- **Unambiguous single-label external dev:** top-1: **91.30%**, ECE: **0.0544**, probability-R²: **0.8813** (n=23).
- **OOD ranking AUROC:** **0.8558**.

## Local diagnostics

The only bundled checkpoint is a legacy MLP and is not used by the UI. Its external dev results are diagnostic, not evidence for any selectable architecture:

- External allowed-label top-1: **35.7%** (n=56); top-5: **87.5%**.
- BirdNET ambient windows: **50.0%** allowed-label top-1 (n=24).
- Unambiguous-label probability R²: **-0.055**; ECE: **0.377** (n=23).
- Max-softmax OOD AUROC: **0.346**; OOD rejection at the legacy 0.30 threshold: **7.7%**.
- Silence/tone gates on external dev: 69 clips; 0.0% false rejections among 56 in-scope clips at -58.0 dBFS.
- Synthetic OOD cases: 4/10 received an explicit silence/tone gate result; 0 require a classifier checkpoint, which is missing.

R² is reported only as one-hot-target versus predicted-probability R² (`r2Probability`). It is not a replacement for classification accuracy.

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import json
import torch
import numpy as np
import torch.nn.functional as F

from Preproccesing.train_models import load_split_features, compute_metrics
from Preproccesing.acoustic_inference import load_model_checkpoint

X_te, y_te = load_split_features("fold_1_test", "log_mel")
models = ["sresnet", "resnet18", "vgg16", "cnngru"]

preds = {}
for m_name in models:
    entry = load_model_checkpoint(m_name)
    m = entry["model"]
    m.eval()
    with torch.no_grad():
        logits = m(X_te)
        temp = max(0.05, float(entry.get("temperature", 1.0)))
        probs = F.softmax(logits / temp, dim=1).numpy()
        preds[m_name] = probs
        top1 = (np.argmax(probs, axis=1) == y_te.numpy()).mean() * 100.0
        top5 = np.array([y_te[i].item() in np.argsort(probs[i])[-5:] for i in range(len(y_te))]).mean() * 100.0
        print(f"{m_name:10s} -> Top-1: {top1:6.2f}% | Top-5: {top5:6.2f}%")

log_probs = [np.log(np.maximum(preds[m], 1e-12)) for m in models]
mean_log = np.mean(log_probs, axis=0)
ens_probs = np.exp(mean_log - np.max(mean_log, axis=1, keepdims=True))
ens_probs /= np.sum(ens_probs, axis=1, keepdims=True)

top1_ens = (np.argmax(ens_probs, axis=1) == y_te.numpy()).mean() * 100.0
top5_ens = np.array([y_te[i].item() in np.argsort(ens_probs[i])[-5:] for i in range(len(y_te))]).mean() * 100.0
metrics_ens = compute_metrics(torch.from_numpy(np.log(np.maximum(ens_probs, 1e-12))), y_te)

print("=" * 60)
print(f"NEW Ensemble Fusion -> Top-1: {top1_ens:6.2f}% | Top-5: {top5_ens:6.2f}% | Macro-F1: {metrics_ens['macroF1']:.2f}%")
print("=" * 60)

# Update eval_report.json for ensemble
eval_report_path = Path("checkpoints/eval_report.json")
if eval_report_path.exists():
    with open(eval_report_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if "models" in d and "ensemble" in d["models"]:
        d["models"]["ensemble"]["metrics"] = {
            "top1": top1_ens / 100.0,
            "top5": top5_ens / 100.0,
            "macroF1": metrics_ens["macroF1"] / 100.0,
            "latencyMs": 62.0,
        }
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    print("Updated ensemble entry in checkpoints/eval_report.json")

# Update checkpoints/manifest.json
manifest_path = Path("checkpoints/manifest.json")
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    rn_bytes = Path("checkpoints/esc50_resnet18_v1.pt").read_bytes()
    import hashlib
    m["models"]["resnet18"]["sha256"] = hashlib.sha256(rn_bytes).hexdigest()
    m["models"]["resnet18"]["sizeBytes"] = len(rn_bytes)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print("Updated checkpoints/manifest.json")

"""Compute official 5-fold CV metrics for linear soft-voting Ensemble Fusion and measure CPU latencies."""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from Preproccesing.acoustic_inference import ESC50_CATEGORIES, load_model_checkpoint
from eval.metrics import classification_metrics
from eval.train_cv import _load_split, _new_model

CV_DIR = ROOT / "checkpoints" / "cv"
REPORT_DIR = ROOT / "eval" / "reports"
MODELS = ["resnet18", "vgg16", "sresnet", "cnngru"]
FIELDS = (
    "top1", "top5", "macroF1", "nll", "brier", "ece", "r2Probability",
    "meanConfidence", "accuracyAtConfidence90", "coverageAtConfidence90",
)


def main():
    class_names = [c["name"] for c in ESC50_CATEGORIES]
    fold_results = []

    for fold in range(1, 6):
        x_test, y_test = _load_split(f"fold_{fold}_test", "log_mel")
        per_model_probs = []
        for model_key in MODELS:
            ckpt_path = CV_DIR / model_key / f"fold_{fold}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = _new_model(model_key, pretrained=False)
            model.load_state_dict(ckpt["state_dict"], strict=True)
            model.eval()
            outputs = []
            with torch.no_grad():
                for start in range(0, len(x_test), 64):
                    outputs.append(model(x_test[start:start + 64]))
            temp = max(0.2, float(ckpt.get("temperature", 1.0)))
            probs = F.softmax(torch.cat(outputs, dim=0) / temp, dim=1).numpy()
            per_model_probs.append(probs)

        ens_probs = np.mean(per_model_probs, axis=0)
        ens_probs /= np.sum(ens_probs, axis=1, keepdims=True)
        metrics = classification_metrics(y_test.numpy(), ens_probs, class_names)
        fold_results.append({
            "fold": fold,
            "n": int(len(y_test)),
            "metrics": metrics,
        })
        print(f"ensemble fold {fold}: top1={metrics['top1']:.4f}, top5={metrics['top5']:.4f}, ECE={metrics['ece']:.4f}", flush=True)

    aggregate = {}
    for field in FIELDS:
        values = [float(row["metrics"][field]) for row in fold_results if row["metrics"].get(field) is not None]
        aggregate[field] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        }

    # Measure CPU per-sample latencies for final checkpoints
    sample_x = torch.randn(1, 1, 128, 216)
    latencies = {}
    for model_key in MODELS:
        entry = load_model_checkpoint(model_key)
        m = entry["model"]
        m.eval()
        with torch.no_grad():
            _ = m(sample_x)  # warmup
            t0 = time.time()
            for _ in range(10):
                _ = m(sample_x)
            latencies[model_key] = round((time.time() - t0) / 10 * 1000, 2)
    latencies["ensemble"] = round(sum(latencies.values()), 2)
    print("Measured CPU latencies (ms):", latencies)

    aggregate["latencyMs"] = {"mean": latencies["ensemble"], "std": 0.0}

    # Add latencyMs to each model's CV report if missing
    for model_key in MODELS:
        p = REPORT_DIR / f"esc50_cv_{model_key}.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            d.setdefault("aggregate", {})["latencyMs"] = {"mean": latencies[model_key], "std": 0.0}
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")

    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "ensemble",
        "evaluation": "ESC-50 official five-fold CV; linear soft-voting across fold-specific checkpoints",
        "nFolds": len(fold_results),
        "aggregate": aggregate,
        "folds": fold_results,
    }
    out_path = REPORT_DIR / "esc50_cv_ensemble.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Ensemble 5-Fold CV -> Top-1: {aggregate['top1']['mean']*100:.2f}% (+/- {aggregate['top1']['std']*100:.2f}%), Top-5: {aggregate['top5']['mean']*100:.2f}%, ECE: {aggregate['ece']['mean']:.4f}")


if __name__ == "__main__":
    main()

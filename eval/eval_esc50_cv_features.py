"""Evaluate held-out ESC-50 folds against the repository's precomputed features."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from Preproccesing.acoustic_inference import ESC50_CATEGORIES
from eval.metrics import classification_metrics
from eval.train_cv import MODEL_CONFIG, _load_split, _new_model

CV_DIR = ROOT / "checkpoints" / "cv"
REPORT_DIR = ROOT / "eval" / "reports"
FIELDS = (
    "top1", "top5", "macroF1", "nll", "brier", "ece", "r2Probability",
    "meanConfidence", "accuracyAtConfidence90", "coverageAtConfidence90",
)


def evaluate(model_key: str, batch_size: int = 64, device_name: str = "auto") -> dict[str, Any]:
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model: {model_key}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    _, feature_type, _, _ = MODEL_CONFIG[model_key]
    fold_results = []

    for fold in range(1, 6):
        checkpoint_path = CV_DIR / model_key / f"fold_{fold}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing {checkpoint_path}; train one model per held-out fold. A final model cannot be reused as CV evidence."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != model_key or checkpoint.get("feature_type") != feature_type:
            raise ValueError(f"Checkpoint metadata does not match requested model/features: {checkpoint_path}")
        model = _new_model(model_key, pretrained=False).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        x_test, y_test = _load_split(f"fold_{fold}_test", feature_type)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_test), batch_size):
                logits = model(x_test[start:start + batch_size].to(device))
                outputs.append(logits.cpu())
        temperature = max(0.05, min(20.0, float(checkpoint.get("temperature", 1.0))))
        probabilities = F.softmax(torch.cat(outputs, dim=0) / temperature, dim=1).numpy()
        metrics = classification_metrics(
            y_test.numpy(),
            probabilities,
            [category["name"] for category in ESC50_CATEGORIES],
        )
        fold_results.append({
            "fold": fold,
            "n": int(len(y_test)),
            "temperature": temperature,
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
        })
        print(f"{model_key} fold {fold}: top1={metrics['top1']:.4f}, ECE={metrics['ece']:.4f}", flush=True)
        del model, x_test, y_test
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {}
    for field in FIELDS:
        values = [float(row["metrics"][field]) for row in fold_results if row["metrics"].get(field) is not None]
        aggregate[field] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        }
    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": model_key,
        "evaluation": "ESC-50 official five-fold CV; each test fold scored by its fold-specific checkpoint",
        "featureSource": "Preproccesing/processed/features; same serving feature extractor pipeline",
        "nFolds": len(fold_results),
        "aggregate": aggregate,
        "folds": fold_results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / f"esc50_cv_{model_key}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_CONFIG, "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    models = list(MODEL_CONFIG) if args.model == "all" else [args.model]
    for model_key in models:
        evaluate(model_key, args.batch_size, args.device)


if __name__ == "__main__":
    main()

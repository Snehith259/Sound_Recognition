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
from eval.train_cv import MODEL_CONFIG, _load_split, _new_model, _temperature_scale

CV_DIR = ROOT / "checkpoints" / "cv"
REPORT_DIR = ROOT / "eval" / "reports"
ENSEMBLE_WEIGHTS_V2 = {"sresnet": 0.34, "vgg16": 0.28, "resnet18": 0.26, "cnngru": 0.12}
ENSEMBLE_WEIGHTS_V1 = {"sresnet": 0.30, "vgg16": 0.32, "resnet18": 0.28, "cnngru": 0.10}
FIELDS = (
    "top1", "top5", "macroF1", "nll", "brier", "ece", "r2Probability",
    "meanConfidence", "accuracyAtConfidence90", "coverageAtConfidence90",
)


def _checkpoint_arch_version(checkpoint: dict[str, Any]) -> int:
    if checkpoint.get("arch_version") is not None:
        return int(checkpoint["arch_version"])
    return 2 if str(checkpoint.get("format", "")).endswith("-v2") else 1


def _batched_logits(model: torch.nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(model(x[start:start + batch_size].to(device)).cpu())
    return torch.cat(outputs, dim=0)


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
        arch_version = _checkpoint_arch_version(checkpoint)
        model = _new_model(model_key, pretrained=False, arch_version=arch_version).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        x_test, y_test = _load_split(f"fold_{fold}_test", feature_type)
        logits = _batched_logits(model, x_test, batch_size, device)
        temperature = max(0.05, min(20.0, float(checkpoint.get("temperature", 1.0))))
        probabilities = F.softmax(logits / temperature, dim=1).numpy()
        metrics = classification_metrics(
            y_test.numpy(),
            probabilities,
            [category["name"] for category in ESC50_CATEGORIES],
        )
        fold_results.append({
            "fold": fold,
            "n": int(len(y_test)),
            "temperature": temperature,
            "archVersion": arch_version,
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
        })
        print(f"{model_key} (v{arch_version}) fold {fold}: top1={metrics['top1']:.4f}, ECE={metrics['ece']:.4f}", flush=True)
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


def evaluate_ensemble(batch_size: int = 64, device_name: str = "auto") -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available()) else "cpu")
    model_keys = list(MODEL_CONFIG)
    fold_results = []
    fold_temps: list[float] = []
    detected_versions: set[int] = set()

    for fold in range(1, 6):
        x_val, y_val = _load_split(f"fold_{fold}_val", "log_mel")
        x_test, y_test = _load_split(f"fold_{fold}_test", "log_mel")
        val_log_probs: dict[str, torch.Tensor] = {}
        test_log_probs: dict[str, torch.Tensor] = {}
        fold_versions: list[int] = []

        for model_key in model_keys:
            checkpoint_path = CV_DIR / model_key / f"fold_{fold}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            arch_version = _checkpoint_arch_version(checkpoint)
            fold_versions.append(arch_version)
            detected_versions.add(arch_version)
            model = _new_model(model_key, pretrained=False, arch_version=arch_version).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            temp = max(0.05, min(20.0, float(checkpoint.get("temperature", 1.0))))
            val_log_probs[model_key] = F.log_softmax(_batched_logits(model, x_val, batch_size, device) / temp, dim=1)
            test_log_probs[model_key] = F.log_softmax(_batched_logits(model, x_test, batch_size, device) / temp, dim=1)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        weights = ENSEMBLE_WEIGHTS_V2 if all(v == 2 for v in fold_versions) else ENSEMBLE_WEIGHTS_V1
        val_fused = sum(weights[k] * val_log_probs[k] for k in model_keys)
        test_fused = sum(weights[k] * test_log_probs[k] for k in model_keys)
        ens_temp = _temperature_scale(val_fused, y_val)
        fold_temps.append(ens_temp)
        probabilities = F.softmax(test_fused / ens_temp, dim=1).numpy()
        metrics = classification_metrics(
            y_test.numpy(),
            probabilities,
            [category["name"] for category in ESC50_CATEGORIES],
        )
        fold_results.append({
            "fold": fold,
            "n": int(len(y_test)),
            "temperature": ens_temp,
            "metrics": metrics,
        })
        print(f"ensemble fold {fold}: top1={metrics['top1']:.4f}, ECE={metrics['ece']:.4f}, T_ens={ens_temp:.4f}", flush=True)

    aggregate = {}
    for field in FIELDS:
        values = [float(row["metrics"][field]) for row in fold_results if row["metrics"].get(field) is not None]
        aggregate[field] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
        }
    arch_ver = 2 if detected_versions == {2} else 1
    weights = ENSEMBLE_WEIGHTS_V2 if arch_ver == 2 else ENSEMBLE_WEIGHTS_V1
    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "ensemble",
        "evaluation": "ESC-50 official five-fold CV; weighted temperature-scaled logit fusion calibrated on validation folds",
        "featureSource": "Preproccesing/processed/features; same serving feature extractor pipeline",
        "nFolds": len(fold_results),
        "weights": weights,
        "aggregate": aggregate,
        "folds": fold_results,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "esc50_cv_ensemble.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")

    ens_cfg_path = ROOT / "checkpoints" / "ensemble_config.json"
    ens_cfg_path.write_text(json.dumps({
        "schemaVersion": 1,
        "archVersion": arch_ver,
        "fusion": "weighted_log_prob",
        "weights": weights,
        "temperature": float(np.mean(fold_temps)),
        "foldTemperatures": fold_temps,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {ens_cfg_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_CONFIG, "ensemble", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.model == "ensemble":
        evaluate_ensemble(args.batch_size, args.device)
        return
    models = list(MODEL_CONFIG) if args.model == "all" else [args.model]
    for model_key in models:
        evaluate(model_key, args.batch_size, args.device)
    if args.model == "all":
        evaluate_ensemble(args.batch_size, args.device)


if __name__ == "__main__":
    main()

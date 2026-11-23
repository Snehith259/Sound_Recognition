"""Score one independently trained checkpoint per official ESC-50 test fold."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import soundfile as sf

from eval.metrics import classification_metrics
from Preproccesing.acoustic_inference import ESC50_CATEGORIES, load_audio_waveform
from Preproccesing.feature_extractor import extract_log_mel_spectrogram, extract_mfcc
from Preproccesing.train_models import ResNet18Transfer, VGG16Transfer, SResNetFast, CNNGRUFast

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = {
    "resnet18": (ResNet18Transfer, "log_mel"),
    "vgg16": (VGG16Transfer, "log_mel"),
    "sresnet": (SResNetFast, "mfcc"),
    "cnngru": (CNNGRUFast, "log_mel"),
}


def _load_wav(path: Path) -> np.ndarray:
    y, _ = load_audio_waveform(path, target_sr=22050, target_duration=5.0)
    return y


def evaluate(model_name: str, audio_dir: Path, metadata_csv: Path, checkpoint_dir: Path) -> dict:
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unsupported model: {model_name}")
    cls, feature_type = MODEL_CONFIG[model_name]
    metadata = pd.read_csv(metadata_csv)
    if not {"filename", "fold", "target"}.issubset(metadata.columns):
        raise ValueError("ESC-50 metadata must contain filename, fold, and target columns")
    fold_results = []
    for fold in range(1, 6):
        checkpoint = checkpoint_dir / model_name / f"fold_{fold}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing cross-validation checkpoint {checkpoint}. A single final model cannot be reused as CV evidence."
            )
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if model_name in {"resnet18", "vgg16"}:
            model = cls(num_classes=50, pretrained=False)
        else:
            model = cls(num_classes=50)
        model.load_state_dict(data["state_dict"], strict=True)
        model.eval()
        temperature = max(0.05, float(data.get("temperature", 1.0)))
        held_out = metadata[metadata["fold"] == fold]
        targets, probabilities = [], []
        with torch.no_grad():
            for row in held_out.itertuples(index=False):
                audio = _load_wav(audio_dir / row.filename)
                feat = extract_mfcc(audio) if feature_type == "mfcc" else extract_log_mel_spectrogram(audio)
                tensor = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
                logits = model(tensor) / temperature
                probabilities.append(torch.softmax(logits, dim=1)[0].numpy())
                targets.append(int(row.target))
        metrics = classification_metrics(targets, np.asarray(probabilities), [c["name"] for c in ESC50_CATEGORIES])
        fold_results.append({"fold": fold, "n": len(targets), "metrics": metrics, "checkpoint": str(checkpoint)})

    score_fields = ["top1", "top5", "macroF1", "nll", "brier", "ece", "r2Probability", "meanConfidence", "accuracyAtConfidence90"]
    aggregate = {}
    for field in score_fields:
        values = [r["metrics"].get(field) for r in fold_results if r["metrics"].get(field) is not None]
        aggregate[field] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return {"createdAt": datetime.now(timezone.utc).isoformat(), "model": model_name, "evaluation": "ESC-50 official 5-fold CV; each fold scored by its held-out checkpoint", "aggregate": aggregate, "folds": fold_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_CONFIG, "all"], default="all")
    parser.add_argument("--audio-dir", type=Path, default=ROOT / "ESC-50" / "audio")
    parser.add_argument("--metadata", type=Path, default=ROOT / "ESC-50" / "meta" / "esc50.csv")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints" / "cv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "eval" / "reports")
    args = parser.parse_args()
    models = list(MODEL_CONFIG) if args.model == "all" else [args.model]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in models:
        result = evaluate(model, args.audio_dir, args.metadata, args.checkpoint_dir)
        target = args.output_dir / f"esc50_cv_{model}.json"
        target.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{model}: {result['aggregate']['top1']['mean']:.4f} ± {result['aggregate']['top1']['std']:.4f} -> {target}")

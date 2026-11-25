"""Train five fold-specific models and optional deployment checkpoints from ESC-50 features.

Run on a Kaggle GPU session after copying this repository. The five official
fold manifests are used as train/validation/test groups; no test fold is used
for early stopping or temperature fitting. This runner uses the checked-in
precomputed feature tensors plus SpecAugment. It does not train on external
audio.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import math
from Preproccesing.train_models import (
    CNNGRUFast,
    CNNGRUV2,
    ResNet18Transfer,
    ResNet18V2,
    SResNetFast,
    SResNetV2,
    VGG16Transfer,
    VGG16V2,
)

FEATURE_DIR = ROOT / "Preproccesing" / "processed" / "features"
SPLIT_DIR = ROOT / "Preproccesing" / "splits"
CHECKPOINT_DIR = ROOT / "checkpoints"
MODEL_CONFIG = {
    "resnet18": (ResNet18V2, "log_mel", "esc50_resnet18_v1.pt", True),
    "vgg16": (VGG16V2, "log_mel", "esc50_vgg16_v1.pt", True),
    "sresnet": (SResNetV2, "log_mel", "esc50_sresnet_linear_v09.pt", True),
    "cnngru": (CNNGRUV2, "log_mel", "esc50_cnngru_recurrent_v08.pt", False),
}
MODEL_CONFIG_V1 = {
    "resnet18": (ResNet18Transfer, "log_mel", "esc50_resnet18_v1.pt", True),
    "vgg16": (VGG16Transfer, "log_mel", "esc50_vgg16_v1.pt", True),
    "sresnet": (SResNetFast, "log_mel", "esc50_sresnet_linear_v09.pt", True),
    "cnngru": (CNNGRUFast, "log_mel", "esc50_cnngru_recurrent_v08.pt", False),
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_split(split_name: str, feature_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    manifest_path = SPLIT_DIR / f"{split_name}.csv"
    feature_path = FEATURE_DIR / feature_type
    manifest = pd.read_csv(manifest_path)
    if not {"filename", "target"}.issubset(manifest.columns):
        raise ValueError(f"Split manifest is missing filename/target columns: {manifest_path}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    missing: list[str] = []
    for row in manifest.itertuples(index=False):
        path = feature_path / f"{Path(row.filename).stem}.npy"
        if not path.is_file():
            missing.append(str(path))
            continue
        features.append(np.load(path).astype(np.float32, copy=False))
        labels.append(int(row.target))
    if missing:
        raise FileNotFoundError(f"{len(missing)} feature files are missing; first: {missing[0]}")
    if not features:
        raise RuntimeError(f"No features were loaded for {split_name} ({feature_type}).")
    return torch.from_numpy(np.stack(features)).unsqueeze(1), torch.tensor(labels, dtype=torch.long)


def _loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(y) >= batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def _time_shift(x: torch.Tensor, max_shift_frac: float = 0.25) -> torch.Tensor:
    """Apply random circular time-roll per sample in a spectrogram batch."""
    out = x.clone()
    width = out.shape[-1]
    max_shift = max(1, int(width * max_shift_frac))
    for i in range(out.shape[0]):
        if torch.rand((), device=out.device).item() < 0.5:
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=out.device).item())
            if shift != 0:
                out[i] = torch.roll(out[i], shifts=shift, dims=-1)
    return out


def _spec_augment(x: torch.Tensor, time_masks: int = 2, freq_masks: int = 2) -> torch.Tensor:
    """Apply frequency/time masking and circular time-shift to spectrogram batches."""
    out = _time_shift(x)
    height, width = out.shape[-2:]
    max_freq = min(20, max(1, height // 6))
    max_time = min(36, max(1, width // 5))
    for sample in range(out.shape[0]):
        for _ in range(freq_masks):
            span = int(torch.randint(0, max_freq + 1, (1,), device=out.device).item())
            if span:
                start = int(torch.randint(0, max(1, height - span + 1), (1,), device=out.device).item())
                out[sample, :, start:start + span, :] = 0
        for _ in range(time_masks):
            span = int(torch.randint(0, max_time + 1, (1,), device=out.device).item())
            if span:
                start = int(torch.randint(0, max(1, width - span + 1), (1,), device=out.device).item())
                out[sample, :, :, start:start + span] = 0
    return out


def _soft_targets(y: torch.Tensor, num_classes: int = 50, smoothing: float = 0.05) -> torch.Tensor:
    targets = torch.full((y.size(0), num_classes), smoothing / num_classes, device=y.device, dtype=torch.float32)
    targets.scatter_(1, y.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)
    return targets


def _mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
    prob: float = 0.65,
    smoothing: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = _soft_targets(y, num_classes=50, smoothing=smoothing)
    if alpha > 0 and x.size(0) > 1 and float(np.random.random()) < prob:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1.0 - lam) * x[perm]
        targets = lam * targets + (1.0 - lam) * targets[perm]
    return x, targets


def _new_model(model_key: str, pretrained: bool, arch_version: int = 2) -> torch.nn.Module:
    cfg = MODEL_CONFIG if arch_version == 2 else MODEL_CONFIG_V1
    cls, _feature_type, _filename, uses_pretrained = cfg[model_key]
    kwargs = {"num_classes": 50}
    if model_key in {"resnet18", "vgg16"} or (arch_version == 2 and model_key == "sresnet"):
        kwargs["pretrained"] = bool(pretrained and uses_pretrained)
    elif arch_version == 1 and model_key == "sresnet":
        kwargs["pretrained"] = bool(pretrained and uses_pretrained)
    return cls(**kwargs)


def _forward(model: torch.nn.Module, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=use_amp):
        return model(x)


def _predict_logits(model: torch.nn.Module, x: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(model(x[start:start + batch_size].to(device)))
    return torch.cat(outputs, dim=0)


def _temperature_scale(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single positive temperature on the validation fold only."""
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    logits = logits.detach().double().cpu()
    labels = labels.detach().long().cpu()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def _train_epochs(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_x: torch.Tensor | None,
    validation_y: torch.Tensor | None,
    epochs: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
    batch_size: int,
    uses_pretrained: bool = True,
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor | None]:
    use_amp = device.type == "cuda"
    model.to(device)
    if uses_pretrained and hasattr(model, "backbone") and hasattr(model, "head"):
        param_groups = [
            {"params": model.backbone.parameters(), "lr": learning_rate},
            {"params": model.head.parameters(), "lr": learning_rate * 5.0},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": learning_rate}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    warmup_epochs = min(5, max(1, epochs // 10))

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return 0.2 + 0.8 * ((ep + 1) / warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1.0
    best_epoch = epochs
    stale_epochs = 0
    best_logits: torch.Tensor | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x_batch, y_batch in train_loader:
            x_batch = _spec_augment(x_batch.to(device))
            y_batch = y_batch.to(device)
            x_batch, soft_y = _mixup_batch(x_batch, y_batch, alpha=0.4, prob=0.65, smoothing=0.05)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x_batch)
                loss = -(soft_y * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if validation_x is None or validation_y is None:
            print(f"    epoch {epoch}/{epochs} loss={float(np.mean(losses)):.4f}", flush=True)
            continue

        validation_logits = _predict_logits(model, validation_x, batch_size, device).cpu()
        accuracy = float((validation_logits.argmax(dim=1) == validation_y).float().mean().item())
        print(f"    epoch {epoch}/{epochs} loss={float(np.mean(losses)):.4f} val_top1={accuracy:.4f}", flush=True)
        if accuracy > best_score:
            best_score = accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_logits = validation_logits
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        return best_state, best_epoch, None
    return best_state, best_epoch, best_logits


def _checkpoint_path(model_key: str, fold: int) -> Path:
    return CHECKPOINT_DIR / "cv" / model_key / f"fold_{fold}.pt"


def _save(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}. Pass --overwrite to replace it.")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def train_model(
    model_key: str,
    folds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    include_final: bool,
    overwrite: bool,
    arch_version: int = 2,
) -> dict[str, Any]:
    if include_final and folds != [1, 2, 3, 4, 5]:
        raise ValueError("--include-final requires folds 1 through 5 in order.")
    cfg = MODEL_CONFIG if arch_version == 2 else MODEL_CONFIG_V1
    _cls, feature_type, final_filename, uses_pretrained = cfg[model_key]
    learning_rate = 1e-4 if uses_pretrained else 1.5e-3
    fold_epochs: list[int] = []
    fold_temperatures: list[float] = []

    for fold in folds:
        print(f"\n[{model_key} v{arch_version}] fold {fold}/5", flush=True)
        _seed_everything(seed + fold)
        x_train, y_train = _load_split(f"fold_{fold}_train", feature_type)
        x_val, y_val = _load_split(f"fold_{fold}_val", feature_type)
        train_loader = _loader(x_train, y_train, batch_size, True, device)
        model = _new_model(model_key, pretrained=True, arch_version=arch_version)
        best_state, best_epoch, validation_logits = _train_epochs(
            model, train_loader, x_val, y_val, epochs, patience, learning_rate, device, batch_size, uses_pretrained
        )
        if validation_logits is None:
            raise RuntimeError(f"Validation logits missing for {model_key}, fold {fold}.")
        temperature = _temperature_scale(validation_logits, y_val)
        split_data = pd.read_csv(SPLIT_DIR / f"fold_{fold}_train.csv")
        train_folds = sorted(int(value) for value in split_data["fold"].unique())
        payload = {
            "format": f"soundsense-cv-v{arch_version}",
            "arch_version": arch_version,
            "model_name": model_key,
            "feature_type": feature_type,
            "num_classes": 50,
            "temperature": temperature,
            "state_dict": best_state,
            "training": {
                "heldOutFold": fold,
                "trainFolds": train_folds,
                "validationFold": int(pd.read_csv(SPLIT_DIR / f"fold_{fold}_val.csv")["fold"].iloc[0]),
                "bestEpoch": best_epoch,
                "validationTop1": float((validation_logits.argmax(dim=1) == y_val).float().mean().item()),
                "augmentation": "SpecAugment + circular time-shift + Mixup (alpha=0.4)",
                "seed": seed + fold,
                "device": str(device),
                "pretrainedImageNet": uses_pretrained,
                "archVersion": arch_version,
            },
        }
        target = _checkpoint_path(model_key, fold)
        _save(target, payload, overwrite)
        fold_epochs.append(best_epoch)
        fold_temperatures.append(temperature)
        print(f"  saved {target}; validation temperature={temperature:.4f}", flush=True)
        del model, x_train, y_train, x_val, y_val, train_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_path = CHECKPOINT_DIR / final_filename
    if include_final:
        if len(fold_epochs) != 5:
            raise ValueError("--include-final requires all five folds to finish for that model.")
        final_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
        print(f"\n[{model_key} v{arch_version}] final model on all five folds for {final_epochs} epochs", flush=True)
        _seed_everything(seed + 1000)
        x_all, y_all = _load_split("dataset_manifest", feature_type) if (SPLIT_DIR / "dataset_manifest.csv").exists() else (None, None)
        # The master manifest is a dataset manifest; load its rows with the same strict loader.
        if x_all is None:
            raise FileNotFoundError("The master dataset manifest is required to train a final model.")
        all_loader = _loader(x_all, y_all, batch_size, True, device)
        model = _new_model(model_key, pretrained=True, arch_version=arch_version)
        final_state, _actual_epochs, _ = _train_epochs(
            model, all_loader, None, None, final_epochs, patience, learning_rate, device, batch_size, uses_pretrained
        )
        final_payload = {
            "format": f"soundsense-final-v{arch_version}",
            "arch_version": arch_version,
            "model_name": model_key,
            "feature_type": feature_type,
            "num_classes": 50,
            "temperature": float(np.mean(fold_temperatures)),
            "state_dict": final_state,
            "training": {
                "folds": [1, 2, 3, 4, 5],
                "epochs": final_epochs,
                "temperatureSource": "mean of five validation-fold temperatures",
                "cvMeanEpoch": float(np.mean(fold_epochs)),
                "cvMeanTemperature": float(np.mean(fold_temperatures)),
                "augmentation": "SpecAugment + circular time-shift + Mixup (alpha=0.4)",
                "pretrainedImageNet": uses_pretrained,
                "archVersion": arch_version,
                "warning": "This final model has no in-sample quality score; use CV and external reports.",
            },
        }
        _save(final_path, final_payload, overwrite)
        del model, x_all, y_all, all_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "model": model_key,
        "archVersion": arch_version,
        "foldCheckpoints": [str(_checkpoint_path(model_key, fold)) for fold in folds],
        "bestEpochs": fold_epochs,
        "temperatures": fold_temperatures,
        "finalCheckpoint": str(final_path) if include_final else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_CONFIG, "all"], default="all")
    parser.add_argument("--folds", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arch-version", type=int, choices=[1, 2], default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--include-final", action="store_true", help="Train deployment checkpoints after all five folds finish")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing checkpoints at the target paths")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    models = list(MODEL_CONFIG) if args.model == "all" else [args.model]
    started = time.time()
    results = []
    for model_key in models:
        results.append(train_model(
            model_key=model_key,
            folds=args.folds,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            include_final=args.include_final,
            overwrite=args.overwrite,
            arch_version=args.arch_version,
        ))
    output = {
        "createdAt": time.time(),
        "elapsedSeconds": round(time.time() - started, 1),
        "device": str(device),
        "models": results,
        "note": "This is training metadata, not an evaluation report. Run eval_esc50_cv_features and external evaluation before publishing scores.",
    }
    output_path = ROOT / "eval" / "reports" / "training_runs.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

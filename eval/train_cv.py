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
    CNNGRUV3,
    Cnn10AudioV4,
    DyMNAudioV4,
    MN20GRUAudioV4,
    ResNet18Transfer,
    ResNet18V2,
    ResNet18V3,
    ResNet22AudioV4,
    SResNetFast,
    SResNetV2,
    SResNetV3,
    VGG16Transfer,
    VGG16V2,
    VGG16V3,
)

FEATURE_DIR = ROOT / "Preproccesing" / "processed" / "features"
AUDIO_22K_DIR = ROOT / "Preproccesing" / "processed" / "audio_22k"
ESC50_RAW_DIR = ROOT / "ESC-50" / "wav_files"
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
MODEL_CONFIG_V3 = {
    "resnet18": (ResNet18V3, "log_mel", "esc50_resnet18_v1.pt", True),
    "vgg16": (VGG16V3, "log_mel", "esc50_vgg16_v1.pt", True),
    "sresnet": (SResNetV3, "log_mel", "esc50_sresnet_linear_v09.pt", True),
    "cnngru": (CNNGRUV3, "log_mel", "esc50_cnngru_recurrent_v08.pt", True),
}
MODEL_CONFIG_V4 = {
    "resnet18": (ResNet22AudioV4, "wave_32k", "esc50_resnet18_v1.pt", True),
    "vgg16": (Cnn10AudioV4, "wave_32k", "esc50_vgg16_v1.pt", True),
    "sresnet": (DyMNAudioV4, "wave_32k", "esc50_sresnet_linear_v09.pt", True),
    "cnngru": (MN20GRUAudioV4, "wave_32k", "esc50_cnngru_recurrent_v08.pt", True),
}
MODEL_CONFIG_BY_VERSION = {
    1: MODEL_CONFIG_V1,
    2: MODEL_CONFIG,
    3: MODEL_CONFIG_V3,
    4: MODEL_CONFIG_V4,
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_WAVE_32K_CACHE: dict[str, np.ndarray] = {}


def _load_wave_32k(filename: str) -> np.ndarray:
    if filename in _WAVE_32K_CACHE:
        return _WAVE_32K_CACHE[filename]
    import soundfile as sf
    import soxr
    stem = Path(filename).stem
    for d in (ESC50_RAW_DIR, AUDIO_22K_DIR):
        p = d / filename
        if not p.is_file():
            p = d / f"{stem}.wav"
        if p.is_file():
            y, sr = sf.read(str(p), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != 32000:
                y = soxr.resample(y, sr, 32000)
            y = np.asarray(y, dtype=np.float32)
            target_len = 32000 * 5
            if y.size < target_len:
                y = np.pad(y, (0, target_len - y.size))
            elif y.size > target_len:
                y = y[:target_len]
            if peak := (float(np.max(np.abs(y))) if y.size else 0.0):
                if peak > 1e-8:
                    y = y / peak
            _WAVE_32K_CACHE[filename] = y
            return y
    raise FileNotFoundError(f"Audio clip not found for {filename}")


def _load_split(split_name: str, feature_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    manifest_path = SPLIT_DIR / f"{split_name}.csv"
    manifest = pd.read_csv(manifest_path)
    if not {"filename", "target"}.issubset(manifest.columns):
        raise ValueError(f"Split manifest is missing filename/target columns: {manifest_path}")

    if feature_type == "wave_32k":
        waves = [_load_wave_32k(str(row.filename)) for row in manifest.itertuples(index=False)]
        labels = [int(row.target) for row in manifest.itertuples(index=False)]
        return torch.from_numpy(np.stack(waves)).float(), torch.tensor(labels, dtype=torch.long)

    feature_path = FEATURE_DIR / feature_type
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
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.float()
    targets = torch.full((y.size(0), num_classes), smoothing / num_classes, device=y.device, dtype=torch.float32)
    targets.scatter_(1, y.long().unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)
    return targets


def _stem_augment(s: torch.Tensor) -> torch.Tensor:
    """Lightweight spatial-temporal masking directly in cached stem feature space."""
    out = s.clone()
    height, width = out.shape[-2:]
    max_f = max(1, height // 4)
    max_t = max(1, width // 5)
    for idx in range(out.shape[0]):
        if torch.rand((), device=out.device).item() < 0.55:
            span = int(torch.randint(1, max_f + 1, (1,), device=out.device).item())
            start = int(torch.randint(0, max(1, height - span + 1), (1,), device=out.device).item())
            out[idx, :, start:start + span, :] = 0.0
        if torch.rand((), device=out.device).item() < 0.55:
            span = int(torch.randint(1, max_t + 1, (1,), device=out.device).item())
            start = int(torch.randint(0, max(1, width - span + 1), (1,), device=out.device).item())
            out[idx, :, :, start:start + span] = 0.0
    return out


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
    cfg = MODEL_CONFIG_BY_VERSION.get(arch_version)
    if cfg is None:
        raise ValueError(f"Unsupported arch_version={arch_version}")
    cls, _feature_type, _filename, uses_pretrained = cfg[model_key]
    kwargs = {"num_classes": 50}
    if model_key in {"resnet18", "vgg16", "sresnet"} or (arch_version >= 3 and model_key == "cnngru"):
        kwargs["pretrained"] = bool(pretrained and uses_pretrained)
    return cls(**kwargs)


def _forward(model: torch.nn.Module, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=use_amp):
        return model(x)


def _predict_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
    stem_mode: bool = False,
) -> torch.Tensor:
    model.eval()
    forward_fn = model.forward_tail if stem_mode else model
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(forward_fn(x[start:start + batch_size].to(device)))
    return torch.cat(outputs, dim=0)


def _extract_stem_features(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute frozen stem activations once using no_grad (safe for downstream autograd)."""
    model.to(device)
    model.stem.eval()
    feats = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            feats.append(model.forward_stem(x[start:start + batch_size].to(device)).cpu())
    return torch.cat(feats, dim=0)


def _load_training_data(
    split_name: str,
    feature_type: str,
    arch_version: int,
    mix_samples: int,
    mix_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Load canonical features, or generate real acoustic waveform mixtures for V3 training."""
    if arch_version >= 3 and mix_samples > 0:
        from Preproccesing.waveform_mixer import generate_mixed_spectrogram_dataset, load_fold_waveforms

        manifest_df = pd.read_csv(SPLIT_DIR / f"{split_name}.csv")
        t0 = time.time()
        waves, targets = load_fold_waveforms(manifest_df)
        n_gen = max(mix_samples, int(round(mix_samples * len(waves) / 1200.0))) if len(waves) > 1200 else mix_samples
        x_mix, y_mix = generate_mixed_spectrogram_dataset(
            waves=waves,
            targets=targets,
            n_samples=n_gen,
            num_classes=50,
            mix_fraction=mix_fraction,
            short_crop_fraction=0.18,
            label_smoothing=0.04,
            seed=seed,
        )
        print(
            f"  generated {len(x_mix)} acoustic mix+domain samples from {len(waves)} clips "
            f"in {time.time() - t0:.1f}s (mix_frac={mix_fraction:.2f})",
            flush=True,
        )
        return x_mix, y_mix, True
    x_raw, y_raw = _load_split(split_name, feature_type)
    return x_raw, y_raw, False


def _temperature_scale(logits: torch.Tensor, labels: torch.Tensor, min_temp: float = 0.05) -> float:
    """Fit a single positive temperature on the validation fold only."""
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    logits = logits.detach().double().cpu()
    labels = labels.detach().long().cpu()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(min_temp, 20.0)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min_temp, 20.0).item())


def _build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    uses_pretrained: bool,
    stem_mode: bool,
    e2e_polish: bool,
) -> torch.optim.Optimizer:
    if stem_mode and hasattr(model, "tail") and hasattr(model, "head"):
        return torch.optim.AdamW(
            [
                {"params": model.tail.parameters(), "lr": learning_rate * 3.0},
                {"params": model.head.parameters(), "lr": learning_rate * 8.0},
            ],
            weight_decay=2e-3,
        )
    if e2e_polish and hasattr(model, "stem") and hasattr(model, "tail") and hasattr(model, "head"):
        return torch.optim.AdamW(
            [
                {"params": model.stem.parameters(), "lr": learning_rate * 0.5},
                {"params": model.tail.parameters(), "lr": learning_rate * 1.2},
                {"params": model.head.parameters(), "lr": learning_rate * 3.0},
            ],
            weight_decay=2e-3,
        )
    if uses_pretrained and hasattr(model, "stem") and hasattr(model, "tail") and hasattr(model, "head"):
        return torch.optim.AdamW(
            [
                {"params": model.stem.parameters(), "lr": learning_rate * 1.0},
                {"params": model.tail.parameters(), "lr": learning_rate * 1.8},
                {"params": model.head.parameters(), "lr": learning_rate * 5.0},
            ],
            weight_decay=1e-3,
        )
    if uses_pretrained and hasattr(model, "backbone") and hasattr(model, "head"):
        return torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": learning_rate},
                {"params": model.head.parameters(), "lr": learning_rate * 5.0},
            ],
            weight_decay=1e-3,
        )
    if uses_pretrained and hasattr(model, "tail") and hasattr(model, "head"):
        return torch.optim.AdamW(
            [
                {"params": model.tail.parameters(), "lr": learning_rate * 2.0},
                {"params": model.head.parameters(), "lr": learning_rate * 5.0},
            ],
            weight_decay=1e-3,
        )
    return torch.optim.AdamW([{"params": model.parameters(), "lr": learning_rate}], weight_decay=1e-3)


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
    stem_mode: bool = False,
    e2e_polish: bool = False,
    is_mixed_dataset: bool = False,
    init_best_state: dict[str, torch.Tensor] | None = None,
    init_best_score: tuple[float, float] = (-1.0, -1e9),
    init_best_logits: torch.Tensor | None = None,
    max_batches_per_epoch: int | None = None,
    freeze_stem_bn: bool = False,
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor | None, tuple[float, float]]:
    if epochs <= 0:
        state = init_best_state or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return state, 0, init_best_logits, init_best_score

    use_amp = device.type == "cuda"
    model.to(device)
    optimizer = _build_optimizer(model, learning_rate, uses_pretrained, stem_mode, e2e_polish)
    warmup_epochs = min(3, max(1, epochs // 6))

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return 0.25 + 0.75 * ((ep + 1) / warmup_epochs)
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state = init_best_state
    best_score = init_best_score
    best_epoch = epochs
    stale_epochs = 0
    best_logits = init_best_logits
    aug_fn = _stem_augment if stem_mode else _spec_augment
    forward_fn = model.forward_tail if stem_mode else model
    mix_alpha = 0.0 if is_mixed_dataset else 0.40
    mix_prob = 0.0 if is_mixed_dataset else 0.65
    tag = "stem-tail" if stem_mode else ("e2e-polish" if e2e_polish else "full-e2e")

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        if (stem_mode or freeze_stem_bn) and hasattr(model, "stem"):
            model.stem.eval()
        losses = []
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            x_batch = aug_fn(x_batch.to(device))
            y_batch = y_batch.to(device)
            x_batch, soft_y = _mixup_batch(x_batch, y_batch, alpha=mix_alpha, prob=mix_prob, smoothing=0.04)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = forward_fn(x_batch)
                loss = -(soft_y * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        if validation_x is None or validation_y is None:
            print(f"    [{tag}] epoch {epoch}/{epochs} loss={float(np.mean(losses)):.4f} ({time.time() - ep_t0:.1f}s)", flush=True)
            continue

        validation_logits = _predict_logits(model, validation_x, batch_size, device, stem_mode=stem_mode).cpu()
        accuracy = float((validation_logits.argmax(dim=1) == validation_y).float().mean().item())
        val_nll = float(F.cross_entropy(validation_logits, validation_y).item())
        score = (round(accuracy, 6), -val_nll)
        print(
            f"    [{tag}] epoch {epoch}/{epochs} loss={float(np.mean(losses)):.4f} "
            f"val_top1={accuracy:.4f} val_nll={val_nll:.4f} ({time.time() - ep_t0:.1f}s)",
            flush=True,
        )
        if score > best_score:
            best_score = score
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
        return best_state, best_epoch, best_logits, best_score
    return best_state, best_epoch, best_logits, best_score


def _checkpoint_path(model_key: str, fold: int) -> Path:
    return CHECKPOINT_DIR / "cv" / model_key / f"fold_{fold}.pt"


def _save(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}. Pass --overwrite to replace it.")
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def _fit_model_stages(
    model: torch.nn.Module,
    model_key: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor | None,
    y_val: torch.Tensor | None,
    epochs: int,
    e2e_epochs: int,
    patience: int,
    learning_rate: float,
    device: torch.device,
    batch_size: int,
    uses_pretrained: bool,
    is_mixed: bool,
    max_batches_per_epoch: int | None = None,
) -> tuple[dict[str, torch.Tensor], int, int, torch.Tensor | None]:
    # ResNet18V3 & CNNGRUV3 are fast enough to train 100% end-to-end (~34-45s/ep), avoiding frozen-stem BN bias.
    # Heavy stems (VGG16V3 & SResNetV3 ConvNeXt) warm up tail+head via cached stem features then polish E2E.
    use_stem_first = hasattr(model, "forward_stem") and hasattr(model, "forward_tail") and model_key in {"vgg16", "sresnet"}
    if use_stem_first and epochs > 0:
        stem_epochs = max(1, epochs - max(0, e2e_epochs))
        t0 = time.time()
        x_train_stem = _extract_stem_features(model, x_train, batch_size, device)
        x_val_stem = _extract_stem_features(model, x_val, batch_size, device) if x_val is not None else None
        print(f"  cached stem features {tuple(x_train_stem.shape)} in {time.time() - t0:.1f}s", flush=True)
        stem_loader = _loader(x_train_stem, y_train, batch_size, True, device)
        best_state, best_stem_ep, best_logits, best_score = _train_epochs(
            model,
            stem_loader,
            x_val_stem,
            y_val,
            stem_epochs,
            patience,
            learning_rate,
            device,
            batch_size,
            uses_pretrained,
            stem_mode=True,
            is_mixed_dataset=is_mixed,
        )
        del stem_loader, x_train_stem, x_val_stem

        best_e2e_ep = 0
        if e2e_epochs > 0:
            model.load_state_dict(best_state, strict=True)
            e2e_loader = _loader(x_train, y_train, batch_size, True, device)
            # Keep stem BN frozen only if Conv2d-heavy VGG/ResNet polish; ConvNeXt uses LayerNorm (always safe to train!)
            freeze_bn = model_key == "vgg16"
            best_state, best_e2e_ep, best_logits, best_score = _train_epochs(
                model,
                e2e_loader,
                x_val,
                y_val,
                e2e_epochs,
                patience,
                learning_rate,
                device,
                batch_size,
                uses_pretrained,
                stem_mode=False,
                e2e_polish=True,
                is_mixed_dataset=is_mixed,
                init_best_state=best_state,
                init_best_score=best_score,
                init_best_logits=best_logits,
                max_batches_per_epoch=max_batches_per_epoch,
                freeze_stem_bn=freeze_bn,
            )
            del e2e_loader
        return best_state, best_stem_ep, best_e2e_ep, best_logits

    train_loader = _loader(x_train, y_train, batch_size, True, device)
    best_state, best_ep, best_logits, _ = _train_epochs(
        model,
        train_loader,
        x_val,
        y_val,
        epochs,
        patience,
        learning_rate,
        device,
        batch_size,
        uses_pretrained,
        is_mixed_dataset=is_mixed,
        max_batches_per_epoch=max_batches_per_epoch,
    )
    del train_loader
    return best_state, best_ep, 0, best_logits


def _active_crop_32k(wave: np.ndarray, crop_len: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a high-energy active window from a 32 kHz 5 s ESC-50 clip."""
    n = wave.shape[0]
    if crop_len >= n:
        shift = int(rng.integers(0, n))
        return np.roll(wave, shift).copy()
    frame = 1600
    hop = 800
    n_frames = max(1, (n - frame) // hop + 1)
    rms = np.empty(n_frames, dtype=np.float32)
    for idx in range(n_frames):
        seg = wave[idx * hop: idx * hop + frame]
        rms[idx] = float(np.sqrt(np.mean(seg * seg)))
    thresh = max(1e-5, 0.22 * float(np.max(rms)))
    active = np.flatnonzero(rms >= thresh)
    if active.size > 0:
        center = int(rng.choice(active)) * hop + frame // 2
        start = int(np.clip(center - crop_len // 2 + int(rng.integers(-4000, 4001)), 0, n - crop_len))
    else:
        start = int(rng.integers(0, max(1, n - crop_len + 1)))
    seg = wave[start: start + crop_len].copy()
    fade = min(384, crop_len // 6)
    if fade > 4:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
    return seg


def _extract_v4_clean_cache(
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, tuple[torch.Tensor, ...]]:
    """Pre-extract deterministic frozen AudioSet-2M features once for all 2,000 ESC-50 clips."""
    manifest = pd.read_csv(SPLIT_DIR / "dataset_manifest.csv")
    filenames = [str(f) for f in manifest["filename"]]
    t0 = time.time()
    cache: dict[str, tuple[torch.Tensor, ...]] = {}
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(filenames), batch_size):
            batch_names = filenames[start: start + batch_size]
            wavs = torch.from_numpy(np.stack([_load_wave_32k(fn) for fn in batch_names])).float().to(device)
            feats = model.extract_frozen_features(wavs)
            # Last element is as_logits; keep only classifier input tensors.
            head_tensors = feats[:-1]
            for i, fn in enumerate(batch_names):
                stored = tuple(
                    (t[i].half() if i_t > 0 and t.ndim == 3 else t[i].float()).cpu()
                    for i_t, t in enumerate(head_tensors)
                )
                cache[fn] = stored
    print(f"  cached {len(cache)} clean 32kHz AudioSet embeddings in {time.time() - t0:.1f}s", flush=True)
    return cache


def _extract_v4_fold_mixes(
    model: torch.nn.Module,
    split_df: pd.DataFrame,
    n_samples: int,
    mix_fraction: float,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Generate fold-strict 32 kHz waveform mixtures/short bursts and extract frozen features."""
    rng = np.random.default_rng(seed)
    waves = [_load_wave_32k(str(fn)) for fn in split_df["filename"]]
    labels = np.asarray([int(t) for t in split_df["target"]], dtype=np.int64)
    by_class: dict[int, list[int]] = {}
    for idx, cls_id in enumerate(labels):
        by_class.setdefault(int(cls_id), []).append(idx)
    classes = list(by_class.keys())

    target_len = 32000 * 5
    smooth = 0.03
    n_classes = 50
    all_feat_chunks: list[list[torch.Tensor]] = []
    all_targets = torch.zeros((n_samples, n_classes), dtype=torch.float32)

    t0 = time.time()
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            batch_waves = np.zeros((end - start, target_len), dtype=np.float32)
            for b_i, g_i in enumerate(range(start, end)):
                idx_a = int(rng.integers(0, len(waves)))
                cls_a = int(labels[idx_a])
                y_a = np.roll(waves[idx_a], int(rng.integers(0, target_len)))
                if rng.random() < mix_fraction and len(classes) > 1:
                    cls_b = int(rng.choice([c for c in classes if c != cls_a]))
                    idx_b = int(rng.choice(by_class[cls_b]))
                    y_b = np.roll(waves[idx_b], int(rng.integers(0, target_len)))
                    rms_a = float(np.sqrt(np.mean(y_a * y_a))) + 1e-8
                    rms_b = float(np.sqrt(np.mean(y_b * y_b))) + 1e-8
                    gain_b = float(rng.uniform(0.32, 0.85))
                    mix = (y_a / rms_a) + gain_b * (y_b / rms_b)
                    pk = float(np.max(np.abs(mix)))
                    if pk > 1e-8:
                        mix = mix / pk
                    batch_waves[b_i] = mix
                    # Acoustic energy-power ratio matching dominant+secondary events cleanly
                    p_a = 1.0 / (1.0 + gain_b * gain_b)
                    p_b = 1.0 - p_a
                    t_vec = torch.full((n_classes,), smooth / n_classes, dtype=torch.float32)
                    t_vec[cls_a] += (1.0 - smooth) * p_a
                    t_vec[cls_b] += (1.0 - smooth) * p_b
                    all_targets[g_i] = t_vec
                else:
                    len_a = int(rng.integers(int(0.65 * 32000), int(3.2 * 32000) + 1))
                    seg = _active_crop_32k(waves[idx_a], len_a, rng)
                    buf = np.zeros(target_len, dtype=np.float32)
                    off = 0 if rng.random() < 0.4 else int(rng.integers(0, max(1, target_len - len_a + 1)))
                    buf[off: off + len_a] = seg
                    pk = float(np.max(np.abs(buf)))
                    if pk > 1e-8:
                        buf = buf / pk
                    batch_waves[b_i] = buf
                    t_vec = torch.full((n_classes,), smooth / n_classes, dtype=torch.float32)
                    t_vec[cls_a] += (1.0 - smooth)
                    all_targets[g_i] = t_vec

            feats = model.extract_frozen_features(torch.from_numpy(batch_waves).to(device))
            head_tensors = feats[:-1]
            if not all_feat_chunks:
                all_feat_chunks = [[] for _ in head_tensors]
            for i_t, t in enumerate(head_tensors):
                all_feat_chunks[i_t].append((t.half() if i_t > 0 and t.ndim == 3 else t.float()).cpu())

    merged_feats = tuple(torch.cat(chunks, dim=0) for chunks in all_feat_chunks)
    print(
        f"  extracted {n_samples} fold-strict 32kHz mixes/crops in {time.time() - t0:.1f}s "
        f"(mix_frac={mix_fraction:.2f})",
        flush=True,
    )
    return merged_feats, all_targets


def _gather_v4_split_features(
    cache: dict[str, tuple[torch.Tensor, ...]],
    split_df: pd.DataFrame,
    smoothing: float = 0.03,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    rows = [cache[str(fn)] for fn in split_df["filename"]]
    n_t = len(rows[0])
    feats = tuple(torch.stack([r[i] for r in rows], dim=0) for i in range(n_t))
    labels = torch.tensor([int(t) for t in split_df["target"]], dtype=torch.long)
    targets = _soft_targets(labels, num_classes=50, smoothing=smoothing)
    return feats, targets


def _v4_serialize_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Serialize V4 checkpoint: store frozen backbone weights in fp16 to halve disk footprint."""
    sd = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu().clone()
        if k.startswith("backbone.") and t.is_floating_point():
            t = t.half()
        sd[k] = t
    return sd


def _fit_v4_head(
    head: torch.nn.Module,
    train_feats: tuple[torch.Tensor, ...],
    train_targets: torch.Tensor,
    val_feats: tuple[torch.Tensor, ...] | None,
    val_labels: torch.Tensor | None,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, torch.Tensor | None]:
    head.to(device)
    ds = TensorDataset(*train_feats, train_targets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1.5e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=2e-5)

    best_score = -1e9
    best_ep = 1
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    best_val_logits: torch.Tensor | None = None

    for epoch in range(1, epochs + 1):
        head.train()
        for batch in loader:
            *x_parts, yb = batch
            inputs = [part.to(device).float() for part in x_parts]
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = head(*inputs)
            loss = -(yb * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=3.0)
            optimizer.step()
        scheduler.step()

        if val_feats is not None and val_labels is not None:
            head.eval()
            with torch.inference_mode():
                val_in = [p.to(device).float() for p in val_feats]
                val_logits = head(*val_in).cpu()
                val_acc = float((val_logits.argmax(dim=1) == val_labels).float().mean().item())
                val_nll = float(F.cross_entropy(val_logits, val_labels).item())
                score = val_acc - 0.02 * val_nll
            if score > best_score:
                best_score = score
                best_ep = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                best_val_logits = val_logits

    if val_feats is None:
        best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        best_ep = epochs
    return best_state, best_ep, best_val_logits


def _train_v4_model(
    model_key: str,
    folds: list[int],
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    include_final: bool,
    overwrite: bool,
    mix_samples: int,
    mix_fraction: float,
) -> dict[str, Any]:
    _cls, feature_type, final_filename, _ = MODEL_CONFIG_V4[model_key]
    v4_mix = 1200 if mix_samples < 0 else mix_samples
    v4_epochs = max(24, epochs)
    v4_mix_frac = 0.62 if mix_fraction <= 0.45 else mix_fraction
    lr = 1.6e-3

    print(f"\n[{model_key} v4] loading pretrained AudioSet-2M backbone...", flush=True)
    model = _new_model(model_key, pretrained=True, arch_version=4)
    clean_cache = _extract_v4_clean_cache(model, device=device, batch_size=32)

    fold_epochs: list[int] = []
    fold_temperatures: list[float] = []

    for fold in folds:
        print(f"\n[{model_key} v4] fold {fold}/5", flush=True)
        _seed_everything(seed + fold)
        split_df = pd.read_csv(SPLIT_DIR / f"fold_{fold}_train.csv")
        val_df = pd.read_csv(SPLIT_DIR / f"fold_{fold}_val.csv")
        test_df = pd.read_csv(SPLIT_DIR / f"fold_{fold}_test.csv")
        train_folds = sorted(int(v) for v in split_df["fold"].unique())
        val_fold = int(val_df["fold"].iloc[0])
        test_fold = int(test_df["fold"].iloc[0])
        if val_fold in train_folds or test_fold in train_folds or val_fold == test_fold:
            raise RuntimeError(f"Leakage detected in fold {fold}: train={train_folds}, val={val_fold}, test={test_fold}")

        clean_feats, clean_targets = _gather_v4_split_features(clean_cache, split_df, smoothing=0.03)
        val_feats, _ = _gather_v4_split_features(clean_cache, val_df, smoothing=0.0)
        y_val = torch.tensor([int(t) for t in val_df["target"]], dtype=torch.long)

        if v4_mix > 0:
            mix_feats, mix_targets = _extract_v4_fold_mixes(
                model, split_df, v4_mix, v4_mix_frac, seed + fold, device=device, batch_size=32
            )
            train_feats = tuple(torch.cat([cf, mf], dim=0) for cf, mf in zip(clean_feats, mix_feats))
            train_targets = torch.cat([clean_targets, mix_targets], dim=0)
        else:
            train_feats = clean_feats
            train_targets = clean_targets

        fresh_head = _new_model(model_key, pretrained=False, arch_version=4).head
        model.head.load_state_dict(fresh_head.state_dict())
        del fresh_head

        best_head_state, best_epoch, val_logits = _fit_v4_head(
            model.head,
            train_feats,
            train_targets,
            val_feats,
            y_val,
            epochs=v4_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        assert val_logits is not None
        model.head.load_state_dict(best_head_state)
        temperature = _temperature_scale(val_logits, y_val, min_temp=0.85)
        payload = {
            "format": "soundsense-cv-v4",
            "arch_version": 4,
            "model_name": model_key,
            "feature_type": feature_type,
            "num_classes": 50,
            "temperature": temperature,
            "state_dict": _v4_serialize_state(model),
            "training": {
                "heldOutFold": fold,
                "trainFolds": train_folds,
                "validationFold": val_fold,
                "bestEpoch": best_epoch,
                "bestE2EEpochs": 0,
                "mixSamples": int(len(train_targets)),
                "validationTop1": float((val_logits.argmax(dim=1) == y_val).float().mean().item()),
                "augmentation": (
                    f"32kHz clean+active 2-sound mix ({v4_mix_frac:.0%} of {v4_mix} synths) + short burst crops"
                ),
                "seed": seed + fold,
                "device": str(device),
                "pretrainedAudioSet": True,
                "archVersion": 4,
            },
        }
        target = _checkpoint_path(model_key, fold)
        _save(target, payload, overwrite)
        fold_epochs.append(best_epoch)
        fold_temperatures.append(temperature)
        print(
            f"  saved {target}; val_top1={payload['training']['validationTop1']:.4f}, "
            f"T={temperature:.4f} (ep={best_epoch}/{v4_epochs})",
            flush=True,
        )

    final_path = CHECKPOINT_DIR / final_filename
    if include_final:
        if len(fold_epochs) != 5:
            raise ValueError("--include-final requires all five folds to finish for that model.")
        final_ep = max(1, int(round(float(np.mean(fold_epochs)))))
        print(f"\n[{model_key} v4] final model on all five folds (ep={final_ep})", flush=True)
        _seed_everything(seed + 1000)
        all_df = pd.read_csv(SPLIT_DIR / "dataset_manifest.csv")
        clean_feats, clean_targets = _gather_v4_split_features(clean_cache, all_df, smoothing=0.03)
        n_final_mix = int(round(v4_mix * len(all_df) / 1200.0)) if v4_mix > 0 else 0
        if n_final_mix > 0:
            mix_feats, mix_targets = _extract_v4_fold_mixes(
                model, all_df, n_final_mix, v4_mix_frac, seed + 1000, device=device, batch_size=32
            )
            train_feats = tuple(torch.cat([cf, mf], dim=0) for cf, mf in zip(clean_feats, mix_feats))
            train_targets = torch.cat([clean_targets, mix_targets], dim=0)
        else:
            train_feats = clean_feats
            train_targets = clean_targets

        fresh_head = _new_model(model_key, pretrained=False, arch_version=4).head
        model.head.load_state_dict(fresh_head.state_dict())
        del fresh_head

        best_head_state, _, _ = _fit_v4_head(
            model.head,
            train_feats,
            train_targets,
            None,
            None,
            epochs=final_ep,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
        model.head.load_state_dict(best_head_state)
        final_payload = {
            "format": "soundsense-final-v4",
            "arch_version": 4,
            "model_name": model_key,
            "feature_type": feature_type,
            "num_classes": 50,
            "temperature": float(np.mean(fold_temperatures)),
            "state_dict": _v4_serialize_state(model),
            "training": {
                "folds": [1, 2, 3, 4, 5],
                "epochs": final_ep,
                "stemEpochs": final_ep,
                "e2eEpochs": 0,
                "mixSamples": int(len(train_targets)),
                "temperatureSource": "mean of five validation-fold temperatures",
                "cvMeanEpoch": float(np.mean(fold_epochs)),
                "cvMeanTemperature": float(np.mean(fold_temperatures)),
                "augmentation": (
                    f"32kHz clean+active 2-sound mix ({v4_mix_frac:.0%} of {n_final_mix} synths) + short burst crops"
                ),
                "pretrainedAudioSet": True,
                "archVersion": 4,
                "warning": "This final model has no in-sample quality score; use CV and external reports.",
            },
        }
        _save(final_path, final_payload, overwrite)
        print(f"  saved {final_path}; T={final_payload['temperature']:.4f}", flush=True)

    del model, clean_cache
    return {
        "model": model_key,
        "archVersion": 4,
        "foldCheckpoints": [str(_checkpoint_path(model_key, fold)) for fold in folds],
        "bestEpochs": fold_epochs,
        "bestE2EEpochs": [0 for _ in fold_epochs],
        "temperatures": fold_temperatures,
        "finalCheckpoint": str(final_path) if include_final else None,
    }


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
    arch_version: int = 4,
    mix_samples: int = -1,
    mix_fraction: float = 0.40,
    e2e_epochs: int = 3,
    max_batches_per_epoch: int | None = None,
) -> dict[str, Any]:
    if include_final and folds != [1, 2, 3, 4, 5]:
        raise ValueError("--include-final requires folds 1 through 5 in order.")
    if arch_version == 4:
        return _train_v4_model(
            model_key=model_key,
            folds=folds,
            epochs=max(epochs, 34) if epochs <= 25 else epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            include_final=include_final,
            overwrite=overwrite,
            mix_samples=mix_samples,
            mix_fraction=mix_fraction,
        )
    resolved_mix = 2800 if mix_samples < 0 else mix_samples
    cfg = MODEL_CONFIG_BY_VERSION.get(arch_version)
    if cfg is None:
        raise ValueError(f"Unsupported arch_version={arch_version}")
    _cls, feature_type, final_filename, uses_pretrained = cfg[model_key]
    learning_rate = 1.5e-4 if uses_pretrained else 1.5e-3
    fold_epochs: list[int] = []
    fold_e2e_epochs: list[int] = []
    fold_temperatures: list[float] = []

    for fold in folds:
        print(f"\n[{model_key} v{arch_version}] fold {fold}/5", flush=True)
        _seed_everything(seed + fold)
        split_data = pd.read_csv(SPLIT_DIR / f"fold_{fold}_train.csv")
        val_df = pd.read_csv(SPLIT_DIR / f"fold_{fold}_val.csv")
        test_df = pd.read_csv(SPLIT_DIR / f"fold_{fold}_test.csv")
        train_folds = sorted(int(value) for value in split_data["fold"].unique())
        val_fold = int(val_df["fold"].iloc[0])
        test_fold = int(test_df["fold"].iloc[0])
        if val_fold in train_folds or test_fold in train_folds or val_fold == test_fold:
            raise RuntimeError(f"Leakage detected in fold {fold}: train={train_folds}, val={val_fold}, test={test_fold}")

        x_train, y_train, is_mixed = _load_training_data(
            f"fold_{fold}_train", feature_type, arch_version, resolved_mix, mix_fraction, seed + fold
        )
        x_val, y_val = _load_split(f"fold_{fold}_val", feature_type)
        model = _new_model(model_key, pretrained=True, arch_version=arch_version)
        best_state, best_epoch, best_e2e_ep, validation_logits = _fit_model_stages(
            model,
            model_key,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs,
            e2e_epochs if arch_version >= 3 else 0,
            patience,
            learning_rate,
            device,
            batch_size,
            uses_pretrained,
            is_mixed,
            max_batches_per_epoch=max_batches_per_epoch,
        )
        if validation_logits is None:
            raise RuntimeError(f"Validation logits missing for {model_key}, fold {fold}.")
        temperature = _temperature_scale(validation_logits, y_val)
        aug_desc = (
            f"Real waveform 2-sound mix ({mix_fraction:.0%}) + short bursts + spectral tilt + colored noise + SpecAugment"
            if is_mixed
            else "SpecAugment + circular time-shift + Mixup (alpha=0.4)"
        )
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
                "validationFold": val_fold,
                "bestEpoch": best_epoch,
                "bestE2EEpochs": best_e2e_ep,
                "mixSamples": int(len(x_train)) if is_mixed else 0,
                "validationTop1": float((validation_logits.argmax(dim=1) == y_val).float().mean().item()),
                "augmentation": aug_desc,
                "seed": seed + fold,
                "device": str(device),
                "pretrainedImageNet": uses_pretrained,
                "archVersion": arch_version,
            },
        }
        target = _checkpoint_path(model_key, fold)
        _save(target, payload, overwrite)
        fold_epochs.append(best_epoch)
        fold_e2e_epochs.append(best_e2e_ep)
        fold_temperatures.append(temperature)
        print(
            f"  saved {target}; val_top1={payload['training']['validationTop1']:.4f}, "
            f"T={temperature:.4f} (stem={best_epoch}, e2e={best_e2e_ep})",
            flush=True,
        )
        del model, x_train, y_train, x_val, y_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_path = CHECKPOINT_DIR / final_filename
    if include_final:
        if len(fold_epochs) != 5:
            raise ValueError("--include-final requires all five folds to finish for that model.")
        final_stem_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
        final_e2e_epochs = int(round(float(np.mean(fold_e2e_epochs)))) if any(fold_e2e_epochs) else 0
        final_total_epochs = final_stem_epochs + final_e2e_epochs
        print(
            f"\n[{model_key} v{arch_version}] final model on all five folds "
            f"(stem={final_stem_epochs}, e2e={final_e2e_epochs})",
            flush=True,
        )
        _seed_everything(seed + 1000)
        if not (SPLIT_DIR / "dataset_manifest.csv").exists():
            raise FileNotFoundError("The master dataset manifest is required to train a final model.")
        x_all, y_all, is_mixed = _load_training_data(
            "dataset_manifest", feature_type, arch_version, resolved_mix, mix_fraction, seed + 1000
        )
        model = _new_model(model_key, pretrained=True, arch_version=arch_version)
        final_state, _, _, _ = _fit_model_stages(
            model,
            model_key,
            x_all,
            y_all,
            None,
            None,
            final_total_epochs,
            final_e2e_epochs,
            patience,
            learning_rate,
            device,
            batch_size,
            uses_pretrained,
            is_mixed,
            max_batches_per_epoch=max_batches_per_epoch,
        )
        aug_desc = (
            f"Real waveform 2-sound mix ({mix_fraction:.0%}) + short bursts + spectral tilt + colored noise + SpecAugment"
            if is_mixed
            else "SpecAugment + circular time-shift + Mixup (alpha=0.4)"
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
                "epochs": final_total_epochs,
                "stemEpochs": final_stem_epochs,
                "e2eEpochs": final_e2e_epochs,
                "mixSamples": int(len(x_all)) if is_mixed else 0,
                "temperatureSource": "mean of five validation-fold temperatures",
                "cvMeanEpoch": float(np.mean(fold_epochs)),
                "cvMeanE2EEpochs": float(np.mean(fold_e2e_epochs)),
                "cvMeanTemperature": float(np.mean(fold_temperatures)),
                "augmentation": aug_desc,
                "pretrainedImageNet": uses_pretrained,
                "archVersion": arch_version,
                "warning": "This final model has no in-sample quality score; use CV and external reports.",
            },
        }
        _save(final_path, final_payload, overwrite)
        del model, x_all, y_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "model": model_key,
        "archVersion": arch_version,
        "foldCheckpoints": [str(_checkpoint_path(model_key, fold)) for fold in folds],
        "bestEpochs": fold_epochs,
        "bestE2EEpochs": fold_e2e_epochs,
        "temperatures": fold_temperatures,
        "finalCheckpoint": str(final_path) if include_final else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODEL_CONFIG, "all"], default="all")
    parser.add_argument("--folds", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=19)
    parser.add_argument("--e2e-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mix-samples", type=int, default=-1)
    parser.add_argument("--mix-fraction", type=float, default=0.40)
    parser.add_argument("--max-batches", type=int, default=26)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--arch-version", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--include-final", action="store_true", help="Train deployment checkpoints after all five folds finish")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing checkpoints at the target paths")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if device.type == "cpu":
        try:
            torch.set_num_threads(min(8, max(1, (os.cpu_count() or 8) - 2)))
        except (RuntimeError, ValueError):
            pass
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
            mix_samples=args.mix_samples,
            mix_fraction=args.mix_fraction,
            e2e_epochs=args.e2e_epochs,
            max_batches_per_epoch=args.max_batches if args.max_batches > 0 else None,
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

"""Five-fold AST fine-tuning on ESC-50.

Fine-tunes `MIT/ast-finetuned-audioset-10-10-0.4593` (AudioSet-pretrained AST)
on the official ESC-50 five-fold split. Each fold produces a HuggingFace
`save_pretrained()` directory under ``checkpoints/cv/ast/fold_{k}/``. The best
fold (highest validation accuracy) is copied to ``checkpoints/ast_finetuned/``
for deployment.

Requires a CUDA GPU for practical speed (15 epochs × 5 folds ≈ 30-60 min on T4).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from Preproccesing.acoustic_inference import ESC50_CATEGORIES

SPLIT_DIR = ROOT / "Preproccesing" / "splits"
DEFAULT_AUDIO_DIR = ROOT / "Preproccesing" / "processed" / "audio_22k"
CHECKPOINT_DIR = ROOT / "checkpoints"
CV_DIR = CHECKPOINT_DIR / "cv" / "ast"
FINETUNED_DIR = CHECKPOINT_DIR / "ast_finetuned"

AST_BASE_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_SAMPLE_RATE = 16000
CLIP_DURATION = 5.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _amp_autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _make_grad_scaler(use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def find_esc50_audio_path(filename: str, audio_root: Path | None = None) -> Path:
    """Locate the real waveform file for an ESC-50 manifest entry."""
    stem = Path(filename).stem
    candidates: list[Path] = []
    search_dirs = [
        audio_root,
        DEFAULT_AUDIO_DIR,
        ROOT / "ESC-50" / "audio",
        ROOT / "ESC-50" / "wav_files",
        ROOT / "data" / "ESC-50" / "audio",
        ROOT / "Preproccesing" / "processed" / "esc50_audio",
    ]
    for directory in search_dirs:
        if directory is None:
            continue
        candidates.append(directory / filename)
        candidates.append(directory / f"{stem}.wav")
        candidates.append(directory / f"{stem}.ogg")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Real ESC-50 audio file not found for '{filename}'. Checked {DEFAULT_AUDIO_DIR} and ESC-50 directories."
    )


class ESC50WaveformDataset(Dataset):
    """Loads real 5-second waveforms resampled to 16 kHz for AST fine-tuning."""

    def __init__(
        self,
        split_csv: Path,
        audio_root: Path | None = None,
        sr: int = AST_SAMPLE_RATE,
        augment: bool = False,
    ):
        self.sr = sr
        self.augment = augment
        self.samples: list[tuple[Path, int]] = []

        df = pd.read_csv(split_csv)
        if not {"filename", "target"}.issubset(df.columns):
            raise ValueError(f"Split manifest missing filename/target columns: {split_csv}")
        for row in df.itertuples(index=False):
            audio_path = find_esc50_audio_path(str(row.filename), audio_root=audio_root)
            self.samples.append((audio_path, int(row.target)))

        if not self.samples:
            raise FileNotFoundError(f"No audio files loaded for split {split_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_waveform(self, path: Path) -> np.ndarray:
        """Load and resample a real audio file to self.sr mono float32."""
        import librosa
        import soundfile as sf

        target_samples = int(self.sr * CLIP_DURATION)
        try:
            y, sr = sf.read(str(path), dtype="float32")
        except Exception:
            y, sr = librosa.load(str(path), sr=self.sr, mono=True)

        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != self.sr and y.size:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        elif len(y) > target_samples:
            y = y[:target_samples]
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        """Apply conservative waveform-level augmentations."""
        if random.random() < 0.7:
            gain_db = random.uniform(-6.0, 6.0)
            y = y * (10.0 ** (gain_db / 20.0))

        if random.random() < 0.5:
            snr_db = random.uniform(15.0, 40.0)
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms > 1e-8:
                noise_rms = rms / (10.0 ** (snr_db / 20.0))
                y = y + np.random.randn(len(y)).astype(np.float32) * noise_rms

        if random.random() < 0.5 and len(y) > 0:
            shift = random.randint(-int(len(y) * 0.2), int(len(y) * 0.2))
            y = np.roll(y, shift)

        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        path, label = self.samples[idx]
        y = self._load_waveform(path)
        if self.augment:
            y = self._augment_waveform(y)
        return y, label


def _collate_fn(batch: list[tuple[np.ndarray, int]]) -> tuple[list[np.ndarray], torch.Tensor]:
    waveforms = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return waveforms, labels


def _temperature_scale(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single positive temperature on validation logits."""
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    logits_d = logits.detach().double().cpu()
    labels_d = labels.detach().long().cpu()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(logits_d / temperature, labels_d)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def _spec_augment_inputs(input_values: torch.Tensor, freq_masks: int = 2, time_masks: int = 2) -> torch.Tensor:
    """Apply SpecAugment masking to ASTFeatureExtractor inputs (B, 1024, 128)."""
    out = input_values.clone()
    if out.dim() != 3:
        return out
    time_steps, freq_bins = out.shape[1], out.shape[2]
    for i in range(out.shape[0]):
        for _ in range(freq_masks):
            width = random.randint(0, min(24, max(1, freq_bins // 4)))
            if width:
                start = random.randint(0, max(0, freq_bins - width))
                out[i, :, start:start + width] = 0.0
        for _ in range(time_masks):
            width = random.randint(0, min(48, max(1, time_steps // 4)))
            if width:
                start = random.randint(0, max(0, time_steps - width))
                out[i, start:start + width, :] = 0.0
    return out


def _mixup_labels(
    labels: torch.Tensor,
    num_classes: int = 50,
    alpha: float = 0.3,
    prob: float = 0.5,
    smoothing: float = 0.05,
) -> tuple[torch.Tensor, float | None, torch.Tensor | None]:
    """Create label-smoothed soft targets with optional Mixup."""
    targets = torch.full(
        (labels.size(0), num_classes),
        smoothing / num_classes,
        device=labels.device,
        dtype=torch.float32,
    )
    targets.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing + smoothing / num_classes)

    if alpha > 0 and labels.size(0) > 1 and random.random() < prob:
        lam = float(np.random.beta(alpha, alpha))
        perm = torch.randperm(labels.size(0), device=labels.device)
        targets = lam * targets + (1.0 - lam) * targets[perm]
        return targets, lam, perm
    return targets, None, None


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def train_fold(
    fold: int,
    epochs: int,
    patience: int,
    batch_size: int,
    grad_accum: int,
    lr_backbone: float,
    lr_head: float,
    device: torch.device,
    seed: int,
    overwrite: bool,
    audio_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one AST fold on real audio and save HuggingFace fold checkpoint."""
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    _seed_everything(seed + fold)
    print(f"\n[AST] fold {fold}/5", flush=True)

    train_csv = SPLIT_DIR / f"fold_{fold}_train.csv"
    val_csv = SPLIT_DIR / f"fold_{fold}_val.csv"

    train_ds = ESC50WaveformDataset(train_csv, audio_root=audio_dir, augment=True)
    val_ds = ESC50WaveformDataset(val_csv, audio_root=audio_dir, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_fn,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn,
    )

    id2label = {cat["id"]: cat["name"] for cat in ESC50_CATEGORIES}
    label2id = {cat["name"]: cat["id"] for cat in ESC50_CATEGORIES}

    feature_extractor = ASTFeatureExtractor.from_pretrained(AST_BASE_MODEL)
    model = ASTForAudioClassification.from_pretrained(
        AST_BASE_MODEL,
        num_labels=50,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
    ])

    grad_accum = max(1, int(grad_accum))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = steps_per_epoch

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(0.1, (step + 1) / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    use_amp = device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    best_val_acc = -1.0
    best_epoch = 0
    stale = 0
    best_state = None
    val_logits_best = None
    val_labels_best = None
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (waveforms, labels) in enumerate(train_loader):
            inputs = feature_extractor(
                waveforms,
                sampling_rate=AST_SAMPLE_RATE,
                return_tensors="pt",
                padding="max_length",
            )
            input_values = _spec_augment_inputs(inputs.input_values.to(device))
            labels_dev = labels.to(device)

            soft_targets, lam, perm = _mixup_labels(labels_dev, num_classes=50)
            if lam is not None and perm is not None:
                input_values = lam * input_values + (1.0 - lam) * input_values[perm]

            with _amp_autocast(device):
                logits = model(input_values=input_values).logits
                loss = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * grad_accum)

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for waveforms, labels in val_loader:
                inputs = feature_extractor(
                    waveforms,
                    sampling_rate=AST_SAMPLE_RATE,
                    return_tensors="pt",
                    padding="max_length",
                )
                input_values = inputs.input_values.to(device)
                with _amp_autocast(device):
                    logits = model(input_values=input_values).logits
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        val_logits = torch.cat(all_logits, dim=0)
        val_labels = torch.cat(all_labels, dim=0)
        val_acc = float((val_logits.argmax(dim=1) == val_labels).float().mean().item())

        mean_loss = float(np.mean(losses)) if losses else 0.0
        print(f"    epoch {epoch}/{epochs} loss={mean_loss:.4f} val_top1={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            val_logits_best = val_logits.clone()
            val_labels_best = val_labels.clone()
        else:
            stale += 1
            if stale >= patience:
                print(f"    early stop at epoch {epoch} (patience={patience})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = _temperature_scale(val_logits_best, val_labels_best) if val_logits_best is not None else 1.0

    fold_dir = CV_DIR / f"fold_{fold}"
    if fold_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing fold directory: {fold_dir}. Pass --overwrite.")
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(fold_dir)
    feature_extractor.save_pretrained(fold_dir)

    meta = {
        "fold": fold,
        "best_epoch": best_epoch,
        "val_top1": best_val_acc,
        "temperature": temperature,
        "base_model": AST_BASE_MODEL,
        "seed": seed + fold,
        "epochs_trained": epochs_ran,
    }
    (fold_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved fold {fold} -> {fold_dir} | val_top1={best_val_acc:.4f} T={temperature:.4f}", flush=True)

    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return meta


def train_all_folds(
    folds: list[int],
    epochs: int,
    patience: int,
    batch_size: int,
    grad_accum: int,
    lr_backbone: float,
    lr_head: float,
    device: torch.device,
    seed: int,
    include_final: bool,
    overwrite: bool,
    audio_dir: Path | None = None,
) -> dict[str, Any]:
    """Train requested AST folds and optionally copy the best validation fold to ast_finetuned/."""
    if include_final and sorted(folds) != [1, 2, 3, 4, 5]:
        raise ValueError("--include-final requires folds 1 through 5.")

    fold_results = []
    for fold in folds:
        fold_results.append(
            train_fold(
                fold=fold,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                grad_accum=grad_accum,
                lr_backbone=lr_backbone,
                lr_head=lr_head,
                device=device,
                seed=seed,
                overwrite=overwrite,
                audio_dir=audio_dir,
            )
        )

    if include_final:
        best_fold = max(fold_results, key=lambda r: r["val_top1"])
        best_fold_num = best_fold["fold"]
        source_dir = CV_DIR / f"fold_{best_fold_num}"

        if FINETUNED_DIR.exists():
            if not overwrite:
                raise FileExistsError(f"Refusing to overwrite {FINETUNED_DIR}. Pass --overwrite.")
            shutil.rmtree(FINETUNED_DIR)
        shutil.copytree(source_dir, FINETUNED_DIR)

        sf_path = FINETUNED_DIR / "model.safetensors"
        temp_config = {
            "temperature": best_fold["temperature"],
            "source_fold": best_fold_num,
            "val_top1": best_fold["val_top1"],
            "mean_temperature": float(np.mean([r["temperature"] for r in fold_results])),
            "fold_temperatures": {str(r["fold"]): r["temperature"] for r in fold_results},
            "sha256": _file_sha256(sf_path) if sf_path.is_file() else None,
        }
        (FINETUNED_DIR / "ast_temperature.json").write_text(
            json.dumps(temp_config, indent=2), encoding="utf-8"
        )
        print(
            f"\n[AST] Deployed best fold {best_fold_num} (val_top1={best_fold['val_top1']:.4f}) -> {FINETUNED_DIR}",
            flush=True,
        )

    summary = {
        "createdAt": time.time(),
        "model": "ast",
        "baseModel": AST_BASE_MODEL,
        "folds": fold_results,
        "meanValTop1": float(np.mean([r["val_top1"] for r in fold_results])),
        "stdValTop1": float(np.std([r["val_top1"] for r in fold_results], ddof=1)) if len(fold_results) > 1 else 0.0,
        "device": str(device),
        "note": "Training metadata only. Run `python -m eval.eval_esc50_cv_features --model ast` to generate held-out CV test metrics.",
    }
    report_dir = ROOT / "eval" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ast_training_runs.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, choices=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Copy best fold to checkpoints/ast_finetuned/ for deployment",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing fold checkpoints",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    print(f"[AST fine-tuning] device={device}, base={AST_BASE_MODEL}", flush=True)
    print(f"  epochs={args.epochs}, patience={args.patience}, bs={args.batch_size}x{args.grad_accum}", flush=True)

    train_all_folds(
        folds=args.folds,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        device=device,
        seed=args.seed,
        include_final=args.include_final,
        overwrite=args.overwrite,
        audio_dir=args.audio_dir,
    )


if __name__ == "__main__":
    main()

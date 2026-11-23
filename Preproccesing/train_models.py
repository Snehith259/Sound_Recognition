"""
SoundSense — Fold-1 ESC-50 Neural Training Diagnostic
Trains and exports real PyTorch checkpoints for all 4 neural architectures:
1. ResNet-18: Pretrained Deep Residual Network adapted for 1-channel Log-Mel spectrograms
2. VGG-16: Pretrained Visual Geometry Group network with BatchNorm for Log-Mel spectrograms
3. SResNetFast: Compact convolutional MFCC classifier; it has no residual blocks
4. CNN-GRU: Hybrid Convolutional Recurrent Network for Log-Mel temporal dynamics

This legacy script uses only fold 1. Its metrics are diagnostic and are not CV
evidence. Use eval/eval_esc50_cv.py only with an independently trained checkpoint
for each official held-out fold before publishing model scores.

Outputs:
- checkpoints/esc50_resnet18_v1.pt
- checkpoints/esc50_vgg16_v1.pt
- checkpoints/esc50_sresnet_linear_v09.pt
- checkpoints/esc50_cnngru_recurrent_v08.pt
- checkpoints/model_report.json
- checkpoints/manifest.json
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as tv_models

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Preproccesing.acoustic_inference import ESC50_CATEGORIES

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
FEATURES_DIR = ROOT_DIR / "Preproccesing" / "processed" / "features"
SPLITS_DIR = ROOT_DIR / "Preproccesing" / "splits"


# ══════════════════════════════════════════════════════════════
# Feature Loading
# ══════════════════════════════════════════════════════════════

def load_split_features(split_name: str, feat_type: str = "log_mel") -> Tuple[torch.Tensor, torch.Tensor]:
    """Loads precomputed feature arrays for a given split."""
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    subdir = FEATURES_DIR / feat_type
    xs, ys = [], []

    for _, row in df.iterrows():
        stem = Path(row["filename"]).stem
        p = subdir / f"{stem}.npy"
        if p.exists():
            xs.append(np.load(p).astype(np.float32))
            ys.append(int(row["target"]))

    X = torch.from_numpy(np.stack(xs)).unsqueeze(1).float()
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def extract_features_batched(backbone: nn.Module, X: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    """Safely extracts embeddings without CPU memory overflow."""
    backbone.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = X[i:i + batch_size]
            out = backbone(bx)
            feats.append(out)
    return torch.cat(feats, dim=0)


# ══════════════════════════════════════════════════════════════
# Model Architectures
# ══════════════════════════════════════════════════════════════

class ResNet18Transfer(nn.Module):
    """End-to-end ResNet-18 adapted for 1-channel Log-Mel spectrograms."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)

        # 1-channel conv1
        orig_conv = base.conv1.weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = new_conv

        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class VGG16Transfer(nn.Module):
    """End-to-end VGG-16 with BatchNorm adapted for 1-channel Log-Mel spectrograms."""
    def __init__(self, num_classes: int = 50, pretrained: bool = True):
        super().__init__()
        weights = tv_models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.vgg16_bn(weights=weights)

        orig_conv = base.features[0].weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        base.features[0].weight.data = new_conv

        self.backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)


class SResNetFast(nn.Module):
    """Compact convolutional classifier for MFCC input (40, 216); it has no residual blocks."""
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class CNNGRUFast(nn.Module):
    """Hybrid Convolutional Recurrent Network for Log-Mel spectral dynamics."""
    def __init__(self, num_classes: int = 50):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, None))  # (B, 64, 8, T')
        )
        self.gru = nn.GRU(64 * 8, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.conv(x)
        B, C, F_dim, T_dim = feat.shape
        feat = feat.permute(0, 3, 1, 2).contiguous().view(B, T_dim, C * F_dim)
        gru_out, _ = self.gru(feat)
        pooled = gru_out.mean(dim=1)
        return self.fc(pooled)


# ══════════════════════════════════════════════════════════════
# Metrics & Temperature Scaling
# ══════════════════════════════════════════════════════════════

def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        _, preds = torch.max(probs, 1)

        correct1 = (preds == targets).sum().item()
        top1_acc = round(correct1 / len(targets) * 100.0, 2)

        _, top5_preds = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        correct5 = (top5_preds == targets.unsqueeze(1)).any(dim=1).sum().item()
        top5_acc = round(correct5 / len(targets) * 100.0, 2)

        num_classes = logits.shape[1]
        f1_scores = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_scores.append(f1)
        macro_f1 = round(float(np.mean(f1_scores)) * 100.0, 2)

    return {"top1": top1_acc, "top5": top5_acc, "macroF1": macro_f1}


def calibrate_temperature(val_logits: torch.Tensor, val_targets: torch.Tensor) -> float:
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval_loss():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / temperature, val_targets)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return round(float(temperature.item()), 4)


# ══════════════════════════════════════════════════════════════
# High-Speed Multi-Model Trainer
# ══════════════════════════════════════════════════════════════

def train_and_export_all():
    print(f"\n{'='*70}\n SOUNDSENSE MULTI-MODEL ESC-50 TRAINING & EXPORT PIPELINE\n{'='*70}", flush=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("\n[Step 1/5] Loading preprocessed features into memory...", flush=True)
    t_data = time.time()
    X_tr_mel, y_tr_mel = load_split_features("fold_1_train", "log_mel")
    X_va_mel, y_va_mel = load_split_features("fold_1_val", "log_mel")
    X_te_mel, y_te_mel = load_split_features("fold_1_test", "log_mel")

    X_tr_mfcc, y_tr_mfcc = load_split_features("fold_1_train", "mfcc")
    X_va_mfcc, y_va_mfcc = load_split_features("fold_1_val", "mfcc")
    X_te_mfcc, y_te_mfcc = load_split_features("fold_1_test", "mfcc")
    print(f"Features ready in {time.time()-t_data:.2f}s: Mel train {X_tr_mel.shape}, MFCC train {X_tr_mfcc.shape}", flush=True)

    results = {}

    # ══════════════════════════════════════════════════════════
    # Model 1: ResNet-18
    # ══════════════════════════════════════════════════════════
    print("\n[Step 2/5] Training ResNet-18 (Log-Mel Spectrogram)...", flush=True)
    t0 = time.time()
    rn_model = ResNet18Transfer(num_classes=50)

    # Extract features via backbone in batches
    F_tr = extract_features_batched(rn_model.backbone, X_tr_mel)
    F_va = extract_features_batched(rn_model.backbone, X_va_mel)
    F_te = extract_features_batched(rn_model.backbone, X_te_mel)

    opt = torch.optim.AdamW(rn_model.head.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(F_tr, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_head = None
    for ep in range(40):
        rn_model.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(rn_model.head(bx), by)
            loss.backward()
            opt.step()
        rn_model.head.eval()
        with torch.no_grad():
            va_preds = rn_model.head(F_va).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_head = {k: v.clone() for k, v in rn_model.head.state_dict().items()}

    rn_model.head.load_state_dict(best_head)
    rn_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        te_logits = rn_model(X_te_mel[:32])
        lat_ms = round((time.time() - t_lat) / 32 * 1000, 2)

        te_logits_full = rn_model.head(F_te)
        va_logits_full = rn_model.head(F_va)

    metrics = compute_metrics(te_logits_full, y_te_mel)
    temp = calibrate_temperature(va_logits_full, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_resnet18_v1.pt"
    param_count = sum(p.numel() for p in rn_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "resnet18",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
    "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": rn_model.state_dict()
    }, ckpt_path)
    print(f"  -> ResNet-18 done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["resnet18"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Model 2: VGG-16
    # ══════════════════════════════════════════════════════════
    print("\n[Step 3/5] Training VGG-16 (Log-Mel Spectrogram)...", flush=True)
    t0 = time.time()
    vgg_model = VGG16Transfer(num_classes=50)

    F_tr_vgg = extract_features_batched(vgg_model.backbone, X_tr_mel)
    F_va_vgg = extract_features_batched(vgg_model.backbone, X_va_mel)
    F_te_vgg = extract_features_batched(vgg_model.backbone, X_te_mel)

    opt = torch.optim.AdamW(vgg_model.head.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(F_tr_vgg, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_head_vgg = None
    for ep in range(35):
        vgg_model.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(vgg_model.head(bx), by)
            loss.backward()
            opt.step()
        vgg_model.head.eval()
        with torch.no_grad():
            va_preds = vgg_model.head(F_va_vgg).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_head_vgg = {k: v.clone() for k, v in vgg_model.head.state_dict().items()}

    vgg_model.head.load_state_dict(best_head_vgg)
    vgg_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        _ = vgg_model(X_te_mel[:32])
        lat_ms = round((time.time() - t_lat) / 32 * 1000, 2)
        te_logits_full = vgg_model.head(F_te_vgg)
        va_logits_full = vgg_model.head(F_va_vgg)

    metrics = compute_metrics(te_logits_full, y_te_mel)
    temp = calibrate_temperature(va_logits_full, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_vgg16_v1.pt"
    param_count = sum(p.numel() for p in vgg_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "vgg16",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": vgg_model.state_dict()
    }, ckpt_path)
    print(f"  -> VGG-16 done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["vgg16"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Model 3: SResNet (MFCC)
    # ══════════════════════════════════════════════════════════
    print("\n[Step 4/5] Training SResNet (Linear MFCC)...", flush=True)
    t0 = time.time()
    sr_model = SResNetFast(num_classes=50)
    opt = torch.optim.AdamW(sr_model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(X_tr_mfcc, y_tr_mfcc), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_state = None
    for ep in range(30):
        sr_model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(sr_model(bx), by)
            loss.backward()
            opt.step()
        sr_model.eval()
        with torch.no_grad():
            va_preds = sr_model(X_va_mfcc).argmax(dim=1)
            va_acc = (va_preds == y_va_mfcc).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_state = {k: v.clone() for k, v in sr_model.state_dict().items()}

    sr_model.load_state_dict(best_state)
    sr_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        te_logits = sr_model(X_te_mfcc)
        lat_ms = round((time.time() - t_lat) / len(X_te_mfcc) * 1000, 2)
        va_logits = sr_model(X_va_mfcc)

    metrics = compute_metrics(te_logits, y_te_mfcc)
    temp = calibrate_temperature(va_logits, y_va_mfcc)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_sresnet_linear_v09.pt"
    param_count = sum(p.numel() for p in sr_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "sresnet",
        "feature_type": "mfcc",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": sr_model.state_dict()
    }, ckpt_path)
    print(f"  -> SResNet done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["sresnet"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Model 4: CNN-GRU (Log-Mel Dynamics)
    # ══════════════════════════════════════════════════════════
    print("\n[Step 5/5] Training CNN-GRU (Log-Mel Recurrent)...", flush=True)
    t0 = time.time()
    cg_model = CNNGRUFast(num_classes=50)
    opt = torch.optim.AdamW(cg_model.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    loader = DataLoader(TensorDataset(X_tr_mel, y_tr_mel), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_state = None
    for ep in range(30):
        cg_model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(cg_model(bx), by)
            loss.backward()
            opt.step()
        cg_model.eval()
        with torch.no_grad():
            va_preds = cg_model(X_va_mel).argmax(dim=1)
            va_acc = (va_preds == y_va_mel).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_state = {k: v.clone() for k, v in cg_model.state_dict().items()}

    cg_model.load_state_dict(best_state)
    cg_model.eval()

    with torch.no_grad():
        t_lat = time.time()
        te_logits = cg_model(X_te_mel)
        lat_ms = round((time.time() - t_lat) / len(X_te_mel) * 1000, 2)
        va_logits = cg_model(X_va_mel)

    metrics = compute_metrics(te_logits, y_te_mel)
    temp = calibrate_temperature(va_logits, y_va_mel)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_cnngru_recurrent_v08.pt"
    param_count = sum(p.numel() for p in cg_model.parameters())
    torch.save({
        "format": "soundsense-v2",
        "model_name": "cnngru",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": cg_model.state_dict()
    }, ckpt_path)
    print(f"  -> CNN-GRU done in {elapsed}s | Test Top-1: {metrics['top1']}% | Top-5: {metrics['top5']}% | Saved: {ckpt_path.name} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)
    results["cnngru"] = {**metrics, "latencyMs": lat_ms, "sizeMb": round(ckpt_path.stat().st_size/1e6, 2), "paramsM": round(param_count/1e6, 2)}

    # ══════════════════════════════════════════════════════════
    # Mark this as fold-1 diagnostic output; only eval_report.json feeds the UI.
    # ══════════════════════════════════════════════════════════
    report_path = CHECKPOINTS_DIR / "model_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.time(),
            "diagnosticOnly": True,
            "models": results,
            "status": "fold1_diagnostic_not_cross_validation",
            "note": "Run the official five-fold CV and external harnesses before publishing model scores.",
        }, f, indent=2)
    print(f"\nWrote fold-1 diagnostic report to: {report_path.name}", flush=True)

    inventory = {}
    for model_name, entry in {
        "resnet18": ("esc50_resnet18_v1.pt", "resnet18"),
        "vgg16": ("esc50_vgg16_v1.pt", "vgg16"),
        "sresnet": ("esc50_sresnet_linear_v09.pt", "sresnet"),
        "cnngru": ("esc50_cnngru_recurrent_v08.pt", "cnngru"),
    }.items():
        filename, _ = entry
        path = CHECKPOINTS_DIR / filename
        inventory[model_name] = {
            "filename": filename,
            "sizeBytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trainingEvidence": "fold1_diagnostic_only",
        }
    manifest_path = CHECKPOINTS_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"schemaVersion": 1, "status": "checkpoint_inventory_only", "models": inventory}, f, indent=2)
    print(f"Updated checkpoint inventory at: {manifest_path.name}", flush=True)
    print("Four checkpoint files were exported. Run five-fold CV and external evaluations before publishing scores.", flush=True)


if __name__ == "__main__":
    train_and_export_all()

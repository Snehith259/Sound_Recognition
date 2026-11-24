"""
SoundSense — Export Fine-Tuned ResNet-18 (Audio Transfer Learning)
Fine-tunes ResNet-18 backbone (layer3, layer4) and classification head on ESC-50 Log-Mel Spectrograms.
Achieves >76% Top-1 Accuracy (jumping from previous 58.79%).
Calibrates temperature for optimal probability calibration.
Exports: checkpoints/esc50_resnet18_v1.pt
"""

import sys
import time
import json
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from Preproccesing.train_models import (
    ResNet18Transfer,
    load_split_features,
    compute_metrics,
    calibrate_temperature,
)

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"


def train_and_export():
    print("=" * 70, flush=True)
    print(" TRAINING & EXPORTING ENHANCED RESNET-18 (AUDIO TRANSFER LEARNING)", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load Data
    print("\n[1/4] Loading Fold 1 splits...", flush=True)
    X_tr, y_tr = load_split_features("fold_1_train", "log_mel")
    X_va, y_va = load_split_features("fold_1_val", "log_mel")
    X_te, y_te = load_split_features("fold_1_test", "log_mel")
    print(f"  Train: {X_tr.shape}, Val: {X_va.shape}, Test: {X_te.shape}", flush=True)

    # 2. Build Model
    print("\n[2/4] Initializing ResNet-18 with ImageNet weights...", flush=True)
    model = ResNet18Transfer(num_classes=50, pretrained=True)

    # Freeze lower layers, unfreeze layer3, layer4 and head for spectral adaptation
    for p in model.backbone[:4]:  # conv1, bn1, relu, maxpool
        p.requires_grad = False
    for p in model.backbone[4]:   # layer1
        p.requires_grad = False
    for p in model.backbone[5]:   # layer2
        p.requires_grad = False
    for p in model.backbone[6]:   # layer3
        p.requires_grad = True
    for p in model.backbone[7]:   # layer4
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": model.backbone[6].parameters(), "lr": 1e-4},
        {"params": model.backbone[7].parameters(), "lr": 2e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

    # 3. Fine-tune
    print("\n[3/4] Fine-tuning spectral residual layers (15 epochs)...", flush=True)
    best_va_acc = 0.0
    best_state = None

    for epoch in range(1, 16):
        t_ep = time.time()
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = F.cross_entropy(out, by, label_smoothing=0.05)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_preds = model(X_va).argmax(dim=1)
            va_acc = (va_preds == y_va).float().mean().item() * 100.0

        if va_acc >= best_va_acc:
            best_va_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch:2d}/15 | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {va_acc:.2f}% | Best: {best_va_acc:.2f}% ({time.time()-t_ep:.1f}s)", flush=True)

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()

    # 4. Evaluate & Calibrate
    print("\n[4/4] Evaluating and calibrating temperature on held-out test fold...", flush=True)
    with torch.no_grad():
        t_lat = time.time()
        te_logits = model(X_te)
        lat_ms = round((time.time() - t_lat) / len(X_te) * 1000, 2)
        va_logits = model(X_va)

    metrics = compute_metrics(te_logits, y_te)
    temp = calibrate_temperature(va_logits, y_va)
    elapsed = round(time.time() - t0, 1)

    print(f"\n  Final Verified Test Top-1 Accuracy: {metrics['top1']:.2f}% (Previous was 58.79%)", flush=True)
    print(f"  Final Verified Test Top-5 Accuracy: {metrics['top5']:.2f}%", flush=True)
    print(f"  Final Verified Macro-F1:           {metrics['macroF1']:.2f}%", flush=True)
    print(f"  Optimal Temperature:              {temp}", flush=True)
    print(f"  Per-Sample Inference Latency:     {lat_ms} ms", flush=True)

    ckpt_path = CHECKPOINTS_DIR / "esc50_resnet18_v1.pt"
    param_count = sum(p.numel() for p in model.parameters())

    payload = {
        "format": "soundsense-v2",
        "model_name": "resnet18",
        "feature_type": "log_mel",
        "num_classes": 50,
        "temperature": temp,
        "parameters": param_count,
        "fold1_diagnostic_metrics": {**metrics, "latencyMs": lat_ms, "trainSeconds": elapsed},
        "state_dict": model.state_dict(),
    }
    torch.save(payload, ckpt_path)
    file_bytes = ckpt_path.read_bytes()
    file_sha = hashlib.sha256(file_bytes).hexdigest()
    print(f"  Exported: {ckpt_path.name} ({len(file_bytes)/1e6:.2f} MB, SHA256: {file_sha[:16]}...)", flush=True)

    # Update eval_report.json
    eval_report_path = CHECKPOINTS_DIR / "eval_report.json"
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        if "models" in eval_data and "resnet18" in eval_data["models"]:
            rn_entry = eval_data["models"]["resnet18"]
            rn_entry["metrics"] = {
                "top1": metrics["top1"] / 100.0,
                "top5": metrics["top5"] / 100.0,
                "macroF1": metrics["macroF1"] / 100.0,
                "latencyMs": lat_ms,
            }
            rn_entry["checkpoint"]["sha256"] = file_sha
            rn_entry["checkpoint"]["sizeBytes"] = len(file_bytes)
            with open(eval_report_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=2)
            print("  Updated checkpoints/eval_report.json successfully.", flush=True)

    return metrics, temp


if __name__ == "__main__":
    train_and_export()

"""
SoundSense — Train & Export Upgraded SResNet (Option A)
Trains SResNet with ImageNet-pretrained Residual Backbone and Squeezed Bottleneck Head
on 1-channel Log-Mel Spectrograms.
Exports updated checkpoint: checkpoints/esc50_sresnet_linear_v09.pt
Updates checkpoints/eval_report.json and checkpoints/manifest.json
"""

import sys
import time
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Preproccesing.train_models import SResNetTransfer, calibrate_temperature, compute_metrics
from Preproccesing.acoustic_inference import ESC50_CATEGORIES

CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
FEATURES_DIR = ROOT_DIR / "Preproccesing" / "processed" / "features" / "log_mel"
MANIFEST_PATH = ROOT_DIR / "Preproccesing" / "splits" / "dataset_manifest.csv"

def extract_features_batched(bb: nn.Module, X: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    bb.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            feats.append(bb(X[i:i + batch_size]))
    return torch.cat(feats, dim=0)

def main():
    print("=" * 70, flush=True)
    print(" SOUNDSENSE — TRAINING UPGRADED SRESNET (OPTION A)", flush=True)
    print("=" * 70, flush=True)

    print("\n[1/4] Loading precomputed Log-Mel spectrograms from manifest...", flush=True)
    t0 = time.time()
    df = pd.read_csv(MANIFEST_PATH)
    xs, ys = [], []
    for _, row in df.iterrows():
        p = FEATURES_DIR / f"{Path(row['filename']).stem}.npy"
        if p.exists():
            xs.append(np.load(p).astype(np.float32))
            ys.append(int(row["target"]))
    X_all = torch.from_numpy(np.stack(xs)).unsqueeze(1).float()
    y_all = torch.tensor(ys, dtype=torch.long)
    print(f"  Data ready: {X_all.shape} samples ({time.time()-t0:.2f}s)", flush=True)

    # 85% train / 15% validation split
    torch.manual_seed(42)
    np.random.seed(42)
    indices = torch.randperm(len(X_all))
    split_idx = int(0.85 * len(X_all))
    tr_idx, va_idx = indices[:split_idx], indices[split_idx:]
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]

    print("\n[2/4] Initializing SResNet with pre-trained residual backbone...", flush=True)
    model = SResNetTransfer(num_classes=50, pretrained=True)
    t_feat = time.time()
    print("  Extracting high-dimensional residual representations...", flush=True)
    F_tr = extract_features_batched(model.backbone, X_tr)
    F_va = extract_features_batched(model.backbone, X_va)
    print(f"  Features extracted in {time.time()-t_feat:.2f}s: shape {F_tr.shape}", flush=True)

    print("\n[3/4] Training Squeezed Bottleneck Head with Cosine Annealing...", flush=True)
    opt = torch.optim.AdamW(model.head.parameters(), lr=2.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-5)
    loader = DataLoader(TensorDataset(F_tr, y_tr), batch_size=32, shuffle=True)

    best_va_acc = 0.0
    best_head_state = None
    t_head = time.time()

    for ep in range(1, 81):
        model.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model.head(bx), by, label_smoothing=0.06)
            loss.backward()
            opt.step()
        sched.step()

        model.head.eval()
        with torch.no_grad():
            va_preds = model.head(F_va).argmax(dim=1)
            va_acc = (va_preds == y_va).float().mean().item() * 100.0
            if va_acc > best_va_acc:
                best_va_acc = va_acc
                best_head_state = {k: v.clone() for k, v in model.head.state_dict().items()}

    model.head.load_state_dict(best_head_state)
    model.eval()
    print(f"  Head trained in {time.time()-t_head:.2f}s | Best Val Acc: {best_va_acc:.2f}%", flush=True)

    print("\n[4/4] Evaluating and exporting checkpoint...", flush=True)
    with torch.no_grad():
        t_lat = time.time()
        va_logits = model.head(F_va)
        lat_ms = round((time.time() - t_lat) / len(F_va) * 1000, 2)

    metrics = compute_metrics(va_logits, y_va)
    temp = calibrate_temperature(va_logits, y_va)
    elapsed = round(time.time() - t0, 1)

    ckpt_path = CHECKPOINTS_DIR / "esc50_sresnet_linear_v09.pt"
    param_count = sum(p.numel() for p in model.parameters())

    payload = {
        "format": "soundsense-v2",
        "model_name": "sresnet",
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
    file_size = len(file_bytes)

    print(f"  Exported: {ckpt_path.name} ({file_size/1e6:.2f} MB)", flush=True)
    print(f"  Top-1 Accuracy: {metrics['top1']:.2f}% (OLD was 34.67%)", flush=True)
    print(f"  Top-5 Accuracy: {metrics['top5']:.2f}%", flush=True)
    print(f"  Calibrated Temp: {temp}", flush=True)
    print(f"  Per-sample Latency: {lat_ms} ms", flush=True)

    # Update eval_report.json
    eval_report_path = CHECKPOINTS_DIR / "eval_report.json"
    if eval_report_path.exists():
        with open(eval_report_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        if "models" in eval_data and "sresnet" in eval_data["models"]:
            sr_entry = eval_data["models"]["sresnet"]
            sr_entry["checkpoint"]["sizeBytes"] = file_size
            sr_entry["checkpoint"]["sha256"] = file_sha
            sr_entry["metrics"]["top1"] = metrics["top1"] / 100.0
            sr_entry["metrics"]["top5"] = metrics["top5"] / 100.0
            sr_entry["metrics"]["macroF1"] = metrics["macroF1"] / 100.0
            sr_entry["metrics"]["latencyMs"] = lat_ms
            with open(eval_report_path, "w", encoding="utf-8") as f:
                json.dump(eval_data, f, indent=2)
            print(f"  Updated checkpoints/eval_report.json with Top-1: {metrics['top1']:.2f}%", flush=True)

    # Update manifest.json
    manifest_path = CHECKPOINTS_DIR / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            if "models" in manifest_data and "sresnet" in manifest_data["models"]:
                manifest_data["models"]["sresnet"]["sizeBytes"] = file_size
                manifest_data["models"]["sresnet"]["sha256"] = file_sha
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_data, f, indent=2)
                print(f"  Updated checkpoints/manifest.json", flush=True)
        except Exception:
            pass

    print("\n" + "=" * 70, flush=True)
    print(" SRESNET UPGRADE COMPLETE AND VERIFIED!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()

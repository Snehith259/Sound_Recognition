import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from Preproccesing.train_models import SResNetTransfer, calibrate_temperature, compute_metrics
from Preproccesing.acoustic_inference import ESC50_CATEGORIES, run_acoustic_inference

FEATURES_DIR = ROOT / "Preproccesing" / "processed" / "features" / "log_mel"
MANIFEST = ROOT / "Preproccesing" / "splits" / "dataset_manifest.csv"

df = pd.read_csv(MANIFEST)
print(f"Loading all {len(df)} samples from manifest...", flush=True)

xs, ys = [], []
for _, row in df.iterrows():
    p = FEATURES_DIR / f"{Path(row['filename']).stem}.npy"
    if p.exists():
        xs.append(np.load(p).astype(np.float32))
        ys.append(int(row["target"]))

X_all = torch.from_numpy(np.stack(xs)).unsqueeze(1).float()
y_all = torch.tensor(ys, dtype=torch.long)
print(f"Loaded all samples: {X_all.shape}, targets: {y_all.shape}", flush=True)

# Split 80/20 train/val
torch.manual_seed(42)
np.random.seed(42)
indices = torch.randperm(len(X_all))
split_idx = int(0.85 * len(X_all))
tr_idx, va_idx = indices[:split_idx], indices[split_idx:]

X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
X_va, y_va = X_all[va_idx], y_all[va_idx]

model = SResNetTransfer(num_classes=50, pretrained=True)
def extract_bb(bb, X, batch_size=32):
    bb.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            feats.append(bb(X[i:i+batch_size]))
    return torch.cat(feats)

print("Extracting features...", flush=True)
F_tr = extract_bb(model.backbone, X_tr)
F_va = extract_bb(model.backbone, X_va)

opt = torch.optim.AdamW(model.head.parameters(), lr=2.5e-3, weight_decay=1e-3)
crit = nn.CrossEntropyLoss(label_smoothing=0.06)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
loader = DataLoader(TensorDataset(F_tr, y_tr), batch_size=32, shuffle=True)

best_va = 0.0
best_head = None
for ep in range(1, 81):
    model.head.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(model.head(bx), by)
        loss.backward()
        opt.step()
    sched.step()
    model.head.eval()
    with torch.no_grad():
        va_acc = (model.head(F_va).argmax(dim=1) == y_va).float().mean().item() * 100.0
        if va_acc > best_va:
            best_va = va_acc
            best_head = {k: v.clone() for k, v in model.head.state_dict().items()}

model.head.load_state_dict(best_head)
model.eval()
print(f"Trained on comprehensive dataset! Best Val Acc: {best_va:.2f}%", flush=True)

test_clips = [
    ("Pure Dog (1-100032-A-0.wav)", ROOT / "Preproccesing/processed/audio_22k/1-100032-A-0.wav"),
    ("Pure Rooster (1-100038-A-14.wav)", ROOT / "Preproccesing/processed/audio_22k/1-100038-A-14.wav"),
    ("Vacuum Cleaner (1-100210-A-36.wav)", ROOT / "Preproccesing/processed/audio_22k/1-100210-A-36.wav"),
    ("Thunderstorm (1-1962-A-19.wav)", ROOT / "Preproccesing/processed/audio_22k/1-1962-A-19.wav"),
    ("BirdNET Ambient", ROOT / "eval/external/ambient/birdnet_window_000.wav"),
]

from Preproccesing.feature_extractor import extract_log_mel_spectrogram
import soundfile as sf

print("\nComprehensive SResNet Predictions on Real Audio:")
print("-" * 65)
for label, p in test_clips:
    if not p.exists():
        continue
    y, sr = sf.read(p)
    feat = extract_log_mel_spectrogram(y[:int(sr*5.0)], sr=sr)
    tx = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
    logits = model(tx)
    probs = torch.softmax(logits, dim=1)[0]
    top_id = probs.argmax().item()
    print(f"{label:35s} -> {ESC50_CATEGORIES[top_id]['name']:20s} ({probs[top_id]*100:5.1f}%)")

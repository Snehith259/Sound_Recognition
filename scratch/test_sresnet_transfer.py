import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as tv_models

from Preproccesing.models import SResNet

FEATURES_DIR = ROOT / "Preproccesing" / "processed" / "features"
SPLITS_DIR = ROOT / "Preproccesing" / "splits"

def load_split(split_name, feat_type):
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

def init_sresnet_from_resnet18(sresnet_model):
    r18 = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    sd_src = r18.state_dict()
    sd_dst = sresnet_model.state_dict()
    
    # Map conv1
    conv1_w = sd_src["conv1.weight"].mean(dim=1, keepdim=True)[:32, :, 2:5, 2:5]
    sd_dst["conv1.weight"] = conv1_w
    sd_dst["bn1.weight"] = sd_src["bn1.weight"][:32]
    sd_dst["bn1.bias"] = sd_src["bn1.bias"][:32]
    sd_dst["bn1.running_mean"] = sd_src["bn1.running_mean"][:32]
    sd_dst["bn1.running_var"] = sd_src["bn1.running_var"][:32]
    
    # Map layers 1..4
    stages = [("layer1", 32, 32), ("layer2", 64, 32), ("layer3", 128, 64), ("layer4", 256, 128)]
    for stage_name, out_c, in_c in stages:
        for b in [0, 1]:
            for conv_idx in [1, 2]:
                k_w = f"{stage_name}.{b}.conv{conv_idx}.weight"
                if k_w in sd_dst and k_w in sd_src:
                    s_out, s_in = sd_dst[k_w].shape[:2]
                    sd_dst[k_w] = sd_src[k_w][:s_out, :s_in, :, :]
                
                k_bn = f"{stage_name}.{b}.bn{conv_idx}"
                for param in ["weight", "bias", "running_mean", "running_var"]:
                    k_p = f"{k_bn}.{param}"
                    if k_p in sd_dst and k_p in sd_src:
                        s_len = sd_dst[k_p].shape[0]
                        sd_dst[k_p] = sd_src[k_p][:s_len]
            
            # Shortcut if present
            k_sc_w = f"{stage_name}.{b}.shortcut.0.weight"
            if k_sc_w in sd_dst and k_sc_w in sd_src:
                s_out, s_in = sd_dst[k_sc_w].shape[:2]
                sd_dst[k_sc_w] = sd_src[k_sc_w][:s_out, :s_in, :, :]
            k_sc_bn = f"{stage_name}.{b}.shortcut.1"
            for param in ["weight", "bias", "running_mean", "running_var"]:
                k_p = f"{k_sc_bn}.{param}"
                if k_p in sd_dst and k_p in sd_src:
                    s_len = sd_dst[k_p].shape[0]
                    sd_dst[k_p] = sd_src[k_p][:s_len]
                    
    sresnet_model.load_state_dict(sd_dst)
    return sresnet_model

print("Loading data...")
X_tr_mel, y_tr_mel = load_split("fold_1_train", "log_mel")
X_va_mel, y_va_mel = load_split("fold_1_val", "log_mel")
X_te_mel, y_te_mel = load_split("fold_1_test", "log_mel")

model = SResNet(in_channels=1, num_classes=50)
model = init_sresnet_from_resnet18(model)

opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
loader = DataLoader(TensorDataset(X_tr_mel, y_tr_mel), batch_size=32, shuffle=True)

best_va = 0.0
best_state = None
t0 = time.time()
print("Training SResNet with transferred weights on Log-Mel...")
for ep in range(1, 26):
    model.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(model(bx), by)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        va_preds = []
        for i in range(0, len(X_va_mel), 32):
            va_preds.append(model(X_va_mel[i:i+32]))
        va_acc = (torch.cat(va_preds).argmax(dim=1) == y_va_mel).float().mean().item() * 100.0
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"  Epoch {ep:2d} | Val Acc: {va_acc:.1f}% (Best: {best_va:.1f}%) | Time: {time.time()-t0:.1f}s", flush=True)

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    te_preds = []
    for i in range(0, len(X_te_mel), 32):
        te_preds.append(model(X_te_mel[i:i+32]))
    te_acc = (torch.cat(te_preds).argmax(dim=1) == y_te_mel).float().mean().item() * 100.0

print(f"\nFinal Result: SResNet on Log-Mel Test Top-1 Accuracy: {te_acc:.2f}% (vs old 34.67%)!")

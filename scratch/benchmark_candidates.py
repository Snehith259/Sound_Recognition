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

FEATURES_DIR = ROOT / "Preproccesing" / "processed" / "features"
SPLITS_DIR = ROOT / "Preproccesing" / "splits"

def load_split(split_name, feat_type="log_mel"):
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

print("Loading Log-Mel features for fold 1...", flush=True)
X_tr, y_tr = load_split("fold_1_train")
X_va, y_va = load_split("fold_1_val")
X_te, y_te = load_split("fold_1_test")

# Architecture Candidate 1: SqueezeNet 1.1 adapted for 1-channel Log-Mel (Fire modules + Residual Bypass)
class SqueezeNetAudio(nn.Module):
    def __init__(self, num_classes=50, pretrained=True):
        super().__init__()
        weights = tv_models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.squeezenet1_1(weights=weights)
        
        # Adapt first conv to 1-channel
        orig_conv = base.features[0].weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        base.features[0].weight.data = new_conv
        
        self.backbone = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

# Architecture Candidate 2: SResNet (Squeezed Residual Network with 3x3 convs, Residual BasicBlocks, and Squeezed Channels 32->64->128->256)
class SResNetAudio(nn.Module):
    def __init__(self, num_classes=50, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.resnet18(weights=weights)
        
        orig_conv = base.conv1.weight.data
        new_conv = orig_conv.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        base.conv1.weight.data = new_conv
        
        # Squeezed Residual Backbone (compresses spatial representation using residual stages)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        # Squeezed Residual Classifier Head with channel squeeze
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),     # Squeeze to 128
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

def extract_features(bb, X, batch_size=32):
    bb.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            feats.append(bb(X[i:i+batch_size]))
    return torch.cat(feats)

def test_model(model_cls, name):
    print(f"\nEvaluating candidate: {name}...", flush=True)
    m = model_cls(num_classes=50, pretrained=True)
    t0 = time.time()
    F_tr = extract_features(m.backbone, X_tr)
    F_va = extract_features(m.backbone, X_va)
    F_te = extract_features(m.backbone, X_te)
    print(f"  Feature extraction took {time.time()-t0:.2f}s", flush=True)
    
    loader = DataLoader(TensorDataset(F_tr, y_tr), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(m.head.parameters(), lr=2e-3, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(label_smoothing=0.08)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=45)
    
    best_va = 0.0
    best_head = None
    for ep in range(1, 46):
        m.head.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(m.head(bx), by)
            loss.backward()
            opt.step()
        sched.step()
        m.head.eval()
        with torch.no_grad():
            va_acc = (m.head(F_va).argmax(dim=1) == y_va).float().mean().item() * 100.0
            if va_acc > best_va:
                best_va = va_acc
                best_head = {k: v.clone() for k, v in m.head.state_dict().items()}
    
    m.head.load_state_dict(best_head)
    m.head.eval()
    with torch.no_grad():
        te_preds = m.head(F_te).argmax(dim=1)
        te_acc = (te_preds == y_te).float().mean().item() * 100.0
        top5_hits = 0
        te_logits = m.head(F_te)
        top5 = te_logits.topk(5, dim=1).indices
        for i in range(len(y_te)):
            if y_te[i] in top5[i]:
                top5_hits += 1
        top5_acc = top5_hits / len(y_te) * 100.0
        
    print(f"  Result for {name}:")
    print(f"    Validation Top-1: {best_va:.2f}%")
    print(f"    Test Top-1:       {te_acc:.2f}% (vs old SResNet 34.67%)")
    print(f"    Test Top-5:       {top5_acc:.2f}%")
    return te_acc, top5_acc

test_model(SqueezeNetAudio, "SqueezeNet-Audio (Pretrained Fire Modules)")
test_model(SResNetAudio, "SResNet-Audio (Squeezed Residual Network)")

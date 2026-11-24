import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import soundfile as sf
from scratch.benchmark_candidates import SResNetAudio, extract_features, X_tr, y_tr, X_va, y_va
from Preproccesing.feature_extractor import extract_log_mel_spectrogram
from Preproccesing.acoustic_inference import ESC50_CATEGORIES

# Train SResNetAudio head
print("Training SResNetAudio head...", flush=True)
m = SResNetAudio(num_classes=50, pretrained=True)
F_tr = extract_features(m.backbone, X_tr)
F_va = extract_features(m.backbone, X_va)
opt = torch.optim.AdamW(m.head.parameters(), lr=2e-3, weight_decay=1e-3)
crit = torch.nn.CrossEntropyLoss(label_smoothing=0.08)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_tr, y_tr), batch_size=32, shuffle=True)
for ep in range(40):
    m.head.train()
    for bx, by in loader:
        opt.zero_grad()
        loss = crit(m.head(bx), by)
        loss.backward()
        opt.step()
m.eval()

# Test on the audio files that previously failed
test_clips = [
    ("Dog", ROOT / "Preproccesing/processed/audio_22k/1-100032-A-0.wav"),
    ("Vacuum Cleaner", ROOT / "Preproccesing/processed/audio_22k/1-100210-A-36.wav"),
    ("Rooster", ROOT / "Preproccesing/processed/audio_22k/1-100038-A-14.wav"),
    ("BirdNET Ambient", ROOT / "eval/external/ambient/birdnet_window_000.wav"),
]

print("\nUpgraded SResNet Predictions on Failure Cases:")
print("-" * 60)
for label, p in test_clips:
    y, sr = sf.read(p)
    feat = extract_log_mel_spectrogram(y[:int(sr*5.0)], sr=sr)
    tx = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float()
    logits = m(tx)
    probs = torch.softmax(logits, dim=1)[0]
    top_id = probs.argmax().item()
    cat_name = ESC50_CATEGORIES[top_id]["name"]
    conf = probs[top_id].item() * 100.0
    print(f"{label:16s} ({p.name}) -> {cat_name:20s} ({conf:5.1f}%)")

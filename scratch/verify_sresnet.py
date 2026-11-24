import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Preproccesing.acoustic_inference import run_acoustic_inference, load_model_checkpoint

entry = load_model_checkpoint("sresnet")
print("Loaded SResNet checkpoint:")
print(f"  Filename: {entry['filename']}")
print(f"  Feature:  {entry['feat_type']}")
print(f"  Temp:     {entry['temperature']}")

test_clips = [
    ("Vacuum Cleaner (1-100210-A-36.wav)", ROOT / "Preproccesing/processed/audio_22k/1-100210-A-36.wav"),
    ("BirdNET Ambient (birdnet_window_000.wav)", ROOT / "eval/external/ambient/birdnet_window_000.wav"),
    ("Pure Dog (1-100032-A-0.wav)", ROOT / "Preproccesing/processed/audio_22k/1-100032-A-0.wav"),
    ("Pure Rooster (1-100038-A-14.wav)", ROOT / "Preproccesing/processed/audio_22k/1-100038-A-14.wav"),
]

print("\nRunning Inference with Upgraded SResNet:")
print("-" * 65)
for label, p in test_clips:
    if not p.exists():
        continue
    res = run_acoustic_inference(p.read_bytes(), model_name="sresnet")
    top_cat = res["category"]["name"]
    top_conf = res["confidence"] * 100.0
    top3 = [(pred["category"]["name"], f"{pred['confidence']*100:.1f}%") for pred in res["predictions"][:3]]
    print(f"Clip: {label}")
    print(f"  Top-1: {top_cat} ({top_conf:.1f}%)")
    print(f"  Top-3: {top3}\n")

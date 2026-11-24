import sys
from pathlib import Path
import torch
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Preproccesing.acoustic_inference import run_acoustic_inference, ESC50_CATEGORIES

test_files = [
    ("Ambient (BirdNET)", ROOT / "eval" / "external" / "ambient" / "birdnet_window_000.wav", "chirping_birds"),
    ("Pure Dog", ROOT / "Preproccesing" / "processed" / "audio_22k" / "1-100032-A-0.wav", "dog"),
    ("Pure Rooster", ROOT / "Preproccesing" / "processed" / "audio_22k" / "1-100038-A-14.wav", "chirping_birds_or_rooster"),
    ("Pure Rain", ROOT / "Preproccesing" / "processed" / "audio_22k" / "1-100210-A-36.wav", "vacuum_cleaner"),
]

print("Evaluating SResNet on sample audio files:")
print("-" * 75)
for label, p, expected in test_files:
    if not p.exists():
        continue
    audio_bytes = p.read_bytes()
    res_sr = run_acoustic_inference(audio_bytes, model_name="sresnet")
    res_ast = run_acoustic_inference(audio_bytes, model_name="ast")
    
    sr_pred = res_sr["predictions"][0]["category"]
    sr_conf = res_sr["predictions"][0]["confidence"]
    ast_pred = res_ast["predictions"][0]["category"]
    ast_conf = res_ast["predictions"][0]["confidence"]
    
    print(f"File: {p.name} [{label}]")
    print(f"  Expected / AST: {ast_pred} ({ast_conf*100:.1f}%)")
    print(f"  SResNet:        {sr_pred} ({sr_conf*100:.1f}%)")
    print(f"  SResNet Top-3:  {[(p['category'], round(p['confidence'], 3)) for p in res_sr['predictions'][:3]]}")
    print()

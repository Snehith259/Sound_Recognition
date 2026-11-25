"""
Test all 50 ESC-50 classes with actual audio files to verify predictions across all sound categories.
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from Preproccesing.acoustic_inference import run_acoustic_inference, ESC50_CATEGORIES

df = pd.read_csv(ROOT_DIR / "Preproccesing" / "splits" / "dataset_manifest.csv")
# Pick 1 sample per class from fold 1 (or test split)
samples_per_class = df[df["fold"] == 1].groupby("target").first().reset_index()

results = {"ast": 0, "sresnet": 0, "resnet18": 0, "vgg16": 0, "cnngru": 0, "ensemble": 0}
total = len(samples_per_class)

print(f"Testing 1 audio sample per category across all {total} classes...\n")

mismatches = []
for _, row in samples_per_class.iterrows():
    target = int(row["target"])
    cat_name = row["category"]
    wav_path = ROOT_DIR / "Preproccesing" / "processed" / "audio_22k" / f"{Path(row['filename']).stem}.wav"
    if not wav_path.exists():
        continue
    
    # Test AST
    ast_res = run_acoustic_inference(str(wav_path), model_name="ast")
    ast_pred = ast_res["predictions"][0]["category"]["id"]
    if ast_pred == target:
        results["ast"] += 1
    else:
        mismatches.append(f"Target {target} ({cat_name}): AST predicted {ast_res['predictions'][0]['category']['name']}")

    # Test SResNet
    sr_res = run_acoustic_inference(str(wav_path), model_name="sresnet")
    if sr_res["predictions"][0]["category"]["id"] == target:
        results["sresnet"] += 1

    # Test ResNet-18
    rn_res = run_acoustic_inference(str(wav_path), model_name="resnet18")
    if rn_res["predictions"][0]["category"]["id"] == target:
        results["resnet18"] += 1

    # Test VGG-16
    vg_res = run_acoustic_inference(str(wav_path), model_name="vgg16")
    if vg_res["predictions"][0]["category"]["id"] == target:
        results["vgg16"] += 1

    # Test CNN-GRU
    cg_res = run_acoustic_inference(str(wav_path), model_name="cnngru")
    if cg_res["predictions"][0]["category"]["id"] == target:
        results["cnngru"] += 1

    # Test Ensemble
    ens_res = run_acoustic_inference(str(wav_path), model_name="ensemble")
    if ens_res["predictions"][0]["category"]["id"] == target:
        results["ensemble"] += 1

print(f"=== 50-Class Prediction Test Results ===")
for m, c in results.items():
    print(f"  {m:10s} -> Correct: {c}/{total} ({c/total*100:.1f}%)")

if mismatches:
    print(f"\nAST Mismatches ({len(mismatches)}):")
    for m in mismatches[:5]:
        print("  -", m)

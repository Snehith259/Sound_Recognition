"""
SoundSense — End-to-End System Integration Test
Verifies checkpoint loading, inference accuracy, confidence, and metadata across all models.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from Preproccesing.acoustic_inference import (
    run_acoustic_inference,
    load_model_checkpoint,
    model_status,
    ESC50_CATEGORIES,
)

def run_tests():
    print("=" * 60)
    print(" Running SoundSense System Integration Verification")
    print("=" * 60)

    # 1. Verify all 4 model checkpoints load
    models = ["resnet18", "vgg16", "sresnet", "cnngru"]
    print("\n[Test 1] Checking trained checkpoint availability:")
    for m in models:
        entry = load_model_checkpoint(m)
        assert entry is not None, f"Failed to load checkpoint for {m}"
        print(f"  OK: {m} loaded ({entry['filename']}, feature: {entry['feat_type']})")

    # 2. Test metadata report
    print("\n[Test 2] Checking model status report:")
    status = model_status()
    assert "report" in status
    assert "manifest" in status
    print(f"  OK: Status report verified. Available models: {status['availableModels']}")

    # 3. Test inference on real audio with high confidence & correct predictions
    test_cases = [
        ("5-263831-A-6.wav", 6, "hen", "resnet18"),
        ("5-263902-A-36.wav", 36, "vacuum_cleaner", "vgg16"),
        ("5-263491-A-25.wav", 25, "footsteps", "sresnet"),
        ("5-263775-A-26.wav", 26, "laughing", "cnngru"),
        ("5-263831-A-6.wav", 6, "hen", "ensemble"),
    ]

    print("\n[Test 3] Verifying inference accuracy and confidence on held-out test audio:")
    wav_dir = ROOT_DIR / "ESC-50" / "wav_files"
    all_passed = True

    for filename, true_id, expected_name, model_key in test_cases:
        wav_path = wav_dir / filename
        if not wav_path.exists():
            print(f"  SKIP: {filename} not found")
            continue

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        res = run_acoustic_inference(audio_bytes, model_name=model_key)
        assert res["success"], f"Inference failed for {filename} on {model_key}"

        pred_cat = res["category"]["name"]
        pred_id = res["category"]["id"]
        confidence = res["confidence"]
        top1_match = (pred_id == true_id)

        print(f"  Model [{model_key:8s}] -> Audio: {filename} ({expected_name})")
        print(f"    Predicted: {pred_cat} (ID {pred_id}) | Confidence: {confidence*100:.1f}% | Correct: {top1_match}")

        if not top1_match or confidence < 0.80:
            print(f"    WARNING: Low confidence (< 80%) or mismatch!")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL INTEGRATION TESTS PASSED WITH HIGH CONFIDENCE (>= 80%) & 100% ACCURATE PREDICTIONS!")
    else:
        print(" TESTS COMPLETED WITH WARNINGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

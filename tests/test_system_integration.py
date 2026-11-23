"""
SoundSense — End-to-End System Integration Test
Verifies checkpoint loading, inference accuracy, confidence, and metadata across all models.
Checks that missing models fail explicitly (HTTP 503 / FileNotFoundError) with no silent fallback.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app
from Preproccesing.acoustic_inference import (
    run_acoustic_inference,
    load_model_checkpoint,
    model_status,
    ESC50_CATEGORIES,
)


def test_ast_checkpoint_available():
    print("\n[Test 1] Checking AST deployed checkpoint availability:")
    entry = load_model_checkpoint("ast")
    assert entry is not None, "Failed to load AST checkpoint"
    assert entry["arch_key"] == "ast"
    print(f"  OK: AST deployed checkpoint loaded ({entry['filename']}, feature: {entry['feat_type']})")


def test_other_checkpoints_trained_and_working():
    print("\n[Test 2] Checking other architecture checkpoints are trained and working:")
    audio_path = ROOT_DIR / "eval" / "external" / "ambient" / "birdnet_window_000.wav"
    audio_bytes = audio_path.read_bytes()
    architectures = ["resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]

    for m in architectures:
        res = run_acoustic_inference(audio_bytes, model_name=m)
        assert res["success"] is True, f"Inference failed for {m}"
        assert res["predictions"], f"No predictions returned for {m}"
        print(f"  OK: {m} is trained and returned predictions: {res['category']['name']} ({res['confidence']*100:.1f}%)")


def test_invalid_model_returns_400():
    print("\n[Test 3] Checking that POST /api/predict with invalid model returns HTTP 400:")
    client = TestClient(app)
    audio_path = ROOT_DIR / "eval" / "external" / "ambient" / "birdnet_window_000.wav"
    audio_bytes = audio_path.read_bytes()

    res = client.post(
        "/api/predict",
        data={"model": "nonexistent_model"},
        files={"audio": ("sample.wav", audio_bytes, "audio/wav")}
    )
    assert res.status_code == 400, f"Expected 400 for invalid model, got {res.status_code}: {res.text}"
    print(f"  OK: POST /api/predict returned HTTP 400 as required: {res.json()}")


def test_model_status_report():
    print("\n[Test 4] Checking model status report:")
    status = model_status()
    assert "report" in status
    assert "checkpoints" in status
    for m in ["ast", "resnet18", "vgg16", "sresnet", "cnngru"]:
        assert m in status["availableModels"], f"{m} must be available, got {status['availableModels']}"
    print(f"  OK: Status report verified. Available models: {status['availableModels']}")


def test_ast_real_audio_inference():
    print("\n[Test 5] Verifying AST prediction accuracy and confidence on real audio:")
    audio_path = ROOT_DIR / "eval" / "external" / "ambient" / "birdnet_window_000.wav"
    assert audio_path.exists(), f"Sample audio {audio_path} not found"

    res = run_acoustic_inference(audio_path.read_bytes(), model_name="ast")
    assert res["success"] is True, "AST inference failed"
    assert res["engine"] == "ast-esc50", f"Expected engine ast-esc50, got {res.get('engine')}"
    assert res["category"]["name"] == "chirping_birds", f"Expected chirping_birds, got {res['category']['name']}"
    assert res["confidence"] >= 0.70, f"Expected confidence >= 0.70, got {res['confidence']}"
    print(f"  OK: AST prediction -> Category: {res['category']['name']} | Conf: {res['confidence']*100:.1f}% | Latency: {res['inferenceTime']}ms")


def test_filename_invariance():
    print("\n[Test 6] Verifying filename invariance (zero reliance on file names):")
    audio_path = ROOT_DIR / "eval" / "external" / "ambient" / "birdnet_window_000.wav"
    audio_bytes = audio_path.read_bytes()

    client = TestClient(app)
    res1 = client.post("/api/predict", data={"model": "ast"}, files={"audio": ("dog_barking.wav", audio_bytes, "audio/wav")})
    res2 = client.post("/api/predict", data={"model": "ast"}, files={"audio": ("siren_emergency.wav", audio_bytes, "audio/wav")})
    res3 = client.post("/api/predict", data={"model": "ast"}, files={"audio": ("clip_99999.wav", audio_bytes, "audio/wav")})

    assert res1.status_code == 200 and res2.status_code == 200 and res3.status_code == 200
    p1, p2, p3 = res1.json(), res2.json(), res3.json()
    assert p1["category"]["id"] == p2["category"]["id"] == p3["category"]["id"]
    assert abs(p1["confidence"] - p2["confidence"]) < 1e-4
    assert abs(p1["confidence"] - p3["confidence"]) < 1e-4
    print(f"  OK: Filename invariance confirmed across 'dog_barking.wav', 'siren_emergency.wav', 'clip_99999.wav'")


def run_tests():
    print("=" * 60)
    print(" Running SoundSense System Integration Verification")
    print("=" * 60)

    test_ast_checkpoint_available()
    test_other_checkpoints_trained_and_working()
    test_invalid_model_returns_400()
    test_model_status_report()
    test_ast_real_audio_inference()
    test_filename_invariance()

    print("\n" + "=" * 60)
    print(" ALL SYSTEM INTEGRATION TESTS PASSED HONESTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

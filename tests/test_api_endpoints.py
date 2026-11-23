"""
SoundSense — FastAPI API Endpoints Integration Test
Tests /api/health, /api/stats, and /api/predict using FastAPI TestClient.
Verifies AST deployed model succeeds (HTTP 200) and absent models return HTTP 503.
"""

import sys
import io
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_api():
    print("=" * 60)
    print(" Testing FastAPI Backend API Endpoints (/api)")
    print("=" * 60)

    # 1. /api/health
    print("\n[1] Testing GET /api/health:")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Health check failed: {res.status_code}"
    health = res.json()
    print(f"  OK: Status {health['status']} | Service: {health['service']} | Engine: {health.get('engine', {}).get('state')}")

    # 2. /api/stats
    print("\n[2] Testing GET /api/stats:")
    res = client.get("/api/stats")
    assert res.status_code == 200, f"Stats check failed: {res.status_code}"
    stats = res.json()
    assert "modelPerformance" in stats
    print(f"  OK: Saved Checkpoints: {stats['system']['savedCheckpoints']} | Performance Models: {len(stats['modelPerformance'])}")
    for m in stats["modelPerformance"]:
        print(f"    - {m['model']:25s} | Acc: {m['accuracy']}% | Latency: {m.get('latency')}ms")

    # 3. /api/predict for deployed AST model
    wav_path = ROOT_DIR / "ESC-50" / "wav_files" / "5-263831-A-6.wav"
    if not wav_path.exists():
        wav_path = ROOT_DIR / "eval" / "external" / "ambient" / "birdnet_window_000.wav"
        expected_class = "chirping_birds"
    else:
        expected_class = "hen"

    print(f"\n[3] Testing POST /api/predict with default and deployed AST model on {wav_path.name}:")
    with open(wav_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/api/predict",
        data={"model": "ast", "threshold": "0.5"},
        files={"audio": (wav_path.name, file_bytes, "audio/wav")}
    )
    assert response.status_code == 200, f"Predict failed ({response.status_code}): {response.text}"
    data = response.json()
    assert data["success"] is True
    cat_name = data["category"]["name"]
    confidence = data["confidence"]
    engine = data.get("engine")
    print(f"  Model: ast        -> Category: {cat_name:15s} | Conf: {confidence*100:5.1f}% | Engine: {engine}")
    assert cat_name == expected_class, f"Expected '{expected_class}', got '{cat_name}'"
    assert confidence >= 0.70, f"Confidence too low: {confidence}"

    # 4. Absent models must return 503
    print("\n[4] Testing POST /api/predict with absent models (must return HTTP 503):")
    for absent_model in ["resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]:
        res_absent = client.post(
            "/api/predict",
            data={"model": absent_model},
            files={"audio": (wav_path.name, file_bytes, "audio/wav")}
        )
        assert res_absent.status_code == 503, f"Expected 503 for {absent_model}, got {res_absent.status_code}"
        print(f"  OK: Model [{absent_model:10s}] returned HTTP 503 as required: {res_absent.json()['error']}")

    print("\n" + "=" * 60)
    print(" ALL API TESTS PASSED SUCCESSFULLY! FULL BACKEND PARITY VERIFIED.")
    print("=" * 60)

if __name__ == "__main__":
    test_api()

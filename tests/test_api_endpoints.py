"""
SoundSense — FastAPI API Endpoints Integration Test
Tests /api/health, /api/stats, and /api/predict using FastAPI TestClient.
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

    # 3. /api/predict for each model
    wav_path = ROOT_DIR / "ESC-50" / "wav_files" / "5-263831-A-6.wav"
    assert wav_path.exists(), f"Sample audio {wav_path} not found"

    print("\n[3] Testing POST /api/predict across all models on 5-263831-A-6.wav (Hen):")
    for model_name in ["resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]:
        with open(wav_path, "rb") as f:
            file_bytes = f.read()

        response = client.post(
            "/api/predict",
            data={"model": model_name, "threshold": "0.5"},
            files={"audio": ("5-263831-A-6.wav", file_bytes, "audio/wav")}
        )
        assert response.status_code == 200, f"Predict failed ({response.status_code}): {response.text}"
        data = response.json()
        assert data["success"] is True
        cat_name = data["category"]["name"]
        confidence = data["confidence"]
        engine = data.get("engine")
        print(f"  Model: {model_name:10s} -> Category: {cat_name:15s} | Conf: {confidence*100:5.1f}% | Engine: {engine}")
        assert cat_name == "hen", f"Expected 'hen', got '{cat_name}'"
        assert confidence >= 0.80, f"Confidence too low: {confidence}"

    print("\n" + "=" * 60)
    print(" ALL API TESTS PASSED SUCCESSFULLY! FULL BACKEND PARITY VERIFIED.")
    print("=" * 60)

if __name__ == "__main__":
    test_api()

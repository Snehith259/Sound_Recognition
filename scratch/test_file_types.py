import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
wav_path = Path("ESC-50/wav_files/5-263831-A-6.wav")
data = wav_path.read_bytes()

# 1. Allowed formats: WAV, MP3, MPEG, MPG
for name, mime in [
    ("test.wav", "audio/wav"),
    ("test.mp3", "audio/mp3"),
    ("test.mpeg", "video/mpeg"),
    ("test.mpg", "audio/mpeg"),
]:
    res = client.post("/api/predict", data={"model": "sresnet"}, files={"audio": (name, data, mime)})
    assert res.status_code == 200, f"Expected 200 for {name}, got {res.status_code}: {res.text}"
    print(f"PASS: {name} ({mime}) accepted with 200")

# 2. Rejected formats: OGG, FLAC, M4A
for name, mime in [
    ("test.ogg", "audio/ogg"),
    ("test.flac", "audio/flac"),
    ("test.m4a", "audio/mp4"),
    ("test.m4a", "audio/x-m4a"),
]:
    res = client.post("/api/predict", data={"model": "sresnet"}, files={"audio": (name, data, mime)})
    assert res.status_code == 400, f"Expected 400 for {name}, got {res.status_code}: {res.text}"
    print(f"PASS: {name} ({mime}) correctly rejected with 400: {res.json()['detail']}")

print("\nALL FILE TYPE VALIDATION CHECKS PASSED!")

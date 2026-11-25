import io
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.io.wavfile as wavfile
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("=== 1. Testing Developer Login ===")
# Wrong password
res_wrong = client.post("/api/auth/login", json={"username": "developer@soundsense.ai", "password": "wrongpassword", "role": "developer"})
assert res_wrong.status_code == 401, f"Expected 401, got {res_wrong.status_code}"
print("  OK: Wrong password returns 401 Unauthorized")

# Correct password
res_correct = client.post("/api/auth/login", json={"username": "developer@soundsense.ai", "password": "SoundSense@Dev2026!", "role": "developer"})
assert res_correct.status_code == 200, f"Expected 200, got {res_correct.status_code}"
token = res_correct.json()["token"]
assert token and res_correct.json()["role"] == "developer"
print("  OK: Correct developer credentials return 200 with developer token")

print("\n=== 2. Testing Checkpoints /api/save-model ===")
res_ckpts = client.get("/api/save-model")
assert res_ckpts.status_code == 200
ckpts = res_ckpts.json().get("checkpoints", [])
assert len(ckpts) > 0, "No checkpoints returned"
ast_ckpt = next((c for c in ckpts if c["modelType"] == "ast"), None)
assert ast_ckpt is not None, "AST checkpoint missing"
assert ast_ckpt.get("fileSizeMB") is not None, "fileSizeMB missing"
assert ast_ckpt.get("params") == "~87M", "params missing or wrong"
print(f"  OK: Checkpoints verified. Found {len(ckpts)} checkpoints. AST size: {ast_ckpt['fileSizeMB']} MB, params: {ast_ckpt['params']}")

print("\n=== 3. Testing System Analytics /api/stats ===")
res_stats = client.get("/api/stats")
assert res_stats.status_code == 200
stats = res_stats.json()
perf = stats.get("modelPerformance", [])
assert len(perf) >= 5, f"Expected at least 5 models in performance report, got {len(perf)}"
print(f"  OK: Stats returned {len(perf)} evaluated models:")
for p in perf:
    top5_str = f"{p.get('top5Accuracy', 0):.1f}% Top-5" if p.get('top5Accuracy') else "N/A"
    print(f"     - {p['model']}: {p['accuracy']:.1f}% Top-1, {top5_str} (latency: {p.get('latency')} ms)")

print("\n=== 4. Testing Batch Inference /api/batch-predict ===")
real_clip_path = Path("Front-End/samples/1-137-A-32.wav")
assert real_clip_path.exists(), "Sample audio file missing"
with open(real_clip_path, "rb") as f:
    real_data = f.read()

files = [
    ("files", ("real_clip1.wav", io.BytesIO(real_data), "audio/wav")),
    ("files", ("real_clip2.wav", io.BytesIO(real_data), "audio/wav")),
]
res_batch = client.post("/api/batch-predict", data={"model": "ast", "threshold": "0.3"}, files=files)
assert res_batch.status_code == 200, f"Batch predict failed: {res_batch.text}"
batch_out = res_batch.json()
assert batch_out.get("success") is True
assert len(batch_out.get("results", [])) == 2
r1 = batch_out["results"][0]
assert r1.get("status") == "success"
assert len(r1.get("predictions", [])) == 5
assert "topPrediction" in r1
category_name = r1["topPrediction"]["category"]
conf_str = r1["topPrediction"]["confidencePercent"]
print(f"  OK: In-domain batch predict succeeded for 2 real files. Top prediction: {category_name} ({conf_str}), predictions count: {len(r1['predictions'])}")

print("\n========================================")
print(" ALL VERIFICATION CHECKS PASSED 100%! ")
print("========================================")

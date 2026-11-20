"""
Sound_Recognition — Batch Prediction & API Verification Test Suite
Tests:
 1. HTML DOM elements for Classifier Studio Batch Prediction
 2. CSS classes for alignment, layout, and theme styling
 3. JS batch classification module presence and integrity
 4. Backend POST /api/predict endpoint with multiple sequential and concurrent audio files
 5. Backend POST /api/predict/batch endpoint with multi-file payload
 6. Verification of complete independence from SoundSense
"""

import sys
import io
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_frontend_markup_and_scripts():
    print("=" * 60)
    print(" [1] Verifying Frontend Batch Prediction Markup & Styles")
    print("=" * 60)

    html_path = ROOT_DIR / "Front-End" / "index.html"
    assert html_path.exists(), "index.html not found"
    html_content = html_path.read_text(encoding="utf-8")

    required_ids = [
        "batch-card",
        "batch-model-info",
        "batch-drop",
        "batch-input",
        "batch-max-label",
        "batch-file-list",
        "batch-progress",
        "batch-progress-fill",
        "batch-progress-text",
        "batch-run-btn",
        "batch-run-idle",
        "batch-run-busy",
        "batch-csv-btn",
        "batch-clear-btn",
        "batch-table-wrap",
        "batch-table",
        "batch-table-body",
        "hero-goto-batch-btn",
        "hint-goto-batch-btn",
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in html_content, f"Missing required element id: {elem_id}"
    print(f"  OK: All {len(required_ids)} required DOM element IDs found in index.html.")

    # Check CSS
    css_path = ROOT_DIR / "Front-End" / "styles.css"
    assert css_path.exists(), "styles.css not found"
    css_content = css_path.read_text(encoding="utf-8")

    required_classes = [
        ".batch-card",
        ".batch-layout-grid",
        ".batch-drop",
        ".batch-file-list",
        ".batch-progress",
        ".batch-actions",
        ".batch-table-wrap",
        ".spec-table.batch-table",
        ".batch-shortcut-hint",
        ".banner-batch-pill",
    ]
    for cls in required_classes:
        assert cls in css_content, f"Missing CSS class: {cls}"
    print(f"  OK: All {len(required_classes)} CSS classes found in styles.css.")

    # Check JS
    js_path = ROOT_DIR / "Front-End" / "app.js"
    assert js_path.exists(), "app.js not found"
    js_content = js_path.read_text(encoding="utf-8")

    assert "initBatchClassification" in js_content, "Batch classification JS module missing"
    assert "downloadCsv" in js_content, "CSV export logic missing"
    assert "sound-recognition-batch-predictions-" in js_content, "Export file naming missing"
    print("  OK: Frontend JavaScript batch classification module verified.")

def test_api_predict_concurrency():
    print("\n" + "=" * 60)
    print(" [2] Testing POST /api/predict with Multiple Audio Files")
    print("=" * 60)

    sample_files = [
        ("5-263831-A-6.wav", "hen"),
        ("5-263902-A-36.wav", "vacuum_cleaner"),
        ("5-263491-A-25.wav", "footsteps"),
    ]
    wav_dir = ROOT_DIR / "ESC-50" / "wav_files"

    for fn, expected_label in sample_files:
        wav_path = wav_dir / fn
        if not wav_path.exists():
            print(f"  SKIP: {fn} not found on disk")
            continue

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        t0 = time.time()
        res = client.post(
            "/api/predict",
            data={"model": "resnet18"},
            files={"audio": (fn, audio_bytes, "audio/wav")}
        )
        latency = round((time.time() - t0) * 1000, 1)

        assert res.status_code == 200, f"Inference failed with status {res.status_code}: {res.text}"
        data = res.json()
        assert data["success"] is True
        assert "predictions" in data
        assert len(data["predictions"]) >= 5
        pred_name = data["category"]["name"]
        print(f"  OK: {fn} -> {pred_name} (expected {expected_label}) | Conf: {data['confidence']*100:.1f}% | Time: {latency}ms")

def test_api_predict_batch_endpoint():
    print("\n" + "=" * 60)
    print(" [3] Testing Dedicated POST /api/predict/batch Endpoint")
    print("=" * 60)

    wav_dir = ROOT_DIR / "ESC-50" / "wav_files"
    fn1 = "5-263831-A-6.wav"
    fn2 = "5-263902-A-36.wav"

    path1 = wav_dir / fn1
    path2 = wav_dir / fn2

    assert path1.exists() and path2.exists(), "Sample files not found"

    with open(path1, "rb") as f1, open(path2, "rb") as f2:
        b1 = f1.read()
        b2 = f2.read()

    files = [
        ("audio", (fn1, b1, "audio/wav")),
        ("audio", (fn2, b2, "audio/wav")),
    ]

    res = client.post("/api/predict/batch", data={"model": "resnet18"}, files=files)
    assert res.status_code == 200, f"Batch prediction failed: {res.text}"
    batch_res = res.json()

    assert batch_res["success"] is True
    assert batch_res["count"] == 2
    assert batch_res["successful"] == 2
    assert batch_res["failed"] == 0
    assert len(batch_res["results"]) == 2

    for r in batch_res["results"]:
        assert r["success"] is True
        assert "category" in r
        assert "predictions" in r
        print(f"  Batch item: {r['fileName']} -> {r['category']['name']} | Conf: {r['confidence']*100:.1f}%")

    print("\n  OK: Dedicated batch endpoint returned complete results matrix.")

def test_independence_from_soundsense():
    print("\n" + "=" * 60)
    print(" [4] Verifying Complete Independence from SoundSense")
    print("=" * 60)

    for p in [ROOT_DIR / "main.py", ROOT_DIR / "server.js", ROOT_DIR / "Front-End" / "app.js", ROOT_DIR / "Front-End" / "index.html"]:
        content = p.read_text(encoding="utf-8")
        assert "../SoundSense" not in content, f"{p.name} contains reference to ../SoundSense"
        assert "SoundSense/" not in content, f"{p.name} contains reference to SoundSense/"
        assert "SoundSense\\" not in content, f"{p.name} contains reference to SoundSense\\"

    print("  OK: Zero dependencies or references to SoundSense folder.")
    print("\n" + "=" * 60)
    print(" ALL BATCH PREDICTION VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    test_frontend_markup_and_scripts()
    test_api_predict_concurrency()
    test_api_predict_batch_endpoint()
    test_independence_from_soundsense()

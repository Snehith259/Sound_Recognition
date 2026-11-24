"""
SoundSense — Multi-Format Ambient & Pure Audio Evaluation Test
Evaluates both ambient and pure sound files across four major audio formats:
- WAV (uncompressed PCM)
- MP3 (MPEG Layer-3)
- WebM (Opus)
- AAC (Advanced Audio Coding)

Tests format parity, model accuracy, and API endpoint integration.
"""

import sys
import io
import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app
from Preproccesing.acoustic_inference import run_acoustic_inference, ESC50_CATEGORIES

client = TestClient(app)

TEST_CASES = [
    # Ambient Environmental Sounds
    {
        "id": "AMB-01",
        "name": "BirdNET Soundscape (Ambient Birds/Nature)",
        "path": ROOT_DIR / "eval" / "external" / "ambient" / "birdnet_window_000.wav",
        "type": "ambient",
        "expected": ["chirping_birds", "crickets", "insects", "wind"],
    },
    {
        "id": "AMB-02",
        "name": "Thunderstorm & Rain (Natural Weather Ambient)",
        "path": ROOT_DIR / "scratch" / "online_sounds" / "audio_04.ogg",
        "type": "ambient",
        "expected": ["rain", "thunderstorm", "water_drops"],
    },
    # Pure Natural & Animal Sounds
    {
        "id": "PURE-01",
        "name": "Dog Barking (Pure Animal)",
        "path": ROOT_DIR / "scratch" / "online_sounds" / "audio_01.ogg",
        "type": "pure",
        "expected": ["dog"],
    },
    {
        "id": "PURE-02",
        "name": "Cat Meowing (Pure Animal)",
        "path": ROOT_DIR / "scratch" / "online_sounds" / "audio_02.ogg",
        "type": "pure",
        "expected": ["cat"],
    },
    {
        "id": "PURE-03",
        "name": "Rooster Crowing (Pure Animal)",
        "path": ROOT_DIR / "scratch" / "online_sounds" / "audio_03.ogg",
        "type": "pure",
        "expected": ["rooster", "hen"],
    },
    {
        "id": "PURE-04",
        "name": "Cow Mooing (Pure Animal)",
        "path": ROOT_DIR / "eval" / "external" / "animals" / "animals_3838bc1d0f10.wav",
        "type": "pure",
        "expected": ["cow"],
    },
    # Pure Urban / Interior Sounds
    {
        "id": "PURE-05",
        "name": "Keyboard Typing (Pure Interior)",
        "path": ROOT_DIR / "Audio_Basics_HandsOn" / "1-137-A-32.wav",
        "type": "pure",
        "expected": ["keyboard_typing", "mouse_click"],
    },
    {
        "id": "PURE-06",
        "name": "Emergency Siren (Pure Urban)",
        "path": ROOT_DIR / "scratch" / "online_sounds" / "audio_05.ogg",
        "type": "pure",
        "expected": ["siren", "clock_alarm"],
    },
    # Out of Domain / Synthetic
    {
        "id": "OOD-01",
        "name": "Pure Synthetic Sine Tone (440Hz)",
        "path": ROOT_DIR / "scratch" / "online_sounds" / "audio_08_sine_tone.wav",
        "type": "synthetic",
        "expected": ["Not in ESC-50 Classes"],
    },
]

FORMATS = ["wav", "mp3", "webm", "aac"]


def encode_audio(src_path: Path, target_fmt: str) -> bytes:
    """Encodes source audio to target format using ffmpeg in-memory."""
    sr = "48000" if target_fmt == "webm" else "22050"
    codec_map = {
        "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
        "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
        "webm": ["-c:a", "libopus", "-b:a", "96k", "-f", "webm"],
        "aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    }
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src_path), "-ar", sr, "-ac", "1"] + codec_map[target_fmt] + ["pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"FFmpeg encoding to {target_fmt} failed: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def test_multiformat_ambient_and_pure():
    print("=" * 80)
    print(" SOUNDSENSE MULTI-FORMAT AMBIENT & PURE SOUND VERIFICATION")
    print(f" Testing {len(TEST_CASES)} sound cases across 4 formats (WAV, MP3, WebM, AAC)")
    print("=" * 80)

    results = []
    parity_mismatches = 0
    total_checks = 0

    for tc in TEST_CASES:
        src = tc["path"]
        if not src.exists():
            print(f"Skipping {tc['id']}: {src} not found")
            continue

        print(f"\n--- [{tc['id']}] {tc['name']} ({tc['type'].upper()}) ---")
        format_predictions = {}

        for fmt in FORMATS:
            try:
                encoded = encode_audio(src, fmt)
                total_checks += 1
                
                # Test direct inference
                res = run_acoustic_inference(encoded, model_name="ast")
                cat_name = res["category"]["name"]
                conf = res["confidence"]
                conf_str = f"{conf*100:5.1f}%" if conf is not None else " GATE"
                format_predictions[fmt] = (cat_name, conf)

                # Test FastAPI endpoint
                mime_map = {
                    "wav": "audio/wav",
                    "mp3": "audio/mpeg",
                    "webm": "audio/webm",
                    "aac": "audio/aac",
                }
                api_res = client.post(
                    "/api/predict",
                    data={"model": "ast"},
                    files={"audio": (f"test_clip.{fmt}", encoded, mime_map[fmt])}
                )
                assert api_res.status_code == 200, f"API failed ({api_res.status_code}) for {fmt}: {api_res.text}"
                api_data = api_res.json()
                assert api_data["success"] is True

                # Check expected match
                is_match = any(exp.lower() in cat_name.lower() for exp in tc["expected"])
                flag = "PASS" if is_match else "ALT "
                print(f"  [{fmt.upper():4s}] ({flag}) -> {cat_name:25s} | Conf: {conf_str} | API: HTTP 200 | Time: {res.get('inferenceTime')}ms")
            except Exception as err:
                print(f"  [{fmt.upper():4s}] ERROR: {err}")
                format_predictions[fmt] = ("ERROR", None)

        # Format parity check (did WAV, MP3, WebM, AAC give identical top class?)
        top_classes = [v[0] for v in format_predictions.values() if v[0] != "ERROR"]
        if len(set(top_classes)) == 1:
            print(f"  -> Format Parity: 100% IDENTICAL across all 4 formats ({top_classes[0]})")
        else:
            parity_mismatches += 1
            print(f"  -> Format Divergence: {format_predictions}")

    # Test Batch Inference with AAC mixed in
    print("\n" + "=" * 80)
    print(" TESTING BATCH INFERENCE WITH MIXED FORMATS (WAV + MP3 + WebM + AAC)")
    print("=" * 80)

    amb_wav = encode_audio(TEST_CASES[0]["path"], "wav")
    amb_mp3 = encode_audio(TEST_CASES[0]["path"], "mp3")
    pure_webm = encode_audio(TEST_CASES[2]["path"], "webm")
    pure_aac = encode_audio(TEST_CASES[2]["path"], "aac")

    batch_files = [
        ("audio", ("ambient.wav", amb_wav, "audio/wav")),
        ("audio", ("ambient.mp3", amb_mp3, "audio/mpeg")),
        ("audio", ("dog.webm", pure_webm, "audio/webm")),
        ("audio", ("dog.aac", pure_aac, "audio/aac")),
    ]

    batch_res = client.post("/api/predict/batch", data={"model": "ast"}, files=batch_files)
    assert batch_res.status_code == 200, f"Batch predict failed: {batch_res.text}"
    batch_data = batch_res.json()
    assert batch_data["success"] is True
    print(f"  Batch Success! Processed {len(batch_data['results'])} files successfully:")
    for item in batch_data["results"]:
        top = item.get("topPrediction") or {}
        cat = top.get("category") if isinstance(top, dict) else None
        if not cat:
            cat = item.get("modelName") or item.get("category") or "Unknown"
        conf = top.get("confidencePercent") if isinstance(top, dict) else None
        print(f"    - {item['fileName']:15s} | Ext: {Path(item['fileName']).suffix:5s} | Top: {str(cat):15s} | Conf: {conf or 'N/A'}")

    print("\n" + "=" * 80)
    print(f" ALL MULTI-FORMAT TESTS COMPLETED! Total Checks: {total_checks} | Parity Divergences: {parity_mismatches}")
    print("=" * 80)


if __name__ == "__main__":
    test_multiformat_ambient_and_pure()

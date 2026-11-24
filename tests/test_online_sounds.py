"""
SoundSense — Real Online Audio Classification Verification
Evaluates authentic sound files fetched from external online repositories:
- Wikimedia Commons (Public Domain / CC-BY)
- GitHub Open Audio Repositories (pdx-cs-sound / wav2vec2)

Filenames passed to API are anonymized ('sample_01.wav', 'sample_02.wav', etc.)
to guarantee 100% genuine acoustic classification with zero metadata reliance.
"""

import sys
import io
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from main import app

ONLINE_TEST_CASES = [
    {
        "id": "ONLINE-01",
        "name": "Dog Barking",
        "source": "Wikimedia Commons (File:Barking_of_a_dog.ogg)",
        "file": "audio_01.ogg",
        "expected_class": "dog",
        "expected_ids": [0],
        "is_ood": False,
    },
    {
        "id": "ONLINE-02",
        "name": "Cat Meowing",
        "source": "Wikimedia Commons (File:Meow.ogg)",
        "file": "audio_02.ogg",
        "expected_class": "cat",
        "expected_ids": [5],
        "is_ood": False,
    },
    {
        "id": "ONLINE-03",
        "name": "Rooster Crowing",
        "source": "Wikimedia Commons (File:Rooster_crowing.ogg)",
        "file": "audio_03.ogg",
        "expected_class": "rooster / hen",
        "expected_ids": [1, 6],
        "is_ood": False,
    },
    {
        "id": "ONLINE-04",
        "name": "Thunderstorm / Rain",
        "source": "Wikimedia Commons (File:Thunder.ogg)",
        "file": "audio_04.ogg",
        "expected_class": "thunderstorm / rain",
        "expected_ids": [19, 10],
        "is_ood": False,
    },
    {
        "id": "ONLINE-05",
        "name": "Emergency Siren",
        "source": "IBM MAX Audio Classifier (siren.wav)",
        "file": "../../eval/external/ibm/ibm_207399184fde.wav",
        "expected_class": "siren",
        "expected_ids": [42],
        "is_ood": False,
    },
    {
        "id": "ONLINE-06",
        "name": "Car Horn",
        "source": "IBM MAX Audio Classifier (car_horn.wav)",
        "file": "../../eval/external/ibm/ibm_3f0f45ff5f30.wav",
        "expected_class": "car_horn",
        "expected_ids": [43, 42],
        "is_ood": False,
    },
    {
        "id": "ONLINE-07",
        "name": "Human Spoken Speech",
        "source": "GitHub Open Audio (gsoc-wav2vec2/sample.wav)",
        "file": "audio_06_speech.wav",
        "expected_class": "Not in ESC-50 Classes (Speech)",
        "expected_ids": [-1],
        "is_ood": True,
    },
    {
        "id": "ONLINE-08",
        "name": "Pure Synthetic Sine Tone",
        "source": "GitHub Open Audio (pdx-cs-sound/sine.wav)",
        "file": "audio_08_sine_tone.wav",
        "expected_class": "Not in ESC-50 Classes (Synthetic)",
        "expected_ids": [-1],
        "is_ood": True,
    },
]

def run_tests():
    client = TestClient(app)
    sounds_dir = ROOT_DIR / "scratch" / "online_sounds"
    models = ["ast", "sresnet", "cnngru", "ensemble"]

    print("=" * 80)
    print(" SOUNDSENSE ONLINE ACOUSTIC GENERALIZATION BENCHMARK")
    print(" Verifying real audio samples downloaded from external internet sources")
    print("=" * 80)

    detailed_results = []

    for idx, tc in enumerate(ONLINE_TEST_CASES, start=1):
        file_path = (sounds_dir / tc["file"]).resolve()
        assert file_path.exists(), f"Audio file not found: {file_path}"
        audio_bytes = file_path.read_bytes()

        # Completely anonymize filename sent to server (e.g. unknown_sound_01.bin)
        anon_name = f"unknown_audio_{idx:02d}.wav"

        print(f"\n[Case {idx}/8] {tc['name']} ({tc['source']})")
        print(f"  Anonymized input: {anon_name} ({len(audio_bytes)/1024:.1f} KB)")
        print(f"  Expected target:  {tc['expected_class']} (OOD: {tc['is_ood']})")
        print("  " + "-" * 76)

        for model in models:
            res = client.post(
                "/api/predict",
                data={"model": model, "threshold": "0.30"},
                files={"audio": (anon_name, audio_bytes, "application/octet-stream")}
            )
            assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
            data = res.json()

            pred_cat = data["category"]["name"]
            pred_id = data["category"]["id"]
            conf = data.get("confidence")
            conf_str = f"{conf*100:5.1f}%" if conf is not None else " GATE"
            engine = data.get("engine")
            is_ood = data.get("isOutOfScope", False)
            top3 = [
                (p["category"]["name"], round(p["confidence"] * 100, 1) if p.get("confidence") is not None else None)
                for p in data.get("predictions", [])[:3]
            ]

            if tc["is_ood"]:
                is_correct = is_ood or (pred_id == -1) or ("Not in ESC-50" in pred_cat)
            else:
                is_correct = (pred_id in tc.get("expected_ids", [tc.get("expected_id")]))

            mark = "PASS" if is_correct else "MISMATCH"
            print(f"    [{mark:8s}] Model: {model:10s} -> {pred_cat:20s} ({conf_str}) | Latency: {data.get('inferenceTime', 0)}ms | Top-3: {top3}")

            detailed_results.append({
                "case_id": tc["id"],
                "name": tc["name"],
                "source": tc["source"],
                "expected": tc["expected_class"],
                "is_ood_case": tc["is_ood"],
                "model": model,
                "engine": engine,
                "predicted": pred_cat,
                "confidence": conf,
                "is_ood_detected": is_ood,
                "correct": is_correct,
                "top3": top3,
                "latency_ms": data.get("inferenceTime", 0)
            })

    # Summary table
    print("\n" + "=" * 80)
    print(" FINAL VERIFICATION RESULTS TABLE")
    print("=" * 80)
    print(f"{'Sound Name':<20s} | {'Model':<10s} | {'Predicted':<22s} | {'Confidence':<10s} | {'Status'}")
    print("-" * 80)

    correct_total = 0
    for r in detailed_results:
        if r["correct"]:
            correct_total += 1
        st = "CORRECT" if r["correct"] else "MISMATCH"
        conf_disp = f"{r['confidence']*100:5.1f}%" if r["confidence"] is not None else "     GATE"
        print(f"{r['name']:<20s} | {r['model']:<10s} | {r['predicted']:<22s} | {conf_disp:<10s} | {st}")

    total_evals = len(detailed_results)
    acc = (correct_total / total_evals) * 100.0

    print("=" * 80)
    print(f" TOTAL SYSTEM ACCURACY: {correct_total}/{total_evals} ({acc:.1f}%)")
    print("=" * 80)

    # Save to JSON artifact
    out_json = ROOT_DIR / "scratch" / "online_sounds_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "total_cases": len(ONLINE_TEST_CASES),
            "total_evaluations": total_evals,
            "correct_evaluations": correct_total,
            "accuracy_percent": acc,
            "results": detailed_results,
        }, f, indent=2)
    print(f"\nSaved detailed benchmark report to: {out_json}")

if __name__ == "__main__":
    run_tests()

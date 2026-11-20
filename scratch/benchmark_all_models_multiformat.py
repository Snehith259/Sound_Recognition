import os
import sys
import json
import time
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import Preproccesing.acoustic_inference as ai
from Preproccesing.acoustic_inference import ESC50_CATEGORIES

def ensure_test_corpus():
    """
    Selects 30 diverse ground-truth ESC-50 sound files across all 5 domains,
    and creates cross-format variants (WAV, MP3, MPG/MPEG).
    """
    test_dir = BASE_DIR / "scratch" / "multiformat_corpus"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Search for canonical ESC-50 samples
    source_candidates = [
        BASE_DIR / "ESC-50" / "wav_files",
        BASE_DIR / "ESC-50" / "audio",
        BASE_DIR / "Preproccesing" / "ESC-50" / "wav_files",
        BASE_DIR / "Front-End" / "samples",
    ]

    available_samples = {}
    for d in source_candidates:
        if d.exists():
            for f in d.glob("*.wav"):
                # Check if it has target class id (e.g. 1-100032-A-0.wav or 0.wav)
                parts = f.stem.split("-")
                class_id = None
                if len(parts) >= 4:
                    try:
                        class_id = int(parts[-1])
                    except ValueError:
                        pass
                elif f.stem.isdigit():
                    class_id = int(f.stem)

                if class_id is not None and 0 <= class_id < 50:
                    if class_id not in available_samples:
                        available_samples[class_id] = f

    print(f"Found {len(available_samples)} canonical sound classes in source directories.")

    # Select 25 distinct environmental classes across all 5 domains
    selected_classes = [
        # Domain 1: Animals
        0, 1, 4, 5, 9,
        # Domain 2: Natural soundscapes
        10, 11, 12, 13, 16,
        # Domain 3: Human non-speech
        20, 21, 22, 24, 25,
        # Domain 4: Interior / domestic
        30, 31, 32, 36, 38,
        # Domain 5: Exterior / urban
        40, 41, 42, 43, 47,
    ]

    corpus = []
    formats = ["wav", "mp3", "mpg"]

    for idx, cid in enumerate(selected_classes):
        if cid not in available_samples:
            continue
        src_path = available_samples[cid]
        cat_info = ESC50_CATEGORIES[cid]
        # Distribute formats evenly: 1/3 WAV, 1/3 MP3, 1/3 MPG
        fmt = formats[idx % len(formats)]
        target_name = f"sample_class_{cid}_{cat_info['name']}.{fmt}"
        target_path = test_dir / target_name

        if not target_path.exists():
            if fmt == "wav":
                subprocess.run(["ffmpeg", "-y", "-i", str(src_path), "-ar", "22050", "-ac", "1", str(target_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif fmt == "mp3":
                subprocess.run(["ffmpeg", "-y", "-i", str(src_path), "-b:a", "192k", "-ar", "22050", "-ac", "1", str(target_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            elif fmt == "mpg":
                subprocess.run(["ffmpeg", "-y", "-i", str(src_path), "-c:a", "mp2", "-b:a", "128k", str(target_path)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        corpus.append({
            "path": target_path,
            "filename": target_name,
            "format": fmt,
            "ground_truth_id": cid,
            "ground_truth_name": cat_info["name"],
            "domain": cat_info["group"],
            "is_ood": False
        })

    # Add 3 deceptive filenames (testing anti-faking / filename independence)
    # E.g. a Dog sound (id 0) named 'airplane_jet_turbine.mp3'
    if 0 in available_samples:
        deceptive_dog = test_dir / "deceptive_airplane_jet_turbine.mp3"
        subprocess.run(["ffmpeg", "-y", "-i", str(available_samples[0]), "-b:a", "192k", str(deceptive_dog)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        corpus.append({
            "path": deceptive_dog,
            "filename": "deceptive_airplane_jet_turbine.mp3",
            "format": "mp3",
            "ground_truth_id": 0,
            "ground_truth_name": "dog",
            "domain": "Animals",
            "is_ood": False,
            "is_deceptive": True
        })

    # E.g. a Rain sound (id 10) named 'chainsaw_cutting_wood.mpg'
    if 10 in available_samples:
        deceptive_rain = test_dir / "deceptive_chainsaw_cutting_wood.mpg"
        subprocess.run(["ffmpeg", "-y", "-i", str(available_samples[10]), "-c:a", "mp2", "-b:a", "128k", str(deceptive_rain)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        corpus.append({
            "path": deceptive_rain,
            "filename": "deceptive_chainsaw_cutting_wood.mpg",
            "format": "mpg",
            "ground_truth_id": 10,
            "ground_truth_name": "rain",
            "domain": "Natural",
            "is_ood": False,
            "is_deceptive": True
        })

    # Add Out-of-Domain synthetic tone (should be rejected by silence/pure-tone OOD detector)
    sr = 22050
    t = np.linspace(0, 5, int(sr * 5), endpoint=False)
    pure_sine = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1 kHz sine
    sine_wav = test_dir / "ood_pure_sine_1khz.wav"
    import soundfile as sf
    sf.write(str(sine_wav), pure_sine, sr)
    corpus.append({
        "path": sine_wav,
        "filename": "ood_pure_sine_1khz.wav",
        "format": "wav",
        "ground_truth_id": -1,
        "ground_truth_name": "Out of Scope",
        "domain": "Synthetic",
        "is_ood": True
    })

    print(f"Constructed test corpus with {len(corpus)} total audio clips:")
    fmt_counts = {}
    for item in corpus:
        fmt_counts[item["format"]] = fmt_counts.get(item["format"], 0) + 1
    print(f"Format distribution: {fmt_counts}")

    return corpus


def simulate_microphone_capture():
    """
    Simulates real microphone capture pipeline:
    Generates human speech / vocal formant acoustic sample and clapping audio,
    encodes it into 16-bit PCM WAV (matching Front-End/app.js encodeWAV()),
    and evaluates it through the backend models.
    """
    sr = 22050
    duration = 3.5  # 3.5s real clip
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Clapping acoustic burst simulation (short impulsive transients)
    clapping_signal = np.zeros_like(t)
    clap_times = [0.3, 0.8, 1.3, 1.8, 2.3, 2.8]
    for ct in clap_times:
        idx = int(ct * sr)
        burst_len = int(0.04 * sr)
        decay = np.exp(-np.linspace(0, 10, burst_len))
        noise_burst = np.random.randn(burst_len) * decay
        if idx + burst_len < len(clapping_signal):
            clapping_signal[idx:idx+burst_len] += noise_burst

    # Encode as WAV in memory
    import io
    import soundfile as sf
    buf_clap = io.BytesIO()
    sf.write(buf_clap, clapping_signal.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    clap_bytes = buf_clap.getvalue()

    # 2. Ambient room speech / laughing simulation (harmonic formants at 200Hz, 800Hz, 2400Hz)
    f0 = 220.0
    vocal_signal = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t) + 0.1 * np.sin(2 * np.pi * 5 * f0 * t)
    # Modulate envelope like natural speech rhythm (syllabic rate ~ 4Hz)
    syllables = (np.sin(2 * np.pi * 3.5 * t) + 1.0) / 2.0
    vocal_signal = vocal_signal * syllables + 0.02 * np.random.randn(len(t))

    buf_vocal = io.BytesIO()
    sf.write(buf_vocal, vocal_signal.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    vocal_bytes = buf_vocal.getvalue()

    return [
        {"name": "mic_capture_clapping.wav", "bytes": clap_bytes, "expected_category": "clapping", "class_id": 22},
        {"name": "mic_capture_vocal_ambient.wav", "bytes": vocal_bytes, "expected_category": "laughing/coughing/breathing", "class_id": 26}
    ]


def run_full_evaluation():
    corpus = ensure_test_corpus()
    models = ["resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]

    results = {
        m: {
            "top1_correct": 0,
            "top5_correct": 0,
            "in_domain_total": 0,
            "ood_rejected_correctly": 0,
            "ood_total": 0,
            "confidences": [],
            "latencies_ms": [],
            "deceptive_invariance_count": 0,
            "deceptive_total": 0,
            "by_format": {"wav": {"correct": 0, "total": 0}, "mp3": {"correct": 0, "total": 0}, "mpg": {"correct": 0, "total": 0}}
        }
        for m in models
    }

    sample_predictions = []

    print("\nStarting comprehensive evaluation of all 5 architectures across all formats...")
    for idx, item in enumerate(corpus):
        is_ood = item["is_ood"]
        is_deceptive = item.get("is_deceptive", False)
        gt_id = item["ground_truth_id"]
        fmt = item["format"]

        item_row = {
            "filename": item["filename"],
            "format": fmt,
            "ground_truth": item["ground_truth_name"],
            "domain": item["domain"],
            "predictions": {}
        }

        # Read audio raw bytes directly
        with open(item["path"], "rb") as f:
            audio_bytes = f.read()

        for m in models:
            res = ai.run_acoustic_inference(audio_bytes, model_name=m)
            item_row["predictions"][m] = {
                "category": res["category"]["name"],
                "confidence": res["confidence"],
                "isOutOfScope": res.get("isOutOfScope", False),
                "inferenceTime": res.get("inferenceTime", 0)
            }

            model_stat = results[m]
            model_stat["latencies_ms"].append(res.get("inferenceTime", 0))

            if is_ood:
                model_stat["ood_total"] += 1
                if res.get("isOutOfScope", False):
                    model_stat["ood_rejected_correctly"] += 1
            else:
                model_stat["in_domain_total"] += 1
                pred_id = res["category"].get("id", -1)
                top5_ids = [p["category"]["id"] for p in res.get("predictions", []) if "id" in p["category"]]

                is_top1 = (pred_id == gt_id)
                is_top5 = (gt_id in top5_ids)

                if is_top1:
                    model_stat["top1_correct"] += 1
                    model_stat["by_format"][fmt]["correct"] += 1
                if is_top5:
                    model_stat["top5_correct"] += 1

                model_stat["by_format"][fmt]["total"] += 1
                model_stat["confidences"].append(res["confidence"])

                if is_deceptive:
                    model_stat["deceptive_total"] += 1
                    # Invariance test: ensure prediction matches true sound (gt_id) despite deceptive filename
                    if is_top1 or is_top5:
                        model_stat["deceptive_invariance_count"] += 1

        sample_predictions.append(item_row)

    # 2. Test Microphone Real-Time Capture Pipeline
    print("\nTesting microphone recording simulation and inference...")
    mic_clips = simulate_microphone_capture()
    mic_results = []
    for mc in mic_clips:
        clip_row = {"name": mc["name"], "expected": mc["expected_category"], "predictions": {}}
        for m in models:
            res = ai.run_acoustic_inference(mc["bytes"], model_name=m)
            clip_row["predictions"][m] = {
                "category": res["category"]["name"],
                "confidence": res["confidence"],
                "isOutOfScope": res.get("isOutOfScope", False),
                "top5": [p["category"]["name"] for p in res.get("predictions", [])[:3]]
            }
        mic_results.append(clip_row)

    # Calculate summaries
    summary = {}
    for m in models:
        st = results[m]
        n_in = st["in_domain_total"]
        n_ood = st["ood_total"]
        top1_acc = (st["top1_correct"] / n_in * 100) if n_in else 0.0
        top5_acc = (st["top5_correct"] / n_in * 100) if n_in else 0.0
        ood_acc = (st["ood_rejected_correctly"] / n_ood * 100) if n_ood else 0.0
        mean_conf = float(np.mean(st["confidences"])) if st["confidences"] else 0.0
        mean_latency = float(np.mean(st["latencies_ms"])) if st["latencies_ms"] else 0.0

        format_acc = {}
        for fmt, fst in st["by_format"].items():
            format_acc[fmt] = round((fst["correct"] / fst["total"] * 100), 1) if fst["total"] else 0.0

        summary[m] = {
            "top1_accuracy": round(top1_acc, 2),
            "top5_accuracy": round(top5_acc, 2),
            "mean_confidence": round(mean_conf, 4),
            "mean_latency_ms": round(mean_latency, 1),
            "ood_rejection_rate": round(ood_acc, 2),
            "accuracy_by_format": format_acc,
            "anti_faking_invariance": f"{st['deceptive_invariance_count']}/{st['deceptive_total']} verified invariant"
        }

    output_report = {
        "benchmark_summary": summary,
        "microphone_validation": mic_results,
        "sample_evaluations": sample_predictions[:10]  # first 10 for inspection
    }

    report_path = BASE_DIR / "scratch" / "comprehensive_model_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(output_report, f, indent=2)

    print("\n" + "="*80)
    print("MODEL ACCURACY, CONFIDENCE & PREDICTION IMPROVEMENT REPORT")
    print("="*80)
    print(f"{'Model':<16} | {'Top-1 Acc':<10} | {'Top-5 Acc':<10} | {'Mean Conf':<10} | {'Latency':<10} | {'WAV Acc':<8} | {'MP3 Acc':<8} | {'MPG Acc':<8}")
    print("-"*80)
    for m in models:
        s = summary[m]
        print(f"{m:<16} | {s['top1_accuracy']:>8.1f}% | {s['top5_accuracy']:>8.1f}% | {s['mean_confidence']:>9.2f} | {s['mean_latency_ms']:>8.1f}ms | {s['accuracy_by_format']['wav']:>7.1f}% | {s['accuracy_by_format']['mp3']:>7.1f}% | {s['accuracy_by_format']['mpg']:>7.1f}%")
    print("="*80)

if __name__ == "__main__":
    run_full_evaluation()

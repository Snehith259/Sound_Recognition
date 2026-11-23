"""Evaluate selected dev clips under controlled audio transformations."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from eval.eval_external import _encode
from Preproccesing.acoustic_inference import run_acoustic_inference

ROOT = Path(__file__).resolve().parents[1]
SR = 22050


def _wav_bytes(audio: np.ndarray) -> bytes:
    import io
    stream = io.BytesIO()
    sf.write(stream, audio.astype(np.float32), SR, format="WAV", subtype="FLOAT")
    return stream.getvalue()


def _snr_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(size=len(y))
    signal_rms = np.sqrt(np.mean(y ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    noise *= signal_rms / (10 ** (snr_db / 20)) / noise_rms
    return (y + noise).astype(np.float32)


def _variants(y: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    rng = np.random.default_rng(21)
    gain_up = y * (10 ** (20 / 20))
    gain_down = y * (10 ** (-20 / 20))
    shifted = np.roll(y, int(0.75 * SR))
    short = y[: int(1.5 * SR)]
    sos = signal.butter(6, 8000, btype="lowpass", fs=SR, output="sos")
    band_limited = signal.sosfiltfilt(sos, y).astype(np.float32)
    pitch_up = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=2).astype(np.float32)
    return {
        "clean": (y, "wav"),
        "noise_30db_snr": (_snr_noise(y, 30, rng), "wav"),
        "noise_20db_snr": (_snr_noise(y, 20, rng), "wav"),
        "noise_10db_snr": (_snr_noise(y, 10, rng), "wav"),
        "gain_plus_20db": (gain_up.astype(np.float32), "wav"),
        "gain_minus_20db": (gain_down.astype(np.float32), "wav"),
        "pitch_plus_2st": (pitch_up, "wav"),
        "time_shift_0_75s": (shifted.astype(np.float32), "wav"),
        "mp3_64kbps": (y, "mp3_64"),
        "band_limit_8khz": (band_limited, "wav"),
        "short_1_5s": (short, "wav"),
    }


def run(model: str, limit: int = 30, threshold: float = 0.5) -> dict[str, Any]:
    with (ROOT / "eval" / "external_manifest.csv").open(newline="", encoding="utf-8") as stream:
        clips = [row for row in csv.DictReader(stream) if row["split"] == "dev" and row["in_scope"] == "true"][:limit]
    if not clips:
        raise RuntimeError("No in-scope dev clips found. Run eval.build_external_set first.")
    records = []
    counter = 0
    for row in clips:
        audio, sr = sf.read(ROOT / "eval" / "external" / row["file"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        allowed = json.loads(row.get("allowed_labels") or "[]")
        for variant, (changed, output_format) in _variants(audio).items():
            counter += 1
            raw = _wav_bytes(changed)
            if output_format == "mp3_64":
                raw, content_type, ext = _encode(raw, "mp3")
                # Re-encode at the specified 64 kbps via the same codec path.
                import subprocess
                proc = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3", "pipe:1"], input=_wav_bytes(changed), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                if proc.returncode == 0 and proc.stdout:
                    raw = proc.stdout
                content_type, ext = "audio/mpeg", ".mp3"
            else:
                content_type, ext = "audio/wav", ".wav"
            try:
                body = run_acoustic_inference(raw, model, threshold)
                status_code = 200 if body.get("success") else 500
            except FileNotFoundError as exc:
                body = {"error": "Model unavailable", "details": str(exc)}
                status_code = 503
            except Exception as exc:
                body = {"error": "Inference failed", "details": str(exc)}
                status_code = 500
            category = body.get("category") or {}
            records.append({
                "sourceFile": row["file"], "sourceRecording": row["source_recording"],
                "variant": variant, "statusCode": status_code,
                "label": row["label"], "allowedLabels": allowed,
                "predictedCategory": category.get("name"), "predictedClassId": category.get("id"),
                "isOutOfScope": body.get("isOutOfScope"), "confidence": body.get("confidence"),
                "correct": status_code == 200 and not body.get("isOutOfScope") and category.get("id") in allowed,
                "error": body.get("detail") or body.get("details") or body.get("error"),
            })
    variants = sorted({r["variant"] for r in records})
    summary = {
        v: {
            "n": sum(r["variant"] == v for r in records),
            "accuracy": (sum(r["correct"] for r in records if r["variant"] == v) / max(1, sum(r["variant"] == v and r["statusCode"] == 200 for r in records)))
            if any(r["variant"] == v and r["statusCode"] == 200 for r in records) else None,
            "failedRequests": sum(r["variant"] == v and r["statusCode"] != 200 for r in records),
        }
        for v in variants
    }
    result = {"createdAt": datetime.now(timezone.utc).isoformat(), "split": "dev", "model": model, "nClips": len(clips), "summary": summary, "results": records}
    report_dir = ROOT / "eval" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"robustness_{model}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet18", "vgg16", "sresnet", "cnngru", "ensemble"], default="ensemble")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.model, args.limit, args.threshold)

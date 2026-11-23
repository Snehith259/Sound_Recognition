"""Measure silence/tone gates and classifier OOD scores on synthetic signals."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from Preproccesing.acoustic_inference import run_acoustic_inference

ROOT = Path(__file__).resolve().parents[1]
SR = 22050
DURATION = 5


def _signals() -> list[tuple[str, np.ndarray]]:
    t = np.arange(SR * DURATION, dtype=np.float64) / SR
    rng = np.random.default_rng(20261121)
    result = [("digital_silence", np.zeros_like(t))]
    for dbfs in (-60, -50, -40):
        noise = rng.normal(size=t.size)
        noise *= 10 ** (dbfs / 20) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
        result.append((f"room_noise_{dbfs}dbfs", noise))
    white = rng.normal(size=t.size)
    result.append(("white_noise", white / (np.max(np.abs(white)) + 1e-12) * 0.5))
    white_for_color = rng.normal(size=t.size)
    spectrum = np.fft.rfft(white_for_color)
    frequencies = np.fft.rfftfreq(t.size, d=1 / SR)
    pink = np.fft.irfft(spectrum / np.sqrt(np.maximum(frequencies, 1 / DURATION)), n=t.size)
    result.append(("pink_noise", pink / (np.max(np.abs(pink)) + 1e-12) * 0.5))
    brown = np.fft.irfft(spectrum / np.maximum(frequencies, 1 / DURATION), n=t.size)
    result.append(("brown_noise", brown / (np.max(np.abs(brown)) + 1e-12) * 0.5))
    result.append(("sine_440hz", 0.5 * np.sin(2 * np.pi * 440 * t)))
    result.append(("hum_50hz", 0.4 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t)))
    sweep = np.sin(2 * np.pi * (40 * t + (18000 - 40) * t ** 2 / (2 * DURATION)))
    result.append(("sine_sweep", 0.5 * sweep))
    return [(name, audio.astype(np.float32)) for name, audio in result]


def run(models: list[str], threshold: float = 0.5) -> dict[str, Any]:
    records = []
    for model in models:
        for index, (name, audio) in enumerate(_signals(), 1):
            try:
                body = run_acoustic_inference(audio, model, threshold)
                status_code = 200 if body.get("success") else 500
            except FileNotFoundError as exc:
                body = {"error": "Model unavailable", "details": str(exc)}
                status_code = 503
            except Exception as exc:
                body = {"error": "Inference failed", "details": str(exc)}
                status_code = 500
            metrics = body.get("metrics") or {}
            records.append({
                "name": name,
                "model": model,
                "statusCode": status_code,
                "category": (body.get("category") or {}).get("name"),
                "isOutOfScope": body.get("isOutOfScope"),
                "confidence": body.get("confidence"),
                "confidenceType": body.get("confidenceType"),
                "oodScore": body.get("oodScore"),
                "dbfs": metrics.get("dbfs"),
                "flatness": metrics.get("flatness"),
                "spectralPeakRatio": metrics.get("spectral_peak_ratio"),
                "error": body.get("details") or body.get("detail") or body.get("error"),
            })
    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "threshold": threshold,
        "syntheticCaseCount": len(_signals()),
        "results": records,
        "summary": {
            model: {
                "expected": len(_signals()),
                "completed": sum(r["model"] == model and r["statusCode"] == 200 for r in records),
                "failed": sum(r["model"] == model and r["statusCode"] != 200 for r in records),
                "rejectedSyntheticCount": sum(r["model"] == model and r["statusCode"] == 200 and r["isOutOfScope"] for r in records),
                "rejectionRateAmongCompleted": (
                    sum(r["model"] == model and r["statusCode"] == 200 and r["isOutOfScope"] for r in records)
                    / max(1, sum(r["model"] == model and r["statusCode"] == 200 for r in records))
                ),
                "systemSuccessRate": sum(r["model"] == model and r["statusCode"] == 200 for r in records) / max(1, len(_signals())),
            }
            for model in models
        },
    }
    report_dir = ROOT / "eval" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "ood_synthetic.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["resnet18", "vgg16", "sresnet", "cnngru", "ensemble"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.models, args.threshold)

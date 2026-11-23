"""Measure silence and narrow-band rule gates on external audio and synthetic signals."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from Preproccesing.acoustic_inference import (
    SILENCE_DBFS_THRESHOLD,
    compute_signal_metrics,
    load_audio_waveform,
)

ROOT = Path(__file__).resolve().parents[1]
TONE_FLATNESS_THRESHOLD = 0.003
TONE_PEAK_RATIO_THRESHOLD = 0.25
TONE_MIN_FREQUENCY_HZ = 30.0
TONE_FREQUENCY_SPREAD_THRESHOLD_HZ = 20.0


def _gate(audio: np.ndarray, sample_rate: int, label: str, source: str, in_scope: bool | None) -> dict[str, Any]:
    metrics = compute_signal_metrics(audio, sample_rate)
    if metrics["rms"] < 10.0 ** (SILENCE_DBFS_THRESHOLD / 20.0):
        gate = "silence"
    elif (
        metrics["flatness_raw"] < TONE_FLATNESS_THRESHOLD
        and metrics["spectral_peak_ratio"] > TONE_PEAK_RATIO_THRESHOLD
        and metrics["dominant_frequency_frame_count"] >= 3
        and metrics["dominant_frequency_median_hz"] > TONE_MIN_FREQUENCY_HZ
        and metrics["dominant_frequency_spread_hz"] < TONE_FREQUENCY_SPREAD_THRESHOLD_HZ
    ):
        gate = "tone"
    else:
        gate = "pass"
    return {
        "label": label,
        "source": source,
        "inScope": in_scope,
        "gate": gate,
        "dbfs": metrics["dbfs"],
        "flatness": metrics["flatness_raw"],
        "spectralPeakRatio": metrics["spectral_peak_ratio"],
        "dominantFrequencySpreadHz": metrics["dominant_frequency_spread_hz"],
        "dominantFrequencyFrameCount": metrics["dominant_frequency_frame_count"],
        "dominantFrequencyMedianHz": metrics["dominant_frequency_median_hz"],
    }


def run(split: str = "dev") -> dict[str, Any]:
    with (ROOT / "eval" / "external_manifest.csv").open(newline="", encoding="utf-8") as stream:
        source_rows = [row for row in csv.DictReader(stream) if row["split"] == split]
    records = []
    for row in source_rows:
        y, sr = load_audio_waveform(ROOT / "eval" / "external" / row["file"], target_sr=22050, target_duration=None)
        result = _gate(y, sr, row["label"], row["source_dataset"], row["in_scope"].lower() == "true")
        result["file"] = row["file"]
        records.append(result)

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    label_summary = {
        label: {
            "n": len(items),
            "gateCounts": dict(Counter(item["gate"] for item in items)),
            "falseRejectionRate": (
                sum(item["gate"] != "pass" for item in items if item["inScope"] is True)
                / max(1, sum(item["inScope"] is True for item in items))
            ) if any(item["inScope"] is True for item in items) else None,
        }
        for label, items in sorted(by_label.items())
    }

    sr = 22050
    t = np.arange(sr * 5, dtype=np.float32) / sr
    rng = np.random.default_rng(20261121)
    white = rng.normal(0, 0.1, size=len(t)).astype(np.float32)
    synthetic = [
        ("digital_silence", np.zeros_like(t)),
        ("room_noise_-60dBFS", rng.normal(0, 10 ** (-60 / 20), size=len(t)).astype(np.float32)),
        ("room_noise_-50dBFS", rng.normal(0, 10 ** (-50 / 20), size=len(t)).astype(np.float32)),
        ("white_noise", white),
        ("sine_440hz", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)),
        ("hum_50hz", (0.4 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)),
        ("sine_sweep", (0.5 * np.sin(2 * np.pi * (40 * t + (18000 - 40) * t ** 2 / 10))).astype(np.float32)),
    ]
    synthetic_results = [_gate(audio, sr, name, "synthetic", None) for name, audio in synthetic]
    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "silenceThresholdDbfs": SILENCE_DBFS_THRESHOLD,
        "toneFlatnessThreshold": TONE_FLATNESS_THRESHOLD,
        "toneSpectralPeakRatioThreshold": TONE_PEAK_RATIO_THRESHOLD,
        "toneMinimumFrequencyHz": TONE_MIN_FREQUENCY_HZ,
        "toneFrequencySpreadThresholdHz": TONE_FREQUENCY_SPREAD_THRESHOLD_HZ,
        "externalSummary": {
            "n": len(records),
            "gateCounts": dict(Counter(item["gate"] for item in records)),
            "inScopeCount": sum(item["inScope"] is True for item in records),
            "inScopeFalseRejectionRate": (
                sum(item["gate"] != "pass" for item in records if item["inScope"] is True)
                / max(1, sum(item["inScope"] is True for item in records))
            ),
        },
        "byLabel": label_summary,
        "synthetic": synthetic_results,
        "records": records,
    }
    output = ROOT / "eval" / "reports" / f"signal_gates_{split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    print(json.dumps(result["externalSummary"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "final"], default="dev")
    args = parser.parse_args()
    run(args.split)

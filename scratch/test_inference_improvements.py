"""
Ablation test on all 182 external audio files for fast PyTorch CNN/RNN models:
Tests:
1. Baseline (zero-pad short clips, uniform window mean, equal 0.25 ensemble weights)
2. Tile-padding short clips (< 4.0s) instead of zero-padding 80% silence
3. RMS-energy weighted window pooling (prevent silent windows from diluting active sound windows)
4. Accuracy-weighted ensemble ([sresnet: 0.33, resnet18: 0.32, vgg16: 0.28, cnngru: 0.07])
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from Preproccesing.acoustic_inference import (
    ANALYSIS_WINDOW_HOP_SECONDS,
    ANALYSIS_WINDOW_SECONDS,
    MAX_ANALYSIS_WINDOWS,
    SILENCE_DBFS_THRESHOLD,
    compute_signal_metrics,
    load_audio_waveform,
    load_model_checkpoint,
)
from Preproccesing.feature_extractor import extract_log_mel_spectrogram

MANIFEST = ROOT / "eval" / "external_manifest.csv"
ONLINE_DIR = ROOT / "scratch" / "online_sounds"
MODELS = ["sresnet", "resnet18", "vgg16", "cnngru"]


def load_items():
    items = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = ROOT / "eval" / "external" / row["file"]
            if p.is_file():
                items.append({
                    "name": row["file"],
                    "path": p,
                    "source": row["source_dataset"],
                    "allowed_labels": json.loads(row.get("allowed_labels") or "[]"),
                    "in_scope": row["in_scope"].strip().lower() == "true",
                })
    online_specs = [
        ("audio_01.ogg", [0], True),
        ("audio_02.ogg", [5], True),
        ("audio_03.ogg", [1], True),
        ("audio_04.ogg", [19], True),
        ("audio_05_ambulance_siren.ogg", [42], True),
        ("audio_07_car_horn.wav", [43], True),
        ("audio_06_speech.wav", [], False),
        ("audio_08_sine_tone.wav", [], False),
    ]
    for fname, allowed, in_scope in online_specs:
        p = ONLINE_DIR / fname
        if p.is_file():
            items.append({
                "name": f"online/{fname}",
                "path": p,
                "source": "online",
                "allowed_labels": allowed,
                "in_scope": in_scope,
            })
    return items


def build_windows(y: np.ndarray, tile_cutoff_sec: float, trim_silence: bool):
    win = int(22050 * ANALYSIS_WINDOW_SECONDS)
    hop = int(22050 * ANALYSIS_WINDOW_HOP_SECONDS)
    if y.size < win:
        if tile_cutoff_sec > 0 and y.size < int(22050 * tile_cutoff_sec) and y.size > 0:
            y_core = y
            if trim_silence:
                peak = float(np.max(np.abs(y)))
                if peak > 1e-6:
                    active = np.where(np.abs(y) > peak * 0.02)[0]
                    if active.size > int(22050 * 0.15):
                        pad_margin = int(22050 * 0.05)
                        s_idx = max(0, int(active[0]) - pad_margin)
                        e_idx = min(y.size, int(active[-1]) + pad_margin)
                        y_core = y[s_idx:e_idx]
            reps = int(np.ceil(win / max(1, y_core.size)))
            return [np.tile(y_core, reps)[:win]]
        return [np.pad(y, (0, win - y.size))]
    starts = list(range(0, max(1, y.size - win + 1), hop))
    final_start = y.size - win
    if starts[-1] != final_start:
        starts.append(final_start)
    if len(starts) > MAX_ANALYSIS_WINDOWS:
        starts = starts[:MAX_ANALYSIS_WINDOWS - 1] + [final_start]
    return [y[s:s + win] for s in starts]


def run_variant(items, entries, tile_cutoff_sec: float, trim_silence: bool, rms_weight: bool, ens_weights: list[float]):
    stats = {m: {"ok1": 0, "ok5": 0, "n": 0, "ood_s": [], "ood_y": []} for m in MODELS + ["ensemble"]}
    silence_rms = 10.0 ** (SILENCE_DBFS_THRESHOLD / 20.0)

    for item in items:
        y, sr = load_audio_waveform(item["path"], target_sr=22050, target_duration=None)
        metrics = compute_signal_metrics(y, sr)
        gate_ood = (
            metrics["rms"] < silence_rms
            or (
                metrics["flatness_raw"] < 0.0005
                and metrics["spectral_peak_ratio"] > 0.45
                and metrics["dominant_frequency_frame_count"] >= 3
                and metrics["dominant_frequency_median_hz"] > 30.0
                and metrics["dominant_frequency_spread_hz"] < 5.0
            )
        )
        if gate_ood:
            for m in MODELS + ["ensemble"]:
                stats[m]["ood_y"].append(0 if item["in_scope"] else 1)
                stats[m]["ood_s"].append(1.0)
                if item["in_scope"]:
                    stats[m]["n"] += 1
            continue

        windows = build_windows(y, tile_cutoff_sec=tile_cutoff_sec, trim_silence=trim_silence)
        w_rms = np.array([float(np.sqrt(np.mean(w ** 2))) if w.size else 0.0 for w in windows], dtype=np.float64)
        if rms_weight and len(windows) > 1 and np.max(w_rms) > 1e-8:
            weights = np.where(w_rms >= max(silence_rms, 0.15 * np.max(w_rms)), (w_rms / np.max(w_rms)), 0.0)
            if np.sum(weights) <= 1e-8:
                weights = np.ones(len(windows))
            weights /= np.sum(weights)
        else:
            weights = np.ones(len(windows)) / len(windows)

        # Precompute features for each window once
        feats = []
        for w in windows:
            pk = float(np.max(np.abs(w))) if w.size else 0.0
            mw = w / pk if pk > 1e-8 else w
            feats.append(torch.from_numpy(extract_log_mel_spectrogram(mw, sr=22050)).unsqueeze(0).unsqueeze(0).float())
        batch_x = torch.cat(feats, dim=0)

        per_model_probs = []
        for m_name, entry in zip(MODELS, entries):
            temp = max(0.2, float(entry.get("temperature", 1.0)))
            with torch.no_grad():
                logits = entry["model"](batch_x)
                w_probs = torch.softmax(logits / temp, dim=1).numpy()
            probs = np.sum(w_probs * weights[:, None], axis=0)
            probs /= np.sum(probs)
            per_model_probs.append(probs)

            top1 = int(np.argmax(probs))
            top5 = np.argsort(probs)[-5:]
            stats[m_name]["ood_y"].append(0 if item["in_scope"] else 1)
            stats[m_name]["ood_s"].append(float(1.0 - probs[top1]))
            if item["in_scope"]:
                stats[m_name]["n"] += 1
                if top1 in item["allowed_labels"]:
                    stats[m_name]["ok1"] += 1
                if any(t in item["allowed_labels"] for t in top5):
                    stats[m_name]["ok5"] += 1

        ens_p = np.zeros(50, dtype=np.float64)
        for w, p in zip(ens_weights, per_model_probs):
            ens_p += w * p
        ens_p /= np.sum(ens_p)
        top1 = int(np.argmax(ens_p))
        top5 = np.argsort(ens_p)[-5:]
        stats["ensemble"]["ood_y"].append(0 if item["in_scope"] else 1)
        stats["ensemble"]["ood_s"].append(float(1.0 - ens_p[top1]))
        if item["in_scope"]:
            stats["ensemble"]["n"] += 1
            if top1 in item["allowed_labels"]:
                stats["ensemble"]["ok1"] += 1
            if any(t in item["allowed_labels"] for t in top5):
                stats["ensemble"]["ok5"] += 1

    return stats


def main():
    items = load_items()
    entries = [load_model_checkpoint(m) for m in MODELS]
    configs = [
        ("baseline (zero-pad, uniform win, equal)", 0.0, False, False, [0.25, 0.25, 0.25, 0.25]),
        ("tile<2.5s no-trim + rms + equal", 2.5, False, True, [0.25, 0.25, 0.25, 0.25]),
        ("tile<2.5s no-trim + rms + [0.30,0.28,0.32,0.10]", 2.5, False, True, [0.30, 0.28, 0.32, 0.10]),
        ("tile<3.5s no-trim + rms + [0.30,0.28,0.32,0.10]", 3.5, False, True, [0.30, 0.28, 0.32, 0.10]),
        ("tile<5.0s no-trim + rms + [0.30,0.28,0.32,0.10]", 5.0, False, True, [0.30, 0.28, 0.32, 0.10]),
        ("tile<5.0s no-trim + rms + equal", 5.0, False, True, [0.25, 0.25, 0.25, 0.25]),
    ]
    for label, tc, ts, rms_w, ew in configs:
        st = run_variant(items, entries, tc, ts, rms_w, ew)
        print(f"\n=== {label} ===")
        for m in ["sresnet", "resnet18", "vgg16", "cnngru", "ensemble"]:
            s = st[m]
            t1 = 100.0 * s["ok1"] / s["n"]
            t5 = 100.0 * s["ok5"] / s["n"]
            auc = roc_auc_score(s["ood_y"], s["ood_s"])
            print(f"  {m:10s}: Top-1={t1:5.2f}% | Top-5={t5:5.2f}% | OOD AUROC={auc:.4f}")


if __name__ == "__main__":
    main()

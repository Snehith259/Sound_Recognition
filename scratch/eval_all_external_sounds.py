"""
Comprehensive External Sound Benchmark across ALL 183 external audio files:
- 174 clips in eval/external/ (dev + final splits; BirdNET ambient, Animal-Sound-Dataset, IBM MAX)
- 9 clips in scratch/online_sounds/ (Wikimedia Commons / GitHub Open Audio: .ogg and .wav)
Tests all 6 models via the production inference entrypoint:
  ast, ensemble, sresnet, resnet18, vgg16, cnngru
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score

from Preproccesing.acoustic_inference import run_acoustic_inference

MODELS = ["ast", "ensemble", "sresnet", "resnet18", "vgg16", "cnngru"]
MANIFEST = ROOT / "eval" / "external_manifest.csv"
ONLINE_DIR = ROOT / "scratch" / "online_sounds"
OUT_PATH = ROOT / "scratch" / "external_all_models_report.json"


def load_external_items():
    items = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = ROOT / "eval" / "external" / row["file"]
            if not p.is_file():
                continue
            items.append({
                "name": row["file"],
                "path": p,
                "source": row["source_dataset"],
                "split": row["split"],
                "label": row["label"],
                "allowed_labels": json.loads(row.get("allowed_labels") or "[]"),
                "in_scope": row["in_scope"].strip().lower() == "true",
            })

    online_specs = [
        ("audio_01.ogg", "dog", [0], True),
        ("audio_02.ogg", "cat", [5], True),
        ("audio_03.ogg", "rooster", [1], True),
        ("audio_04.ogg", "thunderstorm", [19], True),
        ("audio_05_ambulance_siren.ogg", "siren", [42], True),
        ("audio_07_car_horn.wav", "car_horn", [43], True),
        ("audio_06_speech.wav", "ood_speech", [], False),
        ("audio_08_sine_tone.wav", "ood_sine_tone", [], False),
    ]
    for fname, label, allowed, in_scope in online_specs:
        p = ONLINE_DIR / fname
        if p.is_file():
            items.append({
                "name": f"online/{fname}",
                "path": p,
                "source": "online_real_world",
                "split": "online",
                "label": label,
                "allowed_labels": allowed,
                "in_scope": in_scope,
            })
    return items


def main():
    items = load_external_items()
    print(f"Loaded {len(items)} external audio files ({sum(i['in_scope'] for i in items)} in-scope, {sum(not i['in_scope'] for i in items)} OOD)", flush=True)

    summary = {}
    detailed = []

    for model in MODELS:
        t_model = time.time()
        records = []
        for idx, item in enumerate(items, 1):
            res = run_acoustic_inference(item["path"], model_name=model, confidence_threshold=None)
            preds = res.get("predictions") or []
            cat = res.get("category") or {}
            top1_id = cat.get("id")
            top5_ids = [p.get("category", {}).get("id") for p in preds[:5]]
            is_gate_ood = bool(res.get("isOutOfScope", False))
            conf = res.get("confidence")
            ood_score = float(res.get("oodScore", 1.0 if is_gate_ood else 0.0))

            top1_ok = (not is_gate_ood) and (top1_id in item["allowed_labels"]) if item["in_scope"] else is_gate_ood
            top5_ok = (not is_gate_ood) and any(tid in item["allowed_labels"] for tid in top5_ids) if item["in_scope"] else is_gate_ood

            records.append({
                "name": item["name"],
                "source": item["source"],
                "split": item["split"],
                "label": item["label"],
                "in_scope": item["in_scope"],
                "predicted": cat.get("name"),
                "predicted_id": top1_id,
                "confidence": conf,
                "ood_score": ood_score,
                "is_gate_ood": is_gate_ood,
                "top1_ok": bool(top1_ok),
                "top5_ok": bool(top5_ok),
                "latency_ms": float(res.get("inferenceTime", 0.0)),
            })
            if idx % 60 == 0:
                print(f"  [{model}] {idx}/{len(items)} clips evaluated...", flush=True)

        in_scope_rows = [r for r in records if r["in_scope"]]
        ood_rows = [r for r in records if not r["in_scope"]]
        dev_in = [r for r in in_scope_rows if r["split"] == "dev"]
        final_in = [r for r in in_scope_rows if r["split"] == "final"]
        online_in = [r for r in in_scope_rows if r["split"] == "online"]

        by_source = {}
        for src in sorted({r["source"] for r in in_scope_rows}):
            sub = [r for r in in_scope_rows if r["source"] == src]
            by_source[src] = {
                "n": len(sub),
                "top1_pct": round(100.0 * sum(r["top1_ok"] for r in sub) / len(sub), 2),
                "top5_pct": round(100.0 * sum(r["top5_ok"] for r in sub) / len(sub), 2),
            }

        # OOD AUROC using ood_score (1 - top_prob; 1.0 for rule gate)
        y_ood = [0 if r["in_scope"] else 1 for r in records]
        s_ood = [r["ood_score"] for r in records]
        auroc = round(float(roc_auc_score(y_ood, s_ood)), 4)

        # OOD rejection at confidence threshold 0.40 and 0.50 (plus rule gates)
        ood_rej_40 = round(100.0 * sum((r["is_gate_ood"] or (r["confidence"] or 0.0) < 0.40) for r in ood_rows) / len(ood_rows), 2)
        in_false_rej_40 = round(100.0 * sum((r["is_gate_ood"] or (r["confidence"] or 0.0) < 0.40) for r in in_scope_rows) / len(in_scope_rows), 2)

        correct_confs = [r["confidence"] for r in in_scope_rows if r["top1_ok"] and r["confidence"] is not None]
        wrong_confs = [r["confidence"] for r in in_scope_rows if (not r["top1_ok"]) and r["confidence"] is not None]

        m_sum = {
            "in_scope_n": len(in_scope_rows),
            "in_scope_top1_pct": round(100.0 * sum(r["top1_ok"] for r in in_scope_rows) / len(in_scope_rows), 2),
            "in_scope_top5_pct": round(100.0 * sum(r["top5_ok"] for r in in_scope_rows) / len(in_scope_rows), 2),
            "dev_in_scope_top1_pct": round(100.0 * sum(r["top1_ok"] for r in dev_in) / len(dev_in), 2) if dev_in else None,
            "final_holdout_top1_pct": round(100.0 * sum(r["top1_ok"] for r in final_in) / len(final_in), 2) if final_in else None,
            "online_top1_pct": round(100.0 * sum(r["top1_ok"] for r in online_in) / len(online_in), 2) if online_in else None,
            "by_source": by_source,
            "ood_n": len(ood_rows),
            "ood_auroc": auroc,
            "ood_rejection_at_040_pct": ood_rej_40,
            "in_scope_false_rejection_at_040_pct": in_false_rej_40,
            "mean_conf_correct": round(float(np.mean(correct_confs)), 4) if correct_confs else None,
            "mean_conf_wrong": round(float(np.mean(wrong_confs)), 4) if wrong_confs else None,
            "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in records])), 1),
            "elapsed_sec": round(time.time() - t_model, 1),
        }
        summary[model] = m_sum
        detailed.extend([{"model": model, **r} for r in records])
        print(
            f"DONE {model:10s} | In-Scope Top-1: {m_sum['in_scope_top1_pct']:5.2f}% | Top-5: {m_sum['in_scope_top5_pct']:5.2f}% | "
            f"OOD AUROC: {auroc:.4f} | Conf(ok/bad): {m_sum['mean_conf_correct']}/{m_sum['mean_conf_wrong']} | "
            f"Latency: {m_sum['mean_latency_ms']}ms ({m_sum['elapsed_sec']}s)",
            flush=True,
        )

    OUT_PATH.write_text(json.dumps({"summary": summary, "records": detailed}, indent=2), encoding="utf-8")
    print(f"\nSaved full external benchmark report to {OUT_PATH}")


if __name__ == "__main__":
    main()

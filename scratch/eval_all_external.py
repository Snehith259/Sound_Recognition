"""
Comprehensive Evaluation of 175 External Online Audio Files across All Models
Tests: AST, ResNet-18, VGG-16, SResNet, CNN-GRU, Ensemble
Evaluates:
- 24 BirdNET ambient soundscapes
- 132 YashNita animal recordings (dogs, cats, cows, frogs, sheep, hens, birds, OOD animals)
- 19 IBM MAX Audio Classifier samples (typing, engine, siren, thunder, train, car horn, etc.)
"""
import os
import sys
import csv
import json
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Preproccesing.acoustic_inference import run_acoustic_inference, ESC50_CATEGORIES

def evaluate():
    manifest_path = ROOT / "eval" / "external_manifest.csv"
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} clips from {manifest_path}")

    models = ["ast", "resnet18", "vgg16", "sresnet", "cnngru", "ensemble"]
    results = {m: [] for m in models}

    for idx, row in enumerate(rows, start=1):
        rel_path = row["file"]
        full_path = ROOT / "eval" / "external" / rel_path
        if not full_path.exists():
            print(f"WARNING: File missing {full_path}")
            continue

        raw_bytes = full_path.read_bytes()
        expected_label = row["label"]
        allowed = json.loads(row.get("allowed_labels") or "[]")
        in_scope = row["in_scope"].lower() == "true"
        dataset = row["source_dataset"]

        if idx % 25 == 0 or idx == 1 or idx == len(rows):
            print(f"Processing clip {idx}/{len(rows)}: {rel_path} ({expected_label}, in_scope={in_scope})...")

        for m in models:
            t0 = time.time()
            try:
                out = run_acoustic_inference(raw_bytes, m, confidence_threshold=0.30)
                elapsed = (time.time() - t0) * 1000
                cat = out.get("category", {})
                pred_id = cat.get("id", -1)
                pred_name = cat.get("name", "unknown")
                conf = out.get("confidence")
                is_ood = out.get("isOutOfScope", False)
                gate = out.get("engine", "")

                top3 = [(p["category"]["name"], p["confidence"]) for p in out.get("predictions", [])[:3]]

                # Check correctness
                if in_scope:
                    # Correct if predicted class is in allowed labels and not OOD
                    is_correct = (not is_ood) and (pred_id in allowed)
                else:
                    # OOD is correct if flagged as OOD or class is -1
                    is_correct = is_ood or (pred_id == -1)

                results[m].append({
                    "file": rel_path,
                    "dataset": dataset,
                    "expected_label": expected_label,
                    "allowed_labels": allowed,
                    "in_scope": in_scope,
                    "pred_id": pred_id,
                    "pred_name": pred_name,
                    "confidence": conf,
                    "is_ood": is_ood,
                    "engine": gate,
                    "correct": is_correct,
                    "top3": top3,
                    "elapsed_ms": elapsed
                })
            except Exception as e:
                results[m].append({
                    "file": rel_path,
                    "dataset": dataset,
                    "expected_label": expected_label,
                    "allowed_labels": allowed,
                    "in_scope": in_scope,
                    "error": str(e),
                    "correct": False
                })

    # Summary
    print("\n" + "=" * 85)
    print(f"{'Model':<12s} | {'In-Scope Top-1':<16s} | {'OOD Rejection':<16s} | {'Overall Acc':<14s} | {'Avg Latency'}")
    print("=" * 85)

    summary = {}
    for m in models:
        m_res = results[m]
        in_scope_res = [r for r in m_res if r.get("in_scope")]
        ood_res = [r for r in m_res if not r.get("in_scope")]

        in_scope_correct = sum(1 for r in in_scope_res if r.get("correct"))
        ood_correct = sum(1 for r in ood_res if r.get("correct"))
        total_correct = sum(1 for r in m_res if r.get("correct"))

        in_acc = (in_scope_correct / len(in_scope_res) * 100) if in_scope_res else 0.0
        ood_acc = (ood_correct / len(ood_res) * 100) if ood_res else 0.0
        tot_acc = (total_correct / len(m_res) * 100) if m_res else 0.0
        avg_lat = sum(r.get("elapsed_ms", 0) for r in m_res) / len(m_res) if m_res else 0.0

        print(f"{m:<12s} | {in_scope_correct}/{len(in_scope_res)} ({in_acc:5.1f}%) | {ood_correct}/{len(ood_res)} ({ood_acc:5.1f}%) | {total_correct}/{len(m_res)} ({tot_acc:5.1f}%) | {avg_lat:6.1f} ms")

        summary[m] = {
            "in_scope_total": len(in_scope_res),
            "in_scope_correct": in_scope_correct,
            "in_scope_accuracy": in_acc,
            "ood_total": len(ood_res),
            "ood_correct": ood_correct,
            "ood_accuracy": ood_acc,
            "total_clips": len(m_res),
            "total_correct": total_correct,
            "overall_accuracy": tot_acc,
            "avg_latency_ms": avg_lat
        }

    # Save detailed report
    out_file = ROOT / "scratch" / "external_dataset_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nWrote full evaluation report to: {out_file}")

if __name__ == "__main__":
    evaluate()

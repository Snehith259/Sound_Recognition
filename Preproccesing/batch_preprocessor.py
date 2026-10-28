"""
High-performance batch preprocessing pipeline for the ESC-50 dataset.
Processes audio files concurrently:
1. Loads raw audio, converts to mono, resamples to 22,050 Hz, aligns duration to 5.0s.
2. Applies peak amplitude normalization.
3. Detects longest active acoustic segment.
4. Saves resampled 22.05 kHz audio into processed/audio_22k/.
5. Computes and saves:
   - Log-Mel Spectrogram (.npy) in processed/features/log_mel/
   - MFCC (.npy) in processed/features/mfcc/
   - Threshold-Free Recurrence Plot (.npy) in processed/features/recurrence_plot/
6. Generates full preprocessing telemetry & validation reports.
"""

import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Ensure module import works when run directly
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from Preproccesing import config
from Preproccesing.audio_processor import (
    load_and_resample,
    peak_normalize,
    detect_active_segment,
    save_audio,
)
from Preproccesing.feature_extractor import extract_all_features
from Preproccesing.dataset_splits import generate_5fold_splits


def process_single_audio(file_path: Path, meta_row: dict) -> dict:
    """Process a single audio file and save all features and normalized audio."""
    base_name = file_path.stem
    result = {
        "filename": file_path.name,
        "base_name": base_name,
        "status": "pending",
        "error": None,
    }

    try:
        # 1. Load, resample to 22,050 Hz, and duration align
        audio, sr = load_and_resample(file_path, target_sr=config.SAMPLE_RATE, target_duration=config.TARGET_DURATION)

        # 2. Peak normalization
        audio = peak_normalize(audio)

        # 3. Detect active segment
        active_seg = detect_active_segment(audio, sr=sr)

        # 4. Save normalized 22,050 Hz audio
        out_wav_path = config.PROCESSED_AUDIO_DIR / file_path.name
        save_audio(audio, out_wav_path, sr=sr)

        # 5. Extract multi-modal features
        features = extract_all_features(audio, sr=sr)
        log_mel = features["log_mel"]       # (128, 216)
        mfcc = features["mfcc"]             # (40, 216)
        rp = features["recurrence_plot"]    # (128, 128)

        # 6. Save .npy feature files
        np.save(config.LOG_MEL_DIR / f"{base_name}.npy", log_mel)
        np.save(config.MFCC_DIR / f"{base_name}.npy", mfcc)
        np.save(config.RP_DIR / f"{base_name}.npy", rp)

        result.update({
            "status": "success",
            "sample_rate": sr,
            "duration_sec": config.TARGET_DURATION,
            "samples": len(audio),
            "log_mel_shape": list(log_mel.shape),
            "mfcc_shape": list(mfcc.shape),
            "rp_shape": list(rp.shape),
            "active_start_time": active_seg["start_time"],
            "active_end_time": active_seg["end_time"],
            "active_ratio": active_seg["active_ratio"],
            "max_energy": active_seg["max_energy"],
        })
        if meta_row:
            result.update({
                "target": meta_row.get("target"),
                "category": meta_row.get("category"),
                "fold": meta_row.get("fold"),
            })

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)

    return result


def run_batch_preprocessing(
    max_workers: int = 6,
    limit: int = None,
) -> dict:
    """Run concurrent preprocessing across the entire dataset."""
    start_time = time.time()
    print("=" * 70)
    print("STARTING ESC-50 DATA PREPROCESSING PIPELINE")
    print(f"Target Sample Rate: {config.SAMPLE_RATE} Hz")
    print(f"Target Duration:    {config.TARGET_DURATION} seconds ({config.TARGET_SAMPLES} samples)")
    print(f"Feature Shapes:     Log-Mel {config.N_MELS}x216 | MFCC {config.N_MFCC}x216 | RP {config.PAA_DIM}x{config.PAA_DIM}")
    print("=" * 70)

    # 1. Generate 5-fold cross validation splits first
    print("\n[Step 1/3] Generating ESC-50 5-Fold Cross-Validation splits...")
    split_stats = generate_5fold_splits()
    print(f"Verified {split_stats['available_samples']} available samples across {config.NUM_FOLDS} folds.")

    # Load labels dataframe for metadata enrichment
    df_labels = pd.read_csv(config.LABELS_CSV)
    meta_lookup = {row["filename"]: row.to_dict() for _, row in df_labels.iterrows()}

    # List all audio files to process
    wav_files = sorted(list(config.RAW_DATA_DIR.glob("*.wav")))
    if limit:
        wav_files = wav_files[:limit]
        print(f"Processing test subset of {limit} files.")
    else:
        print(f"Processing all {len(wav_files)} dataset audio files.")

    # 2. Concurrently process files
    print(f"\n[Step 2/3] Processing audio files with {max_workers} threads...")
    results = []
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_audio, wav_file, meta_lookup.get(wav_file.name)): wav_file
            for wav_file in wav_files
        }

        total_files = len(wav_files)
        for i, future in enumerate(as_completed(future_to_file), 1):
            res = future.result()
            results.append(res)
            if res["status"] == "success":
                success_count += 1
            else:
                fail_count += 1

            if i % 100 == 0 or i == total_files:
                elapsed = time.time() - start_time
                speed = i / max(elapsed, 0.001)
                print(f"  Processed {i}/{total_files} files ({i/total_files*100:.1f}%) | "
                      f"Speed: {speed:.1f} files/sec | Success: {success_count} | Failed: {fail_count}")

    # 3. Compile report and metadata
    print("\n[Step 3/3] Compiling preprocessing telemetry and validation report...")
    elapsed_total = time.time() - start_time

    df_results = pd.DataFrame(results)
    features_manifest_path = config.PREPROCESSING_DIR / "processed_dataset_manifest.csv"
    df_results.to_csv(features_manifest_path, index=False)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(wav_files),
        "success_count": success_count,
        "fail_count": fail_count,
        "elapsed_seconds": round(elapsed_total, 2),
        "average_speed_fps": round(len(wav_files) / max(elapsed_total, 0.001), 2),
        "specifications": {
            "sample_rate": config.SAMPLE_RATE,
            "target_duration_sec": config.TARGET_DURATION,
            "target_samples": config.TARGET_SAMPLES,
            "log_mel_dimension": [config.N_MELS, 216],
            "mfcc_dimension": [config.N_MFCC, 216],
            "recurrence_plot_dimension": [config.PAA_DIM, config.PAA_DIM],
            "paa_dimension": config.PAA_DIM,
            "num_classes": 50,
            "num_folds": config.NUM_FOLDS,
        },
        "output_directories": {
            "processed_audio": str(config.PROCESSED_AUDIO_DIR),
            "log_mel_spectrograms": str(config.LOG_MEL_DIR),
            "mfcc_features": str(config.MFCC_DIR),
            "recurrence_plots": str(config.RP_DIR),
            "splits": str(config.SPLITS_DIR),
        },
    }

    report_path = config.REPORTS_DIR / "preprocessing_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPREPROCESSING COMPLETE in {elapsed_total:.2f} seconds!")
    print(f"Report saved to: {report_path}")
    print(f"Manifest saved to: {features_manifest_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESC-50 Audio Preprocessing Pipeline")
    parser.add_argument("--workers", type=int, default=6, help="Worker threads")
    parser.add_argument("--limit", type=int, default=None, help="Process subset of files")
    args = parser.parse_args()

    run_batch_preprocessing(max_workers=args.workers, limit=args.limit)

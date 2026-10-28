"""
Dataset splitting and validation protocol module:
- Implements official ESC-50 5-fold cross-validation scheme
- Ensures ZERO data leakage across folds (strict separation of takes/recordings)
- Produces training, validation, and test split manifests for each fold
"""

import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any

from . import config


def generate_5fold_splits(
    labels_csv: Path = config.LABELS_CSV,
    wav_dir: Path = config.RAW_DATA_DIR,
    output_dir: Path = config.SPLITS_DIR,
) -> Dict[str, Any]:
    """
    Read ESC-50 ground truth labels, verify audio file presence,
    and generate structured CSV splits for 5-fold cross validation.
    """
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check file existence in wav_dir
    existing_files = set(p.name for p in wav_dir.glob("*.wav"))
    df["file_exists"] = df["filename"].apply(lambda fn: fn in existing_files)

    # Filter only available files
    valid_df = df[df["file_exists"]].copy()

    # Save full verified manifest
    manifest_path = output_dir / "dataset_manifest.csv"
    valid_df.to_csv(manifest_path, index=False)

    stats = {
        "total_catalogued_samples": int(len(df)),
        "available_samples": int(len(valid_df)),
        "num_classes": int(valid_df["target"].nunique()),
        "categories": sorted(valid_df["category"].unique().tolist()),
        "fold_counts": valid_df["fold"].value_counts().to_dict(),
        "folds": {},
    }

    # Generate train / val / test splits for folds 1 to 5
    for test_fold in range(1, config.NUM_FOLDS + 1):
        # Val fold is the next cyclic fold
        val_fold = (test_fold % config.NUM_FOLDS) + 1
        train_folds = [f for f in range(1, config.NUM_FOLDS + 1) if f not in (test_fold, val_fold)]

        test_data = valid_df[valid_df["fold"] == test_fold]
        val_data = valid_df[valid_df["fold"] == val_fold]
        train_data = valid_df[valid_df["fold"].isin(train_folds)]

        # Save to CSV
        train_path = output_dir / f"fold_{test_fold}_train.csv"
        val_path = output_dir / f"fold_{test_fold}_val.csv"
        test_path = output_dir / f"fold_{test_fold}_test.csv"

        train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)
        test_data.to_csv(test_path, index=False)

        stats["folds"][f"fold_{test_fold}"] = {
            "train_samples": int(len(train_data)),
            "train_folds": train_folds,
            "val_samples": int(len(val_data)),
            "val_fold": val_fold,
            "test_samples": int(len(test_data)),
            "test_fold": test_fold,
            "class_distribution_train": int(train_data["target"].nunique()),
        }

    stats_path = output_dir / "split_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    summary = generate_5fold_splits()
    print("ESC-50 5-Fold Cross-Validation splits successfully generated:")
    print(f"Total available samples: {summary['available_samples']}")
    print(f"Folds generated: {list(summary['folds'].keys())}")

"""Vocabulary selection and controlled manifest generation for SIGNASSIST."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def select_vocabulary_from_train(
    train_csv_path: Path,
    min_samples: int = 15,
    class_mapping_path: Path | None = None,
) -> dict[str, Any]:
    """Select class vocabulary programmatically based on minimum training sample threshold.
    
    Args:
        train_csv_path: Path to data/processed/include/splits/train.csv
        min_samples: Minimum number of training samples required per class (default: 15)
        class_mapping_path: Optional path to data/processed/include/class_mapping.json
        
    Returns:
        Structured dictionary containing vocabulary selection and mappings.
    """
    train_df = pd.read_csv(train_csv_path)
    if "status" in train_df.columns:
        train_df = train_df[train_df["status"] == "clean"].copy()

    # Load original class mapping if available
    orig_class_to_id: dict[str, int] = {}
    if class_mapping_path and Path(class_mapping_path).exists():
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            orig_class_to_id = mapping_data.get("class_to_id", {})

    counts = train_df["label"].value_counts()
    selected_labels = [label for label, count in counts.items() if count >= min_samples]

    # Sort deterministically by original_class_id if available, else by label
    if orig_class_to_id:
        selected_labels.sort(key=lambda lbl: orig_class_to_id.get(lbl, 999999))
    else:
        selected_labels.sort()

    classes_list = []
    label_to_controlled_id = {}
    controlled_id_to_label = {}

    for new_id, label in enumerate(selected_labels):
        sample_count = int(counts[label])
        orig_id = orig_class_to_id.get(label, new_id)
        
        entry = {
            "controlled_class_id": new_id,
            "original_class_id": orig_id,
            "label": label,
            "train_sample_count": sample_count,
        }
        classes_list.append(entry)
        label_to_controlled_id[label] = new_id
        controlled_id_to_label[str(new_id)] = label

    vocab_data = {
        "dataset": "INCLUDE",
        "min_samples_threshold": min_samples,
        "total_dataset_classes": len(counts),
        "num_controlled_classes": len(classes_list),
        "classes": classes_list,
        "label_to_controlled_id": label_to_controlled_id,
        "controlled_id_to_label": controlled_id_to_label,
    }

    return vocab_data


def save_vocabulary(vocab_data: dict[str, Any], output_path: Path) -> None:
    """Save vocabulary selection JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)


def load_vocabulary(vocab_path: Path) -> dict[str, Any]:
    """Load vocabulary selection JSON artifact."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_split_manifest(
    split_csv_path: Path,
    vocab_data: dict[str, Any],
    output_csv_path: Path,
) -> pd.DataFrame:
    """Filter split manifest (train, validation, or test) using selected vocabulary."""
    df = pd.read_csv(split_csv_path)

    # Ensure clean status
    if "status" in df.columns:
        df = df[df["status"] == "clean"].copy()

    label_to_id = vocab_data["label_to_controlled_id"]
    filtered_df = df[df["label"].isin(label_to_id)].copy()

    # Assign controlled_class_id without modifying original label
    filtered_df["controlled_class_id"] = filtered_df["label"].map(label_to_id).astype(int)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_csv_path, index=False)
    return filtered_df


def verify_controlled_manifests(
    controlled_dir: Path,
    vocab_data: dict[str, Any],
) -> dict[str, Any]:
    """Verify controlled manifests for data leakage, class alignment, and balance.
    
    Returns:
        Verification metrics and statistics report.
    """
    train_path = controlled_dir / "train.csv"
    val_path = controlled_dir / "validation.csv"
    test_path = controlled_dir / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(f"Missing one or more controlled manifest files in {controlled_dir}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Check status
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if "status" in df.columns and (df["status"] != "clean").any():
            raise ValueError(f"Quarantined/non-clean samples found in controlled {name}.csv")

    # Check leakage
    train_paths = set(train_df["video_path"])
    val_paths = set(val_df["video_path"])
    test_paths = set(test_df["video_path"])

    if train_paths & val_paths:
        raise ValueError("Train/Validation video leakage detected in controlled manifests!")
    if train_paths & test_paths:
        raise ValueError("Train/Test video leakage detected in controlled manifests!")
    if val_paths & test_paths:
        raise ValueError("Validation/Test video leakage detected in controlled manifests!")

    # Check class coverage
    train_classes = set(train_df["label"])
    val_classes = set(val_df["label"])
    test_classes = set(test_df["label"])

    allowed_labels = set(vocab_data["label_to_controlled_id"].keys())
    if not train_classes.issubset(allowed_labels):
        raise ValueError("Train manifest contains classes outside selected vocabulary!")

    missing_from_train = val_classes - train_classes
    if missing_from_train:
        raise ValueError(f"Validation classes missing from training: {sorted(missing_from_train)}")

    missing_test_from_train = test_classes - train_classes
    if missing_test_from_train:
        raise ValueError(f"Test classes missing from training: {sorted(missing_test_from_train)}")

    # Statistics
    train_counts = train_df["label"].value_counts()
    val_counts = val_df["label"].value_counts()
    test_counts = test_df["label"].value_counts()

    stats = {
        "num_controlled_classes": vocab_data["num_controlled_classes"],
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "total_controlled_samples": len(train_df) + len(val_df) + len(test_df),
        "train_class_counts": {
            "min": int(train_counts.min()),
            "max": int(train_counts.max()),
            "mean": float(train_counts.mean()),
            "std": float(train_counts.std()),
        },
        "validation_class_counts": {
            "min": int(val_counts.min()),
            "max": int(val_counts.max()),
            "mean": float(val_counts.mean()),
            "std": float(val_counts.std()),
        },
        "test_class_counts": {
            "min": int(test_counts.min()),
            "max": int(test_counts.max()),
            "mean": float(test_counts.mean()),
            "std": float(test_counts.std()),
        },
        "leakage_free": True,
        "quarantine_free": True,
    }

    return stats

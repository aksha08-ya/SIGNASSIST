"""Create deterministic, leakage-free train/validation/test splits for INCLUDE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SEED = 42


def load_unique_clean_manifest(path: Path) -> pd.DataFrame:
    """Load the clean manifest and retain exactly one row per physical video."""
    frame = pd.read_csv(path)

    required = {
        "video_path",
        "label",
        "parent_label",
        "include_50",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required manifest columns: {sorted(missing)}")

    clean = frame[frame["status"] == "clean"].copy()

    # A physical video must occur only once in the modeling dataset.
    clean = clean.drop_duplicates(subset=["video_path"], keep="first")

    if clean["video_path"].duplicated().any():
        raise ValueError("Duplicate physical videos remain after deduplication.")

    return clean.reset_index(drop=True)


def _allocate_counts(n: int) -> tuple[int, int, int]:
    """Allocate samples as train/validation/test with minimum test coverage."""
    if n < 3:
        raise ValueError(f"At least 3 samples are required, got {n}.")

    if n == 3:
        return 1, 1, 1

    if n == 4:
        return 2, 1, 1

    if n == 5:
        return 3, 1, 1

    test = max(1, round(n * 0.15))
    validation = max(1, round(n * 0.15))
    train = n - validation - test

    if train < validation:
        train, validation = validation, train

    return train, validation, test


def create_splits(
    manifest: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Create deterministic class-stratified splits."""
    if manifest["video_path"].duplicated().any():
        raise ValueError("Input manifest contains duplicate physical videos.")

    rng = __import__("numpy").random.default_rng(seed)

    train_parts = []
    validation_parts = []
    test_parts = []

    for label in sorted(manifest["label"].unique()):
        group = manifest[manifest["label"] == label].copy()

        if len(group) < 3:
            raise ValueError(
                f"Class {label!r} has only {len(group)} videos; "
                "at least 3 are required for three-way evaluation."
            )

        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)

        n_train, n_validation, n_test = _allocate_counts(len(indices))

        train_indices = indices[:n_train]
        validation_indices = indices[n_train : n_train + n_validation]
        test_indices = indices[n_train + n_validation :]

        train_parts.append(group.loc[train_indices])
        validation_parts.append(group.loc[validation_indices])
        test_parts.append(group.loc[test_indices])

    splits = {
        "train": pd.concat(train_parts, ignore_index=True),
        "validation": pd.concat(validation_parts, ignore_index=True),
        "test": pd.concat(test_parts, ignore_index=True),
    }

    for split_name, frame in splits.items():
        frame = frame.sort_values(["label", "video_path"]).reset_index(drop=True)
        frame["split"] = split_name
        splits[split_name] = frame

    _validate_splits(splits)

    return splits


def _validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Validate class coverage and physical-video isolation."""
    expected = {"train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"Expected splits {expected}, got {set(splits)}")

    paths = {
        name: set(frame["video_path"])
        for name, frame in splits.items()
    }

    if paths["train"] & paths["validation"]:
        raise ValueError("Train/validation video leakage detected.")

    if paths["train"] & paths["test"]:
        raise ValueError("Train/test video leakage detected.")

    if paths["validation"] & paths["test"]:
        raise ValueError("Validation/test video leakage detected.")

    classes = {
        name: set(frame["label"])
        for name, frame in splits.items()
    }

    missing_train = classes["validation"] - classes["train"]
    if missing_train:
        raise ValueError(
            f"Validation classes missing from train: {sorted(missing_train)}"
        )

    missing_test = classes["test"] - classes["train"]
    if missing_test:
        raise ValueError(
            f"Test classes missing from train: {sorted(missing_test)}"
        )


def write_splits(
    splits: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Write train/validation/test manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    manifest_path = Path("data/processed/include/manifest.csv")
    output_dir = Path("data/processed/include/splits")

    manifest = load_unique_clean_manifest(manifest_path)
    splits = create_splits(manifest, seed=DEFAULT_SEED)
    write_splits(splits, output_dir)

    print(f"Unique clean videos: {len(manifest)}")
    print(f"Classes: {manifest['label'].nunique()}")

    for name, frame in splits.items():
        print(
            f"{name}: videos={len(frame)}, "
            f"classes={frame['label'].nunique()}"
        )


if __name__ == "__main__":
    main()

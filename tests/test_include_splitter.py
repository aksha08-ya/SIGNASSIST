from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from src.data.include_splitter import (
    _allocate_counts,
    create_splits,
    load_unique_clean_manifest,
)


MANIFEST = Path("data/processed/include/manifest.csv")


def test_allocation_for_small_classes():
    assert _allocate_counts(3) == (1, 1, 1)
    assert _allocate_counts(4) == (2, 1, 1)
    assert _allocate_counts(5) == (3, 1, 1)


def test_allocation_rejects_too_few_samples():
    with pytest.raises(ValueError):
        _allocate_counts(2)


def test_include_split_sizes_and_class_coverage():
    manifest = load_unique_clean_manifest(MANIFEST)
    splits = create_splits(manifest, seed=42)

    assert len(manifest) == 4210
    assert len(splits["train"]) == 2980
    assert len(splits["validation"]) == 615
    assert len(splits["test"]) == 615

    assert splits["train"]["label"].nunique() == 262
    assert splits["validation"]["label"].nunique() == 262
    assert splits["test"]["label"].nunique() == 262


def test_no_video_leakage():
    manifest = load_unique_clean_manifest(MANIFEST)
    splits = create_splits(manifest, seed=42)

    train = set(splits["train"]["video_path"])
    validation = set(splits["validation"]["video_path"])
    test = set(splits["test"]["video_path"])

    assert not train & validation
    assert not train & test
    assert not validation & test


def test_validation_and_test_classes_exist_in_training():
    manifest = load_unique_clean_manifest(MANIFEST)
    splits = create_splits(manifest, seed=42)

    train_classes = set(splits["train"]["label"])

    assert set(splits["validation"]["label"]) <= train_classes
    assert set(splits["test"]["label"]) <= train_classes


def test_split_is_deterministic():
    manifest = load_unique_clean_manifest(MANIFEST)

    first = create_splits(manifest, seed=42)
    second = create_splits(manifest, seed=42)

    for name in ("train", "validation", "test"):
        pdt.assert_frame_equal(first[name], second[name])


def test_different_seed_changes_allocation():
    manifest = load_unique_clean_manifest(MANIFEST)

    first = create_splits(manifest, seed=42)
    second = create_splits(manifest, seed=123)

    first_train = set(first["train"]["video_path"])
    second_train = set(second["train"]["video_path"])

    assert first_train != second_train

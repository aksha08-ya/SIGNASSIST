"""Tests for controlled manifest generation and data leakage prevention."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.data.vocabulary import (
    filter_split_manifest,
    select_vocabulary_from_train,
    verify_controlled_manifests,
)


@pytest.fixture
def controlled_setup(tmp_path: Path) -> tuple[Path, Path]:
    splits_dir = tmp_path / "splits"
    controlled_dir = tmp_path / "controlled"
    splits_dir.mkdir()

    train_data = {
        "video_path": [f"train/vid_{i}.mov" for i in range(30)],
        "label": ["Class_A"] * 18 + ["Class_B"] * 12,
        "status": ["clean"] * 30,
    }
    val_data = {
        "video_path": [f"val/vid_{i}.mov" for i in range(10)],
        "label": ["Class_A"] * 6 + ["Class_B"] * 4,
        "status": ["clean"] * 10,
    }
    test_data = {
        "video_path": [f"test/vid_{i}.mov" for i in range(10)],
        "label": ["Class_A"] * 6 + ["Class_B"] * 4,
        "status": ["clean"] * 10,
    }

    pd.DataFrame(train_data).to_csv(splits_dir / "train.csv", index=False)
    pd.DataFrame(val_data).to_csv(splits_dir / "validation.csv", index=False)
    pd.DataFrame(test_data).to_csv(splits_dir / "test.csv", index=False)

    return splits_dir, controlled_dir


def test_controlled_manifest_filtering(controlled_setup: tuple[Path, Path]) -> None:
    splits_dir, controlled_dir = controlled_setup
    vocab = select_vocabulary_from_train(splits_dir / "train.csv", min_samples=10)

    train_f = filter_split_manifest(splits_dir / "train.csv", vocab, controlled_dir / "train.csv")
    val_f = filter_split_manifest(splits_dir / "validation.csv", vocab, controlled_dir / "validation.csv")
    test_f = filter_split_manifest(splits_dir / "test.csv", vocab, controlled_dir / "test.csv")

    assert len(train_f) == 30
    assert len(val_f) == 10
    assert len(test_f) == 10
    assert "controlled_class_id" in train_f.columns


def test_controlled_leakage_and_class_verification(controlled_setup: tuple[Path, Path]) -> None:
    splits_dir, controlled_dir = controlled_setup
    vocab = select_vocabulary_from_train(splits_dir / "train.csv", min_samples=10)

    filter_split_manifest(splits_dir / "train.csv", vocab, controlled_dir / "train.csv")
    filter_split_manifest(splits_dir / "validation.csv", vocab, controlled_dir / "validation.csv")
    filter_split_manifest(splits_dir / "test.csv", vocab, controlled_dir / "test.csv")

    stats = verify_controlled_manifests(controlled_dir, vocab)
    assert stats["leakage_free"] is True
    assert stats["quarantine_free"] is True
    assert stats["train_samples"] == 30
    assert stats["validation_samples"] == 10
    assert stats["test_samples"] == 10

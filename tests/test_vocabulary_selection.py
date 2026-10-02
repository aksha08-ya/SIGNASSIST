"""Tests for controlled vocabulary selection and class mapping."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.vocabulary import (
    load_vocabulary,
    save_vocabulary,
    select_vocabulary_from_train,
)


@pytest.fixture
def sample_train_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "train.csv"
    data = {
        "video_path": [f"path/vid_{i}.mov" for i in range(50)],
        "label": (
            ["Class_A"] * 20 +
            ["Class_B"] * 15 +
            ["Class_C"] * 10 +
            ["Class_D"] * 5
        ),
        "parent_label": ["Group1"] * 50,
        "include_50": [True] * 50,
        "status": ["clean"] * 50,
        "split": ["train"] * 50,
    }
    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path


def test_vocabulary_selection_threshold(sample_train_csv: Path) -> None:
    vocab = select_vocabulary_from_train(sample_train_csv, min_samples=15)
    assert vocab["num_controlled_classes"] == 2
    assert "Class_A" in vocab["label_to_controlled_id"]
    assert "Class_B" in vocab["label_to_controlled_id"]
    assert "Class_C" not in vocab["label_to_controlled_id"]
    assert "Class_D" not in vocab["label_to_controlled_id"]


def test_vocabulary_preserves_original_labels(sample_train_csv: Path) -> None:
    vocab = select_vocabulary_from_train(sample_train_csv, min_samples=15)
    for entry in vocab["classes"]:
        assert "controlled_class_id" in entry
        assert "original_class_id" in entry
        assert "label" in entry
        assert "train_sample_count" in entry
        assert entry["label"] in ["Class_A", "Class_B"]


def test_save_and_load_vocabulary(sample_train_csv: Path, tmp_path: Path) -> None:
    vocab = select_vocabulary_from_train(sample_train_csv, min_samples=15)
    save_path = tmp_path / "vocab.json"
    save_vocabulary(vocab, save_path)

    loaded = load_vocabulary(save_path)
    assert loaded["num_controlled_classes"] == vocab["num_controlled_classes"]
    assert loaded["label_to_controlled_id"] == vocab["label_to_controlled_id"]

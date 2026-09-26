"""Unit tests for SIGNASSIST Phase 16 EvaluationDataset and LOSO split engine."""

import numpy as np
import pytest

from src.utils.evaluation_dataset import (
    EvaluationDataset,
    EvaluationSample,
    LOSOFold,
)


def test_valid_dataset_construction() -> None:
    """Tests constructing a valid EvaluationDataset with static and dynamic samples."""
    s1 = EvaluationSample(
        sample_id="s1",
        label=0,
        signer_id="signer_A",
        features=np.random.randn(225).astype(np.float32),
        metadata={"modality": "static"},
    )
    s2 = EvaluationSample(
        sample_id="s2",
        label=1,
        signer_id="signer_B",
        features=np.random.randn(5, 60, 115).astype(np.float32),
        metadata={"modality": "dynamic"},
    )

    dataset = EvaluationDataset(samples=[s1, s2])

    assert len(dataset) == 2
    assert dataset[0].sample_id == "s1"
    assert dataset[1].sample_id == "s2"
    assert dataset.signer_ids == ["signer_A", "signer_B"]
    assert dataset.unique_signers == ["signer_A", "signer_B"]


def test_static_and_dynamic_feature_shapes() -> None:
    """Tests model-agnostic feature shape support (1D static, 3D/4D dynamic)."""
    s_static = EvaluationSample(
        sample_id="static_1",
        label=0,
        signer_id="signer_1",
        features=np.zeros(225, dtype=np.float32),
    )
    s_dynamic_3d = EvaluationSample(
        sample_id="dynamic_3d",
        label=1,
        signer_id="signer_2",
        features=np.zeros((5, 60, 115), dtype=np.float32),
    )
    s_dynamic_4d = EvaluationSample(
        sample_id="dynamic_4d",
        label=2,
        signer_id="signer_1",
        features=np.zeros((1, 5, 60, 115), dtype=np.float32),
    )

    dataset = EvaluationDataset([s_static, s_dynamic_3d, s_dynamic_4d])

    assert dataset[0].features.shape == (225,)
    assert dataset[1].features.shape == (5, 60, 115)
    assert dataset[2].features.shape == (1, 5, 60, 115)


def test_duplicate_sample_id_rejection() -> None:
    """Tests that duplicate sample IDs raise a ValueError."""
    s1 = EvaluationSample(
        sample_id="dup_id", label=0, signer_id="signer_A", features=np.zeros(10)
    )
    s2 = EvaluationSample(
        sample_id="dup_id", label=1, signer_id="signer_B", features=np.zeros(10)
    )

    with pytest.raises(ValueError, match="[Dd]uplicate sample_id"):
        EvaluationDataset([s1, s2])


def test_invalid_sample_and_signer_ids() -> None:
    """Tests validation error on empty or whitespace sample_id and signer_id."""
    features = np.zeros(5)

    with pytest.raises(ValueError, match="sample_id"):
        EvaluationSample(sample_id="", label=0, signer_id="signer_1", features=features)

    with pytest.raises(ValueError, match="sample_id"):
        EvaluationSample(sample_id="   ", label=0, signer_id="signer_1", features=features)

    with pytest.raises(ValueError, match="signer_id"):
        EvaluationSample(sample_id="s1", label=0, signer_id="", features=features)

    with pytest.raises(ValueError, match="signer_id"):
        EvaluationSample(sample_id="s1", label=0, signer_id="   ", features=features)


def test_invalid_label_rejection() -> None:
    """Tests rejection of negative or non-integral labels."""
    features = np.zeros(5)

    with pytest.raises(ValueError, match="label"):
        EvaluationSample(sample_id="s1", label=-1, signer_id="signer_1", features=features)

    with pytest.raises(ValueError, match="label"):
        EvaluationSample(sample_id="s1", label=1.5, signer_id="signer_1", features=features)  # type: ignore[arg-type]


def test_empty_dataset_rejection() -> None:
    """Tests that an empty dataset sequence raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        EvaluationDataset([])


def test_deterministic_unique_signer_ordering() -> None:
    """Tests that unique_signers returns a sorted list in deterministic order."""
    samples = [
        EvaluationSample(sample_id="s1", label=0, signer_id="Zebra", features=np.zeros(2)),
        EvaluationSample(sample_id="s2", label=1, signer_id="Alpha", features=np.zeros(2)),
        EvaluationSample(sample_id="s3", label=0, signer_id="Beta", features=np.zeros(2)),
        EvaluationSample(sample_id="s4", label=1, signer_id="Alpha", features=np.zeros(2)),
    ]

    dataset = EvaluationDataset(samples)
    assert dataset.unique_signers == ["Alpha", "Beta", "Zebra"]


def test_loso_fold_generation_and_isolation() -> None:
    """Tests LOSO fold generation, train/test isolation, and 100% sample coverage."""
    s1 = EvaluationSample(sample_id="s1", label=0, signer_id="Signer_1", features=np.zeros(2))
    s2 = EvaluationSample(sample_id="s2", label=1, signer_id="Signer_1", features=np.zeros(2))
    s3 = EvaluationSample(sample_id="s3", label=0, signer_id="Signer_2", features=np.zeros(2))
    s4 = EvaluationSample(sample_id="s4", label=1, signer_id="Signer_3", features=np.zeros(2))

    dataset = EvaluationDataset([s1, s2, s3, s4])
    folds = dataset.get_loso_folds()

    assert len(folds) == 3
    expected_signers = ["Signer_1", "Signer_2", "Signer_3"]
    assert [f.held_out_signer for f in folds] == expected_signers

    # Fold 1: Held out Signer_1
    f1 = folds[0]
    assert f1.held_out_signer == "Signer_1"
    assert [s.sample_id for s in f1.test_samples] == ["s1", "s2"]
    assert [s.sample_id for s in f1.train_samples] == ["s3", "s4"]

    # Verify no signer overlap between train and test
    for fold in folds:
        test_signers = set(s.signer_id for s in fold.test_samples)
        train_signers = set(s.signer_id for s in fold.train_samples)
        assert len(test_signers.intersection(train_signers)) == 0
        assert test_signers == {fold.held_out_signer}

        # Verify complete coverage across train + test indices
        all_indices = set(fold.train_indices).union(set(fold.test_indices))
        assert all_indices == set(range(len(dataset)))


def test_fewer_than_two_signers_rejection() -> None:
    """Tests that LOSO fold generation raises ValueError when < 2 unique signers exist."""
    s1 = EvaluationSample(sample_id="s1", label=0, signer_id="Single_Signer", features=np.zeros(2))
    s2 = EvaluationSample(sample_id="s2", label=1, signer_id="Single_Signer", features=np.zeros(2))

    dataset = EvaluationDataset([s1, s2])
    with pytest.raises(ValueError, match="at least 2 unique signers"):
        dataset.get_loso_folds()


def test_metadata_preservation() -> None:
    """Tests that optional metadata is preserved and does not alter dataset/fold behavior."""
    s1 = EvaluationSample(
        sample_id="s1",
        label=0,
        signer_id="Signer_1",
        features=np.zeros(2),
        metadata={"session": "A", "lighting": "bright"},
    )
    s2 = EvaluationSample(
        sample_id="s2",
        label=1,
        signer_id="Signer_2",
        features=np.zeros(2),
        metadata={"session": "B"},
    )

    dataset = EvaluationDataset([s1, s2])
    assert dataset[0].metadata == {"session": "A", "lighting": "bright"}
    assert dataset[1].metadata == {"session": "B"}

    folds = dataset.get_loso_folds()
    assert len(folds) == 2


def test_conversion_helpers() -> None:
    """Tests labels_array, signer_ids_list, sample_ids_list, and features_list helpers."""
    s1 = EvaluationSample(sample_id="s1", label=0, signer_id="Signer_1", features=np.ones(3))
    s2 = EvaluationSample(sample_id="s2", label=2, signer_id="Signer_2", features=np.zeros(3))

    dataset = EvaluationDataset([s1, s2])

    np.testing.assert_array_equal(dataset.labels_array(), np.array([0, 2], dtype=np.int64))
    assert dataset.signer_ids_list() == ["Signer_1", "Signer_2"]
    assert dataset.sample_ids_list() == ["s1", "s2"]

    features_list = dataset.features_list()
    assert len(features_list) == 2
    np.testing.assert_array_equal(features_list[0], np.ones(3))
    np.testing.assert_array_equal(features_list[1], np.zeros(3))


def test_deterministic_repeated_fold_generation() -> None:
    """Tests that get_loso_folds is completely deterministic across repeated calls."""
    s1 = EvaluationSample(sample_id="s1", label=0, signer_id="B", features=np.zeros(2))
    s2 = EvaluationSample(sample_id="s2", label=1, signer_id="A", features=np.zeros(2))

    dataset = EvaluationDataset([s1, s2])

    folds1 = dataset.get_loso_folds()
    folds2 = dataset.get_loso_folds()

    assert len(folds1) == len(folds2) == 2
    assert [f.held_out_signer for f in folds1] == [f.held_out_signer for f in folds2]
    assert [f.train_indices for f in folds1] == [f.train_indices for f in folds2]
    assert [f.test_indices for f in folds1] == [f.test_indices for f in folds2]

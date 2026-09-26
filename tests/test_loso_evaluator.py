"""Unit tests for SIGNASSIST Phase 16 LOSOEvaluator and LOSO cross-validation runner."""

import json
from pathlib import Path
from typing import Any, List
import numpy as np
import pytest

from src.utils.evaluation_dataset import EvaluationDataset, EvaluationSample
from src.utils.loso_evaluator import (
    LOSOEvaluator,
    LOSOEvaluationResult,
    LOSOFoldResult,
)
from src.utils.metrics import MetricsEvaluator


@pytest.fixture
def synthetic_loso_dataset() -> EvaluationDataset:
    """Fixture providing a deterministic 3-signer synthetic EvaluationDataset."""
    samples = [
        EvaluationSample(sample_id="s01", label=0, signer_id="Signer_1", features=np.zeros(10)),
        EvaluationSample(sample_id="s02", label=1, signer_id="Signer_1", features=np.zeros(10)),
        EvaluationSample(sample_id="s03", label=0, signer_id="Signer_2", features=np.zeros(10)),
        EvaluationSample(sample_id="s04", label=1, signer_id="Signer_2", features=np.zeros(10)),
        EvaluationSample(sample_id="s05", label=0, signer_id="Signer_3", features=np.zeros(10)),
        EvaluationSample(sample_id="s06", label=1, signer_id="Signer_3", features=np.zeros(10)),
    ]
    return EvaluationDataset(samples)


def mock_perfect_inference(
    train_samples: List[EvaluationSample], test_samples: List[EvaluationSample]
) -> List[int]:
    """Deterministic mock inference function returning true test labels."""
    return [s.label for s in test_samples]


def test_basic_loso_execution(synthetic_loso_dataset: EvaluationDataset) -> None:
    """Tests basic LOSOEvaluator execution."""
    evaluator = LOSOEvaluator()
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    assert isinstance(result, LOSOEvaluationResult)
    assert result.num_signers == 3
    assert result.num_samples == 6
    assert result.aggregate_metrics.accuracy == 1.0


def test_one_fold_per_signer(synthetic_loso_dataset: EvaluationDataset) -> None:
    """Tests that exactly one fold is generated per unique signer."""
    evaluator = LOSOEvaluator()
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    assert len(result.folds) == 3
    fold_signers = [f.held_out_signer for f in result.folds]
    assert fold_signers == ["Signer_1", "Signer_2", "Signer_3"]


def test_held_out_signer_isolation(synthetic_loso_dataset: EvaluationDataset) -> None:
    """Tests that held-out signer is isolated in test set with no train/test overlap."""
    evaluator = LOSOEvaluator()
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    for fold in result.folds:
        test_ids = set(fold.test_sample_ids)
        train_ids = set(fold.train_sample_ids)
        assert len(test_ids.intersection(train_ids)) == 0


def test_inference_callback_receives_correct_samples(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests that inference_fn receives exact training and test samples for each fold."""
    received_folds = []

    def tracking_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[int]:
        received_folds.append((
            [s.signer_id for s in train_s],
            [s.signer_id for s in test_s],
        ))
        return [s.label for s in test_s]

    evaluator = LOSOEvaluator()
    evaluator.evaluate(synthetic_loso_dataset, tracking_inference)

    assert len(received_folds) == 3
    # Check Fold 1 (held out Signer_1)
    train_signers, test_signers = received_folds[0]
    assert set(test_signers) == {"Signer_1"}
    assert "Signer_1" not in train_signers


def test_prediction_count_mismatch_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests ValueError when inference_fn returns wrong prediction count."""

    def bad_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[int]:
        return [0]  # Only 1 prediction for 2 test samples

    evaluator = LOSOEvaluator()
    with pytest.raises(ValueError, match="Prediction count .* does not match"):
        evaluator.evaluate(synthetic_loso_dataset, bad_inference)


def test_invalid_prediction_shape_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests ValueError when inference_fn returns 2D array."""

    def bad_shape_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> np.ndarray:
        return np.zeros((len(test_s), 1), dtype=int)

    evaluator = LOSOEvaluator()
    with pytest.raises(ValueError, match="Predictions must be a 1D sequence"):
        evaluator.evaluate(synthetic_loso_dataset, bad_shape_inference)


def test_negative_prediction_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests ValueError when predictions contain negative integer values."""

    def negative_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[int]:
        return [-1 for _ in test_s]

    evaluator = LOSOEvaluator()
    with pytest.raises(ValueError, match="negative label values"):
        evaluator.evaluate(synthetic_loso_dataset, negative_inference)


def test_non_integral_prediction_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests ValueError when predictions contain fractional floating point values."""

    def float_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[float]:
        return [0.5 for _ in test_s]

    evaluator = LOSOEvaluator()
    with pytest.raises(ValueError, match="non-integer numeric labels"):
        evaluator.evaluate(synthetic_loso_dataset, float_inference)


def test_deterministic_repeated_execution(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests that repeated evaluations produce identical dictionary outputs."""
    evaluator = LOSOEvaluator()
    res1 = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)
    res2 = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    assert res1.to_dict() == res2.to_dict()


def test_aggregate_metrics_equals_concatenated_predictions(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests that aggregate metrics match direct MetricsEvaluator evaluation on concatenated fold predictions."""

    def mixed_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[int]:
        # Return label 0 for all test samples
        return [0 for _ in test_s]

    evaluator = LOSOEvaluator()
    result = evaluator.evaluate(synthetic_loso_dataset, mixed_inference)

    # Manual aggregation
    all_true = [s.label for s in synthetic_loso_dataset]
    all_pred = [0 for _ in synthetic_loso_dataset]

    direct_metrics = MetricsEvaluator().evaluate(all_true, all_pred)
    assert result.aggregate_metrics.to_dict() == direct_metrics.to_dict()


def test_per_fold_metrics_exist(synthetic_loso_dataset: EvaluationDataset) -> None:
    """Tests that each fold result contains valid per-fold metrics."""
    evaluator = LOSOEvaluator()
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    for fold_res in result.folds:
        assert isinstance(fold_res, LOSOFoldResult)
        assert fold_res.metrics.accuracy == 1.0
        assert fold_res.test_sample_count == len(fold_res.test_sample_ids)


def test_optional_num_classes_behavior(synthetic_loso_dataset: EvaluationDataset) -> None:
    """Tests passing explicit num_classes to LOSOEvaluator."""
    evaluator = LOSOEvaluator(num_classes=5)
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    assert result.num_classes == 5
    assert result.aggregate_metrics.num_classes == 5


def test_optional_class_names_behavior(synthetic_loso_dataset: EvaluationDataset) -> None:
    """Tests passing explicit class_names list to LOSOEvaluator."""
    class_names = ["SIGN_A", "SIGN_B"]
    evaluator = LOSOEvaluator(num_classes=2, class_names=class_names)
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    assert result.class_names == class_names
    assert result.aggregate_metrics.class_names == class_names


def test_json_serialization_no_features(
    synthetic_loso_dataset: EvaluationDataset, tmp_path: Path
) -> None:
    """Tests JSON serialization and confirms no raw feature arrays exist in JSON output."""
    evaluator = LOSOEvaluator()
    result = evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)

    output_file = tmp_path / "loso_results.json"
    json_str = result.to_json(output_path=output_file)

    assert output_file.exists()
    assert "features" not in json_str
    parsed = json.loads(json_str)
    assert parsed["num_signers"] == 3
    assert len(parsed["folds"]) == 3


def test_non_callable_inference_fn_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests TypeError when passing a non-callable object as inference_fn."""
    evaluator = LOSOEvaluator()
    with pytest.raises(TypeError, match="inference_fn must be a callable"):
        evaluator.evaluate(synthetic_loso_dataset, "not_a_function")  # type: ignore[arg-type]


def test_less_than_two_signers_rejection() -> None:
    """Tests ValueError when evaluating a dataset with fewer than 2 signers."""
    samples = [
        EvaluationSample(sample_id="s1", label=0, signer_id="Signer_1", features=np.zeros(2)),
        EvaluationSample(sample_id="s2", label=1, signer_id="Signer_1", features=np.zeros(2)),
    ]
    single_signer_dataset = EvaluationDataset(samples)

    evaluator = LOSOEvaluator()
    with pytest.raises(ValueError, match="at least 2 unique signers"):
        evaluator.evaluate(single_signer_dataset, mock_perfect_inference)


def test_train_test_sample_id_overlap_defensive_check(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests defensive check rejecting train/test sample ID overlap."""
    evaluator = LOSOEvaluator()

    from unittest.mock import patch
    bad_fold = synthetic_loso_dataset.get_loso_folds()[0]
    # Create a train sample with the same sample_id as a test sample but different signer_id
    overlap_sample = EvaluationSample(
        sample_id=bad_fold.test_samples[0].sample_id,
        label=0,
        signer_id="Other_Signer",
        features=np.zeros(10),
    )
    bad_fold.train_samples.append(overlap_sample)

    with patch.object(EvaluationDataset, "get_loso_folds", return_value=[bad_fold, bad_fold]):
        with pytest.raises(ValueError, match="sample ID overlap"):
            evaluator.evaluate(synthetic_loso_dataset, mock_perfect_inference)


def test_integer_and_float_integer_predictions_accepted(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests that int lists, NumPy int arrays, and float arrays with integer values are all accepted."""
    evaluator = LOSOEvaluator()

    # Float array with integer values
    def float_int_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> np.ndarray:
        return np.array([float(s.label) for s in test_s], dtype=np.float32)

    res_float = evaluator.evaluate(synthetic_loso_dataset, float_int_inference)
    assert res_float.aggregate_metrics.accuracy == 1.0

    # NumPy int64 array
    def numpy_int_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> np.ndarray:
        return np.array([s.label for s in test_s], dtype=np.int64)

    res_numpy = evaluator.evaluate(synthetic_loso_dataset, numpy_int_inference)
    assert res_numpy.aggregate_metrics.accuracy == 1.0


def test_nan_and_inf_prediction_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests ValueError rejection when predictions contain NaN, positive infinity, or negative infinity."""
    evaluator = LOSOEvaluator()

    def nan_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[float]:
        return [np.nan for _ in test_s]

    with pytest.raises(ValueError, match="NaN or Infinity values"):
        evaluator.evaluate(synthetic_loso_dataset, nan_inference)

    def pos_inf_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[float]:
        return [np.inf for _ in test_s]

    with pytest.raises(ValueError, match="NaN or Infinity values"):
        evaluator.evaluate(synthetic_loso_dataset, pos_inf_inference)

    def neg_inf_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[float]:
        return [-np.inf for _ in test_s]

    with pytest.raises(ValueError, match="NaN or Infinity values"):
        evaluator.evaluate(synthetic_loso_dataset, neg_inf_inference)


def test_string_and_object_prediction_rejection(
    synthetic_loso_dataset: EvaluationDataset,
) -> None:
    """Tests ValueError rejection when predictions contain non-numeric strings or object lists."""
    evaluator = LOSOEvaluator()

    def string_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[str]:
        return [f"class_{s.label}" for s in test_s]

    with pytest.raises(ValueError, match="must be numeric"):
        evaluator.evaluate(synthetic_loso_dataset, string_inference)  # type: ignore[arg-type]

    def object_inference(train_s: List[EvaluationSample], test_s: List[EvaluationSample]) -> List[Any]:
        return [None for _ in test_s]

    with pytest.raises(ValueError, match="must be numeric"):
        evaluator.evaluate(synthetic_loso_dataset, object_inference)


def test_cli_demo_mode_execution(capsys: pytest.CaptureFixture[str]) -> None:
    """Tests running evaluate_loso CLI script in --demo mode."""
    from scripts.evaluate_loso import main as cli_main

    exit_code = cli_main(["--demo"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "[DEMONSTRATION/TEST ONLY - Synthetic Dataset Execution]" in captured.out
    assert "Folds Evaluated : 3" in captured.out


def test_cli_non_demo_clean_exit(capsys: pytest.CaptureFixture[str]) -> None:
    """Tests running evaluate_loso CLI script without --demo flag."""
    from scripts.evaluate_loso import main as cli_main

    exit_code = cli_main([])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Notice: Real LOSO evaluation requires an explicit EvaluationDataset" in captured.out
    assert "DEMONSTRATION/TEST ONLY" not in captured.out

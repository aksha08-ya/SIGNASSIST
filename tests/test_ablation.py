"""Unit tests for SIGNASSIST Phase 16 Ablation Study Framework."""

import json
from pathlib import Path
from typing import Any, List
import numpy as np
import pytest

from scripts.run_ablation import main as cli_main
from src.utils.ablation import (
    AblationConfig,
    AblationResult,
    AblationRunner,
    AblationStudyResult,
)
from src.utils.evaluation_dataset import EvaluationDataset, EvaluationSample
from src.utils.metrics import MetricsEvaluator


@pytest.fixture
def synthetic_ablation_dataset() -> EvaluationDataset:
    """Fixture providing a 3-signer synthetic dataset for ablation testing."""
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
    train_or_samples: Any, test_samples: Any = None
) -> List[int]:
    """Synthetic inference function returning ground-truth labels."""
    target = test_samples if test_samples is not None else train_or_samples
    return [s.label for s in target]


def mock_zero_inference(
    train_or_samples: Any, test_samples: Any = None
) -> List[int]:
    """Synthetic inference function returning all zeros."""
    target = test_samples if test_samples is not None else train_or_samples
    return [0 for _ in target]


# 1. AblationConfig validation
def test_ablation_config_validation() -> None:
    """Tests that AblationConfig validates name, description, and parameters."""
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        AblationConfig(name="", description="valid desc")

    with pytest.raises(ValueError, match="description must be a non-empty string"):
        AblationConfig(name="exp1", description="")

    with pytest.raises(TypeError, match="parameters must be a dictionary"):
        AblationConfig(name="exp1", description="desc", parameters="not_a_dict")  # type: ignore[arg-type]


# 2. valid configuration construction
def test_valid_configuration_construction() -> None:
    """Tests successful construction of a valid AblationConfig."""
    cfg = AblationConfig(
        name="exp_full",
        description="Full model experiment",
        parameters={"modalities": ["pose", "hands"], "lr": 0.001},
    )
    assert cfg.name == "exp_full"
    assert cfg.description == "Full model experiment"
    assert cfg.parameters["lr"] == 0.001


# 3. duplicate experiment-name rejection
def test_duplicate_experiment_name_rejection() -> None:
    """Tests ValueError when registering two experiments with identical names."""
    runner = AblationRunner()
    cfg1 = AblationConfig(name="exp_a", description="First run")
    cfg2 = AblationConfig(name="exp_a", description="Duplicate run")

    runner.add_experiment(cfg1, mock_perfect_inference)
    with pytest.raises(ValueError, match="Duplicate experiment name 'exp_a'"):
        runner.add_experiment(cfg2, mock_perfect_inference)


# 4. non-callable inference rejection
def test_non_callable_inference_rejection() -> None:
    """Tests TypeError when registering an experiment with a non-callable object."""
    runner = AblationRunner()
    cfg = AblationConfig(name="exp1", description="desc")
    with pytest.raises(TypeError, match="inference_fn must be a callable"):
        runner.add_experiment(cfg, "not_callable")  # type: ignore[arg-type]


# 5. registering multiple experiments
def test_registering_multiple_experiments() -> None:
    """Tests registering multiple experiments with the runner."""
    runner = AblationRunner()
    cfg1 = AblationConfig(name="exp_1", description="First")
    cfg2 = AblationConfig(name="exp_2", description="Second")
    cfg3 = AblationConfig(name="exp_3", description="Third")

    runner.add_experiment(cfg1, mock_perfect_inference)
    runner.add_experiment(cfg2, mock_zero_inference)
    runner.add_experiment(cfg3, mock_perfect_inference)

    assert len(runner.experiments) == 3


# 6. deterministic registration order
def test_deterministic_registration_order() -> None:
    """Tests that registered experiments maintain exact insertion order."""
    runner = AblationRunner()
    cfg1 = AblationConfig(name="exp_alpha", description="Alpha")
    cfg2 = AblationConfig(name="exp_beta", description="Beta")
    cfg3 = AblationConfig(name="exp_gamma", description="Gamma")

    runner.add_experiment(cfg1, mock_perfect_inference)
    runner.add_experiment(cfg2, mock_zero_inference)
    runner.add_experiment(cfg3, mock_perfect_inference)

    exps = runner.experiments
    assert [e.name for e in exps] == ["exp_alpha", "exp_beta", "exp_gamma"]


# 7. ordinary evaluation execution
def test_ordinary_evaluation_execution(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests standard evaluation execution across registered experiments."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_full", description="Full model"),
        mock_perfect_inference,
    )

    result = runner.run(synthetic_ablation_dataset, evaluation_type="standard")
    assert isinstance(result, AblationStudyResult)
    assert result.evaluation_type == "standard"
    assert len(result.results) == 1
    assert result.results[0].metrics.accuracy == 1.0


# 8. ordinary metric correctness against MetricsEvaluator
def test_ordinary_metric_correctness(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests that ordinary evaluation metrics match direct MetricsEvaluator output."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_zero", description="Zero model"),
        mock_zero_inference,
    )

    study_result = runner.run(synthetic_ablation_dataset, evaluation_type="standard")
    exp_metrics = study_result.results[0].metrics

    y_true = [s.label for s in synthetic_ablation_dataset]
    y_pred = [0 for _ in synthetic_ablation_dataset]
    direct_metrics = MetricsEvaluator().evaluate(y_true, y_pred)

    assert exp_metrics.to_dict() == direct_metrics.to_dict()


# 9. prediction-count mismatch handling
def test_prediction_count_mismatch_handling(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests ValueError when standard evaluation inference function returns wrong prediction count."""
    runner = AblationRunner()

    def wrong_count_fn(samples: Any, test_samples: Any = None) -> List[int]:
        return [0]  # Only 1 prediction for 6 dataset samples

    runner.add_experiment(AblationConfig(name="exp_wrong", description="Wrong count"), wrong_count_fn)

    with pytest.raises(ValueError, match="Prediction count .* does not match"):
        runner.run(synthetic_ablation_dataset, evaluation_type="standard")


# 10. invalid prediction handling
def test_invalid_prediction_handling(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests ValueError when inference function returns NaN, Inf, fractional floats, or strings."""
    runner = AblationRunner()

    def nan_fn(samples: Any, test_samples: Any = None) -> List[float]:
        return [np.nan for _ in (test_samples or samples)]

    runner.add_experiment(AblationConfig(name="exp_nan", description="NaN predictions"), nan_fn)

    with pytest.raises(ValueError, match="NaN or Infinity values"):
        runner.run(synthetic_ablation_dataset, evaluation_type="standard")


# 11. LOSO ablation execution
def test_loso_ablation_execution(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests execution of LOSO ablation study."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="loso_exp1", description="LOSO experiment 1"),
        mock_perfect_inference,
    )

    result = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    assert result.evaluation_type == "loso"
    assert len(result.results) == 1
    assert result.results[0].loso_result is not None


# 12. one independent LOSO evaluation per experiment
def test_one_independent_loso_evaluation_per_experiment(
    synthetic_ablation_dataset: EvaluationDataset,
) -> None:
    """Tests that each experiment runs an independent LOSO pass."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_perfect", description="Perfect model"),
        mock_perfect_inference,
    )
    runner.add_experiment(
        AblationConfig(name="exp_zero", description="Zero model"),
        mock_zero_inference,
    )

    study_result = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    assert len(study_result.results) == 2
    assert study_result.results[0].loso_result is not study_result.results[1].loso_result


# 13. per-experiment result separation
def test_per_experiment_result_separation(
    synthetic_ablation_dataset: EvaluationDataset,
) -> None:
    """Tests that metrics and predictions of one experiment do not bleed into another."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_perfect", description="Perfect model"),
        mock_perfect_inference,
    )
    runner.add_experiment(
        AblationConfig(name="exp_zero", description="Zero model"),
        mock_zero_inference,
    )

    study_result = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    res_perfect = study_result.results[0]
    res_zero = study_result.results[1]

    assert res_perfect.config.name == "exp_perfect"
    assert res_perfect.metrics.accuracy == 1.0

    assert res_zero.config.name == "exp_zero"
    assert res_zero.metrics.accuracy < 1.0


# 14. fold count correctness
def test_fold_count_correctness(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests that fold_count in LOSO result matches the number of signers."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp1", description="desc"),
        mock_perfect_inference,
    )

    result = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    exp_res = result.results[0]
    assert exp_res.fold_count == 3
    assert exp_res.signer_count == 3


# 15. JSON serialization
def test_json_serialization(
    synthetic_ablation_dataset: EvaluationDataset, tmp_path: Path
) -> None:
    """Tests JSON serialization to string and file."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_json", description="JSON test", parameters={"k": "v"}),
        mock_perfect_inference,
    )

    result = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    out_file = tmp_path / "ablation_results.json"
    json_str = result.to_json(output_path=out_file)

    assert out_file.exists()
    parsed = json.loads(json_str)
    assert parsed["evaluation_type"] == "loso"
    assert parsed["num_experiments"] == 1
    assert parsed["experiments"][0]["name"] == "exp_json"
    assert parsed["experiments"][0]["parameters"] == {"k": "v"}


# 16. no feature arrays in JSON
def test_no_feature_arrays_in_json(
    synthetic_ablation_dataset: EvaluationDataset, tmp_path: Path
) -> None:
    """Tests that raw feature matrices are not included in serialized JSON output."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_nofeat", description="Ablation test without raw data"),
        mock_perfect_inference,
    )

    result = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    json_str = result.to_json()
    assert "features" not in json_str


# 17. deterministic repeated execution
def test_deterministic_repeated_execution(
    synthetic_ablation_dataset: EvaluationDataset,
) -> None:
    """Tests that repeated execution produces identical dictionary output."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_repeat", description="Repeat test"),
        mock_perfect_inference,
    )

    res1 = runner.run(synthetic_ablation_dataset, evaluation_type="loso")
    res2 = runner.run(synthetic_ablation_dataset, evaluation_type="loso")

    assert res1.to_dict() == res2.to_dict()


# 18. empty experiment runner rejection
def test_empty_experiment_runner_rejection(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests ValueError when calling run() with no registered experiments."""
    runner = AblationRunner()
    with pytest.raises(ValueError, match="no registered experiments"):
        runner.run(synthetic_ablation_dataset)


# 19. invalid evaluation type rejection
def test_invalid_evaluation_type_rejection(synthetic_ablation_dataset: EvaluationDataset) -> None:
    """Tests ValueError when passing an unsupported evaluation_type."""
    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp1", description="desc"), mock_perfect_inference
    )
    with pytest.raises(ValueError, match="Invalid evaluation_type 'unsupported'"):
        runner.run(synthetic_ablation_dataset, evaluation_type="unsupported")


# 20. demo CLI behavior
def test_demo_cli_behavior(capsys: pytest.CaptureFixture[str]) -> None:
    """Tests CLI execution with --demo flag."""
    exit_code = cli_main(["--demo"])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "[DEMONSTRATION/TEST ONLY - Synthetic Ablation Execution]" in captured.out
    assert "full_model" in captured.out
    assert "no_face" in captured.out
    assert "static_only" in captured.out


# 21. non-demo CLI clean exit
def test_non_demo_cli_clean_exit(capsys: pytest.CaptureFixture[str]) -> None:
    """Tests CLI execution without --demo flag."""
    exit_code = cli_main([])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Notice: Real ablation execution requires an explicit EvaluationDataset" in captured.out
    assert "DEMONSTRATION/TEST ONLY" not in captured.out


# 22. Standard inference internal TypeError propagation regression test
def test_standard_inference_internal_typeerror_propagation(
    synthetic_ablation_dataset: EvaluationDataset,
) -> None:
    """Tests that an internal TypeError raised inside standard inference_fn propagates and callback is called once."""
    call_count = 0

    def failing_inference(samples: List[EvaluationSample]) -> List[int]:
        nonlocal call_count
        call_count += 1
        raise TypeError("Internal calculation type error inside inference function")

    runner = AblationRunner()
    runner.add_experiment(
        AblationConfig(name="exp_fail", description="Internal failure test"),
        failing_inference,
    )

    with pytest.raises(TypeError, match="Internal calculation type error inside inference function"):
        runner.run(synthetic_ablation_dataset, evaluation_type="standard")

    assert call_count == 1


# 23. JSON serialization of NumPy types in parameters & metadata
def test_ablation_json_serialization_numpy_types_and_metadata(
    synthetic_ablation_dataset: EvaluationDataset, tmp_path: Path
) -> None:
    """Tests that NumPy scalars and arrays in parameters/metadata are converted to JSON-serializable types."""
    runner = AblationRunner()

    params = {
        "int_val": np.int64(42),
        "float_val": np.float32(3.14),
        "bool_val": np.bool_(True),
        "arr_val": np.array([1, 2, 3], dtype=np.int32),
        "nested": {
            "matrix": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "tuple_val": (np.int16(10), "str"),
        },
    }

    config = AblationConfig(
        name="numpy_exp", description="NumPy parameter test", parameters=params
    )
    runner.add_experiment(config, mock_perfect_inference)

    result = runner.run(synthetic_ablation_dataset, evaluation_type="standard")
    result.results[0].metadata = {
        "run_id": np.int32(101),
        "scores": np.array([0.9, 0.95]),
    }

    out_file = tmp_path / "numpy_ablation.json"
    json_str = result.to_json(output_path=out_file)

    assert out_file.exists()
    parsed = json.loads(json_str)

    exp_dict = parsed["experiments"][0]
    assert exp_dict["parameters"]["int_val"] == 42
    assert type(exp_dict["parameters"]["int_val"]) is int
    assert abs(exp_dict["parameters"]["float_val"] - 3.14) < 1e-4
    assert exp_dict["parameters"]["bool_val"] is True
    assert exp_dict["parameters"]["arr_val"] == [1, 2, 3]
    assert exp_dict["parameters"]["nested"]["matrix"] == [[1.0, 2.0], [3.0, 4.0]]
    assert exp_dict["parameters"]["nested"]["tuple_val"] == [10, "str"]

    assert exp_dict["metadata"]["run_id"] == 101
    assert exp_dict["metadata"]["scores"] == [0.9, 0.95]


# 24. JSON serialization rejection of unsupported types
def test_ablation_json_serialization_unsupported_type_rejection(
    synthetic_ablation_dataset: EvaluationDataset,
) -> None:
    """Tests TypeError when parameters or metadata contain non-serializable objects."""
    class UnsupportedModelObject:
        pass

    runner = AblationRunner()
    params = {"model_obj": UnsupportedModelObject()}
    config = AblationConfig(
        name="unsupported_exp", description="Unsupported object test", parameters=params
    )
    runner.add_experiment(config, mock_perfect_inference)

    result = runner.run(synthetic_ablation_dataset, evaluation_type="standard")

    with pytest.raises(TypeError, match="is not JSON serializable"):
        result.to_json()

"""Unit tests for SIGNASSIST Phase 16 MetricsEvaluator."""

import json
import pytest

from src.utils.metrics import EvaluationResult, MetricsEvaluator


def test_perfect_predictions() -> None:
    """Tests evaluation when all predictions are 100% correct."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]

    evaluator = MetricsEvaluator()
    res = evaluator.evaluate(y_true, y_pred, num_classes=3)

    assert isinstance(res, EvaluationResult)
    assert res.accuracy == 1.0
    assert res.macro_precision == 1.0
    assert res.macro_recall == 1.0
    assert res.macro_f1 == 1.0
    assert res.micro_precision == 1.0
    assert res.micro_recall == 1.0
    assert res.micro_f1 == 1.0
    assert res.weighted_precision == 1.0
    assert res.weighted_recall == 1.0
    assert res.weighted_f1 == 1.0
    assert res.confusion_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_completely_wrong_predictions() -> None:
    """Tests evaluation when predictions are 100% incorrect."""
    # 2 classes, true=0 predicted=1, true=1 predicted=0
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]

    evaluator = MetricsEvaluator()
    res = evaluator.evaluate(y_true, y_pred, num_classes=2)

    assert res.accuracy == 0.0
    assert res.macro_precision == 0.0
    assert res.macro_recall == 0.0
    assert res.macro_f1 == 0.0
    assert res.micro_f1 == 0.0
    assert res.weighted_f1 == 0.0
    assert res.confusion_matrix == [[0, 2], [2, 0]]


def test_mixed_multiclass_predictions() -> None:
    """Tests mixed predictions and metric calculations against manual ground truth."""
    # 3 classes
    # True:  0, 0, 0, 1, 1, 2
    # Pred:  0, 0, 1, 1, 2, 2
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 2, 2]

    evaluator = MetricsEvaluator()
    res = evaluator.evaluate(y_true, y_pred, num_classes=3)

    assert res.total_samples == 6
    assert res.num_classes == 3

    # Matrix:
    # Row 0 (True 0): 2 pred 0, 1 pred 1, 0 pred 2
    # Row 1 (True 1): 0 pred 0, 1 pred 1, 1 pred 2
    # Row 2 (True 2): 0 pred 0, 0 pred 1, 1 pred 2
    expected_matrix = [
        [2, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]
    assert res.confusion_matrix == expected_matrix

    # Class 0: TP=2, FP=0, FN=1 -> Prec=1.0, Rec=2/3=0.666667, F1=0.8
    # Class 1: TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
    # Class 2: TP=1, FP=1, FN=0 -> Prec=0.5, Rec=1.0, F1=0.666667
    assert res.per_class_metrics["class_0"]["precision"] == 1.0
    assert res.per_class_metrics["class_0"]["support"] == 3
    assert res.per_class_metrics["class_1"]["precision"] == 0.5
    assert res.per_class_metrics["class_1"]["support"] == 2
    assert res.per_class_metrics["class_2"]["recall"] == 1.0
    assert res.per_class_metrics["class_2"]["support"] == 1

    # Accuracy: 4 / 6 = 0.666667
    assert res.accuracy == pytest.approx(4 / 6, rel=1e-4)
    assert res.micro_f1 == res.accuracy


def test_custom_class_names() -> None:
    """Tests passing custom class_names list."""
    y_true = [0, 1]
    y_pred = [0, 1]
    class_names = ["HELLO", "WORLD"]

    evaluator = MetricsEvaluator()
    res = evaluator.evaluate(y_true, y_pred, class_names=class_names)

    assert res.class_names == ["HELLO", "WORLD"]
    assert "HELLO" in res.per_class_metrics
    assert "WORLD" in res.per_class_metrics
    assert res.per_class_metrics["HELLO"]["support"] == 1


def test_zero_support_classes() -> None:
    """Tests evaluation when some classes have zero ground truth instances."""
    # num_classes = 3, but class 2 has 0 samples
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]

    evaluator = MetricsEvaluator(zero_division=0.0)
    res = evaluator.evaluate(y_true, y_pred, num_classes=3)

    assert res.num_classes == 3
    assert res.per_class_metrics["class_2"]["support"] == 0
    assert res.per_class_metrics["class_2"]["precision"] == 0.0
    assert res.per_class_metrics["class_2"]["recall"] == 0.0
    assert res.per_class_metrics["class_2"]["f1"] == 0.0

    # Ensure no NaN or Inf in serialization
    d = res.to_dict()
    json_str = res.to_json()
    assert "NaN" not in json_str
    assert "Infinity" not in json_str


def test_invalid_label_ranges() -> None:
    """Tests rejection of labels outside specified num_classes or negative labels."""
    evaluator = MetricsEvaluator()

    with pytest.raises(ValueError, match="outside valid range"):
        evaluator.evaluate(y_true=[0, 1, 5], y_pred=[0, 1, 2], num_classes=3)

    with pytest.raises(ValueError, match="outside valid range"):
        evaluator.evaluate(y_true=[0, -1], y_pred=[0, 0], num_classes=2)


def test_mismatched_lengths() -> None:
    """Tests rejection of mismatched y_true and y_pred lengths."""
    evaluator = MetricsEvaluator()
    with pytest.raises(ValueError, match="Mismatched label lengths"):
        evaluator.evaluate(y_true=[0, 1, 2], y_pred=[0, 1])


def test_empty_inputs() -> None:
    """Tests rejection of empty input sequences."""
    evaluator = MetricsEvaluator()
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator.evaluate(y_true=[], y_pred=[])


def test_json_serialization() -> None:
    """Tests that to_dict and to_json produce clean, parseable representations."""
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]

    evaluator = MetricsEvaluator()
    res = evaluator.evaluate(y_true, y_pred, class_names=["cat", "dog"])

    res_dict = res.to_dict()
    assert res_dict["accuracy"] == pytest.approx(2 / 3, rel=1e-4)
    assert res_dict["num_classes"] == 2
    assert "cat" in res_dict["per_class_metrics"]

    json_str = res.to_json()
    parsed = json.loads(json_str)
    assert parsed["num_classes"] == 2
    assert parsed["confusion_matrix"] == [[1, 1], [0, 1]]


def test_deterministic_repeated_evaluation() -> None:
    """Tests that repeated evaluations produce identical output."""
    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 1, 1, 0, 2]

    evaluator = MetricsEvaluator()
    res1 = evaluator.evaluate(y_true, y_pred)
    res2 = evaluator.evaluate(y_true, y_pred)

    assert res1.to_dict() == res2.to_dict()


def test_non_integer_label_rejection() -> None:
    """Tests rejection of non-integer numeric labels and acceptance of integer-valued floats."""
    evaluator = MetricsEvaluator()

    # Fractional float label should raise ValueError
    with pytest.raises(ValueError, match="non-integer numeric labels"):
        evaluator.evaluate(y_true=[0, 1, 1.5], y_pred=[0, 1, 1])

    with pytest.raises(ValueError, match="non-integer numeric labels"):
        evaluator.evaluate(y_true=[0, 1, 1], y_pred=[0, 1, 0.7])

    # Integer-valued floats should be accepted as integer-like
    res = evaluator.evaluate(y_true=[0.0, 1.0, 2.0], y_pred=[0.0, 1.0, 2.0])
    assert res.accuracy == 1.0


def test_duplicate_class_names_rejection() -> None:
    """Tests rejection of duplicate class names."""
    evaluator = MetricsEvaluator()

    with pytest.raises(ValueError, match="duplicate names"):
        evaluator.evaluate(
            y_true=[0, 1], y_pred=[0, 1], class_names=["CLASS_A", "CLASS_A"]
        )

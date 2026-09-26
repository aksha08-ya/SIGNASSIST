"""Model-agnostic multiclass classification evaluation module for SIGNASSIST Phase 16.

Provides MetricsEvaluator and EvaluationResult for computing macro, micro,
and weighted precision, recall, F1, per-class metrics, accuracy, and confusion matrix
without third-party ML framework dependencies.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class EvaluationResult:
    """Container for multiclass evaluation metrics and confusion matrix.

    Attributes:
        accuracy: Overall classification accuracy in [0.0, 1.0].
        macro_precision: Unweighted mean of per-class precision scores.
        macro_recall: Unweighted mean of per-class recall scores.
        macro_f1: Unweighted mean of per-class F1 scores.
        micro_precision: Total true positives divided by total predictions.
        micro_recall: Total true positives divided by total ground truth labels.
        micro_f1: Harmonic mean of micro precision and recall.
        weighted_precision: Support-weighted mean of per-class precision scores.
        weighted_recall: Support-weighted mean of per-class recall scores.
        weighted_f1: Support-weighted mean of per-class F1 scores.
        total_samples: Total number of evaluated sample instances.
        num_classes: Total number of evaluated categories/classes.
        class_names: Ordered list of string names for each class.
        per_class_metrics: Dictionary mapping class name to precision, recall, f1, and support.
        confusion_matrix: 2D matrix of shape [num_classes, num_classes] where rows are true and cols are predicted.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    total_samples: int
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    per_class_metrics: Dict[str, Dict[str, Union[float, int]]] = field(
        default_factory=dict
    )
    confusion_matrix: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a clean JSON-serializable dictionary representation of evaluation results."""
        return {
            "accuracy": float(self.accuracy),
            "macro_precision": float(self.macro_precision),
            "macro_recall": float(self.macro_recall),
            "macro_f1": float(self.macro_f1),
            "micro_precision": float(self.micro_precision),
            "micro_recall": float(self.micro_recall),
            "micro_f1": float(self.micro_f1),
            "weighted_precision": float(self.weighted_precision),
            "weighted_recall": float(self.weighted_recall),
            "weighted_f1": float(self.weighted_f1),
            "total_samples": int(self.total_samples),
            "num_classes": int(self.num_classes),
            "class_names": list(self.class_names),
            "per_class_metrics": self.per_class_metrics,
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes evaluation results to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class MetricsEvaluator:
    """Evaluates 1D true and predicted integer labels for multiclass classification metrics."""

    def __init__(self, zero_division: float = 0.0) -> None:
        if zero_division < 0.0 or zero_division > 1.0:
            raise ValueError("zero_division must be a float between 0.0 and 1.0")
        self.zero_division = float(zero_division)

    def evaluate(
        self,
        y_true: Any,
        y_pred: Any,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """Computes multiclass classification metrics from ground truth and predictions.

        Args:
            y_true: Ground truth integer-like labels (1D array or list).
            y_pred: Predicted integer-like labels (1D array or list).
            num_classes: Optional explicit number of target classes.
            class_names: Optional list of class string identifiers of length num_classes.

        Returns:
            EvaluationResult dataclass populated with overall, per-class, and matrix results.
        """
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred cannot be None")

        def _parse_int_array(name: str, val: Any) -> np.ndarray:
            try:
                arr = np.asarray(val)
            except (ValueError, TypeError) as err:
                raise ValueError(f"Could not parse {name} as a numeric array: {err}")

            if not np.issubdtype(arr.dtype, np.number) and not np.issubdtype(arr.dtype, np.bool_):
                raise ValueError(f"{name} must contain numeric labels")

            if arr.size > 0 and not np.all(np.equal(arr, np.round(arr))):
                raise ValueError(f"{name} contains non-integer numeric labels")

            return arr.astype(np.int64)

        y_true_arr = _parse_int_array("y_true", y_true)
        y_pred_arr = _parse_int_array("y_pred", y_pred)

        if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
            raise ValueError("y_true and y_pred must be 1D 1-dimensional sequences")

        total_samples = len(y_true_arr)
        if total_samples == 0 or len(y_pred_arr) == 0:
            raise ValueError("y_true and y_pred cannot be empty sequences")

        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                f"Mismatched label lengths: len(y_true)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}"
            )

        max_true = int(np.max(y_true_arr))
        max_pred = int(np.max(y_pred_arr))

        if num_classes is None:
            num_classes = max(max_true, max_pred) + 1
        else:
            num_classes = int(num_classes)
            if num_classes <= 0:
                raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

        if np.any(y_true_arr < 0) or np.any(y_true_arr >= num_classes):
            raise ValueError(
                f"y_true contains labels outside valid range [0, {num_classes - 1}]"
            )
        if np.any(y_pred_arr < 0) or np.any(y_pred_arr >= num_classes):
            raise ValueError(
                f"y_pred contains labels outside valid range [0, {num_classes - 1}]"
            )

        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"class_names length ({len(class_names)}) does not match num_classes ({num_classes})"
                )
            names = [str(name) for name in class_names]
            if len(set(names)) != len(names):
                raise ValueError("class_names contains duplicate names")
        else:
            names = [f"class_{i}" for i in range(num_classes)]

        # Compute Confusion Matrix [num_classes, num_classes]
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true_arr, y_pred_arr):
            matrix[t, p] += 1

        per_class_metrics: Dict[str, Dict[str, Union[float, int]]] = {}
        precision_list: List[float] = []
        recall_list: List[float] = []
        f1_list: List[float] = []
        support_list: List[int] = []

        for i in range(num_classes):
            c_name = names[i]
            tp = int(matrix[i, i])
            fp = int(np.sum(matrix[:, i]) - tp)
            fn = int(np.sum(matrix[i, :]) - tp)
            support = int(np.sum(matrix[i, :]))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else float(self.zero_division)
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else float(self.zero_division)
            f1 = (
                float(2.0 * prec * rec / (prec + rec))
                if (prec + rec) > 0
                else float(self.zero_division)
            )

            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            support_list.append(support)

            per_class_metrics[c_name] = {
                "precision": float(round(prec, 6)),
                "recall": float(round(rec, 6)),
                "f1": float(round(f1, 6)),
                "support": support,
            }

        # Overall Accuracy
        total_tp = sum(int(matrix[i, i]) for i in range(num_classes))
        accuracy = float(total_tp / total_samples)

        # Macro Metrics
        macro_precision = float(np.mean(precision_list))
        macro_recall = float(np.mean(recall_list))
        macro_f1 = float(np.mean(f1_list))

        # Micro Metrics
        micro_precision = accuracy
        micro_recall = accuracy
        micro_f1 = accuracy

        # Weighted Metrics
        total_support = sum(support_list)
        if total_support > 0:
            weighted_precision = float(
                sum(support_list[i] * precision_list[i] for i in range(num_classes))
                / total_support
            )
            weighted_recall = float(
                sum(support_list[i] * recall_list[i] for i in range(num_classes))
                / total_support
            )
            weighted_f1 = float(
                sum(support_list[i] * f1_list[i] for i in range(num_classes))
                / total_support
            )
        else:
            weighted_precision = float(self.zero_division)
            weighted_recall = float(self.zero_division)
            weighted_f1 = float(self.zero_division)

        matrix_list = [list(map(int, row)) for row in matrix]

        return EvaluationResult(
            accuracy=float(round(accuracy, 6)),
            macro_precision=float(round(macro_precision, 6)),
            macro_recall=float(round(macro_recall, 6)),
            macro_f1=float(round(macro_f1, 6)),
            micro_precision=float(round(micro_precision, 6)),
            micro_recall=float(round(micro_recall, 6)),
            micro_f1=float(round(micro_f1, 6)),
            weighted_precision=float(round(weighted_precision, 6)),
            weighted_recall=float(round(weighted_recall, 6)),
            weighted_f1=float(round(weighted_f1, 6)),
            total_samples=total_samples,
            num_classes=num_classes,
            class_names=names,
            per_class_metrics=per_class_metrics,
            confusion_matrix=matrix_list,
        )

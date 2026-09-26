"""Leave-One-Signer-Out (LOSO) Cross-Validation Evaluation Runner for SIGNASSIST Phase 16.

Provides LOSOEvaluator, LOSOFoldResult, and LOSOEvaluationResult for executing
model-agnostic LOSO cross-validation with exact metric aggregation and JSON export.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from src.utils.evaluation_dataset import EvaluationDataset, EvaluationSample, LOSOFold
from src.utils.metrics import EvaluationResult, MetricsEvaluator


@dataclass
class LOSOFoldResult:
    """Result container for a single held-out signer fold in LOSO evaluation."""

    held_out_signer: str
    train_sample_count: int
    test_sample_count: int
    train_sample_ids: List[str]
    test_sample_ids: List[str]
    true_labels: List[int]
    predicted_labels: List[int]
    metrics: EvaluationResult

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary representation of the fold result."""
        return {
            "held_out_signer": str(self.held_out_signer),
            "train_sample_count": int(self.train_sample_count),
            "test_sample_count": int(self.test_sample_count),
            "train_sample_ids": list(self.train_sample_ids),
            "test_sample_ids": list(self.test_sample_ids),
            "true_labels": list(map(int, self.true_labels)),
            "predicted_labels": list(map(int, self.predicted_labels)),
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class LOSOEvaluationResult:
    """Result container for a full Leave-One-Signer-Out (LOSO) evaluation run."""

    num_signers: int
    num_samples: int
    num_classes: int
    class_names: List[str]
    folds: List[LOSOFoldResult]
    aggregate_metrics: EvaluationResult
    evaluation_type: str = "leave_one_signer_out"

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary representation of full LOSO evaluation."""
        return {
            "evaluation_type": str(self.evaluation_type),
            "num_signers": int(self.num_signers),
            "num_samples": int(self.num_samples),
            "num_classes": int(self.num_classes),
            "class_names": list(self.class_names),
            "folds": [f.to_dict() for f in self.folds],
            "aggregate_metrics": self.aggregate_metrics.to_dict(),
        }

    def to_json(
        self, output_path: Optional[Union[str, Path]] = None, indent: int = 2
    ) -> str:
        """Serializes evaluation result to JSON string and optionally writes to file."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json_str, encoding="utf-8")
        return json_str


class LOSOEvaluator:
    """Model-agnostic Leave-One-Signer-Out (LOSO) evaluation orchestrator."""

    def __init__(
        self,
        zero_division: float = 0.0,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.zero_division = float(zero_division)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.class_names = list(class_names) if class_names is not None else None
        self.metrics_evaluator = MetricsEvaluator(zero_division=self.zero_division)

    def evaluate(
        self,
        dataset: EvaluationDataset,
        inference_fn: Callable[[List[EvaluationSample], List[EvaluationSample]], Any],
    ) -> LOSOEvaluationResult:
        """Executes Leave-One-Signer-Out (LOSO) cross-validation evaluation.

        Args:
            dataset: EvaluationDataset instance containing samples.
            inference_fn: Callable taking (train_samples, test_samples) and returning predicted labels.

        Returns:
            LOSOEvaluationResult containing fold results and aggregate metrics.
        """
        if not isinstance(dataset, EvaluationDataset):
            raise TypeError(f"Expected EvaluationDataset, got {type(dataset)}")

        if not callable(inference_fn):
            raise TypeError("inference_fn must be a callable function or object")

        folds: List[LOSOFold] = dataset.get_loso_folds()
        if len(folds) < 2:
            raise ValueError("LOSO evaluation requires at least 2 folds / signers")

        all_true_labels: List[int] = []
        all_predicted_labels: List[int] = []
        fold_results: List[LOSOFoldResult] = []

        for fold in folds:
            train_ids = set(s.sample_id for s in fold.train_samples)
            test_ids = set(s.sample_id for s in fold.test_samples)

            # Defensive train/test isolation checks
            train_signers = set(s.signer_id for s in fold.train_samples)
            if fold.held_out_signer in train_signers:
                raise ValueError(
                    f"Defensive check failed: held_out_signer '{fold.held_out_signer}' found in train set"
                )
            if len(train_ids.intersection(test_ids)) > 0:
                raise ValueError(
                    "Defensive check failed: sample ID overlap between train and test sets"
                )

            # Invoke inference callback
            preds_raw = inference_fn(fold.train_samples, fold.test_samples)

            if preds_raw is None:
                raise ValueError("Inference callback returned None instead of predictions")

            try:
                preds_arr = np.asarray(preds_raw)
            except (ValueError, TypeError) as err:
                raise ValueError(f"Could not parse inference predictions as NumPy array: {err}")

            if not np.issubdtype(preds_arr.dtype, np.integer) and not np.issubdtype(preds_arr.dtype, np.floating):
                raise ValueError(f"Predictions must be numeric, got dtype {preds_arr.dtype}")

            if preds_arr.ndim != 1:
                raise ValueError(f"Predictions must be a 1D sequence, got shape {preds_arr.shape}")

            if len(preds_arr) != len(fold.test_samples):
                raise ValueError(
                    f"Prediction count ({len(preds_arr)}) does not match test sample count ({len(fold.test_samples)})"
                )

            if np.any(np.isnan(preds_arr)) or np.any(np.isinf(preds_arr)):
                raise ValueError("Predictions contain NaN or Infinity values")

            if len(preds_arr) > 0 and not np.all(np.equal(preds_arr, np.round(preds_arr))):
                raise ValueError("Predictions contain non-integer numeric labels")

            if np.any(preds_arr < 0):
                raise ValueError("Predictions contain negative label values")

            preds_list = [int(x) for x in preds_arr]
            true_list = [int(s.label) for s in fold.test_samples]

            all_true_labels.extend(true_list)
            all_predicted_labels.extend(preds_list)

            # Compute per-fold metrics
            fold_metrics = self.metrics_evaluator.evaluate(
                y_true=true_list,
                y_pred=preds_list,
                num_classes=self.num_classes,
                class_names=self.class_names,
            )

            fold_result = LOSOFoldResult(
                held_out_signer=fold.held_out_signer,
                train_sample_count=len(fold.train_samples),
                test_sample_count=len(fold.test_samples),
                train_sample_ids=[s.sample_id for s in fold.train_samples],
                test_sample_ids=[s.sample_id for s in fold.test_samples],
                true_labels=true_list,
                predicted_labels=preds_list,
                metrics=fold_metrics,
            )
            fold_results.append(fold_result)

        # Compute aggregate metrics across all held-out fold predictions
        aggregate_metrics = self.metrics_evaluator.evaluate(
            y_true=all_true_labels,
            y_pred=all_predicted_labels,
            num_classes=self.num_classes,
            class_names=self.class_names,
        )

        effective_num_classes = aggregate_metrics.num_classes
        effective_class_names = aggregate_metrics.class_names

        return LOSOEvaluationResult(
            num_signers=len(dataset.unique_signers),
            num_samples=len(dataset),
            num_classes=effective_num_classes,
            class_names=effective_class_names,
            folds=fold_results,
            aggregate_metrics=aggregate_metrics,
        )

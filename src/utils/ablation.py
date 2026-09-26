"""Ablation Study Framework for SIGNASSIST Phase 16.

Provides AblationConfig, AblationResult, AblationStudyResult, and AblationRunner
for executing model-agnostic ablation experiments with standard or LOSO evaluation pipelines.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from src.utils.evaluation_dataset import EvaluationDataset, EvaluationSample
from src.utils.loso_evaluator import (
    LOSOEvaluator,
    LOSOEvaluationResult,
    validate_and_parse_predictions,
)
from src.utils.metrics import EvaluationResult, MetricsEvaluator


@dataclass
class AblationConfig:
    """Configuration container for a single ablation experiment."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("AblationConfig name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("AblationConfig description must be a non-empty string")
        if not isinstance(self.parameters, dict):
            raise TypeError("AblationConfig parameters must be a dictionary")
        self.name = self.name.strip()
        self.description = self.description.strip()


def _sanitize_json_value(val: Any) -> Any:
    """Recursively converts NumPy scalars, NumPy arrays, tuples, lists, and dicts to JSON-serializable types.

    Raises:
        TypeError: If val is an unsupported object type (e.g. callable, tensor, arbitrary class instance).
    """
    if val is None or isinstance(val, (int, float, str, bool)):
        return val

    # NumPy scalar types (e.g. np.int64, np.float32, np.bool_)
    if isinstance(val, np.generic):
        return val.item()

    # NumPy arrays
    if isinstance(val, np.ndarray):
        return val.tolist()

    # Dictionaries
    if isinstance(val, dict):
        return {str(k): _sanitize_json_value(v) for k, v in val.items()}

    # Lists, tuples, sets
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_json_value(item) for item in val]

    # Unsupported object types
    raise TypeError(
        f"Object of type {type(val).__name__} is not JSON serializable in Ablation framework"
    )


@dataclass
class AblationResult:
    """Result container for an individual ablation experiment."""

    config: AblationConfig
    evaluation_type: str
    metrics: EvaluationResult
    sample_count: int
    signer_count: int
    fold_count: Optional[int] = None
    loso_result: Optional[LOSOEvaluationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary representation of the ablation result."""
        res: Dict[str, Any] = {
            "name": str(self.config.name),
            "description": str(self.config.description),
            "parameters": _sanitize_json_value(self.config.parameters),
            "evaluation_type": str(self.evaluation_type),
            "sample_count": int(self.sample_count),
            "signer_count": int(self.signer_count),
            "metrics": self.metrics.to_dict(),
        }
        if self.fold_count is not None:
            res["fold_count"] = int(self.fold_count)
        if self.loso_result is not None:
            res["loso_details"] = self.loso_result.to_dict()
        if self.metadata:
            res["metadata"] = _sanitize_json_value(self.metadata)
        return res


@dataclass
class AblationStudyResult:
    """Top-level result container for a full ablation study across multiple configurations."""

    evaluation_type: str
    num_experiments: int
    results: List[AblationResult]

    def to_dict(self) -> Dict[str, Any]:
        """Returns JSON-serializable dictionary representation of full ablation study."""
        return {
            "evaluation_type": str(self.evaluation_type),
            "num_experiments": int(self.num_experiments),
            "experiments": [r.to_dict() for r in self.results],
        }

    def to_json(
        self, output_path: Optional[Union[str, Path]] = None, indent: int = 2
    ) -> str:
        """Serializes ablation study result to JSON string and optionally writes to file."""
        json_str = json.dumps(self.to_dict(), indent=indent)
        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json_str, encoding="utf-8")
        return json_str


class AblationRunner:
    """Model-agnostic ablation study orchestrator."""

    def __init__(self) -> None:
        self._experiments: List[Dict[str, Any]] = []
        self._experiment_names: set = set()

    def add_experiment(
        self, config: AblationConfig, inference_fn: Callable[..., Any]
    ) -> None:
        """Registers a new ablation experiment configuration and inference function.

        Args:
            config: AblationConfig defining experiment metadata and parameters.
            inference_fn: Callable returning predicted labels for dataset samples.

        Raises:
            TypeError: If config is not AblationConfig or inference_fn is not callable.
            ValueError: If an experiment with the same name is already registered.
        """
        if not isinstance(config, AblationConfig):
            raise TypeError(f"Expected AblationConfig, got {type(config)}")

        if not callable(inference_fn):
            raise TypeError("inference_fn must be a callable function or object")

        if config.name in self._experiment_names:
            raise ValueError(f"Duplicate experiment name '{config.name}' already registered")

        self._experiments.append({"config": config, "inference_fn": inference_fn})
        self._experiment_names.add(config.name)

    @property
    def experiments(self) -> List[AblationConfig]:
        """Exposes the list of registered AblationConfig instances in registration order."""
        return [entry["config"] for entry in self._experiments]

    def run(
        self,
        dataset: EvaluationDataset,
        evaluation_type: str = "standard",
        zero_division: float = 0.0,
        num_classes: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> AblationStudyResult:
        """Runs registered ablation experiments against the provided dataset.

        Args:
            dataset: EvaluationDataset instance containing evaluation samples.
            evaluation_type: "standard" or "loso".
            zero_division: Precision/Recall/F1 fallback value when division by zero occurs.
            num_classes: Optional explicit number of target classes.
            class_names: Optional list of class names.

        Returns:
            AblationStudyResult containing AblationResult for each registered experiment.

        Raises:
            TypeError: If dataset is not EvaluationDataset.
            ValueError: If evaluation_type is invalid or no experiments are registered.
        """
        if not isinstance(dataset, EvaluationDataset):
            raise TypeError(f"Expected EvaluationDataset, got {type(dataset)}")

        if evaluation_type not in ("standard", "loso"):
            raise ValueError(
                f"Invalid evaluation_type '{evaluation_type}'. Supported types: 'standard', 'loso'"
            )

        if not self._experiments:
            raise ValueError("Cannot run ablation study with no registered experiments")

        results: List[AblationResult] = []

        for entry in self._experiments:
            config: AblationConfig = entry["config"]
            inference_fn: Callable = entry["inference_fn"]

            if evaluation_type == "loso":
                loso_evaluator = LOSOEvaluator(
                    zero_division=zero_division,
                    num_classes=num_classes,
                    class_names=class_names,
                )
                loso_res: LOSOEvaluationResult = loso_evaluator.evaluate(dataset, inference_fn)

                ablation_res = AblationResult(
                    config=config,
                    evaluation_type="loso",
                    metrics=loso_res.aggregate_metrics,
                    sample_count=loso_res.num_samples,
                    signer_count=loso_res.num_signers,
                    fold_count=len(loso_res.folds),
                    loso_result=loso_res,
                )
            else:
                # Standard evaluation: strictly inference_fn(samples) -> predictions
                preds_raw = inference_fn(dataset.samples)

                preds_list = validate_and_parse_predictions(preds_raw, len(dataset.samples))
                true_list = [int(s.label) for s in dataset.samples]

                metrics_evaluator = MetricsEvaluator(zero_division=zero_division)
                metrics: EvaluationResult = metrics_evaluator.evaluate(
                    y_true=true_list,
                    y_pred=preds_list,
                    num_classes=num_classes,
                    class_names=class_names,
                )

                ablation_res = AblationResult(
                    config=config,
                    evaluation_type="standard",
                    metrics=metrics,
                    sample_count=len(dataset.samples),
                    signer_count=len(dataset.unique_signers),
                    fold_count=None,
                    loso_result=None,
                )

            results.append(ablation_res)

        return AblationStudyResult(
            evaluation_type=evaluation_type,
            num_experiments=len(results),
            results=results,
        )

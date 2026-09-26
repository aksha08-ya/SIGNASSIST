"""Evaluation runner script for SIGNASSIST Phase 16 Leave-One-Signer-Out (LOSO) Cross-Validation.

Provides CLI interface for running Leave-One-Signer-Out evaluation on research datasets.
Demonstration mode uses synthetic data explicitly labeled as [DEMONSTRATION/TEST ONLY].
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.evaluation_dataset import EvaluationDataset, EvaluationSample
from src.utils.loso_evaluator import LOSOEvaluator, LOSOEvaluationResult


def mock_deterministic_classifier(
    train_samples: List[EvaluationSample], test_samples: List[EvaluationSample]
) -> List[int]:
    """Mock deterministic inference function for pipeline validation only.

    NOTE: This is a synthetic demonstration helper. It does NOT perform actual model inference.
    """
    # Deterministic mock rule: return true label for demo
    return [s.label for s in test_samples]


def create_demo_dataset() -> EvaluationDataset:
    """Generates a small synthetic EvaluationDataset with 3 signers for demonstration."""
    samples = []
    signers = ["Signer_A", "Signer_B", "Signer_C"]
    num_samples_per_signer = 4
    num_classes = 3

    idx = 0
    for signer in signers:
        for i in range(num_samples_per_signer):
            idx += 1
            label = i % num_classes
            sample = EvaluationSample(
                sample_id=f"demo_sample_{idx:03d}",
                label=label,
                signer_id=signer,
                features=np.zeros(225, dtype=np.float32),
                metadata={"synthetic": True},
            )
            samples.append(sample)

    return EvaluationDataset(samples)


def run_loso_evaluation(
    dataset: EvaluationDataset,
    inference_fn: Any = mock_deterministic_classifier,
    output_json_path: Optional[Union[str, Path]] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> LOSOEvaluationResult:
    """Runs LOSOEvaluator on the provided dataset and optional inference function."""
    evaluator = LOSOEvaluator(
        zero_division=0.0, num_classes=num_classes, class_names=class_names
    )
    result = evaluator.evaluate(dataset, inference_fn)

    if output_json_path is not None:
        result.to_json(output_json_path)

    return result


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SIGNASSIST Phase 16 Leave-One-Signer-Out (LOSO) Cross-Validation Runner"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination file path for saving JSON evaluation results",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run evaluation in synthetic demonstration mode",
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.demo:
        print("=== SIGNASSIST Phase 16 LOSO Evaluation Framework ===")
        print("Notice: Real LOSO evaluation requires an explicit EvaluationDataset and injected model inference function.")
        print("To run the synthetic pipeline demonstration, execute: python scripts/evaluate_loso.py --demo")
        return 0

    print("=== SIGNASSIST Phase 16 LOSO Evaluation Framework ===")
    print("[DEMONSTRATION/TEST ONLY - Synthetic Dataset Execution]\n")

    dataset = create_demo_dataset()
    print(f"Loaded synthetic dataset: {len(dataset)} samples across {len(dataset.unique_signers)} signers.")
    print(f"Unique signers: {dataset.unique_signers}")

    result = run_loso_evaluation(
        dataset=dataset,
        inference_fn=mock_deterministic_classifier,
        output_json_path=parsed_args.output,
        num_classes=3,
        class_names=["CLASS_0", "CLASS_1", "CLASS_2"],
    )

    print(f"\n[LOSO Evaluation Complete]")
    print(f"  Folds Evaluated : {len(result.folds)}")
    print(f"  Total Samples   : {result.num_samples}")
    print(f"  Macro F1        : {result.aggregate_metrics.macro_f1:.4f}")
    print(f"  Accuracy        : {result.aggregate_metrics.accuracy:.4f}")

    if parsed_args.output:
        print(f"\nResults successfully written to: {parsed_args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

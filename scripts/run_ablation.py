"""Ablation Study CLI Runner script for SIGNASSIST Phase 16.

Provides CLI interface for running model-agnostic ablation studies.
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

from src.utils.ablation import AblationConfig, AblationRunner, AblationStudyResult
from src.utils.evaluation_dataset import EvaluationDataset, EvaluationSample


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


def mock_full_model_inference(
    train_or_samples: Any, test_samples: Optional[List[EvaluationSample]] = None
) -> List[int]:
    """Synthetic inference function for 'full_model' configuration (100% accuracy mock)."""
    target = test_samples if test_samples is not None else train_or_samples
    return [s.label for s in target]


def mock_no_face_inference(
    train_or_samples: Any, test_samples: Optional[List[EvaluationSample]] = None
) -> List[int]:
    """Synthetic inference function for 'no_face' configuration (ablated accuracy mock)."""
    target = test_samples if test_samples is not None else train_or_samples
    # Slightly degraded prediction: misclassify class 2 as class 0
    return [s.label if s.label != 2 else 0 for s in target]


def mock_static_only_inference(
    train_or_samples: Any, test_samples: Optional[List[EvaluationSample]] = None
) -> List[int]:
    """Synthetic inference function for 'static_only' configuration (baseline mock)."""
    target = test_samples if test_samples is not None else train_or_samples
    # Constant baseline prediction (class 0)
    return [0 for _ in target]


def main(args: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SIGNASSIST Phase 16 Ablation Study Runner"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination file path for saving JSON ablation study results",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run ablation study in synthetic demonstration mode",
    )
    parser.add_argument(
        "--loso",
        action="store_true",
        default=False,
        help="Use Leave-One-Signer-Out (LOSO) evaluation mode in demo",
    )

    parsed_args = parser.parse_args(args)

    if not parsed_args.demo:
        print("=== SIGNASSIST Phase 16 Ablation Study Framework ===")
        print(
            "Notice: Real ablation execution requires an explicit EvaluationDataset and model-specific inference callbacks."
        )
        print(
            "To run the synthetic pipeline demonstration, execute: python scripts/run_ablation.py --demo"
        )
        return 0

    print("=== SIGNASSIST Phase 16 Ablation Study Framework ===")
    print("[DEMONSTRATION/TEST ONLY - Synthetic Ablation Execution]\n")

    dataset = create_demo_dataset()
    print(
        f"Loaded synthetic dataset: {len(dataset)} samples across {len(dataset.unique_signers)} signers."
    )

    eval_type = "loso" if parsed_args.loso else "loso"  # Default demo mode uses LOSO

    runner = AblationRunner()

    config_full = AblationConfig(
        name="full_model",
        description="Full SIGNASSIST Multimodal Model (Pose + Hands + Face)",
        parameters={"use_face": True, "use_pose": True, "use_hands": True},
    )
    runner.add_experiment(config_full, mock_full_model_inference)

    config_no_face = AblationConfig(
        name="no_face",
        description="Ablation: Exclude Facial Keypoints (Pose + Hands)",
        parameters={"use_face": False, "use_pose": True, "use_hands": True},
    )
    runner.add_experiment(config_no_face, mock_no_face_inference)

    config_static = AblationConfig(
        name="static_only",
        description="Ablation: Static Features Only (Hands)",
        parameters={"use_face": False, "use_pose": False, "use_hands": True},
    )
    runner.add_experiment(config_static, mock_static_only_inference)

    print(
        f"Running ablation study across {len(runner.experiments)} configurations ({eval_type.upper()} mode)...\n"
    )

    result: AblationStudyResult = runner.run(
        dataset=dataset,
        evaluation_type=eval_type,
        num_classes=3,
        class_names=["CLASS_0", "CLASS_1", "CLASS_2"],
    )

    print("Ablation Study Results Summary:")
    print("-" * 65)
    for exp_res in result.results:
        print(f"  Experiment  : {exp_res.config.name}")
        print(f"  Description : {exp_res.config.description}")
        print(f"  Accuracy    : {exp_res.metrics.accuracy:.4f}")
        print(f"  Macro F1    : {exp_res.metrics.macro_f1:.4f}")
        print("-" * 65)

    if parsed_args.output:
        result.to_json(parsed_args.output)
        print(f"\nAblation study results successfully written to: {parsed_args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

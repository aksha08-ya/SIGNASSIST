"""Smoke test script for Confidence Estimation & Rejection (SIGNASSIST Phase 8).

Instantiates ConfidenceEstimator, evaluates high-confidence and low-confidence synthetic logits,
prints top-3 predictions and acceptance/rejection status, and asserts correct numerical behavior.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.models.confidence import ConfidenceEstimator


def main() -> None:
    print("=== SIGNASSIST Phase 8 Confidence Estimation & Rejection Smoke Test ===")

    # 1. Instantiate ConfidenceEstimator
    estimator = ConfidenceEstimator(confidence_threshold=0.70, top_k=3)
    estimator.eval()

    total_params = sum(p.numel() for p in estimator.parameters())
    print("ConfidenceEstimator successfully constructed.")
    print(f"Total Model Parameters: {total_params} (Pure post-processing evaluation engine)")
    print(f"Confidence Threshold: {estimator.confidence_threshold}")
    print(f"Default Top-K: {estimator.top_k}")

    # 2. Create synthetic test logits: Item 0 = High Confidence (Peak at Class 12), Item 1 = Low Confidence (Flat)
    num_classes = 250
    logits = torch.zeros(2, num_classes)
    logits[0, 12] = 10.0  # Dominant logit -> High confidence
    # Logits[1] remains zero -> Uniform distribution (0.004 probability) -> Low confidence

    print(f"Input Logits Shape: {list(logits.shape)}")

    # 3. Forward execution
    with torch.no_grad():
        out = estimator(logits)

    print(f"Predicted Class Indices: {out.predicted_class.tolist()}")
    print(f"Top-1 Confidences: {[round(c, 4) for c in out.confidence.tolist()]}")
    print(f"Top-3 Class Indices:\n{out.top_k_indices.numpy()}")
    print(f"Top-3 Probabilities:\n{np.round(out.top_k_probabilities.numpy(), 4)}")
    print(f"Is Accepted Status: {out.is_accepted.tolist()}")
    print(f"Is Rejected Status: {out.is_rejected.tolist()}")

    # 4. Assertions
    # Item 0 (High Confidence)
    assert out.predicted_class[0].item() == 12
    assert out.confidence[0].item() > 0.90
    assert out.is_accepted[0].item() is True
    assert out.is_rejected[0].item() is False

    # Item 1 (Low Confidence / Uniform)
    assert out.confidence[1].item() < 0.05
    assert out.is_accepted[1].item() is False
    assert out.is_rejected[1].item() is True

    # Shapes
    assert list(out.top_k_indices.shape) == [2, 3]
    assert list(out.top_k_probabilities.shape) == [2, 3]
    assert torch.isfinite(out.probabilities).all()

    prob_sums = out.probabilities.sum(dim=-1).numpy()
    assert np.allclose(prob_sums, 1.0, atol=1e-6)

    print("=== Confidence Estimation & Rejection Smoke Test PASSED ===")


if __name__ == "__main__":
    main()

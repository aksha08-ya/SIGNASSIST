"""Confidence Estimation and Rejection Module for SIGNASSIST Phase 8.

Processes classification logits [B, num_classes], computes numerically stable softmax
probabilities, extracts top-K predictions (default K=3), and evaluates predictions against
a configurable confidence threshold to determine acceptance or low-confidence rejection.

Note on Calibration:
This module computes raw softmax confidence metrics (uncalibrated probabilities).
Softmax outputs should be interpreted as model confidence scores, not calibrated empirical probabilities.

TODO / Follow-Up:
Entropy/margin-based rejection and empirical unknown/rest-pose false-positive evaluation
require a trained dataset/checkpoint and will be addressed in a future calibration phase.

Outputs:
- predicted_class: [B] (int64 top-1 class index)
- confidence: [B] (float32 top-1 softmax probability)
- top_k_indices: [B, K] (int64 top-K class indices)
- top_k_probabilities: [B, K] (float32 top-K softmax probabilities)
- is_accepted: [B] (bool: True if confidence >= confidence_threshold else False)
- is_rejected: [B] (bool: True if confidence < confidence_threshold else False)
- probabilities: [B, num_classes] (full softmax probability distribution)

Strictly excludes LSTM, GRU, RNN, or any recurrent architectures.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None

from src.utils.config import InferenceConfig, load_config


@dataclass
class ConfidenceOutput:
    """Output container for ConfidenceEstimator forward execution."""

    predicted_class: Any  # torch.Tensor shaped [B] (int64)
    confidence: Any  # torch.Tensor shaped [B] (float32)
    top_k_indices: Any  # torch.Tensor shaped [B, K] (int64)
    top_k_probabilities: Any  # torch.Tensor shaped [B, K] (float32)
    is_accepted: Any  # torch.Tensor shaped [B] (bool)
    is_rejected: Any  # torch.Tensor shaped [B] (bool)
    probabilities: Optional[Any] = None  # torch.Tensor shaped [B, num_classes]
    predicted_gloss: Optional[Any] = None  # Resolved gloss string or List[str]
    top_k_glosses: Optional[Any] = None  # List of top-K gloss strings


if _TORCH_AVAILABLE:

    class ConfidenceEstimator(nn.Module):
        """Confidence Estimator & Rejection Engine for SIGNASSIST predictions.

        Computes raw softmax confidence metrics and performs top-1 confidence thresholding.
        Note: Softmax output is raw confidence, not calibrated empirical probability.

        TODO: Entropy/margin-based rejection and empirical unknown/rest-pose false-positive
        evaluation require a trained dataset/checkpoint and will be addressed later.
        """

        def __init__(
            self,
            confidence_threshold: float = 0.70,
            top_k: int = 3,
        ) -> None:
            super().__init__()
            if not (0.0 <= confidence_threshold <= 1.0):
                raise ValueError(
                    f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
                )
            if top_k < 1:
                raise ValueError(f"top_k must be >= 1, got {top_k}")

            self.confidence_threshold = float(confidence_threshold)
            self.top_k = int(top_k)

        def forward(
            self,
            logits: torch.Tensor,
            confidence_threshold: Optional[float] = None,
            top_k: Optional[int] = None,
        ) -> ConfidenceOutput:
            """Evaluates confidence and rejection status from classification logits.

            Args:
                logits: Input logits tensor of shape [B, num_classes].
                confidence_threshold: Optional override for confidence threshold.
                top_k: Optional override for top-K parameter.

            Returns:
                ConfidenceOutput container containing predicted_class, confidence,
                top_k_indices, top_k_probabilities, is_accepted, is_rejected, probabilities.
            """
            if logits.dim() != 2:
                raise ValueError(
                    f"Expected 2D logits tensor [B, num_classes], got shape {list(logits.shape)}"
                )

            B, num_classes = logits.shape
            threshold = (
                float(confidence_threshold)
                if confidence_threshold is not None
                else self.confidence_threshold
            )
            k = int(top_k) if top_k is not None else self.top_k

            if k > num_classes:
                raise ValueError(
                    f"top_k={k} cannot exceed number of classes num_classes={num_classes}"
                )

            # 1. Numerically stable softmax probability computation
            shift_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
            probabilities = torch.softmax(shift_logits, dim=-1)

            # 2. Extract Top-K probabilities and class indices
            top_k_values, top_k_indices = torch.topk(probabilities, k=k, dim=-1)

            # Top-1 prediction
            predicted_class = top_k_indices[:, 0]
            confidence = top_k_values[:, 0]

            # 3. Acceptance / Rejection thresholding
            is_accepted = confidence >= threshold
            is_rejected = ~is_accepted

            return ConfidenceOutput(
                predicted_class=predicted_class,
                confidence=confidence,
                top_k_indices=top_k_indices,
                top_k_probabilities=top_k_values,
                is_accepted=is_accepted,
                is_rejected=is_rejected,
                probabilities=probabilities,
            )

        @classmethod
        def from_config(
            cls, config: Optional[InferenceConfig] = None
        ) -> "ConfidenceEstimator":
            """Factory method to instantiate ConfidenceEstimator from InferenceConfig."""
            if config is None:
                app_cfg = load_config()
                config = app_cfg.model.inference

            return cls(
                confidence_threshold=config.confidence_threshold,
                top_k=getattr(config, "top_k", 3),
            )

    # Aliases for convenience
    ConfidenceRejectionEngine = ConfidenceEstimator

else:
    ConfidenceOutput = None
    ConfidenceEstimator = None
    ConfidenceRejectionEngine = None

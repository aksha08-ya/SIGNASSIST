"""Unit tests for Confidence Estimation & Rejection Subsystem (SIGNASSIST Phase 8).

Verifies:
- ConfidenceEstimator construction and configuration loading (from_config);
- Package exports via src.models;
- Probability normalization (softmax sums to 1.0);
- Top-K ordering (probabilities strictly descending);
- Top-K values match sorted full probabilities;
- High-confidence acceptance (confidence >= threshold -> is_accepted=True);
- Low-confidence rejection (confidence < threshold -> is_rejected=True);
- Batch sizes 1, 2, and 8;
- Configurable confidence_threshold override;
- Configurable top_k override (K=1, K=3, K=5);
- Compatibility with 36-class static logits and 262-class dynamic logits;
- Invalid input shape and invalid top-K error handling;
- Deterministic behavior in eval mode;
- Zero recurrent (LSTM/GRU/RNN) modules.
"""

import pytest
import torch
import torch.nn as nn

from src.models import (
    ConfidenceEstimator,
    ConfidenceOutput,
    ConfidenceRejectionEngine,
)


def test_package_exports():
    """Verifies package exports and class alias in src.models."""
    from src.models import ConfidenceEstimator as Estimator_pkg
    from src.models import ConfidenceRejectionEngine as Alias_pkg

    assert Estimator_pkg is ConfidenceEstimator
    assert Alias_pkg is ConfidenceEstimator
    assert issubclass(ConfidenceEstimator, nn.Module)


def test_estimator_construction():
    """Verifies construction and default hyperparameters."""
    estimator = ConfidenceEstimator()
    assert estimator.confidence_threshold == 0.70
    assert estimator.top_k == 3
    assert isinstance(estimator, nn.Module)


def test_construction_from_config():
    """Verifies factory construction via ConfidenceEstimator.from_config()."""
    estimator = ConfidenceEstimator.from_config()
    assert estimator.confidence_threshold == 0.70
    assert estimator.top_k == 3


def test_probability_normalization():
    """Verifies that softmax probabilities sum to 1.0 for all batch items."""
    estimator = ConfidenceEstimator()
    estimator.eval()

    B, num_classes = 4, 36
    logits = torch.randn(B, num_classes)

    with torch.no_grad():
        out = estimator(logits)

    assert isinstance(out, ConfidenceOutput)
    prob_sums = out.probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
    assert (out.probabilities >= 0.0).all() and (out.probabilities <= 1.0).all()


def test_top_k_ordering_and_values():
    """Verifies that top_k_probabilities are strictly in descending order and match full probabilities."""
    estimator = ConfidenceEstimator(top_k=3)
    estimator.eval()

    B, num_classes = 2, 262
    logits = torch.randn(B, num_classes)

    with torch.no_grad():
        out = estimator(logits)

    assert out.top_k_indices.shape == (B, 3)
    assert out.top_k_probabilities.shape == (B, 3)

    for b in range(B):
        probs_row = out.top_k_probabilities[b]
        # Check strictly non-increasing / descending order
        assert (probs_row[:-1] >= probs_row[1:]).all(), "Top-K probabilities not in descending order"

        # Check values match full probabilities at corresponding indices
        indices_row = out.top_k_indices[b]
        expected_top_values = out.probabilities[b, indices_row]
        assert torch.allclose(probs_row, expected_top_values, atol=1e-6)


def test_high_confidence_acceptance():
    """Verifies that a high peak in logits produces is_accepted=True and is_rejected=False."""
    estimator = ConfidenceEstimator(confidence_threshold=0.70)
    estimator.eval()

    # Create logits with a large peak at index 5
    logits = torch.zeros(1, 36)
    logits[0, 5] = 10.0  # Dominant logit -> probability ~0.99

    with torch.no_grad():
        out = estimator(logits)

    assert out.predicted_class[0].item() == 5
    assert out.confidence[0].item() > 0.90
    assert out.is_accepted[0].item() is True
    assert out.is_rejected[0].item() is False


def test_low_confidence_rejection():
    """Verifies that flat/uniform logits produce is_accepted=False and is_rejected=True."""
    estimator = ConfidenceEstimator(confidence_threshold=0.70)
    estimator.eval()

    # Create uniform/flat logits -> probability for each of 36 classes = 1/36 ~ 0.027
    logits = torch.zeros(1, 36)

    with torch.no_grad():
        out = estimator(logits)

    assert out.confidence[0].item() < 0.10
    assert out.is_accepted[0].item() is False
    assert out.is_rejected[0].item() is True


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batch_sizes(batch_size: int):
    """Verifies forward execution across batch sizes 1, 2, and 8."""
    estimator = ConfidenceEstimator()
    estimator.eval()

    logits = torch.randn(batch_size, 262)
    with torch.no_grad():
        out = estimator(logits)

    assert out.predicted_class.shape == (batch_size,)
    assert out.confidence.shape == (batch_size,)
    assert out.top_k_indices.shape == (batch_size, 3)
    assert out.top_k_probabilities.shape == (batch_size, 3)
    assert out.is_accepted.shape == (batch_size,)
    assert out.is_rejected.shape == (batch_size,)


def test_configurable_threshold_override():
    """Verifies that overriding confidence_threshold at runtime dynamically alters acceptance."""
    estimator = ConfidenceEstimator(confidence_threshold=0.70)
    estimator.eval()

    # Logits yielding probability ~ 0.92
    logits = torch.zeros(1, 36)
    logits[0, 0] = 6.0  # Peak logit yields prob ~ 0.92

    with torch.no_grad():
        # With threshold 0.70 -> Accepted (0.92 >= 0.70)
        out_default = estimator(logits)
        # With threshold 0.95 -> Rejected (0.92 < 0.95)
        out_strict = estimator(logits, confidence_threshold=0.95)

    assert out_default.is_accepted[0].item() is True
    assert out_strict.is_accepted[0].item() is False


@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_configurable_top_k(k: int):
    """Verifies that top-K shapes dynamically scale with configurable K values."""
    estimator = ConfidenceEstimator()
    estimator.eval()

    logits = torch.randn(2, 36)
    with torch.no_grad():
        out = estimator(logits, top_k=k)

    assert out.top_k_indices.shape == (2, k)
    assert out.top_k_probabilities.shape == (2, k)


def test_compatibility_static_and_dynamic_logits():
    """Verifies compatibility with 36-class static GCN logits and 262-class dynamic classifier logits."""
    estimator = ConfidenceEstimator()
    estimator.eval()

    static_logits = torch.randn(2, 36)
    dynamic_logits = torch.randn(2, 262)

    with torch.no_grad():
        out_static = estimator(static_logits)
        out_dynamic = estimator(dynamic_logits)

    assert out_static.probabilities.shape == (2, 36)
    assert out_dynamic.probabilities.shape == (2, 262)
    assert torch.isfinite(out_static.confidence).all()
    assert torch.isfinite(out_dynamic.confidence).all()


def test_invalid_input_shapes_and_parameters():
    """Verifies ValueError handling for invalid input shapes, thresholds, or top-K values."""
    # Invalid constructor threshold
    with pytest.raises(ValueError, match="confidence_threshold must be in"):
        ConfidenceEstimator(confidence_threshold=1.5)

    # Invalid constructor top_k
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        ConfidenceEstimator(top_k=0)

    estimator = ConfidenceEstimator(top_k=3)

    # 1D logits input
    with pytest.raises(ValueError, match="Expected 2D logits tensor"):
        estimator(torch.randn(36))

    # 3D logits input
    with pytest.raises(ValueError, match="Expected 2D logits tensor"):
        estimator(torch.randn(2, 5, 36))

    # top_k > num_classes (K=50 for 36 classes)
    with pytest.raises(ValueError, match="cannot exceed number of classes"):
        estimator(torch.randn(2, 36), top_k=50)


def test_deterministic_eval_behavior():
    """Verifies deterministic forward execution in eval mode."""
    estimator = ConfidenceEstimator()
    estimator.eval()

    logits = torch.randn(2, 262)
    with torch.no_grad():
        out1 = estimator(logits)
        out2 = estimator(logits)

    assert torch.allclose(out1.probabilities, out2.probabilities, atol=1e-7)
    assert torch.equal(out1.predicted_class, out2.predicted_class)
    assert torch.equal(out1.is_accepted, out2.is_accepted)


def test_no_lstm_gru_rnn_modules():
    """Verifies zero recurrent (LSTM/GRU/RNN) modules in the estimator."""
    estimator = ConfidenceEstimator()
    forbidden_types = (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    for name, module in estimator.named_modules():
        assert not isinstance(module, forbidden_types), f"Module '{name}' is forbidden recurrent type {type(module)}"

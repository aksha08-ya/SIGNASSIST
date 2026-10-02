"""Focused unit tests for temperature scaling calibration study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_calibration_study import compute_ece, compute_nll, compute_softmax


def test_temperature_scaling_prediction_invariance():
    """Verify temperature scaling T > 0 preserves top-1 class predictions identically."""
    np.random.seed(42)
    logits = np.random.randn(50, 42).astype(np.float32)

    base_preds = np.argmax(logits, axis=1)

    for T in [0.2, 0.5, 0.68, 1.0, 1.5, 2.0, 3.0]:
        probs = compute_softmax(logits, T)
        t_preds = np.argmax(probs, axis=1)
        assert np.array_equal(base_preds, t_preds), f"Prediction invariance failed for T={T}"


def test_ece_computation_bounds_and_validity():
    """Verify ECE computation returns a non-negative float bounded in [0.0, 1.0]."""
    np.random.seed(42)
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
    predictions = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    targets = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int64)

    ece = compute_ece(confidences, predictions, targets, num_bins=5)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_temperature_scaling_confidence_monotonicity():
    """Verify lowering temperature T < 1.0 increases peak softmax probability for non-uniform logits."""
    logits = np.array([[3.0, 1.0, 0.5, 0.1]], dtype=np.float32)

    p_t1 = compute_softmax(logits, T=1.0)
    p_t068 = compute_softmax(logits, T=0.68)
    p_t05 = compute_softmax(logits, T=0.5)

    conf_t1 = np.max(p_t1)
    conf_t068 = np.max(p_t068)
    conf_t05 = np.max(p_t05)

    assert conf_t05 > conf_t068 > conf_t1, "Lowering T must sharpen top-1 confidence score"


def test_nll_computation_validity():
    """Verify NLL computation returns positive finite float."""
    logits = np.array([[2.0, 0.5], [0.1, 1.5]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)

    nll = compute_nll(logits, targets, T=1.0)
    assert isinstance(nll, float)
    assert nll > 0.0
    assert np.isfinite(nll)

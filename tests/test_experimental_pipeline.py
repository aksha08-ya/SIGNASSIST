"""Unit tests for dual-backend recognition pipeline (production vs experimental_42)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from src.pipeline.recognition import RecognitionPipeline
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier


def test_production_backend_initialization():
    """Verify production backend initializes with 262 classes and production defaults."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="production")
    assert pipeline.backend == "production"
    assert pipeline.class_mapping_path == Path("data/processed/include/class_mapping.json")
    assert pipeline.temperature == 1.0
    assert pipeline.id_to_class is not None
    assert len(pipeline.id_to_class) == 262
    assert pipeline.is_available()


def test_experimental_backend_initialization():
    """Verify experimental_42 backend initializes with 42 classes and calibration."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    assert pipeline.backend == "experimental_42"
    assert pipeline.class_mapping_path == Path("data/processed/include/vocabulary_initial.json")
    assert pipeline.id_to_class is not None
    assert len(pipeline.id_to_class) == 42
    assert isinstance(pipeline.dynamic_classifier, ControlledSTGCNClassifier)
    assert pipeline.is_available()


def test_experimental_output_dimension():
    """Verify experimental model output dimension is exactly 42."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    dummy_input = torch.randn(1, 5, 30, 115)
    with torch.no_grad():
        out = pipeline.dynamic_classifier(dummy_input)
    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape == (1, 42)


def test_experimental_label_mapping():
    """Verify all 42 experimental class IDs map to non-empty string labels."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    for idx in range(42):
        label = pipeline.get_dynamic_label(idx)
        assert isinstance(label, str)
        assert len(label) > 0


def test_experimental_temperature_loading():
    """Verify experimental temperature calibration is correctly loaded from calibration.json."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    calib_path = Path("models/checkpoints/experimental/calibration.json")
    if calib_path.is_file():
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        expected_t = float(data.get("temperature", 1.0))
        assert pipeline.temperature == pytest.approx(expected_t, rel=1e-3)


def test_both_backends_coexist():
    """Verify production and experimental backends can exist simultaneously without interference."""
    prod_pipeline = RecognitionPipeline(mode="dynamic_only", backend="production")
    exp_pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")

    assert prod_pipeline.backend == "production"
    assert exp_pipeline.backend == "experimental_42"
    assert len(prod_pipeline.id_to_class) == 262
    assert len(exp_pipeline.id_to_class) == 42
    assert prod_pipeline.is_available()
    assert exp_pipeline.is_available()


def test_invalid_backend_name_fails():
    """Verify invalid backend names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid backend"):
        RecognitionPipeline(mode="dynamic_only", backend="invalid_backend")


def test_invalid_experimental_checkpoint_fails_loudly():
    """Verify invalid/missing experimental checkpoint fails loudly in dynamic_only mode."""
    with pytest.raises(FileNotFoundError):
        RecognitionPipeline(
            mode="dynamic_only",
            backend="experimental_42",
            dynamic_checkpoint_path="non_existent_ckpt.pt",
        )


def test_dummy_inference_experimental():
    """Verify dummy inference through experimental 42-class model generates valid logits."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    dummy_seq = torch.randn(1, 5, 30, 115)
    with torch.no_grad():
        out = pipeline.dynamic_classifier(dummy_seq)
        logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape == (1, 42)
    assert not torch.isnan(logits).any()

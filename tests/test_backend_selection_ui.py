"""Unit tests verifying UI backend selection (production vs experimental_42) invocation across services."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.ui.services import (
    get_default_recognition_pipeline,
    process_video_file,
    _create_default_video_processor,
)
from src.ui.webcam import WebcamRecognitionSession
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier
from src.models.dynamic_classifier import DynamicSignClassifier


def test_service_pipeline_factory_production_backend():
    """Verify service factory returns 262-class production backend when explicitly requested."""
    pipeline = get_default_recognition_pipeline(backend="production")
    assert pipeline.backend == "production"
    assert pipeline.class_mapping_path == Path("data/processed/include/class_mapping.json")
    assert len(pipeline.id_to_class) == 262
    assert isinstance(pipeline.dynamic_classifier, DynamicSignClassifier)


def test_service_pipeline_factory_experimental_backend():
    """Verify service factory returns 42-class experimental backend when requested."""
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    assert pipeline.backend == "experimental_42"
    assert pipeline.class_mapping_path == Path("data/processed/include/vocabulary_initial.json")
    assert len(pipeline.id_to_class) == 42
    assert isinstance(pipeline.dynamic_classifier, ControlledSTGCNClassifier)


def test_video_processor_factory_backend_routing():
    """Verify VideoProcessor factory correctly passes backend parameter to underlying pipeline."""
    prod_processor = _create_default_video_processor(backend="production")
    assert prod_processor.recognition_pipeline.backend == "production"
    assert len(prod_processor.recognition_pipeline.id_to_class) == 262

    exp_processor = _create_default_video_processor(backend="experimental_42")
    assert exp_processor.recognition_pipeline.backend == "experimental_42"
    assert len(exp_processor.recognition_pipeline.id_to_class) == 42


def test_webcam_session_backend_switching():
    """Verify WebcamRecognitionSession maintains explicit backend routing."""
    prod_pipeline = get_default_recognition_pipeline(backend="production")
    prod_session = WebcamRecognitionSession(recognition_pipeline=prod_pipeline)
    assert prod_session.pipeline.backend == "production"
    assert len(prod_session.pipeline.id_to_class) == 262

    exp_pipeline = get_default_recognition_pipeline(backend="experimental_42")
    exp_session = WebcamRecognitionSession(recognition_pipeline=exp_pipeline)
    assert exp_session.pipeline.backend == "experimental_42"
    assert len(exp_session.pipeline.id_to_class) == 42


# --- Regression Tests for Demo Backend Configuration ---

def test_experimental_backend_is_default_in_services():
    """Regression test: verify get_default_recognition_pipeline defaults to experimental_42."""
    pipeline = get_default_recognition_pipeline()
    assert pipeline.backend == "experimental_42"
    assert len(pipeline.id_to_class) == 42


def test_no_silent_production_fallback():
    """Regression test: verify experimental_42 uses ControlledSTGCNClassifier and 42-class mapping without silent fallback."""
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    assert pipeline.backend == "experimental_42"
    assert isinstance(pipeline.dynamic_classifier, ControlledSTGCNClassifier)
    assert pipeline.class_mapping_path == Path("data/processed/include/vocabulary_initial.json")


def test_experimental_configuration_uses_t_068():
    """Regression test: verify experimental_42 configuration loads calibrated temperature T=0.68."""
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    assert pipeline.temperature == pytest.approx(0.68, abs=1e-3)


def test_experimental_configuration_boundary_trimming_disabled():
    """Regression test: verify experimental_42 configuration disables boundary trimming by default."""
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    assert pipeline.use_boundary_trimming is False


def test_confidence_threshold_remains_070():
    """Regression test: verify confidence threshold remains exactly 0.70 across backends."""
    exp_pipeline = get_default_recognition_pipeline(backend="experimental_42")
    prod_pipeline = get_default_recognition_pipeline(backend="production")

    assert exp_pipeline.confidence_estimator.confidence_threshold == pytest.approx(0.70)
    assert prod_pipeline.confidence_estimator.confidence_threshold == pytest.approx(0.70)


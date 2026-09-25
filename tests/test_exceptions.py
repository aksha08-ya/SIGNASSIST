"""
Unit tests for SIGNASSIST custom exception hierarchy.
"""

import pytest
from src.utils.exceptions import (
    SignAssistException,
    ConfigurationError,
    DataError,
    LandmarkExtractionError,
    ModelError,
    ModelInferenceError,
    VideoProcessingError,
    URLExtractionError,
    PipelineError,
)


def test_base_exception():
    err = SignAssistException("Base error message", details={"key": "val"})
    assert str(err) == "Base error message | Details: {'key': 'val'}"
    assert err.message == "Base error message"
    assert err.details == {"key": "val"}


def test_base_exception_no_details():
    err = SignAssistException("Base error message without details")
    assert str(err) == "Base error message without details"
    assert err.details == {}


def test_exception_inheritance():
    cfg_err = ConfigurationError("Invalid YAML config")
    assert isinstance(cfg_err, SignAssistException)

    model_inf_err = ModelInferenceError("CUDA OOM during inference")
    assert isinstance(model_inf_err, ModelError)
    assert isinstance(model_inf_err, SignAssistException)

    url_err = URLExtractionError("Failed to fetch stream")
    assert isinstance(url_err, VideoProcessingError)
    assert isinstance(url_err, SignAssistException)


def test_all_custom_exceptions():
    exceptions = [
        ConfigurationError("config issue"),
        DataError("data issue"),
        LandmarkExtractionError("landmark issue"),
        ModelError("model issue"),
        ModelInferenceError("inference issue"),
        VideoProcessingError("video issue"),
        URLExtractionError("url issue"),
        PipelineError("pipeline issue"),
    ]
    for exc in exceptions:
        assert isinstance(exc, SignAssistException)

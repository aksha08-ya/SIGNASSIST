"""
Custom exceptions taxonomy for the SIGNASSIST application.

This module defines the hierarchy of exceptions used across the project
to provide clear diagnostic context and structured error handling.
"""

from typing import Optional


class SignAssistException(Exception):
    """Base exception class for all SIGNASSIST errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(SignAssistException):
    """Raised when configuration files or environment parameters are invalid or missing."""
    pass


class DataError(SignAssistException):
    """Raised when dataset loading, validation, or pre-processing fails."""
    pass


class LandmarkExtractionError(SignAssistException):
    """Raised when landmark detection or extraction fails on video frame/sequence."""
    pass


class ModelError(SignAssistException):
    """Base exception for model architecture, loading, and training errors."""
    pass


class ModelInferenceError(ModelError):
    """Raised when model inference execution fails or produces invalid outputs."""
    pass


class VideoProcessingError(SignAssistException):
    """Raised when video loading, decoding, frame extraction, or web stream processing fails."""
    pass


class URLExtractionError(VideoProcessingError):
    """Raised when extracting video content from a URL (e.g. YouTube/web) fails."""
    pass


class PipelineError(SignAssistException):
    """Raised when end-to-end processing pipeline fails."""
    pass

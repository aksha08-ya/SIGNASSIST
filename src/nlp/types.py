"""Typed data structures and containers for SIGNASSIST Phase 10 sentence reconstruction."""

from dataclasses import dataclass, field
from typing import Any, List, Optional
import numpy as np

from src.models.confidence import ConfidenceOutput
from src.pipeline.boundary_detector import SignSegment


@dataclass
class SignToken:
    """Typed representation of a recognized sign token.

    Attributes:
        gloss: Resolved sign word / vocabulary gloss string.
        class_index: Integer class index from classification head.
        confidence: Top-1 softmax model confidence float in [0.0, 1.0].
        is_accepted: Authoritative boolean acceptance flag directly from Phase 8.
        top_k_glosses: List of top-K candidate gloss strings for UI/editing.
        top_k_probabilities: List of top-K candidate softmax probabilities.
        boundary_evidence: Bounded 0..1 segmentation motion evidence from Phase 9.
        start_frame: Starting temporal frame index (0-indexed).
        end_frame: Ending temporal frame index (inclusive).
    """

    gloss: str
    class_index: int
    confidence: float
    is_accepted: bool
    top_k_glosses: List[str] = field(default_factory=list)
    top_k_probabilities: List[float] = field(default_factory=list)
    boundary_evidence: Optional[float] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    def __post_init__(self) -> None:
        self.class_index = int(self.class_index)
        self.confidence = float(self.confidence)
        self.is_accepted = bool(self.is_accepted)
        if self.boundary_evidence is not None:
            self.boundary_evidence = float(self.boundary_evidence)
        if self.start_frame is not None:
            self.start_frame = int(self.start_frame)
        if self.end_frame is not None:
            self.end_frame = int(self.end_frame)


def _validate_single_sample_field(field_name: str, val: Any, is_top_k: bool = False) -> None:
    """Validates that a field from ConfidenceOutput represents a single sample."""
    if val is None:
        return
    if hasattr(val, "shape"):
        shape = tuple(val.shape)
        if len(shape) == 0:
            return  # 0D scalar
        if is_top_k:
            if len(shape) == 1:
                return  # 1D top-k array [K]
            if len(shape) >= 2 and shape[0] > 1:
                raise ValueError(
                    f"BoundedSignPrediction expects single-sample prediction, but {field_name} has batch size {shape[0]} > 1."
                )
        else:
            if len(shape) >= 1 and shape[0] > 1:
                raise ValueError(
                    f"BoundedSignPrediction expects single-sample prediction, but {field_name} has batch size {shape[0]} > 1."
                )
    elif isinstance(val, (list, tuple)):
        if is_top_k:
            if len(val) > 0 and isinstance(val[0], (list, tuple)) and len(val) > 1:
                raise ValueError(
                    f"BoundedSignPrediction expects single-sample prediction, but {field_name} has batch size {len(val)} > 1."
                )
        else:
            if len(val) > 1:
                raise ValueError(
                    f"BoundedSignPrediction expects single-sample prediction, but {field_name} has batch size {len(val)} > 1."
                )


@dataclass
class BoundedSignPrediction:
    """Explicit association binding exactly ONE Phase 9 SignSegment with ONE Phase 8 prediction."""

    segment: SignSegment
    prediction: ConfidenceOutput

    def validate(self) -> None:
        """Validates that segment and prediction represent valid single-sample outputs."""
        if not isinstance(self.segment, SignSegment):
            raise TypeError(
                f"Expected segment to be instance of SignSegment, got {type(self.segment)}"
            )
        if not isinstance(self.prediction, ConfidenceOutput):
            raise TypeError(
                f"Expected prediction to be instance of ConfidenceOutput, got {type(self.prediction)}"
            )

        pred = self.prediction
        _validate_single_sample_field("predicted_class", pred.predicted_class, is_top_k=False)
        _validate_single_sample_field("confidence", pred.confidence, is_top_k=False)
        _validate_single_sample_field("is_accepted", pred.is_accepted, is_top_k=False)
        _validate_single_sample_field("is_rejected", pred.is_rejected, is_top_k=False)
        _validate_single_sample_field("top_k_indices", pred.top_k_indices, is_top_k=True)
        _validate_single_sample_field("top_k_probabilities", pred.top_k_probabilities, is_top_k=True)


@dataclass
class SentenceTranscript:
    """Structured, editable transcript representation of a reconstructed sentence."""

    text: str
    raw_tokens: List[SignToken] = field(default_factory=list)
    filtered_tokens: List[SignToken] = field(default_factory=list)
    mean_confidence: float = 0.0
    is_edited: bool = False

    def update_text(self, new_text: str) -> None:
        """Updates text content and sets is_edited to True."""
        self.text = str(new_text)
        self.is_edited = True

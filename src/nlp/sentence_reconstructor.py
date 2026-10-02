"""SentenceReconstructor module for SIGNASSIST Phase 10."""

from typing import Dict, List, Optional, Union
import numpy as np

from src.models.confidence import ConfidenceOutput
from src.nlp.context_builder import ContextAwareSentenceBuilder
from src.nlp.duplicate_suppressor import DuplicateSuppressor
from src.nlp.types import (
    BoundedSignPrediction,
    SentenceTranscript,
    SignToken,
)
from src.pipeline.boundary_detector import SignSegment
from src.utils.config import ModelConfig, NLPConfig, load_config


def _to_scalar_int(val) -> int:
    if hasattr(val, "item"):
        return int(val.item())
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return int(val[0])
    return int(val)


def _to_scalar_float(val) -> float:
    if hasattr(val, "item"):
        return float(val.item())
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return float(val[0])
    return float(val)


def _to_scalar_bool(val) -> bool:
    if hasattr(val, "item"):
        return bool(val.item())
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        return bool(val[0])
    return bool(val)


def _to_1d_list(val) -> list:
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    if isinstance(val, np.ndarray):
        if val.ndim > 1:
            val = val.squeeze(0)
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]


class SentenceReconstructor:
    """Sentence reconstruction engine supporting batch and streaming modes.

    Connects Phase 8 confidence estimation and Phase 9 boundary segmentation outputs
    to produce clean, formatted English sentence transcripts.
    """

    def __init__(
        self,
        config: Optional[Union[NLPConfig, ModelConfig]] = None,
        vocab_map: Optional[Dict[int, str]] = None,
    ) -> None:
        if isinstance(config, ModelConfig):
            nlp_cfg = config.nlp
        elif isinstance(config, NLPConfig):
            nlp_cfg = config
        elif config is None:
            nlp_cfg = NLPConfig()
        else:
            raise TypeError(
                f"Expected config to be NLPConfig or ModelConfig, got {type(config)}"
            )

        self.config = nlp_cfg
        self.vocab_map = vocab_map or {}

        self.suppressor = DuplicateSuppressor(
            enable_suppression=nlp_cfg.enable_duplicate_suppression
        )
        self.builder = ContextAwareSentenceBuilder(
            enable_context_rules=nlp_cfg.enable_context_rules,
            auto_punctuation=nlp_cfg.auto_punctuation,
        )

        # Streaming buffer
        self._streaming_raw_tokens: List[SignToken] = []
        self._streaming_filtered_tokens: List[SignToken] = []

    def resolve_gloss(self, class_index: int) -> str:
        """Resolves integer class index to a string gloss."""
        return self.vocab_map.get(class_index, f"SIGN_{class_index}")

    def create_token(
        self,
        prediction: ConfidenceOutput,
        segment: Optional[SignSegment] = None,
    ) -> SignToken:
        """Constructs a SignToken from a Phase 8 ConfidenceOutput and optional Phase 9 SignSegment."""
        cls_idx = _to_scalar_int(prediction.predicted_class)
        conf = _to_scalar_float(prediction.confidence)
        is_acc = _to_scalar_bool(prediction.is_accepted)

        top_k_indices = _to_1d_list(prediction.top_k_indices)
        top_k_probs = _to_1d_list(prediction.top_k_probabilities)

        if getattr(prediction, "top_k_glosses", None):
            top_k_glosses = prediction.top_k_glosses
        else:
            top_k_glosses = [self.resolve_gloss(int(idx)) for idx in top_k_indices]
        top_k_probabilities = [float(p) for p in top_k_probs]

        if getattr(prediction, "predicted_gloss", None):
            gloss = prediction.predicted_gloss
        else:
            gloss = self.resolve_gloss(cls_idx)

        boundary_ev = segment.boundary_evidence if segment is not None else None
        start_f = segment.start_frame if segment is not None else None
        end_f = segment.end_frame if segment is not None else None

        return SignToken(
            gloss=gloss,
            class_index=cls_idx,
            confidence=conf,
            is_accepted=is_acc,
            top_k_glosses=top_k_glosses,
            top_k_probabilities=top_k_probabilities,
            boundary_evidence=boundary_ev,
            start_frame=start_f,
            end_frame=end_f,
        )

    # --- Batch API ---
    def process_batch(
        self,
        segments: List[SignSegment],
        predictions: List[ConfidenceOutput],
    ) -> SentenceTranscript:
        """Processes aligned lists of Phase 9 segments and Phase 8 predictions.

        Args:
            segments: List of SignSegment objects from Phase 9.
            predictions: List of single-sample ConfidenceOutput objects from Phase 8.

        Returns:
            SentenceTranscript container.

        Raises:
            ValueError: If segment and prediction counts do not match.
        """
        if len(segments) != len(predictions):
            raise ValueError(
                f"Mismatched segment and prediction counts: "
                f"got {len(segments)} segments and {len(predictions)} predictions."
            )

        raw_tokens: List[SignToken] = []
        filtered_tokens: List[SignToken] = []

        self.suppressor.reset()

        for seg, pred in zip(segments, predictions):
            bounded = BoundedSignPrediction(segment=seg, prediction=pred)
            bounded.validate()

            token = self.create_token(pred, segment=seg)
            raw_tokens.append(token)

            # Separate segments reset duplicate suppressor state to preserve intentional repeats
            self.suppressor.reset()
            processed = self.suppressor.process_token(token)
            if processed is not None:
                filtered_tokens.append(processed)

        gloss_sequence = [tok.gloss for tok in filtered_tokens]
        text = self.builder.build_sentence(gloss_sequence)

        mean_conf = (
            float(np.mean([t.confidence for t in filtered_tokens]))
            if filtered_tokens
            else 0.0
        )

        return SentenceTranscript(
            text=text,
            raw_tokens=raw_tokens,
            filtered_tokens=filtered_tokens,
            mean_confidence=mean_conf,
            is_edited=False,
        )

    # --- Streaming API ---
    def add_segment_prediction(
        self,
        segment: SignSegment,
        confidence_output: ConfidenceOutput,
    ) -> Optional[SignToken]:
        """Processes an incremental segment and prediction pair.

        Resets suppressor state for new segment and appends token to buffer if accepted.
        """
        bounded = BoundedSignPrediction(segment=segment, prediction=confidence_output)
        bounded.validate()

        token = self.create_token(confidence_output, segment=segment)
        self._streaming_raw_tokens.append(token)

        # Boundary reset between segments
        self.notify_boundary()
        processed = self.suppressor.process_token(token)
        if processed is not None:
            self._streaming_filtered_tokens.append(processed)
            return processed
        return None

    def add_token(self, token: SignToken) -> Optional[SignToken]:
        """Appends a pre-constructed token incrementally to the streaming buffer."""
        self._streaming_raw_tokens.append(token)
        processed = self.suppressor.process_token(token)
        if processed is not None:
            self._streaming_filtered_tokens.append(processed)
            return processed
        return None

    def notify_boundary(self) -> None:
        """Notifies a boundary or rest-pose transition to reset duplicate suppressor state."""
        self.suppressor.reset()

    def get_transcript(self) -> SentenceTranscript:
        """Returns current SentenceTranscript from buffered tokens without resetting state."""
        gloss_sequence = [tok.gloss for tok in self._streaming_filtered_tokens]
        text = self.builder.build_sentence(gloss_sequence)

        mean_conf = (
            float(np.mean([t.confidence for t in self._streaming_filtered_tokens]))
            if self._streaming_filtered_tokens
            else 0.0
        )

        return SentenceTranscript(
            text=text,
            raw_tokens=list(self._streaming_raw_tokens),
            filtered_tokens=list(self._streaming_filtered_tokens),
            mean_confidence=mean_conf,
            is_edited=False,
        )

    def finalize_sentence(self) -> SentenceTranscript:
        """Finalizes current streaming transcript and resets stream buffer."""
        transcript = self.get_transcript()
        self.reset()
        return transcript

    def reset(self) -> None:
        """Clears streaming buffer and resets duplicate suppressor state."""
        self._streaming_raw_tokens.clear()
        self._streaming_filtered_tokens.clear()
        self.suppressor.reset()

    @classmethod
    def from_config(
        cls, config: Optional[ModelConfig] = None
    ) -> "SentenceReconstructor":
        """Factory method to instantiate SentenceReconstructor from ModelConfig."""
        if config is None:
            app_cfg = load_config()
            config = app_cfg.model
        return cls(config=config)

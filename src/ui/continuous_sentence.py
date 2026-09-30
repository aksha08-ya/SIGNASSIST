"""
Thread-safe Continuous Sentence Recognition session backend for SIGNASSIST (Phase 17).

Connects rolling window webcam predictions from WebcamRecognitionSession to the streaming
SentenceReconstructor API to generate real-time, context-smoothed English sentence transcripts.

This module MUST NOT import Streamlit or streamlit_webrtc.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Set
import logging
import threading
import numpy as np

from src.models.confidence import ConfidenceOutput
from src.nlp.sentence_reconstructor import SentenceReconstructor
from src.nlp.types import BoundedSignPrediction, SentenceTranscript, SignToken
from src.pipeline.boundary_detector import SignSegment
from src.ui.webcam import WebcamRecognitionSession, WebcamSessionResult

logger = logging.getLogger(__name__)


def _make_prediction_key(pred: BoundedSignPrediction) -> Any:
    """Generates a composite identity key for tracking consumed BoundedSignPrediction instances."""
    seg = getattr(pred, "segment", None)
    start_f = getattr(seg, "start_frame", 0) if seg else 0
    end_f = getattr(seg, "end_frame", 0) if seg else 0

    p = getattr(pred, "prediction", None)
    cls_val = getattr(p, "predicted_class", 0) if p else 0
    if hasattr(cls_val, "item"):
        cls_idx = int(cls_val.item())
    elif isinstance(cls_val, (int, np.integer)):
        cls_idx = int(cls_val)
    else:
        cls_idx = 0

    return (id(pred), start_f, end_f, cls_idx)


@dataclass(frozen=True)
class ContinuousSentenceResult:
    """Immutable snapshot of the current continuous sentence recognition session state."""

    status: str
    recognition_available: bool
    current_sentence: str
    current_text: str
    mean_confidence: float
    raw_token_count: int
    filtered_token_count: int
    latest_token: Optional[SignToken]
    latest_prediction: Optional[BoundedSignPrediction]
    transcript: SentenceTranscript
    total_frames_received: int
    processed_windows: int
    processing_fps: float
    latency_ms: float
    prediction_history: List[BoundedSignPrediction]


class ContinuousSentenceSession:
    """Thread-safe orchestrator coupling WebcamRecognitionSession to streaming SentenceReconstructor."""

    def __init__(
        self,
        webcam_session: Optional[WebcamRecognitionSession] = None,
        reconstructor: Optional[SentenceReconstructor] = None,
    ) -> None:
        """Initializes continuous sentence recognition session.

        Args:
            webcam_session: Optional pre-configured WebcamRecognitionSession instance.
            reconstructor: Optional pre-configured SentenceReconstructor instance.
        """
        self._lock = threading.Lock()
        self.webcam_session = webcam_session or WebcamRecognitionSession()
        self.reconstructor = reconstructor or SentenceReconstructor()

        self._consumed_prediction_ids: Set[Any] = set()
        self._latest_prediction: Optional[BoundedSignPrediction] = None

    def is_available(self) -> bool:
        """Returns True if the underlying recognition pipeline is available."""
        return self.webcam_session.is_available()

    def _consume_new_predictions(self, webcam_res: WebcamSessionResult) -> None:
        """Passes newly produced predictions from webcam session into SentenceReconstructor.

        Must be called under self._lock.
        """
        if not webcam_res.recognition_available:
            return

        for pred in webcam_res.prediction_history:
            if isinstance(pred, BoundedSignPrediction):
                key = _make_prediction_key(pred)
                if key not in self._consumed_prediction_ids:
                    if pred.segment is not None and pred.prediction is not None:
                        try:
                            self.reconstructor.add_segment_prediction(
                                segment=pred.segment,
                                confidence_output=pred.prediction,
                            )
                        except Exception as e:
                            logger.warning(f"Error adding prediction to SentenceReconstructor: {e}")
                    self._consumed_prediction_ids.add(key)
                    self._latest_prediction = pred

    def add_frame(self, frame: Any) -> ContinuousSentenceResult:
        """Ingests a video frame into webcam session and updates continuous sentence state.

        Args:
            frame: Expected BGR numpy image array [H, W, 3] or raw video frame.

        Returns:
            ContinuousSentenceResult snapshot.
        """
        webcam_res = self.webcam_session.add_frame(frame)
        with self._lock:
            self._consume_new_predictions(webcam_res)
            return self._get_result_unlocked(webcam_res)

    def get_result(self) -> ContinuousSentenceResult:
        """Returns current snapshot of continuous sentence state without resetting stream."""
        with self._lock:
            webcam_res = self.webcam_session.get_result()
            self._consume_new_predictions(webcam_res)
            return self._get_result_unlocked(webcam_res)

    def _get_result_unlocked(self, webcam_res: WebcamSessionResult) -> ContinuousSentenceResult:
        """Internal unlocked helper for constructing ContinuousSentenceResult."""
        transcript = self.reconstructor.get_transcript()

        latest_tok = transcript.filtered_tokens[-1] if transcript.filtered_tokens else (
            transcript.raw_tokens[-1] if transcript.raw_tokens else None
        )

        status = "unavailable" if not webcam_res.recognition_available else webcam_res.status

        return ContinuousSentenceResult(
            status=status,
            recognition_available=webcam_res.recognition_available,
            current_sentence=transcript.text,
            current_text=transcript.text,
            mean_confidence=transcript.mean_confidence,
            raw_token_count=len(transcript.raw_tokens),
            filtered_token_count=len(transcript.filtered_tokens),
            latest_token=latest_tok,
            latest_prediction=self._latest_prediction,
            transcript=transcript,
            total_frames_received=webcam_res.total_frames_received,
            processed_windows=webcam_res.processed_windows,
            processing_fps=webcam_res.processing_fps,
            latency_ms=webcam_res.last_latency_ms,
            prediction_history=list(webcam_res.prediction_history),
        )

    def finalize_sentence(self) -> ContinuousSentenceResult:
        """Finalizes current streaming transcript and resets reconstructor buffer for next sentence.

        Returns:
            ContinuousSentenceResult snapshot containing the finalized transcript.
        """
        with self._lock:
            webcam_res = self.webcam_session.get_result()
            self._consume_new_predictions(webcam_res)

            final_transcript = self.reconstructor.finalize_sentence()

            latest_tok = final_transcript.filtered_tokens[-1] if final_transcript.filtered_tokens else (
                final_transcript.raw_tokens[-1] if final_transcript.raw_tokens else None
            )

            status = "unavailable" if not webcam_res.recognition_available else webcam_res.status

            return ContinuousSentenceResult(
                status=status,
                recognition_available=webcam_res.recognition_available,
                current_sentence=final_transcript.text,
                current_text=final_transcript.text,
                mean_confidence=final_transcript.mean_confidence,
                raw_token_count=len(final_transcript.raw_tokens),
                filtered_token_count=len(final_transcript.filtered_tokens),
                latest_token=latest_tok,
                latest_prediction=self._latest_prediction,
                transcript=final_transcript,
                total_frames_received=webcam_res.total_frames_received,
                processed_windows=webcam_res.processed_windows,
                processing_fps=webcam_res.processing_fps,
                latency_ms=webcam_res.last_latency_ms,
                prediction_history=list(webcam_res.prediction_history),
            )


    def reset(self) -> None:
        """Resets both underlying webcam recognition session and sentence reconstructor."""
        with self._lock:
            self.webcam_session.reset()
            self.reconstructor.reset()
            self._consumed_prediction_ids.clear()
            self._latest_prediction = None

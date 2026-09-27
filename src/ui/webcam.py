"""
Thread-safe Webcam Recognition Session backend for SIGNASSIST (Phase 17).

Provides a Streamlit-independent rolling window frame buffer and recognition
session orchestrator wrapping an injected RecognitionPipeline adapter.

This module MUST NOT import Streamlit or streamlit_webrtc.
"""

from dataclasses import dataclass
from typing import Any, List, Optional
import logging
import threading
import time
import numpy as np

from src.nlp.types import BoundedSignPrediction
from src.video.file_reader import DecodedFrame, format_timestamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebcamSessionResult:
    """Immutable snapshot of the current webcam recognition session state."""

    status: str
    recognition_available: bool
    total_frames_received: int
    processed_windows: int
    processing_fps: float
    last_latency_ms: float
    latest_prediction: Optional[BoundedSignPrediction]
    prediction_history: List[BoundedSignPrediction]


class WebcamRecognitionSession:
    """Thread-safe, Streamlit-independent rolling window webcam recognition session."""

    def __init__(
        self,
        recognition_pipeline: Optional[Any] = None,
        window_size: int = 60,
        stride: int = 15,
        target_fps: float = 30.0,
        duplicate_suppression_window: float = 1.0,
    ) -> None:
        """Initializes webcam recognition session.

        Args:
            recognition_pipeline: Injected RecognitionPipeline adapter instance or None.
            window_size: Number of frames per recognition window (default 60).
            stride: Number of new frames added per recognition step (default 15).
            target_fps: Target video frame rate for timestamp computation (default 30.0).
            duplicate_suppression_window: Temporal window in seconds for duplicate sign suppression.
        """
        self._lock = threading.Lock()
        self.pipeline = recognition_pipeline
        self.window_size = int(window_size) if window_size > 0 else 60
        self.stride = int(stride) if stride > 0 else 15
        self.target_fps = float(target_fps) if target_fps > 0 else 30.0
        self.duplicate_suppression_window = float(duplicate_suppression_window)

        # Session state variables
        self._frame_buffer: List[np.ndarray] = []
        self._total_frames_received = 0
        self._processed_windows = 0
        self._processing_fps = 0.0
        self._last_latency_ms = 0.0
        self._prediction_history: List[BoundedSignPrediction] = []
        self._latest_prediction: Optional[BoundedSignPrediction] = None
        self._start_time: Optional[float] = None
        self._status = "idle"

        self.reset()

    def reset(self) -> None:
        """Resets frame buffer, prediction history, counters, and timing state."""
        with self._lock:
            self._frame_buffer.clear()
            self._total_frames_received = 0
            self._processed_windows = 0
            self._processing_fps = 0.0
            self._last_latency_ms = 0.0
            self._prediction_history.clear()
            self._latest_prediction = None
            self._start_time = None

            if self._check_pipeline_available():
                self._status = "ready"
            else:
                self._status = "unavailable"

    def _check_pipeline_available(self) -> bool:
        """Checks if the injected recognition pipeline is present and available."""
        if self.pipeline is None:
            return False
        if hasattr(self.pipeline, "is_available") and callable(getattr(self.pipeline, "is_available")):
            try:
                return bool(self.pipeline.is_available())
            except Exception as e:
                logger.warning(f"Error checking pipeline availability: {e}")
                return False
        return True

    def is_available(self) -> bool:
        """Returns True if recognition pipeline is available."""
        with self._lock:
            return self._check_pipeline_available()

    def add_frame(self, frame: Any) -> WebcamSessionResult:
        """Adds a single BGR webcam frame array to rolling buffer and processes when window is ready.

        Args:
            frame: Expected BGR numpy image array [H, W, 3].

        Returns:
            WebcamSessionResult snapshot.
        """
        # Validate input frame
        if not isinstance(frame, np.ndarray):
            logger.warning("Rejected invalid frame: expected numpy ndarray.")
            return self.get_result()

        if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
            logger.warning(f"Rejected malformed frame: invalid shape {getattr(frame, 'shape', None)}")
            return self.get_result()

        with self._lock:
            rec_avail = self._check_pipeline_available()
            if not rec_avail:
                self._status = "unavailable"

            self._total_frames_received += 1
            if self._start_time is None:
                self._start_time = time.perf_counter()

            # Store frame copy in rolling buffer
            self._frame_buffer.append(frame.copy())

            # Check if rolling window is complete
            if len(self._frame_buffer) >= self.window_size:
                # Extract window slice
                window_raw_frames = self._frame_buffer[: self.window_size]
                # Slide window: retain overlapping tail (window_size - stride)
                retain_count = max(0, self.window_size - self.stride)
                self._frame_buffer = self._frame_buffer[-retain_count:] if retain_count > 0 else []

                if rec_avail and self.pipeline is not None:
                    self._status = "processing"
                    start_frame_idx = self._total_frames_received - len(window_raw_frames)

                    # Build DecodedFrame objects for pipeline
                    decoded_frames: List[DecodedFrame] = []
                    for idx, raw_img in enumerate(window_raw_frames):
                        f_idx = start_frame_idx + idx
                        t_sec = float(f_idx / self.target_fps)
                        t_str = format_timestamp(t_sec)
                        decoded_frames.append(
                            DecodedFrame(
                                frame=raw_img,
                                frame_index=f_idx,
                                timestamp_sec=t_sec,
                                timestamp=t_str,
                            )
                        )

                    primary_tracks = [None] * len(decoded_frames)

                    t0 = time.perf_counter()
                    try:
                        if hasattr(self.pipeline, "process_frames") and callable(getattr(self.pipeline, "process_frames")):
                            raw_preds = self.pipeline.process_frames(decoded_frames, primary_tracks)
                        elif callable(self.pipeline):
                            raw_preds = self.pipeline(decoded_frames, primary_tracks)
                        else:
                            raw_preds = []
                    except Exception as e:
                        logger.warning(f"Error executing recognition pipeline chunk: {e}")
                        raw_preds = []
                    t1 = time.perf_counter()

                    self._last_latency_ms = (t1 - t0) * 1000.0
                    self._processed_windows += 1

                    # Process predictions with duplicate suppression
                    if raw_preds:
                        for pred in raw_preds:
                            if isinstance(pred, BoundedSignPrediction):
                                if not self._is_duplicate(pred):
                                    self._prediction_history.append(pred)
                                    self._latest_prediction = pred

                    self._status = "ready"
                else:
                    self._status = "unavailable"
                    self._processed_windows += 1

            # Update processing FPS
            if self._start_time is not None:
                elapsed = time.perf_counter() - self._start_time
                if elapsed > 0:
                    self._processing_fps = float(self._total_frames_received / elapsed)

            return self._get_result_unlocked()

    def _is_duplicate(self, new_pred: BoundedSignPrediction) -> bool:
        """Determines if new_pred is a duplicate of the most recent prediction in history."""
        if not self._prediction_history:
            return False

        last_pred = self._prediction_history[-1]

        # Extract class indices safely
        new_class = self._extract_class_index(new_pred)
        last_class = self._extract_class_index(last_pred)

        if new_class is None or last_class is None or new_class != last_class:
            return False

        # Compare temporal frame distance
        new_start = getattr(new_pred.segment, "start_frame", 0)
        last_start = getattr(last_pred.segment, "start_frame", 0)
        frame_diff = abs(new_start - last_start)

        threshold_frames = self.duplicate_suppression_window * self.target_fps
        return frame_diff <= threshold_frames

    @staticmethod
    def _extract_class_index(pred: BoundedSignPrediction) -> Optional[int]:
        """Safely extracts predicted class index integer from BoundedSignPrediction."""
        if pred is None or not hasattr(pred, "prediction") or pred.prediction is None:
            return None
        cls_val = getattr(pred.prediction, "predicted_class", None)
        if cls_val is None:
            return None
        if hasattr(cls_val, "item"):
            return int(cls_val.item())
        if hasattr(cls_val, "__getitem__") and not isinstance(cls_val, (str, int)):
            try:
                return int(cls_val[0])
            except (IndexError, TypeError):
                pass
        try:
            return int(cls_val)
        except (ValueError, TypeError):
            return None

    def get_result(self) -> WebcamSessionResult:
        """Returns an immutable snapshot of the current session state."""
        with self._lock:
            return self._get_result_unlocked()

    def _get_result_unlocked(self) -> WebcamSessionResult:
        """Internal unlocked helper constructing WebcamSessionResult."""
        rec_avail = self._check_pipeline_available()
        return WebcamSessionResult(
            status=self._status,
            recognition_available=rec_avail,
            total_frames_received=self._total_frames_received,
            processed_windows=self._processed_windows,
            processing_fps=self._processing_fps,
            last_latency_ms=self._last_latency_ms,
            latest_prediction=self._latest_prediction,
            prediction_history=list(self._prediction_history),
        )

"""
MediaPipe Holistic landmark extractor for SIGNASSIST.

Extracts raw 543 3D landmarks (face, pose, hands) from BGR video/webcam frames,
handling missing modalities safely without throwing unhandled exceptions.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp

    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    mp = None

from src.features.constants import (
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
)
from src.features.types import RawLandmarks
from src.utils.exceptions import LandmarkExtractionError


class MediaPipeHolisticExtractor:
    """Extracts raw MediaPipe Holistic landmarks from video frames."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._holistic = None
        if _MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Initializes the MediaPipe Holistic model instance."""
        if not _MEDIAPIPE_AVAILABLE:
            raise LandmarkExtractionError(
                "MediaPipe library is not installed or unavailable."
            )
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
                self._mp_holistic = mp.solutions.holistic
            else:
                import mediapipe.python.solutions.holistic as mp_holistic

                self._mp_holistic = mp_holistic

            self._holistic = self._mp_holistic.Holistic(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        except Exception as e:
            raise LandmarkExtractionError(
                f"Failed to initialize MediaPipe Holistic solutions: {e}",
                details={"error": str(e)},
            )

    def extract(self, frame: np.ndarray) -> RawLandmarks:
        """Extracts 543 raw landmarks from a BGR image frame.

        Args:
            frame: Input BGR image frame as a NumPy array (H, W, 3).

        Returns:
            RawLandmarks containing face (468), pose (33), left_hand (21), right_hand (21).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise LandmarkExtractionError(
                "Invalid frame input. Frame must be a non-empty NumPy array."
            )

        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise LandmarkExtractionError(
                f"Frame must be 3-channel BGR image. Got shape {frame.shape}."
            )

        if not _MEDIAPIPE_AVAILABLE or self._holistic is None:
            return RawLandmarks()

        try:
            # Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._holistic.process(rgb_frame)

            return self._parse_results(results)
        except Exception as e:
            raise LandmarkExtractionError(
                f"Failed during MediaPipe landmark extraction: {e}",
                details={"error": str(e)},
            )

    def _parse_results(self, results: any) -> RawLandmarks:
        """Parses MediaPipe Holistic output object into typed RawLandmarks."""
        raw = RawLandmarks()

        # Face Landmarks (468, 3)
        if getattr(results, "face_landmarks", None):
            face_arr = np.zeros((NUM_FACE_LANDMARKS, 3), dtype=np.float32)
            for idx, lm in enumerate(results.face_landmarks.landmark):
                if idx < NUM_FACE_LANDMARKS:
                    face_arr[idx] = [lm.x, lm.y, lm.z]
            raw.face = face_arr
            raw.has_face = True

        # Pose Landmarks (33, 4: x, y, z, visibility)
        if getattr(results, "pose_landmarks", None):
            pose_arr = np.zeros((NUM_POSE_LANDMARKS, 4), dtype=np.float32)
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx < NUM_POSE_LANDMARKS:
                    vis = getattr(lm, "visibility", 1.0)
                    pose_arr[idx] = [lm.x, lm.y, lm.z, vis]
            raw.pose = pose_arr
            raw.has_pose = True

        # Left Hand Landmarks (21, 3)
        if getattr(results, "left_hand_landmarks", None):
            lh_arr = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
            for idx, lm in enumerate(results.left_hand_landmarks.landmark):
                if idx < NUM_HAND_LANDMARKS:
                    lh_arr[idx] = [lm.x, lm.y, lm.z]
            raw.left_hand = lh_arr
            raw.has_left_hand = True

        # Right Hand Landmarks (21, 3)
        if getattr(results, "right_hand_landmarks", None):
            rh_arr = np.zeros((NUM_HAND_LANDMARKS, 3), dtype=np.float32)
            for idx, lm in enumerate(results.right_hand_landmarks.landmark):
                if idx < NUM_HAND_LANDMARKS:
                    rh_arr[idx] = [lm.x, lm.y, lm.z]
            raw.right_hand = rh_arr
            raw.has_right_hand = True

        return raw

    def close(self) -> None:
        """Releases MediaPipe resources."""
        if self._holistic is not None:
            try:
                self._holistic.close()
            except Exception:
                pass
            self._holistic = None

    def __enter__(self) -> "MediaPipeHolisticExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

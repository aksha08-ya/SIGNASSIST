"""Video Quality Assessor for SIGNASSIST (Phase 13).

Provides:
- VideoQualityReport typed dataclass
- VideoQualityAssessor for frame-level and video-level quality metrics:
  - Brightness / lighting score (grayscale mean)
  - Contrast score (grayscale standard deviation)
  - Blur / sharpness score (Laplacian variance)
  - Resolution compliance check
  - Hand / keypoint visibility ratio (supports optional perception pipeline signals)
  - Overall quality rating (EXCELLENT, GOOD, ACCEPTABLE, POOR)
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union
import cv2
import numpy as np

from src.utils.config import AppConfig, VideoConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class VideoQualityReport:
    """Structured report containing video quality metrics and overall rating."""

    average_brightness: float
    average_contrast: float
    average_blur_score: float
    resolution_ok: bool
    hand_visibility_ratio: float
    sampled_frames: int
    overall_rating: str  # EXCELLENT, GOOD, ACCEPTABLE, POOR


class VideoQualityAssessor:
    """Evaluates video quality parameters against configured quality thresholds."""

    def __init__(
        self,
        config: Optional[Union[VideoConfig, AppConfig]] = None,
    ) -> None:
        if isinstance(config, AppConfig):
            self.config = config.video
        elif isinstance(config, VideoConfig):
            self.config = config
        elif config is None:
            self.config = VideoConfig()
        else:
            raise TypeError(
                f"Expected config to be VideoConfig or AppConfig, got {type(config)}"
            )

    def assess_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Calculates brightness, contrast, and blur score for a single BGR image array."""
        if frame is None or frame.size == 0:
            return {"brightness": 0.0, "contrast": 0.0, "blur_score": 0.0}

        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            return {"brightness": 0.0, "contrast": 0.0, "blur_score": 0.0}

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
        }

    def assess_video(
        self,
        video_source: Union["VideoFileReader", Iterable[Union[np.ndarray, "DecodedFrame"]]],
        hand_visibility_signals: Optional[List[float]] = None,
        num_samples: int = 15,
    ) -> VideoQualityReport:
        """Evaluates video quality metrics across frame samples.

        Args:
            video_source: VideoFileReader instance or iterable of DecodedFrame / np.ndarray.
            hand_visibility_signals: Optional list of float visibility scores (0.0..1.0)
                from downstream perception pipeline (e.g., MediaPipe landmark confidence/presence).
            num_samples: Number of frames to sample when a VideoFileReader is passed.

        Returns:
            VideoQualityReport containing metric averages, resolution status, hand visibility ratio,
            and overall quality rating string.
        """
        from src.video.file_reader import DecodedFrame, VideoFileReader

        frames_to_assess: List[np.ndarray] = []
        width, height = 0, 0

        if isinstance(video_source, VideoFileReader):
            metadata = video_source.load_metadata()
            width, height = metadata.width, metadata.height
            samples = video_source.sample_frames(num_samples=num_samples)
            frames_to_assess = [s.frame for s in samples]
        elif isinstance(video_source, Iterable):
            for item in video_source:
                if isinstance(item, DecodedFrame):
                    frames_to_assess.append(item.frame)
                elif isinstance(item, np.ndarray):
                    frames_to_assess.append(item)

            if frames_to_assess:
                h, w = frames_to_assess[0].shape[:2]
                width, height = w, h

        if not frames_to_assess:
            return VideoQualityReport(
                average_brightness=0.0,
                average_contrast=0.0,
                average_blur_score=0.0,
                resolution_ok=False,
                hand_visibility_ratio=0.0,
                sampled_frames=0,
                overall_rating="POOR",
            )

        brightness_list: List[float] = []
        contrast_list: List[float] = []
        blur_list: List[float] = []

        for f in frames_to_assess:
            metrics = self.assess_frame(f)
            brightness_list.append(metrics["brightness"])
            contrast_list.append(metrics["contrast"])
            blur_list.append(metrics["blur_score"])

        avg_brightness = float(np.mean(brightness_list))
        avg_contrast = float(np.mean(contrast_list))
        avg_blur = float(np.mean(blur_list))

        # Resolution check against configured minimum/target dimensions
        resolution_ok = bool(width >= 320 and height >= 240)

        # Hand visibility ratio calculation: accepts optional perception signals if provided
        if hand_visibility_signals is not None and len(hand_visibility_signals) > 0:
            hand_visibility_ratio = float(np.mean(np.clip(hand_visibility_signals, 0.0, 1.0)))
        else:
            hand_visibility_ratio = 1.0

        overall_rating = self._determine_overall_rating(
            avg_brightness, avg_contrast, avg_blur, resolution_ok, hand_visibility_ratio
        )

        return VideoQualityReport(
            average_brightness=avg_brightness,
            average_contrast=avg_contrast,
            average_blur_score=avg_blur,
            resolution_ok=resolution_ok,
            hand_visibility_ratio=hand_visibility_ratio,
            sampled_frames=len(frames_to_assess),
            overall_rating=overall_rating,
        )

    def _determine_overall_rating(
        self,
        brightness: float,
        contrast: float,
        blur_score: float,
        resolution_ok: bool,
        hand_visibility_ratio: float,
    ) -> str:
        """Determines overall quality rating: EXCELLENT, GOOD, ACCEPTABLE, POOR."""
        if not resolution_ok or blur_score < 30.0 or brightness < 30.0 or brightness > 225.0 or hand_visibility_ratio < 0.2:
            return "POOR"

        if blur_score < 80.0 or brightness < 50.0 or brightness > 200.0 or contrast < 20.0 or hand_visibility_ratio < 0.5:
            return "ACCEPTABLE"

        if blur_score >= 150.0 and (80.0 <= brightness <= 180.0) and contrast >= 40.0 and hand_visibility_ratio >= 0.8:
            return "EXCELLENT"

        return "GOOD"

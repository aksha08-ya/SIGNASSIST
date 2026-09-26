"""Unit tests for Phase 13 VideoQualityAssessor & VideoQualityReport."""

import cv2
import numpy as np
import pytest

from src.utils.config import VideoConfig
from src.video.quality import VideoQualityAssessor, VideoQualityReport


def test_assess_frame_metrics():
    """Test assess_frame metric outputs on synthetic dark, bright, contrast, and sharp frames."""
    assessor = VideoQualityAssessor()

    # Dark frame (value 10)
    dark_frame = np.full((100, 100, 3), 10, dtype=np.uint8)
    dark_res = assessor.assess_frame(dark_frame)
    assert dark_res["brightness"] == pytest.approx(10.0)
    assert dark_res["contrast"] == pytest.approx(0.0)
    assert dark_res["blur_score"] == pytest.approx(0.0)

    # Bright frame (value 240)
    bright_frame = np.full((100, 100, 3), 240, dtype=np.uint8)
    bright_res = assessor.assess_frame(bright_frame)
    assert bright_res["brightness"] == pytest.approx(240.0)
    assert bright_res["contrast"] == pytest.approx(0.0)

    # High contrast / sharp checkerboard
    checkerboard = np.zeros((100, 100, 3), dtype=np.uint8)
    checkerboard[:50, :50] = 255
    checkerboard[50:, 50:] = 255
    sharp_res = assessor.assess_frame(checkerboard)
    assert sharp_res["brightness"] == pytest.approx(127.5, abs=1.0)
    assert sharp_res["contrast"] > 100.0
    assert sharp_res["blur_score"] > 500.0


def test_assess_video_synthetic_frames():
    """Test assess_video on a list of synthetic frames."""
    assessor = VideoQualityAssessor()

    # Create 5 normal resolution sharp frames
    frames = []
    for i in range(5):
        f = np.full((480, 640, 3), 128, dtype=np.uint8)
        # Add high contrast pattern for sharpness/contrast
        cv2.rectangle(f, (50, 50), (300, 300), (0, 0, 0), -1)
        cv2.circle(f, (200, 200), 50, (255, 255, 255), -1)
        frames.append(f)

    report = assessor.assess_video(frames)

    assert isinstance(report, VideoQualityReport)
    assert report.sampled_frames == 5
    assert report.resolution_ok is True
    assert 50.0 <= report.average_brightness <= 200.0
    assert report.average_contrast > 20.0
    assert report.average_blur_score > 100.0
    assert report.overall_rating in {"EXCELLENT", "GOOD", "ACCEPTABLE", "POOR"}


def test_low_resolution_rejection():
    """Test resolution_ok becomes False for frames below 320x240."""
    assessor = VideoQualityAssessor()
    tiny_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    report = assessor.assess_video([tiny_frame])
    assert report.resolution_ok is False
    assert report.overall_rating == "POOR"


def test_hand_visibility_signal_integration():
    """Test external hand visibility signals are consumed and reflected in quality report."""
    assessor = VideoQualityAssessor()
    frames = [np.full((480, 640, 3), 128, dtype=np.uint8) for _ in range(3)]

    # 1. External signals provided: 0.9, 0.8, 1.0 -> avg 0.9
    report_high = assessor.assess_video(frames, hand_visibility_signals=[0.9, 0.8, 1.0])
    assert report_high.hand_visibility_ratio == pytest.approx(0.9)

    # 2. External signals provided low: 0.1 -> poor rating
    report_low = assessor.assess_video(frames, hand_visibility_signals=[0.1])
    assert report_low.hand_visibility_ratio == pytest.approx(0.1)
    assert report_low.overall_rating == "POOR"

    # 3. No external signal provided defaults to 1.0
    report_default = assessor.assess_video(frames)
    assert report_default.hand_visibility_ratio == 1.0

"""Unit tests for Phase 13 VideoProcessor pipeline orchestrator."""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pytest

from src.models.confidence import ConfidenceOutput
from src.nlp.types import BoundedSignPrediction, SentenceTranscript
from src.pipeline.boundary_detector import SignSegment
from src.utils.exceptions import VideoProcessingError
from src.video.file_reader import DecodedFrame, format_timestamp
from src.video.processor import (
    RecognitionPipelineAdapter,
    TimestampedSignEntry,
    VideoProcessingResult,
    VideoProcessor,
)
from src.video.signer_tracker import BoundingBox, SignerTracker


def create_synthetic_video(
    file_path: Path,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    duration_sec: float = 1.0,
) -> Path:
    """Creates a synthetic video file for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_frames = int(round(fps * duration_sec))
    file_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.rectangle(frame, (50 + i * 2, 50), (200 + i * 2, 200), (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return file_path


class MockRecognitionAdapter:
    """Mock recognition pipeline adapter for testing VideoProcessor."""

    def __init__(self, predictions: Optional[List[BoundedSignPrediction]] = None):
        self.predictions = predictions or []
        self.called_with_chunks = 0

    def is_available(self) -> bool:
        return True

    def process_frames(
        self,
        frames: List[DecodedFrame],
        primary_tracks: List[Optional[BoundingBox]],
    ) -> List[BoundedSignPrediction]:
        self.called_with_chunks += 1
        if self.called_with_chunks == 1:
            return self.predictions
        return []


def test_processor_no_recognition_pipeline(tmp_path: Path):
    """Test VideoProcessor operating in recognition-unavailable mode."""
    vid_path = tmp_path / "no_rec.mp4"
    create_synthetic_video(vid_path, fps=30.0, duration_sec=1.0)

    processor = VideoProcessor(recognition_pipeline=None)
    result = processor.process_file(vid_path)

    assert isinstance(result, VideoProcessingResult)
    assert result.recognition_available is False
    assert len(result.timeline) == 0
    assert result.transcript is None
    assert result.tts_audio is None
    assert result.frames_processed == 30
    assert result.execution_seconds > 0.0
    assert result.processing_fps > 0.0


def test_processor_with_mock_recognition_pipeline(tmp_path: Path):
    """Test VideoProcessor with mock recognition adapter generating timeline and transcript."""
    vid_path = tmp_path / "mock_rec.mp4"
    create_synthetic_video(vid_path, fps=30.0, duration_sec=1.0)

    seg = SignSegment(start_frame=0, end_frame=15, duration=16, mean_energy=0.5, boundary_evidence=0.8, is_valid=True)
    pred = ConfidenceOutput(
        predicted_class=1,
        confidence=0.92,
        is_accepted=True,
        is_rejected=False,
        top_k_indices=[1, 2, 3],
        top_k_probabilities=[0.92, 0.05, 0.03],
    )
    bounded_pred = BoundedSignPrediction(segment=seg, prediction=pred)

    adapter = MockRecognitionAdapter(predictions=[bounded_pred])
    processor = VideoProcessor(recognition_pipeline=adapter)

    result = processor.process_file(vid_path)

    assert result.recognition_available is True
    assert len(result.timeline) == 1

    entry = result.timeline[0]
    assert isinstance(entry, TimestampedSignEntry)
    assert entry.start_frame == 0
    assert entry.end_frame == 15
    assert entry.start_timestamp_sec == 0.0
    assert entry.end_timestamp_sec == pytest.approx(0.5, abs=1e-2)
    assert entry.start_timestamp == "00:00.000"
    assert entry.end_timestamp == "00:00.500"
    assert entry.recognized_gloss == "1"
    assert entry.confidence == 0.92

    # Verify transcript reconstruction
    assert isinstance(result.transcript, SentenceTranscript)
    assert result.transcript.text != ""


def test_native_timestamp_preservation_non_30fps(tmp_path: Path):
    """Verify processor preserves native FPS (e.g. 25 FPS) for timestamps rather than target 30 FPS."""
    vid_path = tmp_path / "fps25.mp4"
    create_synthetic_video(vid_path, fps=25.0, duration_sec=1.0)

    seg = SignSegment(start_frame=0, end_frame=24, duration=25, mean_energy=0.5, boundary_evidence=0.8, is_valid=True)
    pred = ConfidenceOutput(
        predicted_class=0,
        confidence=0.88,
        is_accepted=True,
        is_rejected=False,
        top_k_indices=[0],
        top_k_probabilities=[0.88],
    )
    bounded_pred = BoundedSignPrediction(segment=seg, prediction=pred)

    adapter = MockRecognitionAdapter(predictions=[bounded_pred])
    processor = VideoProcessor(recognition_pipeline=adapter)

    result = processor.process_file(vid_path)
    entry = result.timeline[0]

    # At 25 FPS, frame 24 end_timestamp_sec = 24 / 25.0 = 0.96s
    assert result.video_metadata.native_fps == pytest.approx(25.0, abs=0.1)
    assert entry.end_timestamp_sec == pytest.approx(24 / 25.0, abs=1e-3)
    assert entry.end_timestamp == format_timestamp(24 / 25.0)


def test_manual_signer_selection(tmp_path: Path):
    """Verify manual_signer_id parameter configures tracker selection lock."""
    vid_path = tmp_path / "manual_signer.mp4"
    create_synthetic_video(vid_path, fps=30.0, duration_sec=1.0)

    processor = VideoProcessor()
    result = processor.process_file(vid_path, manual_signer_id=2)

    assert result.additional_metadata.get("manual_signer_id") == 2
    assert processor.tracker.manual_primary_id == 2


def test_invalid_file_handling(tmp_path: Path):
    """Verify processor raises VideoProcessingError on missing or invalid video path."""
    missing_path = tmp_path / "missing.mp4"
    processor = VideoProcessor()

    with pytest.raises(VideoProcessingError):
        processor.process_file(missing_path)

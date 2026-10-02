"""Focused unit tests for RecognitionPipeline boundary trimming bypass feature."""

from __future__ import annotations

from typing import List
import pytest
import numpy as np
import torch

from src.features.types import SelectedGraphLandmarks
from src.pipeline.recognition import RecognitionPipeline
from src.video.file_reader import DecodedFrame


def _create_mock_decoded_frames(num_frames: int = 15) -> List[DecodedFrame]:
    """Generates synthetic DecodedFrame instances for testing."""
    frames = []
    for idx in range(num_frames):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frames.append(
            DecodedFrame(
                frame=img,
                frame_index=idx,
                timestamp_sec=float(idx * 0.033),
                timestamp=f"00:{idx:02d}.000",
            )
        )
    return frames


def test_default_use_boundary_trimming_is_true():
    """Verify RecognitionPipeline defaults to use_boundary_trimming=True."""
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    assert pipeline.use_boundary_trimming is True


def test_use_boundary_trimming_false_initialization():
    """Verify RecognitionPipeline preserves explicit use_boundary_trimming=False parameter."""
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        backend="experimental_42",
        use_boundary_trimming=False,
    )
    assert pipeline.use_boundary_trimming is False


def test_process_frames_with_boundary_trimming_true(monkeypatch):
    """Verify process_frames with use_boundary_trimming=True operates normally."""
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        backend="experimental_42",
        use_boundary_trimming=True,
    )

    # Mock preprocess_frame to avoid requiring real MediaPipe execution on synthetic images
    def mock_preprocess_frame(self, df):
        nodes = np.random.randn(115, 5).astype(np.float32)
        mask = np.ones((115,), dtype=bool)
        lm = SelectedGraphLandmarks(nodes=nodes, mask=mask)
        return (None, lm, None)

    monkeypatch.setattr(RecognitionPipeline, "preprocess_frame", mock_preprocess_frame)

    frames = _create_mock_decoded_frames(num_frames=20)
    preds = pipeline.process_frames(frames=frames, primary_tracks=[None] * len(frames))

    assert isinstance(preds, list)
    if preds:
        for p in preds:
            p.validate()
            assert hasattr(p, "segment")
            assert hasattr(p, "prediction")


def test_process_frames_with_boundary_trimming_false_bypasses_trimming(monkeypatch):
    """Verify process_frames with use_boundary_trimming=False classifies full sequence without trimming."""
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        backend="experimental_42",
        use_boundary_trimming=False,
    )

    def mock_preprocess_frame(self, df):
        nodes = np.random.randn(115, 5).astype(np.float32)
        mask = np.ones((115,), dtype=bool)
        lm = SelectedGraphLandmarks(nodes=nodes, mask=mask)
        return (None, lm, None)

    monkeypatch.setattr(RecognitionPipeline, "preprocess_frame", mock_preprocess_frame)

    num_frames = 25
    frames = _create_mock_decoded_frames(num_frames=num_frames)
    preds = pipeline.process_frames(frames=frames, primary_tracks=[None] * len(frames))

    assert isinstance(preds, list)
    assert len(preds) == 1, "Bypass mode must return exactly 1 prediction for full sequence"

    pred = preds[0]
    pred.validate()

    assert pred.segment.start_frame == 0
    assert pred.segment.end_frame == num_frames - 1
    assert pred.segment.duration == num_frames
    assert pred.prediction.confidence is not None


def test_experimental_backend_42_class_output_preserved_in_both_modes(monkeypatch):
    """Verify both trimming modes output valid 42-class gloss predictions."""
    def mock_preprocess_frame(self, df):
        nodes = np.random.randn(115, 5).astype(np.float32)
        mask = np.ones((115,), dtype=bool)
        lm = SelectedGraphLandmarks(nodes=nodes, mask=mask)
        return (None, lm, None)

    monkeypatch.setattr(RecognitionPipeline, "preprocess_frame", mock_preprocess_frame)

    frames = _create_mock_decoded_frames(num_frames=15)

    pipe_trim = RecognitionPipeline(mode="dynamic_only", backend="experimental_42", use_boundary_trimming=True)
    pipe_no_trim = RecognitionPipeline(mode="dynamic_only", backend="experimental_42", use_boundary_trimming=False)

    preds_trim = pipe_trim.process_frames(frames=frames, primary_tracks=[None] * len(frames))
    preds_no_trim = pipe_no_trim.process_frames(frames=frames, primary_tracks=[None] * len(frames))

    for pred in preds_trim + preds_no_trim:
        pred.validate()
        gloss = pred.prediction.predicted_gloss
        assert isinstance(gloss, str)
        assert len(gloss) > 0

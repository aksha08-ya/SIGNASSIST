"""Automated smoke test for WebcamRecognitionSession and RecognitionPipeline integration (Phase 17)."""

import numpy as np
import pytest
from src.ui.services import get_default_recognition_pipeline
from src.ui.webcam import WebcamRecognitionSession, WebcamSessionResult


def test_webcam_session_fake_frames_to_prediction():
    """Verifies that pushing fake video frames into WebcamRecognitionSession triggers pipeline inference,

    produces 60-frame windows, calculates confidence, applies rejection policy, and updates prediction history.
    """
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    assert pipeline is not None
    assert pipeline.is_available()

    session = WebcamRecognitionSession(recognition_pipeline=pipeline, window_size=60, stride=15)
    initial_res = session.get_result()
    assert initial_res.total_frames_received == 0
    assert initial_res.processed_windows == 0
    assert initial_res.latest_prediction is None
    assert len(initial_res.prediction_history) == 0

    # Push 65 fake BGR frames (480x640x3) to trigger 1 complete 60-frame window
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(65):
        dummy_frame[100:200, 100:200] = (i * 5) % 255
        res = session.add_frame(dummy_frame)

    final_res = session.get_result()

    # 1. Frames counted correctly
    assert final_res.total_frames_received == 65
    # 2. Window size reached and processed
    assert final_res.processed_windows == 1
    # 3. Processing FPS computed
    assert final_res.processing_fps > 0.0
    # 4. Latency measured
    assert final_res.last_latency_ms >= 0.0
    # 5. Prediction produced
    assert final_res.latest_prediction is not None
    assert len(final_res.prediction_history) == 1

    pred = final_res.latest_prediction.prediction
    assert pred is not None
    assert hasattr(pred, "confidence")
    assert hasattr(pred, "is_accepted")
    assert hasattr(pred, "is_rejected")
    # 6. Rejection policy preserved
    assert float(pred.confidence.item() if hasattr(pred.confidence, "item") else pred.confidence) <= 1.0


def test_webcam_session_reset():
    """Verifies that reset() clears buffers, counters, and prediction history."""
    pipeline = get_default_recognition_pipeline(backend="experimental_42")
    session = WebcamRecognitionSession(recognition_pipeline=pipeline, window_size=60, stride=15)

    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    for _ in range(65):
        session.add_frame(dummy_frame)

    assert session.get_result().total_frames_received == 65
    session.reset()

    reset_res = session.get_result()
    assert reset_res.total_frames_received == 0
    assert reset_res.processed_windows == 0
    assert reset_res.latest_prediction is None
    assert len(reset_res.prediction_history) == 0

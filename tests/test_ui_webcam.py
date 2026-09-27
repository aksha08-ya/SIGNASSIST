"""
Unit tests for Thread-Safe WebcamRecognitionSession Backend (Phase 17).
"""

from unittest.mock import MagicMock
import threading
import numpy as np
import pytest

from src.models.confidence import ConfidenceOutput
from src.nlp.types import BoundedSignPrediction
from src.pipeline.boundary_detector import SignSegment
from src.ui.webcam import WebcamRecognitionSession, WebcamSessionResult


# Helper to construct a valid dummy BoundedSignPrediction
def create_dummy_prediction(class_idx: int = 5, start_frame: int = 10, end_frame: int = 25) -> BoundedSignPrediction:
    seg = SignSegment(
        start_frame=start_frame,
        end_frame=end_frame,
        duration=end_frame - start_frame + 1,
        mean_energy=0.5,
        boundary_evidence=0.8,
        is_valid=True,
    )
    conf = MagicMock(spec=ConfidenceOutput)
    conf.predicted_class = np.array([class_idx])
    conf.confidence = np.array([0.9])
    conf.is_accepted = np.array([True])
    conf.is_rejected = np.array([False])
    conf.top_k_indices = np.array([[class_idx, 0, 1]])
    conf.top_k_probabilities = np.array([[0.9, 0.05, 0.05]])
    return BoundedSignPrediction(segment=seg, prediction=conf)


def test_no_streamlit_import_in_webcam_module() -> None:
    """Verifies that src.ui.webcam does not import Streamlit or streamlit_webrtc."""
    import sys
    assert "src.ui.webcam" in sys.modules
    webcam_mod = sys.modules["src.ui.webcam"]

    # Check module attributes to ensure Streamlit / webrtc were not imported
    for attr_name in dir(webcam_mod):
        attr_val = getattr(webcam_mod, attr_name)
        if hasattr(attr_val, "__name__"):
            assert "streamlit" not in attr_val.__name__.lower()
            assert "webrtc" not in attr_val.__name__.lower()


def test_session_init_defaults() -> None:
    """Tests session initialization defaults and initial result snapshot."""
    session = WebcamRecognitionSession()
    res = session.get_result()

    assert isinstance(res, WebcamSessionResult)
    assert res.status == "unavailable"
    assert res.recognition_available is False
    assert res.total_frames_received == 0
    assert res.processed_windows == 0
    assert res.latest_prediction is None
    assert res.prediction_history == []


def test_session_init_with_available_pipeline() -> None:
    """Tests session initialization with an available mock pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline)
    res = session.get_result()

    assert res.status == "ready"
    assert res.recognition_available is True


def test_session_unavailable_pipeline_handling() -> None:
    """Tests handling of an unavailable pipeline during frame additions."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = False

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=10, stride=5)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add 10 frames to trigger window
    for _ in range(10):
        res = session.add_frame(dummy_frame)

    assert res.status == "unavailable"
    assert res.recognition_available is False
    assert res.total_frames_received == 10
    assert res.processed_windows == 1
    assert res.latest_prediction is None
    assert res.prediction_history == []
    mock_pipeline.process_frames.assert_not_called()


def test_malformed_frame_rejection() -> None:
    """Tests clean rejection of malformed or invalid input frames."""
    session = WebcamRecognitionSession()

    # Invalid types and shapes
    res1 = session.add_frame("not_a_numpy_array")
    res2 = session.add_frame(np.zeros((10,), dtype=np.uint8))
    res3 = session.add_frame(np.zeros((0, 0, 3), dtype=np.uint8))

    assert res1.total_frames_received == 0
    assert res2.total_frames_received == 0
    assert res3.total_frames_received == 0


def test_buffering_before_full_window() -> None:
    """Tests frame buffering before window size (60) is reached."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=60, stride=15)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Add 59 frames
    for i in range(59):
        res = session.add_frame(dummy_frame)

    assert res.total_frames_received == 59
    assert res.processed_windows == 0
    assert res.latest_prediction is None
    mock_pipeline.process_frames.assert_not_called()


def test_first_window_processing_and_prediction_preservation() -> None:
    """Tests processing when the 60th frame arrives, preserving real BoundedSignPrediction."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    pred_obj = create_dummy_prediction(class_idx=5, start_frame=10, end_frame=25)
    mock_pipeline.process_frames.return_value = [pred_obj]

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=60, stride=15)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(60):
        res = session.add_frame(dummy_frame)

    assert res.total_frames_received == 60
    assert res.processed_windows == 1
    assert res.latest_prediction is pred_obj
    assert len(res.prediction_history) == 1
    assert res.prediction_history[0] is pred_obj

    # Verify pipeline process_frames arguments
    mock_pipeline.process_frames.assert_called_once()
    call_args = mock_pipeline.process_frames.call_args[0]
    decoded_frames, primary_tracks = call_args[0], call_args[1]
    assert len(decoded_frames) == 60
    assert len(primary_tracks) == 60
    assert all(t is None for t in primary_tracks)


def test_overlap_and_stride_behavior() -> None:
    """Tests window sliding with overlap (window_size=60, stride=15)."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    mock_pipeline.process_frames.return_value = []

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=60, stride=15)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. First window: 60 frames
    for _ in range(60):
        session.add_frame(dummy_frame)

    assert session.get_result().processed_windows == 1

    # 2. Add 14 more frames (total 74) -> no 2nd window yet
    for _ in range(14):
        session.add_frame(dummy_frame)

    assert session.get_result().processed_windows == 1

    # 3. Add 15th frame (total 75) -> 2nd window processed
    res = session.add_frame(dummy_frame)
    assert res.total_frames_received == 75
    assert res.processed_windows == 2
    assert mock_pipeline.process_frames.call_count == 2


def test_duplicate_suppression() -> None:
    """Tests duplicate prediction suppression across overlapping windows."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    pred1 = create_dummy_prediction(class_idx=5, start_frame=10, end_frame=25)
    pred2_duplicate = create_dummy_prediction(class_idx=5, start_frame=15, end_frame=30)  # Same class, frame diff 5 <= 30
    pred3_different = create_dummy_prediction(class_idx=8, start_frame=20, end_frame=35)  # Different class

    mock_pipeline.process_frames.side_effect = [[pred1], [pred2_duplicate], [pred3_different]]

    session = WebcamRecognitionSession(
        recognition_pipeline=mock_pipeline,
        window_size=10,
        stride=5,
        duplicate_suppression_window=1.0,  # 30 frames
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Window 1: pred1 added
    for _ in range(10):
        session.add_frame(dummy_frame)

    assert len(session.get_result().prediction_history) == 1
    assert session.get_result().latest_prediction is pred1

    # Window 2: pred2 duplicate suppressed
    for _ in range(5):
        session.add_frame(dummy_frame)

    assert len(session.get_result().prediction_history) == 1
    assert session.get_result().latest_prediction is pred1

    # Window 3: pred3 different class added
    for _ in range(5):
        session.add_frame(dummy_frame)

    res = session.get_result()
    assert len(res.prediction_history) == 2
    assert res.latest_prediction is pred3_different
    assert res.prediction_history == [pred1, pred3_different]


def test_reset() -> None:
    """Tests session reset functionality."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    mock_pipeline.process_frames.return_value = [create_dummy_prediction(5, 0, 10)]

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=10, stride=5)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(10):
        session.add_frame(dummy_frame)

    assert session.get_result().processed_windows == 1
    assert len(session.get_result().prediction_history) == 1

    # Reset session
    session.reset()
    res = session.get_result()

    assert res.status == "ready"
    assert res.total_frames_received == 0
    assert res.processed_windows == 0
    assert res.latest_prediction is None
    assert res.prediction_history == []


def test_thread_safe_concurrent_add_frame() -> None:
    """Tests thread-safe concurrent frame additions."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    mock_pipeline.process_frames.return_value = []

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=10, stride=5)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def worker() -> None:
        for _ in range(20):
            session.add_frame(dummy_frame)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    res = session.get_result()
    assert res.total_frames_received == 100
    assert res.processed_windows > 0

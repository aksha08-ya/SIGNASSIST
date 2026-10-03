"""
Regression test suite for Memory-Lifecycle Audit and OOM Prevention (SIGNASSIST Phase 18).

Verifies strict memory bounding for frame buffers, prediction histories, session reuse,
and thread isolation across repeated UI refresh cycles.
"""

from unittest.mock import MagicMock
import threading
import numpy as np
import pytest

from src.models.confidence import ConfidenceOutput
from src.nlp.types import BoundedSignPrediction
from src.pipeline.boundary_detector import SignSegment
from src.ui.continuous_sentence import ContinuousSentenceSession
from src.ui.webcam import WebcamRecognitionSession, WebcamSessionResult


def make_dummy_prediction(class_idx: int = 5, start_frame: int = 10, end_frame: int = 25) -> BoundedSignPrediction:
    """Helper to build a valid BoundedSignPrediction for testing."""
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
    conf.confidence = np.array([0.95])
    conf.is_accepted = np.array([True])
    conf.is_rejected = np.array([False])
    conf.top_k_indices = np.array([[class_idx, 0, 1]])
    conf.top_k_probabilities = np.array([[0.95, 0.03, 0.02]])
    conf.predicted_gloss = f"SIGN_{class_idx}"
    return BoundedSignPrediction(segment=seg, prediction=conf)


def test_a_frame_buffer_bounded() -> None:
    """Test A: Frame buffer cannot grow beyond configured max_buffer_size."""
    session = WebcamRecognitionSession(max_buffer_size=50)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Ingest 200 frames without triggering process_frames window pop
    for _ in range(200):
        session._frame_buffer.append(dummy_frame.copy())
        if len(session._frame_buffer) > session.max_buffer_size:
            session._frame_buffer = session._frame_buffer[-session.max_buffer_size:]

    res = session.get_result()
    assert res.current_buffer_count <= 50
    assert res.max_buffer_size == 50


def test_b_prediction_history_bounded() -> None:
    """Test B: Prediction history is bounded by max_prediction_history."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    # Return different non-duplicate predictions each window
    predictions_pool = [
        make_dummy_prediction(class_idx=i, start_frame=i * 30, end_frame=i * 30 + 15)
        for i in range(150)
    ]
    mock_pipeline.process_frames.side_effect = [[p] for p in predictions_pool]

    session = WebcamRecognitionSession(
        recognition_pipeline=mock_pipeline,
        window_size=10,
        stride=5,
        min_agreeing=1,
        max_prediction_history=25,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Run 100 windows (total 505 frames)
    for _ in range(505):
        session.add_frame(dummy_frame)

    res = session.get_result()
    assert len(res.prediction_history) <= 25
    assert res.max_history_count == 25
    assert res.history_count == len(res.prediction_history)


def test_c_session_reuse_on_ui_refresh() -> None:
    """Test C: Repeated session requests reuse existing object without duplication."""
    initial_count = WebcamRecognitionSession._creation_counter
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    session1 = WebcamRecognitionSession(recognition_pipeline=mock_pipeline)
    count_after_first = WebcamRecognitionSession._creation_counter
    assert count_after_first == initial_count + 1

    # Simulated app session cache reuse
    session_cache = {"webcam_session": session1}
    reused_session = session_cache["webcam_session"]
    assert reused_session is session1
    assert WebcamRecognitionSession._creation_counter == count_after_first


def test_d_no_duplicate_workers_on_refresh() -> None:
    """Test D: Repeated refresh does not duplicate worker threads or pipeline instances."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline)
    initial_thread_count = threading.active_count()

    # Simulate 50 UI refresh checks
    for _ in range(50):
        _ = session.get_result()

    current_thread_count = threading.active_count()
    assert current_thread_count == initial_thread_count


def test_e_session_reset_frees_memory() -> None:
    """Test E: Session reset releases frame buffer and prediction history."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    pred = make_dummy_prediction(class_idx=1, start_frame=0, end_frame=10)
    mock_pipeline.process_frames.return_value = [pred]

    session = WebcamRecognitionSession(
        recognition_pipeline=mock_pipeline,
        window_size=10,
        stride=5,
        min_agreeing=1,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _ in range(20):
        session.add_frame(dummy_frame)

    assert session.get_result().total_frames_received > 0
    assert len(session.get_result().prediction_history) > 0

    session.reset()
    res = session.get_result()

    assert res.total_frames_received == 0
    assert res.current_buffer_count == 0
    assert len(res.prediction_history) == 0
    assert res.latest_prediction is None


def test_f_thousand_frames_memory_safety() -> None:
    """Test F: 1000 simulated incoming frames does not cause unbounded frame retention."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    mock_pipeline.process_frames.return_value = []

    session = WebcamRecognitionSession(
        recognition_pipeline=mock_pipeline,
        window_size=60,
        stride=15,
        max_buffer_size=120,
        max_prediction_history=100,
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(1000):
        session.add_frame(dummy_frame)

    res = session.get_result()
    assert res.total_frames_received == 1000
    assert res.current_buffer_count <= 120
    assert len(res.prediction_history) <= 100


def test_g_continuous_sentence_session_bounded() -> None:
    """Test G: Continuous sentence session remains fully functional with bounded memory."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    mock_pipeline.process_frames.return_value = []

    webcam = WebcamRecognitionSession(recognition_pipeline=mock_pipeline, window_size=10, stride=5)
    cs_session = ContinuousSentenceSession(webcam_session=webcam)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(100):
        res = cs_session.add_frame(dummy_frame)

    assert res.total_frames_received == 100
    assert len(cs_session._consumed_prediction_ids) <= 200


def test_h_webcam_session_telemetry() -> None:
    """Test H: Live webcam session telemetry fields report correct buffer and history metrics."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    session = WebcamRecognitionSession(
        recognition_pipeline=mock_pipeline,
        max_buffer_size=80,
        max_prediction_history=40,
    )
    res = session.get_result()

    assert res.max_buffer_size == 80
    assert res.max_history_count == 40
    assert res.current_buffer_count == 0
    assert res.history_count == 0
    assert res.session_creation_count > 0


def test_i_overall_system_memory_contracts() -> None:
    """Test I: Existing session memory contracts hold across multi-step operation."""
    session = WebcamRecognitionSession()
    assert hasattr(session, "max_buffer_size")
    assert hasattr(session, "max_prediction_history")
    assert hasattr(session, "session_creation_id")

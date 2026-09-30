"""
Unit tests for SIGNASSIST Phase 17 Continuous Sentence Recognition backend (ContinuousSentenceSession).
"""

from unittest.mock import MagicMock
import threading
import numpy as np
import pytest

from src.models.confidence import ConfidenceOutput
from src.nlp.sentence_reconstructor import SentenceReconstructor
from src.nlp.types import BoundedSignPrediction, SentenceTranscript, SignToken
from src.pipeline.boundary_detector import SignSegment
from src.ui.continuous_sentence import (
    ContinuousSentenceResult,
    ContinuousSentenceSession,
)
from src.ui.webcam import WebcamRecognitionSession, WebcamSessionResult


def test_initialization():
    """Verifies ContinuousSentenceSession initializes correctly with defaults or injected instances."""
    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = True
    mock_reconstructor = MagicMock(spec=SentenceReconstructor)

    session = ContinuousSentenceSession(
        webcam_session=mock_webcam,
        reconstructor=mock_reconstructor,
    )

    assert session.webcam_session is mock_webcam
    assert session.reconstructor is mock_reconstructor
    assert session.is_available() is True


def test_unavailable_recognition():
    """Verifies clean unavailable state when recognition pipeline is unavailable."""
    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = False
    mock_webcam.get_result.return_value = WebcamSessionResult(
        status="unavailable",
        recognition_available=False,
        total_frames_received=10,
        processed_windows=0,
        processing_fps=30.0,
        last_latency_ms=0.0,
        latest_prediction=None,
        prediction_history=[],
    )

    session = ContinuousSentenceSession(webcam_session=mock_webcam)

    res = session.get_result()

    assert res.status == "unavailable"
    assert res.recognition_available is False
    assert res.current_sentence == ""
    assert res.raw_token_count == 0
    assert res.latest_prediction is None


def test_frame_delegation_and_malformed_frame_handling():
    """Verifies frame ingestion delegates to webcam session and handles malformed frames safely."""
    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = True
    mock_webcam.add_frame.return_value = WebcamSessionResult(
        status="ready",
        recognition_available=True,
        total_frames_received=1,
        processed_windows=0,
        processing_fps=30.0,
        last_latency_ms=0.0,
        latest_prediction=None,
        prediction_history=[],
    )

    session = ContinuousSentenceSession(webcam_session=mock_webcam)

    # Valid array
    valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = session.add_frame(valid_frame)
    mock_webcam.add_frame.assert_called_with(valid_frame)
    assert res.total_frames_received == 1

    # Malformed frame
    res_invalid = session.add_frame("not_an_array")
    mock_webcam.add_frame.assert_called_with("not_an_array")
    assert isinstance(res_invalid, ContinuousSentenceResult)


def test_prediction_consumption_and_no_duplicates():
    """Verifies new predictions are fed to SentenceReconstructor exactly once without duplicates."""
    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = True

    seg = SignSegment(start_frame=0, end_frame=60, duration=60, mean_energy=0.5, boundary_evidence=0.8)
    pred_output = ConfidenceOutput(
        predicted_class=1,
        confidence=0.9,
        top_k_indices=[1, 0, 2],
        top_k_probabilities=[0.9, 0.08, 0.02],
        is_accepted=True,
        is_rejected=False,
    )
    bounded_pred = BoundedSignPrediction(segment=seg, prediction=pred_output)

    webcam_snapshot = WebcamSessionResult(
        status="ready",
        recognition_available=True,
        total_frames_received=60,
        processed_windows=1,
        processing_fps=30.0,
        last_latency_ms=10.0,
        latest_prediction=bounded_pred,
        prediction_history=[bounded_pred],
    )
    mock_webcam.get_result.return_value = webcam_snapshot

    mock_reconstructor = MagicMock(spec=SentenceReconstructor)
    mock_reconstructor.get_transcript.return_value = SentenceTranscript(text="Hello", mean_confidence=0.9)

    session = ContinuousSentenceSession(webcam_session=mock_webcam, reconstructor=mock_reconstructor)

    # First get_result call consumes prediction
    res1 = session.get_result()
    mock_reconstructor.add_segment_prediction.assert_called_once_with(
        segment=seg,
        confidence_output=pred_output,
    )
    assert res1.current_sentence == "Hello"

    # Second get_result call with same snapshot should NOT call add_segment_prediction again
    mock_reconstructor.add_segment_prediction.reset_mock()
    res2 = session.get_result()
    mock_reconstructor.add_segment_prediction.assert_not_called()
    assert res2.current_sentence == "Hello"


def test_sentence_reconstruction_through_real_reconstructor_api():
    """Verifies end-to-end sentence streaming through real SentenceReconstructor instance."""
    vocab_map = {0: "hello", 1: "world"}
    reconstructor = SentenceReconstructor(vocab_map=vocab_map)

    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = True

    seg1 = SignSegment(start_frame=0, end_frame=60, duration=60, mean_energy=0.5, boundary_evidence=0.8)
    pred1 = ConfidenceOutput(
        predicted_class=0,
        confidence=0.95,
        top_k_indices=[0, 1],
        top_k_probabilities=[0.95, 0.05],
        is_accepted=True,
        is_rejected=False,
    )
    bounded1 = BoundedSignPrediction(segment=seg1, prediction=pred1)

    seg2 = SignSegment(start_frame=61, end_frame=120, duration=60, mean_energy=0.5, boundary_evidence=0.8)
    pred2 = ConfidenceOutput(
        predicted_class=1,
        confidence=0.90,
        top_k_indices=[1, 0],
        top_k_probabilities=[0.90, 0.10],
        is_accepted=True,
        is_rejected=False,
    )
    bounded2 = BoundedSignPrediction(segment=seg2, prediction=pred2)

    mock_webcam.get_result.return_value = WebcamSessionResult(
        status="ready",
        recognition_available=True,
        total_frames_received=120,
        processed_windows=2,
        processing_fps=30.0,
        last_latency_ms=10.0,
        latest_prediction=bounded2,
        prediction_history=[bounded1, bounded2],
    )

    session = ContinuousSentenceSession(webcam_session=mock_webcam, reconstructor=reconstructor)

    res = session.get_result()

    assert isinstance(res.transcript, SentenceTranscript)
    assert res.raw_token_count == 2
    assert "hello" in res.current_sentence.lower()
    assert "world" in res.current_sentence.lower()


def test_get_result_does_not_reset_and_finalize_clears_state():
    """Verifies get_result keeps streaming buffer intact while finalize_sentence returns transcript and clears state."""
    vocab_map = {0: "SIGN_A"}
    reconstructor = SentenceReconstructor(vocab_map=vocab_map)

    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = True

    seg = SignSegment(start_frame=0, end_frame=60, duration=60, mean_energy=0.5, boundary_evidence=0.8)
    pred = ConfidenceOutput(
        predicted_class=0,
        confidence=0.9,
        top_k_indices=[0],
        top_k_probabilities=[0.9],
        is_accepted=True,
        is_rejected=False,
    )
    bounded = BoundedSignPrediction(segment=seg, prediction=pred)

    mock_webcam.get_result.return_value = WebcamSessionResult(
        status="ready",
        recognition_available=True,
        total_frames_received=60,
        processed_windows=1,
        processing_fps=30.0,
        last_latency_ms=10.0,
        latest_prediction=bounded,
        prediction_history=[bounded],
    )

    session = ContinuousSentenceSession(webcam_session=mock_webcam, reconstructor=reconstructor)

    res_streaming = session.get_result()
    assert res_streaming.raw_token_count == 1

    # Finalize sentence
    res_final = session.finalize_sentence()
    assert res_final.raw_token_count == 1

    # State after finalize should be reset for new stream
    res_after = session.get_result()
    assert res_after.raw_token_count == 0


def test_reset_clears_webcam_and_sentence_state():
    """Verifies reset() calls reset on both webcam session and reconstructor."""
    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_reconstructor = MagicMock(spec=SentenceReconstructor)

    session = ContinuousSentenceSession(
        webcam_session=mock_webcam,
        reconstructor=mock_reconstructor,
    )

    session.reset()

    mock_webcam.reset.assert_called_once()
    mock_reconstructor.reset.assert_called_once()


def test_concurrent_access():
    """Verifies thread safety under concurrent add_frame and get_result operations."""
    mock_webcam = MagicMock(spec=WebcamRecognitionSession)
    mock_webcam.is_available.return_value = True
    mock_webcam.add_frame.return_value = WebcamSessionResult(
        status="ready",
        recognition_available=True,
        total_frames_received=1,
        processed_windows=0,
        processing_fps=30.0,
        last_latency_ms=0.0,
        latest_prediction=None,
        prediction_history=[],
    )
    mock_webcam.get_result.return_value = mock_webcam.add_frame.return_value

    session = ContinuousSentenceSession(webcam_session=mock_webcam)

    def worker():
        for _ in range(50):
            session.add_frame(np.zeros((10, 10, 3), dtype=np.uint8))
            session.get_result()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert session.get_result() is not None

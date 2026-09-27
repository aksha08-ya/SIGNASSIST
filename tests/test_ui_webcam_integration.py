"""
Unit and integration tests for SIGNASSIST Phase 17 Live Webcam WebRTC UI integration.
"""

from unittest.mock import MagicMock, patch
import sys
import numpy as np
import pytest

from src.models.confidence import ConfidenceOutput
from src.nlp.types import BoundedSignPrediction, SignSegment
from src.ui.services import get_default_recognition_pipeline
from src.ui.webcam import WebcamRecognitionSession, WebcamSessionResult


def test_lazy_webrtc_import_boundary():
    """Verifies that importing src.app does NOT load streamlit_webrtc at module import time."""
    # Ensure streamlit_webrtc is not loaded before import
    if "streamlit_webrtc" in sys.modules:
        del sys.modules["streamlit_webrtc"]

    import src.app  # noqa: F401

    assert "streamlit_webrtc" not in sys.modules, (
        "streamlit_webrtc was imported at top-level! It must ONLY be imported lazily inside render_mode_live_webcam()."
    )


def test_make_video_frame_callback_frame_conversion_and_isolation():
    """Verifies frame callback converts av.VideoFrame to BGR ndarray and avoids Streamlit state access."""
    from src.app import make_video_frame_callback

    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    mock_pipeline.process_frames.return_value = []

    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline)

    # Mock PyAV VideoFrame
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_av_frame = MagicMock()
    mock_av_frame.to_ndarray.return_value = dummy_bgr

    callback = make_video_frame_callback(session)

    # Verify calling callback with mocked Streamlit state access assertion
    with patch("streamlit.session_state", side_effect=AssertionError("Callback touched st.session_state!")):
        ret_frame = callback(mock_av_frame)

    assert ret_frame is mock_av_frame
    mock_av_frame.to_ndarray.assert_called_once_with(format="bgr24")
    assert session.get_result().total_frames_received == 1


def test_get_or_create_webcam_session_reuse():
    """Verifies get_or_create_webcam_session creates and reuses exactly one session per UI session state."""
    from src.app import get_or_create_webcam_session

    mock_state = {}

    with patch("streamlit.session_state", mock_state):
        session1 = get_or_create_webcam_session()
        assert isinstance(session1, WebcamRecognitionSession)
        assert mock_state.get("webcam_session") is session1

        session2 = get_or_create_webcam_session()
        assert session2 is session1


def test_extract_prediction_info_with_real_bounded_sign_prediction():
    """Verifies extract_prediction_info correctly parses a real BoundedSignPrediction."""
    from src.app import extract_prediction_info

    segment = SignSegment(
        start_frame=0,
        end_frame=60,
        duration=60,
        mean_energy=0.5,
        boundary_evidence=0.8,
        is_valid=True,
    )

    prediction = ConfidenceOutput(
        predicted_class=3,
        confidence=0.92,
        top_k_indices=[3, 1, 0],
        top_k_probabilities=[0.92, 0.05, 0.03],
        is_accepted=True,
        is_rejected=False,
    )
    bounded_pred = BoundedSignPrediction(segment=segment, prediction=prediction)

    info = extract_prediction_info(bounded_pred)

    assert info["class_index"] == 3
    assert info["gloss"] == "Sign #3"
    assert pytest.approx(info["confidence"]) == 0.92
    assert info["accepted"] is True
    assert len(info["top_k"]) == 3
    assert info["top_k"][0] == ("Sign #3", 0.92)


def test_webcam_session_reset_behavior():
    """Verifies WebcamRecognitionSession.reset clears frame buffer, counters, and prediction history."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_available.return_value = True
    session = WebcamRecognitionSession(recognition_pipeline=mock_pipeline)

    # Feed frames
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    for _ in range(10):
        session.add_frame(dummy_img)

    res_before = session.get_result()
    assert res_before.total_frames_received == 10

    session.reset()

    res_after = session.get_result()
    assert res_after.total_frames_received == 0
    assert res_after.processed_windows == 0
    assert res_after.latest_prediction is None
    assert len(res_after.prediction_history) == 0


def test_render_mode_live_webcam_handles_import_error():
    """Verifies render_mode_live_webcam handles missing or failing streamlit_webrtc gracefully."""
    from src.app import render_mode_live_webcam

    with patch("streamlit.error") as mock_st_error, patch.dict("sys.modules", {"streamlit_webrtc": None}):
        render_mode_live_webcam({})
        mock_st_error.assert_called_once()
        err_msg = mock_st_error.call_args[0][0]
        assert "Streamlit WebRTC component unavailable" in err_msg


def test_render_mode_live_webcam_unavailable_recognition_and_rendering():
    """Verifies UI rendering when recognition pipeline is unavailable but WebRTC is present."""
    from src.app import render_mode_live_webcam

    mock_webrtc_ctx = MagicMock()
    mock_webrtc_ctx.state.playing = True

    mock_streamer = MagicMock(return_value=mock_webrtc_ctx)
    mock_webrtc_module = MagicMock()
    mock_webrtc_module.webrtc_streamer = mock_streamer
    mock_webrtc_module.WebRtcMode = MagicMock()
    mock_webrtc_module.RTCConfiguration = MagicMock()

    # Unavailable pipeline session
    unavail_pipeline = MagicMock()
    unavail_pipeline.is_available.return_value = False
    unavail_session = WebcamRecognitionSession(recognition_pipeline=unavail_pipeline)

    mock_state = {"webcam_session": unavail_session}

    with patch.dict("sys.modules", {"streamlit_webrtc": mock_webrtc_module}), \
         patch("streamlit.session_state", mock_state), \
         patch("streamlit.warning") as mock_st_warning:
        render_mode_live_webcam({"recognition_pipeline_available": False})
        mock_st_warning.assert_called()
        warn_msg = mock_st_warning.call_args[0][0]
        assert "Recognition Pipeline Standby" in warn_msg

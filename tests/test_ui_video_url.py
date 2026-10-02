"""Unit tests for SIGNASSIST Phase 17 Block 4 Video URL UI Integration."""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.app import render_mode_video_url
from src.ui.state import init_session_state
from src.video.processor import VideoProcessingResult
from src.video.url_downloader import URLExtractionError


@dataclass
class DummyMetadata:
    native_fps: float = 30.0


@dataclass
class DummyQualityReport:
    overall_rating: str = "EXCELLENT"
    average_brightness: float = 120.0
    average_contrast: float = 55.0
    average_blur_score: float = 200.0


# 1. Test URL input and Process Video button exist
def test_video_url_ui_elements_exist() -> None:
    """Tests that text_input and button widgets are rendered in Video URL mode."""
    state = init_session_state({})
    with patch("streamlit.text_input", return_value="") as mock_input, \
         patch("streamlit.button", return_value=False) as mock_button, \
         patch("src.app.render_header"), \
         patch("src.app.render_empty_state"), \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        mock_input.assert_called_once()
        assert "Enter Video URL" in mock_input.call_args[0][0]
        mock_button.assert_called_once_with("Process Video", type="primary")


# 2. Test empty URL does not invoke process_video_url service
def test_video_url_empty_input_does_not_call_service() -> None:
    """Tests that an empty or whitespace-only URL input does not trigger process_video_url."""
    state = init_session_state({})
    with patch("streamlit.text_input", return_value="   "), \
         patch("streamlit.button", return_value=True), \
         patch("src.app.process_video_url") as mock_service, \
         patch("src.app.render_error_state") as mock_error, \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        mock_service.assert_not_called()
        mock_error.assert_called_once()
        assert "non-empty Video URL" in mock_error.call_args[0][0]


# 3. Test valid URL + button delegates to process_video_url and stores state
def test_video_url_valid_input_delegates_to_service_and_stores_state() -> None:
    """Tests that valid URL + button click delegates to process_video_url and stores result in session_state."""
    mock_result = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        execution_seconds=0.8,
        processing_fps=45.0,
        frames_processed=36,
        recognition_available=False,
        additional_metadata={"platform_name": "youtube"},
    )

    state = init_session_state({})
    with patch("streamlit.text_input", return_value="  https://youtube.com/watch?v=demo  "), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.caption"), \
         patch("src.app.process_video_url", return_value=mock_result) as mock_service, \
         patch("src.video.url_downloader.URLVideoDownloader.detect_platform", return_value="youtube"), \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        mock_service.assert_called_once()
        assert mock_service.call_args[1]["url"] == "https://youtube.com/watch?v=demo"
        assert state["latest_video_result"] is mock_result
        assert state["last_processed_url"] == "https://youtube.com/watch?v=demo"
        assert state["processing"] is False
        assert state["last_error"] is None


# 4. Test recognition unavailable is represented as standby
def test_video_url_recognition_unavailable_standby_warning() -> None:
    """Tests that recognition_available=False renders warning status."""
    mock_result = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        execution_seconds=0.5,
        processing_fps=60.0,
        frames_processed=30,
        recognition_available=False,
    )

    state = init_session_state({})
    state["latest_video_result"] = mock_result
    state["last_processed_url"] = "https://youtube.com/watch?v=demo"

    with patch("streamlit.text_input", return_value="https://youtube.com/watch?v=demo"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.caption"), \
         patch("streamlit.warning") as mock_warning, \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        mock_warning.assert_called()
        msg = mock_warning.call_args[0][0]
        assert "recognition backend is unavailable" in msg.lower() or "standby" in msg.lower()


# 5. Test URLExtractionError produces error state and updates last_error
def test_video_url_extraction_error_produces_error_state() -> None:
    """Tests that URLExtractionError updates last_error and renders error UI."""
    state = init_session_state({})
    err_msg = "Video is private on YOUTUBE."
    with patch("streamlit.text_input", return_value="https://youtube.com/watch?v=private"), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.caption"), \
         patch("src.app.render_error_state") as mock_render_error, \
         patch("src.app.process_video_url", side_effect=URLExtractionError(err_msg)), \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        assert state["processing"] is False
        assert state["last_error"] == err_msg
        assert state["latest_video_result"] is None
        mock_render_error.assert_called_once_with(err_msg)


# 6. Test generic processing exception produces error state
def test_video_url_generic_exception_produces_error_state() -> None:
    """Tests that a generic processing exception updates last_error and renders error UI."""
    state = init_session_state({})
    with patch("streamlit.text_input", return_value="https://example.com/video.mp4"), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.caption"), \
         patch("src.app.render_error_state") as mock_render_error, \
         patch("src.app.process_video_url", side_effect=RuntimeError("Network socket closed")), \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        assert state["processing"] is False
        assert "Network socket closed" in state["last_error"]
        assert state["latest_video_result"] is None
        mock_render_error.assert_called_once()
        assert "Network socket closed" in mock_render_error.call_args[0][0]


# 7. Test repeated Streamlit rerun without button click does not reprocess URL
def test_video_url_rerun_without_button_does_not_reprocess() -> None:
    """Tests that re-rendering without clicking Process Video reuses stored result without calling service."""
    mock_result = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        execution_seconds=0.8,
        processing_fps=45.0,
        frames_processed=36,
        recognition_available=False,
    )

    state = init_session_state({})
    state["latest_video_result"] = mock_result
    state["last_processed_url"] = "https://youtube.com/watch?v=demo"

    with patch("streamlit.text_input", return_value="https://youtube.com/watch?v=demo"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.caption"), \
         patch("src.app.process_video_url") as mock_service, \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        # Service MUST NOT be invoked on rerun without button click
        mock_service.assert_not_called()
        assert state["latest_video_result"] is mock_result


# 8. Test explicit second Process Video click reprocesses the URL
def test_video_url_explicit_second_button_click_reprocesses() -> None:
    """Tests that clicking Process Video again intentionally invokes process_video_url a second time."""
    mock_result_1 = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        frames_processed=30,
        recognition_available=False,
    )
    mock_result_2 = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        frames_processed=30,
        recognition_available=True,
    )

    state = init_session_state({})
    state["latest_video_result"] = mock_result_1
    state["last_processed_url"] = "https://youtube.com/watch?v=demo"

    with patch("streamlit.text_input", return_value="https://youtube.com/watch?v=demo"), \
         patch("streamlit.button", return_value=True), \
         patch("streamlit.caption"), \
         patch("src.app.process_video_url", return_value=mock_result_2) as mock_service, \
         patch("streamlit.session_state", state):

        render_mode_video_url({})

        # Service MUST be invoked again because button was clicked
        mock_service.assert_called_once()
        assert mock_service.call_args[1]["url"] == "https://youtube.com/watch?v=demo"
        assert state["latest_video_result"] is mock_result_2

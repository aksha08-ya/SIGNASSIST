"""Unit tests for SIGNASSIST Phase 17 Block 3 Upload Video UI Integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch
import pytest

from src.app import render_mode_upload_video
from src.ui.state import init_session_state
from src.video.processor import VideoProcessingResult


class DummyUploadedFile:
    """Mock Streamlit UploadedFile object for headless testing."""

    def __init__(self, name: str = "test_sign.mp4", content: bytes = b"fake_video_bytes"):
        self.name = name
        self.size = len(content)
        self._content = content

    def getvalue(self) -> bytes:
        return self._content

    def read(self) -> bytes:
        return self._content


@dataclass
class DummyMetadata:
    native_fps: float = 30.0


@dataclass
class DummyTranscript:
    text: str = "NAMASTE WELCOME"
    mean_confidence: float = 0.95


@dataclass
class DummyQualityReport:
    overall_rating: str = "EXCELLENT"
    average_brightness: float = 120.0
    average_contrast: float = 55.0
    average_blur_score: float = 200.0


# 1. Test supported video formats passed to st.file_uploader
def test_upload_video_supported_formats() -> None:
    """Tests that st.file_uploader is configured with all supported video extensions."""
    state = init_session_state({})
    with patch("streamlit.file_uploader") as mock_uploader, \
         patch("src.app.render_header"), \
         patch("src.app.render_empty_state"), \
         patch("streamlit.session_state", state):

        mock_uploader.return_value = None
        render_mode_upload_video({})

        mock_uploader.assert_called_once()
        _, kwargs = mock_uploader.call_args
        assert "type" in kwargs
        assert kwargs["type"] == ["mp4", "mov", "avi", "mkv", "webm"]


# 2. Test delegation to service layer and session_state storage
def test_upload_video_delegates_to_service_and_stores_state() -> None:
    """Tests that uploaded file is passed to process_video_file and result is stored in session_state."""
    dummy_file = DummyUploadedFile(name="sample.mp4", content=b"12345")
    mock_result = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        timeline=[],
        transcript=None,
        execution_seconds=1.2,
        processing_fps=30.0,
        frames_processed=36,
        recognition_available=False,
    )

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.video"), \
         patch("streamlit.caption"), \
         patch("src.app.process_video_file", return_value=mock_result) as mock_process, \
         patch("streamlit.session_state", state):

        render_mode_upload_video({})

        mock_process.assert_called_once()
        processed_path = mock_process.call_args[0][0]
        assert isinstance(processed_path, Path)

        # Verify result stored in session state
        assert state["latest_video_result"] is mock_result
        assert state["processing"] is False
        assert state["last_error"] is None


# 3. Test temporary file cleanup behavior
def test_upload_video_temp_file_cleanup() -> None:
    """Tests that temporary file created from upload is unlinked after processing completes."""
    dummy_file = DummyUploadedFile(name="temp_test.mp4", content=b"video_data")
    created_temp_paths = []

    def mock_process_side_effect(path: Path):
        created_temp_paths.append(Path(path))
        assert Path(path).exists()
        return VideoProcessingResult(
            video_metadata=DummyMetadata(),
            quality_report=DummyQualityReport(),
            execution_seconds=0.5,
            processing_fps=60.0,
            frames_processed=30,
            recognition_available=False,
        )

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.video"), \
         patch("streamlit.caption"), \
         patch("src.app.process_video_file", side_effect=mock_process_side_effect), \
         patch("streamlit.session_state", state):

        render_mode_upload_video({})

    assert len(created_temp_paths) == 1
    # Verify file was cleaned up after processing
    assert not created_temp_paths[0].exists()


# 4. Test recognition_available=False presented as backend unavailable
def test_upload_video_recognition_unavailable_warning() -> None:
    """Tests that recognition_available=False renders a clear warning about backend standby."""
    dummy_file = DummyUploadedFile()
    mock_result = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        execution_seconds=0.5,
        processing_fps=60.0,
        frames_processed=30,
        recognition_available=False,
    )

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.video"), \
         patch("streamlit.caption"), \
         patch("streamlit.warning") as mock_warning, \
         patch("src.app.process_video_file", return_value=mock_result), \
         patch("streamlit.session_state", state):

        render_mode_upload_video({})

        mock_warning.assert_called()
        warning_msg = mock_warning.call_args[0][0]
        assert "recognition backend is unavailable" in warning_msg.lower() or "standby" in warning_msg.lower()


# 5. Test genuine transcript displayed when present
def test_upload_video_transcript_displayed_when_present() -> None:
    """Tests that genuine transcript text is displayed when transcript is present in VideoProcessingResult."""
    dummy_file = DummyUploadedFile()
    mock_transcript = DummyTranscript(text="NAMASTE WELCOME")
    mock_result = VideoProcessingResult(
        video_metadata=DummyMetadata(),
        quality_report=DummyQualityReport(),
        transcript=mock_transcript,
        execution_seconds=1.0,
        processing_fps=30.0,
        frames_processed=30,
        recognition_available=True,
    )

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.video"), \
         patch("streamlit.caption"), \
         patch("streamlit.success") as mock_success, \
         patch("src.app.process_video_file", return_value=mock_result), \
         patch("streamlit.session_state", state):

        render_mode_upload_video({})

        assert state["transcript"] == "NAMASTE WELCOME"
        success_calls = [call[0][0] for call in mock_success.call_args_list]
        assert any("NAMASTE WELCOME" in msg for msg in success_calls)


# 6. Test exception in processing produces error state
def test_upload_video_exception_produces_error_state() -> None:
    """Tests that processing exceptions update last_error in session_state and render error UI."""
    dummy_file = DummyUploadedFile()

    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=dummy_file), \
         patch("streamlit.video"), \
         patch("streamlit.caption"), \
         patch("src.app.render_error_state") as mock_render_error, \
         patch("src.app.process_video_file", side_effect=ValueError("Corrupt frame header")), \
         patch("streamlit.session_state", state):

        render_mode_upload_video({})

        assert state["processing"] is False
        assert "Corrupt frame header" in state["last_error"]
        assert state["latest_video_result"] is None
        mock_render_error.assert_called_once()
        assert "Corrupt frame header" in mock_render_error.call_args[0][0]


# 7. Test empty state when no file is uploaded
def test_upload_video_no_file_renders_empty_state() -> None:
    """Tests that when no file is uploaded, render_empty_state is displayed."""
    state = init_session_state({})
    with patch("streamlit.file_uploader", return_value=None), \
         patch("src.app.render_empty_state") as mock_empty, \
         patch("streamlit.session_state", state):

        render_mode_upload_video({})

        mock_empty.assert_called_once()
        empty_title = mock_empty.call_args[1].get("title", mock_empty.call_args[0][0] if mock_empty.call_args[0] else "")
        assert "Video Upload Processing" in empty_title

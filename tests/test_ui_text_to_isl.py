"""Unit tests for SIGNASSIST Phase 17 Block 5 Text → ISL UI Integration."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch
import pytest

from src.app import render_mode_text_to_isl
from src.nlp.text_to_isl import DemonstrationSource, ISLDemonstration, ISLDemonstrationSequence
from src.ui.services import TextToISLResult
from src.ui.state import init_session_state


# 1. Test text_area and Convert button render
def test_text_to_isl_ui_elements_exist() -> None:
    """Tests that text_area and Convert to ISL button are rendered."""
    state = init_session_state({})
    with patch("streamlit.text_area", return_value="") as mock_text_area, \
         patch("streamlit.button", return_value=False) as mock_button, \
         patch("src.app.render_header"), \
         patch("src.app.render_empty_state"), \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        mock_text_area.assert_called_once()
        assert "Enter English Text" in mock_text_area.call_args[0][0]
        mock_button.assert_called_once_with("Convert to ISL", type="primary")


# 2. Test empty input shows validation error and does not call service
def test_text_to_isl_empty_input_shows_validation_error() -> None:
    """Tests that empty/whitespace input displays validation error and skips convert_text_to_isl."""
    state = init_session_state({})
    with patch("streamlit.text_area", return_value="   "), \
         patch("streamlit.button", return_value=True), \
         patch("src.app.convert_text_to_isl") as mock_service, \
         patch("src.app.render_error_state") as mock_render_error, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        mock_service.assert_not_called()
        mock_render_error.assert_called_once()
        assert "non-empty" in mock_render_error.call_args[0][0]


# 3. Test valid input delegates to convert_text_to_isl and stores result in session_state
def test_text_to_isl_valid_input_delegates_and_stores_result() -> None:
    """Tests that valid text delegates to convert_text_to_isl and stores result in session_state."""
    seq = ISLDemonstrationSequence(
        original_text="HELLO WORLD",
        normalized_text="hello world",
        demonstrations=[
            ISLDemonstration(
                token="HELLO",
                normalized_token="hello",
                source=DemonstrationSource.DICTIONARY,
                dictionary_gloss="HELLO",
            ),
            ISLDemonstration(
                token="WORLD",
                normalized_token="world",
                source=DemonstrationSource.FINGERSPELL,
                characters=["W", "O", "R", "L", "D"],
            ),
        ],
    )
    mock_result = TextToISLResult(
        sequence=seq,
        stitch_result=None,
        output_video_path=None,
        included_tokens=["HELLO", "WORLD"],
        skipped_tokens=[],
        dictionary_count=1,
        fingerspell_count=1,
        unknown_count=0,
    )

    state = init_session_state({})
    with patch("streamlit.text_area", return_value=" HELLO WORLD "), \
         patch("streamlit.button", return_value=True), \
         patch("src.app.convert_text_to_isl", return_value=mock_result) as mock_service, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        mock_service.assert_called_once()
        _, kwargs = mock_service.call_args
        assert kwargs["text"] == "HELLO WORLD"
        assert state["text_to_isl_result"] is mock_result
        assert state["text_to_isl_last_processed"] == "HELLO WORLD"
        assert state["processing"] is False


# 4. Test counts and demonstration sequence are displayed
def test_text_to_isl_displays_counts_and_sequence() -> None:
    """Tests that dictionary/fingerspell/unknown counts and demonstration sequence table are rendered."""
    seq = ISLDemonstrationSequence(
        original_text="HELLO INDIA",
        normalized_text="hello india",
        demonstrations=[
            ISLDemonstration(
                token="HELLO",
                normalized_token="hello",
                source=DemonstrationSource.DICTIONARY,
                dictionary_gloss="HELLO",
            ),
            ISLDemonstration(
                token="INDIA",
                normalized_token="india",
                source=DemonstrationSource.UNKNOWN,
            ),
        ],
    )
    mock_result = TextToISLResult(
        sequence=seq,
        stitch_result=None,
        output_video_path=None,
        included_tokens=["HELLO"],
        skipped_tokens=["INDIA"],
        dictionary_count=1,
        fingerspell_count=0,
        unknown_count=1,
    )

    state = init_session_state({})
    state["text_to_isl_result"] = mock_result
    state["text_to_isl_last_processed"] = "HELLO INDIA"

    with patch("streamlit.text_area", return_value="HELLO INDIA"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.metric") as mock_metric, \
         patch("streamlit.dataframe") as mock_dataframe, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        # Check metrics called for counts
        metric_labels = [call[0][0] for call in mock_metric.call_args_list]
        assert "Dictionary Signs" in metric_labels
        assert "Fingerspelled Words" in metric_labels
        assert "Unknown Words" in metric_labels

        # Check sequence dataframe called
        mock_dataframe.assert_called_once()
        table_data = mock_dataframe.call_args[0][0]
        assert len(table_data) == 2
        assert table_data[0]["Input Token"] == "HELLO"
        assert table_data[1]["Input Token"] == "INDIA"


# 5. Test real output_video_path displayed when Path.exists() is True
def test_text_to_isl_displays_video_when_path_exists(tmp_path: Path) -> None:
    """Tests that st.video is invoked if output_video_path is present and file exists on disk."""
    dummy_video = tmp_path / "stitched.mp4"
    dummy_video.write_bytes(b"dummy_video_bytes")

    seq = ISLDemonstrationSequence(original_text="TEST", normalized_text="test")
    mock_result = TextToISLResult(
        sequence=seq,
        stitch_result=None,
        output_video_path=str(dummy_video),
        included_tokens=["TEST"],
        skipped_tokens=[],
        dictionary_count=1,
        fingerspell_count=0,
        unknown_count=0,
    )

    state = init_session_state({})
    state["text_to_isl_result"] = mock_result
    state["text_to_isl_last_processed"] = "TEST"

    with patch("streamlit.text_area", return_value="TEST"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.video") as mock_video, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        mock_video.assert_called_once_with(str(dummy_video))


# 6. Test missing/nonexistent output path does NOT call st.video
def test_text_to_isl_does_not_call_video_when_path_missing() -> None:
    """Tests that st.video is NOT called if output_video_path is None or file does not exist."""
    seq = ISLDemonstrationSequence(original_text="TEST", normalized_text="test")
    mock_result = TextToISLResult(
        sequence=seq,
        stitch_result=None,
        output_video_path="/nonexistent/file.mp4",
        included_tokens=[],
        skipped_tokens=["TEST"],
        dictionary_count=0,
        fingerspell_count=0,
        unknown_count=1,
    )

    state = init_session_state({})
    state["text_to_isl_result"] = mock_result
    state["text_to_isl_last_processed"] = "TEST"

    with patch("streamlit.text_area", return_value="TEST"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.video") as mock_video, \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        mock_video.assert_not_called()
        mock_info.assert_called()
        info_msg = mock_info.call_args[0][0]
        assert "Textual ISL demonstration sequence generated" in info_msg


# 7. Test conversion exception sets error state and displays error UI
def test_text_to_isl_exception_sets_error_state() -> None:
    """Tests that conversion exceptions set last_error in session_state and display error UI."""
    state = init_session_state({})
    with patch("streamlit.text_area", return_value="CRASH TEXT"), \
         patch("streamlit.button", return_value=True), \
         patch("src.app.convert_text_to_isl", side_effect=ValueError("Dictionary file corrupted")), \
         patch("src.app.render_error_state") as mock_render_error, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        assert state["processing"] is False
        assert "Dictionary file corrupted" in state["last_error"]
        assert state["text_to_isl_result"] is None
        mock_render_error.assert_called_once()
        assert "Dictionary file corrupted" in mock_render_error.call_args[0][0]


# 8. Test rerunning without Convert button does not call convert_text_to_isl
def test_text_to_isl_rerun_without_button_does_not_call_service() -> None:
    """Tests that re-rendering without clicking Convert reuses cached result without calling service."""
    seq = ISLDemonstrationSequence(original_text="CACHED", normalized_text="cached")
    mock_result = TextToISLResult(
        sequence=seq,
        stitch_result=None,
        output_video_path=None,
        included_tokens=[],
        skipped_tokens=[],
        dictionary_count=0,
        fingerspell_count=0,
        unknown_count=0,
    )

    state = init_session_state({})
    state["text_to_isl_result"] = mock_result
    state["text_to_isl_last_processed"] = "CACHED"

    with patch("streamlit.text_area", return_value="CACHED"), \
         patch("streamlit.button", return_value=False), \
         patch("src.app.convert_text_to_isl") as mock_service, \
         patch("streamlit.session_state", state):

        render_mode_text_to_isl({})

        mock_service.assert_not_called()
        assert state["text_to_isl_result"] is mock_result

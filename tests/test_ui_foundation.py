"""Unit tests for SIGNASSIST Phase 17 Block 1 UI Foundation."""

from pathlib import Path
import pytest

from src.app import APPLICATION_MODES
from src.ui.components import check_model_availability
from src.ui.config import get_ui_config
from src.ui.state import INITIAL_STATE_DEFAULTS, init_session_state
from src.utils.config import AppConfig


def test_ui_config_loading() -> None:
    """Tests loading AppConfig via get_ui_config helper."""
    config = get_ui_config()
    assert isinstance(config, AppConfig)
    assert config.ui.app.title.startswith("SIGNASSIST")
    assert config.ui.app.layout == "wide"


def test_session_state_initialization_headlessly() -> None:
    """Tests session-state initialization headlessly using a standard dictionary."""
    state_dict = {}
    result_state = init_session_state(state_dict)

    assert result_state["current_mode"] == "Live Webcam"
    assert result_state["processing"] is False
    assert result_state["transcript"] == []
    assert result_state["prediction_history"] == []
    assert result_state["selected_signer_id"] is None
    assert result_state["latest_video_result"] is None
    assert result_state["last_error"] is None
    assert result_state["status_message"] == "Ready"


def test_session_state_idempotency() -> None:
    """Tests that init_session_state is idempotent and does not overwrite existing state keys."""
    custom_state = {
        "current_mode": "Text → ISL",
        "processing": True,
        "transcript": ["Hello world"],
        "status_message": "Processing...",
    }

    result = init_session_state(custom_state)

    # Pre-existing values must be preserved
    assert result["current_mode"] == "Text → ISL"
    assert result["processing"] is True
    assert result["transcript"] == ["Hello world"]
    assert result["status_message"] == "Processing..."

    # Missing default keys must be initialized
    assert result["prediction_history"] == []
    assert result["selected_signer_id"] is None


def test_model_availability_check_no_checkpoint(tmp_path: Path) -> None:
    """Tests model availability when no checkpoint exists."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    info = check_model_availability(models_dir)

    assert info["architecture_available"] is True
    assert info["checkpoint_available"] is False
    assert info["recognition_pipeline_available"] is False
    assert info["checkpoint_count"] == 0


def test_model_availability_checkpoint_present_no_adapter(tmp_path: Path) -> None:
    """Tests model availability when checkpoint exists but no recognition adapter/pipeline is provided."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dummy_checkpoint = models_dir / "stgcn_dynamic_isl.pt"
    dummy_checkpoint.write_bytes(b"dummy_checkpoint_data")

    info = check_model_availability(models_dir)

    assert info["architecture_available"] is True
    assert info["checkpoint_available"] is True
    # Checkpoint present does NOT equate to recognition pipeline availability
    assert info["recognition_pipeline_available"] is False
    assert info["checkpoint_count"] == 1


def test_model_availability_explicit_pipeline_available(tmp_path: Path) -> None:
    """Tests model availability when an explicit pipeline/adapter is supplied."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    info = check_model_availability(models_dir, explicit_pipeline_available=True)

    assert info["architecture_available"] is True
    assert info["checkpoint_available"] is False
    assert info["recognition_pipeline_available"] is True


def test_five_application_modes_defined_exactly_once() -> None:
    """Tests that exactly five application mode names are defined without duplicates."""
    assert len(APPLICATION_MODES) == 5
    assert len(set(APPLICATION_MODES)) == 5

    expected_modes = [
        "Live Webcam",
        "Continuous Sentence",
        "Upload Video",
        "Video URL",
        "Text → ISL",
    ]
    assert APPLICATION_MODES == expected_modes

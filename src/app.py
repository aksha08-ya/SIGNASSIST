"""SIGNASSIST - Real-Time Indian Sign Language Recognition & Translation UI Application."""

import os
import sys
from pathlib import Path
from typing import List
import streamlit as st

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui import (
    check_model_availability,
    get_ui_config,
    init_session_state,
    render_empty_state,
    render_error_state,
    render_header,
    render_model_status,
)

APPLICATION_MODES: List[str] = [
    "Live Webcam",
    "Continuous Sentence",
    "Upload Video",
    "Video URL",
    "Text → ISL",
    "Research Analytics",
]


def render_mode_live_webcam(status_info: dict) -> None:
    """Renders Live Webcam mode placeholder."""
    render_header(
        title="Live Webcam Recognition",
        page_icon="📹",
        description="Real-time sign language recognition and translation from your camera feed.",
    )
    if status_info.get("recognition_pipeline_available"):
        st.info("Live webcam inference pipeline is ready.")
    elif status_info.get("checkpoint_available"):
        count = status_info.get("checkpoint_count", 0)
        render_empty_state(
            title="Recognition Adapter Standby",
            message=(
                f"Trained checkpoint file(s) found ({count}), but the live recognition adapter pipeline is not yet wired or verified.\n\n"
                "**Status Details**:\n"
                "- Model Architecture: ✅ Available (ST-GCN / GNN)\n"
                "- Trained Checkpoints: ✅ Found\n"
                "- Recognition Pipeline Adapter: ⚠️ Standby (Unwired)\n\n"
                "Live inference cannot start until the live recognition adapter pipeline is initialized and connected."
            ),
            icon="📹",
        )
    else:
        render_empty_state(
            title="Recognition Backend Standby",
            message=(
                "Model architectures are available, but no trained checkpoints were found under the models directory.\n\n"
                "**Status Details**:\n"
                "- Model Architecture: ✅ Available (ST-GCN / GNN)\n"
                "- Trained Checkpoints: ⚠️ None Found\n"
                "- Recognition Pipeline: ⚠️ Standby\n\n"
                "Live stream inference requires trained checkpoint files and an initialized recognition pipeline adapter."
            ),
            icon="📹",
        )


def render_mode_continuous_sentence(status_info: dict) -> None:
    """Renders Continuous Sentence mode placeholder."""
    render_header(
        title="Continuous Sentence Recognition",
        page_icon="💬",
        description="Translate continuous ISL gesture sequences into structured sentences.",
    )
    render_empty_state(
        title="Continuous Sentence Mode Standby",
        message=(
            "Translates continuous sign sequences into natural text using boundary detection and NLP context smoothing.\n\n"
            "**Backend Status**:\n"
            f"- Architecture: {'✅ Available' if status_info.get('architecture_available') else '❌ Unavailable'}\n"
            f"- Checkpoint: {'✅ Available' if status_info.get('checkpoint_available') else '⚠️ None Found'}"
        ),
        icon="💬",
    )


def render_mode_upload_video(status_info: dict) -> None:
    """Renders Upload Video mode placeholder."""
    render_header(
        title="Upload Video File",
        page_icon="📁",
        description="Upload MP4 or AVI video files for sign language recognition and translation.",
    )
    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.caption(f"Loaded video file: {uploaded_file.name}")
    render_empty_state(
        title="Video Upload Processing",
        message="Upload a video file above to process frames through the landmark extraction and recognition pipeline.",
        icon="📁",
    )


def render_mode_video_url(status_info: dict) -> None:
    """Renders Video URL mode placeholder."""
    render_header(
        title="Video URL Processing",
        page_icon="🔗",
        description="Process external public video URLs (e.g. YouTube or HTTP video streams) for ISL recognition.",
    )
    url_input = st.text_input(
        "Enter Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    if url_input:
        st.caption(f"Target URL: {url_input}")
    render_empty_state(
        title="External Video URL Processing",
        message="Enter a valid video URL above to fetch and process video frames through the SIGNASSIST pipeline.",
        icon="🔗",
    )


def render_mode_text_to_isl(status_info: dict) -> None:
    """Renders Text → ISL mode placeholder."""
    render_header(
        title="Text to ISL Demonstration Engine",
        page_icon="🤟",
        description="Convert input text sentences into Indian Sign Language video demonstrations.",
    )
    text_input = st.text_input(
        "Enter Text to Translate into ISL",
        placeholder="e.g. Hello welcome to India",
    )
    if text_input:
        st.caption(f"Input Text: {text_input}")
    render_empty_state(
        title="Text-to-ISL Generator",
        message=(
            "Converts input text into ISL gloss sequence, dictionary lookup clips, "
            "and fingerspelling fallbacks using OpenCV Video Stitcher."
        ),
        icon="🤟",
    )


def render_mode_research_analytics(status_info: dict) -> None:
    """Renders Research Analytics mode placeholder."""
    render_header(
        title="Research Analytics & Benchmarks",
        page_icon="📊",
        description="Evaluate model metrics, confusion matrices, LOSO cross-validation, and ablation studies.",
    )
    checkpoint_count = status_info.get("checkpoint_count", 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Architecture", "ST-GCN + GNN")
    with col2:
        st.metric("Checkpoints Available", str(checkpoint_count))
    with col3:
        st.metric("Benchmark Status", "Ready for Evaluation")

    render_empty_state(
        title="Research Evaluation Framework",
        message=(
            "Provides model-agnostic MetricsEvaluator, LOSO evaluation engine, and Ablation Study runner.\n\n"
            "Run `python scripts/evaluate_loso.py --demo` or `python scripts/run_ablation.py --demo` from the terminal to view benchmark evaluations."
        ),
        icon="📊",
    )


def main() -> None:
    """Main application entry point."""
    # Load application configuration with error handling
    try:
        app_config = get_ui_config()
    except Exception as err:
        st.set_page_config(page_title="SIGNASSIST - Error", page_icon="⚠️", layout="wide")
        render_error_state(f"Failed to load application configuration: {err}")
        return

    # Configure Streamlit page from config
    st.set_page_config(
        page_title=app_config.ui.app.title,
        page_icon=app_config.ui.app.page_icon,
        layout=app_config.ui.app.layout,
    )

    # Centralized session state initialization
    init_session_state()

    # Sidebar mode selection
    st.sidebar.title("Navigation")
    selected_mode = st.sidebar.selectbox(
        "Application Mode",
        options=APPLICATION_MODES,
        index=APPLICATION_MODES.index(st.session_state.get("current_mode", APPLICATION_MODES[0])),
    )
    st.session_state["current_mode"] = selected_mode

    # Model and checkpoint availability check
    models_dir = getattr(app_config.paths, "models_dir", "models")
    status_info = check_model_availability(models_dir)
    render_model_status(status_info)

    # Render selected mode page
    if selected_mode == "Live Webcam":
        render_mode_live_webcam(status_info)
    elif selected_mode == "Continuous Sentence":
        render_mode_continuous_sentence(status_info)
    elif selected_mode == "Upload Video":
        render_mode_upload_video(status_info)
    elif selected_mode == "Video URL":
        render_mode_video_url(status_info)
    elif selected_mode == "Text → ISL":
        render_mode_text_to_isl(status_info)
    elif selected_mode == "Research Analytics":
        render_mode_research_analytics(status_info)
    else:
        render_error_state(f"Unknown mode: {selected_mode}")


if __name__ == "__main__":
    main()

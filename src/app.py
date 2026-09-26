"""SIGNASSIST - Real-Time Indian Sign Language Recognition & Translation UI Application."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional
import streamlit as st

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui import (
    TextToISLResult,
    check_model_availability,
    convert_text_to_isl,
    get_ui_config,
    init_session_state,
    process_video_file,
    process_video_url,
    render_empty_state,
    render_error_state,
    render_header,
    render_model_status,
)
from src.video.url_downloader import URLExtractionError, URLVideoDownloader

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


def render_video_processing_result(result: Any) -> None:
    """Displays VideoProcessingResult fields in a structured Streamlit summary layout."""
    st.markdown("### 📊 Processing Results")

    # 1. Recognition status banner
    rec_avail = getattr(result, "recognition_available", False)
    if rec_avail:
        st.success("Recognition backend active.")
    else:
        st.warning(
            "Video processing completed, but the recognition backend is unavailable (standby / no model checkpoint)."
        )

    # 2. Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames Processed", getattr(result, "frames_processed", 0))
    with col2:
        fps = getattr(result, "processing_fps", 0.0)
        st.metric("Processing FPS", f"{fps:.1f}")
    with col3:
        exec_sec = getattr(result, "execution_seconds", 0.0)
        st.metric("Execution Time", f"{exec_sec:.2f} s")
    with col4:
        st.metric("Recognition Status", "Active" if rec_avail else "Standby")

    # 3. Quality Report if present
    quality = getattr(result, "quality_report", None)
    if quality is not None:
        st.markdown("#### 🎥 Video Quality Assessment")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.metric("Overall Rating", getattr(quality, "overall_rating", "N/A"))
        with q_col2:
            brightness = getattr(quality, "average_brightness", 0.0)
            st.metric("Brightness", f"{brightness:.1f}")
        with q_col3:
            contrast = getattr(quality, "average_contrast", 0.0)
            st.metric("Contrast", f"{contrast:.1f}")
        with q_col4:
            blur = getattr(quality, "average_blur_score", 0.0)
            st.metric("Sharpness (Blur)", f"{blur:.1f}")

    # 4. Signer / Tracking Information if present
    meta = getattr(result, "additional_metadata", {}) or {}
    manual_signer = meta.get("manual_signer_id")
    if manual_signer is not None:
        st.info(f"**Signer Tracking**: Manual Signer ID `{manual_signer}` assigned.")

    # 5. Transcript if genuinely present
    transcript = getattr(result, "transcript", None)
    if transcript is not None and getattr(transcript, "text", "").strip():
        st.markdown("#### 📝 Sentence Transcript")
        st.success(f"**Transcript**: {transcript.text}")
        if hasattr(transcript, "mean_confidence") and transcript.mean_confidence > 0:
            st.caption(f"Mean Confidence: {transcript.mean_confidence:.2%}")

    # 6. Timeline entries if present
    timeline = getattr(result, "timeline", []) or []
    if timeline:
        st.markdown("#### ⏱️ Sign Recognition Timeline")
        timeline_data = []
        for entry in timeline:
            timeline_data.append(
                {
                    "Start Frame": getattr(entry, "start_frame", 0),
                    "End Frame": getattr(entry, "end_frame", 0),
                    "Timestamp": f"{getattr(entry, 'start_timestamp', '')} - {getattr(entry, 'end_timestamp', '')}",
                    "Gloss": getattr(entry, "recognized_gloss", ""),
                    "Confidence": f"{getattr(entry, 'confidence', 0.0):.2%}",
                    "Signer Track ID": getattr(entry, "primary_signer_track_id", "N/A"),
                }
            )
        st.dataframe(timeline_data, use_container_width=True)


def _handle_video_upload(uploaded_file: Any) -> None:
    """Saves uploaded video file to a temporary file, processes it, updates session state, and cleans up."""
    file_name = getattr(uploaded_file, "name", "uploaded_video.mp4")
    suffix = Path(file_name).suffix or ".mp4"

    st.session_state["processing"] = True
    st.session_state["status_message"] = f"Processing {file_name}..."
    st.session_state["last_error"] = None

    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            if hasattr(uploaded_file, "getvalue"):
                content = uploaded_file.getvalue()
            elif hasattr(uploaded_file, "read"):
                content = uploaded_file.read()
                if hasattr(uploaded_file, "seek"):
                    uploaded_file.seek(0)
            else:
                content = bytes(uploaded_file)
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        result = process_video_file(temp_path)

        st.session_state["latest_video_result"] = result
        st.session_state["processing"] = False
        st.session_state["status_message"] = "Video processing completed."
        st.session_state["uploaded_file_id"] = f"{file_name}_{getattr(uploaded_file, 'size', 0)}"
        if getattr(result, "transcript", None):
            st.session_state["transcript"] = result.transcript.text
    except Exception as err:
        st.session_state["processing"] = False
        st.session_state["last_error"] = str(err)
        st.session_state["status_message"] = f"Error: {err}"
        st.session_state["latest_video_result"] = None
        render_error_state(f"Failed to process video file: {err}")
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def render_mode_upload_video(status_info: dict) -> None:
    """Renders Upload Video mode."""
    render_header(
        title="Upload Video File",
        page_icon="📁",
        description="Upload MP4 or AVI video files for sign language recognition and translation.",
    )
    uploaded_file = st.file_uploader(
        "Choose a video file", type=["mp4", "mov", "avi", "mkv", "webm"]
    )
    if uploaded_file is not None:
        st.video(uploaded_file)
        st.caption(f"Loaded video file: {uploaded_file.name}")

        file_id = f"{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
        if st.session_state.get("uploaded_file_id") != file_id:
            _handle_video_upload(uploaded_file)

        result = st.session_state.get("latest_video_result")
        if result is not None:
            render_video_processing_result(result)
    else:
        render_empty_state(
            title="Video Upload Processing",
            message="Upload a video file above to process frames through the landmark extraction and recognition pipeline.",
            icon="📁",
        )


def _handle_video_url_processing(url: str) -> None:
    """Invokes process_video_url service, handles errors, and updates session state."""
    st.session_state["processing"] = True
    st.session_state["status_message"] = "Downloading and processing video from URL..."
    st.session_state["last_error"] = None

    try:
        result = process_video_url(url=url)
        st.session_state["latest_video_result"] = result
        st.session_state["last_processed_url"] = url
        st.session_state["processing"] = False
        st.session_state["status_message"] = "URL video processing completed."
        if getattr(result, "transcript", None):
            st.session_state["transcript"] = result.transcript.text
    except URLExtractionError as err:
        st.session_state["processing"] = False
        st.session_state["last_error"] = str(err)
        st.session_state["status_message"] = "URL video processing failed."
        st.session_state["latest_video_result"] = None
        render_error_state(str(err))
    except Exception as err:
        st.session_state["processing"] = False
        st.session_state["last_error"] = str(err)
        st.session_state["status_message"] = "URL video processing failed."
        st.session_state["latest_video_result"] = None
        render_error_state(f"Failed to process video URL: {err}")


def render_mode_video_url(status_info: dict) -> None:
    """Renders Video URL mode."""
    render_header(
        title="Video URL Processing",
        page_icon="🔗",
        description="Process external public video URLs (e.g. YouTube or HTTP video streams) for ISL recognition.",
    )
    url_input = st.text_input(
        "Enter Video URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    clean_url = url_input.strip() if url_input else ""

    if clean_url:
        try:
            downloader = URLVideoDownloader()
            platform_name = downloader.detect_platform(clean_url)
        except Exception:
            platform_name = "unknown"

        st.caption(f"Target URL: `{clean_url}` | Detected Platform: `{platform_name.upper()}`")

    process_button = st.button("Process Video", type="primary")

    if process_button:
        if not clean_url:
            st.session_state["last_error"] = "Please enter a valid non-empty Video URL."
            render_error_state("Please enter a valid non-empty Video URL.")
        else:
            _handle_video_url_processing(clean_url)

    result = st.session_state.get("latest_video_result")
    last_processed_url = st.session_state.get("last_processed_url")

    if result is not None and last_processed_url:
        meta = getattr(result, "additional_metadata", {}) or {}
        retrieved_path = meta.get("retrieved_file_path") or getattr(
            getattr(result, "video_metadata", None), "file_path", None
        )
        if retrieved_path and Path(retrieved_path).exists():
            st.video(retrieved_path)

        render_video_processing_result(result)
    elif not clean_url and not st.session_state.get("last_error"):
        render_empty_state(
            title="External Video URL Processing",
            message="Enter a valid video URL above to fetch and process video frames through the SIGNASSIST pipeline.",
            icon="🔗",
        )


def render_text_to_isl_result(result: TextToISLResult) -> None:
    """Renders TextToISLResult fields, metrics, tokens, sequence, and optional video."""
    st.markdown("### 🤟 ISL Translation Summary")
    st.caption(f"Original Input: {result.sequence.original_text}")

    # 1. Token Counts Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dictionary Signs", result.dictionary_count)
    with col2:
        st.metric("Fingerspelled Words", result.fingerspell_count)
    with col3:
        st.metric("Unknown Words", result.unknown_count)

    # 2. Included & Skipped Tokens
    if result.included_tokens:
        st.markdown(f"**Included Tokens**: `{', '.join(result.included_tokens)}`")
    if result.skipped_tokens:
        st.warning(f"**Skipped / Unknown Tokens**: `{', '.join(result.skipped_tokens)}`")

    # 3. Demonstration Sequence in Token Order
    demos = getattr(result.sequence, "demonstrations", [])
    if demos:
        st.markdown("#### 📋 Demonstration Sequence")
        seq_data = []
        for idx, d in enumerate(demos, 1):
            source_str = d.source.value if hasattr(d.source, "value") else str(d.source)
            detail = d.dictionary_gloss if d.dictionary_gloss else (", ".join(d.characters) if d.characters else "N/A")
            seq_data.append(
                {
                    "Step": idx,
                    "Input Token": d.token,
                    "Normalized Token": d.normalized_token,
                    "Source": source_str.upper(),
                    "Detail / Gloss": detail,
                }
            )
        st.dataframe(seq_data, use_container_width=True)

    # 4. Video Stitched Output if valid file path exists
    if result.output_video_path and Path(result.output_video_path).exists():
        st.markdown("#### 🎬 Stitched ISL Demonstration Video")
        st.video(result.output_video_path)
    else:
        st.info(
            "Textual ISL demonstration sequence generated. "
            "No stitched video clip could be produced because video assets for the dictionary entries are unavailable."
        )


def _handle_text_to_isl_conversion(text: str) -> None:
    """Invokes convert_text_to_isl service and updates session state."""
    st.session_state["processing"] = True
    st.session_state["status_message"] = "Converting text to ISL sequence..."
    st.session_state["last_error"] = None

    out_dir = Path(tempfile.gettempdir()) / "signassist_isl_stitch"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / "stitched_output.mp4"

    try:
        result = convert_text_to_isl(
            text=text,
            stitch_video=True,
            output_video_path=out_video_path,
        )
        st.session_state["text_to_isl_result"] = result
        st.session_state["text_to_isl_last_processed"] = text
        st.session_state["processing"] = False
        st.session_state["status_message"] = "Text to ISL conversion complete."
    except Exception as err:
        st.session_state["processing"] = False
        st.session_state["last_error"] = str(err)
        st.session_state["status_message"] = "Text to ISL conversion failed."
        st.session_state["text_to_isl_result"] = None
        render_error_state(f"Failed to convert text to ISL: {err}")


def render_mode_text_to_isl(status_info: dict) -> None:
    """Renders Text → ISL mode."""
    render_header(
        title="Text to ISL Demonstration Engine",
        page_icon="🤟",
        description="Convert input text sentences into Indian Sign Language video demonstrations.",
    )
    input_text = st.text_area(
        "Enter English Text to Translate into ISL",
        placeholder="e.g. Hello welcome to India",
        key="text_to_isl_input",
    )

    clean_text = input_text.strip() if input_text else ""

    convert_button = st.button("Convert to ISL", type="primary")

    if convert_button:
        if not clean_text:
            st.session_state["last_error"] = "Please enter non-empty English text to translate."
            render_error_state("Please enter non-empty English text to translate.")
        else:
            _handle_text_to_isl_conversion(clean_text)

    result: Optional[TextToISLResult] = st.session_state.get("text_to_isl_result")
    last_processed = st.session_state.get("text_to_isl_last_processed")

    if result is not None and last_processed:
        render_text_to_isl_result(result)
    elif not clean_text and not st.session_state.get("last_error"):
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

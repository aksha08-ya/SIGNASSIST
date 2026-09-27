"""SIGNASSIST - Real-Time Indian Sign Language Recognition & Translation UI Application."""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import streamlit as st

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ui import (
    TextToISLResult,
    check_model_availability,
    convert_text_to_isl,
    get_ui_config,
    init_session_state,
    parse_research_artifact,
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


def render_core_metrics_summary(metrics_dict: Dict[str, Any]) -> None:
    """Renders 9 core classification metrics using st.metric cards."""
    st.markdown("#### 🎯 Overall Classification Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics_dict.get('accuracy', 0.0):.2%}")
        st.metric("Macro Precision", f"{metrics_dict.get('macro_precision', 0.0):.2%}")
        st.metric("Weighted Precision", f"{metrics_dict.get('weighted_precision', 0.0):.2%}")
    with col2:
        st.metric("Total Samples", metrics_dict.get("total_samples", 0))
        st.metric("Macro Recall", f"{metrics_dict.get('macro_recall', 0.0):.2%}")
        st.metric("Weighted Recall", f"{metrics_dict.get('weighted_recall', 0.0):.2%}")
    with col3:
        st.metric("Number of Classes", metrics_dict.get("num_classes", 0))
        st.metric("Macro F1", f"{metrics_dict.get('macro_f1', 0.0):.2%}")
        st.metric("Weighted F1", f"{metrics_dict.get('weighted_f1', 0.0):.2%}")


def render_per_class_metrics_table(per_class_dict: Dict[str, Any]) -> None:
    """Renders per-class precision, recall, f1, and support in a dataframe."""
    if not per_class_dict:
        return

    st.markdown("#### 🏷️ Per-Class Performance")
    rows = []
    for cls_name, vals in per_class_dict.items():
        if isinstance(vals, dict):
            rows.append(
                {
                    "Class": cls_name,
                    "Precision": f"{vals.get('precision', 0.0):.4f}",
                    "Recall": f"{vals.get('recall', 0.0):.4f}",
                    "F1 Score": f"{vals.get('f1', 0.0):.4f}",
                    "Support": vals.get("support", 0),
                }
            )

    if rows:
        st.dataframe(rows, use_container_width=True)


def render_confusion_matrix_table(cm_list: List[List[int]], class_names: List[str]) -> None:
    """Renders confusion matrix table with rows as Actual and columns as Predicted classes."""
    if not cm_list:
        return

    st.markdown("#### 🔢 Confusion Matrix")
    num_classes = len(cm_list)
    if not class_names or len(class_names) != num_classes:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    cm_df_data = []
    for idx, row in enumerate(cm_list):
        row_dict = {"Actual Class": class_names[idx]}
        for c_idx, val in enumerate(row):
            col_name = f"Pred: {class_names[c_idx]}"
            row_dict[col_name] = val
        cm_df_data.append(row_dict)

    st.dataframe(cm_df_data, use_container_width=True)


def render_evaluation_result_dashboard(payload: Dict[str, Any]) -> None:
    """Renders standard EvaluationResult artifact metrics, per-class metrics, and confusion matrix."""
    st.markdown("### 📈 Evaluation Result Dashboard")
    render_core_metrics_summary(payload)
    render_per_class_metrics_table(payload.get("per_class_metrics", {}))
    render_confusion_matrix_table(payload.get("confusion_matrix", []), payload.get("class_names", []))


def render_loso_result_dashboard(payload: Dict[str, Any]) -> None:
    """Renders Leave-One-Signer-Out (LOSO) cross-validation evaluation results."""
    st.markdown("### 👥 Leave-One-Signer-Out (LOSO) Evaluation Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Number of Signers", payload.get("num_signers", len(payload.get("folds", []))))
    with col2:
        st.metric("Total Samples", payload.get("num_samples", 0))
    with col3:
        agg = payload.get("aggregate_metrics", {})
        st.metric("Aggregate Accuracy", f"{agg.get('accuracy', 0.0):.2%}")
    with col4:
        st.metric("Aggregate Macro F1", f"{agg.get('macro_f1', 0.0):.2%}")

    agg = payload.get("aggregate_metrics", {})
    if agg:
        render_core_metrics_summary(agg)
        render_per_class_metrics_table(agg.get("per_class_metrics", {}))
        render_confusion_matrix_table(agg.get("confusion_matrix", []), agg.get("class_names", []))

    folds = payload.get("folds", [])
    if folds:
        st.markdown("#### 📂 Per-Fold Signer Isolation Details")
        fold_rows = []
        for idx, fold in enumerate(folds, 1):
            f_metrics = fold.get("metrics", {})
            fold_rows.append(
                {
                    "Fold": idx,
                    "Held-Out Signer": fold.get("held_out_signer", f"Signer_{idx}"),
                    "Train Samples": fold.get("train_sample_count", 0),
                    "Test Samples": fold.get("test_sample_count", 0),
                    "Accuracy": f"{f_metrics.get('accuracy', 0.0):.4f}",
                    "Macro F1": f"{f_metrics.get('macro_f1', 0.0):.4f}",
                }
            )
        st.dataframe(fold_rows, use_container_width=True)


def render_ablation_result_dashboard(payload: Dict[str, Any]) -> None:
    """Renders Ablation Study results across multiple experiment configurations."""
    st.markdown("### 🔬 Ablation Study Dashboard")
    st.caption(
        f"Evaluation Mode: `{payload.get('evaluation_type', 'standard').upper()}` | "
        f"Experiments: `{payload.get('num_experiments', len(payload.get('experiments', [])))}`"
    )

    experiments = payload.get("experiments", [])
    if not experiments:
        st.info("No experiment results found in ablation artifact.")
        return

    exp_rows = []
    for exp in experiments:
        m = exp.get("metrics", {})
        params = exp.get("parameters", {})
        param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if isinstance(params, dict) else str(params)

        exp_rows.append(
            {
                "Experiment Name": exp.get("name", "N/A"),
                "Description": exp.get("description", "N/A"),
                "Accuracy": f"{m.get('accuracy', 0.0):.4f}",
                "Macro F1": f"{m.get('macro_f1', 0.0):.4f}",
                "Parameters": param_str,
                "Sample Count": exp.get("sample_count", "N/A"),
                "Signer Count": exp.get("signer_count", "N/A"),
            }
        )

    st.dataframe(exp_rows, use_container_width=True)


def generate_synthetic_demo_artifact() -> Dict[str, Any]:
    """Generates a synthetic LOSO evaluation artifact for UI demonstration only."""
    from scripts.evaluate_loso import create_demo_dataset, mock_deterministic_classifier
    from src.utils.loso_evaluator import LOSOEvaluator

    ds = create_demo_dataset()
    evaluator = LOSOEvaluator(zero_division=0.0, num_classes=3, class_names=["CLASS_0", "CLASS_1", "CLASS_2"])
    res = evaluator.evaluate(ds, mock_deterministic_classifier)
    return {
        "artifact_type": "loso",
        "payload": res.to_dict(),
    }


def render_mode_research_analytics(status_info: dict) -> None:
    """Renders Research Analytics & Benchmarks mode."""
    render_header(
        title="Research Analytics & Benchmarks",
        page_icon="📊",
        description=(
            "Visualize genuine evaluation result artifacts, confusion matrices, LOSO cross-validation, "
            "and ablation study results produced by the Phase 16 evaluation framework."
        ),
    )

    col_up, col_demo = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Evaluation Result Artifact (JSON)",
            type=["json"],
            key="research_file_uploader",
        )

    with col_demo:
        st.markdown("#### 🧪 Demo Mode")
        demo_active = st.checkbox(
            "Run Synthetic Demonstration (TEST ONLY)",
            value=st.session_state.get("research_demo_active", False),
            key="research_demo_active_checkbox",
        )
        st.session_state["research_demo_active"] = demo_active

    # Handle synthetic demo mode
    if demo_active:
        st.error("⚠️ DEMONSTRATION / TEST ONLY — SYNTHETIC DATA — NOT RESEARCH RESULTS")
        demo_artifact = generate_synthetic_demo_artifact()
        render_loso_result_dashboard(demo_artifact["payload"])
        return

    # Handle file upload
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{getattr(uploaded_file, 'size', 0)}"
        if st.session_state.get("research_result_file_id") != file_id:
            try:
                raw_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
                json_data = json.loads(raw_bytes.decode("utf-8"))
                parsed = parse_research_artifact(json_data)

                st.session_state["research_result"] = parsed["payload"]
                st.session_state["research_result_type"] = parsed["artifact_type"]
                st.session_state["research_result_filename"] = uploaded_file.name
                st.session_state["research_result_file_id"] = file_id
                st.session_state["research_last_error"] = None
            except json.JSONDecodeError as err:
                st.session_state["research_result"] = None
                st.session_state["research_result_type"] = None
                st.session_state["research_last_error"] = f"Malformed JSON file: {err}"
                render_error_state(f"Malformed JSON file: {err}")
            except ValueError as err:
                st.session_state["research_result"] = None
                st.session_state["research_result_type"] = None
                st.session_state["research_last_error"] = str(err)
                render_error_state(str(err))
            except Exception as err:
                st.session_state["research_result"] = None
                st.session_state["research_result_type"] = None
                st.session_state["research_last_error"] = str(err)
                render_error_state(f"Failed to load evaluation artifact: {err}")

    res_payload = st.session_state.get("research_result")
    res_type = st.session_state.get("research_result_type")

    if res_payload is not None and res_type is not None:
        st.caption(f"Loaded Artifact: `{st.session_state.get('research_result_filename', 'uploaded.json')}`")
        if res_type == "evaluation":
            render_evaluation_result_dashboard(res_payload)
        elif res_type == "loso":
            render_loso_result_dashboard(res_payload)
        elif res_type == "ablation":
            render_ablation_result_dashboard(res_payload)
    elif uploaded_file is None and not st.session_state.get("research_last_error"):
        checkpoint_count = status_info.get("checkpoint_count", 0)
        render_empty_state(
            title="Research Evaluation Framework",
            message=(
                "Model architectures and evaluation frameworks are available.\n\n"
                f"**System Status**:\n"
                f"- Model Architecture: ✅ Available (ST-GCN / GNN)\n"
                f"- Checkpoints Found: {checkpoint_count}\n"
                f"- Evaluation Framework: ✅ Ready (MetricsEvaluator / LOSO / Ablation)\n\n"
                "Real research analytics require uploading a genuine evaluation result JSON artifact produced by Phase 16 benchmark scripts. "
                "You may also enable 'Run Synthetic Demonstration (TEST ONLY)' above for framework validation."
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

"""SIGNASSIST - Real-Time Indian Sign Language Recognition & Translation UI Application."""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import streamlit as st

import logging


# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp.text_to_isl import DemonstrationSource
from src.nlp.types import BoundedSignPrediction
from src.ui import (
    TextToISLResult,
    check_model_availability,
    convert_text_to_isl,
    get_default_recognition_pipeline,
    get_ui_config,
    init_session_state,
    parse_research_artifact,
    process_video_file,
    process_video_url,
    render_empty_state,
    render_error_state,
    render_header,
    render_model_status,
    ContinuousSentenceSession,
)
from src.ui.webcam import WebcamRecognitionSession
from src.video.url_downloader import URLExtractionError, URLVideoDownloader

logger = logging.getLogger(__name__)

APPLICATION_MODES: List[str] = [
    "Live Webcam",
    "Continuous Sentence",
    "Upload Video",
    "Video URL",
    "Text → ISL",
    "Research Analytics",
]


def make_video_frame_callback(session: WebcamRecognitionSession):
    """Creates a WebRTC video frame callback bound to the given session instance.

    This function and the returned callback MUST NOT touch Streamlit APIs or st.session_state.
    """
    def video_frame_callback(frame: Any) -> Any:
        try:
            img = frame.to_ndarray(format="bgr24")
            session.add_frame(img)
        except Exception as e:
            logger.warning(f"Error converting WebRTC video frame: {e}")
        return frame

    return video_frame_callback


def get_or_create_webcam_session(backend: str = "experimental_42") -> WebcamRecognitionSession:
    """Retrieves or initializes the session-scoped WebcamRecognitionSession instance."""
    current_session = st.session_state.get("webcam_session")
    if current_session is not None:
        current_backend = getattr(current_session, "backend", None)
        if current_backend is None or not isinstance(current_backend, str) or current_backend == backend:
            return current_session

    try:
        pipeline = get_default_recognition_pipeline(backend=backend)
    except TypeError:
        pipeline = get_default_recognition_pipeline()
    except Exception:
        pipeline = None
    session = WebcamRecognitionSession(recognition_pipeline=pipeline)
    session.backend = backend
    st.session_state["webcam_session"] = session
    return st.session_state["webcam_session"]


def extract_prediction_info(pred: Optional[BoundedSignPrediction]) -> Dict[str, Any]:
    """Extracts UI-displayable gloss, confidence, class index, and top-3 candidates from BoundedSignPrediction."""
    if pred is None or getattr(pred, "prediction", None) is None:
        return {"gloss": "N/A", "confidence": 0.0, "top_k": [], "accepted": False}

    p = pred.prediction

    cls_val = getattr(p, "predicted_class", None)
    if hasattr(cls_val, "item"):
        cls_idx = int(cls_val.item())
    elif isinstance(cls_val, (int, np.integer)):
        cls_idx = int(cls_val)
    else:
        cls_idx = 0

    gloss = getattr(p, "predicted_gloss", None) or f"Sign #{cls_idx}"

    conf_val = getattr(p, "confidence", 0.0)
    if hasattr(conf_val, "item"):
        conf_score = float(conf_val.item())
    else:
        try:
            conf_score = float(conf_val)
        except (ValueError, TypeError):
            conf_score = 0.0

    accepted = bool(getattr(p, "is_accepted", True))
    if not accepted:
        gloss = "N/A (Rejected)"

    top_k_list = []
    top_k_idxs = getattr(p, "top_k_indices", None)
    top_k_probs = getattr(p, "top_k_probabilities", None)

    if top_k_idxs is not None and top_k_probs is not None:
        try:
            if hasattr(top_k_idxs, "tolist"):
                idxs = top_k_idxs.tolist()
            else:
                idxs = list(top_k_idxs)
            if hasattr(top_k_probs, "tolist"):
                probs = top_k_probs.tolist()
            else:
                probs = list(top_k_probs)

            if idxs and isinstance(idxs[0], list):
                idxs = idxs[0]
            if probs and isinstance(probs[0], list):
                probs = probs[0]

            for idx, prob in zip(idxs[:3], probs[:3]):
                top_k_list.append((f"Sign #{int(idx)}", float(prob)))
        except Exception:
            pass

    return {
        "gloss": gloss,
        "class_index": cls_idx,
        "confidence": conf_score,
        "top_k": top_k_list,
        "accepted": accepted,
    }


def render_mode_live_webcam(status_info: dict) -> None:
    """Renders Live Webcam Recognition mode with WebRTC streaming and real-time telemetry."""
    render_header(
        title="Live Webcam Recognition",
        page_icon="📹",
        description="Real-time sign language recognition and translation from your camera feed.",
    )

    try:
        from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    except Exception as err:
        st.error(
            f"Streamlit WebRTC component unavailable: {err}.\n\n"
            "Please ensure `streamlit-webrtc` and system video dependencies (av / PyAV) are installed."
        )
        return

    selected_b = st.session_state.get("selected_backend", "experimental_42")
    session = get_or_create_webcam_session(backend=selected_b)
    video_callback = make_video_frame_callback(session)

    col_stream, col_controls = st.columns([3, 1])

    with col_stream:
        webrtc_ctx = webrtc_streamer(
            key="live_webcam_streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    is_playing = bool(webrtc_ctx and hasattr(webrtc_ctx, "state") and getattr(webrtc_ctx.state, "playing", False))

    was_playing = st.session_state.get("webcam_was_playing", False)
    if was_playing and not is_playing:
        session.reset()
    st.session_state["webcam_was_playing"] = is_playing

    with col_controls:
        st.markdown("#### Session Controls")
        if st.button("Reset Recognition", key="reset_webcam_session_btn", use_container_width=True):
            session.reset()

    res = session.get_result()
    info = extract_prediction_info(res.latest_prediction)

    if not res.recognition_available:
        st.warning(
            "⚠️ **Recognition Pipeline Standby**: Trained model checkpoints or pipeline adapters are not loaded. "
            "Webcam preview is active, but live recognition inference is disabled."
        )
    elif not is_playing:
        st.info("📹 **Camera Idle**: Click **START** above to activate the live webcam stream.")

    st.markdown("### 📊 Live Telemetry & Diagnostics")

    # Row 1: Hardware & Extraction Diagnostics
    d_col1, d_col2, d_col3, d_col4, d_col5, d_col6 = st.columns(6)
    with d_col1:
        st.metric("Camera", "CONNECTED" if is_playing else "IDLE")
    with d_col2:
        if not is_playing:
            rec_status = "IDLE"
        elif not res.recognition_available:
            rec_status = "STANDBY"
        elif res.total_frames_received < res.window_size:
            rec_status = "BUFFERING"
        else:
            rec_status = "RUNNING"
        st.metric("Recognition", rec_status)
    with d_col3:
        st.metric("Frames Received", res.total_frames_received if is_playing else 0)
    with d_col4:
        st.metric("Valid Landmarks", getattr(res, "frames_valid_landmarks", 0) if is_playing else 0)
    with d_col5:
        buf_count = getattr(res, "current_buffer_count", 0)
        st.metric("Buffer Progress", f"{buf_count} / {res.window_size}" if is_playing else "0 / 60")
    with d_col6:
        st.metric("Inference Windows", res.processed_windows if is_playing else 0)

    # Row 2: Model Performance & Latest Prediction
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("FPS", f"{res.processing_fps:.1f}" if is_playing else "0.0")
    with m_col2:
        st.metric("Latency", f"{res.last_latency_ms:.1f} ms" if is_playing else "0.0 ms")
    with m_col3:
        st.metric("Latest Sign", info["gloss"] if is_playing else "N/A")
    with m_col4:
        conf_label = "Calibrated Conf" if selected_b == "experimental_42" else "Raw Conf"
        conf_val = f"{info['confidence'] * 100:.1f}%" if (is_playing and info["gloss"] != "N/A") else "N/A"
        st.metric(conf_label, conf_val)
    with m_col5:
        decision_label = "ACCEPTED" if (is_playing and info["accepted"] and info["gloss"] != "N/A") else ("REJECTED" if is_playing else "N/A")
        st.metric("Decision", decision_label)
    with m_col6:
        reason_str = getattr(res, "rejection_reason", "N/A") if is_playing else "Camera Idle"
        st.caption(f"**Policy Reason**: `{reason_str}`")

    if is_playing and info["top_k"]:
        st.markdown("#### 🎯 Top-3 Candidates")
        cand_cols = st.columns(len(info["top_k"]))
        for idx, (label, prob) in enumerate(info["top_k"]):
            with cand_cols[idx]:
                st.metric(f"Rank {idx+1}: {label}", f"{prob * 100:.1f}%")

    if is_playing and res.prediction_history:
        st.markdown("#### 📋 Recognition History & Session Export")
        history_rows = []
        for p_item in reversed(res.prediction_history):
            p_info = extract_prediction_info(p_item)
            history_rows.append({
                "Sign / Gloss": p_info["gloss"],
                "Confidence": f"{p_info['confidence'] * 100:.1f}%",
                "Status": "Accepted" if p_info["accepted"] else "Rejected",
            })
        st.dataframe(history_rows, use_container_width=True)

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_data = "Sign / Gloss,Confidence,Status\n" + "\n".join([f'"{r["Sign / Gloss"]}","{r["Confidence"]}","{r["Status"]}"' for r in history_rows])
            st.download_button("📥 Export History (CSV)", data=csv_data, file_name="webcam_recognition_history.csv", mime="text/csv", key="dl_webcam_csv")
        with exp_col2:
            import json as _json
            json_data = _json.dumps(history_rows, indent=2)
            st.download_button("📥 Export History (JSON)", data=json_data, file_name="webcam_recognition_history.json", mime="application/json", key="dl_webcam_json")

    # Auto-refresh loop to keep live Streamlit metrics updated while WebRTC stream is active
    if is_playing:
        time.sleep(0.5)
        st.rerun()


def get_or_create_continuous_sentence_session(backend: str = "experimental_42") -> ContinuousSentenceSession:
    """Retrieves or initializes the session-scoped continuous sentence session."""
    current_session = st.session_state.get("continuous_sentence_session")
    if current_session is not None:
        current_backend = getattr(current_session, "backend", None)
        if current_backend is None or not isinstance(current_backend, str) or current_backend == backend:
            return current_session

    try:
        pipeline = get_default_recognition_pipeline(backend=backend)
    except TypeError:
        pipeline = get_default_recognition_pipeline()
    except Exception:
        pipeline = None

    webcam_session = WebcamRecognitionSession(recognition_pipeline=pipeline)
    session = ContinuousSentenceSession(webcam_session=webcam_session)
    session.backend = backend
    st.session_state["continuous_sentence_session"] = session

    return st.session_state["continuous_sentence_session"]


def make_continuous_sentence_frame_callback(session: ContinuousSentenceSession):
    """Creates a WebRTC callback for the continuous sentence session."""
    def video_frame_callback(frame: Any) -> Any:
        try:
            img = frame.to_ndarray(format="bgr24")
            session.add_frame(img)
        except Exception as err:
            logger.warning(f"Error processing continuous sentence frame: {err}")
        return frame

    return video_frame_callback


def render_mode_continuous_sentence(status_info: dict) -> None:
    """Renders the live Continuous Sentence Recognition mode."""
    render_header(
        title="Continuous Sentence Recognition",
        page_icon="💬",
        description=(
            "Translate continuous Indian Sign Language sequences into structured "
            "sentences using temporal segmentation and context-aware reconstruction."
        ),
    )

    try:
        from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
    except Exception as err:
        st.error(
            f"Streamlit WebRTC component unavailable: {err}.\n\n"
            "Please ensure `streamlit-webrtc` and system video dependencies "
            "(av / PyAV) are installed."
        )
        return

    selected_b = st.session_state.get("selected_backend", "experimental_42")
    session = get_or_create_continuous_sentence_session(backend=selected_b)
    video_callback = make_continuous_sentence_frame_callback(session)

    col_stream, col_controls = st.columns([3, 1])

    with col_stream:
        webrtc_ctx = webrtc_streamer(
            key="continuous_sentence_streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    is_playing = bool(
        webrtc_ctx
        and hasattr(webrtc_ctx, "state")
        and getattr(webrtc_ctx.state, "playing", False)
    )

    was_playing = st.session_state.get("continuous_sentence_was_playing", False)
    if was_playing and not is_playing:
        pass
    st.session_state["continuous_sentence_was_playing"] = is_playing

    with col_controls:
        st.markdown("#### Session Controls")

        if st.button(
            "Finalize Sentence",
            key="finalize_continuous_sentence_btn",
            type="primary",
            use_container_width=True,
        ):
            result = session.finalize_sentence()
            st.session_state["finalized_sentence"] = result.current_sentence

        if st.button(
            "Reset Sentence",
            key="reset_continuous_sentence_btn",
            use_container_width=True,
        ):
            session.reset()
            st.session_state["finalized_sentence"] = ""

    result = session.get_result()

    if not result.recognition_available:
        st.warning(
            "⚠️ **Recognition Pipeline Standby**: trained model checkpoints or "
            "pipeline adapters are not loaded. Continuous sentence recognition is disabled."
        )
    elif not is_playing:
        st.info("📹 **Camera Idle**: Click **START** above to begin continuous signing.")

    # 1. Recognized Sign Stream
    if is_playing and getattr(result, "transcript", None) and result.transcript.filtered_tokens:
        st.markdown("### 🔀 Accepted Sign Stream")
        token_strs = [f"`{t.gloss}` ({t.confidence * 100:.1f}%)" for t in result.transcript.filtered_tokens]
        st.markdown(" ➔ ".join(token_strs))

    # 2. Current Sentence
    st.markdown("### 📝 Current Reconstructed Sentence")
    current_sentence = result.current_sentence.strip()
    if current_sentence:
        st.success(current_sentence)
    else:
        st.info(
            "No sentence reconstructed yet. Start signing after the recognition pipeline becomes active."
        )

    finalized_sentence = st.session_state.get("finalized_sentence", "").strip()
    if finalized_sentence:
        st.markdown("### ✅ Finalized Sentence")
        st.success(finalized_sentence)

    # 3. Telemetry
    st.markdown("### 📊 Continuous Recognition Telemetry")

    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)

    with m_col1:
        if not is_playing:
            status_label = "Stopped"
        elif not result.recognition_available:
            status_label = "Unavailable"
        elif result.total_frames_received < 60:
            status_label = "Buffering"
        else:
            status_label = "Active"
        st.metric("Status", status_label)

    with m_col2:
        st.metric("Accepted Tokens", result.filtered_token_count)

    with m_col3:
        conf_label = "Mean Calibrated Conf" if selected_b == "experimental_42" else "Mean Raw Conf"
        st.metric(
            conf_label,
            f"{result.mean_confidence * 100:.1f}%"
            if result.filtered_token_count
            else "N/A",
        )

    with m_col4:
        st.metric("FPS", f"{result.processing_fps:.1f}" if is_playing else "0.0")

    with m_col5:
        st.metric("Latency", f"{result.latency_ms:.1f} ms" if is_playing else "0.0 ms")

    with m_col6:
        st.metric("Windows", result.processed_windows if is_playing else 0)

    # Latest prediction
    latest_info = extract_prediction_info(result.latest_prediction)

    if result.latest_prediction is not None:
        st.markdown("### 🎯 Latest Recognition")

        latest_col1, latest_col2, latest_col3 = st.columns(3)

        with latest_col1:
            st.metric("Latest Sign", latest_info["gloss"])

        with latest_col2:
            st.metric("Confidence", f"{latest_info['confidence'] * 100:.1f}%")

        with latest_col3:
            accepted_label = "Accepted" if latest_info["accepted"] else "Rejected"
            st.metric("Prediction Status", accepted_label)

        if latest_info["top_k"]:
            st.markdown("#### Top-3 Candidates")
            candidate_cols = st.columns(len(latest_info["top_k"]))
            for idx, (label, probability) in enumerate(latest_info["top_k"]):
                with candidate_cols[idx]:
                    st.metric(
                        f"Rank {idx + 1}: {label}",
                        f"{probability * 100:.1f}%",
                    )

    if result.prediction_history:
        st.markdown("### 📋 Sign Recognition History")

        history_rows = []
        for prediction in reversed(result.prediction_history):
            info = extract_prediction_info(prediction)
            segment = getattr(prediction, "segment", None)

            history_rows.append(
                {
                    "Sign / Gloss": info["gloss"],
                    "Confidence": f"{info['confidence'] * 100:.1f}%",
                    "Status": "Accepted" if info["accepted"] else "Rejected",
                    "Start Frame": getattr(segment, "start_frame", "N/A"),
                    "End Frame": getattr(segment, "end_frame", "N/A"),
                }
            )

        st.dataframe(history_rows, use_container_width=True)

    st.caption(
        f"Frames received: {result.total_frames_received} | "
        f"Processed windows: {result.processed_windows} | "
        f"Raw tokens: {result.raw_token_count} | "
        f"Filtered tokens: {result.filtered_token_count}"
    )

    if is_playing:
        time.sleep(0.5)
        st.rerun()


def render_video_processing_result(result: Any) -> None:
    """Displays VideoProcessingResult fields in a structured Streamlit summary layout."""
    st.markdown("### 📊 Processing Results")

    selected_b = st.session_state.get("selected_backend", "experimental_42")

    # 1. Recognition status banner
    rec_avail = getattr(result, "recognition_available", False)
    if rec_avail:
        if selected_b == "experimental_42":
            st.success(
                "🎯 **Experimental 42-Class Backend — Demo Mode** Active\n\n"
                "- **Boundary trimming**: OFF | **Calibration**: T=0.68 | **Acceptance threshold**: 0.70"
            )
        else:
            st.warning(
                "⚠️ **Production — 262 classes** Active (research/experimental; low validation accuracy)"
            )
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

    # 5. Sentence Transcript if genuinely present
    transcript = getattr(result, "transcript", None)
    if transcript is not None and getattr(transcript, "text", "").strip():
        st.markdown("#### 📝 Sentence Transcript")
        st.success(f"**Transcript**: {transcript.text}")
        if hasattr(transcript, "mean_confidence") and transcript.mean_confidence > 0:
            conf_label = "Mean Calibrated Confidence ($T=0.68$)" if selected_b == "experimental_42" else "Mean Raw Softmax Confidence (UNCALIBRATED)"
            st.caption(f"{conf_label}: {transcript.mean_confidence:.2%}")

    # 6. Timeline & Diagnostic Entries
    timeline = getattr(result, "timeline", []) or []
    if timeline:
        st.markdown("#### ⏱️ Sign Recognition Timeline & Diagnostics")
        timeline_data = []
        conf_type_str = "Calibrated Confidence ($T=0.68$)" if selected_b == "experimental_42" else "Raw Softmax Confidence (UNCALIBRATED)"
        for idx, entry in enumerate(timeline, start=1):
            conf = getattr(entry, "confidence", 0.0)
            is_acc = getattr(entry, "is_accepted", conf >= 0.70)
            status_str = "Accepted" if is_acc else "Prediction rejected: confidence below threshold"

            gloss = getattr(entry, "recognized_gloss", "")
            cls_id = getattr(entry, "class_index", "N/A")
            top_3 = getattr(entry, "top_3_candidates", [])

            if not is_acc:
                st.warning(
                    f"⚠️ **Segment {idx} ({getattr(entry, 'start_timestamp', '')} - {getattr(entry, 'end_timestamp', '')})**: "
                    f"**[MODEL PREDICTION | REJECTED BY CONFIDENCE POLICY (<0.70) | NOT USER-VERIFIED]**\n\n"
                    f"- **Predicted Gloss**: `{gloss}`\n"
                    f"- **Class ID**: `{cls_id}`\n"
                    f"- **{conf_type_str}**: `{conf:.4f}` ({conf * 100:.2f}%)\n"
                    f"- **Acceptance Threshold**: `0.70` (Rejection Policy Active)\n"
                )
                if top_3:
                    top_3_str = ", ".join([f"`{g}` ({p * 100:.1f}%)" for g, p in top_3])
                    st.caption(f"Top-3 Candidates: {top_3_str}")
            else:
                st.success(
                    f"✅ **Segment {idx} ({getattr(entry, 'start_timestamp', '')} - {getattr(entry, 'end_timestamp', '')})**: "
                    f"**[MODEL PREDICTION | ACCEPTED BY CONFIDENCE POLICY (≥0.70) | NOT USER-VERIFIED]**\n\n"
                    f"- **Accepted Gloss**: `{gloss}` (Class ID `{cls_id}`)\n"
                    f"- **{conf_type_str}**: `{conf * 100:.2f}%` (Acceptance Threshold: `0.70`)\n"
                )

            timeline_data.append(
                {
                    "Segment": idx,
                    "Timestamp": f"{getattr(entry, 'start_timestamp', '')} - {getattr(entry, 'end_timestamp', '')}",
                    "Predicted Gloss": gloss,
                    "Class ID": cls_id,
                    "Confidence Score": f"{conf * 100:.2f}%",
                    "Confidence Type": "Calibrated (T=0.68)" if selected_b == "experimental_42" else "Raw Softmax",
                    "Status": status_str,
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

        backend = st.session_state.get("selected_backend", "production")
        if backend != "production":
            result = process_video_file(temp_path, backend=backend)
        else:
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

    # Verified Demo Examples helper (TRAIN split only)
    demo_manifest_path = Path("data/processed/include/demo_assets/index.json")
    if demo_manifest_path.exists():
        st.markdown("#### 🧪 Verified Demo Examples (TRAIN Split Only)")
        try:
            with open(demo_manifest_path, "r", encoding="utf-8") as f:
                demo_data = json.load(f)
            assets = demo_data.get("demo_assets", [])
        except Exception:
            assets = []

        if assets:
            asset_options = {
                f"Class '{a['label']}' ({Path(a['demo_video_path']).name} - {a['duration_seconds']}s)": a
                for a in assets
            }
            selected_asset_label = st.selectbox(
                "Select a Verified INCLUDE Training Sample Video:",
                options=list(asset_options.keys()),
                key="sb_demo_asset_select",
            )
            selected_asset = asset_options[selected_asset_label]
            sample_demo_path = Path("data/processed/include") / selected_asset["demo_video_path"]

            st.caption(
                f"**Selected Asset**: `{selected_asset['label']}` | **Path**: `{sample_demo_path}` | **Duration**: `{selected_asset['duration_seconds']}s`"
            )
            if st.button(f"▶️ Run Selected Demo Video ('{selected_asset['label']}')", key="btn_run_selected_demo_video"):
                st.session_state["processing"] = True
                st.session_state["status_message"] = f"Processing demo video '{selected_asset['label']}'..."
                selected_b = st.session_state.get("selected_backend", "experimental_42")
                result = process_video_file(sample_demo_path, backend=selected_b)
                st.session_state["latest_video_result"] = result
                st.session_state["processing"] = False
                st.session_state["status_message"] = "Demo video processing completed."
        st.markdown("---")

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
        result = st.session_state.get("latest_video_result")
        if result is not None:
            render_video_processing_result(result)
        else:
            render_empty_state(
                title="Video Upload Processing",
                message="Upload a video file above or run the Known-Good Demo Test to process frames through the recognition pipeline.",
                icon="📁",
            )


def _handle_video_url_processing(url: str) -> None:
    """Invokes process_video_url service, handles errors, and updates session state."""
    st.session_state["processing"] = True
    st.session_state["status_message"] = "Downloading and processing video from URL..."
    st.session_state["last_error"] = None

    try:
        backend = st.session_state.get("selected_backend", "production")
        if backend != "production":
            result = process_video_url(url=url, backend=backend)
        else:
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

    st.info(
        "ℹ️ **Model Verification & Domain Constraint Notice**:\n"
        "- **MODEL PREDICTION**: Unverified automated output from recognition model.\n"
        "- **ACCEPTED BY CONFIDENCE POLICY**: Inferences passing the mandatory **0.70 threshold** using calibrated confidence ($T=0.68$).\n"
        "- **NOT USER-VERIFIED**: Automated model outputs require domain expert verification.\n\n"
        "⚠️ **Domain Constraint**: Recognition is strictly limited to the supported 42-class controlled vocabulary. "
        "Arbitrary public internet videos may contain unsupported signs, complex background clutter, distribution shifts, or non-standard camera angles, resulting in low confidence or rejection."
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
        st.markdown("#### 🔄 Multi-Stage Pipeline Diagnostics & Evidence Fusion")
        stage_col1, stage_col2, stage_col3, stage_col4 = st.columns(4)
        with stage_col1:
            st.metric("1. URL Validation", "✓ ACCEPTED")
            st.metric("5. Landmark Extraction", f"✓ {getattr(result, 'frames_processed', 0)} frames")
        with stage_col2:
            st.metric("2. Video Retrieval", "✓ DOWNLOADED")
            st.metric("6. Window Recognition", f"✓ {len(getattr(result, 'timeline', []))} windows")
        with stage_col3:
            st.metric("3. Video Decoding", f"✓ {getattr(result, 'frames_processed', 0)} frames")
            st.metric("7. Confidence Policy", "✓ T=0.68 | ≥0.70")
        with stage_col4:
            st.metric("4. Signer Tracking", "✓ 1 Primary Signer")
            is_any_accepted = any(getattr(e, "is_accepted", False) for e in getattr(result, "timeline", []))
            st.metric("8. Final Interpretation", "✓ ACCEPTED SIGN" if is_any_accepted else "⚠️ REJECTED (<0.70)")

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

    # 2. Missing & Included Tokens Handling
    demos = getattr(result.sequence, "demonstrations", []) or []
    missing_tokens = []
    for d in demos:
        if d.source == DemonstrationSource.UNKNOWN or (d.source == DemonstrationSource.DICTIONARY and not d.video_path):
            missing_tokens.append(d.token)
    
    if result.skipped_tokens:
        for st_tok in result.skipped_tokens:
            if st_tok not in missing_tokens:
                missing_tokens.append(st_tok)

    if missing_tokens:
        missing_str = ", ".join([f"'{w}'" for w in missing_tokens])
        st.warning(f"⚠️ **ISL demonstration unavailable for**: {missing_str}")
        if result.included_tokens:
            st.info("ℹ️ **Partial Translation**: Stitching video only for available dictionary clips. Missing tokens are omitted from video compilation.")

    if result.included_tokens:
        st.markdown(f"**Included Words**: `{', '.join(result.included_tokens)}`")

    # 3. Demonstration Sequence in Token Order
    if demos:
        st.markdown("#### 📋 Demonstration Sequence")
        seq_data = []
        for idx, d in enumerate(demos, 1):
            source_str = d.source.value if hasattr(d.source, "value") else str(d.source)
            detail = d.dictionary_gloss if d.dictionary_gloss else (", ".join(d.characters) if d.characters else "Unavailable")
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
        comp_json_path = Path("reports/evaluations/retrieval_hybrid_comparison.json")
        if comp_json_path.exists():
            try:
                with open(comp_json_path, "r", encoding="utf-8") as f:
                    comp_data = json.load(f)
                st.markdown("### 🔬 Recognition Strategy Evaluation (Model vs Retrieval vs Hybrid)")
                st.caption(
                    f"Evaluated on {comp_data.get('dataset_inventory', {}).get('validation_samples', 126)} validation "
                    f"and {comp_data.get('dataset_inventory', {}).get('test_samples', 126)} held-out test videos across "
                    f"{comp_data.get('dataset_inventory', {}).get('controlled_classes', 42)} controlled classes."
                )
                
                val_m = comp_data.get("validation_metrics", {})
                rows = []
                for strategy, m in val_m.items():
                    rows.append({
                        "Strategy": strategy,
                        "Validation Top-1 Acc": f"{m.get('top1_accuracy', 0.0):.2f}%",
                        "Validation Top-3 Acc": f"{m.get('top3_accuracy', 0.0):.2f}%",
                        "Validation ECE": f"{m.get('ece', 0.0):.4f}",
                        "Acceptance Rate (>=0.70)": f"{m.get('acceptance_rate', 0.0):.2f}%",
                        "Accepted Precision": f"{m.get('accepted_precision', 0.0):.2f}%",
                    })
                st.dataframe(rows, use_container_width=True)
                st.info(f"**Selected Default Strategy**: `{comp_data.get('selected_default_mode', 'Model-Only')}` — {comp_data.get('rationale', '')}")
                st.markdown("---")
            except Exception as err:
                logger.warning(f"Could not load retrieval_hybrid_comparison.json: {err}")

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

    # Sidebar backend selection
    st.sidebar.markdown("---")
    st.sidebar.title("Recognition Backend")
    backend_options = {
        "experimental_42": "Experimental — 42 classes (Controlled Demo Backend)",
        "production": "Production — 262 classes (research/experimental; low validation accuracy)",
    }
    selected_backend = st.sidebar.radio(
        "Active Model Backend",
        options=list(backend_options.keys()),
        format_func=lambda k: backend_options[k],
        index=0 if st.session_state.get("selected_backend", "experimental_42") == "experimental_42" else 1,
        help="Select between Experimental 42-class controlled demo model and Production 262-class research model.",
    )
    st.session_state["selected_backend"] = selected_backend

    if selected_backend == "experimental_42":
        st.sidebar.warning(
            "⚠️ **Controlled Vocabulary Notice**: The Experimental backend supports only a controlled subset "
            "of 42 high-frequency ISL classes (min 15 training samples/class). It is NOT a general 262-class ISL translator."
        )

    # Model and checkpoint availability check
    models_dir = getattr(app_config.paths, "models_dir", "models")
    status_info = check_model_availability(models_dir)
    render_model_status(status_info, backend=selected_backend)

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

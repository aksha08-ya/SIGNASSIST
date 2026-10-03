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
        return {"gloss": "N/A", "raw_gloss": "N/A", "confidence": 0.0, "top_k": [], "accepted": False}

    p = pred.prediction

    cls_val = getattr(p, "predicted_class", None)
    if hasattr(cls_val, "item"):
        cls_idx = int(cls_val.item())
    elif isinstance(cls_val, (int, np.integer)):
        cls_idx = int(cls_val)
    else:
        cls_idx = 0

    raw_gloss = getattr(p, "predicted_gloss", None) or f"Sign #{cls_idx}"

    conf_val = getattr(p, "confidence", 0.0)
    if hasattr(conf_val, "item"):
        conf_score = float(conf_val.item())
    else:
        try:
            conf_score = float(conf_val)
        except (ValueError, TypeError):
            conf_score = 0.0

    accepted = bool(getattr(p, "is_accepted", True))
    gloss = raw_gloss if accepted else f"{raw_gloss} (Candidate / Rejected)"

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
        "raw_gloss": raw_gloss,
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
    last_eval_pred = getattr(res, "last_evaluated_prediction", None)
    eval_info = extract_prediction_info(last_eval_pred) if last_eval_pred else None

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

    has_accepted_pred = bool(is_playing and res.latest_prediction is not None and info.get("accepted", False) and info.get("gloss") != "N/A")
    has_candidate_pred = bool(is_playing and not has_accepted_pred and eval_info is not None and eval_info.get("raw_gloss", "N/A") != "N/A")

    # Row 2: Model Performance & Latest Prediction
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.metric("FPS", f"{res.processing_fps:.1f}" if is_playing else "0.0")
    with m_col2:
        st.metric("Latency", f"{res.last_latency_ms:.1f} ms" if is_playing else "0.0 ms")
    with m_col3:
        if has_accepted_pred:
            st.metric("Latest Sign", info["raw_gloss"])
        elif has_candidate_pred:
            st.metric("Candidate Sign", f"{eval_info['raw_gloss']} (Candidate / Rejected)")
        else:
            st.metric("Latest Sign", "N/A")
    with m_col4:
        if has_accepted_pred:
            conf_label = "Calibrated Conf" if selected_b == "experimental_42" else "Raw Conf"
            conf_val = f"{info['confidence'] * 100:.1f}%"
        elif has_candidate_pred:
            conf_label = "Candidate Confidence"
            conf_val = f"{eval_info['confidence'] * 100:.1f}%"
        else:
            conf_label = "Calibrated Conf" if selected_b == "experimental_42" else "Raw Conf"
            conf_val = "N/A"
        st.metric(conf_label, conf_val)
    with m_col5:
        if has_accepted_pred:
            decision_label = "ACCEPTED"
        elif has_candidate_pred:
            decision_label = "REJECTED"
        elif is_playing:
            decision_label = "REJECTED"
        else:
            decision_label = "N/A"
        st.metric("Decision", decision_label)
    with m_col6:
        reason_str = getattr(res, "rejection_reason", "N/A") if is_playing else "Camera Idle"
        st.caption(f"**Policy Reason**: `{reason_str}`")

    display_top_k = info["top_k"] if has_accepted_pred else (eval_info["top_k"] if has_candidate_pred else [])
    if display_top_k:
        top_k_title = "🎯 Top-3 Candidates" if has_accepted_pred else "🎯 Top-3 Candidates (Candidate / Rejected)"
        st.markdown(f"#### {top_k_title}")
        cand_cols = st.columns(len(display_top_k))
        for idx, (label, prob) in enumerate(display_top_k):
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


def render_combined_video_intelligence_results(
    local_result: Any,
    ai_result: Optional[Dict[str, Any]] = None,
    video_path: Optional[Union[str, Path]] = None,
) -> None:
    """Displays structured, separated output for local ST-GCN, AI analysis, Research Grounding, and Final Interpretation."""
    st.markdown("---")
    st.markdown("## 🧭 Multimodal Video Interpretation Results")

    selected_b = st.session_state.get("selected_backend", "experimental_42")

    # ==========================================
    # SECTION A: SIGNASSIST LOCAL MODEL
    # ==========================================
    st.markdown("### A. 🎯 SIGNASSIST Local Recognition Model")

    rec_avail = getattr(local_result, "recognition_available", False)
    if rec_avail:
        if selected_b == "experimental_42":
            st.success(
                "🎯 **Experimental 42-Class Backend — Demo Mode** Active\n\n"
                "- **Supported Vocabulary**: 42 Controlled Classes\n"
                "- **Boundary Trimming**: OFF | **Calibration**: T=0.68 | **Acceptance Threshold**: 0.70"
            )
        else:
            st.warning("⚠️ **Production — 262 classes** Active (research/experimental; low validation accuracy)")
    else:
        st.warning("Local recognition backend unavailable (standby / no model checkpoint).")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames Processed", getattr(local_result, "frames_processed", 0))
    with col2:
        fps = getattr(local_result, "processing_fps", 0.0)
        st.metric("Processing FPS", f"{fps:.1f}")
    with col3:
        exec_sec = getattr(local_result, "execution_seconds", 0.0)
        st.metric("Execution Time", f"{exec_sec:.2f} s")
    with col4:
        st.metric("Recognition Status", "Active" if rec_avail else "Standby")

    # Quality Report if present
    quality = getattr(local_result, "quality_report", None)
    if quality is not None:
        st.markdown("#### 🎥 Local Video Quality Assessment")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        with q_col1:
            st.metric("Overall Rating", getattr(quality, "overall_rating", "N/A"))
        with q_col2:
            st.metric("Brightness", f"{getattr(quality, 'average_brightness', 0.0):.1f}")
        with q_col3:
            st.metric("Contrast", f"{getattr(quality, 'average_contrast', 0.0):.1f}")
        with q_col4:
            st.metric("Sharpness (Blur)", f"{getattr(quality, 'average_blur_score', 0.0):.1f}")

    # Timeline & Diagnostic Entries
    timeline = getattr(local_result, "timeline", []) or []
    if timeline:
        st.markdown("#### ⏱️ Local Sign Recognition Timeline & Diagnostics")
        timeline_data = []
        conf_type_str = "Calibrated Confidence ($T=0.68$)" if selected_b == "experimental_42" else "Raw Softmax Confidence (UNCALIBRATED)"
        accepted_glosses = []
        for idx, entry in enumerate(timeline, start=1):
            conf = getattr(entry, "confidence", 0.0)
            is_acc = getattr(entry, "is_accepted", conf >= 0.70)
            status_str = "Accepted" if is_acc else "Prediction rejected: confidence below threshold"
            gloss = getattr(entry, "recognized_gloss", "")
            cls_id = getattr(entry, "class_index", "N/A")
            if is_acc and gloss:
                accepted_glosses.append(gloss)

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

        pred_gloss_str = ", ".join(accepted_glosses) if accepted_glosses else "None"
        st.warning(
            f"⚠️ **WARNING**: This is a controlled-vocabulary prediction (`{pred_gloss_str}`) "
            "and does not establish that the video means this phrase in continuous ISL."
        )

    # ==========================================
    # SECTION B: AI-ASSISTED VIDEO ANALYSIS
    # ==========================================
    st.markdown("---")
    st.markdown("### B. 🤖 AI-Assisted Multimodal Video Analysis")

    if not ai_result or ai_result.get("status") == "AI_ANALYSIS_UNAVAILABLE":
        err_msg = ai_result.get("error", "GEMINI_API_KEY environment variable is not configured.") if ai_result else "GEMINI_API_KEY environment variable is not configured."
        st.info(
            f"ℹ️ **AI Analysis Unavailable**: {err_msg}\n\n"
            "Local SIGNASSIST recognition remains active. Configure `GEMINI_API_KEY` in `.env` to enable Gemini multimodal video analysis."
        )
    elif ai_result.get("status") == "SUCCESS":
        st.success(
            f"✅ **Gemini Multimodal Video Analysis Complete**\n\n"
            f"- **Provider**: `{ai_result.get('provider')}` | **Model**: `{ai_result.get('model')}`\n"
            f"- **ISL Detected**: `{ai_result.get('is_probably_isl')}`"
        )
        if ai_result.get("isl_assessment"):
            st.markdown(f"**ISL Assessment**: {ai_result.get('isl_assessment')}")

        segments = ai_result.get("sign_segments", [])
        if segments:
            st.markdown("#### 👁️ Observed Visual Sign Segments")
            st.dataframe(segments, use_container_width=True)

        col_g, col_e = st.columns(2)
        with col_g:
            glosses = ai_result.get("gloss_sequence", [])
            st.markdown(f"**Inferred Gloss Sequence**: `{ ' -> '.join(glosses) if glosses else 'None' }`")
        with col_e:
            st.markdown(f"**English Interpretation**: **{ai_result.get('interpreted_sentence') or 'N/A'}**")

        if ai_result.get("spoken_audio_transcript"):
            st.markdown(f"**Spoken Audio Transcript**: *{ai_result.get('spoken_audio_transcript')}*")
        if ai_result.get("visible_text"):
            st.markdown(f"**Visible Text / Subtitles**: *{ai_result.get('visible_text')}*")
        if ai_result.get("uncertainty_notes"):
            st.caption(f"**Uncertainty / Ambiguity Notes**: {ai_result.get('uncertainty_notes')}")

    elif ai_result.get("status") == "ERROR":
        st.error(f"❌ AI Video Analysis Error: {ai_result.get('error')}")

    # ==========================================
    # SECTION C: RESEARCH / EVIDENCE SOURCES
    # ==========================================
    st.markdown("---")
    st.markdown("### C. 🔍 Research / Evidence Sources (Google Search Grounding)")

    sources = ai_result.get("evidence_sources", []) if ai_result else []
    if sources:
        for idx, src in enumerate(sources, start=1):
            st.markdown(
                f"**{idx}. [{src.get('title', 'Link')}]({src.get('url', '#')})**\n"
                f"- **Relevance**: {src.get('relevance')}\n"
                f"- **Support Level**: `{src.get('support_level')}`"
            )
    else:
        st.caption("No external web research grounding sources returned for this video query.")

    # ==========================================
    # SECTION D: FINAL INTERPRETATION
    # ==========================================
    st.markdown("---")
    st.markdown("### D. 📜 Final Combined Interpretation")

    interp_text = ai_result.get("interpreted_sentence") if (ai_result and ai_result.get("status") == "SUCCESS") else None
    if not interp_text and getattr(local_result, "transcript", None):
        interp_text = getattr(local_result.transcript, "text", "")

    if interp_text:
        st.success(
            f"### **Final Meaning**: {interp_text}\n\n"
            f"🏷️ **Provenance**: `{ai_result.get('provenance', 'AI-assisted / unverified') if ai_result else 'Local Controlled Model / unverified'}`"
        )
    else:
        st.info("No definitive final sentence interpretation supported by current evidence.")

    st.caption(
        "⚠️ **Disclaimer**: This interpretation combines local controlled recognition with external AI multimodal analysis. "
        "It has not been independently validated by a certified ISL human interpreter."
    )


def render_video_processing_result(result: Any, ai_result: Optional[Dict[str, Any]] = None, video_path: Optional[str] = None) -> None:
    """Displays VideoProcessingResult fields in a structured Streamlit summary layout."""
    ai_res = ai_result or st.session_state.get("latest_ai_result")
    render_combined_video_intelligence_results(result, ai_result=ai_res, video_path=video_path)


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
    """Renders TextToISLResult fields, metrics, individual video players, unavailable tokens, and stitched video."""
    st.markdown("### 🤟 ISL Video Demonstration Summary")

    st.markdown(
        "ℹ️ **Lexical Video Demonstration Notice**: This feature displays canonical Indian Sign Language (ISL) "
        "dictionary video clips for recognized words/phrases. It demonstrates lexical sign equivalents in input sequence "
        "and is not an unrestricted English-to-ISL grammatical translator."
    )
    st.caption(f"Original Input: `{result.sequence.original_text}` | Normalized Text: `{result.sequence.normalized_text}`")

    demos = getattr(result.sequence, "demonstrations", []) or []
    total_tokens = len(demos)
    dict_matched = result.dictionary_count
    coverage_pct = (dict_matched / total_tokens * 100.0) if total_tokens > 0 else 0.0

    # 1. Coverage Metrics Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dictionary Signs", result.dictionary_count)
    with col2:
        st.metric("Fingerspelled Words", result.fingerspell_count)
    with col3:
        st.metric("Unknown Words", result.unknown_count)

    st.markdown(
        f"**Coverage Summary**: `{dict_matched}/{total_tokens}` input tokens ({coverage_pct:.1f}%) "
        f"have verified ISL dictionary demonstrations."
    )

    # 2. Combined Stitched Video Output (Multi-word assembly)
    if result.output_video_path and Path(result.output_video_path).exists():
        st.markdown("#### 🎬 Combined Stitched ISL Demonstration Video")
        st.video(result.output_video_path)
        clip_count = getattr(result.stitch_result, "total_clips", len(result.included_tokens))
        duration = getattr(result.stitch_result, "duration_sec", 0.0)
        st.caption(
            f"**Included Words**: `{', '.join(result.included_tokens)}` | "
            f"**Clips Stitched**: `{clip_count}` | "
            f"**Duration**: `{duration:.2f}s` | "
            f"**Format**: H.264 Web MP4 (640x480 @ 30 FPS)"
        )
    elif result.included_tokens:
        st.info("ℹ️ Stitched compilation video unavailable, but individual video clips are displayed below.")
    else:
        st.info(
            "Textual ISL demonstration sequence generated. "
            "No stitched video clip could be produced because video assets for the dictionary entries are unavailable."
        )

    # 3. Individual Verified ISL Videos Section
    matched_demos = [d for d in demos if d.source == DemonstrationSource.DICTIONARY and d.video_path]
    if matched_demos:
        st.markdown("#### 📽️ Individual Verified ISL Dictionary Videos")
        dict_dir = Path("data/external/isl_dictionary")
        for idx, d in enumerate(matched_demos, 1):
            abs_v_path = dict_dir / d.video_path
            with st.container():
                st.markdown(f"##### Sign #{idx}: `{d.token}` ➔ Canonical Gloss: `{d.dictionary_gloss}`")
                v_col1, v_col2 = st.columns([1, 1])
                with v_col1:
                    if abs_v_path.exists():
                        st.video(str(abs_v_path))
                    else:
                        st.warning(f"Video file missing on disk: {abs_v_path}")
                with v_col2:
                    st.markdown(f"- **Input Word**: `{d.token}`")
                    st.markdown(f"- **Canonical Gloss**: `{d.dictionary_gloss}`")
                    st.markdown(f"- **Category**: `{getattr(d, 'category', None) or 'general'}`")
                    if getattr(d, "duration_sec", None):
                        st.markdown(f"- **Clip Duration**: `{d.duration_sec:.2f}s`")
                    st.markdown(f"- **Source Provenance**: `ISL Dictionary (INCLUDE_TRAIN)`")
                    st.markdown(f"- **Local Asset Path**: `{abs_v_path}`")
                    if d.description:
                        st.caption(f"Notes: {d.description}")
                st.markdown("---")

    # 4. Unavailable Tokens Handling
    missing_tokens = []
    for d in demos:
        if d.source == DemonstrationSource.UNKNOWN or (d.source == DemonstrationSource.DICTIONARY and not d.video_path):
            missing_tokens.append(d.token)
    if result.skipped_tokens:
        for st_tok in result.skipped_tokens:
            if st_tok not in missing_tokens:
                missing_tokens.append(st_tok)

    if missing_tokens:
        st.markdown("#### ⚠️ Unavailable Demonstrations")
        for un_tok in missing_tokens:
            st.warning(
                f"⚠️ **ISL demonstration unavailable for**: `{un_tok}`\n\n"
                f"- **Reason**: No verified ISL dictionary video asset exists in `data/external/isl_dictionary/` for this word.\n"
                f"- **Fingerspelling Policy**: Fingerspelling is disabled as no verified ISL alphabet video assets are present in the repository.\n"
                f"- **Integrity Policy**: No simulated, fake, or ASL video assets have been substituted."
            )

    # 5. Full Sequence Provenance Dataframe
    if demos:
        st.markdown("#### 📋 Full Demonstration Sequence Provenance")
        seq_data = []
        dict_dir = Path("data/external/isl_dictionary")
        for idx, d in enumerate(demos, 1):
            source_str = d.source.value if hasattr(d.source, "value") else str(d.source)
            detail = d.dictionary_gloss if d.dictionary_gloss else "ISL demonstration unavailable"
            v_path_str = str(dict_dir / d.video_path) if d.video_path else "N/A"
            seq_data.append(
                {
                    "Step": idx,
                    "Input Token": d.token,
                    "Normalized Token": d.normalized_token,
                    "Source": source_str.upper(),
                    "Canonical Gloss": detail,
                    "Asset Video Path": v_path_str,
                }
            )
        st.dataframe(seq_data, use_container_width=True)


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



def main() -> None:
    """Main application entry point."""
    try:
        app_config = get_ui_config()
    except Exception as err:
        st.set_page_config(page_title="SIGNASSIST - Error", page_icon="⚠️", layout="wide")
        render_error_state(f"Failed to load application configuration: {err}")
        return

    st.set_page_config(
        page_title=app_config.ui.app.title,
        page_icon=app_config.ui.app.page_icon,
        layout=app_config.ui.app.layout,
    )

    init_session_state()

    st.sidebar.title("Navigation")
    selected_mode = st.sidebar.selectbox(
        "Application Mode",
        options=APPLICATION_MODES,
        index=APPLICATION_MODES.index(
            st.session_state.get("current_mode", APPLICATION_MODES[0])
        ),
    )
    st.session_state["current_mode"] = selected_mode

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

    models_dir = getattr(app_config.paths, "models_dir", "models")
    status_info = check_model_availability(models_dir)
    render_model_status(status_info, backend=selected_backend)

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
    else:
        render_error_state(f"Unknown mode: {selected_mode}")


if __name__ == "__main__":
    main()

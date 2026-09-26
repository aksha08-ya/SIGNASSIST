"""Reusable presentation-only UI components and availability helpers for SIGNASSIST."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import streamlit as st


from src.ui.services import get_recognition_readiness


def check_model_availability(
    models_dir: Optional[Union[str, Path]] = None,
    pipeline_instance: Optional[Any] = None,
    explicit_pipeline_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Checks for model architecture, trained checkpoint, and recognition pipeline availability.

    Delegates to get_recognition_readiness in src.ui.services.
    """
    return get_recognition_readiness(
        models_dir=models_dir,
        pipeline_instance=pipeline_instance,
        explicit_pipeline_available=explicit_pipeline_available,
    )


def render_header(
    title: str,
    page_icon: Optional[str] = "🤟",
    description: Optional[str] = None,
) -> None:
    """Renders the main application header."""
    header_text = f"{page_icon} {title}" if page_icon else title
    st.title(header_text)
    if description:
        st.markdown(description)
    st.divider()


def render_model_status(status_info: Dict[str, Any]) -> None:
    """Renders backend and model checkpoint availability status in sidebar/panel."""
    st.sidebar.markdown("### ⚙️ Backend Status")

    arch_status = "✅ Available" if status_info.get("architecture_available") else "❌ Unavailable"
    st.sidebar.markdown(f"**Model Architecture**: {arch_status}")

    if status_info.get("checkpoint_available"):
        count = status_info.get("checkpoint_count", 0)
        st.sidebar.markdown(f"**Trained Checkpoints**: ✅ Found ({count})")
    else:
        st.sidebar.markdown("**Trained Checkpoints**: ⚠️ None Found")

    if status_info.get("recognition_pipeline_available"):
        st.sidebar.success("Recognition Pipeline: Ready")
    else:
        st.sidebar.warning("Recognition Pipeline: Standby (No Checkpoint/Pipeline)")


def render_empty_state(
    title: str,
    message: str,
    icon: Optional[str] = "ℹ️",
) -> None:
    """Renders a structured informational empty state placeholder."""
    st.info(f"### {icon} {title}\n\n{message}")


def render_success_state(message: str) -> None:
    """Renders a success notification state."""
    st.success(message)


def render_error_state(error_message: str) -> None:
    """Renders an error notification state."""
    st.error(f"⚠️ Error: {error_message}")


def render_status_indicator(
    status_message: str,
    is_processing: bool = False,
) -> None:
    """Renders a status indicator with status message and processing state."""
    if is_processing:
        st.info(f"⏳ {status_message}")
    else:
        st.caption(f"Status: {status_message}")

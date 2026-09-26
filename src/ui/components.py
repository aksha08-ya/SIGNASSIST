"""Reusable presentation-only UI components and availability helpers for SIGNASSIST."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import streamlit as st


def check_model_availability(
    models_dir: Optional[Union[str, Path]] = None,
    pipeline_instance: Optional[Any] = None,
    explicit_pipeline_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Checks for model architecture, trained checkpoint, and recognition pipeline availability.

    Do NOT instantiate/load models or equate checkpoint presence with pipeline readiness.

    Args:
        models_dir: Optional directory path to check for trained model checkpoints.
                   If None, defaults to Path("models").
        pipeline_instance: Optional instantiated pipeline/adapter object to check for readiness.
        explicit_pipeline_available: Optional explicit boolean flag indicating if a usable pipeline is ready.

    Returns:
        Dictionary containing detailed availability status:
          - architecture_available: bool (True since PyTorch architecture modules exist in codebase)
          - checkpoint_available: bool (True only if checkpoint files exist on disk)
          - recognition_pipeline_available: bool (False unless a usable pipeline is verified/supplied)
          - checkpoint_count: int
          - models_dir: str
    """
    m_dir = Path(models_dir) if models_dir is not None else Path("models")

    checkpoint_files = []
    if m_dir.exists() and m_dir.is_dir():
        for ext in ("*.pt", "*.pth", "*.bin", "*.ckpt"):
            checkpoint_files.extend(list(m_dir.glob(ext)))

    checkpoint_files = sorted(list(set(checkpoint_files)))
    checkpoint_available = len(checkpoint_files) > 0
    architecture_available = True  # ST-GCN, GNN, Transformer, Fusion modules exist in codebase

    # Recognition pipeline is available ONLY if an explicit verified pipeline instance or flag is provided.
    # Checkpoints existing does NOT imply a pipeline adapter is initialized or available.
    if explicit_pipeline_available is not None:
        recognition_pipeline_available = bool(explicit_pipeline_available)
    elif pipeline_instance is not None:
        recognition_pipeline_available = getattr(pipeline_instance, "is_ready", True)
    else:
        recognition_pipeline_available = False

    return {
        "architecture_available": architecture_available,
        "checkpoint_available": checkpoint_available,
        "recognition_pipeline_available": recognition_pipeline_available,
        "checkpoint_count": len(checkpoint_files),
        "models_dir": str(m_dir),
    }


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

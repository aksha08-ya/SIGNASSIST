"""Centralized session state initialization and management for SIGNASSIST UI."""

from typing import Any, Dict, MutableMapping, Optional
import streamlit as st

INITIAL_STATE_DEFAULTS: Dict[str, Any] = {
    "current_mode": "Live Webcam",
    "processing": False,
    "transcript": [],
    "prediction_history": [],
    "selected_signer_id": None,
    "latest_video_result": None,
    "last_error": None,
    "status_message": "Ready",
}


def init_session_state(
    state_store: Optional[MutableMapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    """Initializes session state with default values if keys are not already present.

    This operation is idempotent: existing values in state_store will not be overwritten.

    Args:
        state_store: Optional mutable mapping (e.g. dict or st.session_state).
                    If None, uses streamlit.session_state.

    Returns:
        The initialized mutable mapping instance.
    """
    if state_store is None:
        state_store = st.session_state

    for key, default_value in INITIAL_STATE_DEFAULTS.items():
        if key not in state_store:
            if isinstance(default_value, list):
                state_store[key] = list(default_value)
            elif isinstance(default_value, dict):
                state_store[key] = dict(default_value)
            else:
                state_store[key] = default_value

    return state_store

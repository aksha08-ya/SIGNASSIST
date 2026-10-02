"""Integration tests for Phase 17 Continuous Sentence Streamlit UI wiring."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


def test_continuous_sentence_session_factory_reuses_session(monkeypatch):
    import src.app as app

    fake_pipeline = object()
    fake_webcam_session = MagicMock()
    fake_continuous_session = MagicMock()

    monkeypatch.setattr(app, "get_default_recognition_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(
        app,
        "WebcamRecognitionSession",
        lambda recognition_pipeline=None: (
            fake_webcam_session
            if recognition_pipeline is fake_pipeline
            else pytest.fail("Unexpected recognition pipeline")
        ),
    )
    monkeypatch.setattr(
        app,
        "ContinuousSentenceSession",
        lambda webcam_session=None: (
            fake_continuous_session
            if webcam_session is fake_webcam_session
            else pytest.fail("Unexpected webcam session")
        ),
    )

    state = {}

    monkeypatch.setattr(app.st, "session_state", state)

    first = app.get_or_create_continuous_sentence_session()
    second = app.get_or_create_continuous_sentence_session()

    assert first is fake_continuous_session
    assert second is fake_continuous_session
    assert state["continuous_sentence_session"] is fake_continuous_session


def test_continuous_sentence_frame_callback_delegates_frame():
    import src.app as app

    session = MagicMock()
    callback = app.make_continuous_sentence_frame_callback(session)

    expected = np.zeros((32, 48, 3), dtype=np.uint8)

    frame = MagicMock()
    frame.to_ndarray.return_value = expected

    returned = callback(frame)

    assert returned is frame
    frame.to_ndarray.assert_called_once_with(format="bgr24")
    session.add_frame.assert_called_once_with(expected)


def test_continuous_sentence_frame_callback_handles_processing_error():
    import src.app as app

    session = MagicMock()
    session.add_frame.side_effect = RuntimeError("inference failure")

    callback = app.make_continuous_sentence_frame_callback(session)

    frame = MagicMock()
    frame.to_ndarray.return_value = np.zeros((16, 16, 3), dtype=np.uint8)

    assert callback(frame) is frame


def test_continuous_sentence_session_factory_does_not_create_second_session():
    import src.app as app

    existing = MagicMock()
    state = {"continuous_sentence_session": existing}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(app.st, "session_state", state)

    try:
        result = app.get_or_create_continuous_sentence_session()
    finally:
        monkeypatch.undo()

    assert result is existing


def test_continuous_sentence_renderer_uses_stable_webrtc_key(monkeypatch):
    import src.app as app

    fake_session = MagicMock()
    fake_result = SimpleNamespace(
        recognition_available=False,
        status="unavailable",
        current_sentence="",
        mean_confidence=0.0,
        raw_token_count=0,
        filtered_token_count=0,
        latest_prediction=None,
        prediction_history=[],
        total_frames_received=0,
        processed_windows=0,
        processing_fps=0.0,
        latency_ms=0.0,
    )

    fake_session.get_result.return_value = fake_result

    monkeypatch.setattr(
        app, "get_or_create_continuous_sentence_session", lambda *args, **kwargs: fake_session
    )

    captured = {}

    def fake_webrtc_streamer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(state=SimpleNamespace(playing=False))

    fake_module = SimpleNamespace(
        RTCConfiguration=lambda value: value,
        WebRtcMode=SimpleNamespace(SENDRECV="sendrecv"),
        webrtc_streamer=fake_webrtc_streamer,
    )

    import sys

    monkeypatch.setitem(sys.modules, "streamlit_webrtc", fake_module)

    monkeypatch.setattr(app.st, "session_state", {})

    monkeypatch.setattr(
        app.st,
        "columns",
        lambda spec: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))],
    )
    monkeypatch.setattr(app.st, "button", lambda *args, **kwargs: False)
    monkeypatch.setattr(app.st, "metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "markdown", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "warning", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "success", lambda *args, **kwargs: None)
    monkeypatch.setattr(app.st, "caption", lambda *args, **kwargs: None)

    app.render_mode_continuous_sentence({})

    assert captured["key"] == "continuous_sentence_streamer"
    assert captured["mode"] == "sendrecv"
    assert captured["media_stream_constraints"] == {"video": True, "audio": False}
    assert captured["async_processing"] is True


def test_continuous_sentence_renderer_does_not_replace_existing_session(
    monkeypatch,
):
    import src.app as app

    existing = MagicMock()
    state = {"continuous_sentence_session": existing}

    monkeypatch.setattr(app.st, "session_state", state)

    result = app.get_or_create_continuous_sentence_session()

    assert result is existing


def test_continuous_sentence_module_has_no_streamlit_dependency():
    from pathlib import Path

    source = Path("src/ui/continuous_sentence.py").read_text()

    import ast

    tree = ast.parse(source)
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(
        module == "streamlit" or module.startswith("streamlit.")
        for module in imported_modules
    )
    assert not any(
        module == "streamlit_webrtc" or module.startswith("streamlit_webrtc.")
        for module in imported_modules
    )

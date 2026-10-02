"""Unit and integration tests for SIGNASSIST Phase 11 Text-to-Speech (TTS) Engine."""

import time
from unittest.mock import patch
import pytest
import torch
import torch.nn as nn

from src.models.confidence import ConfidenceOutput
from src.nlp import SentenceReconstructor, SentenceTranscript
from src.pipeline.boundary_detector import SignSegment
from src.tts import TTSEngine, TTSAudioOutput
from src.utils.config import AppConfig, TTSConfig


def test_tts_disabled_mode():
    """Verify enabled=False returns disabled TTSAudioOutput instantly without running engines."""
    cfg = TTSConfig(enabled=False)
    engine = TTSEngine(config=cfg)

    output = engine.synthesize("Hello world")

    assert isinstance(output, TTSAudioOutput)
    assert output.engine_used == "disabled"
    assert output.is_success is True
    assert output.audio_bytes == b""


def test_synthesize_text_and_metadata():
    """Verify synthesize generates audio bytes and populates metadata correctly."""
    cfg = TTSConfig(enabled=True)
    engine = TTSEngine(config=cfg)

    output = engine.synthesize("My name is Akshaya.")

    assert isinstance(output, TTSAudioOutput)
    assert output.is_success is True
    assert len(output.audio_bytes) > 0
    assert output.engine_used in ("pyttsx3", "gtts", "mock")
    assert output.text_spoken == "My name is Akshaya."


def test_gtts_timeout_and_mock_fallback():
    """Verify that when pyttsx3 and gTTS network calls fail or time out, engine falls back to mock."""
    engine = TTSEngine(config=TTSConfig(enabled=True), gtts_timeout=0.1)

    def _failing_pyttsx3(*args, **kwargs):
        raise RuntimeError("pyttsx3 driver unavailable")

    def _timing_out_gtts(*args, **kwargs):
        raise TimeoutError("gTTS network timeout simulated")

    with patch.object(engine, "_synthesize_pyttsx3", side_effect=_failing_pyttsx3):
        with patch.object(engine, "_synthesize_gtts", side_effect=_timing_out_gtts):
            output = engine.synthesize("Fallback test")

            assert isinstance(output, TTSAudioOutput)
            assert output.engine_used == "mock"
            assert output.is_success is True
            assert len(output.audio_bytes) > 0
            assert "Mock audio fallback used" in output.error_message


def test_internal_gtts_timeout_mechanism():
    """Verify that _synthesize_gtts raises TimeoutError within a bounded period when gTTS write_to_fp hangs."""
    engine = TTSEngine(config=TTSConfig(enabled=True), gtts_timeout=0.1)

    def _slow_gtts_write(self_gtts, fp):
        time.sleep(2.0)
        fp.write(b"mock audio bytes")

    start_t = time.time()
    with patch("gtts.gTTS.write_to_fp", new=_slow_gtts_write):
        with pytest.raises(TimeoutError, match="gTTS request timed out"):
            engine._synthesize_gtts("Test timeout")
    elapsed = time.time() - start_t

    # Must raise within bounded time (< 0.8s), not waiting for the 2.0s sleep
    assert elapsed < 0.8


def test_mock_engine_metadata():
    """Verify mock synthesis explicitly sets engine_used='mock'."""
    engine = TTSEngine(config=TTSConfig(enabled=True))
    output = engine._synthesize_mock("Test mock text")

    assert output.engine_used == "mock"
    assert output.is_success is True
    assert len(output.audio_bytes) > 0


def test_playback_backend_failure_graceful():
    """Verify playback backend failure (missing sound hardware) is handled gracefully without crashing."""
    engine = TTSEngine(config=TTSConfig(enabled=True))

    with patch("src.tts.tts_engine._play_audio_backend", side_effect=RuntimeError("No audio device")):
        output = engine.speak("Test speech playback failure", blocking=True)
        assert output.is_success is True


def test_synthesize_transcript_integration():
    """Verify synthesize_transcript extracts text from Phase 10 SentenceTranscript."""
    engine = TTSEngine(config=TTSConfig(enabled=True))
    transcript = SentenceTranscript(text="Thank you very much.")

    output = engine.synthesize_transcript(transcript)

    assert output.is_success is True
    assert output.text_spoken == "Thank you very much."
    assert len(output.audio_bytes) > 0


def test_single_sign_synthesis():
    """Verify synthesis works for a single sign gloss string."""
    engine = TTSEngine(config=TTSConfig(enabled=True))
    output = engine.synthesize("HELLO")

    assert output.is_success is True
    assert output.text_spoken == "HELLO"
    assert len(output.audio_bytes) > 0


def test_non_blocking_speak_thread():
    """Verify speak(blocking=False) returns substantially before a slow playback worker finishes."""
    engine = TTSEngine(config=TTSConfig(enabled=True))

    def _slow_playback(audio_bytes, mime_type, file_path=None):
        time.sleep(3.0)
        return True

    with patch("src.tts.tts_engine._play_audio_backend", new=_slow_playback):
        start_t = time.time()
        output = engine.speak("Testing non blocking playback", blocking=False)
        elapsed = time.time() - start_t

        assert output.is_success is True
        assert len(output.audio_bytes) > 0
        assert elapsed < 1.0  # Substantially before the 3.0s playback completes



def test_config_instantiation():
    """Verify TTSEngine.from_config() instantiates correctly."""
    app_cfg = AppConfig()
    engine = TTSEngine.from_config(app_cfg)

    assert isinstance(engine, TTSEngine)
    assert engine.enabled is True
    assert engine.volume == 1.0


def test_empty_transcript_handling():
    """Verify empty text/transcript returns graceful empty output."""
    engine = TTSEngine(config=TTSConfig(enabled=True))
    t_empty = SentenceTranscript(text="")

    output = engine.synthesize_transcript(t_empty)

    assert output.is_success is True
    assert output.engine_used == "none"
    assert output.audio_bytes == b""


def test_end_to_end_phase10_to_phase11_pipeline():
    """Integration test: Phase 8 prediction + Phase 9 segment -> Phase 10 Reconstructor -> Phase 11 TTS."""
    reconstructor = SentenceReconstructor(vocab_map={0: "THANK", 1: "YOU"})

    seg1 = SignSegment(0, 10, 11, 0.5, 0.8)
    seg2 = SignSegment(11, 20, 10, 0.5, 0.8)

    pred1 = ConfidenceOutput(
        predicted_class=torch.tensor(0, dtype=torch.int64),
        confidence=torch.tensor(0.9, dtype=torch.float32),
        top_k_indices=torch.tensor([0, 1], dtype=torch.int64),
        top_k_probabilities=torch.tensor([0.9, 0.1], dtype=torch.float32),
        is_accepted=torch.tensor(True, dtype=torch.bool),
        is_rejected=torch.tensor(False, dtype=torch.bool),
    )
    pred2 = ConfidenceOutput(
        predicted_class=torch.tensor(1, dtype=torch.int64),
        confidence=torch.tensor(0.9, dtype=torch.float32),
        top_k_indices=torch.tensor([1, 0], dtype=torch.int64),
        top_k_probabilities=torch.tensor([0.9, 0.1], dtype=torch.float32),
        is_accepted=torch.tensor(True, dtype=torch.bool),
        is_rejected=torch.tensor(False, dtype=torch.bool),
    )

    transcript = reconstructor.process_batch([seg1, seg2], [pred1, pred2])
    assert transcript.text == "Thank you."

    tts_engine = TTSEngine(config=TTSConfig(enabled=True))
    audio_output = tts_engine.synthesize_transcript(transcript)

    assert audio_output.is_success is True
    assert audio_output.text_spoken == "Thank you."
    assert len(audio_output.audio_bytes) > 0


def test_no_recurrent_modules():
    """Verify zero LSTM/GRU/RNN dependencies across all src/tts/ modules."""
    import src.tts as tts_pkg
    for attr_name in dir(tts_pkg):
        attr = getattr(tts_pkg, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module if hasattr(nn, 'Module') else object):
            for module in attr().modules():
                assert not isinstance(module, (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell))


def test_tts_audio_output_canonical_engine_used_attribute():
    """Regression test: verify TTSAudioOutput exposes canonical 'engine_used' attribute."""
    output = TTSAudioOutput(audio_bytes=b"123", engine_used="mock")
    assert hasattr(output, "engine_used")
    assert output.engine_used == "mock"


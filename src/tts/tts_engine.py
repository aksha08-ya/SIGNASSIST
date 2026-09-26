"""Text-to-Speech (TTS) Engine for SIGNASSIST Phase 11.

Implements voice synthesis supporting:
- Tier 1: Offline pyttsx3 voice synthesis engine
- Tier 2: Online gTTS fallback engine (with bounded timeout & network error handling)
- Tier 3: Headless mock WAV audio fallback for driverless/offline environments

Note on Metadata & Fallback Transparency:
The engine_used metadata field explicitly reports "pyttsx3", "gtts", "mock", "disabled", or "none"
so downstream components (and tests) can distinguish genuine speech from mock audio artifacts.
"""

from dataclasses import dataclass
import concurrent.futures
import io
import os
import subprocess
import tempfile
import threading
import time
from typing import Optional, Union
import wave

from src.nlp.types import SentenceTranscript
from src.utils.config import AppConfig, TTSConfig, load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Try optional imports
try:
    import pyttsx3

    _PYTTSX3_AVAILABLE = True
except Exception:
    _PYTTSX3_AVAILABLE = False
    pyttsx3 = None

try:
    from gtts import gTTS

    _GTTS_AVAILABLE = True
except Exception:
    _GTTS_AVAILABLE = False
    gTTS = None


@dataclass
class TTSAudioOutput:
    """Container holding synthesized audio data, metadata, and execution status."""

    audio_bytes: bytes
    file_path: Optional[str] = None
    mime_type: str = "audio/wav"
    duration_sec: float = 0.0
    text_spoken: str = ""
    engine_used: str = "offline"  # "pyttsx3", "gtts", "mock", "disabled", "none"
    is_success: bool = True
    error_message: Optional[str] = None


def _generate_mock_silent_wav(duration_sec: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generates a valid silent 16-bit PCM WAV byte stream for headless environments."""
    buf = io.BytesIO()
    n_samples = int(sample_rate * max(0.1, duration_sec))
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def _play_audio_backend(
    audio_bytes: bytes, mime_type: str, file_path: Optional[str] = None
) -> bool:
    """Attempts actual audio playback using available system backends.

    Returns True if playback was dispatched successfully, False if no backend/device exists.
    """
    if not audio_bytes:
        return False

    # 1. Windows winsound backend
    if os.name == "nt":
        try:
            import winsound

            if file_path and os.path.exists(file_path):
                winsound.PlaySound(file_path, winsound.SND_FILENAME)
                return True
            else:
                winsound.PlaySound(audio_bytes, winsound.SND_MEMORY)
                return True
        except Exception as e:
            logger.debug(f"winsound playback failed: {e}")

    # 2. Linux / WSL / Unix system playback backends (aplay, paplay, ffplay, mpv)
    players = [
        ["aplay", "-q"],
        ["paplay"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ["mpv", "--no-terminal"],
    ]

    tmp_path = file_path
    created_tmp = False
    if not tmp_path or not os.path.exists(tmp_path):
        ext = ".mp3" if "mp3" in mime_type else ".wav"
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
                created_tmp = True
        except Exception:
            return False

    try:
        for cmd in players:
            try:
                res = subprocess.run(
                    cmd + [tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
    finally:
        if created_tmp and tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return False


class TTSEngine:
    """Multi-tiered Text-to-Speech engine for SIGNASSIST."""

    def __init__(
        self,
        config: Optional[Union[TTSConfig, AppConfig]] = None,
        **kwargs,
    ) -> None:
        if isinstance(config, AppConfig):
            tts_cfg = config.ui.tts
        elif isinstance(config, TTSConfig):
            tts_cfg = config
        elif config is None:
            tts_cfg = TTSConfig()
        else:
            raise TypeError(
                f"Expected config to be TTSConfig or AppConfig, got {type(config)}"
            )

        self.enabled = kwargs.get("enabled", tts_cfg.enabled)
        self.auto_play = kwargs.get("auto_play", tts_cfg.auto_play)
        self.voice_gender = kwargs.get("voice_gender", tts_cfg.voice_gender)
        self.speech_rate = kwargs.get("speech_rate", tts_cfg.speech_rate)
        self.volume = float(kwargs.get("volume", getattr(tts_cfg, "volume", 1.0)))
        self.gtts_timeout = float(kwargs.get("gtts_timeout", 5.0))

    def synthesize(
        self,
        text: str,
        save_path: Optional[str] = None,
    ) -> TTSAudioOutput:
        """Synthesizes text into speech audio bytes and optional saved file.

        Degrades through:
        Tier 1: pyttsx3 (offline)
        Tier 2: gTTS (online, bounded timeout)
        Tier 3: Mock silent WAV audio (headless fallback)

        Args:
            text: Input text string to synthesize.
            save_path: Optional file path to save synthesized audio output.

        Returns:
            TTSAudioOutput structured container.
        """
        text = text.strip() if text else ""

        if not self.enabled:
            return TTSAudioOutput(
                audio_bytes=b"",
                file_path=None,
                mime_type="audio/wav",
                duration_sec=0.0,
                text_spoken=text,
                engine_used="disabled",
                is_success=True,
                error_message="TTS is disabled in configuration.",
            )

        if not text:
            return TTSAudioOutput(
                audio_bytes=b"",
                file_path=None,
                mime_type="audio/wav",
                duration_sec=0.0,
                text_spoken="",
                engine_used="none",
                is_success=True,
                error_message="Text is empty.",
            )

        errors = []

        # --- Tier 1: Offline pyttsx3 ---
        if _PYTTSX3_AVAILABLE:
            try:
                output = self._synthesize_pyttsx3(text, save_path=save_path)
                if output is not None and output.is_success:
                    return output
            except Exception as e:
                err_msg = f"Tier 1 (pyttsx3) failed: {e}"
                logger.warning(err_msg)
                errors.append(err_msg)

        # --- Tier 2: Online gTTS ---
        if _GTTS_AVAILABLE:
            try:
                output = self._synthesize_gtts(text, save_path=save_path)
                if output is not None and output.is_success:
                    return output
            except Exception as e:
                err_msg = f"Tier 2 (gTTS) failed: {e}"
                logger.warning(err_msg)
                errors.append(err_msg)

        # --- Tier 3: Headless Mock WAV Fallback ---
        logger.info("Engaging Tier 3 Headless Mock WAV Fallback.")
        return self._synthesize_mock(
            text, save_path=save_path, error_history=" | ".join(errors)
        )

    def _synthesize_pyttsx3(
        self, text: str, save_path: Optional[str] = None
    ) -> Optional[TTSAudioOutput]:
        """Tier 1 offline pyttsx3 synthesis."""
        engine = pyttsx3.init()

        # Set rate and volume
        try:
            rate = int(175 * self.speech_rate)
            engine.setProperty("rate", rate)
        except Exception:
            pass

        try:
            vol = max(0.0, min(1.0, float(self.volume)))
            engine.setProperty("volume", vol)
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_wav_path = tmp_file.name

        try:
            engine.save_to_file(text, tmp_wav_path)
            engine.runAndWait()

            if not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) == 0:
                raise RuntimeError("pyttsx3 generated empty audio file.")

            with open(tmp_wav_path, "rb") as f:
                audio_bytes = f.read()

            final_file_path = None
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                with open(save_path, "wb") as f_out:
                    f_out.write(audio_bytes)
                final_file_path = save_path

            words_count = len(text.split())
            approx_duration = max(0.5, words_count * (0.3 / max(0.1, self.speech_rate)))

            return TTSAudioOutput(
                audio_bytes=audio_bytes,
                file_path=final_file_path,
                mime_type="audio/wav",
                duration_sec=approx_duration,
                text_spoken=text,
                engine_used="pyttsx3",
                is_success=True,
                error_message=None,
            )
        finally:
            if os.path.exists(tmp_wav_path):
                try:
                    os.remove(tmp_wav_path)
                except Exception:
                    pass

    def _synthesize_gtts(
        self, text: str, save_path: Optional[str] = None
    ) -> Optional[TTSAudioOutput]:
        """Tier 2 online gTTS synthesis with bounded timeout."""

        def _gtts_fetch():
            tts = gTTS(text=text, lang="en", tld="com")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            return fp.getvalue()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_gtts_fetch)
            audio_bytes = future.result(timeout=self.gtts_timeout)
            executor.shutdown(wait=False)
        except concurrent.futures.TimeoutError:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(f"gTTS request timed out after {self.gtts_timeout}s.")
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

        if not audio_bytes:
            raise RuntimeError("gTTS returned empty audio stream.")

        final_file_path = None
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "wb") as f_out:
                f_out.write(audio_bytes)
            final_file_path = save_path

        words_count = len(text.split())
        approx_duration = max(0.5, words_count * (0.3 / max(0.1, self.speech_rate)))

        return TTSAudioOutput(
            audio_bytes=audio_bytes,
            file_path=final_file_path,
            mime_type="audio/mp3",
            duration_sec=approx_duration,
            text_spoken=text,
            engine_used="gtts",
            is_success=True,
            error_message=None,
        )

    def _synthesize_mock(
        self,
        text: str,
        save_path: Optional[str] = None,
        error_history: Optional[str] = None,
    ) -> TTSAudioOutput:
        """Tier 3 mock silent WAV fallback for headless/driverless environments."""
        words_count = len(text.split()) if text else 1
        duration = max(0.5, words_count * 0.3)
        mock_bytes = _generate_mock_silent_wav(duration_sec=duration)

        final_file_path = None
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "wb") as f_out:
                f_out.write(mock_bytes)
            final_file_path = save_path

        err_msg = (
            f"Mock audio fallback used. Previous errors: {error_history}"
            if error_history
            else "Mock audio fallback used."
        )

        return TTSAudioOutput(
            audio_bytes=mock_bytes,
            file_path=final_file_path,
            mime_type="audio/wav",
            duration_sec=duration,
            text_spoken=text,
            engine_used="mock",
            is_success=True,
            error_message=err_msg,
        )

    def speak(
        self,
        text: str,
        blocking: bool = False,
    ) -> TTSAudioOutput:
        """Plays audio for the provided text. Non-blocking by default using daemon thread."""
        output = self.synthesize(text)
        if not self.enabled or not output.audio_bytes or output.engine_used == "disabled":
            return output

        def _play_worker() -> None:
            try:
                _play_audio_backend(
                    audio_bytes=output.audio_bytes,
                    mime_type=output.mime_type,
                    file_path=output.file_path,
                )
            except Exception as e:
                logger.warning(f"Audio playback worker exception: {e}")

        if blocking:
            _play_worker()
        else:
            thread = threading.Thread(target=_play_worker, daemon=True)
            thread.start()

        return output

    def synthesize_transcript(
        self,
        transcript: SentenceTranscript,
        save_path: Optional[str] = None,
    ) -> TTSAudioOutput:
        """Synthesizes speech from a Phase 10 SentenceTranscript."""
        text = transcript.text if transcript else ""
        return self.synthesize(text, save_path=save_path)

    def speak_transcript(
        self,
        transcript: SentenceTranscript,
        blocking: bool = False,
    ) -> TTSAudioOutput:
        """Plays speech from a Phase 10 SentenceTranscript."""
        text = transcript.text if transcript else ""
        return self.speak(text, blocking=blocking)

    @classmethod
    def from_config(cls, config: Optional[AppConfig] = None) -> "TTSEngine":
        """Factory method to instantiate TTSEngine from AppConfig."""
        if config is None:
            config = load_config()
        return cls(config=config)

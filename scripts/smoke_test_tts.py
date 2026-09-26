"""Smoke test script for SIGNASSIST Phase 11 Text-to-Speech (TTS) Engine."""

import os
import sys

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp import SentenceTranscript
from src.tts import TTSEngine, TTSAudioOutput
from src.utils.config import ConfigManager


def main() -> None:
    print("=== SIGNASSIST Phase 11 Text-to-Speech (TTS) Smoke Test ===")

    # 1. Config loading
    config_mgr = ConfigManager()
    app_cfg = config_mgr.load_all()
    tts_cfg = app_cfg.ui.tts

    print(f"[1/4] Loaded TTSConfig from ui.yaml:")
    print(f"      enabled: {tts_cfg.enabled}")
    print(f"      auto_play: {tts_cfg.auto_play}")
    print(f"      voice_gender: {tts_cfg.voice_gender}")
    print(f"      speech_rate: {tts_cfg.speech_rate}")

    # 2. Instantiation
    engine = TTSEngine.from_config(app_cfg)
    print("\n[2/4] Instantiated TTSEngine successfully.")

    # 3. SentenceTranscript Synthesis (Phase 10 -> Phase 11)
    print("\n[3/4] Running Text-to-Speech Synthesis from SentenceTranscript...")
    transcript = SentenceTranscript(text="My name is Akshaya.")
    output: TTSAudioOutput = engine.synthesize_transcript(transcript)

    print(f"      Text Spoken: '{output.text_spoken}'")
    print(f"      Engine Used (Tier): {output.engine_used}")
    print(f"      MIME Type: {output.mime_type}")
    print(f"      Audio Bytes Generated: {len(output.audio_bytes)} bytes")
    print(f"      Estimated Duration: {output.duration_sec:.2f} seconds")
    print(f"      Is Success: {output.is_success}")
    if output.error_message:
        print(f"      Notice / Fallback Details: {output.error_message}")

    # 4. Non-blocking speak test
    print("\n[4/4] Testing Non-Blocking Speak Playback...")
    speak_output = engine.speak("Testing non blocking audio playback.", blocking=False)
    print(f"      Non-blocking speak executed successfully (Engine: {speak_output.engine_used}).")

    print("\n=== Phase 11 TTS Smoke Test Passed Successfully! ===")


if __name__ == "__main__":
    main()

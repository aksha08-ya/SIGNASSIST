#""Tests for hardened Text to ISL- UI and missing token handling."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex
from src.nlp.text_to_isl import DemonstrationSource, TextToISLConverter
from src.ui.services import convert_text_to_isl


def test_convert_text_to_isl_missing_tokens() -> None:
    empty_dict = ISLDictionaryIndex(entries=[])
    res = convert_text_to_isl(text="hello world", dictionary=empty_dict)

    assert res.dictionary_count == 0
    assert res.unknown_count == 2
    assert "hello" in res.skipped_tokens
    assert "world" in res.skipped_tokens
    assert res.output_video_path is None



def test_convert_text_to_isl_partial_translation() -> None:
    entry = ISLDictionaryEntry(
        gloss="HELLO",
        video_path="videos/hello.mp4",
        aliases=["hello"],
    )
    dict_idx = ISLDictionaryIndex(entries=[entry])
    res = convert_text_to_isl(text="hello world", dictionary=dict_idx)

    assert res.dictionary_count == 1
    assert res.unknown_count == 1
    assert "world" in res.skipped_tokens

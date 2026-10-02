"""Comprehensive E2E hardening unit tests for SIGNASSIST Text-to-ISL pipeline."""

from pathlib import Path
import tempfile
import pytest

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex
from src.nlp.text_to_isl import DemonstrationSource, TextToISLConverter
from src.ui.services import convert_text_to_isl


def test_dictionary_video_existence_validation(tmp_path: Path) -> None:
    """Tests that ISLDictionaryIndex.load with validate_paths=True filters out entries with missing video files."""
    dict_dir = tmp_path / "dict"
    dict_dir.mkdir()
    videos_dir = dict_dir / "videos"
    videos_dir.mkdir()

    # Create one real video file and one missing
    real_video = videos_dir / "real.mp4"
    real_video.write_bytes(b"dummy video data")

    index_json = dict_dir / "index.json"
    index_data = {
        "entries": [
            {
                "gloss": "REAL",
                "video_path": "videos/real.mp4",
                "aliases": ["real"],
            },
            {
                "gloss": "MISSING",
                "video_path": "videos/missing.mp4",
                "aliases": ["missing"],
            },
        ]
    }
    import json
    index_json.write_text(json.dumps(index_data))

    index = ISLDictionaryIndex.load(dict_dir, validate_paths=True)
    assert index.contains("real")
    assert not index.contains("missing")
    assert len(index.entries()) == 1
    assert index.entries()[0].gloss == "REAL"


def test_word_lookup_normalized() -> None:
    """Tests single word lookup with case-folding, whitespace, and punctuation stripping."""
    entry = ISLDictionaryEntry(
        gloss="10. Plane",
        video_path="videos/class_10__plane_MVI_3114.MOV",
        aliases=["plane", "10. plane", "10 plane"],
        category="Means_of_Transportation",
        duration_sec=2.4,
    )
    index = ISLDictionaryIndex(entries=[entry])
    converter = TextToISLConverter(dictionary=index)

    res = converter.convert("  plane!  ")
    assert len(res.demonstrations) == 1
    demo = res.demonstrations[0]
    assert demo.source == DemonstrationSource.DICTIONARY
    assert demo.dictionary_gloss == "10. Plane"
    assert demo.category == "Means_of_Transportation"
    assert demo.duration_sec == 2.4


def test_phrase_lookup_greedy_matching() -> None:
    """Tests greedy multi-word phrase matching over single word tokens."""
    phrase_entry = ISLDictionaryEntry(
        gloss="55. Good Night",
        video_path="videos/good_night.mp4",
        aliases=["good night"],
    )
    word_entry = ISLDictionaryEntry(
        gloss="GOOD",
        video_path="videos/good.mp4",
        aliases=["good"],
    )
    index = ISLDictionaryIndex(entries=[phrase_entry, word_entry])
    converter = TextToISLConverter(dictionary=index)

    res = converter.convert("have a good night")
    demos = res.demonstrations
    # "have", "a" are unknown, "good night" is matched as a phrase
    matched = [d for d in demos if d.source == DemonstrationSource.DICTIONARY]
    assert len(matched) == 1
    assert matched[0].dictionary_gloss == "55. Good Night"


def test_unavailable_words_handling() -> None:
    """Tests that unsupported words are marked UNKNOWN with no fingerspelling pretending when alphabet assets are absent."""
    empty_index = ISLDictionaryIndex(entries=[])
    converter = TextToISLConverter(dictionary=empty_index, alphabet_video_mapping=None)

    res = converter.convert("unsupportedword")
    assert len(res.demonstrations) == 1
    demo = res.demonstrations[0]
    assert demo.source == DemonstrationSource.UNKNOWN
    assert demo.video_path is None
    assert demo.alphabet_video_paths == []
    assert demo.dictionary_gloss is None


def test_mixed_supported_and_unsupported_input() -> None:
    """Tests input sequence with mixed dictionary-matched and unsupported tokens."""
    entry = ISLDictionaryEntry(
        gloss="5. Cow",
        video_path="videos/cow.mp4",
        aliases=["cow"],
    )
    index = ISLDictionaryIndex(entries=[entry])
    res = convert_text_to_isl(text="cow xyz999unsupported", dictionary=index)

    assert res.dictionary_count == 1
    assert res.unknown_count == 1
    assert "xyz999unsupported" in res.skipped_tokens


def test_real_isl_dictionary_asset_provenance() -> None:
    """Tests that real repository ISL dictionary assets exist and have valid TRAIN provenance."""
    dict_dir = Path("data/external/isl_dictionary")
    if not dict_dir.exists() or not (dict_dir / "index.json").exists():
        pytest.skip("ISL dictionary assets missing")

    index = ISLDictionaryIndex.load(dict_dir, validate_paths=True)
    assert len(index.entries()) >= 42

    for entry in index.entries():
        abs_v_path = dict_dir / entry.video_path
        assert abs_v_path.exists(), f"Video file missing: {abs_v_path}"
        assert abs_v_path.stat().st_size > 0, f"Video file zero size: {abs_v_path}"
        assert "videos/" in entry.video_path

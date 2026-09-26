"""Unit tests for SIGNASSIST Phase 15 ISL Dictionary and Index infrastructure."""

import json
from pathlib import Path
import pytest

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex


def test_empty_index_loading(tmp_path: Path) -> None:
    """Tests loading an empty dictionary index from JSON and default empty index instantiation."""
    empty_index = ISLDictionaryIndex()
    assert len(empty_index.entries()) == 0
    assert empty_index.lookup("HELLO") is None
    assert not empty_index.contains("HELLO")

    # Load empty index file
    index_file = tmp_path / "index.json"
    index_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "description": "Empty test index",
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    loaded_index = ISLDictionaryIndex.load(index_file)
    assert len(loaded_index.entries()) == 0
    assert loaded_index.lookup("HELLO") is None

    # Load from directory path
    dir_loaded = ISLDictionaryIndex.load(tmp_path)
    assert len(dir_loaded.entries()) == 0


def test_canonical_and_alias_lookup() -> None:
    """Tests looking up entries by canonical gloss and registered aliases."""
    entry_hello = ISLDictionaryEntry(
        gloss="HELLO",
        video_path="videos/hello.mp4",
        aliases=["hi", "greetings"],
        category="greetings",
    )
    entry_world = ISLDictionaryEntry(
        gloss="WORLD",
        video_path="videos/world.mp4",
        aliases=["earth", "globe"],
        category="general",
    )

    index = ISLDictionaryIndex(entries=[entry_hello, entry_world])

    # Canonical lookup
    res_hello = index.lookup("HELLO")
    assert res_hello is not None
    assert res_hello.gloss == "HELLO"
    assert res_hello.video_path == "videos/hello.mp4"

    # Alias lookup
    res_hi = index.lookup("hi")
    assert res_hi is not None
    assert res_hi.gloss == "HELLO"

    res_greetings = index.lookup("greetings")
    assert res_greetings is not None
    assert res_greetings.gloss == "HELLO"

    res_earth = index.lookup("earth")
    assert res_earth is not None
    assert res_earth.gloss == "WORLD"


def test_case_and_whitespace_normalization() -> None:
    """Tests that leading/trailing whitespace, case variations, and internal spaces normalize correctly."""
    entry = ISLDictionaryEntry(
        gloss="GOOD MORNING",
        video_path="videos/good_morning.mp4",
        aliases=["morning greet", "morning"],
    )
    index = ISLDictionaryIndex(entries=[entry])

    # Search with mixed case and leading/trailing/multiple internal whitespace
    assert index.contains("  gOoD   mOrNiNg  ")
    res = index.lookup("  gOoD   mOrNiNg  ")
    assert res is not None
    assert res.gloss == "GOOD MORNING"

    # Alias with case/whitespace variations
    assert index.contains("   MoRnInG   GReeT  ")
    res_alias = index.lookup("   MoRnInG   GReeT  ")
    assert res_alias is not None
    assert res_alias.gloss == "GOOD MORNING"


def test_lookup_many_ordering() -> None:
    """Tests lookup_many returning entries or None in exact input sequence."""
    e1 = ISLDictionaryEntry(gloss="ALPHA", video_path="videos/alpha.mp4", aliases=["a"])
    e2 = ISLDictionaryEntry(gloss="BETA", video_path="videos/beta.mp4", aliases=["b"])
    index = ISLDictionaryIndex(entries=[e1, e2])

    query_terms = ["a", "MISSING", "BETA", "ALPHA", "unknown", "b"]
    results = index.lookup_many(query_terms)

    assert len(results) == len(query_terms)
    assert results[0] == e1
    assert results[1] is None
    assert results[2] == e2
    assert results[3] == e1
    assert results[4] is None
    assert results[5] == e2


def test_missing_term_behavior() -> None:
    """Tests lookup behavior for nonexistent terms, None, empty string, and non-string inputs."""
    entry = ISLDictionaryEntry(gloss="YES", video_path="videos/yes.mp4")
    index = ISLDictionaryIndex(entries=[entry])

    assert index.lookup("NO") is None
    assert not index.contains("NO")
    assert "NO" not in index

    assert index.lookup("") is None
    assert index.lookup(None) is None  # type: ignore[arg-type]
    assert index.lookup(123) is None  # type: ignore[arg-type]


def test_duplicate_canonical_detection() -> None:
    """Tests rejection of duplicate canonical glosses during index construction."""
    e1 = ISLDictionaryEntry(gloss="CAT", video_path="videos/cat.mp4")
    e2 = ISLDictionaryEntry(gloss="cat", video_path="videos/cat2.mp4")  # Normalized duplicate

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        ISLDictionaryIndex(entries=[e1, e2])


def test_conflicting_alias_detection() -> None:
    """Tests rejection of alias conflicts across different entries."""
    # Entry 1 maps alias "wave" to HELLO
    e1 = ISLDictionaryEntry(gloss="HELLO", video_path="videos/hello.mp4", aliases=["wave"])
    # Entry 2 maps alias "wave" to BYE
    e2 = ISLDictionaryEntry(gloss="BYE", video_path="videos/bye.mp4", aliases=["wave"])

    with pytest.raises(ValueError, match="[Cc]onflict"):
        ISLDictionaryIndex(entries=[e1, e2])

    # Alias of Entry 2 conflicts with canonical gloss of Entry 1
    e3 = ISLDictionaryEntry(gloss="GREET", video_path="videos/greet.mp4", aliases=["HELLO"])
    with pytest.raises(ValueError, match="[Cc]onflict"):
        ISLDictionaryIndex(entries=[e1, e3])


def test_invalid_empty_fields() -> None:
    """Tests validation error on empty gloss or empty video_path."""
    with pytest.raises(ValueError, match="[Gg]loss"):
        ISLDictionaryEntry(gloss="", video_path="videos/test.mp4")

    with pytest.raises(ValueError, match="[Gg]loss"):
        ISLDictionaryEntry(gloss="   ", video_path="videos/test.mp4")

    with pytest.raises(ValueError, match="video_path"):
        ISLDictionaryEntry(gloss="HELLO", video_path="")

    with pytest.raises(ValueError, match="video_path"):
        ISLDictionaryEntry(gloss="HELLO", video_path="   ")


def test_path_traversal_rejection() -> None:
    """Tests rejection of path traversal sequences and absolute paths in video_path."""
    traversal_paths = [
        "../videos/hello.mp4",
        "videos/../../hello.mp4",
        "/absolute/path/hello.mp4",
        "\\absolute\\windows\\hello.mp4",
        "C:\\videos\\hello.mp4",
        "videos/../hello.mp4",
    ]

    for bad_path in traversal_paths:
        with pytest.raises(ValueError, match="[Pp]ath traversal|[Aa]bsolute path"):
            ISLDictionaryEntry(gloss="HELLO", video_path=bad_path)


def test_relative_video_paths_allowed() -> None:
    """Tests that valid relative video paths are accepted."""
    valid_paths = [
        "videos/hello.mp4",
        "isl_clips/chapter1/sign_01.mp4",
        "hello.webm",
        "assets/dict/sign_test.mov",
    ]

    for valid_path in valid_paths:
        entry = ISLDictionaryEntry(gloss="TEST", video_path=valid_path)
        assert entry.video_path == valid_path


def test_missing_video_files_allowed_at_index_load_time(tmp_path: Path) -> None:
    """Tests that entries referencing non-existent video files are valid during index creation and load."""
    non_existent_path = "videos/does_not_exist_yet_12345.mp4"
    assert not (tmp_path / non_existent_path).exists()

    entry = ISLDictionaryEntry(gloss="FUTURE", video_path=non_existent_path)
    index = ISLDictionaryIndex(entries=[entry])
    assert index.lookup("FUTURE") is not None

    # Load from index JSON file
    index_file = tmp_path / "index.json"
    index_file.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "description": "Test index with pending video files",
                "entries": [
                    {
                        "gloss": "FUTURE",
                        "video_path": non_existent_path,
                        "aliases": ["later"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded_index = ISLDictionaryIndex.load(index_file)
    assert loaded_index.lookup("FUTURE") is not None
    assert loaded_index.lookup("FUTURE").video_path == non_existent_path

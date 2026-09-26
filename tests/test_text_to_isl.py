"""Unit tests for SIGNASSIST Phase 15 Block 2 TextToISLConverter."""

import pytest

from src.nlp import (
    DemonstrationSource,
    ISLDemonstration,
    ISLDemonstrationSequence,
    ISLDictionaryEntry,
    ISLDictionaryIndex,
    TextToISLConverter,
)


@pytest.fixture
def sample_dictionary() -> ISLDictionaryIndex:
    """Fixture providing a sample ISLDictionaryIndex for testing."""
    e1 = ISLDictionaryEntry(
        gloss="HELLO",
        video_path="videos/hello.mp4",
        aliases=["hi", "greetings"],
        description="Wave greeting",
    )
    e2 = ISLDictionaryEntry(
        gloss="WORLD",
        video_path="videos/world.mp4",
        aliases=["earth"],
    )
    e3 = ISLDictionaryEntry(
        gloss="THANK YOU",
        video_path="videos/thank_you.mp4",
        aliases=["thanks"],
    )
    return ISLDictionaryIndex(entries=[e1, e2, e3])


@pytest.fixture
def alphabet_mapping() -> dict[str, str]:
    """Fixture providing a full A-Z alphabet video mapping."""
    return {chr(c): f"videos/alphabet/{chr(c)}.mp4" for c in range(ord("A"), ord("Z") + 1)}


def test_dictionary_word_lookup(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests exact dictionary term lookup."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert("HELLO WORLD")

    assert seq.original_text == "HELLO WORLD"
    assert seq.dictionary_count == 2
    assert seq.fingerspell_count == 0
    assert seq.unknown_count == 0
    assert len(seq.demonstrations) == 2

    d1, d2 = seq.demonstrations
    assert d1.token == "HELLO"
    assert d1.source == DemonstrationSource.DICTIONARY
    assert d1.video_path == "videos/hello.mp4"
    assert d1.alphabet_video_paths == []
    assert d1.dictionary_gloss == "HELLO"

    assert d2.token == "WORLD"
    assert d2.source == DemonstrationSource.DICTIONARY
    assert d2.video_path == "videos/world.mp4"
    assert d2.alphabet_video_paths == []


def test_case_insensitive_lookup(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests case-folding lookup for dictionary words and aliases."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert("hELLo hI ThAnKs")

    assert seq.dictionary_count == 3
    assert seq.demonstrations[0].dictionary_gloss == "HELLO"
    assert seq.demonstrations[0].alphabet_video_paths == []
    assert seq.demonstrations[1].dictionary_gloss == "HELLO"
    assert seq.demonstrations[1].alphabet_video_paths == []
    assert seq.demonstrations[2].dictionary_gloss == "THANK YOU"
    assert seq.demonstrations[2].alphabet_video_paths == []


def test_whitespace_and_punctuation_handling(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests whitespace collapsing and stripping punctuation attached to words."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert("  hello,   world!  ")

    assert seq.original_text == "  hello,   world!  "
    assert seq.dictionary_count == 2
    assert len(seq.demonstrations) == 2
    assert seq.demonstrations[0].token == "hello"
    assert seq.demonstrations[0].alphabet_video_paths == []
    assert seq.demonstrations[1].token == "world"
    assert seq.demonstrations[1].alphabet_video_paths == []


def test_mixed_dictionary_and_fallback_sentence(
    sample_dictionary: ISLDictionaryIndex, alphabet_mapping: dict[str, str]
) -> None:
    """Tests a sentence containing dictionary words, fingerspelled fallback words, and missing terms."""
    converter = TextToISLConverter(
        dictionary=sample_dictionary, alphabet_video_mapping=alphabet_mapping
    )
    seq = converter.convert("Hello python world")

    assert seq.dictionary_count == 2
    assert seq.fingerspell_count == 1
    assert seq.unknown_count == 0

    d_python = seq.demonstrations[1]
    assert d_python.token == "python"
    assert d_python.source == DemonstrationSource.FINGERSPELL
    assert d_python.characters == ["P", "Y", "T", "H", "O", "N"]
    assert d_python.video_path is None
    assert d_python.alphabet_video_paths == [
        "videos/alphabet/P.mp4",
        "videos/alphabet/Y.mp4",
        "videos/alphabet/T.mp4",
        "videos/alphabet/H.mp4",
        "videos/alphabet/O.mp4",
        "videos/alphabet/N.mp4",
    ]


def test_pure_fingerspelling_fallback(
    sample_dictionary: ISLDictionaryIndex, alphabet_mapping: dict[str, str]
) -> None:
    """Tests pure fingerspelling fallback when words are not in dictionary."""
    converter = TextToISLConverter(
        dictionary=sample_dictionary, alphabet_video_mapping=alphabet_mapping
    )
    seq = converter.convert("Akshaya Deepmind")

    assert seq.dictionary_count == 0
    assert seq.fingerspell_count == 2
    assert seq.unknown_count == 0

    d1, d2 = seq.demonstrations
    assert d1.characters == ["A", "K", "S", "H", "A", "Y", "A"]
    assert d1.video_path is None
    assert d1.alphabet_video_paths == [
        "videos/alphabet/A.mp4",
        "videos/alphabet/K.mp4",
        "videos/alphabet/S.mp4",
        "videos/alphabet/H.mp4",
        "videos/alphabet/A.mp4",
        "videos/alphabet/Y.mp4",
        "videos/alphabet/A.mp4",
    ]

    assert d2.characters == ["D", "E", "E", "P", "M", "I", "N", "D"]
    assert d2.video_path is None
    assert d2.alphabet_video_paths == [
        "videos/alphabet/D.mp4",
        "videos/alphabet/E.mp4",
        "videos/alphabet/E.mp4",
        "videos/alphabet/P.mp4",
        "videos/alphabet/M.mp4",
        "videos/alphabet/I.mp4",
        "videos/alphabet/N.mp4",
        "videos/alphabet/D.mp4",
    ]


def test_missing_alphabet_mapping_to_unknown(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests that missing alphabet mapping marks non-dictionary words as UNKNOWN with empty alphabet_video_paths."""
    # Partial mapping missing 'X', 'Y', 'Z'
    partial_mapping = {"A": "videos/A.mp4", "B": "videos/B.mp4"}
    converter = TextToISLConverter(
        dictionary=sample_dictionary, alphabet_video_mapping=partial_mapping
    )

    seq = converter.convert("XYZ")
    assert seq.dictionary_count == 0
    assert seq.fingerspell_count == 0
    assert seq.unknown_count == 1

    demo = seq.demonstrations[0]
    assert demo.source == DemonstrationSource.UNKNOWN
    assert demo.characters == ["X", "Y", "Z"]
    assert demo.video_path is None
    assert demo.alphabet_video_paths == []

    # Converter with NO alphabet mapping
    no_alpha_converter = TextToISLConverter(dictionary=sample_dictionary)
    seq_no_alpha = no_alpha_converter.convert("python")
    assert seq_no_alpha.unknown_count == 1
    assert seq_no_alpha.demonstrations[0].source == DemonstrationSource.UNKNOWN
    assert seq_no_alpha.demonstrations[0].alphabet_video_paths == []


def test_empty_and_whitespace_input(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests that empty and whitespace-only inputs return empty sequence without throwing."""
    converter = TextToISLConverter(dictionary=sample_dictionary)

    seq_empty = converter.convert("")
    assert seq_empty.original_text == ""
    assert len(seq_empty.demonstrations) == 0

    seq_spaces = converter.convert("   \n\t  ")
    assert seq_spaces.original_text == "   \n\t  "
    assert len(seq_spaces.demonstrations) == 0


def test_punctuation_only_input(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests that punctuation-only input produces zero video demonstrations."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert("!... ??? ,,,")

    assert seq.original_text == "!... ??? ,,,"
    assert len(seq.demonstrations) == 0
    assert seq.dictionary_count == 0
    assert seq.fingerspell_count == 0
    assert seq.unknown_count == 0


def test_deterministic_repeated_conversion(
    sample_dictionary: ISLDictionaryIndex, alphabet_mapping: dict[str, str]
) -> None:
    """Tests that identical input produces identical output deterministically."""
    converter = TextToISLConverter(
        dictionary=sample_dictionary, alphabet_video_mapping=alphabet_mapping
    )
    text = "Hello, world! Python testing."

    seq1 = converter.convert(text)
    seq2 = converter.convert(text)

    assert seq1 == seq2


def test_no_fabricated_video_path(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests that fallback demonstrations do NOT fabricate concatenated or fake video paths."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert("UNRECOGNIZEDWORD")

    demo = seq.demonstrations[0]
    assert demo.video_path is None
    assert demo.alphabet_video_paths == []


def test_security_and_path_provenance(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests that video paths only originate from verified dictionary/alphabet mappings and user input paths are rejected/ignored."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert("../../../etc/passwd C:\\Windows\\system32")

    for demo in seq.demonstrations:
        assert demo.video_path is None
        assert demo.alphabet_video_paths == []


def test_convert_tokens_helper(sample_dictionary: ISLDictionaryIndex) -> None:
    """Tests convert_tokens helper method."""
    converter = TextToISLConverter(dictionary=sample_dictionary)
    seq = converter.convert_tokens(["hello", "world"])

    assert seq.dictionary_count == 2
    assert seq.original_text == "hello world"


def test_has_dictionary_entry_and_get_supported_terms(
    sample_dictionary: ISLDictionaryIndex,
) -> None:
    """Tests public metadata query methods."""
    converter = TextToISLConverter(dictionary=sample_dictionary)

    assert converter.has_dictionary_entry("hello")
    assert converter.has_dictionary_entry("WORLD")
    assert converter.has_dictionary_entry("thanks")
    assert not converter.has_dictionary_entry("nonexistent")

    terms = converter.get_supported_dictionary_terms()
    assert "HELLO" in terms
    assert "WORLD" in terms
    assert "THANK YOU" in terms


def test_public_package_exports() -> None:
    """Tests that all required symbols are exported from src.nlp."""
    from src.nlp import (
        DemonstrationSource,
        ISLDemonstration,
        ISLDemonstrationSequence,
        TextToISLConverter,
    )

    assert DemonstrationSource.DICTIONARY == "DICTIONARY"
    assert DemonstrationSource.FINGERSPELL == "FINGERSPELL"
    assert DemonstrationSource.UNKNOWN == "UNKNOWN"
    assert ISLDemonstration is not None
    assert ISLDemonstrationSequence is not None
    assert TextToISLConverter is not None

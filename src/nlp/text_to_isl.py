"""Text to ISL Converter module for SIGNASSIST Phase 15 Block 2.

Translates natural language text into a sequence of ISL demonstrations,
prioritizing dictionary-based sign video lookups with structured alphabet
fingerspelling fallback and unknown word tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Dict, List, Optional, Union

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex


class DemonstrationSource(str, Enum):
    """Origin source of an ISL demonstration."""

    DICTIONARY = "DICTIONARY"
    FINGERSPELL = "FINGERSPELL"
    UNKNOWN = "UNKNOWN"


@dataclass
class ISLDemonstration:
    """Represents a single ISL sign demonstration in a translated sequence.

    Attributes:
        token: Original input word/token.
        normalized_token: Normalized lookup token.
        source: Provenance source (DICTIONARY, FINGERSPELL, UNKNOWN).
        video_path: Relative path to dictionary video clip (or None).
        characters: Uppercase A-Z characters for fingerspelling fallback.
        alphabet_video_paths: Ordered list of alphabet video clip paths for fingerspelling.
        dictionary_gloss: Canonical ISL dictionary gloss if matched.
        description: Description/context from dictionary entry if matched.
    """

    token: str
    normalized_token: str
    source: DemonstrationSource
    video_path: Optional[str] = None
    characters: List[str] = field(default_factory=list)
    alphabet_video_paths: List[str] = field(default_factory=list)
    dictionary_gloss: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.source, str) and not isinstance(self.source, DemonstrationSource):
            self.source = DemonstrationSource(self.source)


@dataclass
class ISLDemonstrationSequence:
    """Structured result container for a text-to-ISL conversion.

    Attributes:
        original_text: Exact input text string.
        normalized_text: Cleaned and normalized text representation.
        demonstrations: Ordered list of ISL demonstrations.
        dictionary_count: Count of demonstrations sourced from dictionary.
        fingerspell_count: Count of demonstrations using fingerspelling fallback.
        unknown_count: Count of demonstrations marked as unknown.
    """

    original_text: str
    normalized_text: str
    demonstrations: List[ISLDemonstration] = field(default_factory=list)
    dictionary_count: int = 0
    fingerspell_count: int = 0
    unknown_count: int = 0

    def __post_init__(self) -> None:
        self.dictionary_count = sum(
            1 for d in self.demonstrations if d.source == DemonstrationSource.DICTIONARY
        )
        self.fingerspell_count = sum(
            1 for d in self.demonstrations if d.source == DemonstrationSource.FINGERSPELL
        )
        self.unknown_count = sum(
            1 for d in self.demonstrations if d.source == DemonstrationSource.UNKNOWN
        )


class TextToISLConverter:
    """Converts natural language text into structured ISL demonstration sequences."""

    PUNCTUATION_REGEX = re.compile(
        r"^[\s\.\,\!\?\;\:\'\"\( \)\{\}\[\]\-]+|[\s\.\,\!\?\;\:\'\"\( \)\{\}\[\]\-]+$"
    )

    def __init__(
        self,
        dictionary: ISLDictionaryIndex,
        alphabet_video_mapping: Optional[Dict[str, str]] = None,
        punctuation_policy: Optional[str] = "strip",
    ) -> None:
        if not isinstance(dictionary, ISLDictionaryIndex):
            raise TypeError(f"Expected ISLDictionaryIndex instance, got {type(dictionary)}")
        self.dictionary = dictionary

        self.alphabet_video_mapping: Optional[Dict[str, str]] = None
        if alphabet_video_mapping is not None:
            self.alphabet_video_mapping = {
                k.upper(): v for k, v in alphabet_video_mapping.items() if isinstance(k, str)
            }

        self.punctuation_policy = punctuation_policy or "strip"

    def _strip_punctuation(self, word: str) -> str:
        """Strips leading and trailing punctuation marks from a token."""
        return self.PUNCTUATION_REGEX.sub("", word)

    def convert(self, text: str) -> ISLDemonstrationSequence:
        """Translates raw input text into an ISLDemonstrationSequence."""
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        original_text = text
        cleaned_text = text.strip()

        if not cleaned_text:
            return ISLDemonstrationSequence(
                original_text=original_text,
                normalized_text="",
                demonstrations=[],
            )

        raw_tokens = cleaned_text.split()
        return self._convert_raw_tokens(original_text, raw_tokens)

    def convert_tokens(self, tokens: List[str]) -> ISLDemonstrationSequence:
        """Translates a pre-tokenized list of words into an ISLDemonstrationSequence."""
        if not tokens:
            return ISLDemonstrationSequence(
                original_text="",
                normalized_text="",
                demonstrations=[],
            )
        original_text = " ".join(str(t) for t in tokens)
        return self._convert_raw_tokens(original_text, [str(t) for t in tokens])

    def _convert_raw_tokens(
        self, original_text: str, raw_tokens: List[str]
    ) -> ISLDemonstrationSequence:
        demonstrations: List[ISLDemonstration] = []
        normalized_words: List[str] = []

        for raw_token in raw_tokens:
            clean_word = self._strip_punctuation(raw_token)
            if not clean_word:
                # Entire token was punctuation, skip creating demonstration clip
                continue

            norm_token = self.dictionary.normalize_term(clean_word)
            normalized_words.append(norm_token)

            # 1. Dictionary Lookup
            entry = self.dictionary.lookup(clean_word)
            if entry is not None:
                demo = ISLDemonstration(
                    token=clean_word,
                    normalized_token=norm_token,
                    source=DemonstrationSource.DICTIONARY,
                    video_path=entry.video_path,
                    characters=[],
                    alphabet_video_paths=[],
                    dictionary_gloss=entry.gloss,
                    description=entry.description,
                )
                demonstrations.append(demo)
                continue

            # 2. Extract uppercase A-Z characters for fingerspelling
            chars = [c.upper() for c in clean_word if c.isalpha()]
            if not chars:
                demo = ISLDemonstration(
                    token=clean_word,
                    normalized_token=norm_token,
                    source=DemonstrationSource.UNKNOWN,
                    video_path=None,
                    characters=[],
                    alphabet_video_paths=[],
                )
                demonstrations.append(demo)
                continue

            # Check if alphabet video mapping covers all characters
            has_alphabet_coverage = False
            alphabet_paths: List[str] = []
            if self.alphabet_video_mapping is not None:
                if all(c in self.alphabet_video_mapping for c in chars):
                    has_alphabet_coverage = True
                    alphabet_paths = [self.alphabet_video_mapping[c] for c in chars]

            if has_alphabet_coverage:
                demo = ISLDemonstration(
                    token=clean_word,
                    normalized_token=norm_token,
                    source=DemonstrationSource.FINGERSPELL,
                    video_path=None,
                    characters=chars,
                    alphabet_video_paths=alphabet_paths,
                    dictionary_gloss=None,
                    description=None,
                )
            else:
                demo = ISLDemonstration(
                    token=clean_word,
                    normalized_token=norm_token,
                    source=DemonstrationSource.UNKNOWN,
                    video_path=None,
                    characters=chars,
                    alphabet_video_paths=[],
                    dictionary_gloss=None,
                    description=None,
                )
            demonstrations.append(demo)

        normalized_text = " ".join(normalized_words)
        return ISLDemonstrationSequence(
            original_text=original_text,
            normalized_text=normalized_text,
            demonstrations=demonstrations,
        )

    def has_dictionary_entry(self, term: str) -> bool:
        """Returns True if term exists in the underlying ISL dictionary index."""
        return self.dictionary.contains(term)

    def get_supported_dictionary_terms(self) -> List[str]:
        """Returns a list of all canonical glosses supported by the dictionary index."""
        return [entry.gloss for entry in self.dictionary.entries()]

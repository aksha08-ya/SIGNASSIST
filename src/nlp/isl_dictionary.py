"""ISL Dictionary and Index infrastructure for SIGNASSIST Phase 15.

Provides typed ISLDictionaryEntry structures and ISLDictionaryIndex with exact
normalized term lookup, alias mapping, schema validation, and path safety enforcement.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Union


def _validate_gloss(gloss: str) -> None:
    """Validates that a sign gloss is a non-empty string."""
    if not isinstance(gloss, str) or not gloss.strip():
        raise ValueError("Gloss cannot be empty or whitespace-only")


def _validate_video_path(video_path: str) -> None:
    """Validates that a video path is a non-empty relative path free of path traversal."""
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("video_path cannot be empty or whitespace-only")

    path_str = video_path.strip()

    # Check for absolute paths (POSIX leading slash, Windows drive letters / UNCs)
    if (
        path_str.startswith("/")
        or path_str.startswith("\\")
        or (len(path_str) >= 2 and path_str[1] == ":")
        or Path(path_str).is_absolute()
    ):
        raise ValueError(f"Path traversal / absolute path rejected in video_path: '{video_path}'")

    # Check for '..' path traversal segments
    normalized_slashes = path_str.replace("\\", "/")
    segments = [seg for seg in normalized_slashes.split("/") if seg]
    if ".." in segments or ".." in Path(path_str).parts:
        raise ValueError(f"Path traversal ('..') rejected in video_path: '{video_path}'")


@dataclass
class ISLDictionaryEntry:
    """Typed representation of an Indian Sign Language (ISL) dictionary entry.

    Attributes:
        gloss: Primary canonical gloss identifier (e.g., 'HELLO').
        video_path: Relative path to the ISL video asset (e.g., 'videos/hello.mp4').
        aliases: List of alternative text/gloss search terms mapping to this sign.
        category: Conceptual category or domain tag (default: 'general').
        source: Provenance dataset or dictionary source identifier.
        license: Content licensing specification.
        duration_sec: Duration of the video clip in seconds.
        description: Human-readable description or context notes.
        fingerspelling_available: Flag indicating fallback fingerspelling support.
    """

    gloss: str
    video_path: str
    aliases: List[str] = field(default_factory=list)
    category: str = "general"
    source: Optional[str] = None
    license: Optional[str] = None
    duration_sec: Optional[float] = None
    description: Optional[str] = None
    fingerspelling_available: bool = False

    def __post_init__(self) -> None:
        _validate_gloss(self.gloss)
        _validate_video_path(self.video_path)

        if self.aliases is None:
            self.aliases = []
        elif not isinstance(self.aliases, list):
            self.aliases = list(self.aliases)

        validated_aliases: List[str] = []
        for alias in self.aliases:
            if isinstance(alias, str) and alias.strip():
                validated_aliases.append(alias.strip())
        self.aliases = validated_aliases

        if not self.category or not isinstance(self.category, str):
            self.category = "general"
        else:
            self.category = self.category.strip()

        if self.duration_sec is not None:
            self.duration_sec = float(self.duration_sec)
            if self.duration_sec < 0.0:
                raise ValueError("duration_sec must be non-negative")

        self.fingerspelling_available = bool(self.fingerspelling_available)


class ISLDictionaryIndex:
    """In-memory dictionary index supporting canonical gloss and alias lookups.

    Enforces deterministic term normalization, case/whitespace folding,
    duplicate canonical gloss rejection, and cross-entry alias conflict prevention.
    """

    def __init__(
        self,
        entries: Optional[List[ISLDictionaryEntry]] = None,
        index_path: Optional[Path] = None,
    ) -> None:
        self.index_path = Path(index_path) if index_path else None
        self._entries: List[ISLDictionaryEntry] = []
        self._term_map: Dict[str, ISLDictionaryEntry] = {}

        if entries:
            for entry in entries:
                self._add_entry(entry)

    @staticmethod
    def normalize_term(term: str) -> str:
        """Normalizes search terms by stripping whitespace, collapsing internal whitespace,

        and performing case-folding.
        """
        if term is None:
            return ""
        if not isinstance(term, str):
            term = str(term)
        cleaned = term.strip()
        collapsed = re.sub(r"\s+", " ", cleaned)
        return collapsed.casefold()

    def _add_entry(self, entry: ISLDictionaryEntry) -> None:
        if not isinstance(entry, ISLDictionaryEntry):
            raise TypeError(f"Expected ISLDictionaryEntry instance, got {type(entry)}")

        norm_canonical = self.normalize_term(entry.gloss)
        if not norm_canonical:
            raise ValueError("Entry gloss normalized to an empty string")

        # Check for duplicate canonical gloss
        if norm_canonical in self._term_map:
            existing = self._term_map[norm_canonical]
            raise ValueError(
                f"Duplicate canonical gloss '{entry.gloss}' (normalized: '{norm_canonical}') "
                f"conflicts with existing entry '{existing.gloss}'"
            )

        # Collect all normalized terms for this entry (canonical + valid aliases)
        entry_terms: Set[str] = {norm_canonical}
        for alias in entry.aliases:
            norm_alias = self.normalize_term(alias)
            if norm_alias:
                entry_terms.add(norm_alias)

        # Check for conflicts with existing registered terms in other entries
        for term in entry_terms:
            if term in self._term_map:
                existing = self._term_map[term]
                raise ValueError(
                    f"Conflicting term/alias '{term}' in entry '{entry.gloss}' "
                    f"is already mapped to entry '{existing.gloss}'"
                )

        # Register all terms for this entry
        for term in entry_terms:
            self._term_map[term] = entry

        self._entries.append(entry)

    def lookup(self, term: str) -> Optional[ISLDictionaryEntry]:
        """Looks up an ISLDictionaryEntry by canonical gloss or alias.

        Matching is exact on normalized forms; no fuzzy matching is performed.
        Returns None if term is missing or invalid.
        """
        if not term or not isinstance(term, str):
            return None
        norm = self.normalize_term(term)
        return self._term_map.get(norm, None)

    def contains(self, term: str) -> bool:
        """Returns True if the normalized term exists in the index."""
        return self.lookup(term) is not None

    def __contains__(self, term: str) -> bool:
        return self.contains(term)

    def entries(self) -> List[ISLDictionaryEntry]:
        """Returns a list of all dictionary entries in insertion order."""
        return list(self._entries)

    def lookup_many(self, terms: List[str]) -> List[Optional[ISLDictionaryEntry]]:
        """Looks up multiple terms, returning results in exact input order."""
        return [self.lookup(t) for t in terms]

    @classmethod
    def load(cls, index_path: Union[str, Path]) -> "ISLDictionaryIndex":
        """Loads and validates an ISLDictionaryIndex from a JSON file or directory path."""
        path = Path(index_path)
        if path.is_dir():
            path = path / "index.json"

        if not path.exists():
            raise FileNotFoundError(f"Dictionary index file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid index file format in {path}: root must be a JSON object")

        if "entries" not in data or not isinstance(data["entries"], list):
            raise ValueError(f"Invalid index file format in {path}: missing 'entries' list")

        parsed_entries: List[ISLDictionaryEntry] = []
        for raw in data["entries"]:
            if not isinstance(raw, dict):
                raise ValueError("Each item in 'entries' must be a dictionary")
            entry = ISLDictionaryEntry(
                gloss=raw.get("gloss", ""),
                video_path=raw.get("video_path", ""),
                aliases=raw.get("aliases", []),
                category=raw.get("category", "general"),
                source=raw.get("source"),
                license=raw.get("license"),
                duration_sec=raw.get("duration_sec"),
                description=raw.get("description"),
                fingerspelling_available=raw.get("fingerspelling_available", False),
            )
            parsed_entries.append(entry)

        return cls(entries=parsed_entries, index_path=path)

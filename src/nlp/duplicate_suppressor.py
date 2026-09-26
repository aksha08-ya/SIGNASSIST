"""Segment-aware duplicate suppression module for SIGNASSIST Phase 10."""

from typing import List, Optional
from src.nlp.types import SignToken


class DuplicateSuppressor:
    """Stateful duplicate suppressor operating within sign segments.

    Suppresses consecutive duplicate glosses within the same sign segment or continuous frame buffer.
    A Phase 9 boundary or rest-pose transition resets the suppressor state so intentional repeated
    signs across separate segments are preserved.
    """

    def __init__(self, enable_suppression: bool = True) -> None:
        self.enable_suppression = enable_suppression
        self.last_gloss: Optional[str] = None

    def reset(self) -> None:
        """Resets suppressor state upon a boundary event or rest-pose transition."""
        self.last_gloss = None

    def process_token(self, token: SignToken) -> Optional[SignToken]:
        """Processes a single token and returns it if accepted and non-duplicate.

        Args:
            token: Input SignToken.

        Returns:
            The token if accepted and non-duplicate, or None if filtered/rejected.
        """
        # Phase 8 rejected tokens must not enter active sentence assembly
        if not token.is_accepted:
            return None

        if not self.enable_suppression:
            return token

        # Suppress consecutive identical glosses within the current segment
        if token.gloss == self.last_gloss:
            return None

        self.last_gloss = token.gloss
        return token

    def filter_tokens(
        self, tokens: List[SignToken], is_new_segment: bool = False
    ) -> List[SignToken]:
        """Filters a list of tokens, optionally resetting state for a new segment."""
        if is_new_segment:
            self.reset()

        filtered: List[SignToken] = []
        for token in tokens:
            processed = self.process_token(token)
            if processed is not None:
                filtered.append(processed)
        return filtered

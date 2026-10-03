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


class TemporalConsensusFilter:
    """Implements 3-of-5 temporal consensus filtering for streaming sign recognition.

    Maintains a sliding window of recent predictions. Requires at least `min_agreeing`
    out of the most recent `window_size` eligible predictions to agree on the predicted class/gloss
    before emitting the consensus prediction.
    """

    def __init__(
        self, window_size: int = 5, min_agreeing: int = 3, min_confidence: float = 0.70
    ) -> None:
        self.window_size = window_size
        self.min_agreeing = min_agreeing
        self.min_confidence = min_confidence
        self.recent_predictions: List[dict] = []
        self.last_emitted_class: Optional[int] = None

    def reset(self) -> None:
        """Resets consensus filter state."""
        self.recent_predictions.clear()
        self.last_emitted_class = None

    def add_prediction(
        self, class_id: int, gloss: str, confidence: float, is_accepted: bool
    ) -> Optional[dict]:
        """Adds a prediction and returns consensus result dict if agreed, else None."""
        if not is_accepted or confidence < self.min_confidence:
            return None

        self.recent_predictions.append(
            {
                "class_id": class_id,
                "gloss": gloss,
                "confidence": confidence,
            }
        )
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)

        if len(self.recent_predictions) < self.min_agreeing:
            return None

        # Count frequencies
        counts: dict[int, int] = {}
        for pred in self.recent_predictions:
            cid = pred["class_id"]
            counts[cid] = counts.get(cid, 0) + 1

        top_class, max_count = max(counts.items(), key=lambda x: x[1])
        if max_count >= self.min_agreeing:
            top_pred = next(
                p for p in reversed(self.recent_predictions) if p["class_id"] == top_class
            )
            return {
                "class_id": top_class,
                "gloss": top_pred["gloss"],
                "confidence": top_pred["confidence"],
                "consensus_count": max_count,
            }

        return None


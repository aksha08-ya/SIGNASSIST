import pytest
import torch

from src.models.confidence import ConfidenceOutput
from src.nlp import (
    BoundedSignPrediction,
    SentenceReconstructor,
    SentenceTranscript,
    SignToken,
)
from src.pipeline.boundary_detector import SignSegment


def _make_confidence(
    class_idx: int,
    confidence: float,
    is_accepted: bool,
    gloss: str = None,
) -> ConfidenceOutput:
    return ConfidenceOutput(
        predicted_class=torch.tensor(class_idx, dtype=torch.int64),
        confidence=torch.tensor(confidence, dtype=torch.float32),
        top_k_indices=torch.tensor([class_idx, 0, 1], dtype=torch.int64),
        top_k_probabilities=torch.tensor([confidence, 0.1, 0.05], dtype=torch.float32),
        is_accepted=torch.tensor(is_accepted, dtype=torch.bool),
        is_rejected=torch.tensor(not is_accepted, dtype=torch.bool),
        predicted_gloss=gloss,
    )


def test_sentence_all_rejected_yields_empty_transcript():
    """Verify when all predictions are rejected (below threshold), transcript text is empty."""
    reconstructor = SentenceReconstructor()
    seg1 = SignSegment(0, 15, 16, 0.5, 0.8)
    seg2 = SignSegment(16, 30, 15, 0.5, 0.8)

    pred1 = _make_confidence(0, 0.45, False, gloss="10. Plane")
    pred2 = _make_confidence(1, 0.55, False, gloss="5. Cow")

    transcript = reconstructor.process_batch([seg1, seg2], [pred1, pred2])
    assert len(transcript.raw_tokens) == 2
    assert len(transcript.filtered_tokens) == 0
    assert transcript.text == ""
    assert transcript.mean_confidence == 0.0


def test_sentence_one_accepted_yields_one_token():
    """Verify single accepted sign produces exactly one token transcript."""
    reconstructor = SentenceReconstructor()
    seg1 = SignSegment(0, 30, 31, 0.5, 0.8)
    pred1 = _make_confidence(0, 0.95, True, gloss="83. big large")

    transcript = reconstructor.process_batch([seg1], [pred1])
    assert len(transcript.filtered_tokens) == 1
    assert transcript.text == "83. big large."
    assert transcript.filtered_tokens[0].start_frame == 0
    assert transcript.filtered_tokens[0].end_frame == 30


def test_sentence_accepted_plus_rejected_yields_accepted_only():
    """Verify rejected tokens are completely excluded from final sentence transcript."""
    reconstructor = SentenceReconstructor()
    segs = [
        SignSegment(0, 10, 11, 0.5, 0.8),
        SignSegment(11, 20, 10, 0.5, 0.8),
        SignSegment(21, 30, 10, 0.5, 0.8),
    ]
    preds = [
        _make_confidence(0, 0.47, False, gloss="97. dry"),
        _make_confidence(1, 0.90, True, gloss="83. big large"),
        _make_confidence(2, 0.30, False, gloss="94. good"),
    ]

    transcript = reconstructor.process_batch(segs, preds)
    assert len(transcript.raw_tokens) == 3
    assert len(transcript.filtered_tokens) == 1
    assert transcript.filtered_tokens[0].gloss == "83. big large"
    assert transcript.text == "83. big large."
    assert pytest.approx(transcript.mean_confidence, abs=1e-3) == 0.90


def test_sentence_duplicate_suppression_and_timestamps():
    """Verify within-segment duplicate suppression and exact frame timestamp preservation."""
    reconstructor = SentenceReconstructor()
    seg1 = SignSegment(0, 20, 21, 0.5, 0.8)
    seg2 = SignSegment(21, 40, 20, 0.5, 0.8)

    pred1 = _make_confidence(0, 0.85, True, gloss="10. Plane")
    pred2 = _make_confidence(0, 0.88, True, gloss="10. Plane")

    # In process_batch, each segment resets suppressor so separate sign events are kept
    transcript = reconstructor.process_batch([seg1, seg2], [pred1, pred2])
    assert len(transcript.filtered_tokens) == 2
    assert transcript.filtered_tokens[0].start_frame == 0
    assert transcript.filtered_tokens[0].end_frame == 20
    assert transcript.filtered_tokens[1].start_frame == 21
    assert transcript.filtered_tokens[1].end_frame == 40

"""Unit tests for SIGNASSIST Phase 10 Sentence Reconstruction and NLP Layer."""

import pytest
import torch
import torch.nn as nn

from src.models.confidence import ConfidenceOutput
from src.nlp import (
    BoundedSignPrediction,
    ContextAwareSentenceBuilder,
    DuplicateSuppressor,
    SentenceReconstructor,
    SentenceTranscript,
    SignToken,
)
from src.pipeline.boundary_detector import SignSegment
from src.utils.config import ModelConfig, NLPConfig


def _make_mock_confidence_output(
    class_idx: int,
    confidence: float,
    is_accepted: bool,
    top_k_indices: list = None,
    top_k_probs: list = None,
) -> ConfidenceOutput:
    if top_k_indices is None:
        top_k_indices = [class_idx, 0, 1]
    if top_k_probs is None:
        top_k_probs = [confidence, 0.1, 0.05]

    return ConfidenceOutput(
        predicted_class=torch.tensor(class_idx, dtype=torch.int64),
        confidence=torch.tensor(confidence, dtype=torch.float32),
        top_k_indices=torch.tensor(top_k_indices, dtype=torch.int64),
        top_k_probabilities=torch.tensor(top_k_probs, dtype=torch.float32),
        is_accepted=torch.tensor(is_accepted, dtype=torch.bool),
        is_rejected=torch.tensor(not is_accepted, dtype=torch.bool),
    )


def test_bounded_prediction_mismatch_rejection():
    """Verify ValueError is raised when segment count != prediction count in batch mode."""
    reconstructor = SentenceReconstructor()
    seg = SignSegment(0, 10, 11, 0.5, 0.8)
    pred = _make_mock_confidence_output(0, 0.9, True)

    # 2 segments vs 1 prediction
    with pytest.raises(ValueError, match="Mismatched segment and prediction counts"):
        reconstructor.process_batch([seg, seg], [pred])


def test_phase8_confidence_rejection_honored():
    """Verify tokens rejected by Phase 8 (is_accepted=False) are excluded from active text but in raw_tokens."""
    vocab_map = {0: "ME", 1: "NAME", 2: "JOHN"}
    reconstructor = SentenceReconstructor(vocab_map=vocab_map)

    seg1 = SignSegment(0, 10, 11, 0.5, 0.8)
    seg2 = SignSegment(11, 20, 10, 0.5, 0.8)
    seg3 = SignSegment(21, 30, 10, 0.5, 0.8)

    pred1 = _make_mock_confidence_output(0, 0.9, True)   # ME (accepted)
    pred2 = _make_mock_confidence_output(1, 0.3, False)  # NAME (rejected by Phase 8)
    pred3 = _make_mock_confidence_output(2, 0.85, True)  # JOHN (accepted)

    transcript = reconstructor.process_batch([seg1, seg2, seg3], [pred1, pred2, pred3])

    assert len(transcript.raw_tokens) == 3
    assert len(transcript.filtered_tokens) == 2
    assert "NAME" not in [t.gloss for t in transcript.filtered_tokens]
    assert transcript.text == "I john."  # ME -> I, JOHN -> john


def test_within_segment_duplicate_suppression():
    """Verify consecutive identical predictions within a single segment are suppressed."""
    suppressor = DuplicateSuppressor(enable_suppression=True)

    tok1 = SignToken("HELLO", 0, 0.9, True)
    tok2 = SignToken("HELLO", 0, 0.9, True)  # Duplicate within same segment

    out1 = suppressor.process_token(tok1)
    out2 = suppressor.process_token(tok2)

    assert out1 is not None
    assert out1.gloss == "HELLO"
    assert out2 is None  # Suppressed duplicate


def test_boundary_reset_duplicate_suppression():
    """Verify duplicate signs separated by a Phase 9 boundary/rest-pose transition are preserved."""
    vocab_map = {0: "AGAIN"}
    reconstructor = SentenceReconstructor(vocab_map=vocab_map)

    seg1 = SignSegment(0, 15, 16, 0.5, 0.8)
    seg2 = SignSegment(25, 40, 16, 0.5, 0.8)

    pred1 = _make_mock_confidence_output(0, 0.9, True)
    pred2 = _make_mock_confidence_output(0, 0.9, True)

    # Process separate segments
    transcript = reconstructor.process_batch([seg1, seg2], [pred1, pred2])

    assert len(transcript.filtered_tokens) == 2
    assert [t.gloss for t in transcript.filtered_tokens] == ["AGAIN", "AGAIN"]


def test_phrase_template_reconstruction():
    """Test matching common phrase templates (ME NAME JOHN -> My name is John.)."""
    vocab_map = {0: "ME", 1: "NAME", 2: "JOHN"}
    reconstructor = SentenceReconstructor(vocab_map=vocab_map)

    segs = [SignSegment(0, 10, 11, 0.5, 0.8) for _ in range(3)]
    preds = [
        _make_mock_confidence_output(0, 0.9, True),
        _make_mock_confidence_output(1, 0.9, True),
        _make_mock_confidence_output(2, 0.9, True),
    ]

    transcript = reconstructor.process_batch(segs, preds)
    assert transcript.text == "My name is John."


def test_question_punctuation_inference():
    """Test question words (WHERE, WHAT, WHO) append '?'."""
    builder = ContextAwareSentenceBuilder()
    text = builder.build_sentence(["YOU", "GO", "WHERE"])
    assert text.endswith("?")
    assert "Where" in text or "where" in text


def test_streaming_add_token_and_finalize():
    """Test incremental add_token(), add_segment_prediction(), notify_boundary(), and finalize_sentence()."""
    vocab_map = {0: "THANK", 1: "YOU"}
    reconstructor = SentenceReconstructor(vocab_map=vocab_map)

    seg1 = SignSegment(0, 10, 11, 0.5, 0.8)
    seg2 = SignSegment(11, 20, 10, 0.5, 0.8)

    pred1 = _make_mock_confidence_output(0, 0.9, True)
    pred2 = _make_mock_confidence_output(1, 0.9, True)

    tok1 = reconstructor.add_segment_prediction(seg1, pred1)
    assert tok1 is not None
    assert tok1.gloss == "THANK"

    tok2 = reconstructor.add_segment_prediction(seg2, pred2)
    assert tok2 is not None
    assert tok2.gloss == "YOU"

    transcript = reconstructor.finalize_sentence()
    assert transcript.text == "Thank you."

    # Streaming buffer should be reset after finalize
    empty_transcript = reconstructor.get_transcript()
    assert len(empty_transcript.raw_tokens) == 0


def test_empty_and_single_sign_inputs():
    """Test handling of empty sequences and single-sign inputs."""
    reconstructor = SentenceReconstructor(vocab_map={0: "HELLO"})

    # Empty
    t_empty = reconstructor.process_batch([], [])
    assert t_empty.text == ""
    assert len(t_empty.raw_tokens) == 0

    # Single sign
    seg = SignSegment(0, 10, 11, 0.5, 0.8)
    pred = _make_mock_confidence_output(0, 0.9, True)
    t_single = reconstructor.process_batch([seg], [pred])
    assert t_single.text == "Hello."


def test_sentence_transcript_edit():
    """Verify manual update_text() updates text and sets is_edited=True."""
    transcript = SentenceTranscript(text="Hello.")
    assert not transcript.is_edited

    transcript.update_text("Hello world!")
    assert transcript.text == "Hello world!"
    assert transcript.is_edited


def test_top_k_preservation():
    """Verify top-K predictions are stored in SignToken without altering top-1 acceptance."""
    reconstructor = SentenceReconstructor(vocab_map={0: "CAT", 1: "DOG", 2: "BIRD"})
    pred = _make_mock_confidence_output(0, 0.8, True, top_k_indices=[0, 1, 2], top_k_probs=[0.8, 0.15, 0.05])
    seg = SignSegment(0, 10, 11, 0.5, 0.8)

    token = reconstructor.create_token(pred, segment=seg)
    assert token.gloss == "CAT"
    assert token.top_k_glosses == ["CAT", "DOG", "BIRD"]
    assert token.top_k_probabilities == pytest.approx([0.8, 0.15, 0.05], abs=1e-4)



def test_config_loading():
    """Verify SentenceReconstructor.from_config() instantiates correctly from ModelConfig."""
    model_cfg = ModelConfig()
    reconstructor = SentenceReconstructor.from_config(model_cfg)
    assert isinstance(reconstructor, SentenceReconstructor)
    assert reconstructor.config.enable_duplicate_suppression is True


def test_bounded_sign_prediction_valid_single_sample():
    """Verify BoundedSignPrediction accepts valid scalar/single-sample ConfidenceOutput."""
    seg = SignSegment(0, 10, 11, 0.5, 0.8)
    pred_scalar = _make_mock_confidence_output(0, 0.9, True)
    bounded = BoundedSignPrediction(segment=seg, prediction=pred_scalar)
    bounded.validate()  # Should pass without error

    pred_1d_tensor = ConfidenceOutput(
        predicted_class=torch.tensor([0], dtype=torch.int64),
        confidence=torch.tensor([0.9], dtype=torch.float32),
        top_k_indices=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        top_k_probabilities=torch.tensor([[0.9, 0.08, 0.02]], dtype=torch.float32),
        is_accepted=torch.tensor([True], dtype=torch.bool),
        is_rejected=torch.tensor([False], dtype=torch.bool),
    )
    bounded2 = BoundedSignPrediction(segment=seg, prediction=pred_1d_tensor)
    bounded2.validate()  # Should pass without error


def test_bounded_sign_prediction_batched_output_rejection():
    """Verify BoundedSignPrediction rejects genuinely batched ConfidenceOutput (B > 1) with ValueError."""
    seg = SignSegment(0, 10, 11, 0.5, 0.8)
    batched_pred = ConfidenceOutput(
        predicted_class=torch.tensor([0, 1], dtype=torch.int64),  # Batch size 2
        confidence=torch.tensor([0.9, 0.8], dtype=torch.float32),
        top_k_indices=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64),
        top_k_probabilities=torch.tensor([[0.9, 0.08, 0.02], [0.8, 0.15, 0.05]], dtype=torch.float32),
        is_accepted=torch.tensor([True, True], dtype=torch.bool),
        is_rejected=torch.tensor([False, False], dtype=torch.bool),
    )
    bounded = BoundedSignPrediction(segment=seg, prediction=batched_pred)
    with pytest.raises(ValueError, match="expects single-sample prediction"):
        bounded.validate()


def test_no_recurrent_modules():
    """Verify zero LSTM/GRU/RNN dependencies across all src/nlp/ modules."""
    import src.nlp as nlp_pkg
    for attr_name in dir(nlp_pkg):
        attr = getattr(nlp_pkg, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module if hasattr(nn, 'Module') else object):
            for module in attr().modules():
                assert not isinstance(module, (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell))

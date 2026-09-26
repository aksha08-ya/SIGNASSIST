"""Smoke test script for SIGNASSIST Phase 10 Sentence Reconstruction & NLP Layer."""

import os
import sys

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.confidence import ConfidenceOutput
from src.nlp import (
    SentenceReconstructor,
    SentenceTranscript,
    SignToken,
)
from src.pipeline.boundary_detector import SignSegment
from src.utils.config import ConfigManager


def _make_confidence(class_idx: int, conf: float, accepted: bool) -> ConfidenceOutput:
    return ConfidenceOutput(
        predicted_class=torch.tensor([class_idx], dtype=torch.int64),
        confidence=torch.tensor([conf], dtype=torch.float32),
        top_k_indices=torch.tensor([[class_idx, 0, 1]], dtype=torch.int64),
        top_k_probabilities=torch.tensor([[conf, 0.1, 0.05]], dtype=torch.float32),
        is_accepted=torch.tensor([accepted], dtype=torch.bool),
        is_rejected=torch.tensor([not accepted], dtype=torch.bool),
    )


def main() -> None:
    print("=== SIGNASSIST Phase 10 Sentence Reconstruction Smoke Test ===")

    # 1. Config loading
    config_mgr = ConfigManager()
    model_cfg = config_mgr.load_all().model
    nlp_cfg = model_cfg.nlp
    print(f"[1/4] Loaded NLPConfig:")
    print(f"      enable_duplicate_suppression: {nlp_cfg.enable_duplicate_suppression}")
    print(f"      enable_context_rules: {nlp_cfg.enable_context_rules}")
    print(f"      auto_punctuation: {nlp_cfg.auto_punctuation}")

    # Vocabulary map
    vocab_map = {
        0: "ME",
        1: "NAME",
        2: "AKSHAYA",
        3: "WHERE",
        4: "YOU",
        5: "GO",
    }

    # 2. Instantiation
    reconstructor = SentenceReconstructor(config=model_cfg, vocab_map=vocab_map)
    print("\n[2/4] Instantiated SentenceReconstructor successfully.")

    # 3. Batch Reconstruction Test: "ME NAME AKSHAYA" -> "My name is Akshaya."
    print("\n[3/4] Running Batch Reconstruction (Phase 8 + Phase 9 -> Phase 10)...")
    segments = [
        SignSegment(start_frame=10, end_frame=30, duration=21, mean_energy=0.5, boundary_evidence=0.9),
        SignSegment(start_frame=35, end_frame=55, duration=21, mean_energy=0.6, boundary_evidence=0.95),
        SignSegment(start_frame=60, end_frame=80, duration=21, mean_energy=0.7, boundary_evidence=0.98),
    ]

    predictions = [
        _make_confidence(class_idx=0, conf=0.95, accepted=True),  # ME
        _make_confidence(class_idx=1, conf=0.92, accepted=True),  # NAME
        _make_confidence(class_idx=2, conf=0.88, accepted=True),  # AKSHAYA
    ]

    batch_transcript: SentenceTranscript = reconstructor.process_batch(segments, predictions)

    print(f"      Raw tokens count: {len(batch_transcript.raw_tokens)}")
    print(f"      Filtered tokens count: {len(batch_transcript.filtered_tokens)}")
    print(f"      Mean confidence: {batch_transcript.mean_confidence:.4f}")
    print(f"      Reconstructed sentence text: '{batch_transcript.text}'")

    # 4. Streaming Reconstruction Test: "YOU GO WHERE" -> "Where are you going?" / Question punctuation
    print("\n[4/4] Running Streaming Reconstruction...")
    reconstructor.reset()

    stream_segs = [
        SignSegment(start_frame=0, end_frame=15, duration=16, mean_energy=0.5, boundary_evidence=0.8),
        SignSegment(start_frame=20, end_frame=35, duration=16, mean_energy=0.5, boundary_evidence=0.8),
        SignSegment(start_frame=40, end_frame=55, duration=16, mean_energy=0.5, boundary_evidence=0.8),
    ]
    stream_preds = [
        _make_confidence(class_idx=4, conf=0.90, accepted=True),  # YOU
        _make_confidence(class_idx=5, conf=0.85, accepted=True),  # GO
        _make_confidence(class_idx=3, conf=0.91, accepted=True),  # WHERE
    ]

    for seg, pred in zip(stream_segs, stream_preds):
        reconstructor.add_segment_prediction(seg, pred)

    stream_transcript = reconstructor.finalize_sentence()
    print(f"      Streaming reconstructed sentence: '{stream_transcript.text}'")

    # Test editable transcript functionality
    stream_transcript.update_text("Where are you going, Akshaya?")
    print(f"      Edited transcript text: '{stream_transcript.text}' (is_edited={stream_transcript.is_edited})")

    print("\n=== Phase 10 Smoke Test Passed Successfully! ===")


if __name__ == "__main__":
    main()

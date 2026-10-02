"""Focused unit tests for UI services and video integration in dynamic-only mode."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.models.confidence import ConfidenceOutput
from src.nlp.types import BoundedSignPrediction
from src.pipeline.boundary_detector import SignSegment
from src.pipeline.recognition import RecognitionPipeline
from src.ui.services import get_recognition_readiness, process_video_file, process_video_url
from src.video.file_reader import VideoMetadata
from src.video.processor import VideoProcessingResult
from src.video.quality import VideoQualityReport


def test_video_upload_invokes_dynamic_only_pipeline(tmp_path: Path):
    """Test that process_video_file uses RecognitionPipeline(mode='dynamic_only')."""
    dummy_video = tmp_path / "sample.mp4"
    dummy_video.write_bytes(b"dummy")

    with patch("src.ui.services.VideoProcessor") as mock_processor_cls:
        mock_processor_inst = MagicMock()
        mock_processor_cls.return_value = mock_processor_inst
        mock_processor_inst.process_file.return_value = VideoProcessingResult(
            video_metadata=VideoMetadata(
                file_path=str(dummy_video),
                file_name="sample.mp4",
                file_size_bytes=100,
                duration_sec=1.0,
                total_frames=30,
                native_fps=30.0,
                width=640,
                height=480,
                aspect_ratio=4 / 3,
                codec="mp4v",
            ),
            quality_report=VideoQualityReport(
                average_brightness=100.0,
                average_contrast=50.0,
                average_blur_score=200.0,
                resolution_ok=True,
                hand_visibility_ratio=0.9,
                sampled_frames=30,
                overall_rating="Good",
            ),
            recognition_available=True,
            frames_processed=30,
        )

        res = process_video_file(dummy_video, mode="dynamic_only")
        assert res.recognition_available is True
        mock_processor_cls.assert_called_once()

        pipeline_arg = mock_processor_cls.call_args.kwargs.get("recognition_pipeline")
        assert pipeline_arg is not None
        assert getattr(pipeline_arg, "mode", None) == "dynamic_only"


def test_rejected_low_confidence_prediction_rendered_as_rejected():
    """Test that low-confidence predictions (< 0.70) set is_accepted to False."""
    conf_out = ConfidenceOutput(
        predicted_class=torch.tensor([0]),
        confidence=torch.tensor([0.05]),  # Below 0.70 threshold
        top_k_indices=torch.tensor([[0, 1, 2]]),
        top_k_probabilities=torch.tensor([[0.05, 0.03, 0.02]]),
        is_accepted=torch.tensor([False]),
        is_rejected=torch.tensor([True]),
        predicted_gloss="1. Dog",
        top_k_glosses=["1. Dog", "2. Cat", "3. Bird"],
    )
    segment = SignSegment(start_frame=0, end_frame=30, duration=31, mean_energy=0.5, boundary_evidence=0.8)
    pred = BoundedSignPrediction(segment=segment, prediction=conf_out)
    pred.validate()

    assert pred.prediction.is_accepted.item() is False
    assert float(pred.prediction.confidence.item()) == pytest.approx(0.05)


def test_inference_failure_surfaced(tmp_path: Path):
    """Test that if recognition pipeline throws or fails, recognition_available is handled."""
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        dynamic_checkpoint_path=tmp_path / "non_existent.pt",
        models_dir=tmp_path,
    )
    assert pipeline.is_available() is False
    preds = pipeline.process_frames(frames=[], primary_tracks=[])
    assert preds == []


def test_url_processing_uses_dynamic_only_backend():
    """Test that process_video_url initializes dynamic_only video processor."""
    with patch("src.ui.services.URLVideoDownloader") as mock_dl_cls:
        mock_dl_inst = MagicMock()
        mock_dl_cls.return_value = mock_dl_inst
        mock_dl_inst.process_url.return_value = VideoProcessingResult(
            video_metadata=VideoMetadata(
                file_path="dummy.mp4",
                file_name="dummy.mp4",
                file_size_bytes=100,
                duration_sec=1.0,
                total_frames=30,
                native_fps=30.0,
                width=640,
                height=480,
                aspect_ratio=4 / 3,
                codec="mp4v",
            ),
            quality_report=VideoQualityReport(
                average_brightness=100.0,
                average_contrast=50.0,
                average_blur_score=200.0,
                resolution_ok=True,
                hand_visibility_ratio=0.9,
                sampled_frames=30,
                overall_rating="Good",
            ),
            recognition_available=True,
            frames_processed=30,
        )

        res = process_video_url("https://example.com/video.mp4", mode="dynamic_only")
        assert res.recognition_available is True
        mock_dl_inst.process_url.assert_called_once()
        processor_used = mock_dl_inst.process_url.call_args.kwargs.get("processor")
        assert processor_used is not None
        assert getattr(processor_used.recognition_pipeline, "mode", None) == "dynamic_only"


def test_multimodal_readiness_still_reports_unavailable(tmp_path: Path):
    """Test that get_recognition_readiness(mode='multimodal') reports unavailable when 4 checkpoints missing."""
    readiness = get_recognition_readiness(models_dir=tmp_path, mode="multimodal")
    assert readiness["recognition_pipeline_available"] is False


def test_dynamic_only_readiness_and_checkpoint_detection():
    """Test that get_recognition_readiness(mode='dynamic_only') detects real checkpoint and reports ready."""
    readiness = get_recognition_readiness(mode="dynamic_only")
    assert readiness["recognition_pipeline_available"] is True
    assert readiness["checkpoint_available"] is True
    assert readiness["mode"] == "dynamic_only"


def test_timestamp_conversion_absolute_frame_indices():
    """Test that RecognitionPipeline.process_frames converts chunk-relative indices to absolute video frame indices."""
    from src.video.file_reader import DecodedFrame
    import numpy as np

    pipeline = RecognitionPipeline(mode="dynamic_only")
    if not pipeline.is_available():
        pytest.skip("Dynamic classifier checkpoint unavailable.")

    # Create dummy DecodedFrames representing chunk 2 (frames 60..119)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    chunk_frames = [
        DecodedFrame(
            frame=dummy_img,
            frame_index=idx,
            timestamp_sec=float(idx / 30.0),
            timestamp=f"00:{idx // 30:02d}.000",
        )
        for idx in range(60, 120)
    ]
    tracks = [None] * len(chunk_frames)

    preds = pipeline.process_frames(frames=chunk_frames, primary_tracks=tracks)
    # If boundary detector returns segments, their start_frame must be >= 60 (absolute video frame index)
    for pred in preds:
        assert pred.segment.start_frame >= 60
        assert pred.segment.end_frame >= pred.segment.start_frame
        assert pred.segment.end_frame <= 119


def test_sentence_reconstructor_duplicate_suppression_and_rejection():
    """Test duplicate suppression and rejected prediction handling in DuplicateSuppressor and SentenceReconstructor."""
    from src.nlp.duplicate_suppressor import DuplicateSuppressor
    from src.nlp.sentence_reconstructor import SentenceReconstructor

    reconstructor = SentenceReconstructor()
    suppressor = DuplicateSuppressor(enable_suppression=True)

    # High-confidence accepted token
    conf_acc_1 = ConfidenceOutput(
        predicted_class=torch.tensor([1]),
        confidence=torch.tensor([0.95]),
        top_k_indices=torch.tensor([[1, 2, 3]]),
        top_k_probabilities=torch.tensor([[0.95, 0.03, 0.02]]),
        is_accepted=torch.tensor([True]),
        is_rejected=torch.tensor([False]),
        predicted_gloss="Hello",
        top_k_glosses=["Hello", "World", "Sign"],
    )
    # Consecutive identical accepted token
    conf_acc_2 = ConfidenceOutput(
        predicted_class=torch.tensor([1]),
        confidence=torch.tensor([0.92]),
        top_k_indices=torch.tensor([[1, 2, 3]]),
        top_k_probabilities=torch.tensor([[0.92, 0.05, 0.03]]),
        is_accepted=torch.tensor([True]),
        is_rejected=torch.tensor([False]),
        predicted_gloss="Hello",
        top_k_glosses=["Hello", "World", "Sign"],
    )
    # Low-confidence rejected token
    conf_rej = ConfidenceOutput(
        predicted_class=torch.tensor([5]),
        confidence=torch.tensor([0.05]),
        top_k_indices=torch.tensor([[5, 6, 7]]),
        top_k_probabilities=torch.tensor([[0.05, 0.03, 0.02]]),
        is_accepted=torch.tensor([False]),
        is_rejected=torch.tensor([True]),
        predicted_gloss="Cow",
        top_k_glosses=["Cow", "Dog", "Cat"],
    )

    seg1 = SignSegment(start_frame=0, end_frame=29, duration=30, mean_energy=0.5, boundary_evidence=0.8)
    seg2 = SignSegment(start_frame=30, end_frame=59, duration=30, mean_energy=0.5, boundary_evidence=0.8)
    seg3 = SignSegment(start_frame=60, end_frame=89, duration=30, mean_energy=0.5, boundary_evidence=0.8)

    tok1 = reconstructor.create_token(conf_acc_1, segment=seg1)
    tok2 = reconstructor.create_token(conf_acc_2, segment=seg2)
    tok3 = reconstructor.create_token(conf_rej, segment=seg3)

    # 1. Direct DuplicateSuppressor test on a continuous sequence
    p1 = suppressor.process_token(tok1)
    p2 = suppressor.process_token(tok2)
    p3 = suppressor.process_token(tok3)
    assert p1 is not None and p1.gloss == "Hello"
    assert p2 is None  # Suppressed duplicate within stream
    assert p3 is None  # Filtered rejected token

    # 2. SentenceReconstructor process_batch test (preserves raw tokens & handles separate segments)
    transcript = reconstructor.process_batch(
        segments=[seg1, seg3],
        predictions=[conf_acc_1, conf_rej],
    )
    assert len(transcript.raw_tokens) == 2
    assert len(transcript.filtered_tokens) == 1
    assert transcript.filtered_tokens[0].gloss == "Hello"


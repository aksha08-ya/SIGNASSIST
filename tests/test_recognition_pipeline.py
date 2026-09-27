"""
Unit tests for End-to-End RecognitionPipeline (Phase 17 Block 6).

Note: Temporary test checkpoints in these tests are un-trained structural models.
Tests verify execution flow, shapes, types, and structural validity ONLY,
and do NOT assert recognition accuracy or semantic correctness.
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from src.pipeline.recognition import RecognitionPipeline
from src.features.types import SelectedGraphLandmarks, StaticFeatureVector
from src.models.confidence import ConfidenceOutput
from src.models.router import RouterOutput
from src.nlp.types import BoundedSignPrediction
from src.pipeline.boundary_detector import BoundaryDetectionOutput, SignSegment
from src.video.file_reader import DecodedFrame


def test_pipeline_init_without_checkpoints() -> None:
    """Tests pipeline initialization when checkpoints are absent."""
    pipeline = RecognitionPipeline()
    try:
        assert pipeline.is_available() is False
        assert isinstance(pipeline.models_dir, Path)
    finally:
        pipeline.close()


def test_pipeline_checkpoint_discovery_empty_dir(tmp_path: Path) -> None:
    """Tests checkpoint discovery with an empty directory."""
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    pipeline = RecognitionPipeline(models_dir=empty_dir)
    try:
        assert pipeline.is_available() is False
        assert "Missing required checkpoint files" in pipeline._readiness_reason
    finally:
        pipeline.close()


def test_extract_state_dict_helper() -> None:
    """Tests the private state_dict extraction helper for raw and checkpoint dicts."""
    pipeline = RecognitionPipeline()
    try:
        raw_sd = pipeline.static_gcn.state_dict()

        # Direct raw state dict
        extracted1 = pipeline._extract_state_dict(raw_sd)
        assert isinstance(extracted1, dict)

        # Checkpoint dict with 'state_dict' key
        extracted2 = pipeline._extract_state_dict({"state_dict": raw_sd})
        assert isinstance(extracted2, dict)

        # Checkpoint dict with 'model_state_dict' key
        extracted3 = pipeline._extract_state_dict({"model_state_dict": raw_sd})
        assert isinstance(extracted3, dict)

        # Invalid types or malformed dicts
        with pytest.raises(TypeError):
            pipeline._extract_state_dict("invalid_type")

        with pytest.raises(ValueError):
            pipeline._extract_state_dict({})
    finally:
        pipeline.close()


def test_pipeline_preprocessing_shapes() -> None:
    """Tests shape and type outputs of preprocessing helpers."""
    pipeline = RecognitionPipeline()
    try:
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw, selected, static_vector = pipeline.preprocess_frame(dummy_frame)

        assert selected.nodes.shape == (115, 5)
        assert selected.mask.shape == (115,)
        assert static_vector.features.shape == (225,)
        assert static_vector.mask.shape == (75,)

        # Dynamic tensor building
        tensor, mask = pipeline.build_dynamic_tensor([selected, selected])
        seq_len = pipeline.config.model.dynamic_model.sequence_length
        assert tensor.shape == (1, 5, seq_len, 115)
        assert mask.shape == (1, seq_len, 115)
        assert tensor.dtype == torch.float32
        assert mask.dtype == torch.bool
    finally:
        pipeline.close()


def test_pipeline_decoded_frame_preprocessing() -> None:
    """Tests preprocessing with a DecodedFrame container."""
    pipeline = RecognitionPipeline()
    try:
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        decoded = DecodedFrame(
            frame=dummy_bgr,
            frame_index=0,
            timestamp_sec=0.0,
            timestamp="00:00.000",
        )
        raw, selected, static_vector = pipeline.preprocess_frame(decoded)
        assert selected.nodes.shape == (115, 5)
    finally:
        pipeline.close()


def test_pipeline_invalid_frame_handling() -> None:
    """Tests error handling for invalid/empty frames."""
    pipeline = RecognitionPipeline()
    try:
        with pytest.raises(ValueError):
            pipeline.preprocess_frame(np.zeros((0,), dtype=np.uint8))

        with pytest.raises(TypeError):
            pipeline.preprocess_frame("invalid_frame_type")  # type: ignore
    finally:
        pipeline.close()


def test_process_frames_unavailable_returns_empty() -> None:
    """Tests that process_frames returns [] when pipeline is unavailable."""
    pipeline = RecognitionPipeline()
    try:
        assert pipeline.is_available() is False
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        decoded = DecodedFrame(
            frame=dummy_bgr,
            frame_index=0,
            timestamp_sec=0.0,
            timestamp="00:00.000",
        )

        preds = pipeline.process_frames(frames=[decoded], primary_tracks=[None])
        assert preds == []
    finally:
        pipeline.close()


def test_no_fake_predictions_when_unavailable() -> None:
    """Verifies that no fake predictions are fabricated when checkpoints are absent."""
    pipeline = RecognitionPipeline()
    try:
        assert pipeline.is_available() is False
        preds = pipeline.process_frames(frames=[], primary_tracks=[])
        assert isinstance(preds, list)
        assert len(preds) == 0
    finally:
        pipeline.close()


def test_label_helpers() -> None:
    """Tests label helper functions for static and dynamic classes."""
    pipeline = RecognitionPipeline()
    try:
        assert pipeline.get_static_label(5) == "static_5"
        assert pipeline.get_dynamic_label(42) == "dynamic_42"
    finally:
        pipeline.close()


def test_process_frames_structural_inference_path(tmp_path: Path) -> None:
    """Structural Test: Exercises the complete process_frames inference execution flow.

    Uses temporary test checkpoints with initialized model weights solely to verify pipeline connectivity.
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    base_pipeline = RecognitionPipeline()
    try:
        torch.save(base_pipeline.static_gcn.state_dict(), models_dir / "static_gcn.pt")
        torch.save(
            base_pipeline.dynamic_classifier.state_dict(),
            models_dir / "dynamic_classifier.pt",
        )
        torch.save(base_pipeline.router.state_dict(), models_dir / "router.pt")
        torch.save(base_pipeline.fusion_engine.state_dict(), models_dir / "fusion.pt")
    finally:
        base_pipeline.close()

    pipeline = RecognitionPipeline(models_dir=models_dir)
    try:
        assert pipeline.is_available() is True

        fake_segment = SignSegment(
            start_frame=0,
            end_frame=5,
            duration=6,
            mean_energy=0.5,
            boundary_evidence=0.8,
            is_valid=True,
        )
        pipeline.boundary_detector.detect_boundaries = MagicMock(
            return_value=BoundaryDetectionOutput(
                segments=[[fake_segment]],
                motion_energy=torch.zeros((1, 10)),
                smoothed_energy=torch.zeros((1, 10)),
                boundary_flags=torch.zeros((1, 10), dtype=torch.bool),
                rest_pose_mask=torch.zeros((1, 10), dtype=torch.bool),
            )
        )

        frames = [
            DecodedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=i,
                timestamp_sec=float(i / 30.0),
                timestamp=f"00:{i:02d}",
            )
            for i in range(10)
        ]

        preds = pipeline.process_frames(frames, [None] * 10)
        assert isinstance(preds, list)
        assert len(preds) == 1

        pred = preds[0]
        assert isinstance(pred, BoundedSignPrediction)
        pred.validate()  # Ensure single-sample validation passes cleanly
    finally:
        pipeline.close()


def test_separate_vocabularies_and_branch_selection(tmp_path: Path) -> None:
    """Structural Test: Verifies static (36) vs dynamic (250) branch selection logic."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    base_pipeline = RecognitionPipeline()
    try:
        torch.save(base_pipeline.static_gcn.state_dict(), models_dir / "static_gcn.pt")
        torch.save(
            base_pipeline.dynamic_classifier.state_dict(),
            models_dir / "dynamic_classifier.pt",
        )
        torch.save(base_pipeline.router.state_dict(), models_dir / "router.pt")
        torch.save(base_pipeline.fusion_engine.state_dict(), models_dir / "fusion.pt")
    finally:
        base_pipeline.close()

    pipeline = RecognitionPipeline(models_dir=models_dir)
    try:
        assert pipeline.is_available() is True

        fake_segment = SignSegment(
            start_frame=0,
            end_frame=5,
            duration=6,
            mean_energy=0.5,
            boundary_evidence=0.8,
            is_valid=True,
        )
        pipeline.boundary_detector.detect_boundaries = MagicMock(
            return_value=BoundaryDetectionOutput(
                segments=[[fake_segment]],
                motion_energy=torch.zeros((1, 10)),
                smoothed_energy=torch.zeros((1, 10)),
                boundary_flags=torch.zeros((1, 10), dtype=torch.bool),
                rest_pose_mask=torch.zeros((1, 10), dtype=torch.bool),
            )
        )

        frames = [
            DecodedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=i,
                timestamp_sec=float(i / 30.0),
                timestamp=f"00:{i:02d}",
            )
            for i in range(10)
        ]

        # Case 1: Force static branch selection (static_weight > dynamic_weight)
        pipeline.router.forward = MagicMock(
            return_value=RouterOutput(
                static_weight=torch.tensor([[0.8]]),
                dynamic_weight=torch.tensor([[0.2]]),
                probabilities=torch.tensor([[0.8, 0.2]]),
            )
        )
        preds_static = pipeline.process_frames(frames, [None] * 10)
        assert len(preds_static) == 1
        p_stat = preds_static[0].prediction
        assert isinstance(p_stat, ConfidenceOutput)
        # Class index must be within static vocabulary [0, 35]
        assert 0 <= int(p_stat.predicted_class[0].item()) < 36
        assert p_stat.probabilities.shape[1] == 36

        # Case 2: Force dynamic branch selection (dynamic_weight > static_weight)
        pipeline.router.forward = MagicMock(
            return_value=RouterOutput(
                static_weight=torch.tensor([[0.1]]),
                dynamic_weight=torch.tensor([[0.9]]),
                probabilities=torch.tensor([[0.1, 0.9]]),
            )
        )
        preds_dynamic = pipeline.process_frames(frames, [None] * 10)
        assert len(preds_dynamic) == 1
        p_dyn = preds_dynamic[0].prediction
        assert isinstance(p_dyn, ConfidenceOutput)
        # Class index must be within dynamic vocabulary [0, 249]
        assert 0 <= int(p_dyn.predicted_class[0].item()) < 250
        assert p_dyn.probabilities.shape[1] == 250
    finally:
        pipeline.close()


def test_empty_or_no_segment_case_returns_empty(tmp_path: Path) -> None:
    """Tests that empty input or sequences with no detected boundary segments return []."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    base_pipeline = RecognitionPipeline()
    try:
        torch.save(base_pipeline.static_gcn.state_dict(), models_dir / "static_gcn.pt")
        torch.save(
            base_pipeline.dynamic_classifier.state_dict(),
            models_dir / "dynamic_classifier.pt",
        )
        torch.save(base_pipeline.router.state_dict(), models_dir / "router.pt")
        torch.save(base_pipeline.fusion_engine.state_dict(), models_dir / "fusion.pt")
    finally:
        base_pipeline.close()

    pipeline = RecognitionPipeline(models_dir=models_dir)
    try:
        # Empty frames
        assert pipeline.process_frames([], []) == []

        # Frames present but no segments detected
        pipeline.boundary_detector.detect_boundaries = MagicMock(
            return_value=BoundaryDetectionOutput(
                segments=[[]],
                motion_energy=torch.zeros((1, 10)),
                smoothed_energy=torch.zeros((1, 10)),
                boundary_flags=torch.zeros((1, 10), dtype=torch.bool),
                rest_pose_mask=torch.zeros((1, 10), dtype=torch.bool),
            )
        )
        frames = [
            DecodedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=i,
                timestamp_sec=float(i / 30.0),
                timestamp=f"00:{i:02d}",
            )
            for i in range(10)
        ]
        assert pipeline.process_frames(frames, [None] * 10) == []
    finally:
        pipeline.close()


def test_normalizer_state_reset_between_calls() -> None:
    """Verifies that LandmarkNormalizer history state is reset on each process_frames call."""
    pipeline = RecognitionPipeline()
    try:
        pipeline.is_available = MagicMock(return_value=True)
        pipeline.normalizer.reset = MagicMock(wraps=pipeline.normalizer.reset)
        frames = [
            DecodedFrame(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                frame_index=0,
                timestamp_sec=0.0,
                timestamp="00:00",
            )
        ]
        # Call 1
        pipeline.process_frames(frames, [None])
        call_count_1 = pipeline.normalizer.reset.call_count
        assert call_count_1 == 1

        # Call 2
        pipeline.process_frames(frames, [None])
        call_count_2 = pipeline.normalizer.reset.call_count
        assert call_count_2 == 2
    finally:
        pipeline.close()

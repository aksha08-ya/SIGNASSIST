"""Unit tests for Phase 9 continuous sign boundary detection and temporal smoothing."""

import pytest
import torch
import torch.nn as nn

from src.pipeline.boundary_detector import (
    BoundaryDetectionOutput,
    SignBoundaryDetector,
    SignSegment,
)
from src.pipeline.temporal_smoother import TemporalSmoother


def test_motion_energy_calculation_shape():
    """Verify motion energy output shape [B, T] for input [B, 5, T, 115]."""
    detector = SignBoundaryDetector()
    B, T = 2, 60
    landmarks = torch.randn(B, 5, T, 115)
    energy = detector.compute_motion_energy(landmarks)
    assert energy.shape == (B, T)
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()


def test_zero_motion_sequence():
    """Verify zero motion input leads to zero energy and produces no invalid segments."""
    detector = SignBoundaryDetector()
    B, T = 2, 60
    landmarks = torch.zeros(B, 5, T, 115)
    output = detector.detect_boundaries(landmarks)

    assert isinstance(output, BoundaryDetectionOutput)
    assert output.motion_energy.shape == (B, T)
    assert (output.motion_energy == 0.0).all()
    assert (output.smoothed_energy == 0.0).all()
    assert output.rest_pose_mask.all()
    for sample_segs in output.segments:
        assert len(sample_segs) == 0


def test_constant_motion_sequence():
    """Verify constant motion above threshold is handled properly and boundary_evidence is set."""
    detector = SignBoundaryDetector(min_segment_duration=10, rest_pose_threshold=0.1)
    B, T = 1, 40
    landmarks = torch.zeros(B, 5, T, 115)
    # Set non-zero landmark positions (x, y) so valid_mask is True
    landmarks[:, 0, :, :42] = 0.5
    landmarks[:, 1, :, :42] = 0.5
    # Set constant vx, vy for active nodes
    landmarks[:, 3, :, :42] = 0.5  # vx
    landmarks[:, 4, :, :42] = 0.0  # vy

    output = detector.detect_boundaries(landmarks)

    assert output.motion_energy.shape == (B, T)
    assert (output.motion_energy > 0.4).all()
    assert not output.rest_pose_mask.any()
    assert len(output.segments[0]) == 1
    seg = output.segments[0][0]
    assert seg.duration >= 10
    assert seg.start_frame == 0
    assert seg.end_frame == T - 1
    assert hasattr(seg, "boundary_evidence")
    assert 0.0 <= seg.boundary_evidence <= 1.0


def test_zero_valid_nodes_case():
    """Verify all invalid nodes yield zero energy without zero-division error or NaN/Inf."""
    detector = SignBoundaryDetector()
    B, T = 2, 30
    landmarks = torch.randn(B, 5, T, 115)
    valid_mask = torch.zeros(B, T, 115, dtype=torch.bool)

    energy = detector.compute_motion_energy(landmarks, valid_mask=valid_mask)

    assert energy.shape == (B, T)
    assert (energy == 0.0).all()
    assert not torch.isnan(energy).any()
    assert not torch.isinf(energy).any()

    output = detector.detect_boundaries(landmarks, valid_mask=valid_mask)
    assert not torch.isnan(output.smoothed_energy).any()


def test_local_minima_detection():
    """Verify a sequence with energy dip triggers candidate boundary detection."""
    detector = SignBoundaryDetector(
        energy_threshold=0.3,
        rest_pose_threshold=0.05,
        min_boundary_separation=2,
        min_segment_duration=5,
    )
    B, T = 1, 50
    landmarks = torch.zeros(B, 5, T, 115)
    landmarks[:, 0, :, :42] = 0.5
    landmarks[:, 1, :, :42] = 0.5

    # High velocity 0..15, low velocity dip at 16..22, high velocity 23..45
    landmarks[:, 3, 0:16, :42] = 0.5
    landmarks[:, 3, 16:23, :42] = 0.02  # Energy dip
    landmarks[:, 3, 23:46, :42] = 0.5

    output = detector.detect_boundaries(landmarks)
    assert output.boundary_flags.any()
    assert len(output.segments[0]) >= 1


def test_rest_pose_detection():
    """Verify frames below rest_pose_threshold are classified as rest pose."""
    detector = SignBoundaryDetector(rest_pose_threshold=0.1)
    B, T = 1, 30
    landmarks = torch.zeros(B, 5, T, 115)
    # Frames 0..10 high velocity, 11..29 zero velocity
    landmarks[:, 0, :, :42] = 0.5
    landmarks[:, 1, :, :42] = 0.5
    landmarks[:, 3, 0:10, :42] = 0.5

    output = detector.detect_boundaries(landmarks)

    # Late frames should be marked as rest pose
    assert output.rest_pose_mask[:, 15:].all()


def test_min_segment_duration_enforcement():
    """Verify short active bursts (< min_segment_duration) are ignored."""
    detector = SignBoundaryDetector(min_segment_duration=15, rest_pose_threshold=0.1)
    B, T = 1, 40
    landmarks = torch.zeros(B, 5, T, 115)
    landmarks[:, 0, :, :42] = 0.5
    landmarks[:, 1, :, :42] = 0.5

    # Short burst of 5 active frames
    landmarks[:, 3, 10:15, :42] = 0.5

    output = detector.detect_boundaries(landmarks)

    # The 5-frame burst is less than min_segment_duration (15), so no segment returned
    assert len(output.segments[0]) == 0


def test_minimum_boundary_separation_debouncing():
    """Verify candidate boundaries within min_boundary_separation are debounced."""
    smoother = TemporalSmoother()
    events = torch.tensor([[0, 1, 1, 1, 0, 1, 0, 0, 1, 0]], dtype=torch.int32)
    debounced = smoother.debounce_events(events, min_separation=4)

    # Frame 1 triggers event. Frames 2, 3 suppressed. Frame 5 triggers event (5-1=4 >= 4). Frame 8 suppressed (8-5=3 < 4).
    assert debounced[0, 1] == 1
    assert debounced[0, 2] == 0
    assert debounced[0, 3] == 0
    assert debounced[0, 5] == 1
    assert debounced[0, 8] == 0


def test_batch_dimension_support():
    """Verify batch size > 1 produces independent per-sample segmentation results."""
    detector = SignBoundaryDetector(min_segment_duration=10, rest_pose_threshold=0.1)
    B, T = 2, 40
    landmarks = torch.zeros(B, 5, T, 115)
    landmarks[:, 0, :, :42] = 0.5
    landmarks[:, 1, :, :42] = 0.5

    # Sample 0 has motion across all frames
    landmarks[0, 3, :, :42] = 0.5
    # Sample 1 has no motion
    landmarks[1, 3, :, :42] = 0.0

    output = detector.detect_boundaries(landmarks)

    assert len(output.segments[0]) == 1
    assert len(output.segments[1]) == 0


def test_non_recurrent_architecture_verification():
    """Verify SignBoundaryDetector contains no recurrent neural network modules."""
    detector = SignBoundaryDetector()
    for module in detector.modules():
        assert not isinstance(module, (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell))


def test_invalid_tensor_shapes():
    """Verify ValueError is raised when input tensor is not [B, 5, T, 115]."""
    detector = SignBoundaryDetector()
    invalid_landmarks = torch.randn(2, 3, 60, 100)  # Wrong C and V dimensions
    with pytest.raises(ValueError):
        detector.compute_motion_energy(invalid_landmarks)


def test_even_window_rejection():
    """Verify that even or non-positive window sizes are rejected with ValueError."""
    smoother = TemporalSmoother()
    energy_2d = torch.randn(2, 20)

    # Even window sizes must raise ValueError
    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        smoother.smooth_motion_energy(energy_2d, window_size=4)

    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        smoother.smooth_motion_energy(energy_2d, window_size=0)

    with pytest.raises(ValueError, match="window_size must be a positive odd integer"):
        smoother.majority_vote(energy_2d.long(), window_size=6)


def test_smoothing_output_shape_preservation():
    """Verify smoothing preserves exact temporal dimensions [B, T] or [T]."""
    smoother = TemporalSmoother()
    T = 37
    energy_1d = torch.randn(T)
    energy_2d = torch.randn(3, T)

    res_1d = smoother.smooth_motion_energy(energy_1d, window_size=5)
    res_2d = smoother.smooth_motion_energy(energy_2d, window_size=5)

    assert res_1d.shape == (T,)
    assert res_2d.shape == (3, T)

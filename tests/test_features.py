"""
Unit tests for SIGNASSIST Phase 2: Landmark Processing & Feature Extraction.
"""

import numpy as np
import pytest

from src.features import (
    DEFAULT_SEQUENCE_LENGTH,
    NUM_FEATURE_CHANNELS,
    NUM_GRAPH_NODES,
    NUM_RAW_LANDMARKS,
    NUM_SELECTED_FACE_LANDMARKS,
    NUM_STATIC_FEATURES,
    NUM_STATIC_NODES,
    SELECTED_FACE_INDICES,
    LandmarkNormalizer,
    LandmarkSelector,
    MediaPipeHolisticExtractor,
    RawLandmarks,
    SelectedGraphLandmarks,
    SequenceBuffer,
    StaticFeatureBuilder,
)
from src.utils.exceptions import LandmarkExtractionError


def create_synthetic_raw_landmarks(
    has_face: bool = True,
    has_pose: bool = True,
    has_left_hand: bool = True,
    has_right_hand: bool = True,
    offset: float = 0.0,
    scale: float = 1.0,
) -> RawLandmarks:
    """Helper to generate deterministic synthetic RawLandmarks."""
    raw = RawLandmarks(
        has_face=has_face,
        has_pose=has_pose,
        has_left_hand=has_left_hand,
        has_right_hand=has_right_hand,
    )

    if has_face:
        face = np.zeros((468, 3), dtype=np.float32)
        for i in range(468):
            face[i] = [i * 0.001 * scale + offset, (i + 1) * 0.001 * scale + offset, 0.1 * scale]
        raw.face = face

    if has_pose:
        pose = np.zeros((33, 4), dtype=np.float32)
        for i in range(33):
            pose[i] = [
                (i + 10) * 0.01 * scale + offset,
                (i + 20) * 0.01 * scale + offset,
                0.05 * scale,
                0.9,
            ]
        # Specific shoulders & hips for valid torso normalization
        pose[11] = [-0.2 * scale + offset, 0.5 * scale + offset, 0.0, 0.9]  # Left shoulder
        pose[12] = [0.2 * scale + offset, 0.5 * scale + offset, 0.0, 0.9]  # Right shoulder
        pose[23] = [-0.15 * scale + offset, 0.0 + offset, 0.0, 0.9]  # Left hip
        pose[24] = [0.15 * scale + offset, 0.0 + offset, 0.0, 0.9]  # Right hip
        raw.pose = pose

    if has_left_hand:
        lh = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            lh[i] = [-0.3 * scale + offset, (i * 0.01) * scale + offset, -0.05 * scale]
        raw.left_hand = lh

    if has_right_hand:
        rh = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            rh[i] = [0.3 * scale + offset, (i * 0.01) * scale + offset, -0.05 * scale]
        raw.right_hand = rh

    return raw


def test_raw_landmarks_543_array():
    raw = create_synthetic_raw_landmarks()
    arr = raw.to_543_array()
    assert arr.shape == (NUM_RAW_LANDMARKS, 3)
    assert arr.dtype == np.float32
    assert not np.isnan(arr).any()


def test_landmark_selector_counts():
    selector = LandmarkSelector()
    raw = create_synthetic_raw_landmarks()

    nodes_115, mask_115 = selector.select_115_nodes(raw)
    assert nodes_115.shape == (NUM_GRAPH_NODES, 3)
    assert mask_115.shape == (NUM_GRAPH_NODES,)
    assert mask_115.all()  # All modalities present

    nodes_75, mask_75 = selector.select_75_static_nodes(raw)
    assert nodes_75.shape == (NUM_STATIC_NODES, 3)
    assert mask_75.shape == (NUM_STATIC_NODES,)
    assert mask_75.all()


def test_missing_hand_and_face_handling():
    selector = LandmarkSelector()
    normalizer = LandmarkNormalizer()
    builder = StaticFeatureBuilder()

    # Raw landmarks missing left hand and face
    raw_partial = create_synthetic_raw_landmarks(
        has_face=False, has_left_hand=False
    )
    nodes_115, mask_115 = selector.select_115_nodes(raw_partial)

    # Left hand (0..20) and Face (75..114) should be invalid (False)
    assert not mask_115[0:21].any()
    assert mask_115[21:42].all()  # Right hand valid
    assert mask_115[42:75].all()  # Pose valid
    assert not mask_115[75:115].any()  # Face invalid

    # Normalizer should run cleanly without NaNs
    norm_landmarks = normalizer.normalize(raw_partial)
    assert norm_landmarks.nodes.shape == (115, 5)
    assert not np.isnan(norm_landmarks.nodes).any()

    # Static vector builder
    static_vec = builder.build_from_graph(norm_landmarks)
    assert static_vec.features.shape == (NUM_STATIC_FEATURES,)
    assert static_vec.has_hands is True  # Has right hand


def test_translation_and_scale_invariance():
    normalizer1 = LandmarkNormalizer()
    normalizer2 = LandmarkNormalizer()

    raw1 = create_synthetic_raw_landmarks(offset=0.0, scale=1.0)
    raw2 = create_synthetic_raw_landmarks(offset=100.0, scale=2.5)  # Shifted & scaled

    norm1 = normalizer1.normalize(raw1)
    norm2 = normalizer2.normalize(raw2)

    # Normalized x, y, z coordinates should be virtually identical
    np.testing.assert_allclose(norm1.nodes[:, :3], norm2.nodes[:, :3], atol=1e-4)


def test_first_frame_and_velocity_calculation():
    normalizer = LandmarkNormalizer(target_fps=30.0)

    raw_frame1 = create_synthetic_raw_landmarks(offset=0.0)
    raw_frame2 = create_synthetic_raw_landmarks(offset=0.05)  # Shifted positions

    norm1 = normalizer.normalize(raw_frame1)
    # Frame 0: velocities vx, vy must be zero
    assert np.allclose(norm1.nodes[:, 3:], 0.0)

    norm2 = normalizer.normalize(raw_frame2)
    # Frame 1: velocities vx, vy should reflect shift scaled by target_fps
    assert norm2.nodes[:, 3:].any()  # Non-zero velocities
    assert not np.isnan(norm2.nodes).any()


def test_velocity_fps_scaling():
    normalizer30 = LandmarkNormalizer(target_fps=30.0)
    normalizer60 = LandmarkNormalizer(target_fps=60.0)

    raw1 = create_synthetic_raw_landmarks(offset=0.0)
    raw2 = create_synthetic_raw_landmarks(offset=0.02)

    # Frame 0
    norm1_30 = normalizer30.normalize(raw1)
    norm1_60 = normalizer60.normalize(raw1)
    assert np.allclose(norm1_30.nodes[:, 3:], 0.0)
    assert np.allclose(norm1_60.nodes[:, 3:], 0.0)

    # Frame 1: 60 FPS velocity should be exactly twice 30 FPS velocity
    norm2_30 = normalizer30.normalize(raw2)
    norm2_60 = normalizer60.normalize(raw2)

    v30 = norm2_30.nodes[:, 3:]
    v60 = norm2_60.nodes[:, 3:]
    np.testing.assert_allclose(v60, v30 * 2.0, rtol=1e-5)


def test_static_builder_raw_vs_graph_consistency():
    builder = StaticFeatureBuilder()
    normalizer = LandmarkNormalizer(target_fps=30.0)
    raw = create_synthetic_raw_landmarks(offset=0.5, scale=1.2)

    # Reset normalizer to ensure identical state
    normalizer.reset()
    norm_graph = normalizer.normalize(raw)
    static_from_graph = builder.build_from_graph(norm_graph)

    normalizer.reset()
    static_from_raw = builder.build_from_raw(raw, normalizer=normalizer)

    # Both paths MUST return exact same normalized 225-dimensional static feature vector
    np.testing.assert_allclose(static_from_raw.features, static_from_graph.features)
    np.testing.assert_array_equal(static_from_raw.mask, static_from_graph.mask)
    assert static_from_raw.has_hands == static_from_graph.has_hands


def test_nan_inf_prevention():
    normalizer = LandmarkNormalizer()
    raw = create_synthetic_raw_landmarks()

    # Inject NaN and Inf into raw landmarks
    raw.left_hand[0] = [np.nan, np.inf, -np.inf]
    norm = normalizer.normalize(raw)

    assert not np.isnan(norm.nodes).any()
    assert not np.isinf(norm.nodes).any()


def test_static_feature_output_dimension():
    builder = StaticFeatureBuilder()
    normalizer = LandmarkNormalizer()
    raw = create_synthetic_raw_landmarks()

    norm = normalizer.normalize(raw)
    static_out = builder.build_from_graph(norm)

    assert static_out.features.shape == (NUM_STATIC_FEATURES,)
    assert static_out.features.dtype == np.float32
    assert static_out.mask.shape == (NUM_STATIC_NODES,)


def test_sequence_buffer_tensor_shape_and_padding():
    seq = SequenceBuffer(sequence_length=60)
    normalizer = LandmarkNormalizer()

    # Empty sequence initially
    tensor_obj0 = seq.get_tensor(batch_size=2)
    assert tensor_obj0.tensor.shape == (2, NUM_FEATURE_CHANNELS, 60, NUM_GRAPH_NODES)
    assert tensor_obj0.valid_frames_count == 0
    assert not tensor_obj0.mask.any()

    # Add 5 frames (fewer than 60)
    for _ in range(5):
        raw = create_synthetic_raw_landmarks()
        norm = normalizer.normalize(raw)
        seq.add_frame(norm)

    tensor_obj5 = seq.get_tensor(batch_size=1)
    assert tensor_obj5.tensor.shape == (1, 5, 60, 115)
    assert tensor_obj5.valid_frames_count == 5

    # Check padding: last 5 frames should be valid, first 55 should be masked False
    assert not tensor_obj5.mask[0, :55, :].any()
    assert tensor_obj5.mask[0, 55:, :].all()


def test_extractor_error_handling():
    extractor = MediaPipeHolisticExtractor()

    with pytest.raises(LandmarkExtractionError):
        extractor.extract(None)

    with pytest.raises(LandmarkExtractionError):
        extractor.extract(np.zeros((0, 0, 3), dtype=np.uint8))

    extractor.close()

"""
Phase 2 synthetic landmark smoke test for SIGNASSIST.

Validates end-to-end landmark selection, normalization, static vector generation (225,),
and dynamic sequence tensor construction [B, 5, T, 115].
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.features import (
    DEFAULT_SEQUENCE_LENGTH,
    NUM_FEATURE_CHANNELS,
    NUM_GRAPH_NODES,
    NUM_RAW_LANDMARKS,
    NUM_STATIC_FEATURES,
    LandmarkNormalizer,
    RawLandmarks,
    SequenceBuffer,
    StaticFeatureBuilder,
)
from src.utils.logger import setup_logger


def create_synthetic_frame(frame_idx: int) -> RawLandmarks:
    """Generates synthetic 543 raw landmarks with simulated motion."""
    shift = frame_idx * 0.01

    face = np.random.randn(468, 3).astype(np.float32) * 0.1 + shift
    pose = np.random.randn(33, 4).astype(np.float32) * 0.2 + shift
    pose[:, 3] = 0.95  # visibility

    # Set valid shoulder and hip anchors for torso normalization
    pose[11] = [-0.2 + shift, 0.5 + shift, 0.0, 0.95]  # Left shoulder
    pose[12] = [0.2 + shift, 0.5 + shift, 0.0, 0.95]  # Right shoulder
    pose[23] = [-0.15 + shift, 0.0 + shift, 0.0, 0.95]  # Left hip
    pose[24] = [0.15 + shift, 0.0 + shift, 0.0, 0.95]  # Right hip

    lh = np.random.randn(21, 3).astype(np.float32) * 0.05 + shift
    rh = np.random.randn(21, 3).astype(np.float32) * 0.05 + shift

    return RawLandmarks(
        face=face,
        pose=pose,
        left_hand=lh,
        right_hand=rh,
        has_face=True,
        has_pose=True,
        has_left_hand=True,
        has_right_hand=True,
    )


def main():
    logger = setup_logger(name="signassist_phase2_smoke")
    logger.info("=== SIGNASSIST Phase 2 Feature Pipeline Smoke Test ===")

    normalizer = LandmarkNormalizer()
    static_builder = StaticFeatureBuilder()
    seq_buffer = SequenceBuffer(sequence_length=DEFAULT_SEQUENCE_LENGTH)

    # Process 10 synthetic frames
    num_test_frames = 10
    logger.info(f"Processing {num_test_frames} synthetic frames...")

    for f_idx in range(num_test_frames):
        raw = create_synthetic_frame(f_idx)
        raw_543 = raw.to_543_array()

        norm_landmarks = normalizer.normalize(raw)
        seq_buffer.add_frame(norm_landmarks)

        if f_idx == 0:
            static_vec = static_builder.build_from_graph(norm_landmarks)
            logger.info(f"Frame 0 Raw 543 Array Shape: {raw_543.shape}")
            logger.info(
                f"Frame 0 Normalized Recognition Nodes Shape: {norm_landmarks.nodes.shape} (115 nodes x 5 channels)"
            )
            logger.info(f"Frame 0 Static Feature Vector Shape: {static_vec.features.shape} (225 dim)")

    dynamic_tensor_obj = seq_buffer.get_tensor(batch_size=2)
    tensor = dynamic_tensor_obj.tensor  # (B, 5, T, 115)

    logger.info(
        f"Dynamic Sequence Tensor Shape: {tensor.shape} (B={tensor.shape[0]}, C={tensor.shape[1]}, T={tensor.shape[2]}, V={tensor.shape[3]})"
    )
    logger.info(f"Dynamic Sequence Mask Shape: {dynamic_tensor_obj.mask.shape}")
    logger.info(f"Buffered Frames Count: {dynamic_tensor_obj.valid_frames_count}/{DEFAULT_SEQUENCE_LENGTH}")

    # Assert canonical shapes & sanity checks
    assert raw_543.shape == (NUM_RAW_LANDMARKS, 3)
    assert norm_landmarks.nodes.shape == (NUM_GRAPH_NODES, NUM_FEATURE_CHANNELS)
    assert static_vec.features.shape == (NUM_STATIC_FEATURES,)
    assert tensor.shape == (2, NUM_FEATURE_CHANNELS, DEFAULT_SEQUENCE_LENGTH, NUM_GRAPH_NODES)
    assert not np.isnan(tensor).any()
    assert not np.isinf(tensor).any()

    logger.info("Phase 2 feature pipeline smoke test completed successfully!")


if __name__ == "__main__":
    main()

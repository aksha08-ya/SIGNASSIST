"""Smoke test script for SIGNASSIST Phase 9 Continuous Sign Boundary Detection & Segmentation."""

import os
import sys

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.pipeline import SignBoundaryDetector, TemporalSmoother, BoundaryDetectionOutput
from src.utils.config import ConfigManager


def main() -> None:
    print("=== SIGNASSIST Phase 9 Boundary Detection Smoke Test ===")

    # 1. Config loading
    config_mgr = ConfigManager()
    model_cfg = config_mgr.load_all().model
    b_cfg = model_cfg.boundary_detection
    print(f"[1/4] Loaded BoundaryConfig:")
    print(f"      energy_smoothing_window: {b_cfg.energy_smoothing_window}")
    print(f"      energy_threshold: {b_cfg.energy_threshold}")
    print(f"      rest_pose_threshold: {b_cfg.rest_pose_threshold}")
    print(f"      min_segment_duration: {b_cfg.min_segment_duration}")
    print(f"      min_boundary_separation: {b_cfg.min_boundary_separation}")
    print(f"      active_nodes count: {len(b_cfg.active_nodes)}")

    # 2. Instantiation
    detector = SignBoundaryDetector(config=model_cfg)
    print("\n[2/4] Instantiated SignBoundaryDetector successfully.")

    # 3. Synthetic Continuous Sequence Simulation
    B, T, V = 2, 90, 115
    landmarks = torch.zeros(B, 5, T, V)
    # Set non-zero x, y positions for active nodes
    landmarks[:, 0, :, :42] = 0.5
    landmarks[:, 1, :, :42] = 0.5

    # Sample 0: Rest (0..15), Sign 1 (16..45), Rest/Transition (46..55), Sign 2 (56..85)
    landmarks[0, 3, 16:46, :42] = 0.6  # Sign 1 vx
    landmarks[0, 3, 56:86, :42] = 0.8  # Sign 2 vx

    # Sample 1: Continuous motion with energy dip at frame 45
    landmarks[1, 3, 0:40, :42] = 0.7
    landmarks[1, 3, 40:50, :42] = 0.05  # Energy dip
    landmarks[1, 3, 50:90, :42] = 0.7

    print("\n[3/4] Running Boundary Detector Forward Pass...")
    output: BoundaryDetectionOutput = detector(landmarks)

    print(f"      Output motion_energy shape: {list(output.motion_energy.shape)}")
    print(f"      Output smoothed_energy shape: {list(output.smoothed_energy.shape)}")
    print(f"      Output boundary_flags shape: {list(output.boundary_flags.shape)}")
    print(f"      Output rest_pose_mask shape: {list(output.rest_pose_mask.shape)}")

    print("\n[4/4] Extracted Sign Segments:")
    for b in range(B):
        segs = output.segments[b]
        print(f"      Sample {b} produced {len(segs)} segments:")
        for idx, seg in enumerate(segs):
            print(
                f"        - Segment {idx + 1}: frames {seg.start_frame}..{seg.end_frame} "
                f"(duration={seg.duration}), mean_energy={seg.mean_energy:.4f}, boundary_evidence={seg.boundary_evidence:.4f}"
            )

    print("\n=== Smoke Test Passed Successfully! ===")


if __name__ == "__main__":
    main()

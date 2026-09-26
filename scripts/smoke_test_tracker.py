"""Smoke test script for SIGNASSIST Phase 12 Signer Detection, Tracking & Multi-Person Intelligence."""

import os
import sys

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import ConfigManager
from src.video import BoundingBox, SignerTracker, SignerTrackerOutput


def main() -> None:
    print("=== SIGNASSIST Phase 12 Signer Tracking Smoke Test ===")

    # 1. Config loading
    config_mgr = ConfigManager()
    app_cfg = config_mgr.load_all()
    tracker_cfg = app_cfg.video.tracking

    print(f"[1/4] Loaded TrackerConfig from config:")
    print(f"      iou_threshold: {tracker_cfg.iou_threshold}")
    print(f"      max_age: {tracker_cfg.max_age}")
    print(f"      min_hits: {tracker_cfg.min_hits}")
    print(f"      selection_mode: {tracker_cfg.selection_mode}")
    print(f"      landmark_margin: {tracker_cfg.landmark_margin}")

    # 2. Instantiation
    tracker = SignerTracker(config=app_cfg, min_hits=2)
    print("\n[2/4] Instantiated SignerTracker successfully.")

    # 3. Simulate 2 synthetic people moving across 6 video frames
    print("\n[3/4] Processing Synthetic 2-Person Video Frames...")

    # Person 1 (left signer, moving right)
    # Person 2 (right signer, moving down-left, larger area)
    for frame_idx in range(6):
        shift = frame_idx * 0.015

        person_1 = BoundingBox(
            xmin=0.10 + shift,
            ymin=0.20,
            xmax=0.30 + shift,
            ymax=0.50,
            confidence=0.92,
        )

        person_2 = BoundingBox(
            xmin=0.50 - shift,
            ymin=0.30 + shift,
            xmax=0.85 - shift,
            ymax=0.80 + shift,
            confidence=0.95,
        )

        output: SignerTrackerOutput = tracker.process_frame([person_1, person_2], frame_index=frame_idx)

        print(f"      Frame {frame_idx}:")
        print(f"        Active Tracks count: {len(output.tracks)}")
        for trk in output.tracks:
            print(
                f"          - Track ID {trk.track_id}: state={trk.state.value}, "
                f"center=({trk.bbox.center[0]:.3f}, {trk.bbox.center[1]:.3f}), "
                f"area={trk.bbox.area:.4f}, hits={trk.hits}, is_primary={trk.is_primary}"
            )

        if output.primary_track:
            print(f"        Primary Signer: Track ID {output.primary_track.track_id}")

    # 4. Manual Signer Selection Override
    print("\n[4/4] Testing Manual Signer Selection Lock (Switching to Person 1)...")
    success = tracker.select_signer(1)
    print(f"      Selected Track ID 1 status: {success}")

    p1_frame6 = BoundingBox(xmin=0.19, ymin=0.20, xmax=0.39, ymax=0.50)
    p2_frame6 = BoundingBox(xmin=0.41, ymin=0.39, xmax=0.76, ymax=0.89)

    out6 = tracker.process_frame([p1_frame6, p2_frame6], frame_index=6)
    print(f"      Frame 6 Primary Signer after Manual Selection: Track ID {out6.primary_track.track_id}")
    assert out6.primary_track.track_id == 1

    print("\n=== Phase 12 Signer Tracking Smoke Test Passed Successfully! ===")


if __name__ == "__main__":
    main()

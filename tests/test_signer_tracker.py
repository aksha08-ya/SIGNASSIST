"""Unit tests for SIGNASSIST Phase 12 Signer Detection, Tracking & Multi-Person Intelligence."""

import numpy as np
import pytest
import torch.nn as nn

from src.features.types import RawLandmarks
from src.utils.config import AppConfig, TrackerConfig, VideoConfig
from src.video import (
    BoundingBox,
    KalmanBoxFilter,
    SignerTrack,
    SignerTracker,
    SignerTrackerOutput,
    TrackState,
)


def test_bbox_validation_and_iou():
    """Verify BoundingBox properties, clamping, area, center, and IoU computation."""
    box1 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)
    box2 = BoundingBox(xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.7)

    assert box1.area == pytest.approx(0.16)
    assert box1.center == (0.3, 0.3)
    assert box1.width == pytest.approx(0.4)
    assert box1.height == pytest.approx(0.4)

    # IoU between [0.1..0.5] and [0.3..0.7]
    # Inter: [0.3..0.5] x [0.3..0.7] -> area 0.04
    # Union: 0.16 + 0.16 - 0.04 = 0.28
    # IoU: 0.04 / 0.28 = 0.142857
    iou_val = box1.iou(box2)
    assert iou_val == pytest.approx(0.04 / 0.28, abs=1e-4)

    # Non-overlapping boxes
    box3 = BoundingBox(xmin=0.8, ymin=0.8, xmax=0.9, ymax=0.9)
    assert box1.iou(box3) == 0.0

    # Invalid box where xmin > xmax raises ValueError
    with pytest.raises(ValueError, match="xmin .* cannot be greater than xmax"):
        BoundingBox(xmin=0.8, ymin=0.1, xmax=0.2, ymax=0.5)


def test_kalman_initialization_and_linear_prediction():
    """Verify KalmanBoxFilter state vector initialization and linear prediction under motion."""
    box = BoundingBox(xmin=0.2, ymin=0.2, xmax=0.4, ymax=0.4)
    kf = KalmanBoxFilter(box)

    pred_box = kf.predict()
    assert isinstance(pred_box, BoundingBox)
    assert pred_box.center[0] == pytest.approx(0.3)
    assert pred_box.center[1] == pytest.approx(0.3)

    # Update with shifted position
    new_box = BoundingBox(xmin=0.22, ymin=0.22, xmax=0.42, ymax=0.42)
    updated_box = kf.update(new_box)
    assert updated_box.center[0] > 0.3  # Filter moved towards new detection


def test_track_creation():
    """Verify new detection creates a tentative track."""
    tracker = SignerTracker(min_hits=3)
    box = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)

    output = tracker.process_frame([box], frame_index=0)
    assert len(output.tracks) == 1
    assert output.tracks[0].state == TrackState.TENTATIVE
    assert output.tracks[0].hits == 1


def test_tentative_to_confirmed_transition():
    """Verify track transitions TENTATIVE -> CONFIRMED after min_hits consecutive frames."""
    tracker = SignerTracker(min_hits=3)

    # Frame 0
    box0 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    out0 = tracker.process_frame([box0], frame_index=0)
    assert out0.tracks[0].state == TrackState.TENTATIVE

    # Frame 1
    box1 = BoundingBox(xmin=0.11, ymin=0.11, xmax=0.31, ymax=0.31)
    out1 = tracker.process_frame([box1], frame_index=1)
    assert out1.tracks[0].state == TrackState.TENTATIVE

    # Frame 2 (hit #3) -> CONFIRMED
    box2 = BoundingBox(xmin=0.12, ymin=0.12, xmax=0.32, ymax=0.32)
    out2 = tracker.process_frame([box2], frame_index=2)
    assert out2.tracks[0].state == TrackState.CONFIRMED


def test_iou_association():
    """Verify IoU bipartite matching assigns detections to tracks correctly."""
    tracker = SignerTracker(min_hits=1)

    box_a = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    box_b = BoundingBox(xmin=0.6, ymin=0.6, xmax=0.8, ymax=0.8)

    out = tracker.process_frame([box_a, box_b], frame_index=0)
    assert len(out.tracks) == 2
    id_a = out.tracks[0].track_id
    id_b = out.tracks[1].track_id

    # Next frame slightly shifted
    box_a2 = BoundingBox(xmin=0.11, ymin=0.11, xmax=0.31, ymax=0.31)
    box_b2 = BoundingBox(xmin=0.61, ymin=0.61, xmax=0.81, ymax=0.81)

    out2 = tracker.process_frame([box_a2, box_b2], frame_index=1)
    assert len(out2.tracks) == 2
    assert out2.tracks[0].track_id == id_a
    assert out2.tracks[1].track_id == id_b


def test_multiple_simultaneous_people():
    """Verify multiple simultaneous people maintain stable track IDs across frames."""
    tracker = SignerTracker(min_hits=1)

    # 3 distinct people moving across 5 frames
    for f in range(5):
        shift = f * 0.01
        p1 = BoundingBox(xmin=0.1 + shift, ymin=0.1, xmax=0.3 + shift, ymax=0.3)
        p2 = BoundingBox(xmin=0.4 + shift, ymin=0.4, xmax=0.6 + shift, ymax=0.6)
        p3 = BoundingBox(xmin=0.7, ymin=0.1 + shift, xmax=0.9, ymax=0.3 + shift)

        out = tracker.process_frame([p1, p2, p3], frame_index=f)
        assert len(out.tracks) == 3
        track_ids = [tr.track_id for tr in out.tracks]
        assert track_ids == [1, 2, 3]


def test_missed_detection_to_lost_transition():
    """Verify CONFIRMED track transitions to LOST when detection is missed."""
    tracker = SignerTracker(min_hits=1)

    box = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    out0 = tracker.process_frame([box], frame_index=0)
    assert out0.tracks[0].state == TrackState.CONFIRMED

    # Missed detection in frame 1
    out1 = tracker.process_frame([], frame_index=1)
    assert len(out1.tracks) == 1
    assert out1.tracks[0].state == TrackState.LOST
    assert out1.tracks[0].time_since_update == 1


def test_lost_to_confirmed_reassociation():
    """Verify LOST track returns to CONFIRMED when detection reappears."""
    tracker = SignerTracker(min_hits=1)

    box0 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    tracker.process_frame([box0], frame_index=0)

    # Missed frame 1
    out1 = tracker.process_frame([], frame_index=1)
    assert out1.tracks[0].state == TrackState.LOST

    # Reappears at frame 2
    box2 = BoundingBox(xmin=0.12, ymin=0.12, xmax=0.32, ymax=0.32)
    out2 = tracker.process_frame([box2], frame_index=2)
    assert out2.tracks[0].state == TrackState.CONFIRMED
    assert out2.tracks[0].time_since_update == 0


def test_max_age_to_deleted_transition():
    """Verify LOST track is deleted after max_age frames."""
    tracker = SignerTracker(min_hits=1, max_age=3)

    box = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    tracker.process_frame([box], frame_index=0)

    # Missed frames 1, 2, 3, 4
    for f in range(1, 4):
        out = tracker.process_frame([], frame_index=f)
        assert len(out.tracks) == 1
        assert out.tracks[0].state == TrackState.LOST

    # Frame 4 (time_since_update = 4 > max_age 3) -> track deleted
    out4 = tracker.process_frame([], frame_index=4)
    assert len(out4.tracks) == 0


def test_manual_signer_selection():
    """Verify select_signer(track_id) locks primary flag on specified track."""
    tracker = SignerTracker(min_hits=1)

    box1 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)  # area 0.04
    box2 = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.9, ymax=0.9)  # area 0.25 (larger)

    out = tracker.process_frame([box1, box2], frame_index=0)
    # Default selection mode "largest" selects track 2
    assert out.primary_track.track_id == 2

    # Manually select track 1
    success = tracker.select_signer(1)
    assert success is True

    out2 = tracker.process_frame([box1, box2], frame_index=1)
    assert out2.primary_track.track_id == 1
    assert out2.primary_track.is_primary is True


def test_selection_modes():
    """Test largest, center, and confidence selection modes."""
    # Largest
    t_largest = SignerTracker(min_hits=1, selection_mode="largest")
    box_small = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2, confidence=0.9)
    box_large = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.9, ymax=0.9, confidence=0.5)
    out_l = t_largest.process_frame([box_small, box_large], frame_index=0)
    assert out_l.primary_track.bbox.area == box_large.area

    # Center
    t_center = SignerTracker(min_hits=1, selection_mode="center")
    box_corner = BoundingBox(xmin=0.0, ymin=0.0, xmax=0.5, ymax=0.5)  # center (0.25, 0.25)
    box_middle = BoundingBox(xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.7)  # center (0.5, 0.5)
    out_c = t_center.process_frame([box_corner, box_middle], frame_index=0)
    assert out_c.primary_track.bbox.center == pytest.approx((0.5, 0.5))

    # Confidence
    t_conf = SignerTracker(min_hits=1, selection_mode="confidence")
    box_low = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.9, ymax=0.9, confidence=0.3)
    box_high = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2, confidence=0.95)
    out_conf = t_conf.process_frame([box_low, box_high], frame_index=0)
    assert out_conf.primary_track.bbox.confidence == 0.95


def test_primary_fallback_on_deletion():
    """Verify automatic fallback when manually selected track is deleted."""
    tracker = SignerTracker(min_hits=1, max_age=2)

    box1 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2)  # Track 1
    box2 = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.9, ymax=0.9)  # Track 2

    tracker.process_frame([box1, box2], frame_index=0)
    tracker.select_signer(1)

    # Track 1 disappears in frame 1, 2, 3
    tracker.process_frame([box2], frame_index=1)
    tracker.process_frame([box2], frame_index=2)

    # Track 1 is now deleted due to max_age
    out3 = tracker.process_frame([box2], frame_index=3)
    assert out3.primary_track.track_id == 2


def test_landmark_to_bbox_conversion_missing_modalities():
    """Verify extract_detection_from_landmarks with valid points, missing modalities, and zero/NaN coordinates."""
    tracker = SignerTracker(landmark_margin=0.1)

    # 1. Valid pose landmarks
    raw = RawLandmarks()
    pose_pts = np.zeros((33, 4), dtype=np.float32)
    pose_pts[0] = [0.4, 0.2, 0.0, 1.0]
    pose_pts[1] = [0.6, 0.8, 0.0, 1.0]
    raw.pose = pose_pts
    raw.has_pose = True

    bbox = tracker.extract_detection_from_landmarks(raw)
    assert isinstance(bbox, BoundingBox)
    # raw range x: [0.4, 0.6] (width 0.2 -> margin 0.02 -> [0.38, 0.62])
    # raw range y: [0.2, 0.8] (height 0.6 -> margin 0.06 -> [0.14, 0.86])
    assert bbox.xmin == pytest.approx(0.38, abs=1e-2)
    assert bbox.xmax == pytest.approx(0.62, abs=1e-2)
    assert bbox.ymin == pytest.approx(0.14, abs=1e-2)
    assert bbox.ymax == pytest.approx(0.86, abs=1e-2)

    # 2. Modality not active (has_face = False even if array present)
    raw_inactive = RawLandmarks()
    raw_inactive.face = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)
    raw_inactive.has_face = False
    assert tracker.extract_detection_from_landmarks(raw_inactive) is None

    # 3. Non-finite / zero-filled landmark handling
    raw_nan = RawLandmarks()
    pose_nan = np.zeros((33, 4), dtype=np.float32)
    pose_nan[0] = [np.nan, 0.2, 0.0, 1.0]  # NaN
    pose_nan[1] = [0.0, 0.0, 0.0, 0.0]     # Zero default
    raw_nan.pose = pose_nan
    raw_nan.has_pose = True
    assert tracker.extract_detection_from_landmarks(raw_nan) is None


def test_tracker_reset():
    """Verify reset() clears tracks, filters, and manual selections."""
    tracker = SignerTracker(min_hits=1)
    box = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)

    tracker.process_frame([box], frame_index=0)
    tracker.select_signer(1)
    assert len(tracker.tracks) == 1

    tracker.reset()
    assert len(tracker.tracks) == 0
    assert tracker.manual_primary_id is None


def test_malformed_empty_detections():
    """Verify handling of empty detection lists or malformed inputs without crashing."""
    tracker = SignerTracker()
    out = tracker.process_frame([], frame_index=0)
    assert isinstance(out, SignerTrackerOutput)
    assert len(out.tracks) == 0
    assert out.primary_track is None


def test_optimal_bipartite_matching_vs_greedy():
    """Verify that bipartite matching finds maximum total IoU assignment where greedy fails."""
    # Counterexample cost matrix:
    #      Det 0  Det 1
    # Trk 0 0.90   0.85
    # Trk 1 0.85   0.10
    #
    # Greedy picks (0,0)->0.90, (1,1)->0.10 => total = 1.00
    # Optimal picks (0,1)->0.85, (1,0)->0.85 => total = 1.70

    from src.video.signer_tracker import associate_detections_to_tracks, max_weight_bipartite_matching

    cost_matrix = np.array([[0.90, 0.85], [0.85, 0.10]], dtype=np.float64)
    matches = max_weight_bipartite_matching(cost_matrix)
    assert set(matches) == {(0, 1), (1, 0)}

    class MockBox(BoundingBox):
        def __init__(self, box_id: int):
            super().__init__(xmin=0.0, ymin=0.0, xmax=0.5, ymax=0.5)
            self.box_id = box_id

    class MockTrackBox(MockBox):
        def __init__(self, box_id: int, iou_map: dict):
            super().__init__(box_id)
            self.iou_map = iou_map

        def iou(self, other: BoundingBox) -> float:
            return self.iou_map.get(getattr(other, "box_id", -1), 0.0)

    t0 = MockTrackBox(0, {0: 0.90, 1: 0.85})
    t1 = MockTrackBox(1, {0: 0.85, 1: 0.10})
    d0 = MockBox(0)
    d1 = MockBox(1)

    matches_assoc, unmatched_trks, unmatched_dets = associate_detections_to_tracks(
        [t0, t1], [d0, d1], iou_threshold=0.05
    )
    assert matches_assoc == [(0, 1), (1, 0)]
    assert unmatched_trks == []
    assert unmatched_dets == []


def test_manual_selection_mode_without_selected_id():
    """Verify that selection_mode='manual' with no manual_primary_id yields primary_track=None."""
    tracker = SignerTracker(min_hits=1, selection_mode="manual")
    box1 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    box2 = BoundingBox(xmin=0.4, ymin=0.4, xmax=0.8, ymax=0.8)

    out = tracker.process_frame([box1, box2], frame_index=0)
    assert out.primary_track is None
    for trk in out.tracks:
        assert trk.is_primary is False

    # Setting manual primary ID activates primary_track
    tracker.select_signer(1)
    out2 = tracker.process_frame([box1, box2], frame_index=1)
    assert out2.primary_track is not None
    assert out2.primary_track.track_id == 1
    assert out2.primary_track.is_primary is True


def test_no_recurrent_modules():
    """Verify zero LSTM/GRU/RNN dependencies across all src/video/ modules."""
    import src.video as video_pkg
    for attr_name in dir(video_pkg):
        attr = getattr(video_pkg, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module if hasattr(nn, 'Module') else object):
            for module in attr().modules():
                assert not isinstance(module, (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell))


def test_signer_tracker_canonical_tracks_and_select_signer_signature():
    """Regression test: verify SignerTrackerOutput exposes 'tracks' and SignerTracker accepts 'track_id'."""
    box = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3)
    tracker = SignerTracker()
    out = tracker.process_frame([box], frame_index=0)
    assert hasattr(out, "tracks")
    assert len(out.tracks) == 1
    # Verify select_signer accepts keyword argument track_id
    success = tracker.select_signer(track_id=out.tracks[0].track_id)
    assert success is True
    assert tracker.manual_primary_id == out.tracks[0].track_id



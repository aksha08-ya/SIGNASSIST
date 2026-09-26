"""Signer Detection, Tracking, and Multi-Person Intelligence for SIGNASSIST (Phase 12).

Provides:
- Normalized BoundingBox representation and IoU computations
- NumPy constant-velocity KalmanBoxFilter for state estimation
- Exact Maximum Total IoU Bipartite Association (Kuhn-Munkres algorithm)
- Track lifecycle management (TENTATIVE -> CONFIRMED -> LOST -> DELETED)
- Manual and automatic primary signer selection
- MediaPipe RawLandmarks to BoundingBox extraction utility
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.features.types import RawLandmarks
from src.utils.config import AppConfig, TrackerConfig, VideoConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BoundingBox:
    """Normalized bounding box representation [xmin, ymin, xmax, ymax] in [0.0, 1.0]."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        # Clamp slightly out-of-bound values to [0.0, 1.0]
        self.xmin = float(np.clip(self.xmin, 0.0, 1.0))
        self.ymin = float(np.clip(self.ymin, 0.0, 1.0))
        self.xmax = float(np.clip(self.xmax, 0.0, 1.0))
        self.ymax = float(np.clip(self.ymax, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

        if self.xmin > self.xmax:
            raise ValueError(f"xmin ({self.xmin}) cannot be greater than xmax ({self.xmax}).")
        if self.ymin > self.ymax:
            raise ValueError(f"ymin ({self.ymin}) cannot be greater than ymax ({self.ymax}).")

    @property
    def area(self) -> float:
        """Returns bounding box area."""
        return max(0.0, self.xmax - self.xmin) * max(0.0, self.ymax - self.ymin)

    @property
    def center(self) -> Tuple[float, float]:
        """Returns bounding box center (cx, cy)."""
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    @property
    def width(self) -> float:
        """Returns bounding box width."""
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        """Returns bounding box height."""
        return max(0.0, self.ymax - self.ymin)

    def iou(self, other: "BoundingBox") -> float:
        """Computes Intersection over Union (IoU) with another BoundingBox."""
        inter_xmin = max(self.xmin, other.xmin)
        inter_ymin = max(self.ymin, other.ymin)
        inter_xmax = min(self.xmax, other.xmax)
        inter_ymax = min(self.ymax, other.ymax)

        inter_w = max(0.0, inter_xmax - inter_xmin)
        inter_h = max(0.0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)


class TrackState(Enum):
    """Lifecycle states for tracked signers."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


@dataclass
class SignerTrack:
    """Represents a tracked person/signer across video frames."""

    track_id: int
    bbox: BoundingBox
    state: TrackState
    is_primary: bool = False
    hits: int = 1
    age: int = 1
    time_since_update: int = 0


@dataclass
class SignerTrackerOutput:
    """Structured output from SignerTracker per video frame."""

    tracks: List[SignerTrack] = field(default_factory=list)
    primary_track: Optional[SignerTrack] = None
    frame_index: int = 0


class KalmanBoxFilter:
    """Constant-velocity Kalman Filter for bounding box tracking in NumPy.

    State vector (8x1): [cx, cy, w, h, v_cx, v_cy, v_w, v_h]^T
    Measurement vector (4x1): [cx, cy, w, h]^T
    """

    def __init__(self, bbox: BoundingBox) -> None:
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height

        # State x: [cx, cy, w, h, v_cx, v_cy, v_w, v_h]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(8, 1)

        # State transition matrix F (8x8)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # pos += vel * dt

        # Measurement matrix H (4x8)
        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        # Process noise covariance Q
        self.Q = np.eye(8, dtype=np.float64) * 1e-2
        self.Q[4:, 4:] *= 1e-1  # Velocity process noise

        # Measurement noise covariance R
        self.R = np.eye(4, dtype=np.float64) * 1e-2

        # Error covariance matrix P
        self.P = np.eye(8, dtype=np.float64) * 10.0
        self.P[4:, 4:] *= 100.0  # High uncertainty in initial velocity

    def predict(self) -> BoundingBox:
        """Advances state vector x and covariance P by 1 time step."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self._to_bbox()

    def update(self, bbox: BoundingBox) -> BoundingBox:
        """Updates filter state with a new bounding box measurement."""
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height
        z = np.array([cx, cy, w, h], dtype=np.float64).reshape(4, 1)

        # Innovation
        y = z - np.dot(self.H, self.x)

        # Innovation covariance
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # State and covariance update
        self.x = self.x + np.dot(K, y)
        I = np.eye(8, dtype=np.float64)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

        return self._to_bbox(confidence=bbox.confidence)

    def _to_bbox(self, confidence: float = 1.0) -> BoundingBox:
        """Converts internal state [cx, cy, w, h] to a clamped BoundingBox."""
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        w = max(1e-4, float(self.x[2, 0]))
        h = max(1e-4, float(self.x[3, 0]))

        xmin = max(0.0, min(1.0, cx - w / 2.0))
        ymin = max(0.0, min(1.0, cy - h / 2.0))
        xmax = max(0.0, min(1.0, cx + w / 2.0))
        ymax = max(0.0, min(1.0, cy + h / 2.0))

        if xmin >= xmax:
            xmax = min(1.0, xmin + 1e-3)
        if ymin >= ymax:
            ymax = min(1.0, ymin + 1e-3)

        return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=confidence)


def max_weight_bipartite_matching(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    """Computes exact maximum weight bipartite matching for a non-negative weight matrix."""
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return []

    max_val = float(np.max(cost_matrix)) if cost_matrix.size > 0 else 0.0
    if max_val <= 0:
        return []

    dim = max(N, M)
    C = np.zeros((dim, dim), dtype=np.float64)
    C[:N, :M] = max_val - cost_matrix
    if dim > N:
        C[N:, :] = max_val
    if dim > M:
        C[:, M:] = max_val

    u = np.zeros(dim + 1, dtype=np.float64)
    v = np.zeros(dim + 1, dtype=np.float64)
    p = np.zeros(dim + 1, dtype=int)
    way = np.zeros(dim + 1, dtype=int)

    for i in range(1, dim + 1):
        p[0] = i
        j0 = 0
        minv = np.full(dim + 1, np.inf)
        used = np.full(dim + 1, False)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, dim + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(dim + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments = []
    for j in range(1, dim + 1):
        if p[j] != 0:
            row_idx = p[j] - 1
            col_idx = j - 1
            if row_idx < N and col_idx < M:
                assignments.append((row_idx, col_idx))

    return assignments


def associate_detections_to_tracks(
    tracks_boxes: List[BoundingBox],
    detections: List[BoundingBox],
    iou_threshold: float = 0.3,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Performs maximum total IoU bipartite assignment between tracks and detections."""
    if not tracks_boxes or not detections:
        return [], list(range(len(tracks_boxes))), list(range(len(detections)))

    num_tracks = len(tracks_boxes)
    num_dets = len(detections)

    iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float64)
    for i, trk in enumerate(tracks_boxes):
        for j, det in enumerate(detections):
            score = trk.iou(det)
            if score >= iou_threshold:
                iou_matrix[i, j] = score

    candidate_matches = max_weight_bipartite_matching(iou_matrix)

    matches = []
    matched_tracks = set()
    matched_dets = set()

    candidate_matches.sort(key=lambda pair: pair[0])

    for trk_idx, det_idx in candidate_matches:
        if iou_matrix[trk_idx, det_idx] >= iou_threshold:
            matches.append((trk_idx, det_idx))
            matched_tracks.add(trk_idx)
            matched_dets.add(det_idx)

    unmatched_tracks = [i for i in range(num_tracks) if i not in matched_tracks]
    unmatched_dets = [j for j in range(num_dets) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets


class SignerTracker:
    """Bounding box tracker and primary signer selection engine for SIGNASSIST."""

    def __init__(
        self,
        config: Optional[Union[TrackerConfig, VideoConfig, AppConfig]] = None,
        **kwargs,
    ) -> None:
        if isinstance(config, AppConfig):
            t_cfg = config.video.tracking
        elif isinstance(config, VideoConfig):
            t_cfg = config.tracking
        elif isinstance(config, TrackerConfig):
            t_cfg = config
        elif config is None:
            t_cfg = TrackerConfig()
        else:
            raise TypeError(
                f"Expected config to be TrackerConfig, VideoConfig, or AppConfig, got {type(config)}"
            )

        self.iou_threshold = kwargs.get("iou_threshold", t_cfg.iou_threshold)
        self.max_age = kwargs.get("max_age", t_cfg.max_age)
        self.min_hits = kwargs.get("min_hits", t_cfg.min_hits)
        self.selection_mode = kwargs.get("selection_mode", t_cfg.selection_mode)
        self.landmark_margin = kwargs.get("landmark_margin", t_cfg.landmark_margin)

        self.tracks: List[SignerTrack] = []
        self._filters: Dict[int, KalmanBoxFilter] = {}
        self._next_id: int = 1
        self.manual_primary_id: Optional[int] = None

    def reset(self) -> None:
        """Resets tracker state, tracks, filters, and manual selections."""
        self.tracks.clear()
        self._filters.clear()
        self._next_id = 1
        self.manual_primary_id = None

    def select_signer(self, track_id: Optional[int]) -> bool:
        """Manually sets target signer track ID for isolation."""
        if track_id is None:
            self.manual_primary_id = None
            return True

        exists = any(tr.track_id == track_id and tr.state != TrackState.DELETED for tr in self.tracks)
        self.manual_primary_id = track_id
        return exists

    def extract_detection_from_landmarks(
        self,
        landmarks: RawLandmarks,
        margin: Optional[float] = None,
    ) -> Optional[BoundingBox]:
        """Extracts a normalized BoundingBox from MediaPipe RawLandmarks."""
        if landmarks is None:
            return None

        eff_margin = margin if margin is not None else self.landmark_margin
        coords: List[Tuple[float, float]] = []

        modalities = [
            (landmarks.has_face, landmarks.face),
            (landmarks.has_pose, landmarks.pose),
            (landmarks.has_left_hand, landmarks.left_hand),
            (landmarks.has_right_hand, landmarks.right_hand),
        ]

        for has_mod, arr in modalities:
            if has_mod and arr is not None and len(arr) > 0:
                for row in arr:
                    x, y = float(row[0]), float(row[1])
                    if np.isfinite(x) and np.isfinite(y) and (abs(x) > 1e-6 or abs(y) > 1e-6):
                        coords.append((x, y))

        if len(coords) < 2:
            return None

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        raw_xmin, raw_xmax = min(xs), max(xs)
        raw_ymin, raw_ymax = min(ys), max(ys)

        w = raw_xmax - raw_xmin
        h = raw_ymax - raw_ymin

        margin_x = w * eff_margin
        margin_y = h * eff_margin

        xmin = float(np.clip(raw_xmin - margin_x, 0.0, 1.0))
        xmax = float(np.clip(raw_xmax + margin_x, 0.0, 1.0))
        ymin = float(np.clip(raw_ymin - margin_y, 0.0, 1.0))
        ymax = float(np.clip(raw_ymax + margin_y, 0.0, 1.0))

        if xmin >= xmax or ymin >= ymax:
            return None

        return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=1.0)

    def process_frame(
        self,
        detections: List[BoundingBox],
        frame_index: int = 0,
    ) -> SignerTrackerOutput:
        """Processes detections for a new video frame, updating tracks and primary signer."""
        valid_dets = [d for d in detections if isinstance(d, BoundingBox)]

        # Step 1: Kalman Predict
        predicted_boxes: List[BoundingBox] = []
        for trk in self.tracks:
            kf = self._filters[trk.track_id]
            pred_box = kf.predict()
            predicted_boxes.append(pred_box)

        # Step 2: Optimal Bipartite IoU Association
        matches, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
            predicted_boxes, valid_dets, iou_threshold=self.iou_threshold
        )

        # Step 3: Handle Matched Tracks
        for trk_idx, det_idx in matches:
            trk = self.tracks[trk_idx]
            det = valid_dets[det_idx]
            kf = self._filters[trk.track_id]

            updated_box = kf.update(det)
            trk.bbox = updated_box
            trk.hits += 1
            trk.age += 1
            trk.time_since_update = 0

            if trk.state == TrackState.TENTATIVE and trk.hits >= self.min_hits:
                trk.state = TrackState.CONFIRMED
            elif trk.state == TrackState.LOST:
                trk.state = TrackState.CONFIRMED

        # Step 4: Handle Unmatched Tracks
        for trk_idx in unmatched_tracks:
            trk = self.tracks[trk_idx]
            kf = self._filters[trk.track_id]
            trk.bbox = kf._to_bbox()
            trk.age += 1
            trk.time_since_update += 1

            if trk.state == TrackState.CONFIRMED:
                trk.state = TrackState.LOST

            if trk.time_since_update > self.max_age:
                trk.state = TrackState.DELETED

        # Step 5: Handle Unmatched Detections
        for det_idx in unmatched_dets:
            det = valid_dets[det_idx]
            new_id = self._next_id
            self._next_id += 1

            kf = KalmanBoxFilter(det)
            self._filters[new_id] = kf

            initial_state = (
                TrackState.CONFIRMED if self.min_hits <= 1 else TrackState.TENTATIVE
            )
            new_track = SignerTrack(
                track_id=new_id,
                bbox=det,
                state=initial_state,
                hits=1,
                age=1,
                time_since_update=0,
            )
            self.tracks.append(new_track)

        # Step 6: Remove DELETED tracks
        deleted_ids = {tr.track_id for tr in self.tracks if tr.state == TrackState.DELETED}
        self.tracks = [tr for tr in self.tracks if tr.state != TrackState.DELETED]
        for did in deleted_ids:
            self._filters.pop(did, None)

        # Step 7: Select Primary Signer
        primary_track = self._select_primary_signer()

        for trk in self.tracks:
            trk.is_primary = False

        if primary_track is not None:
            primary_track.is_primary = True

        return SignerTrackerOutput(
            tracks=list(self.tracks),
            primary_track=primary_track,
            frame_index=frame_index,
        )

    def _select_primary_signer(self) -> Optional[SignerTrack]:
        """Determines primary signer track using manual lock or automatic heuristic."""
        active_tracks = [tr for tr in self.tracks if tr.state != TrackState.DELETED]
        if not active_tracks:
            return None

        # 1. If selection mode is explicitly "manual"
        if self.selection_mode == "manual":
            if self.manual_primary_id is not None:
                for tr in active_tracks:
                    if tr.track_id == self.manual_primary_id:
                        return tr
            return None

        # 2. For non-manual modes ("largest", "center", "confidence"):
        if self.manual_primary_id is not None:
            for tr in active_tracks:
                if tr.track_id == self.manual_primary_id:
                    return tr
            logger.info(
                f"Manually selected primary track {self.manual_primary_id} is unavailable. Falling back to auto-selection."
            )

        if self.selection_mode == "center":
            best_track = min(
                active_tracks,
                key=lambda tr: (tr.bbox.center[0] - 0.5) ** 2 + (tr.bbox.center[1] - 0.5) ** 2,
            )
            return best_track
        elif self.selection_mode == "confidence":
            best_track = max(active_tracks, key=lambda tr: tr.bbox.confidence)
            return best_track
        else:
            # Default "largest"
            best_track = max(active_tracks, key=lambda tr: tr.bbox.area)
            return best_track

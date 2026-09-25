"""
Landmark normalizer for SIGNASSIST.

Performs translation centering, distance-invariant scale normalization, and
frame-aware kinematic velocity computation across sequential frames.
"""

from typing import Optional, Tuple
import numpy as np

from src.features.constants import (
    GRAPH_POSE_START,
    NUM_FEATURE_CHANNELS,
    NUM_GRAPH_NODES,
    POSE_LEFT_HIP_IDX,
    POSE_LEFT_SHOULDER_IDX,
    POSE_RIGHT_HIP_IDX,
    POSE_RIGHT_SHOULDER_IDX,
)
from src.features.selector import LandmarkSelector
from src.features.types import RawLandmarks, SelectedGraphLandmarks


class LandmarkNormalizer:
    """Normalizes landmark spatial coordinates and computes frame-aware temporal velocities."""

    def __init__(
        self,
        target_fps: Optional[float] = None,
        eps: float = 1e-6,
    ) -> None:
        self.eps = eps
        if target_fps is None:
            try:
                from src.utils.config import ConfigManager

                cfg = ConfigManager().load_all()
                self.target_fps = float(cfg.video.target_fps)
            except Exception:
                self.target_fps = 30.0
        else:
            self.target_fps = float(target_fps)

        if self.target_fps <= 0:
            self.target_fps = 30.0

        self.selector = LandmarkSelector()
        self._prev_normalized_xy: Optional[np.ndarray] = None
        self._prev_mask: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets temporal velocity tracking history."""
        self._prev_normalized_xy = None
        self._prev_mask = None

    def normalize(self, raw_landmarks: RawLandmarks) -> SelectedGraphLandmarks:
        """Processes raw landmarks into normalized (115, 5) graph nodes.

        Normalization steps:
        1. Select 115 graph nodes (LH: 21, RH: 21, Pose: 33, Face: 40).
        2. Translate coordinates to mid-shoulder origin.
        3. Scale coordinates by torso height (or shoulder width fallback).
        4. Calculate frame-aware kinematic velocities vx, vy relative to previous frame:
           velocity = (current_xy - previous_xy) / dt = displacement * target_fps.
        5. Replace any NaN/Inf values with 0.0.

        Args:
            raw_landmarks: RawLandmarks object containing 543 extracted keypoints.

        Returns:
            SelectedGraphLandmarks containing (115, 5) normalized nodes and (115,) validity mask.
        """
        nodes_3d, mask = self.selector.select_115_nodes(raw_landmarks)

        # 1. Compute translation origin (mid-shoulder point)
        origin = self._compute_translation_origin(nodes_3d, mask, raw_landmarks.has_pose)

        # 2. Compute scale factor (torso height or shoulder width)
        scale = self._compute_scale_factor(nodes_3d, mask, raw_landmarks.has_pose)

        # 3. Center and scale 3D coordinates for valid nodes
        normalized_3d = np.zeros_like(nodes_3d, dtype=np.float32)
        valid_indices = np.where(mask)[0]

        if len(valid_indices) > 0:
            centered = nodes_3d[valid_indices] - origin
            scaled = centered / max(scale, self.eps)
            normalized_3d[valid_indices] = scaled

        # 4. Compute frame-aware kinematic velocities (vx, vy)
        normalized_xy = normalized_3d[:, :2]
        vx_vy = np.zeros((NUM_GRAPH_NODES, 2), dtype=np.float32)

        if self._prev_normalized_xy is not None and self._prev_mask is not None:
            # Velocity is valid only where landmark is present in both current and previous frame
            common_mask = mask & self._prev_mask
            valid_common = np.where(common_mask)[0]
            if len(valid_common) > 0:
                displacement = (
                    normalized_xy[valid_common] - self._prev_normalized_xy[valid_common]
                )
                vx_vy[valid_common] = displacement * self.target_fps

        # Update history for next frame
        self._prev_normalized_xy = normalized_xy.copy()
        self._prev_mask = mask.copy()

        # 5. Assemble final (115, 5) tensor: x, y, z, vx, vy
        nodes_5d = np.hstack([normalized_3d, vx_vy]).astype(np.float32)

        # Guarantee no NaN / Inf values
        nodes_5d = np.nan_to_num(nodes_5d, nan=0.0, posinf=0.0, neginf=0.0)

        return SelectedGraphLandmarks(
            nodes=nodes_5d,
            mask=mask,
            has_left_hand=raw_landmarks.has_left_hand,
            has_right_hand=raw_landmarks.has_right_hand,
            has_pose=raw_landmarks.has_pose,
            has_face=raw_landmarks.has_face,
        )

    def _compute_translation_origin(
        self, nodes_3d: np.ndarray, mask: np.ndarray, has_pose: bool
    ) -> np.ndarray:
        """Computes mid-shoulder origin point (x, y, z)."""
        if not has_pose:
            return np.zeros(3, dtype=np.float32)

        left_shoulder_node = GRAPH_POSE_START + POSE_LEFT_SHOULDER_IDX
        right_shoulder_node = GRAPH_POSE_START + POSE_RIGHT_SHOULDER_IDX

        if mask[left_shoulder_node] and mask[right_shoulder_node]:
            return 0.5 * (nodes_3d[left_shoulder_node] + nodes_3d[right_shoulder_node])
        elif mask[left_shoulder_node]:
            return nodes_3d[left_shoulder_node].copy()
        elif mask[right_shoulder_node]:
            return nodes_3d[right_shoulder_node].copy()
        else:
            return np.zeros(3, dtype=np.float32)

    def _compute_scale_factor(
        self, nodes_3d: np.ndarray, mask: np.ndarray, has_pose: bool
    ) -> float:
        """Computes distance-invariant scale factor from torso height or shoulder width."""
        if not has_pose:
            return 1.0

        left_shoulder_node = GRAPH_POSE_START + POSE_LEFT_SHOULDER_IDX
        right_shoulder_node = GRAPH_POSE_START + POSE_RIGHT_SHOULDER_IDX
        left_hip_node = GRAPH_POSE_START + POSE_LEFT_HIP_IDX
        right_hip_node = GRAPH_POSE_START + POSE_RIGHT_HIP_IDX

        # Try computing torso height: distance between mid-shoulder and mid-hip
        shoulders_valid = mask[left_shoulder_node] and mask[right_shoulder_node]
        hips_valid = mask[left_hip_node] and mask[right_hip_node]

        if shoulders_valid and hips_valid:
            mid_shoulder = 0.5 * (
                nodes_3d[left_shoulder_node] + nodes_3d[right_shoulder_node]
            )
            mid_hip = 0.5 * (nodes_3d[left_hip_node] + nodes_3d[right_hip_node])
            torso_height = float(np.linalg.norm(mid_shoulder - mid_hip))
            if torso_height > self.eps:
                return torso_height

        # Fallback 1: Shoulder width
        if shoulders_valid:
            shoulder_width = float(
                np.linalg.norm(
                    nodes_3d[left_shoulder_node] - nodes_3d[right_shoulder_node]
                )
            )
            if shoulder_width > self.eps:
                return shoulder_width

        # Fallback 2: Default scale
        return 1.0

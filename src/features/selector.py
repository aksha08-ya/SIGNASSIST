"""
Landmark selector for SIGNASSIST.

Extracts the canonical 115 recognition graph nodes and 75 static gesture nodes
from raw 543 MediaPipe Holistic landmarks deterministically.
"""

from typing import Tuple
import numpy as np

from src.features.constants import (
    NUM_GRAPH_NODES,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    NUM_SELECTED_FACE_LANDMARKS,
    NUM_STATIC_NODES,
    SELECTED_FACE_INDICES,
)
from src.features.types import RawLandmarks


class LandmarkSelector:
    """Selects canonical node subsets from raw MediaPipe Holistic landmarks."""

    def select_115_nodes(
        self, raw_landmarks: RawLandmarks
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the 115 recognition graph nodes and validity mask from raw landmarks.

        Order of 115 nodes:
        - Nodes 0..20: Left Hand (21 nodes)
        - Nodes 21..41: Right Hand (21 nodes)
        - Nodes 42..74: Body Pose (33 nodes)
        - Nodes 75..114: Facial Non-Manual Contour (40 selected nodes)

        Returns:
            Tuple of:
            - nodes_3d: np.ndarray shape (115, 3) float32
            - validity_mask: np.ndarray shape (115,) bool
        """
        nodes_3d = np.zeros((NUM_GRAPH_NODES, 3), dtype=np.float32)
        mask = np.zeros(NUM_GRAPH_NODES, dtype=bool)

        # 1. Left Hand (0..20)
        if raw_landmarks.has_left_hand and raw_landmarks.left_hand is not None:
            nodes_3d[0:21] = raw_landmarks.left_hand[:, :3]
            mask[0:21] = True

        # 2. Right Hand (21..41)
        if raw_landmarks.has_right_hand and raw_landmarks.right_hand is not None:
            nodes_3d[21:42] = raw_landmarks.right_hand[:, :3]
            mask[21:42] = True

        # 3. Body Pose (42..74)
        if raw_landmarks.has_pose and raw_landmarks.pose is not None:
            nodes_3d[42:75] = raw_landmarks.pose[:, :3]
            mask[42:75] = True

        # 4. Facial Non-Manual Contour (75..114)
        if raw_landmarks.has_face and raw_landmarks.face is not None:
            selected_face = raw_landmarks.face[SELECTED_FACE_INDICES, :3]
            nodes_3d[75:115] = selected_face
            mask[75:115] = True

        return nodes_3d, mask

    def select_75_static_nodes(
        self, raw_landmarks: RawLandmarks
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts the 75 primary gesture nodes (LH + RH + Pose) and validity mask.

        Order of 75 nodes:
        - Nodes 0..20: Left Hand (21 nodes)
        - Nodes 21..41: Right Hand (21 nodes)
        - Nodes 42..74: Body Pose (33 nodes)

        Returns:
            Tuple of:
            - nodes_3d: np.ndarray shape (75, 3) float32
            - validity_mask: np.ndarray shape (75,) bool
        """
        nodes_3d = np.zeros((NUM_STATIC_NODES, 3), dtype=np.float32)
        mask = np.zeros(NUM_STATIC_NODES, dtype=bool)

        if raw_landmarks.has_left_hand and raw_landmarks.left_hand is not None:
            nodes_3d[0:21] = raw_landmarks.left_hand[:, :3]
            mask[0:21] = True

        if raw_landmarks.has_right_hand and raw_landmarks.right_hand is not None:
            nodes_3d[21:42] = raw_landmarks.right_hand[:, :3]
            mask[21:42] = True

        if raw_landmarks.has_pose and raw_landmarks.pose is not None:
            nodes_3d[42:75] = raw_landmarks.pose[:, :3]
            mask[42:75] = True

        return nodes_3d, mask

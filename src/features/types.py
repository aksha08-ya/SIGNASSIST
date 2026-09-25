"""
Typed dataclasses and explicit data structures for landmark extraction and feature engineering.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.features.constants import (
    NUM_FACE_LANDMARKS,
    NUM_FEATURE_CHANNELS,
    NUM_GRAPH_NODES,
    NUM_HAND_LANDMARKS,
    NUM_POSE_LANDMARKS,
    NUM_RAW_LANDMARKS,
    NUM_STATIC_FEATURES,
    NUM_STATIC_NODES,
)


@dataclass
class RawLandmarks:
    """Raw MediaPipe Holistic output landmarks representation (543 total landmarks)."""

    face: Optional[np.ndarray] = None  # (468, 3) or (468, 4)
    pose: Optional[np.ndarray] = None  # (33, 4) x, y, z, visibility
    left_hand: Optional[np.ndarray] = None  # (21, 3)
    right_hand: Optional[np.ndarray] = None  # (21, 3)
    has_face: bool = False
    has_pose: bool = False
    has_left_hand: bool = False
    has_right_hand: bool = False

    def to_543_array(self) -> np.ndarray:
        """Converts raw landmark components into a canonical (543, 3) float32 array.

        Missing modalities default to zeroed coordinates.
        """
        arr = np.zeros((NUM_RAW_LANDMARKS, 3), dtype=np.float32)

        if self.has_face and self.face is not None:
            arr[0:NUM_FACE_LANDMARKS] = self.face[:, :3]

        if self.has_pose and self.pose is not None:
            arr[NUM_FACE_LANDMARKS : NUM_FACE_LANDMARKS + NUM_POSE_LANDMARKS] = (
                self.pose[:, :3]
            )

        pose_offset = NUM_FACE_LANDMARKS + NUM_POSE_LANDMARKS
        if self.has_left_hand and self.left_hand is not None:
            arr[pose_offset : pose_offset + NUM_HAND_LANDMARKS] = self.left_hand[:, :3]

        lh_offset = pose_offset + NUM_HAND_LANDMARKS
        if self.has_right_hand and self.right_hand is not None:
            arr[lh_offset : lh_offset + NUM_HAND_LANDMARKS] = self.right_hand[:, :3]

        return arr


@dataclass
class SelectedGraphLandmarks:
    """Canonical 115-node spatial-temporal graph landmark representation."""

    nodes: np.ndarray  # (115, 5) x, y, z, vx, vy (float32)
    mask: np.ndarray  # (115,) bool validity mask
    has_left_hand: bool = False
    has_right_hand: bool = False
    has_pose: bool = False
    has_face: bool = False

    def __post_init__(self) -> None:
        if self.nodes.shape != (NUM_GRAPH_NODES, NUM_FEATURE_CHANNELS):
            raise ValueError(
                f"Expected nodes shape ({NUM_GRAPH_NODES}, {NUM_FEATURE_CHANNELS}), got {self.nodes.shape}"
            )
        if self.mask.shape != (NUM_GRAPH_NODES,):
            raise ValueError(
                f"Expected mask shape ({NUM_GRAPH_NODES},), got {self.mask.shape}"
            )


@dataclass
class StaticFeatureVector:
    """Canonical 225-dimensional static feature vector (75 nodes x 3 spatial coords)."""

    features: np.ndarray  # (225,) float32
    mask: np.ndarray  # (75,) bool validity mask
    has_hands: bool = False

    def __post_init__(self) -> None:
        if self.features.shape != (NUM_STATIC_FEATURES,):
            raise ValueError(
                f"Expected static features shape ({NUM_STATIC_FEATURES},), got {self.features.shape}"
            )
        if self.mask.shape != (NUM_STATIC_NODES,):
            raise ValueError(
                f"Expected static mask shape ({NUM_STATIC_NODES},), got {self.mask.shape}"
            )


@dataclass
class DynamicSequenceTensor:
    """Canonical dynamic sequence tensor representation [B, 5, T, 115]."""

    tensor: np.ndarray  # (B, 5, T, 115) float32
    mask: np.ndarray  # (B, T, 115) bool validity mask
    valid_frames_count: int = 0
    sequence_length: int = 60

    def __post_init__(self) -> None:
        if self.tensor.ndim != 4 or self.tensor.shape[1] != NUM_FEATURE_CHANNELS or self.tensor.shape[3] != NUM_GRAPH_NODES:
            raise ValueError(
                f"Expected tensor shape (B, {NUM_FEATURE_CHANNELS}, T, {NUM_GRAPH_NODES}), got {self.tensor.shape}"
            )

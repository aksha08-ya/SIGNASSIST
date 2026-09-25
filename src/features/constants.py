"""
Landmark constants, topology indices, and canonical mapping definitions for SIGNASSIST.

Centralizes indices for MediaPipe Holistic raw landmarks (543 total) and the canonical
115-node recognition graph representation.
"""

from typing import List

# Landmark counts
NUM_RAW_LANDMARKS: int = 543
NUM_FACE_LANDMARKS: int = 468
NUM_POSE_LANDMARKS: int = 33
NUM_HAND_LANDMARKS: int = 21

NUM_SELECTED_FACE_LANDMARKS: int = 40
NUM_GRAPH_NODES: int = 115  # 21 Left Hand + 21 Right Hand + 33 Pose + 40 Face
NUM_STATIC_NODES: int = 75  # 21 Left Hand + 21 Right Hand + 33 Pose
NUM_STATIC_FEATURES: int = 225  # 75 nodes x 3 spatial coordinates (x, y, z)
NUM_FEATURE_CHANNELS: int = 5  # x, y, z, vx, vy
DEFAULT_SEQUENCE_LENGTH: int = 60  # Temporal sliding window size T

# MediaPipe Raw Index Ranges
FACE_START_IDX: int = 0
FACE_END_IDX: int = 468  # Exclusive: 0..467 (468 points)

POSE_START_IDX: int = 468
POSE_END_IDX: int = 501  # Exclusive: 468..500 (33 points)

LH_START_IDX: int = 501
LH_END_IDX: int = 522  # Exclusive: 501..521 (21 points)

RH_START_IDX: int = 522
RH_END_IDX: int = 543  # Exclusive: 522..542 (21 points)

# Selected 40 Facial Non-Manual Contour Landmark Indices (from 468 MediaPipe Face Mesh topology)
SELECTED_FACE_INDICES: List[int] = [
    # Left Eyebrow (5 points)
    70,
    63,
    105,
    66,
    107,
    # Right Eyebrow (5 points)
    336,
    296,
    334,
    293,
    300,
    # Left Eye Contour (6 points)
    33,
    160,
    158,
    133,
    153,
    144,
    # Right Eye Contour (6 points)
    362,
    385,
    387,
    263,
    373,
    380,
    # Outer Lip Contour (12 points)
    61,
    185,
    40,
    39,
    37,
    0,
    267,
    269,
    270,
    291,
    321,
    314,
    # Jawline, Nose, Chin Anchors (6 points)
    1,
    10,
    14,
    152,
    172,
    397,
]

# MediaPipe Pose Anchor Indices (from 33 Pose landmarks)
POSE_LEFT_SHOULDER_IDX: int = 11
POSE_RIGHT_SHOULDER_IDX: int = 12
POSE_LEFT_HIP_IDX: int = 23
POSE_RIGHT_HIP_IDX: int = 24
POSE_NOSE_IDX: int = 0

# Graph Node Range Partitioning (0..114)
GRAPH_LH_START: int = 0
GRAPH_LH_END: int = 21  # 21 nodes (0..20)

GRAPH_RH_START: int = 21
GRAPH_RH_END: int = 42  # 21 nodes (21..41)

GRAPH_POSE_START: int = 42
GRAPH_POSE_END: int = 75  # 33 nodes (42..74)

GRAPH_FACE_START: int = 75
GRAPH_FACE_END: int = 115  # 40 nodes (75..114)

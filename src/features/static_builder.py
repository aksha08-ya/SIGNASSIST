"""
Static feature builder for SIGNASSIST.

Converts normalized keyframe landmarks into the canonical 225-dimensional 1D static
gesture feature vector (75 nodes x 3 spatial coordinates).
"""

from typing import TYPE_CHECKING, Optional
import numpy as np

from src.features.constants import NUM_STATIC_FEATURES, NUM_STATIC_NODES
from src.features.selector import LandmarkSelector
from src.features.types import RawLandmarks, SelectedGraphLandmarks, StaticFeatureVector

if TYPE_CHECKING:
    from src.features.normalizer import LandmarkNormalizer


class StaticFeatureBuilder:
    """Builds canonical 225-dimensional static feature vectors."""

    def __init__(self) -> None:
        self.selector = LandmarkSelector()

    def build_from_graph(
        self, graph_landmarks: SelectedGraphLandmarks
    ) -> StaticFeatureVector:
        """Builds static vector from SelectedGraphLandmarks (115 nodes x 5 channels).

        Extracts spatial coordinates (x, y, z) of the first 75 nodes (21 LH + 21 RH + 33 Pose).

        Args:
            graph_landmarks: SelectedGraphLandmarks instance.

        Returns:
            StaticFeatureVector containing (225,) float32 array and (75,) validity mask.
        """
        # First 75 nodes correspond to LH (21) + RH (21) + Pose (33)
        static_nodes_3d = graph_landmarks.nodes[0:NUM_STATIC_NODES, :3]  # (75, 3)
        static_mask = graph_landmarks.mask[0:NUM_STATIC_NODES]  # (75,)

        features_1d = static_nodes_3d.flatten().astype(np.float32)  # (225,)
        features_1d = np.nan_to_num(features_1d, nan=0.0, posinf=0.0, neginf=0.0)

        has_hands = bool(graph_landmarks.has_left_hand or graph_landmarks.has_right_hand)

        return StaticFeatureVector(
            features=features_1d, mask=static_mask, has_hands=has_hands
        )

    def build_from_raw(
        self,
        raw_landmarks: RawLandmarks,
        normalizer: Optional["LandmarkNormalizer"] = None,
    ) -> StaticFeatureVector:
        """Builds static vector directly from RawLandmarks using standard normalization.

        Args:
            raw_landmarks: RawLandmarks instance containing 543 raw keypoints.
            normalizer: Optional LandmarkNormalizer instance. If None, instantiates a default.

        Returns:
            StaticFeatureVector containing normalized 225-dimensional static vector.
        """
        if normalizer is None:
            from src.features.normalizer import LandmarkNormalizer

            normalizer = LandmarkNormalizer()

        graph_landmarks = normalizer.normalize(raw_landmarks)
        return self.build_from_graph(graph_landmarks)

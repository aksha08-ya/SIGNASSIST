"""
Features package for SIGNASSIST landmark processing, feature extraction, and graph topology.
"""

from src.features.constants import (
    DEFAULT_SEQUENCE_LENGTH,
    NUM_FEATURE_CHANNELS,
    NUM_GRAPH_NODES,
    NUM_RAW_LANDMARKS,
    NUM_SELECTED_FACE_LANDMARKS,
    NUM_STATIC_FEATURES,
    NUM_STATIC_NODES,
    SELECTED_FACE_INDICES,
)
from src.features.extractor import MediaPipeHolisticExtractor
from src.features.graph import (
    GraphPartitions,
    GraphTopology,
    build_graph_edges,
    get_canonical_graph,
)
from src.features.normalizer import LandmarkNormalizer
from src.features.selector import LandmarkSelector
from src.features.sequence import SequenceBuffer
from src.features.static_builder import StaticFeatureBuilder
from src.features.types import (
    DynamicSequenceTensor,
    RawLandmarks,
    SelectedGraphLandmarks,
    StaticFeatureVector,
)

__all__ = [
    "NUM_RAW_LANDMARKS",
    "NUM_GRAPH_NODES",
    "NUM_STATIC_NODES",
    "NUM_STATIC_FEATURES",
    "NUM_FEATURE_CHANNELS",
    "NUM_SELECTED_FACE_LANDMARKS",
    "DEFAULT_SEQUENCE_LENGTH",
    "SELECTED_FACE_INDICES",
    "RawLandmarks",
    "SelectedGraphLandmarks",
    "StaticFeatureVector",
    "DynamicSequenceTensor",
    "MediaPipeHolisticExtractor",
    "LandmarkSelector",
    "LandmarkNormalizer",
    "StaticFeatureBuilder",
    "SequenceBuffer",
    "GraphTopology",
    "GraphPartitions",
    "build_graph_edges",
    "get_canonical_graph",
]
"""
Skeleton Graph Topology and Normalization for SIGNASSIST.

Defines the canonical 115-node spatial skeleton graph, intra-part anatomical connections,
cross-part anchor links, graph partitions, and degree-normalized adjacency matrices.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.features.constants import (
    GRAPH_FACE_END,
    GRAPH_FACE_START,
    GRAPH_LH_END,
    GRAPH_LH_START,
    GRAPH_POSE_END,
    GRAPH_POSE_START,
    GRAPH_RH_END,
    GRAPH_RH_START,
    NUM_GRAPH_NODES,
)


@dataclass(frozen=True)
class GraphPartitions:
    """Node partitions mapping canonical subgraphs to zero-indexed node lists."""

    left_hand: List[int] = field(default_factory=lambda: list(range(GRAPH_LH_START, GRAPH_LH_END)))
    right_hand: List[int] = field(default_factory=lambda: list(range(GRAPH_RH_START, GRAPH_RH_END)))
    pose: List[int] = field(default_factory=lambda: list(range(GRAPH_POSE_START, GRAPH_POSE_END)))
    face: List[int] = field(default_factory=lambda: list(range(GRAPH_FACE_START, GRAPH_FACE_END)))


# Base 21-point MediaPipe Hand Skeleton Topology (relative joint indices 0..20)
HAND_EDGES_RAW: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (0, 17),  # Palm base closure
]

# Base 33-point MediaPipe Pose Skeleton Topology (relative joint indices 0..32)
POSE_EDGES_RAW: List[Tuple[int, int]] = [
    # Facial anchors
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    # Torso / Shoulders / Hips
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Left Arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    # Right Arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Left Leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    # Right Leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
]


class GraphTopology:
    """Canonical 115-node spatial skeleton graph builder and adjacency normalizer."""

    def __init__(
        self,
        num_nodes: int = NUM_GRAPH_NODES,
        self_loops: bool = True,
        cross_part_edges: bool = True,
        normalization: str = "symmetric",
    ) -> None:
        self.num_nodes = num_nodes
        self.self_loops = self_loops
        self.cross_part_edges = cross_part_edges
        self.normalization = normalization
        self.partitions = GraphPartitions()

    def get_edges(
        self,
        include_self_loops: Optional[bool] = None,
        include_cross_part: Optional[bool] = None,
    ) -> List[Tuple[int, int]]:
        """Generates deterministic edge list for the canonical graph.

        Args:
            include_self_loops: Whether to include self-loops (i, i). Defaults to self.self_loops.
            include_cross_part: Whether to include cross-part links. Defaults to self.cross_part_edges.

        Returns:
            List of unique undirected edge tuples (u, v) with u <= v.
        """
        with_self = include_self_loops if include_self_loops is not None else self.self_loops
        with_cross = (
            include_cross_part if include_cross_part is not None else self.cross_part_edges
        )

        edges_set = set()

        # 1. Left Hand Intra-Part Edges (nodes 0..20)
        for u, v in HAND_EDGES_RAW:
            u_lh, v_lh = GRAPH_LH_START + u, GRAPH_LH_START + v
            edges_set.add((min(u_lh, v_lh), max(u_lh, v_lh)))

        # 2. Right Hand Intra-Part Edges (nodes 21..41)
        for u, v in HAND_EDGES_RAW:
            u_rh, v_rh = GRAPH_RH_START + u, GRAPH_RH_START + v
            edges_set.add((min(u_rh, v_rh), max(u_rh, v_rh)))

        # 3. Body Pose Intra-Part Edges (nodes 42..74)
        for u, v in POSE_EDGES_RAW:
            u_pose, v_pose = GRAPH_POSE_START + u, GRAPH_POSE_START + v
            edges_set.add((min(u_pose, v_pose), max(u_pose, v_pose)))

        # 4. Facial Non-Manual Contour Intra-Part Edges (nodes 75..114)
        # Left Eyebrow (5 points: 75..79)
        for i in range(75, 79):
            edges_set.add((i, i + 1))
        # Right Eyebrow (5 points: 80..84)
        for i in range(80, 84):
            edges_set.add((i, i + 1))
        # Left Eye Contour (6 points: 85..90 loop)
        for i in range(85, 90):
            edges_set.add((i, i + 1))
        edges_set.add((85, 90))
        # Right Eye Contour (6 points: 91..96 loop)
        for i in range(91, 96):
            edges_set.add((i, i + 1))
        edges_set.add((91, 96))
        # Outer Lip Contour (12 points: 97..108 loop)
        for i in range(97, 108):
            edges_set.add((i, i + 1))
        edges_set.add((97, 108))

        # Sparse Facial Anchor Topology (nodes 109..114):
        # 109: Nose Tip, 110: Forehead/Glabella, 111: Inner Lip Center,
        # 112: Chin Tip, 113: Jaw Left, 114: Jaw Right
        # Connect anchors to the local facial structures they spatially support:
        facial_anchor_edges = [
            # Forehead (110) <-> Inner Eyebrows (79, 80) & Nose Tip (109)
            (110, 79),
            (110, 80),
            (110, 109),
            # Nose Tip (109) <-> Inner Eyebrows (79, 80) & Lip Top (102)
            (109, 79),
            (109, 80),
            (109, 102),
            # Inner Lip Center (111) <-> Outer Lip Top (102) & Outer Lip Bottom (108)
            (111, 102),
            (111, 108),
            # Chin Tip (112) <-> Outer Lip Bottom (108), Lip Center (111), Jaw Left (113), Jaw Right (114)
            (112, 108),
            (112, 111),
            (112, 113),
            (112, 114),
            # Jaw Left (113) <-> Outer Lip Left Corner (97)
            (113, 97),
            # Jaw Right (114) <-> Outer Lip Right Corner (105)
            (114, 105),
        ]
        for u, v in facial_anchor_edges:
            edges_set.add((min(u, v), max(u, v)))

        # 5. Cross-Part Anchor Links
        if with_cross:
            # LH Wrist (0) <-> Pose Left Wrist (42+15 = 57)
            edges_set.add((0, 57))
            # RH Wrist (21) <-> Pose Right Wrist (42+16 = 58)
            edges_set.add((21, 58))
            # Pose Nose (42+0 = 42) <-> Facial Nose Tip (109)
            edges_set.add((42, 109))
            # Pose Nose/Head (42) <-> Facial Forehead Anchor (110)
            edges_set.add((42, 110))
            # Pose Left Ear (42+7 = 49) <-> Facial Jaw Left Anchor (113)
            edges_set.add((49, 113))
            # Pose Right Ear (42+8 = 50) <-> Facial Jaw Right Anchor (114)
            edges_set.add((50, 114))
            # Pose Mouth Left (42+9 = 51) <-> Facial Outer Lip Left Corner (97)
            edges_set.add((51, 97))
            # Pose Mouth Right (42+10 = 52) <-> Facial Outer Lip Right Corner (105)
            edges_set.add((52, 105))

        # 6. Self-Loops
        if with_self:
            for i in range(self.num_nodes):
                edges_set.add((i, i))

        return sorted(list(edges_set))

    def get_adjacency(
        self,
        include_self_loops: Optional[bool] = None,
        include_cross_part: Optional[bool] = None,
    ) -> np.ndarray:
        """Constructs binary symmetric adjacency matrix A of shape (115, 115).

        Returns:
            np.ndarray of shape (115, 115) float32.
        """
        edges = self.get_edges(
            include_self_loops=include_self_loops, include_cross_part=include_cross_part
        )
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for u, v in edges:
            adj[u, v] = 1.0
            adj[v, u] = 1.0

        return adj

    def get_normalized_adjacency(
        self,
        include_self_loops: Optional[bool] = None,
        include_cross_part: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """Constructs degree-normalized adjacency matrix of shape (115, 115).

        Default normalization mode is symmetric: D^(-1/2) A D^(-1/2).

        Args:
            include_self_loops: Whether to include self-loops.
            include_cross_part: Whether to include cross-part links.
            mode: Normalization mode ("symmetric" or "row").

        Returns:
            np.ndarray of shape (115, 115) float32 containing finite normalized values.
        """
        norm_mode = mode or self.normalization
        adj = self.get_adjacency(
            include_self_loops=include_self_loops, include_cross_part=include_cross_part
        )

        degrees = np.sum(adj, axis=1)  # Degree per node

        if norm_mode == "symmetric":
            # D^(-1/2)
            deg_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
            valid_mask = degrees > 0
            deg_inv_sqrt[valid_mask] = 1.0 / np.sqrt(degrees[valid_mask])
            deg_mat = np.diag(deg_inv_sqrt)
            norm_adj = deg_mat @ adj @ deg_mat
        elif norm_mode == "row":
            # D^(-1) A
            deg_inv = np.zeros_like(degrees, dtype=np.float32)
            valid_mask = degrees > 0
            deg_inv[valid_mask] = 1.0 / degrees[valid_mask]
            deg_mat = np.diag(deg_inv)
            norm_adj = deg_mat @ adj
        else:
            raise ValueError(f"Unsupported normalization mode '{norm_mode}'. Use 'symmetric' or 'row'.")

        return np.nan_to_num(norm_adj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def build_graph_edges(
    self_loops: bool = True, cross_part: bool = True
) -> List[Tuple[int, int]]:
    """Convenience helper returning deterministic edge list for canonical 115-node graph."""
    topo = GraphTopology(self_loops=self_loops, cross_part_edges=cross_part)
    return topo.get_edges()


def get_canonical_graph() -> GraphTopology:
    """Returns a default canonical GraphTopology instance."""
    return GraphTopology()

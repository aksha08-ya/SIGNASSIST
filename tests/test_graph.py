"""
Unit tests for SIGNASSIST Phase 3: Skeleton Graph Topology & Normalization.
"""

import numpy as np
import pytest

from src.features.constants import NUM_GRAPH_NODES
from src.features.graph import (
    HAND_EDGES_RAW,
    GraphPartitions,
    GraphTopology,
    build_graph_edges,
    get_canonical_graph,
)


def test_graph_node_count_and_indices():
    graph = get_canonical_graph()
    assert graph.num_nodes == NUM_GRAPH_NODES == 115

    edges = graph.get_edges(include_self_loops=True, include_cross_part=True)
    for u, v in edges:
        assert 0 <= u < 115
        assert 0 <= v < 115


def test_deterministic_edge_generation():
    edges1 = build_graph_edges(self_loops=True, cross_part=True)
    edges2 = build_graph_edges(self_loops=True, cross_part=True)
    assert edges1 == edges2

    graph1 = GraphTopology()
    graph2 = GraphTopology()
    assert np.array_equal(graph1.get_adjacency(), graph2.get_adjacency())
    assert np.array_equal(graph1.get_normalized_adjacency(), graph2.get_normalized_adjacency())


def test_self_loops_presence_and_no_duplicates():
    graph = get_canonical_graph()
    edges_with_self = graph.get_edges(include_self_loops=True)
    edges_no_self = graph.get_edges(include_self_loops=False)

    self_loops = [e for e in edges_with_self if e[0] == e[1]]
    assert len(self_loops) == 115  # Exactly one self-loop per node
    assert len(edges_with_self) == len(edges_no_self) + 115

    # Verify no duplicate edge pairs in list
    assert len(edges_with_self) == len(set(edges_with_self))


def test_adjacency_matrix_properties():
    graph = get_canonical_graph()
    adj = graph.get_adjacency()

    assert adj.shape == (115, 115)
    assert adj.dtype == np.float32
    assert np.array_equal(adj, adj.T)  # Symmetric

    # Check non-zero degree for every node with self-loops enabled
    degrees = np.sum(adj, axis=1)
    assert (degrees > 0).all()


def test_normalized_adjacency_properties():
    graph = get_canonical_graph()
    norm_adj = graph.get_normalized_adjacency(mode="symmetric")

    assert norm_adj.shape == (115, 115)
    assert norm_adj.dtype == np.float32
    assert np.allclose(norm_adj, norm_adj.T)  # Symmetric
    assert not np.isnan(norm_adj).any()
    assert not np.isinf(norm_adj).any()
    assert (norm_adj >= 0.0).all()
    assert (norm_adj <= 1.0).all()


def test_hand_subgraphs_and_isomorphism():
    graph = get_canonical_graph()
    adj = graph.get_adjacency(include_self_loops=False, include_cross_part=False)

    lh_adj = adj[0:21, 0:21]
    rh_adj = adj[21:42, 21:42]

    # Hand subgraphs must be structurally identical (isomorphic)
    assert np.array_equal(lh_adj, rh_adj)

    # Check that base hand edges are present in LH subgraph
    for u, v in HAND_EDGES_RAW:
        assert lh_adj[u, v] == 1.0
        assert lh_adj[v, u] == 1.0


def test_graph_partitions_completeness_and_disjointness():
    partitions = GraphPartitions()

    lh = set(partitions.left_hand)
    rh = set(partitions.right_hand)
    pose = set(partitions.pose)
    face = set(partitions.face)

    # Check correct size of each partition
    assert len(lh) == 21
    assert len(rh) == 21
    assert len(pose) == 33
    assert len(face) == 40

    # Check pairwise disjoint (no node overlap)
    assert lh.isdisjoint(rh)
    assert lh.isdisjoint(pose)
    assert lh.isdisjoint(face)
    assert rh.isdisjoint(pose)
    assert rh.isdisjoint(face)
    assert pose.isdisjoint(face)

    # Union must cover all 115 nodes exactly [0..114]
    union_nodes = lh | rh | pose | face
    assert union_nodes == set(range(115))


def test_corrected_facial_anchor_topology():
    graph = get_canonical_graph()
    adj = graph.get_adjacency(include_self_loops=False, include_cross_part=False)

    # Verify naive sequential chain (109-110-111-112-113-114) is NOT used
    assert adj[110, 111] == 0.0  # Forehead is not connected directly to inner lip center

    # Verify sparse anchor connections to supported facial features
    assert adj[110, 79] == 1.0  # Forehead <-> Left eyebrow inner
    assert adj[110, 80] == 1.0  # Forehead <-> Right eyebrow inner
    assert adj[110, 109] == 1.0  # Forehead <-> Nose tip
    assert adj[109, 102] == 1.0  # Nose tip <-> Outer lip top
    assert adj[111, 102] == 1.0  # Inner lip center <-> Outer lip top
    assert adj[111, 108] == 1.0  # Inner lip center <-> Outer lip bottom
    assert adj[112, 113] == 1.0  # Chin tip <-> Jaw left
    assert adj[112, 114] == 1.0  # Chin tip <-> Jaw right
    assert adj[113, 97] == 1.0  # Jaw left <-> Outer lip left corner
    assert adj[114, 105] == 1.0  # Jaw right <-> Outer lip right corner


def test_facial_to_pose_cross_part_edges():
    graph = get_canonical_graph()
    adj = graph.get_adjacency(include_self_loops=False, include_cross_part=True)

    # 1. Verify direct shoulder to chin links are removed
    assert adj[53, 112] == 0.0  # Pose left shoulder (53) <-> Chin (112) removed
    assert adj[54, 112] == 0.0  # Pose right shoulder (54) <-> Chin (112) removed

    # 2. Verify wrist to pose arm links exist
    assert adj[0, 57] == 1.0  # LH wrist (0) <-> Pose left wrist (57)
    assert adj[21, 58] == 1.0  # RH wrist (21) <-> Pose right wrist (58)

    # 3. Verify facial head/ear/mouth to pose cross-part links exist
    assert adj[42, 109] == 1.0  # Pose nose (42) <-> Facial nose tip (109)
    assert adj[42, 110] == 1.0  # Pose nose (42) <-> Facial forehead (110)
    assert adj[49, 113] == 1.0  # Pose left ear (49) <-> Facial jaw left (113)
    assert adj[50, 114] == 1.0  # Pose right ear (50) <-> Facial jaw right (114)
    assert adj[51, 97] == 1.0  # Pose mouth left (51) <-> Facial outer lip left corner (97)
    assert adj[52, 105] == 1.0  # Pose mouth right (52) <-> Facial outer lip right corner (105)

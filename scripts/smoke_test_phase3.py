"""
Phase 3 skeleton graph topology smoke test for SIGNASSIST.

Constructs canonical 115-node graph, verifies edge counts, partition sizes,
and degree-normalized adjacency matrix properties.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.features.graph import GraphTopology, get_canonical_graph
from src.utils.logger import setup_logger


def main():
    logger = setup_logger(name="signassist_phase3_smoke")
    logger.info("=== SIGNASSIST Phase 3 Skeleton Graph Topology Smoke Test ===")

    graph = get_canonical_graph()

    edges_no_self = graph.get_edges(include_self_loops=False, include_cross_part=True)
    edges_with_self = graph.get_edges(include_self_loops=True, include_cross_part=True)

    adj = graph.get_adjacency()
    norm_adj = graph.get_normalized_adjacency(mode="symmetric")
    partitions = graph.partitions

    logger.info(f"Canonical Graph Nodes: {graph.num_nodes}")
    logger.info(f"Edge Count (without self-loops): {len(edges_no_self)}")
    logger.info(f"Edge Count (with self-loops): {len(edges_with_self)}")
    logger.info(f"Binary Adjacency Shape: {adj.shape}")
    logger.info(f"Symmetric Normalized Adjacency Shape: {norm_adj.shape}")
    logger.info(
        f"Graph Partitions: Left Hand={len(partitions.left_hand)}, "
        f"Right Hand={len(partitions.right_hand)}, "
        f"Pose={len(partitions.pose)}, "
        f"Face={len(partitions.face)}"
    )

    # Sanity checks
    assert graph.num_nodes == 115
    assert adj.shape == (115, 115)
    assert norm_adj.shape == (115, 115)
    assert not np.isnan(norm_adj).any()
    assert not np.isinf(norm_adj).any()
    assert len(partitions.left_hand) == 21
    assert len(partitions.right_hand) == 21
    assert len(partitions.pose) == 33
    assert len(partitions.face) == 40

    logger.info("Phase 3 skeleton graph topology smoke test completed successfully!")


if __name__ == "__main__":
    main()

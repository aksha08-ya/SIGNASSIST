"""
Spatial Graph Convolutional Network (GCN) for Static ISL Sign Recognition.

Implements a 3-layer spatial GCN operating on the canonical 75-node spatial subgraph
(21 LH + 21 RH + 33 Pose) with symmetric degree normalization, exposing a 256-dimensional
embedding for multimodal fusion and 36-class prediction logits (A-Z, 0-9).
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


@dataclass
class StaticGCNOutput:
    """Output container for StaticGraphGCN forward execution."""

    logits: Any  # torch.Tensor shaped [B, 36]
    embedding: Any  # torch.Tensor shaped [B, 256]
    node_features: Optional[Any] = None  # torch.Tensor shaped [B, 75, 128]


def get_static_75_node_adjacency(
    self_loops: bool = True,
    cross_part: bool = True,
    normalization: str = "symmetric",
) -> np.ndarray:
    """Derives the canonical 75x75 normalized adjacency matrix for static sign recognition.

    Extracts LH (0..20), RH (21..41), and Pose (42..74) nodes and their intra-part/cross-part
    edges from the canonical Phase 3 graph topology, computing symmetric degree normalization.

    Returns:
        np.ndarray shape (75, 75) float32.
    """
    from src.features.graph import get_canonical_graph

    full_graph = get_canonical_graph()
    full_edges = full_graph.get_edges(
        include_self_loops=False, include_cross_part=cross_part
    )

    # Filter edges belonging strictly to the first 75 static nodes (LH + RH + Pose)
    static_edges = [(u, v) for u, v in full_edges if u < 75 and v < 75]

    if self_loops:
        for i in range(75):
            static_edges.append((i, i))

    adj_75 = np.zeros((75, 75), dtype=np.float32)
    for u, v in static_edges:
        adj_75[u, v] = 1.0
        adj_75[v, u] = 1.0

    degrees = np.sum(adj_75, axis=1)
    if normalization == "symmetric":
        deg_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
        valid = degrees > 0
        deg_inv_sqrt[valid] = 1.0 / np.sqrt(degrees[valid])
        deg_mat = np.diag(deg_inv_sqrt)
        norm_adj = deg_mat @ adj_75 @ deg_mat
    else:
        norm_adj = adj_75

    return norm_adj.astype(np.float32)


if _TORCH_AVAILABLE:

    class StaticGraphConv(nn.Module):
        """Spatial Graph Convolutional Layer.

        H^(l+1) = A_norm * H^(l) * W + b
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            """Forward pass for spatial graph convolution.

            Args:
                x: Input feature tensor of shape [B, V, C_in].
                adj_norm: Normalized adjacency matrix of shape [V, V].

            Returns:
                Output feature tensor of shape [B, V, C_out].
            """
            if x.dim() == 3 and x.shape[1] != adj_norm.shape[0] and x.shape[2] == adj_norm.shape[0]:
                x = x.transpose(1, 2)  # [B, C_in, V] -> [B, V, C_in]

            # 1. Feature projection: [B, V, C_in] -> [B, V, C_out]
            h = self.linear(x)

            # 2. Graph propagation: [V, V] @ [B, V, C_out] -> [B, V, C_out]
            out = torch.einsum("vw, bwc -> bvc", adj_norm, h)
            return out

    class StaticGraphGCN(nn.Module):
        """Spatial Graph Convolutional Network for static sign recognition."""

        def __init__(
            self,
            input_dim: int = 225,
            num_nodes: int = 75,
            spatial_channels: int = 3,
            hidden_dims: Optional[List[int]] = None,
            embedding_dim: int = 256,
            num_classes: int = 36,
            dropout: float = 0.2,
            adj_norm: Optional[np.ndarray] = None,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.num_nodes = num_nodes
            self.spatial_channels = spatial_channels
            self.hidden_dims = hidden_dims or [512, 256, 128]
            self.embedding_dim = embedding_dim
            self.num_classes = num_classes
            self.dropout_rate = dropout

            # Load or derive 75x75 static normalized adjacency matrix
            if adj_norm is None:
                raw_adj = get_static_75_node_adjacency()
            else:
                raw_adj = adj_norm

            self.register_buffer("adj_norm", torch.from_numpy(raw_adj).float())

            # GCN Layer 1: 3 -> 512
            self.conv1 = StaticGraphConv(spatial_channels, self.hidden_dims[0])
            self.bn1 = nn.BatchNorm1d(self.hidden_dims[0])
            self.drop1 = nn.Dropout(dropout)

            # GCN Layer 2: 512 -> 256
            self.conv2 = StaticGraphConv(self.hidden_dims[0], self.hidden_dims[1])
            self.bn2 = nn.BatchNorm1d(self.hidden_dims[1])
            self.drop2 = nn.Dropout(dropout)

            # GCN Layer 3: 256 -> 128
            self.conv3 = StaticGraphConv(self.hidden_dims[1], self.hidden_dims[2])
            self.bn3 = nn.BatchNorm1d(self.hidden_dims[2])

            # Embedding Projection Head: 128 -> 256
            self.embedding_fc = nn.Linear(self.hidden_dims[2], embedding_dim)
            self.embedding_bn = nn.BatchNorm1d(embedding_dim)

            # Classification Head: 256 -> 36
            self.classifier = nn.Linear(embedding_dim, num_classes)

            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> StaticGCNOutput:
            """Forward pass for static sign classification.

            Args:
                x: Input tensor shaped [B, 225] or [B, 75, 3].

            Returns:
                StaticGCNOutput containing logits [B, 36] and embedding [B, 256].
            """
            # Shape adapter: convert flattened [B, 225] to node tensor [B, 75, 3]
            if x.dim() == 2 and x.shape[1] == self.input_dim:
                B = x.shape[0]
                x = x.view(B, self.num_nodes, self.spatial_channels)
            elif x.dim() == 3 and (
                x.shape[1] != self.num_nodes or x.shape[2] != self.spatial_channels
            ):
                if x.shape[2] == self.num_nodes and x.shape[1] == self.spatial_channels:
                    x = x.transpose(1, 2)
                else:
                    raise ValueError(
                        f"Expected input shape [B, 225] or [B, {self.num_nodes}, {self.spatial_channels}], got {x.shape}"
                    )

            # Layer 1: [B, 75, 3] -> [B, 75, 512]
            h1 = self.conv1(x, self.adj_norm)
            h1 = self.bn1(h1.transpose(1, 2)).transpose(1, 2)
            h1 = self.drop1(self.relu(h1))

            # Layer 2: [B, 75, 512] -> [B, 75, 256]
            h2 = self.conv2(h1, self.adj_norm)
            h2 = self.bn2(h2.transpose(1, 2)).transpose(1, 2)
            h2 = self.drop2(self.relu(h2))

            # Layer 3: [B, 75, 256] -> [B, 75, 128]
            h3 = self.conv3(h2, self.adj_norm)
            h3 = self.bn3(h3.transpose(1, 2)).transpose(1, 2)
            h3 = self.relu(h3)  # Node features [B, 75, 128]

            # Global Mean Pooling across 75 nodes: [B, 75, 128] -> [B, 128]
            graph_pooled = torch.mean(h3, dim=1)

            # Embedding Projection: [B, 128] -> [B, 256]
            embedding = self.embedding_fc(graph_pooled)
            embedding = self.embedding_bn(embedding)
            embedding = self.relu(embedding)  # [B, 256]

            # Classification Head: [B, 256] -> [B, 36]
            logits = self.classifier(embedding)

            return StaticGCNOutput(
                logits=logits,
                embedding=embedding,
                node_features=h3,
            )

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            """Helper method returning only the 256-dimensional embedding [B, 256]."""
            output = self.forward(x)
            return output.embedding

else:
    # Dummy fallback definitions if PyTorch is absent
    StaticGraphConv = None
    StaticGraphGCN = None

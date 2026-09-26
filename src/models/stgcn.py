"""ST-GCN Spatial-Temporal Backbone for SIGNASSIST Phase 5.1.

Implements the ST-GCN spatial-temporal backbone for dynamic landmark sequence processing.
Input shape: [B, 5, T, 115] (B=batch size, C=5 landmark channels [x, y, z, vx, vy],
T=sequence length, V=115 recognition graph nodes).
Channel progression: 5 -> 64 -> 128 -> 256 -> 256 over 4 ST-GCN layers.
Output shape: [B, 256, T, 115].
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

from src.features.graph import get_canonical_graph
from src.utils.config import DynamicModelConfig, load_config


if _TORCH_AVAILABLE:

    class SpatialGraphConv(nn.Module):
        """Spatial Graph Convolutional Layer.

        Propagates features across graph nodes using a normalized adjacency matrix.
        Computes: H = A_norm * (X * W)
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
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            """Forward pass for spatial graph convolution.

            Args:
                x: Input feature tensor of shape [B, C_in, T, V].
                adj_norm: Normalized adjacency matrix of shape [V, V].

            Returns:
                Output feature tensor of shape [B, C_out, T, V].
            """
            if x.dim() != 4:
                raise ValueError(
                    f"Expected 4D input tensor [B, C_in, T, V], got shape {list(x.shape)}"
                )

            # 1. Feature projection: [B, C_in, T, V] -> [B, C_out, T, V]
            h = self.conv(x)

            # 2. Graph propagation across node dimension V:
            # einsum aggregates node features from neighbor nodes w into node v using A_norm[v, w]
            out = torch.einsum("vw, bctw -> bctv", adj_norm, h)
            return out

    class STGCNBlock(nn.Module):
        """Single Spatial-Temporal Graph Convolutional Layer (ST-GCN Layer).

        Combines spatial graph convolution (over V) and temporal convolution (over T)
        with residual connections, batch normalization, ReLU activation, and dropout.
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 9,
            stride: int = 1,
            dropout: float = 0.3,
            residual: bool = True,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            # Padding to preserve temporal length T: (kernel_size - 1) // 2
            padding = (kernel_size - 1) // 2

            # 1. Spatial Graph Convolution
            self.gcn = SpatialGraphConv(in_channels, out_channels)
            self.bn_gcn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

            # 2. Temporal Convolution
            self.tcn = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                bias=True,
            )
            self.bn_tcn = nn.BatchNorm2d(out_channels)
            self.drop = nn.Dropout(dropout)

            # 3. Residual Path
            if not residual:
                self.residual = lambda x: 0
            elif in_channels == out_channels and stride == 1:
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(1, 1),
                        stride=(stride, 1),
                    ),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            """Forward pass through ST-GCN block.

            Args:
                x: Input tensor shaped [B, C_in, T, V].
                adj_norm: Normalized adjacency matrix tensor [V, V].

            Returns:
                Output tensor shaped [B, C_out, T, V].
            """
            res = self.residual(x)

            # Spatial GCN -> BN -> ReLU
            x_spatial = self.gcn(x, adj_norm)
            x_spatial = self.bn_gcn(x_spatial)
            x_spatial = self.relu(x_spatial)

            # Temporal Conv -> BN -> Dropout
            x_temporal = self.tcn(x_spatial)
            x_temporal = self.bn_tcn(x_temporal)
            x_temporal = self.drop(x_temporal)

            # Add Residual -> ReLU
            out = self.relu(x_temporal + res)
            return out

    class STGCNBackbone(nn.Module):
        """ST-GCN Spatial-Temporal Backbone for SIGNASSIST Phase 5.1.

        Extracts spatial-temporal features from dynamic landmark sequences [B, 5, T, 115]
        producing a feature tensor of shape [B, 256, T, 115].
        """

        def __init__(
            self,
            in_channels: int = 5,
            num_nodes: int = 115,
            sequence_length: int = 60,
            channel_progression: Optional[List[int]] = None,
            dropout: float = 0.3,
            temporal_kernel_size: int = 9,
            adj_norm: Optional[np.ndarray] = None,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.num_nodes = num_nodes
            self.sequence_length = sequence_length
            self.dropout_rate = dropout
            self.temporal_kernel_size = temporal_kernel_size

            # Channel progression default: 5 -> 64 -> 128 -> 256 -> 256
            self.channel_progression = channel_progression or [64, 128, 256, 256]

            # 1. Load canonical Phase 3 115x115 normalized adjacency matrix if not provided
            if adj_norm is None:
                graph = get_canonical_graph()
                raw_adj = graph.get_normalized_adjacency(mode="symmetric")
            else:
                raw_adj = adj_norm

            if raw_adj.shape != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"Adjacency matrix shape {raw_adj.shape} does not match expected shape ({self.num_nodes}, {self.num_nodes})."
                )

            # Register normalized adjacency matrix as a model buffer
            self.register_buffer("adj_norm", torch.from_numpy(raw_adj).float())

            # 2. Build ST-GCN layers according to channel progression
            layers = []
            in_c = self.in_channels
            for out_c in self.channel_progression:
                layers.append(
                    STGCNBlock(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=temporal_kernel_size,
                        stride=1,
                        dropout=dropout,
                        residual=True,
                    )
                )
                in_c = out_c

            self.layers = nn.ModuleList(layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through ST-GCN backbone.

            Args:
                x: Input landmark sequence tensor of shape [B, 5, T, 115].

            Returns:
                Extracted spatial-temporal feature tensor of shape [B, 256, T, 115].
            """
            if x.dim() != 4:
                raise ValueError(
                    f"Expected 4D tensor input [B, C, T, V], got shape {list(x.shape)}"
                )

            B, C, T, V = x.shape
            if C != self.in_channels:
                raise ValueError(
                    f"Expected input with {self.in_channels} channels, got {C} channels (shape {list(x.shape)})"
                )
            if V != self.num_nodes:
                raise ValueError(
                    f"Expected input with {self.num_nodes} nodes, got {V} nodes (shape {list(x.shape)})"
                )

            out = x
            for layer in self.layers:
                out = layer(out, self.adj_norm)

            return out

        @classmethod
        def from_config(cls, config: Optional[DynamicModelConfig] = None) -> "STGCNBackbone":
            """Factory method to instantiate STGCNBackbone from DynamicModelConfig."""
            if config is None:
                app_cfg = load_config()
                config = app_cfg.model.dynamic_model

            return cls(
                in_channels=config.landmark_channels,
                num_nodes=config.num_nodes,
                sequence_length=config.sequence_length,
                dropout=config.dropout,
            )

    # Alias for convenience
    STGCN = STGCNBackbone

else:
    SpatialGraphConv = None
    STGCNBlock = None
    STGCNBackbone = None
    STGCN = None

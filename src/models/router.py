"""Learned Static/Dynamic Routing Network for SIGNASSIST Phase 6.

Implements a lightweight MLP gating network that computes soft routing weights
between static spatial GCN embeddings [B, 256] and dynamic spatial-temporal
Transformer embeddings [B, 256].

Outputs:
- static_weight: [B, 1]
- dynamic_weight: [B, 1]
- probabilities: [B, 2] (static_weight + dynamic_weight = 1)

Strictly excludes LSTM, GRU, or recurrent modules, and uses learned feature representations
rather than hard-coded movement/velocity thresholds.
"""

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None

from src.utils.config import RoutingConfig, load_config


@dataclass
class RouterOutput:
    """Output container for LearnedStaticDynamicRouter forward execution."""

    static_weight: Any  # torch.Tensor shaped [B, 1]
    dynamic_weight: Any  # torch.Tensor shaped [B, 1]
    probabilities: Any  # torch.Tensor shaped [B, 2]
    logits: Optional[Any] = None  # torch.Tensor shaped [B, 2]


if _TORCH_AVAILABLE:

    class LearnedStaticDynamicRouter(nn.Module):
        """Learned MLP Routing Network for static vs dynamic sign recognition pathways.

        Concatenates static embedding [B, 256] and dynamic embedding [B, 256] into [B, 512],
        passing through MLP layers to generate 2 routing probabilities summing to 1.
        """

        def __init__(
            self,
            static_dim: int = 256,
            dynamic_dim: int = 256,
            hidden_dim: int = 128,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.static_dim = static_dim
            self.dynamic_dim = dynamic_dim
            self.hidden_dim = hidden_dim
            self.dropout_rate = dropout

            input_dim = static_dim + dynamic_dim  # 256 + 256 = 512

            # Gating MLP Network
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.relu = nn.ReLU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, 2)  # 2 routing logits: static vs dynamic
            self.softmax = nn.Softmax(dim=-1)

        def forward(
            self,
            static_embedding: torch.Tensor,
            dynamic_embedding: torch.Tensor,
        ) -> RouterOutput:
            """Forward pass through learned router.

            Args:
                static_embedding: Static GCN feature embedding tensor of shape [B, 256].
                dynamic_embedding: Dynamic ST-GCN+Transformer feature embedding tensor of shape [B, 256].

            Returns:
                RouterOutput containing static_weight [B, 1], dynamic_weight [B, 1],
                probabilities [B, 2], and logits [B, 2].
            """
            if static_embedding.dim() != 2:
                raise ValueError(
                    f"Expected 2D static_embedding [B, {self.static_dim}], got shape {list(static_embedding.shape)}"
                )
            if dynamic_embedding.dim() != 2:
                raise ValueError(
                    f"Expected 2D dynamic_embedding [B, {self.dynamic_dim}], got shape {list(dynamic_embedding.shape)}"
                )

            B_stat, C_stat = static_embedding.shape
            B_dyn, C_dyn = dynamic_embedding.shape

            if B_stat != B_dyn:
                raise ValueError(
                    f"Batch size mismatch between static ({B_stat}) and dynamic ({B_dyn}) embeddings."
                )
            if C_stat != self.static_dim:
                raise ValueError(
                    f"Expected static embedding dimension {self.static_dim}, got {C_stat}"
                )
            if C_dyn != self.dynamic_dim:
                raise ValueError(
                    f"Expected dynamic embedding dimension {self.dynamic_dim}, got {C_dyn}"
                )

            # 1. Concatenate feature embeddings along channel dimension: [B, 512]
            combined = torch.cat([static_embedding, dynamic_embedding], dim=1)

            # 2. MLP Forward Pass
            h = self.fc1(combined)

            # Handle batch size 1 in train mode for BatchNorm1d
            if self.training and B_stat == 1:
                self.bn1.eval()
                h = self.bn1(h)
                self.bn1.train()
            else:
                h = self.bn1(h)

            h = self.relu(h)
            h = self.drop(h)
            logits = self.fc2(h)  # [B, 2]

            # 3. Softmax routing weights
            probs = self.softmax(logits)  # [B, 2]

            # Separate routing weights
            static_weight = probs[:, 0:1]  # [B, 1]
            dynamic_weight = probs[:, 1:2]  # [B, 1]

            return RouterOutput(
                static_weight=static_weight,
                dynamic_weight=dynamic_weight,
                probabilities=probs,
                logits=logits,
            )

        @classmethod
        def from_config(
            cls, config: Optional[RoutingConfig] = None
        ) -> "LearnedStaticDynamicRouter":
            """Factory method to instantiate LearnedStaticDynamicRouter from RoutingConfig."""
            if config is None:
                app_cfg = load_config()
                config = app_cfg.model.routing

            return cls(
                static_dim=getattr(config, "static_dim", 256),
                dynamic_dim=getattr(config, "dynamic_dim", 256),
                hidden_dim=getattr(config, "hidden_dim", 128),
                dropout=getattr(config, "dropout", 0.2),
            )

    # Aliases for convenience
    LearnedRouter = LearnedStaticDynamicRouter

else:
    RouterOutput = None
    LearnedStaticDynamicRouter = None
    LearnedRouter = None

"""Multimodal Feature Fusion Engine for SIGNASSIST Phase 7.

Combines 256-dimensional static feature representations (from Static GCN)
and 256-dimensional dynamic feature representations (from ST-GCN + Transformer)
using learned soft-routing weights produced by LearnedStaticDynamicRouter.

Outputs:
- fused_embedding: [B, 256]
- static_embedding: [B, 256]
- dynamic_embedding: [B, 256]
- static_weight: [B, 1]
- dynamic_weight: [B, 1]
- probabilities: [B, 2]
- fused_raw: [B, 256]

Strictly excludes LSTM, GRU, RNN, or any recurrent architectures.
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

from src.models.router import LearnedStaticDynamicRouter, RouterOutput
from src.utils.config import FusionConfig, load_config


@dataclass
class MultimodalFusionOutput:
    """Output container for MultimodalFusionEngine forward execution."""

    fused_embedding: Any  # torch.Tensor shaped [B, 256]
    static_embedding: Any  # torch.Tensor shaped [B, 256]
    dynamic_embedding: Any  # torch.Tensor shaped [B, 256]
    static_weight: Any  # torch.Tensor shaped [B, 1]
    dynamic_weight: Any  # torch.Tensor shaped [B, 1]
    probabilities: Optional[Any] = None  # torch.Tensor shaped [B, 2]
    fused_raw: Optional[Any] = None  # torch.Tensor shaped [B, 256]


if _TORCH_AVAILABLE:

    class MultimodalFusionEngine(nn.Module):
        """Multimodal Feature Fusion Engine combining static and dynamic embeddings using learned routing.

        Formulas:
            weighted_static = static_embedding * static_weight
            weighted_dynamic = dynamic_embedding * dynamic_weight
            fused_raw = weighted_static + weighted_dynamic  # [B, 256]
            fused_embedding = refinement(fused_raw) if use_refinement else fused_raw
        """

        def __init__(
            self,
            input_dim: int = 256,
            output_dim: int = 256,
            dropout: float = 0.2,
            use_refinement: bool = True,
            router: Optional[LearnedStaticDynamicRouter] = None,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.dropout_rate = dropout
            self.use_refinement = use_refinement

            # Router module instance
            if router is not None:
                self.router = router
            else:
                self.router = LearnedStaticDynamicRouter(
                    static_dim=input_dim,
                    dynamic_dim=input_dim,
                    dropout=dropout,
                )

            # Lightweight Refinement Block: 256 -> 256
            if use_refinement:
                self.refinement = nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                self.refinement = nn.Identity()

        def forward(
            self,
            static_embedding: torch.Tensor,
            dynamic_embedding: torch.Tensor,
            static_weight: Optional[torch.Tensor] = None,
            dynamic_weight: Optional[torch.Tensor] = None,
            probabilities: Optional[torch.Tensor] = None,
        ) -> MultimodalFusionOutput:
            """Forward pass through multimodal feature fusion engine.

            Args:
                static_embedding: Static GCN feature tensor of shape [B, 256].
                dynamic_embedding: Dynamic ST-GCN+Transformer feature tensor of shape [B, 256].
                static_weight: Optional static pathway weight of shape [B, 1].
                dynamic_weight: Optional dynamic pathway weight of shape [B, 1].
                probabilities: Optional routing probabilities of shape [B, 2].

            Returns:
                MultimodalFusionOutput containing fused_embedding [B, 256], static_embedding [B, 256],
                dynamic_embedding [B, 256], static_weight [B, 1], dynamic_weight [B, 1],
                probabilities [B, 2], and fused_raw [B, 256].
            """
            if static_embedding.dim() != 2:
                raise ValueError(
                    f"Expected 2D static_embedding tensor [B, {self.input_dim}], got shape {list(static_embedding.shape)}"
                )
            if dynamic_embedding.dim() != 2:
                raise ValueError(
                    f"Expected 2D dynamic_embedding tensor [B, {self.input_dim}], got shape {list(dynamic_embedding.shape)}"
                )

            B_stat, C_stat = static_embedding.shape
            B_dyn, C_dyn = dynamic_embedding.shape

            if B_stat != B_dyn:
                raise ValueError(
                    f"Batch size mismatch between static ({B_stat}) and dynamic ({B_dyn}) embeddings."
                )
            if C_stat != self.input_dim:
                raise ValueError(
                    f"Expected static_embedding channel dimension {self.input_dim}, got {C_stat}"
                )
            if C_dyn != self.input_dim:
                raise ValueError(
                    f"Expected dynamic_embedding channel dimension {self.input_dim}, got {C_dyn}"
                )

            # 1. Determine routing weights
            if static_weight is not None and dynamic_weight is not None:
                s_weight = static_weight
                d_weight = dynamic_weight
                probs = (
                    probabilities
                    if probabilities is not None
                    else torch.cat([s_weight, d_weight], dim=-1)
                )
            elif probabilities is not None:
                probs = probabilities
                s_weight = probs[:, 0:1]
                d_weight = probs[:, 1:2]
            else:
                router_out = self.router(static_embedding, dynamic_embedding)
                s_weight = router_out.static_weight
                d_weight = router_out.dynamic_weight
                probs = router_out.probabilities

            # 2. Weighted feature fusion
            weighted_static = static_embedding * s_weight
            weighted_dynamic = dynamic_embedding * d_weight
            fused_raw = weighted_static + weighted_dynamic  # [B, 256]

            # 3. Refinement projection block
            fused_embedding = self.refinement(fused_raw)

            return MultimodalFusionOutput(
                fused_embedding=fused_embedding,
                static_embedding=static_embedding,
                dynamic_embedding=dynamic_embedding,
                static_weight=s_weight,
                dynamic_weight=d_weight,
                probabilities=probs,
                fused_raw=fused_raw,
            )

        @classmethod
        def from_config(
            cls, config: Optional[FusionConfig] = None
        ) -> "MultimodalFusionEngine":
            """Factory method to instantiate MultimodalFusionEngine from FusionConfig."""
            if config is None:
                app_cfg = load_config()
                config = app_cfg.model.fusion

            return cls(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                dropout=config.dropout,
                use_refinement=config.use_refinement,
            )

    # Aliases for convenience
    MultimodalFusionModule = MultimodalFusionEngine

else:
    MultimodalFusionOutput = None
    MultimodalFusionEngine = None
    MultimodalFusionModule = None

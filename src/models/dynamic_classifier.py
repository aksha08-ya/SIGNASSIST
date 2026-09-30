"""ST-GCN + Transformer Dynamic Sign Recognition Model for SIGNASSIST Phase 5.3.

Integrates the Phase 5.1 ST-GCN spatial-temporal backbone and Phase 5.2 Transformer
temporal encoder into an end-to-end dynamic sign classifier.

Input shape: [B, 5, T, 115] (B=batch size, C=5 landmark channels, T=sequence length, V=115 graph nodes).
ST-GCN backbone output: [B, 256, T, 115].
Transformer temporal encoder output: [B, 256] (fixed-size temporal representation).
Classification head output: [B, 262] (logits across dynamic ISL sign vocabulary).

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

from src.models.stgcn import STGCNBackbone
from src.models.transformer_encoder import TransformerTemporalEncoder
from src.utils.config import DynamicModelConfig, load_config


@dataclass
class DynamicClassifierOutput:
    """Output container for DynamicSignClassifier forward execution."""

    logits: Any  # torch.Tensor shaped [B, 262]
    embedding: Any  # torch.Tensor shaped [B, 256]
    stgcn_features: Optional[Any] = None  # torch.Tensor shaped [B, 256, T, 115]


if _TORCH_AVAILABLE:

    class DynamicSignClassifier(nn.Module):
        """Dynamic Sign Classifier integrating ST-GCN backbone & Transformer temporal encoder.

        Pipeline:
            [B, 5, T, 115] -> STGCNBackbone -> [B, 256, T, 115] -> TransformerTemporalEncoder -> [B, 256] -> Classifier Head -> [B, 262]
        """

        def __init__(
            self,
            in_channels: int = 5,
            num_nodes: int = 115,
            sequence_length: int = 60,
            hidden_dim: int = 256,
            num_stgcn_layers: int = 4,
            num_transformer_layers: int = 4,
            num_heads: int = 8,
            dropout: float = 0.3,
            num_classes: int = 262,
            stgcn_backbone: Optional[STGCNBackbone] = None,
            transformer_encoder: Optional[TransformerTemporalEncoder] = None,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.num_nodes = num_nodes
            self.sequence_length = sequence_length
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.dropout_rate = dropout

            # 1. Instantiate or reuse ST-GCN spatial-temporal backbone
            if stgcn_backbone is not None:
                self.stgcn_backbone = stgcn_backbone
            else:
                self.stgcn_backbone = STGCNBackbone(
                    in_channels=in_channels,
                    num_nodes=num_nodes,
                    sequence_length=sequence_length,
                    dropout=dropout,
                )

            # 2. Instantiate or reuse Transformer temporal encoder
            if transformer_encoder is not None:
                self.transformer_encoder = transformer_encoder
            else:
                self.transformer_encoder = TransformerTemporalEncoder(
                    hidden_dim=hidden_dim,
                    num_transformer_layers=num_transformer_layers,
                    num_heads=num_heads,
                    dropout=dropout,
                    sequence_length=sequence_length,
                )

            # 3. Classification Head: 256 -> 256 -> 262
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> DynamicClassifierOutput:
            """Forward pass through dynamic sign classifier.

            Args:
                x: Input landmark tensor of shape [B, 5, T, 115].
                mask: Optional validity mask of shape [B, T] or [B, T, 115] (True=valid, False=padded).

            Returns:
                DynamicClassifierOutput containing logits [B, 262], embedding [B, 256], and stgcn_features [B, 256, T, 115].
            """
            if x.dim() != 4:
                raise ValueError(
                    f"Expected 4D tensor input [B, C, T, V], got shape {list(x.shape)}"
                )

            B, C, T, V = x.shape
            if C != self.in_channels:
                raise ValueError(
                    f"Expected input with {self.in_channels} channels, got {C} (shape {list(x.shape)})"
                )
            if V != self.num_nodes:
                raise ValueError(
                    f"Expected input with {self.num_nodes} nodes, got {V} (shape {list(x.shape)})"
                )

            # Stage 1: Spatial-Temporal Feature Extraction via ST-GCN [B, 5, T, 115] -> [B, 256, T, 115]
            stgcn_features = self.stgcn_backbone(x)

            # Stage 2: Temporal Encoding via Transformer [B, 256, T, 115] -> [B, 256]
            embedding = self.transformer_encoder(stgcn_features, mask=mask)

            # Stage 3: Classification Head [B, 256] -> [B, 262]
            if self.training and embedding.shape[0] == 1:
                self.classifier[1].eval()
                logits = self.classifier(embedding)
                self.classifier[1].train()
            else:
                logits = self.classifier(embedding)

            return DynamicClassifierOutput(
                logits=logits,
                embedding=embedding,
                stgcn_features=stgcn_features,
            )

        def forward_features(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Extracts only the 256-dimensional temporal embedding vector [B, 256]."""
            out = self.forward(x, mask=mask)
            return out.embedding

        @classmethod
        def from_config(
            cls, config: Optional[DynamicModelConfig] = None
        ) -> "DynamicSignClassifier":
            """Factory method to instantiate DynamicSignClassifier from DynamicModelConfig."""
            if config is None:
                app_cfg = load_config()
                config = app_cfg.model.dynamic_model

            return cls(
                in_channels=config.landmark_channels,
                num_nodes=config.num_nodes,
                sequence_length=config.sequence_length,
                hidden_dim=config.hidden_dim,
                num_stgcn_layers=config.num_stgcn_layers,
                num_transformer_layers=config.num_transformer_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                num_classes=config.num_classes,
            )

    # Aliases for convenience
    DynamicSTGCNTransformer = DynamicSignClassifier

else:
    DynamicClassifierOutput = None
    DynamicSignClassifier = None
    DynamicSTGCNTransformer = None

"""Transformer Temporal Encoder for SIGNASSIST Phase 5.2.

Implements a Transformer-based temporal encoder for processing spatial-temporal
feature sequences produced by the ST-GCN backbone (or feature tensors of shape [B, C, T, V],
[B, C, T], or [B, T, C]).

Output shape: [B, 256] (fixed-size temporal representation).
Strictly excludes LSTM, GRU, and RNN recurrent modules.
"""

import math
from typing import Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None
    nn = None

from src.utils.config import DynamicModelConfig, load_config


if _TORCH_AVAILABLE:

    class SinusoidalPositionalEncoding(nn.Module):
        """Sinusoidal Positional Encoding for variable-length temporal sequences.

        Adds position-dependent signals PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)) to feature representations.
        """

        def __init__(
            self, d_model: int = 256, max_len: int = 1000, dropout: float = 0.1
        ) -> None:
            super().__init__()
            self.d_model = d_model
            self.max_len = max_len
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # Register pe as buffer of shape [1, max_len, d_model] for batch broadcasting
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Adds positional encoding to input tensor [B, T, d_model].

            Args:
                x: Input sequence tensor of shape [B, T, d_model].

            Returns:
                Tensor of shape [B, T, d_model] with positional encodings added.
            """
            T = x.size(1)
            if T > self.max_len:
                raise ValueError(
                    f"Sequence length T={T} exceeds maximum supported length max_len={self.max_len}"
                )

            # Slice positional encoding buffer up to sequence length T
            x = x + self.pe[:, :T, :]
            return self.dropout(x)

    class TransformerTemporalEncoder(nn.Module):
        """Transformer Temporal Encoder Module for SIGNASSIST Phase 5.2.

        Processes spatial-temporal features from ST-GCN backbone [B, 256, T, 115]
        (or pre-pooled sequence [B, 256, T] / [B, T, 256]) using multi-head self-attention
        and produces a fixed-size temporal embedding [B, 256].
        """

        def __init__(
            self,
            hidden_dim: int = 256,
            num_transformer_layers: int = 4,
            num_heads: int = 8,
            dropout: float = 0.3,
            sequence_length: int = 60,
            max_len: int = 1000,
            dim_feedforward: Optional[int] = None,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_transformer_layers = num_transformer_layers
            self.num_heads = num_heads
            self.dropout_rate = dropout
            self.sequence_length = sequence_length
            self.max_len = max_len
            self.dim_feedforward = dim_feedforward or (hidden_dim * 4)

            # Positional encoding
            self.pos_encoder = SinusoidalPositionalEncoding(
                d_model=hidden_dim,
                max_len=max_len,
                dropout=dropout,
            )

            # PyTorch Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_transformer_layers,
                enable_nested_tensor=False,
            )

            # Final Norm & Projection Layer
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def _process_input_shape(self, x: torch.Tensor) -> torch.Tensor:
            """Adapts input tensor of shape [B, C, T, V], [B, C, T], or [B, T, C] to [B, T, C]."""
            if x.dim() == 4:
                # [B, C, T, V] -> spatial mean pooling over V -> [B, C, T]
                x = x.mean(dim=-1)
                # [B, C, T] -> transpose to [B, T, C]
                x = x.transpose(1, 2)
            elif x.dim() == 3:
                # Distinguish [B, C, T] vs [B, T, C]
                if x.shape[1] == self.hidden_dim and x.shape[2] != self.hidden_dim:
                    x = x.transpose(1, 2)
                elif x.shape[2] == self.hidden_dim:
                    pass  # Already [B, T, C]
                else:
                    raise ValueError(
                        f"Expected 3D tensor [B, T, {self.hidden_dim}] or [B, {self.hidden_dim}, T], got {list(x.shape)}"
                    )
            else:
                raise ValueError(
                    f"Expected 3D or 4D input tensor, got shape {list(x.shape)}"
                )

            return x

        def _prepare_masks(
            self,
            B: int,
            T: int,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Constructs PyTorch key_padding_mask [B, T] (True=padded) and valid_mask [B, T, 1] (True=valid).

            Returns:
                Tuple of (key_padding_mask [B, T] bool, valid_mask [B, T, 1] float32).
            """
            if key_padding_mask is not None:
                # Key padding mask provided directly (True = padded, False = valid)
                k_mask = key_padding_mask
                if k_mask.dim() == 3 and k_mask.shape[-1] == 1:
                    k_mask = k_mask.squeeze(-1)
                v_mask = (~k_mask).unsqueeze(-1).float()
            elif mask is not None:
                # Validity mask provided (True = valid, False = padded)
                m = mask
                if m.dim() == 3:
                    # [B, T, V] -> boolean reduction across nodes V (True if any node valid)
                    m = m.any(dim=-1)
                elif m.dim() == 2:
                    pass  # [B, T]
                else:
                    raise ValueError(f"Invalid mask shape {list(m.shape)}")

                k_mask = ~m  # True for padded
                v_mask = m.unsqueeze(-1).float()
            else:
                # No mask provided: all positions valid
                k_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
                v_mask = torch.ones((B, T, 1), dtype=torch.float32, device=device)

            # Edge case safeguard: If a batch row has ALL positions masked (all True in k_mask),
            # unmask index 0 in k_mask to prevent NaN in Transformer attention.
            # Output pooling will multiply by v_mask (which is 0.0), yielding finite 0.0 representation.
            all_masked_rows = k_mask.all(dim=1)
            if all_masked_rows.any():
                k_mask = k_mask.clone()
                k_mask[all_masked_rows, 0] = False

            return k_mask, v_mask

        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass through Transformer Temporal Encoder.

            Args:
                x: Input tensor of shape [B, 256, T, 115], [B, 256, T], or [B, T, 256].
                mask: Optional boolean validity mask (True=valid, False=padded) of shape [B, T] or [B, T, V].
                key_padding_mask: Optional PyTorch key padding mask (True=padded, False=valid) of shape [B, T].

            Returns:
                Fixed-size temporal embedding tensor of shape [B, 256].
            """
            # 1. Adapt input shape to [B, T, C] where C = hidden_dim (256)
            x = self._process_input_shape(x)
            B, T, C = x.shape

            if C != self.hidden_dim:
                raise ValueError(
                    f"Expected channel dimension {self.hidden_dim}, got {C} (input shape {list(x.shape)})"
                )

            # 2. Build masks
            k_mask, v_mask = self._prepare_masks(
                B=B,
                T=T,
                mask=mask,
                key_padding_mask=key_padding_mask,
                device=x.device,
            )

            # 3. Apply positional encoding
            x_pe = self.pos_encoder(x)

            # 4. Pass through PyTorch Transformer Encoder
            enc_out = self.transformer_encoder(x_pe, src_key_padding_mask=k_mask)

            # 5. Masked temporal pooling over T dimension
            masked_enc = enc_out * v_mask
            valid_counts = v_mask.sum(dim=1)  # [B, 1]
            valid_counts_clamped = torch.clamp(valid_counts, min=1.0)

            # Masked average pooling across temporal dimension T -> [B, 256]
            pooled = masked_enc.sum(dim=1) / valid_counts_clamped

            # Layer norm
            output = self.layer_norm(pooled)
            return output

        @classmethod
        def from_config(
            cls, config: Optional[DynamicModelConfig] = None
        ) -> "TransformerTemporalEncoder":
            """Factory method to instantiate TransformerTemporalEncoder from DynamicModelConfig."""
            if config is None:
                app_cfg = load_config()
                config = app_cfg.model.dynamic_model

            return cls(
                hidden_dim=config.hidden_dim,
                num_transformer_layers=config.num_transformer_layers,
                num_heads=config.num_heads,
                dropout=config.dropout,
                sequence_length=config.sequence_length,
            )

else:
    SinusoidalPositionalEncoding = None
    TransformerTemporalEncoder = None

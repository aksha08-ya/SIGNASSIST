"""Smoke test script for Transformer Temporal Encoder (SIGNASSIST Phase 5.2).

Instantiates TransformerTemporalEncoder, passes synthetic ST-GCN backbone output [2, 256, 60, 115],
prints model parameter count and output shape [2, 256], and asserts finite output.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.transformer_encoder import TransformerTemporalEncoder


def main() -> None:
    print("=== SIGNASSIST Phase 5.2 Transformer Temporal Encoder Smoke Test ===")

    # 1. Instantiate model
    model = TransformerTemporalEncoder()
    model.eval()

    # 2. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("TransformerTemporalEncoder successfully constructed.")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 3. Create synthetic input matching ST-GCN backbone output: [B=2, C=256, T=60, V=115]
    batch_size = 2
    channels = 256
    seq_len = 60
    num_nodes = 115
    dummy_stgcn_output = torch.randn(batch_size, channels, seq_len, num_nodes)
    print(f"Input ST-GCN Tensor Shape: {list(dummy_stgcn_output.shape)}")

    # 4. Forward execution
    with torch.no_grad():
        temporal_embedding = model(dummy_stgcn_output)

    print(f"Output Temporal Embedding Shape: {list(temporal_embedding.shape)}")

    # 5. Assertions
    expected_shape = (batch_size, 256)
    assert temporal_embedding.shape == expected_shape, f"Expected shape {expected_shape}, got {temporal_embedding.shape}"
    assert torch.isfinite(temporal_embedding).all(), "Output contains NaN or Inf values!"

    print("=== Transformer Temporal Encoder Smoke Test PASSED ===")


if __name__ == "__main__":
    main()

"""Smoke test script for ST-GCN Spatial-Temporal Backbone (SIGNASSIST Phase 5.1).

Instantiates STGCNBackbone, runs forward execution on synthetic input [2, 5, 60, 115],
prints parameter count and output shape, and asserts finite output.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.stgcn import STGCNBackbone


def main() -> None:
    print("=== SIGNASSIST Phase 5.1 ST-GCN Smoke Test ===")

    # 1. Instantiate model
    model = STGCNBackbone()
    model.eval()

    # 2. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model successfully constructed.")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 3. Create synthetic input [2, 5, 60, 115]
    batch_size = 2
    in_channels = 5
    seq_len = 60
    num_nodes = 115
    dummy_input = torch.randn(batch_size, in_channels, seq_len, num_nodes)
    print(f"Input Shape: {list(dummy_input.shape)}")

    # 4. Forward execution
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output Shape: {list(output.shape)}")

    # 5. Assertions
    expected_shape = (batch_size, 256, seq_len, num_nodes)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert torch.isfinite(output).all(), "Output contains NaN or Inf values!"

    # 6. Verify buffer registration
    buffers = dict(model.named_buffers())
    assert "adj_norm" in buffers, "Normalized adjacency matrix buffer 'adj_norm' missing!"
    print(f"Adjacency Buffer Registered: {buffers['adj_norm'].shape}")

    print("=== ST-GCN Smoke Test PASSED ===")


if __name__ == "__main__":
    main()

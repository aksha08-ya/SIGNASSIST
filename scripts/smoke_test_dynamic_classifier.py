"""Smoke test script for ST-GCN + Transformer Dynamic Sign Classifier (SIGNASSIST Phase 5.3).

Instantiates DynamicSignClassifier, runs forward execution on synthetic input [2, 5, 60, 115],
prints stage intermediate shapes, verifies sub-module presence, prints parameter counts, and asserts finite output.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.dynamic_classifier import DynamicSignClassifier
from src.models.stgcn import STGCNBackbone
from src.models.transformer_encoder import TransformerTemporalEncoder


def main() -> None:
    print("=== SIGNASSIST Phase 5.3 Dynamic Sign Classifier Integration Smoke Test ===")

    # 1. Instantiate model
    model = DynamicSignClassifier()
    model.eval()

    # 2. Check presence of sub-modules
    has_stgcn = hasattr(model, "stgcn_backbone") and isinstance(model.stgcn_backbone, STGCNBackbone)
    has_transformer = hasattr(model, "transformer_encoder") and isinstance(model.transformer_encoder, TransformerTemporalEncoder)
    print(f"ST-GCN Backbone Present: {has_stgcn}")
    print(f"Transformer Encoder Present: {has_transformer}")
    assert has_stgcn, "ST-GCN Backbone is missing!"
    assert has_transformer, "Transformer Temporal Encoder is missing!"

    # 3. Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("DynamicSignClassifier successfully constructed.")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 4. Input [2, 5, 60, 115]
    batch_size = 2
    in_channels = 5
    seq_len = 60
    num_nodes = 115
    dummy_input = torch.randn(batch_size, in_channels, seq_len, num_nodes)
    print(f"Input Shape: {list(dummy_input.shape)}")

    # 5. Forward execution
    with torch.no_grad():
        out = model(dummy_input)

    stgcn_shape = list(out.stgcn_features.shape)
    embedding_shape = list(out.embedding.shape)
    logits_shape = list(out.logits.shape)

    print(f"ST-GCN Output Shape: {stgcn_shape}")
    print(f"Transformer Embedding Shape: {embedding_shape}")
    print(f"Final Logits Shape: {logits_shape}")

    # 6. Assertions
    assert stgcn_shape == [batch_size, 256, seq_len, num_nodes], f"Unexpected ST-GCN output shape: {stgcn_shape}"
    assert embedding_shape == [batch_size, 256], f"Unexpected Transformer embedding shape: {embedding_shape}"
    assert logits_shape == [batch_size, 250], f"Unexpected final logits shape: {logits_shape}"

    assert torch.isfinite(out.logits).all(), "Logits contain NaN or Inf values!"
    assert torch.isfinite(out.embedding).all(), "Embedding contains NaN or Inf values!"
    assert torch.isfinite(out.stgcn_features).all(), "ST-GCN features contain NaN or Inf values!"

    print("=== Dynamic Sign Classifier Integration Smoke Test PASSED ===")


if __name__ == "__main__":
    main()

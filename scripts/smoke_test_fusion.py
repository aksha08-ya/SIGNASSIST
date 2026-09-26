"""Smoke test script for Multimodal Feature Fusion Engine (SIGNASSIST Phase 7).

Instantiates MultimodalFusionEngine, passes synthetic embeddings [2, 256] static & [2, 256] dynamic,
prints parameter counts, input/output shapes, weight shapes, maximum deviation from weight sum = 1.0,
and asserts finite output.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.fusion import MultimodalFusionEngine


def main() -> None:
    print("=== SIGNASSIST Phase 7 Multimodal Feature Fusion Engine Smoke Test ===")

    # 1. Instantiate fusion engine
    engine = MultimodalFusionEngine()
    engine.eval()

    # 2. Count parameters
    total_params = sum(p.numel() for p in engine.parameters())
    trainable_params = sum(p.numel() for p in engine.parameters() if p.requires_grad)
    print("MultimodalFusionEngine successfully constructed.")
    print(f"Total Parameters (Router + Refinement): {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 3. Create synthetic static & dynamic embeddings [2, 256]
    batch_size = 2
    embedding_dim = 256
    static_embedding = torch.randn(batch_size, embedding_dim)
    dynamic_embedding = torch.randn(batch_size, embedding_dim)

    print(f"Static Embedding Shape: {list(static_embedding.shape)}")
    print(f"Dynamic Embedding Shape: {list(dynamic_embedding.shape)}")

    # 4. Forward execution through fusion engine
    with torch.no_grad():
        out = engine(static_embedding, dynamic_embedding)

    static_weight_shape = list(out.static_weight.shape)
    dynamic_weight_shape = list(out.dynamic_weight.shape)
    fused_embedding_shape = list(out.fused_embedding.shape)

    print(f"Static Weight Shape: {static_weight_shape}")
    print(f"Dynamic Weight Shape: {dynamic_weight_shape}")
    print(f"Fused Embedding Shape: {fused_embedding_shape}")

    # 5. Calculate weight sum deviation
    weight_sum = out.static_weight + out.dynamic_weight
    max_dev = torch.max(torch.abs(weight_sum - 1.0)).item()
    print(f"Maximum Deviation from Weight Sum = 1.0: {max_dev:.8e}")

    is_finite = torch.isfinite(out.fused_embedding).all().item()
    print(f"Is Fused Embedding Strictly Finite: {is_finite}")

    # 6. Assertions
    assert static_weight_shape == [batch_size, 1], f"Unexpected static weight shape: {static_weight_shape}"
    assert dynamic_weight_shape == [batch_size, 1], f"Unexpected dynamic weight shape: {dynamic_weight_shape}"
    assert fused_embedding_shape == [batch_size, embedding_dim], f"Unexpected fused embedding shape: {fused_embedding_shape}"
    assert max_dev < 1e-6, f"Weight sum deviation {max_dev} exceeds tolerance 1e-6!"
    assert is_finite, "Fused embedding contains NaN or Inf values!"

    print("=== Multimodal Feature Fusion Engine Smoke Test PASSED ===")


if __name__ == "__main__":
    main()

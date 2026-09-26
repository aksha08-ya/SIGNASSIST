"""Smoke test script for Learned Static/Dynamic Router (SIGNASSIST Phase 6).

Instantiates LearnedStaticDynamicRouter, passes synthetic embeddings [2, 256] static & [2, 256] dynamic,
prints parameter count, weight shapes, example probabilities, and asserts weight sum = 1.0.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.router import LearnedStaticDynamicRouter


def main() -> None:
    print("=== SIGNASSIST Phase 6 Learned Static/Dynamic Router Smoke Test ===")

    # 1. Instantiate router
    router = LearnedStaticDynamicRouter()
    router.eval()

    # 2. Count parameters
    total_params = sum(p.numel() for p in router.parameters())
    trainable_params = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print("LearnedStaticDynamicRouter successfully constructed.")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 3. Create synthetic embeddings [2, 256]
    batch_size = 2
    embedding_dim = 256
    static_embedding = torch.randn(batch_size, embedding_dim)
    dynamic_embedding = torch.randn(batch_size, embedding_dim)

    print(f"Static Embedding Shape: {list(static_embedding.shape)}")
    print(f"Dynamic Embedding Shape: {list(dynamic_embedding.shape)}")

    # 4. Run router forward pass
    with torch.no_grad():
        out = router(static_embedding, dynamic_embedding)

    static_weight_shape = list(out.static_weight.shape)
    dynamic_weight_shape = list(out.dynamic_weight.shape)
    prob_shape = list(out.probabilities.shape)

    print(f"Static Weight Shape: {static_weight_shape}")
    print(f"Dynamic Weight Shape: {dynamic_weight_shape}")
    print(f"Probability Shape: {prob_shape}")
    print(f"Example Routing Probabilities [static, dynamic]:\n{out.probabilities.numpy()}")

    # 5. Calculate weight sum deviation
    weight_sum = out.static_weight + out.dynamic_weight
    max_dev = torch.max(torch.abs(weight_sum - 1.0)).item()
    print(f"Maximum Deviation from Weight Sum = 1.0: {max_dev:.8e}")

    # 6. Assertions
    assert static_weight_shape == [batch_size, 1], f"Unexpected static weight shape: {static_weight_shape}"
    assert dynamic_weight_shape == [batch_size, 1], f"Unexpected dynamic weight shape: {dynamic_weight_shape}"
    assert prob_shape == [batch_size, 2], f"Unexpected probability shape: {prob_shape}"
    assert max_dev < 1e-6, f"Weight sum deviation {max_dev} exceeds tolerance 1e-6!"
    assert torch.isfinite(out.probabilities).all(), "Probabilities contain NaN or Inf!"

    print("=== Learned Router Smoke Test PASSED ===")


if __name__ == "__main__":
    main()

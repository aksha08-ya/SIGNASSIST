"""
Phase 4 static sign recognition GNN pipeline smoke test for SIGNASSIST.

Instantiates StaticGraphGCN, verifies 75x75 static graph derivation, executes forward
inference on synthetic inputs [2, 225], and checks 256-dim embeddings and 36-class logits.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    import torch
    from src.models.static_gcn import StaticGraphGCN, get_static_75_node_adjacency
    from src.utils.logger import setup_logger

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def main():
    if not _TORCH_AVAILABLE:
        print("PyTorch is not available. Smoke test aborted.")
        sys.exit(1)

    logger = setup_logger(name="signassist_phase4_smoke")
    logger.info("=== SIGNASSIST Phase 4 Static Spatial GNN Smoke Test ===")

    # 1. Derive 75-node static adjacency
    adj_75 = get_static_75_node_adjacency()
    logger.info(f"Derived Static Subgraph Adjacency Shape: {adj_75.shape}")

    # 2. Instantiate StaticGraphGCN model
    model = StaticGraphGCN()
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"StaticGraphGCN Model Instantiated successfully. "
        f"Total Params: {total_params:,}, Trainable: {trainable_params:,}"
    )

    # 3. Create synthetic input [B=2, 225]
    B = 2
    x_synthetic = torch.randn(B, 225)
    logger.info(f"Synthetic Input Shape: {list(x_synthetic.shape)}")

    # 4. Run forward inference
    with torch.no_grad():
        out = model(x_synthetic)

    logits = out.logits
    embedding = out.embedding
    predicted_classes = torch.argmax(logits, dim=1).tolist()

    logger.info(f"Multimodal Fusion Embedding Shape: {list(embedding.shape)}")
    logger.info(f"Classification Logits Shape: {list(logits.shape)}")
    logger.info(f"Predicted Class Indices: {predicted_classes}")

    # 5. Assert finite output checks
    assert adj_75.shape == (75, 75)
    assert embedding.shape == (2, 256)
    assert logits.shape == (2, 36)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(embedding).any()
    assert not torch.isinf(embedding).any()

    logger.info("Phase 4 static spatial GNN smoke test completed successfully!")


if __name__ == "__main__":
    main()

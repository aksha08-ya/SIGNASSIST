"""
Unit tests for SIGNASSIST Phase 4: Static Sign Recognition Pipeline using Spatial GNN.
"""

import numpy as np
import pytest

from src.models.static_gcn import (
    StaticGCNOutput,
    StaticGraphConv,
    StaticGraphGCN,
    get_static_75_node_adjacency,
)

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_static_75_node_graph_adjacency():
    adj_75 = get_static_75_node_adjacency(self_loops=True, normalization="symmetric")
    assert adj_75.shape == (75, 75)
    assert adj_75.dtype == np.float32
    assert np.allclose(adj_75, adj_75.T)  # Symmetric
    assert not np.isnan(adj_75).any()
    assert not np.isinf(adj_75).any()

    # All nodes must have non-zero degree with self-loops
    degrees = np.sum(adj_75 > 0, axis=1)
    assert (degrees > 0).all()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_static_gcn_model_construction():
    model = StaticGraphGCN(
        input_dim=225,
        num_nodes=75,
        spatial_channels=3,
        hidden_dims=[512, 256, 128],
        embedding_dim=256,
        num_classes=36,
    )

    assert isinstance(model.conv1, StaticGraphConv)
    assert isinstance(model.conv2, StaticGraphConv)
    assert isinstance(model.conv3, StaticGraphConv)
    assert model.hidden_dims == [512, 256, 128]
    assert model.embedding_dim == 256
    assert model.num_classes == 36


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_flattened_input_forward():
    model = StaticGraphGCN()
    model.eval()

    # Flattened input [B, 225]
    for B in [1, 2, 8]:
        x_flat = torch.randn(B, 225)
        with torch.no_grad():
            out = model(x_flat)

        assert isinstance(out, StaticGCNOutput)
        assert out.logits.shape == (B, 36)
        assert out.embedding.shape == (B, 256)
        assert out.node_features.shape == (B, 75, 128)
        assert not torch.isnan(out.logits).any()
        assert not torch.isnan(out.embedding).any()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_node_input_forward():
    model = StaticGraphGCN()
    model.eval()

    # Node input [B, 75, 3]
    for B in [1, 2, 8]:
        x_node = torch.randn(B, 75, 3)
        with torch.no_grad():
            out = model(x_node)

        assert out.logits.shape == (B, 36)
        assert out.embedding.shape == (B, 256)
        assert not torch.isnan(out.logits).any()
        assert not torch.isnan(out.embedding).any()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_gradient_flow():
    model = StaticGraphGCN()
    model.train()

    x = torch.randn(4, 225, requires_grad=True)
    out = model(x)

    loss = out.logits.sum() + out.embedding.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_deterministic_eval_output():
    model = StaticGraphGCN()
    model.eval()

    x = torch.randn(2, 225)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1.logits, out2.logits)
    assert torch.allclose(out1.embedding, out2.embedding)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_forward_features_api():
    model = StaticGraphGCN()
    model.eval()

    x = torch.randn(2, 225)
    with torch.no_grad():
        embedding = model.forward_features(x)

    assert embedding.shape == (2, 256)
    assert not torch.isnan(embedding).any()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_flattened_and_node_inputs_are_equivalent():
    """The two supported input representations must produce identical outputs."""
    model = StaticGraphGCN()
    model.eval()

    x_node = torch.randn(2, 75, 3)
    x_flat = x_node.reshape(2, 225)

    with torch.no_grad():
        out_node = model(x_node)
        out_flat = model(x_flat)

    assert torch.allclose(out_node.logits, out_flat.logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_node.embedding, out_flat.embedding, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_node.node_features, out_flat.node_features, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch is required for GCN tests")
def test_registered_adjacency_matches_canonical_graph():
    """The model must register the canonical 75-node normalized adjacency."""
    expected = get_static_75_node_adjacency(
        self_loops=True,
        cross_part=True,
        normalization="symmetric",
    )

    model = StaticGraphGCN()
    actual = model.adj_norm.detach().cpu().numpy()

    assert actual.shape == (75, 75)
    assert np.allclose(actual, expected, atol=1e-7)

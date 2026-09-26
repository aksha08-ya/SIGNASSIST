"""Unit tests for ST-GCN Spatial-Temporal Backbone (SIGNASSIST Phase 5.1).

Verifies:
- Canonical adjacency is 115x115 and symmetric;
- Model construction;
- Input shape [B, 5, 60, 115];
- Output shape [B, 256, 60, 115];
- Batch sizes 1, 2, 8;
- No NaN / Inf in outputs or gradients;
- Gradient flow through all trainable parameters;
- Deterministic eval output;
- Temporal dimension is preserved;
- Adjacency matrix is registered as a model buffer;
- Model does not contain LSTM or GRU modules;
- Package exports via src.models.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.features.graph import get_canonical_graph
from src.models.stgcn import STGCN, STGCNBackbone, STGCNBlock, SpatialGraphConv


def test_stgcn_package_exports():
    """Verifies that STGCNBackbone and STGCN import successfully from src.models and STGCN is the alias for STGCNBackbone."""
    from src.models import STGCN as STGCN_pkg
    from src.models import STGCNBackbone as STGCNBackbone_pkg
    from src.models import STGCNBlock as STGCNBlock_pkg
    from src.models import SpatialGraphConv as SpatialGraphConv_pkg

    assert STGCN_pkg is STGCNBackbone_pkg
    assert STGCNBackbone_pkg is STGCNBackbone
    assert STGCNBlock_pkg is STGCNBlock
    assert SpatialGraphConv_pkg is SpatialGraphConv


def test_canonical_adjacency_properties():
    """Verifies that the canonical adjacency matrix is 115x115 and symmetric."""
    graph = get_canonical_graph()
    adj_norm = graph.get_normalized_adjacency(mode="symmetric")

    assert adj_norm.shape == (115, 115), f"Expected shape (115, 115), got {adj_norm.shape}"
    assert np.allclose(adj_norm, adj_norm.T, atol=1e-6), "Adjacency matrix is not symmetric"
    assert np.isfinite(adj_norm).all(), "Adjacency matrix contains NaN or Inf"


def test_stgcn_model_construction():
    """Verifies STGCNBackbone construction and configuration parameters."""
    model = STGCNBackbone()
    assert model.in_channels == 5
    assert model.num_nodes == 115
    assert model.sequence_length == 60
    assert model.channel_progression == [64, 128, 256, 256]
    assert len(model.layers) == 4
    assert isinstance(model, nn.Module)


def test_stgcn_from_config():
    """Verifies construction via STGCNBackbone.from_config()."""
    model = STGCNBackbone.from_config()
    assert model.in_channels == 5
    assert model.num_nodes == 115
    assert len(model.layers) == 4


def test_stgcn_forward_default_shape():
    """Verifies forward pass with standard input [B, 5, 60, 115] -> [B, 256, 60, 115]."""
    model = STGCNBackbone()
    model.eval()
    B, C, T, V = 2, 5, 60, 115
    x = torch.randn(B, C, T, V)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (B, 256, T, V), f"Expected output shape {(B, 256, T, V)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values"


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_stgcn_batch_sizes(batch_size: int):
    """Verifies that the model operates correctly across batch sizes 1, 2, and 8 in both train and eval modes."""
    model = STGCNBackbone()
    C, T, V = 5, 60, 115
    x = torch.randn(batch_size, C, T, V)

    # Test in eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (batch_size, 256, T, V)
    assert torch.isfinite(out_eval).all()

    # Test in train mode
    model.train()
    out_train = model(x)
    assert out_train.shape == (batch_size, 256, T, V)
    assert torch.isfinite(out_train).all()


def test_no_nan_or_inf_output():
    """Verifies that synthetic inputs produce strictly finite outputs with no NaNs or Infs."""
    model = STGCNBackbone()
    model.eval()
    x = torch.randn(4, 5, 60, 115)

    with torch.no_grad():
        out = model(x)

    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


def test_gradient_flow():
    """Verifies that gradients flow through all trainable parameters during backward pass."""
    model = STGCNBackbone()
    model.train()
    x = torch.randn(2, 5, 60, 115, requires_grad=True)

    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"

    # Check parameter gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found"

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for parameter '{name}' is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for parameter '{name}' contains NaN"
            assert not torch.isinf(param.grad).any(), f"Gradient for parameter '{name}' contains Inf"
            assert (param.grad != 0).any(), f"Gradient for parameter '{name}' is all zeros"


def test_deterministic_eval_output():
    """Verifies that forward pass in eval mode is strictly deterministic for identical inputs."""
    model = STGCNBackbone()
    model.eval()
    x = torch.randn(2, 5, 60, 115)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-7), "Eval mode outputs are not deterministic"


@pytest.mark.parametrize("seq_length", [10, 30, 60, 90, 120])
def test_temporal_dimension_preservation(seq_length: int):
    """Verifies that sequence length T is strictly preserved for arbitrary values of T."""
    model = STGCNBackbone()
    model.eval()
    x = torch.randn(2, 5, seq_length, 115)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 256, seq_length, 115), f"Expected sequence length {seq_length}, got {out.shape[2]}"


def test_adjacency_registered_as_buffer():
    """Verifies that normalized adjacency matrix is registered as a model buffer and not a parameter."""
    model = STGCNBackbone()
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())

    assert "adj_norm" in buffers, "'adj_norm' buffer is not registered in model buffers"
    assert "adj_norm" not in parameters, "'adj_norm' should be a buffer, not a parameter"
    assert buffers["adj_norm"].shape == (115, 115)
    assert isinstance(buffers["adj_norm"], torch.Tensor)


def test_no_lstm_or_gru_modules():
    """Verifies that the model contains no LSTM, GRU, or RNN modules."""
    model = STGCNBackbone()
    forbidden_types = (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    for name, module in model.named_modules():
        assert not isinstance(module, forbidden_types), f"Module '{name}' is forbidden type {type(module)}"


def test_invalid_input_shapes():
    """Verifies that passing invalid input shapes raises ValueError."""
    model = STGCNBackbone()

    # 3D input
    with pytest.raises(ValueError, match="Expected 4D tensor input"):
        model(torch.randn(2, 5, 60))

    # Wrong channel dimension (e.g., 3 instead of 5)
    with pytest.raises(ValueError, match="Expected input with 5 channels"):
        model(torch.randn(2, 3, 60, 115))

    # Wrong node dimension (e.g., 75 instead of 115)
    with pytest.raises(ValueError, match="Expected input with 115 nodes"):
        model(torch.randn(2, 5, 60, 75))

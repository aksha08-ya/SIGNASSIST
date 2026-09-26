"""Unit tests for Transformer Temporal Encoder (SIGNASSIST Phase 5.2).

Verifies:
- Model construction and configuration loading (from_config);
- Normal forward pass with ST-GCN backbone tensor [B, 256, T, 115] and [B, T, 256];
- Batch sizes 1, 2, 8;
- Output shape [B, 256];
- Variable temporal lengths 10, 30, 60, 90, 120;
- Padding-mask behavior;
- Padded timesteps do not alter valid sequence temporal representation;
- All-masked input edge-case handling without producing NaNs;
- Finite outputs and gradient flow through all parameters;
- Deterministic eval output;
- Zero LSTM / GRU / RNN recurrent components;
- Package exports via src.models.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models import (
    SinusoidalPositionalEncoding,
    TransformerTemporalEncoder,
)


def test_package_exports():
    """Verifies that TransformerTemporalEncoder exports cleanly from src.models."""
    from src.models import TransformerTemporalEncoder as Trans_pkg
    assert Trans_pkg is TransformerTemporalEncoder
    assert issubclass(TransformerTemporalEncoder, nn.Module)


def test_model_construction():
    """Verifies construction and default hyperparameters."""
    model = TransformerTemporalEncoder()
    assert model.hidden_dim == 256
    assert model.num_transformer_layers == 4
    assert model.num_heads == 8
    assert model.dropout_rate == 0.3
    assert model.sequence_length == 60
    assert isinstance(model, nn.Module)


def test_construction_from_config():
    """Verifies factory construction via from_config()."""
    model = TransformerTemporalEncoder.from_config()
    assert model.hidden_dim == 256
    assert model.num_transformer_layers == 4
    assert model.num_heads == 8


def test_forward_stgcn_backbone_input_shape():
    """Verifies forward pass with ST-GCN backbone tensor [B, 256, T, 115] -> [B, 256]."""
    model = TransformerTemporalEncoder()
    model.eval()
    B, C, T, V = 2, 256, 60, 115
    stgcn_output = torch.randn(B, C, T, V)

    with torch.no_grad():
        out = model(stgcn_output)

    assert out.shape == (B, 256), f"Expected shape {(B, 256)}, got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


def test_forward_sequence_input_shapes():
    """Verifies forward pass with 3D sequence inputs [B, T, 256] and [B, 256, T]."""
    model = TransformerTemporalEncoder()
    model.eval()

    # [B, T, 256]
    x_btc = torch.randn(2, 60, 256)
    with torch.no_grad():
        out1 = model(x_btc)
    assert out1.shape == (2, 256)

    # [B, 256, T]
    x_bct = torch.randn(2, 256, 60)
    with torch.no_grad():
        out2 = model(x_bct)
    assert out2.shape == (2, 256)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batch_sizes(batch_size: int):
    """Verifies batch sizes 1, 2, and 8 in both train and eval modes."""
    model = TransformerTemporalEncoder()
    x = torch.randn(batch_size, 256, 60, 115)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (batch_size, 256)
    assert torch.isfinite(out_eval).all()

    # Train mode
    model.train()
    out_train = model(x)
    assert out_train.shape == (batch_size, 256)
    assert torch.isfinite(out_train).all()


@pytest.mark.parametrize("seq_length", [10, 30, 60, 90, 120])
def test_variable_temporal_lengths(seq_length: int):
    """Verifies support for variable sequence lengths T without shape errors."""
    model = TransformerTemporalEncoder()
    model.eval()
    x = torch.randn(2, 256, seq_length, 115)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 256)
    assert torch.isfinite(out).all()


def test_padding_mask_invariance():
    """Verifies that padded timesteps do not alter the temporal representation of valid frames."""
    model = TransformerTemporalEncoder(dropout=0.0)  # Zero dropout for exact determinism
    model.eval()

    B = 2
    T_valid = 30
    T_total = 60
    C = 256

    # Generate 30 frames of valid features
    valid_features = torch.randn(B, T_valid, C)

    # Create total 60 frames tensor by appending zero padded frames
    padded_features = torch.zeros(B, T_total, C)
    padded_features[:, :T_valid, :] = valid_features

    # Random padding frames to test that padded values are ignored when masked
    random_padded_features = torch.randn(B, T_total, C)
    random_padded_features[:, :T_valid, :] = valid_features

    # Mask: True for valid frames (0..29), False for padded frames (30..59)
    mask = torch.zeros(B, T_total, dtype=torch.bool)
    mask[:, :T_valid] = True

    with torch.no_grad():
        out_valid = model(valid_features)
        out_zero_padded = model(padded_features, mask=mask)
        out_rand_padded = model(random_padded_features, mask=mask)

    # Output representations must be identical regardless of padded frame values
    assert torch.allclose(out_valid, out_zero_padded, atol=1e-5), "Zero-padded sequence output differs from unpadded output"
    assert torch.allclose(out_valid, out_rand_padded, atol=1e-5), "Random-padded sequence output differs from unpadded output"


def test_node_mask_support():
    """Verifies that 3D mask [B, T, V] is properly handled."""
    model = TransformerTemporalEncoder()
    model.eval()

    B, T, V = 2, 60, 115
    x = torch.randn(B, 256, T, V)

    # Mask where first 30 frames are True (valid) and rest False (padded)
    mask_3d = torch.zeros(B, T, V, dtype=torch.bool)
    mask_3d[:, :30, :] = True

    with torch.no_grad():
        out = model(x, mask=mask_3d)

    assert out.shape == (B, 256)
    assert torch.isfinite(out).all()


def test_all_masked_edge_case():
    """Verifies that all-masked sequence does not produce NaN or Inf values."""
    model = TransformerTemporalEncoder()
    model.eval()

    B, T, C = 2, 60, 256
    x = torch.randn(B, T, C)

    # All positions masked out
    mask = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        out = model(x, mask=mask)

    assert out.shape == (B, 256)
    assert not torch.isnan(out).any(), "All-masked input produced NaN"
    assert not torch.isinf(out).any(), "All-masked input produced Inf"


def test_finite_outputs_and_gradient_flow():
    """Verifies backward pass and gradient flow through all trainable parameters."""
    model = TransformerTemporalEncoder()
    model.train()

    x = torch.randn(2, 256, 60, 115, requires_grad=True)
    out = model(x)

    loss = out.sum()
    loss.backward()

    # Check input gradient
    assert x.grad is not None, "Input gradient is None"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"

    # Check parameter gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for '{name}' is None"
            assert not torch.isnan(param.grad).any(), f"Gradient for '{name}' contains NaN"
            assert not torch.isinf(param.grad).any(), f"Gradient for '{name}' contains Inf"


def test_deterministic_eval_output():
    """Verifies deterministic output in eval mode."""
    model = TransformerTemporalEncoder()
    model.eval()

    x = torch.randn(2, 256, 60, 115)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-7), "Outputs in eval mode are not deterministic"


def test_no_lstm_gru_rnn_modules():
    """Verifies that the encoder contains zero recurrent (LSTM/GRU/RNN) modules."""
    model = TransformerTemporalEncoder()
    forbidden_types = (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    for name, module in model.named_modules():
        assert not isinstance(module, forbidden_types), f"Module '{name}' is forbidden recurrent type {type(module)}"


def test_invalid_input_shapes():
    """Verifies ValueError on invalid input shapes or channel dimensions."""
    model = TransformerTemporalEncoder()

    # 2D input
    with pytest.raises(ValueError, match="Expected 3D or 4D input tensor"):
        model(torch.randn(2, 256))

    # Invalid channels (e.g. 128 instead of 256)
    with pytest.raises(ValueError, match="Expected 3D tensor"):
        model(torch.randn(2, 128, 60))

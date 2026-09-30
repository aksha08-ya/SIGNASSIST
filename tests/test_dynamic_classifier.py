"""Unit tests for ST-GCN + Transformer Dynamic Sign Classifier (SIGNASSIST Phase 5.3).

Verifies:
- Model construction and configuration loading (from_config);
- Package exports via src.models;
- Single internal ST-GCN backbone and Transformer encoder instances without duplicates;
- Input shape [B, 5, T, 115] -> Output logits [B, 262] & embedding [B, 256];
- Batch sizes 1, 2, and 8;
- Variable temporal lengths T = 10, 30, 60, 90, 120;
- [B, T] and [B, T, 115] validity masks;
- Padded-frame invariance;
- All-masked edge-case handling without producing NaNs/Infs;
- Finite outputs;
- Independent gradient flow through ST-GCN, Transformer, and Classifier head;
- Deterministic eval behavior;
- Zero recurrent (LSTM/GRU/RNN) modules;
- Invalid input shape handling.
"""

import pytest
import torch
import torch.nn as nn

from src.models import (
    DynamicClassifierOutput,
    DynamicSTGCNTransformer,
    DynamicSignClassifier,
    STGCNBackbone,
    TransformerTemporalEncoder,
)


def test_package_exports():
    """Verifies package exports and class aliases in src.models."""
    from src.models import DynamicSignClassifier as Dynamic_pkg
    from src.models import DynamicSTGCNTransformer as Alias_pkg

    assert Dynamic_pkg is DynamicSignClassifier
    assert Alias_pkg is DynamicSignClassifier
    assert issubclass(DynamicSignClassifier, nn.Module)


def test_model_construction():
    """Verifies construction, component instances, and default hyperparameters."""
    model = DynamicSignClassifier()
    assert model.in_channels == 5
    assert model.num_nodes == 115
    assert model.sequence_length == 60
    assert model.hidden_dim == 256
    assert model.num_classes == 262
    assert isinstance(model.stgcn_backbone, STGCNBackbone)
    assert isinstance(model.transformer_encoder, TransformerTemporalEncoder)


def test_construction_from_config():
    """Verifies construction via DynamicSignClassifier.from_config()."""
    model = DynamicSignClassifier.from_config()
    assert model.in_channels == 5
    assert model.num_nodes == 115
    assert model.sequence_length == 60
    assert model.hidden_dim == 256
    assert model.num_classes == 262


def test_no_duplicate_instances():
    """Verifies that custom backbone/encoder injection reuses existing instances without duplication."""
    backbone = STGCNBackbone()
    encoder = TransformerTemporalEncoder()

    model = DynamicSignClassifier(
        stgcn_backbone=backbone,
        transformer_encoder=encoder,
    )

    assert model.stgcn_backbone is backbone
    assert model.transformer_encoder is encoder

    # Verify that model parameter search finds exact module references
    stgcn_modules = [m for m in model.modules() if isinstance(m, STGCNBackbone)]
    transformer_modules = [m for m in model.modules() if isinstance(m, TransformerTemporalEncoder)]
    assert len(stgcn_modules) == 1, "Duplicate ST-GCN backbone instances found"
    assert len(transformer_modules) == 1, "Duplicate Transformer encoder instances found"


def test_forward_pass_default_shapes():
    """Verifies forward pass output shapes: logits [B, 262], embedding [B, 256], ST-GCN [B, 256, 60, 115]."""
    model = DynamicSignClassifier()
    model.eval()

    B, C, T, V = 2, 5, 60, 115
    x = torch.randn(B, C, T, V)

    with torch.no_grad():
        out = model(x)

    assert isinstance(out, DynamicClassifierOutput)
    assert out.logits.shape == (B, 262), f"Expected logits shape {(B, 262)}, got {out.logits.shape}"
    assert out.embedding.shape == (B, 256), f"Expected embedding shape {(B, 256)}, got {out.embedding.shape}"
    assert out.stgcn_features.shape == (B, 256, T, V), f"Expected ST-GCN features shape {(B, 256, T, V)}, got {out.stgcn_features.shape}"
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.embedding).all()


def test_forward_features_helper():
    """Verifies forward_features() returns 256-dim temporal embedding [B, 256]."""
    model = DynamicSignClassifier()
    model.eval()

    x = torch.randn(2, 5, 60, 115)
    with torch.no_grad():
        embedding = model.forward_features(x)

    assert embedding.shape == (2, 256)
    assert torch.isfinite(embedding).all()


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batch_sizes(batch_size: int):
    """Verifies forward pass across batch sizes 1, 2, and 8 in train and eval modes."""
    model = DynamicSignClassifier()
    C, T, V = 5, 60, 115
    x = torch.randn(batch_size, C, T, V)

    # Eval mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.logits.shape == (batch_size, 262)
    assert out_eval.embedding.shape == (batch_size, 256)
    assert torch.isfinite(out_eval.logits).all()

    # Train mode
    model.train()
    out_train = model(x)
    assert out_train.logits.shape == (batch_size, 262)
    assert out_train.embedding.shape == (batch_size, 256)
    assert torch.isfinite(out_train.logits).all()


@pytest.mark.parametrize("seq_length", [10, 30, 60, 90, 120])
def test_variable_temporal_lengths(seq_length: int):
    """Verifies that sequence length T is dynamically supported for arbitrary values of T."""
    model = DynamicSignClassifier()
    model.eval()

    x = torch.randn(2, 5, seq_length, 115)
    with torch.no_grad():
        out = model(x)

    assert out.logits.shape == (2, 262)
    assert out.embedding.shape == (2, 256)
    assert out.stgcn_features.shape == (2, 256, seq_length, 115)


def test_temporal_2d_mask():
    """Verifies forward pass with [B, T] validity mask."""
    model = DynamicSignClassifier()
    model.eval()

    B, T = 2, 60
    x = torch.randn(B, 5, T, 115)
    mask_2d = torch.zeros(B, T, dtype=torch.bool)
    mask_2d[:, :30] = True

    with torch.no_grad():
        out = model(x, mask=mask_2d)

    assert out.logits.shape == (B, 262)
    assert out.embedding.shape == (B, 256)
    assert torch.isfinite(out.logits).all()


def test_node_3d_mask():
    """Verifies forward pass with [B, T, 115] validity mask."""
    model = DynamicSignClassifier()
    model.eval()

    B, T, V = 2, 60, 115
    x = torch.randn(B, 5, T, V)
    mask_3d = torch.zeros(B, T, V, dtype=torch.bool)
    mask_3d[:, :30, :] = True

    with torch.no_grad():
        out = model(x, mask=mask_3d)

    assert out.logits.shape == (B, 262)
    assert out.embedding.shape == (B, 256)
    assert torch.isfinite(out.logits).all()


def test_padded_frame_invariance():
    """Verifies that padded frame values under mask produce consistent embeddings for zero-padded sequence buffers."""
    model = DynamicSignClassifier(dropout=0.0)
    model.eval()

    B = 2
    T_valid = 30
    T_total = 60
    C = 5
    V = 115

    # 30 valid frames
    valid_frames = torch.randn(B, C, T_valid, V)

    # 60 frames tensor with zero padding as produced by SequenceBuffer
    zero_padded = torch.zeros(B, C, T_total, V)
    zero_padded[:, :, :T_valid, :] = valid_frames

    # Mask: True for frames 0..29, False for 30..59
    mask = torch.zeros(B, T_total, dtype=torch.bool)
    mask[:, :T_valid] = True

    with torch.no_grad():
        out_zero = model(zero_padded, mask=mask)
        out_zero_again = model(zero_padded, mask=mask)

    assert torch.allclose(out_zero.embedding, out_zero_again.embedding, atol=1e-6)
    assert torch.allclose(out_zero.logits, out_zero_again.logits, atol=1e-6)
    assert torch.isfinite(out_zero.logits).all()
    assert torch.isfinite(out_zero.embedding).all()


def test_all_masked_edge_case():
    """Verifies that an all-masked input does not produce NaN or Inf values."""
    model = DynamicSignClassifier()
    model.eval()

    B, T = 2, 60
    x = torch.randn(B, 5, T, 115)
    mask_all_false = torch.zeros(B, T, dtype=torch.bool)

    with torch.no_grad():
        out = model(x, mask=mask_all_false)

    assert out.logits.shape == (B, 262)
    assert out.embedding.shape == (B, 256)
    assert not torch.isnan(out.logits).any()
    assert not torch.isinf(out.logits).any()
    assert not torch.isnan(out.embedding).any()
    assert not torch.isinf(out.embedding).any()


def test_gradient_flow_all_submodules():
    """Verifies that gradients flow back into ST-GCN, Transformer, and Classifier head parameters."""
    model = DynamicSignClassifier()
    model.train()

    x = torch.randn(2, 5, 60, 115, requires_grad=True)
    out = model(x)

    loss = out.logits.sum()
    loss.backward()

    # 1. Check input gradients
    assert x.grad is not None, "Input x gradient is None"
    assert not torch.isnan(x.grad).any(), "Input x gradient contains NaN"

    # 2. Check ST-GCN backbone gradients
    stgcn_params = list(model.stgcn_backbone.parameters())
    assert len(stgcn_params) > 0
    for param in stgcn_params:
        if param.requires_grad:
            assert param.grad is not None, "ST-GCN parameter gradient is None"
            assert not torch.isnan(param.grad).any(), "ST-GCN parameter gradient contains NaN"

    # 3. Check Transformer encoder gradients
    trans_params = list(model.transformer_encoder.parameters())
    assert len(trans_params) > 0
    for param in trans_params:
        if param.requires_grad:
            assert param.grad is not None, "Transformer parameter gradient is None"
            assert not torch.isnan(param.grad).any(), "Transformer parameter gradient contains NaN"

    # 4. Check Classifier head gradients
    head_params = list(model.classifier.parameters())
    assert len(head_params) > 0
    for param in head_params:
        if param.requires_grad:
            assert param.grad is not None, "Classifier head parameter gradient is None"
            assert not torch.isnan(param.grad).any(), "Classifier head parameter gradient contains NaN"


def test_deterministic_eval_output():
    """Verifies deterministic output in eval mode for identical inputs."""
    model = DynamicSignClassifier()
    model.eval()

    x = torch.randn(2, 5, 60, 115)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1.logits, out2.logits, atol=1e-7)
    assert torch.allclose(out1.embedding, out2.embedding, atol=1e-7)


def test_no_lstm_gru_rnn_modules():
    """Verifies that zero recurrent (LSTM/GRU/RNN) modules exist in the integrated model."""
    model = DynamicSignClassifier()
    forbidden_types = (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    for name, module in model.named_modules():
        assert not isinstance(module, forbidden_types), f"Module '{name}' is forbidden recurrent type {type(module)}"


def test_invalid_input_shapes():
    """Verifies ValueError when invalid input tensor shape, channel count, or node count is passed."""
    model = DynamicSignClassifier()

    # 3D input
    with pytest.raises(ValueError, match="Expected 4D tensor input"):
        model(torch.randn(2, 5, 60))

    # Invalid channel count (3 instead of 5)
    with pytest.raises(ValueError, match="Expected input with 5 channels"):
        model(torch.randn(2, 3, 60, 115))

    # Invalid node count (75 instead of 115)
    with pytest.raises(ValueError, match="Expected input with 115 nodes"):
        model(torch.randn(2, 5, 60, 75))

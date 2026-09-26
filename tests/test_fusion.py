"""Unit tests for Multimodal Feature Fusion Engine (SIGNASSIST Phase 7).

Verifies:
- Fusion module construction and configuration loading (from_config);
- Package exports via src.models;
- Input shapes static [B, 256] + dynamic [B, 256];
- Output fused_embedding [B, 256];
- Batch sizes 1, 2, 8 in train and eval modes;
- Finite outputs;
- Gradient flow through static embedding, dynamic embedding, router parameters, and refinement parameters;
- Routing weight mathematical correctness:
  - static_weight=1.0, dynamic_weight=0.0 -> fused_raw == static_embedding
  - static_weight=0.0, dynamic_weight=1.0 -> fused_raw == dynamic_embedding
  - static_weight=0.5, dynamic_weight=0.5 -> fused_raw == 0.5*static + 0.5*dynamic
- Changing routing weights dynamically changes fused representation;
- Non-silent routing weight usage;
- Refinement projection output shape, finite values, and gradient flow;
- Invalid input dimensions and batch size mismatch error handling;
- Deterministic eval mode;
- Zero recurrent (LSTM/GRU/RNN) modules.
"""

import pytest
import torch
import torch.nn as nn

from src.models import (
    MultimodalFusionEngine,
    MultimodalFusionModule,
    MultimodalFusionOutput,
)


def test_package_exports():
    """Verifies package exports and class alias in src.models."""
    from src.models import MultimodalFusionEngine as Engine_pkg
    from src.models import MultimodalFusionModule as Module_pkg

    assert Engine_pkg is MultimodalFusionEngine
    assert Module_pkg is MultimodalFusionEngine
    assert issubclass(MultimodalFusionEngine, nn.Module)


def test_fusion_construction():
    """Verifies construction and default hyperparameters."""
    engine = MultimodalFusionEngine()
    assert engine.input_dim == 256
    assert engine.output_dim == 256
    assert engine.dropout_rate == 0.2
    assert engine.use_refinement is True
    assert isinstance(engine, nn.Module)


def test_construction_from_config():
    """Verifies factory construction via MultimodalFusionEngine.from_config()."""
    engine = MultimodalFusionEngine.from_config()
    assert engine.input_dim == 256
    assert engine.output_dim == 256
    assert engine.use_refinement is True


def test_forward_pass_default_shapes():
    """Verifies forward pass with default inputs produces fused_embedding [B, 256]."""
    engine = MultimodalFusionEngine()
    engine.eval()

    B = 4
    static_emb = torch.randn(B, 256)
    dynamic_emb = torch.randn(B, 256)

    with torch.no_grad():
        out = engine(static_emb, dynamic_emb)

    assert isinstance(out, MultimodalFusionOutput)
    assert out.fused_embedding.shape == (B, 256), f"Expected fused_embedding shape {(B, 256)}, got {out.fused_embedding.shape}"
    assert out.static_embedding.shape == (B, 256)
    assert out.dynamic_embedding.shape == (B, 256)
    assert out.static_weight.shape == (B, 1)
    assert out.dynamic_weight.shape == (B, 1)
    assert out.probabilities.shape == (B, 2)
    assert out.fused_raw.shape == (B, 256)
    assert torch.isfinite(out.fused_embedding).all()


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batch_sizes(batch_size: int):
    """Verifies forward execution across batch sizes 1, 2, and 8 in train and eval modes."""
    engine = MultimodalFusionEngine()
    static_emb = torch.randn(batch_size, 256)
    dynamic_emb = torch.randn(batch_size, 256)

    # Eval mode
    engine.eval()
    with torch.no_grad():
        out_eval = engine(static_emb, dynamic_emb)
    assert out_eval.fused_embedding.shape == (batch_size, 256)
    assert torch.isfinite(out_eval.fused_embedding).all()

    # Train mode
    engine.train()
    out_train = engine(static_emb, dynamic_emb)
    assert out_train.fused_embedding.shape == (batch_size, 256)
    assert torch.isfinite(out_train.fused_embedding).all()


def test_weighted_sum_math_and_weight_influence():
    """Explicitly tests mathematical correctness of weighted sum without refinement."""
    engine = MultimodalFusionEngine(use_refinement=False)
    engine.eval()

    B = 2
    static_emb = torch.randn(B, 256)
    dynamic_emb = torch.randn(B, 256)

    # 1. Test static_weight = 1.0, dynamic_weight = 0.0 -> fused == static_emb
    s_weight_1 = torch.tensor([[1.0], [1.0]])
    d_weight_0 = torch.tensor([[0.0], [0.0]])
    with torch.no_grad():
        out_static_only = engine(
            static_emb, dynamic_emb, static_weight=s_weight_1, dynamic_weight=d_weight_0
        )
    assert torch.allclose(out_static_only.fused_embedding, static_emb, atol=1e-6)
    assert torch.allclose(out_static_only.fused_raw, static_emb, atol=1e-6)

    # 2. Test static_weight = 0.0, dynamic_weight = 1.0 -> fused == dynamic_emb
    s_weight_0 = torch.tensor([[0.0], [0.0]])
    d_weight_1 = torch.tensor([[1.0], [1.0]])
    with torch.no_grad():
        out_dynamic_only = engine(
            static_emb, dynamic_emb, static_weight=s_weight_0, dynamic_weight=d_weight_1
        )
    assert torch.allclose(out_dynamic_only.fused_embedding, dynamic_emb, atol=1e-6)
    assert torch.allclose(out_dynamic_only.fused_raw, dynamic_emb, atol=1e-6)

    # 3. Test 50/50 split -> fused == 0.5 * static + 0.5 * dynamic
    s_weight_half = torch.tensor([[0.5], [0.5]])
    d_weight_half = torch.tensor([[0.5], [0.5]])
    with torch.no_grad():
        out_half = engine(
            static_emb, dynamic_emb, static_weight=s_weight_half, dynamic_weight=d_weight_half
        )
    expected_half = 0.5 * static_emb + 0.5 * dynamic_emb
    assert torch.allclose(out_half.fused_embedding, expected_half, atol=1e-6)


def test_changing_weights_changes_fused_representation():
    """Verifies that changing routing weights dynamically alters the fused representation."""
    engine = MultimodalFusionEngine(use_refinement=False)
    engine.eval()

    B = 2
    static_emb = torch.randn(B, 256)
    dynamic_emb = torch.randn(B, 256)

    weights_a_stat = torch.tensor([[0.8], [0.8]])
    weights_a_dyn = torch.tensor([[0.2], [0.2]])

    weights_b_stat = torch.tensor([[0.2], [0.2]])
    weights_b_dyn = torch.tensor([[0.8], [0.8]])

    with torch.no_grad():
        out_a = engine(static_emb, dynamic_emb, static_weight=weights_a_stat, dynamic_weight=weights_a_dyn)
        out_b = engine(static_emb, dynamic_emb, static_weight=weights_b_stat, dynamic_weight=weights_b_dyn)

    assert not torch.allclose(out_a.fused_embedding, out_b.fused_embedding, atol=1e-4), "Changing routing weights failed to alter fused representation"


def test_gradient_flow():
    """Verifies gradient flow through static embedding, dynamic embedding, router, and refinement block."""
    engine = MultimodalFusionEngine(use_refinement=True)
    engine.train()

    static_emb = torch.randn(2, 256, requires_grad=True)
    dynamic_emb = torch.randn(2, 256, requires_grad=True)

    out = engine(static_emb, dynamic_emb)
    loss = out.fused_embedding.sum()
    loss.backward()

    # 1. Check input embedding gradients
    assert static_emb.grad is not None, "Static embedding gradient is None"
    assert not torch.isnan(static_emb.grad).any(), "Static embedding gradient contains NaN"
    assert dynamic_emb.grad is not None, "Dynamic embedding gradient is None"
    assert not torch.isnan(dynamic_emb.grad).any(), "Dynamic embedding gradient contains NaN"

    # 2. Check router parameter gradients
    router_params = list(engine.router.parameters())
    assert len(router_params) > 0
    for param in router_params:
        if param.requires_grad:
            assert param.grad is not None, "Router parameter gradient is None"
            assert not torch.isnan(param.grad).any(), "Router parameter gradient contains NaN"

    # 3. Check refinement block parameter gradients
    refinement_params = list(engine.refinement.parameters())
    assert len(refinement_params) > 0
    for param in refinement_params:
        if param.requires_grad:
            assert param.grad is not None, "Refinement parameter gradient is None"
            assert not torch.isnan(param.grad).any(), "Refinement parameter gradient contains NaN"


def test_refinement_projection_behavior():
    """Verifies that refinement projection maintains [B, 256] output shape and finite values."""
    engine_with_refinement = MultimodalFusionEngine(use_refinement=True)
    engine_without_refinement = MultimodalFusionEngine(use_refinement=False)

    engine_with_refinement.eval()
    engine_without_refinement.eval()

    static_emb = torch.randn(2, 256)
    dynamic_emb = torch.randn(2, 256)

    with torch.no_grad():
        out_refined = engine_with_refinement(static_emb, dynamic_emb)
        out_unrefined = engine_without_refinement(static_emb, dynamic_emb)

    assert out_refined.fused_embedding.shape == (2, 256)
    assert out_unrefined.fused_embedding.shape == (2, 256)
    assert torch.isfinite(out_refined.fused_embedding).all()
    assert torch.isfinite(out_unrefined.fused_embedding).all()


def test_deterministic_eval_mode():
    """Verifies deterministic forward pass in eval mode."""
    engine = MultimodalFusionEngine()
    engine.eval()

    static_emb = torch.randn(2, 256)
    dynamic_emb = torch.randn(2, 256)

    with torch.no_grad():
        out1 = engine(static_emb, dynamic_emb)
        out2 = engine(static_emb, dynamic_emb)

    assert torch.allclose(out1.fused_embedding, out2.fused_embedding, atol=1e-7)
    assert torch.allclose(out1.static_weight, out2.static_weight, atol=1e-7)
    assert torch.allclose(out1.dynamic_weight, out2.dynamic_weight, atol=1e-7)


def test_no_lstm_gru_rnn_modules():
    """Verifies zero recurrent (LSTM/GRU/RNN) modules in the fusion engine."""
    engine = MultimodalFusionEngine()
    forbidden_types = (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    for name, module in engine.named_modules():
        assert not isinstance(module, forbidden_types), f"Module '{name}' is forbidden recurrent type {type(module)}"


def test_invalid_input_shapes():
    """Verifies ValueError handling for invalid shapes and batch dimension mismatches."""
    engine = MultimodalFusionEngine()

    # 1D static input
    with pytest.raises(ValueError, match="Expected 2D static_embedding tensor"):
        engine(torch.randn(256), torch.randn(1, 256))

    # 1D dynamic input
    with pytest.raises(ValueError, match="Expected 2D dynamic_embedding tensor"):
        engine(torch.randn(1, 256), torch.randn(256))

    # Batch size mismatch (B=2 vs B=4)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        engine(torch.randn(2, 256), torch.randn(4, 256))

    # Static channel dimension mismatch (128 instead of 256)
    with pytest.raises(ValueError, match="Expected static_embedding channel dimension 256"):
        engine(torch.randn(2, 128), torch.randn(2, 256))

    # Dynamic channel dimension mismatch (128 instead of 256)
    with pytest.raises(ValueError, match="Expected dynamic_embedding channel dimension 256"):
        engine(torch.randn(2, 256), torch.randn(2, 128))

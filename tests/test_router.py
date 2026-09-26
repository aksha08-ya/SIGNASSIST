"""Unit tests for Learned Static/Dynamic Router (SIGNASSIST Phase 6).

Verifies:
- Router construction and configuration loading (from_config);
- Package exports via src.models;
- Input shape [B, 256] static + [B, 256] dynamic;
- Output probabilities shape [B, 2], static weight shape [B, 1], dynamic weight shape [B, 1];
- Weights sum to 1.0 and remain strictly within [0.0, 1.0];
- Batch sizes 1, 2, 8 in train and eval modes;
- Finite outputs;
- Gradient flow through router parameters and back into static & dynamic embeddings;
- Deterministic eval output;
- Zero recurrent (LSTM/GRU/RNN) modules;
- Invalid input shape and batch mismatch error handling;
- Learned function test: output routing weights dynamically change as a function of feature embeddings.
"""

import pytest
import torch
import torch.nn as nn

from src.models import (
    LearnedRouter,
    LearnedStaticDynamicRouter,
    RouterOutput,
)


def test_package_exports():
    """Verifies package exports and class alias in src.models."""
    from src.models import LearnedRouter as Alias_pkg
    from src.models import LearnedStaticDynamicRouter as Router_pkg

    assert Router_pkg is LearnedStaticDynamicRouter
    assert Alias_pkg is LearnedStaticDynamicRouter
    assert issubclass(LearnedStaticDynamicRouter, nn.Module)


def test_router_construction():
    """Verifies construction and default hyperparameters."""
    router = LearnedStaticDynamicRouter()
    assert router.static_dim == 256
    assert router.dynamic_dim == 256
    assert router.hidden_dim == 128
    assert router.dropout_rate == 0.2
    assert isinstance(router, nn.Module)


def test_construction_from_config():
    """Verifies factory construction via LearnedStaticDynamicRouter.from_config()."""
    router = LearnedStaticDynamicRouter.from_config()
    assert router.static_dim == 256
    assert router.dynamic_dim == 256
    assert router.hidden_dim == 128


def test_forward_pass_shapes():
    """Verifies output shapes: probabilities [B, 2], static_weight [B, 1], dynamic_weight [B, 1]."""
    router = LearnedStaticDynamicRouter()
    router.eval()

    B = 4
    static_emb = torch.randn(B, 256)
    dynamic_emb = torch.randn(B, 256)

    with torch.no_grad():
        out = router(static_emb, dynamic_emb)

    assert isinstance(out, RouterOutput)
    assert out.probabilities.shape == (B, 2), f"Expected probabilities shape {(B, 2)}, got {out.probabilities.shape}"
    assert out.static_weight.shape == (B, 1), f"Expected static_weight shape {(B, 1)}, got {out.static_weight.shape}"
    assert out.dynamic_weight.shape == (B, 1), f"Expected dynamic_weight shape {(B, 1)}, got {out.dynamic_weight.shape}"
    assert out.logits.shape == (B, 2), f"Expected logits shape {(B, 2)}, got {out.logits.shape}"


def test_routing_weights_properties():
    """Verifies that static_weight + dynamic_weight == 1.0 and all weights remain in [0.0, 1.0]."""
    router = LearnedStaticDynamicRouter()
    router.eval()

    B = 8
    static_emb = torch.randn(B, 256)
    dynamic_emb = torch.randn(B, 256)

    with torch.no_grad():
        out = router(static_emb, dynamic_emb)

    # Weights within [0.0, 1.0]
    assert (out.static_weight >= 0.0).all() and (out.static_weight <= 1.0).all()
    assert (out.dynamic_weight >= 0.0).all() and (out.dynamic_weight <= 1.0).all()
    assert (out.probabilities >= 0.0).all() and (out.probabilities <= 1.0).all()

    # Sums to 1.0
    weight_sum = out.static_weight + out.dynamic_weight
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-6)

    prob_sum = out.probabilities.sum(dim=-1, keepdim=True)
    assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_batch_sizes(batch_size: int):
    """Verifies router forward execution across batch sizes 1, 2, and 8 in train and eval modes."""
    router = LearnedStaticDynamicRouter()
    static_emb = torch.randn(batch_size, 256)
    dynamic_emb = torch.randn(batch_size, 256)

    # Eval mode
    router.eval()
    with torch.no_grad():
        out_eval = router(static_emb, dynamic_emb)
    assert out_eval.probabilities.shape == (batch_size, 2)
    assert torch.isfinite(out_eval.probabilities).all()

    # Train mode
    router.train()
    out_train = router(static_emb, dynamic_emb)
    assert out_train.probabilities.shape == (batch_size, 2)
    assert torch.isfinite(out_train.probabilities).all()


def test_gradient_flow_and_embedding_backprop():
    """Verifies gradient flow through router parameters and back into input embeddings."""
    router = LearnedStaticDynamicRouter()
    router.train()

    static_emb = torch.randn(2, 256, requires_grad=True)
    dynamic_emb = torch.randn(2, 256, requires_grad=True)

    out = router(static_emb, dynamic_emb)
    loss = out.probabilities.sum()
    loss.backward()

    # Check backpropagation into static embedding
    assert static_emb.grad is not None, "Static embedding gradient is None"
    assert not torch.isnan(static_emb.grad).any(), "Static embedding gradient contains NaN"

    # Check backpropagation into dynamic embedding
    assert dynamic_emb.grad is not None, "Dynamic embedding gradient is None"
    assert not torch.isnan(dynamic_emb.grad).any(), "Dynamic embedding gradient contains NaN"

    # Check router layer parameters
    for name, param in router.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Router parameter '{name}' gradient is None"
            assert not torch.isnan(param.grad).any(), f"Router parameter '{name}' gradient contains NaN"


def test_deterministic_eval_behavior():
    """Verifies deterministic forward pass in eval mode."""
    router = LearnedStaticDynamicRouter()
    router.eval()

    static_emb = torch.randn(2, 256)
    dynamic_emb = torch.randn(2, 256)

    with torch.no_grad():
        out1 = router(static_emb, dynamic_emb)
        out2 = router(static_emb, dynamic_emb)

    assert torch.allclose(out1.probabilities, out2.probabilities, atol=1e-7)
    assert torch.allclose(out1.static_weight, out2.static_weight, atol=1e-7)
    assert torch.allclose(out1.dynamic_weight, out2.dynamic_weight, atol=1e-7)


def test_learned_function_response():
    """Verifies that changing input feature embeddings alters routing probabilities, confirming a learned function."""
    router = LearnedStaticDynamicRouter()
    router.eval()

    emb_a = torch.randn(2, 256)
    emb_b = torch.randn(2, 256)
    emb_c = torch.randn(2, 256) * 5.0 + 3.0

    with torch.no_grad():
        out_ab = router(emb_a, emb_b)
        out_ac = router(emb_a, emb_c)

    # Differing inputs produce different routing outputs
    assert not torch.allclose(out_ab.probabilities, out_ac.probabilities, atol=1e-4), "Router output did not respond to embedding changes"


def test_no_lstm_gru_rnn_modules():
    """Verifies that zero recurrent (LSTM/GRU/RNN) modules exist in the router."""
    router = LearnedStaticDynamicRouter()
    forbidden_types = (nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell)

    for name, module in router.named_modules():
        assert not isinstance(module, forbidden_types), f"Module '{name}' is forbidden recurrent type {type(module)}"


def test_invalid_input_shapes():
    """Verifies ValueError handling for invalid shapes and batch dimension mismatches."""
    router = LearnedStaticDynamicRouter()

    # 1D static input
    with pytest.raises(ValueError, match="Expected 2D static_embedding"):
        router(torch.randn(256), torch.randn(1, 256))

    # 1D dynamic input
    with pytest.raises(ValueError, match="Expected 2D dynamic_embedding"):
        router(torch.randn(1, 256), torch.randn(256))

    # Batch size mismatch (B=2 vs B=3)
    with pytest.raises(ValueError, match="Batch size mismatch"):
        router(torch.randn(2, 256), torch.randn(3, 256))

    # Static channel dimension mismatch (128 instead of 256)
    with pytest.raises(ValueError, match="Expected static embedding dimension 256"):
        router(torch.randn(2, 128), torch.randn(2, 256))

    # Dynamic channel dimension mismatch (128 instead of 256)
    with pytest.raises(ValueError, match="Expected dynamic embedding dimension 256"):
        router(torch.randn(2, 256), torch.randn(2, 128))

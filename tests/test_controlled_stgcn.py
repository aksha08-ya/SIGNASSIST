"""Tests for Controlled ST-GCN model and checkpoint handling."""

from __future__ import annotations

from pathlib import Path
import torch
import pytest

from src.training.train_controlled_stgcn import ControlledSTGCNClassifier


def test_controlled_stgcn_initialization() -> None:
    model = ControlledSTGCNClassifier(num_classes=42)
    assert model.num_classes == 42
    assert model.classifier[-1].out_features == 42


def test_controlled_stgcn_forward_pass() -> None:
    model = ControlledSTGCNClassifier(num_classes=42)
    x = torch.randn(2, 5, 60, 115)
    logits = model(x)
    assert logits.shape == (2, 42)


def test_controlled_stgcn_checkpoint_saving_loading(tmp_path: Path) -> None:
    model = ControlledSTGCNClassifier(num_classes=42)
    ckpt_path = tmp_path / "test_checkpoint.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 42,
        },
        ckpt_path,
    )

    loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
    new_model = ControlledSTGCNClassifier(num_classes=42)
    new_model.load_state_dict(loaded_ckpt["model_state_dict"])

    model.eval()
    new_model.eval()

    x = torch.randn(1, 5, 60, 115)
    with torch.no_grad():
        out1 = model(x)
        out2 = new_model(x)
    assert torch.allclose(out1, out2, atol=1e-5)

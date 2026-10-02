"""Phase 5: Controlled Vocabulary Model Improvement Experiments Framework for SIGNASSIST."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.include_dataset import INCLUDEFeatureDataset
from src.models.stgcn import STGCNBackbone
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier, seed_everything, unpack_batch

# -----------------------------------------------------------------
# 1. LANDMARK DATA AUGMENTATION DATASET WRAPPER
# -----------------------------------------------------------------
class AugmentedFeatureDataset(Dataset):
    """Wraps an INCLUDEFeatureDataset with realistic landmark-space data augmentations."""

    def __init__(self, dataset: INCLUDEFeatureDataset, augment: bool = True) -> None:
        self.dataset = dataset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = self.dataset[idx]
        if isinstance(batch, dict):
            x = batch["dynamic"].clone()
            y = batch["label"]
        else:
            x = batch[0].clone()
            y = batch[1]
        
        if not self.augment:
            return {"dynamic": x, "label": y}

        # x shape: [5, T, 115]
        # Channels: 0=x, 1=y, 2=z, 3=vx, 4=vy
        # 1. Gaussian noise on spatial channels (0, 1, 2)
        noise = torch.randn_like(x[:3, :, :]) * 0.003
        mask = (x[:3, :, :].abs() > 1e-6)
        x[:3, :, :] += noise * mask

        # 2. Random spatial scale
        scale = random.uniform(0.95, 1.05)
        x[:3, :, :] *= scale

        # 3. Random spatial translation
        tx = random.uniform(-0.03, 0.03)
        ty = random.uniform(-0.03, 0.03)
        x[0, :, :] += tx * mask[0, :, :]
        x[1, :, :] += ty * mask[1, :, :]

        # 4. Recompute velocities (channel 3 and 4)
        vx = torch.zeros_like(x[0])
        vy = torch.zeros_like(x[1])
        vx[:, 1:] = x[0, :, 1:] - x[0, :, :-1]
        vy[:, 1:] = x[1, :, 1:] - x[1, :, :-1]
        x[3] = vx
        x[4] = vy

        return {"dynamic": x, "label": y}


# -----------------------------------------------------------------
# 2. EXPERIMENTAL ARCHITECTURE: ST-GCN WITH TEMPORAL ATTENTION POOLING
# -----------------------------------------------------------------
class TemporalAttentionPooling(nn.Module):
    """Multi-Head Temporal Attention Pooling over frame dimension T."""

    def __init__(self, in_features: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, T]
        B, C, T = x.shape
        x_perm = x.permute(0, 2, 1) # [B, T, C]

        Q = self.query(x_perm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T, D]
        K = self.key(x_perm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # [B, H, T, D]
        V = self.value(x_perm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # [B, H, T, D]

        attn_weights = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1) # [B, H, T, T]
        attn_out = attn_weights @ V # [B, H, T, D]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C) # [B, T, C]

        return attn_out.mean(dim=1) # [B, C]


class STGCNAttentionClassifier(nn.Module):
    """ST-GCN Classifier with Learnable Temporal Attention Pooling."""

    def __init__(
        self,
        num_classes: int = 42,
        in_channels: int = 5,
        num_nodes: int = 115,
        channel_progression: list[int] | None = None,
        dropout: float = 0.3,
        classifier_hidden: int = 256,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if channel_progression is None:
            channel_progression = [64, 128, 256, 256]

        self.num_classes = num_classes
        self.backbone = STGCNBackbone(
            in_channels=in_channels,
            channel_progression=channel_progression,
            num_nodes=num_nodes,
            dropout=dropout,
        )
        self.temp_attn = TemporalAttentionPooling(in_features=channel_progression[-1], num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(channel_progression[-1], classifier_hidden),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.backbone(x) # [B, C, T, V]
        spatial_pooled = features.mean(dim=3) # [B, C, T]
        temporal_pooled = self.temp_attn(spatial_pooled) # [B, C]
        return self.classifier(temporal_pooled)


# -----------------------------------------------------------------
# 3. EVALUATION AND TRAINING RUNNER
# -----------------------------------------------------------------
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top3_correct = 0

    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            _, top3_preds = logits.topk(min(3, logits.size(1)), dim=1)
            top1_correct += (top3_preds[:, 0] == y).sum().item()
            top3_correct += (top3_preds == y.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    top1_acc = top1_correct / total_samples if total_samples > 0 else 0.0
    top3_acc = top3_correct / total_samples if total_samples > 0 else 0.0

    return avg_loss, top1_acc, top3_acc, top1_acc


def run_experiment(
    exp_id: str,
    exp_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_label_smoothing: bool = False,
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"RUNNING EXPERIMENT: {exp_id} ({exp_name})")
    print(f"=======================================================")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if use_label_smoothing else 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_top1 = -1.0
    best_val_top3 = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            x, y = unpack_batch(batch)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_samples += x.size(0)

        scheduler.step()
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0

        val_loss, val_top1, val_top3, _ = eval_model(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_top1": round(val_top1, 4),
            "val_top3": round(val_top3, 4),
        })

        if val_top1 > best_val_top1 or (abs(val_top1 - best_val_top1) < 1e-6 and val_loss < best_val_loss):
            best_val_top1 = val_top1
            best_val_top3 = val_top3
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Top-1: {val_top1*100:.2f}% | Val Top-3: {val_top3*100:.2f}%")

    print(f"\n--> BEST VALIDATION RESULT ({exp_id}):")
    print(f"    Best Epoch: {best_epoch}")
    print(f"    Val Top-1:  {best_val_top1*100:.2f}%")
    print(f"    Val Top-3:  {best_val_top3*100:.2f}%")
    print(f"    Val Loss:   {best_val_loss:.4f}")

    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "best_epoch": best_epoch,
        "val_top1": best_val_top1,
        "val_top3": best_val_top3,
        "val_loss": best_val_loss,
        "weights": best_weights,
        "history": history
    }


def main():
    seed_everything(42)
    device = torch.device("cpu")

    controlled_dir = Path("data/processed/include/controlled")
    train_dataset_base = INCLUDEFeatureDataset(split_csv=controlled_dir / "train.csv")
    val_dataset = INCLUDEFeatureDataset(split_csv=controlled_dir / "validation.csv")

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    baseline_val_top1 = 0.6667
    baseline_val_top3 = 0.8730

    print(f"ESTABLISHED BASELINE VALIDATION MARKS:")
    print(f"Baseline Val Top-1: {baseline_val_top1*100:.2f}%")
    print(f"Baseline Val Top-3: {baseline_val_top3*100:.2f}%")

    results = []

    # EXP 1: LANDMARK DATA AUGMENTATION + LABEL SMOOTHING
    aug_train_dataset = AugmentedFeatureDataset(train_dataset_base, augment=True)
    aug_train_loader = DataLoader(aug_train_dataset, batch_size=32, shuffle=True)

    model_exp1 = ControlledSTGCNClassifier(num_classes=42).to(device)
    res_exp1 = run_experiment(
        exp_id="EXP1_AUG_SMOOTHING",
        exp_name="Landmark Data Augmentation + Label Smoothing (0.1)",
        model=model_exp1,
        train_loader=aug_train_loader,
        val_loader=val_loader,
        device=device,
        epochs=35,
        lr=1e-3,
        use_label_smoothing=True
    )
    results.append(res_exp1)

    # EXP 2: ST-GCN WITH TEMPORAL MULTI-HEAD ATTENTION POOLING
    clean_train_loader = DataLoader(train_dataset_base, batch_size=32, shuffle=True)
    model_exp2 = STGCNAttentionClassifier(num_classes=42).to(device)
    res_exp2 = run_experiment(
        exp_id="EXP2_TEMP_ATTN",
        exp_name="ST-GCN + Multi-Head Temporal Attention Pooling",
        model=model_exp2,
        train_loader=clean_train_loader,
        val_loader=val_loader,
        device=device,
        epochs=35,
        lr=1e-3,
        use_label_smoothing=False
    )
    results.append(res_exp2)

    # EXP 3: COMBINED AUGMENTATION + TEMPORAL ATTENTION + LABEL SMOOTHING
    model_exp3 = STGCNAttentionClassifier(num_classes=42).to(device)
    res_exp3 = run_experiment(
        exp_id="EXP3_COMBINED",
        exp_name="Augmentation + Temporal Attention + Label Smoothing",
        model=model_exp3,
        train_loader=aug_train_loader,
        val_loader=val_loader,
        device=device,
        epochs=35,
        lr=1e-3,
        use_label_smoothing=True
    )
    results.append(res_exp3)

    # COMPARISON AND PROMOTION EVALUATION
    print("\n=======================================================")
    print("EXPERIMENTAL RESULTS COMPARISON (VALIDATION ONLY)")
    print("=======================================================")
    print(f"BASELINE:             Val Top-1: {baseline_val_top1*100:.2f}% | Val Top-3: {baseline_val_top3*100:.2f}%")

    promoted_exp = None

    for r in results:
        top1 = r["val_top1"]
        top3 = r["val_top3"]
        status = "REJECTED ❌"

        if top1 > baseline_val_top1 and top3 >= (baseline_val_top3 - 0.01):
            status = "CANDIDATE FOR PROMOTION 🏆"
            if promoted_exp is None or top1 > promoted_exp["val_top1"]:
                promoted_exp = r
        elif top1 == baseline_val_top1 and top3 > baseline_val_top3:
            status = "CANDIDATE FOR PROMOTION (Top-3 Improvement) 🏆"

        print(f"{r['exp_id']} ({r['exp_name']}): Val Top-1: {top1*100:.2f}% | Val Top-3: {top3*100:.2f}% | Status: {status}")

    summary_report = {
        "baseline_val_top1": baseline_val_top1,
        "baseline_val_top3": baseline_val_top3,
        "experiments": [
            {
                "exp_id": r["exp_id"],
                "exp_name": r["exp_name"],
                "val_top1": r["val_top1"],
                "val_top3": r["val_top3"],
                "val_loss": r["val_loss"],
                "best_epoch": r["best_epoch"]
            }
            for r in results
        ],
        "promoted_experiment": promoted_exp["exp_id"] if promoted_exp else None,
        "final_decision": "KEEP CURRENT BASELINE" if promoted_exp is None else f"EVALUATE {promoted_exp['exp_id']} ON HELD-OUT TEST ONCE"
    }

    out_json = Path("reports/experimental_model_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nSaved experimental summary to {out_json}")

    if promoted_exp:
        print(f"\n🏆 PROMOTED EXPERIMENT: {promoted_exp['exp_id']}")
        cand_ckpt_path = Path("models/checkpoints/experimental/candidate_model.pt")
        cand_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(promoted_exp["weights"], cand_ckpt_path)
        print(f"Saved candidate checkpoint to {cand_ckpt_path}")
    else:
        print("\nDecision: Baseline model remains superior on Validation. Keeping current baseline checkpoint!")

if __name__ == "__main__":
    main()

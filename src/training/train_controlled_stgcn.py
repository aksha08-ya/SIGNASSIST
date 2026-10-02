"""Training entrypoint for Controlled ST-GCN experimental model."""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.data.include_dataset import INCLUDEFeatureDataset
from src.models.stgcn import STGCNBackbone


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if _TORCH_AVAILABLE:

    class ControlledSTGCNClassifier(nn.Module):
        """Pure ST-GCN Classifier for controlled vocabulary training (no Transformer)."""

        def __init__(
            self,
            num_classes: int,
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
            self.classifier = nn.Sequential(
                nn.Linear(channel_progression[-1], classifier_hidden),
                nn.ReLU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(classifier_hidden, num_classes),
            )

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            features = self.backbone(x)
            embedding = features.mean(dim=(2, 3))
            return self.classifier(embedding)


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["dynamic"], batch["label"]
    return batch[0], batch[1]


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 42,
) -> tuple[float, float, float, float]:
    """Evaluate loss, top-1, top-3 accuracy, and macro F1 on a data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    top3_correct = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = logits.argmax(dim=1)
            top1_correct += (preds == y).sum().item()
            k = min(3, logits.size(1))
            top3_pred = logits.topk(k, dim=1).indices
            top3_correct += (top3_pred == y.unsqueeze(1)).any(dim=1).sum().item()

            all_targets.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    f1s = []
    for c_id in range(num_classes):
        tp = np.sum((targets_np == c_id) & (preds_np == c_id))
        fp = np.sum((targets_np != c_id) & (preds_np == c_id))
        fn = np.sum((targets_np == c_id) & (preds_np != c_id))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    return (
        total_loss / total_samples,
        top1_correct / total_samples,
        top3_correct / total_samples,
        macro_f1,
    )


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.exists():
        return path
    root = Path(__file__).resolve().parent.parent.parent
    rel = root / path
    if rel.exists():
        return rel
    return path


def run_training(config_path: Path) -> None:
    """Run full training loop based on config YAML."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model training.")

    resolved_config_path = resolve_path(config_path)
    cfg = load_config(resolved_config_path)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    seed_everything(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = resolve_path(data_cfg["vocabulary_path"])
    train_csv = resolve_path(data_cfg["train_csv"])
    val_csv = resolve_path(data_cfg["val_csv"])
    test_csv = resolve_path(data_cfg["test_csv"])
    feature_root = resolve_path(data_cfg["feature_root"])

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    num_classes = vocab_data["num_controlled_classes"]

    train_dataset = INCLUDEFeatureDataset(
        split_csv=train_csv,
        feature_root=feature_root,
        include_static=data_cfg.get("include_static", False),
    )
    val_dataset = INCLUDEFeatureDataset(
        split_csv=val_csv,
        feature_root=feature_root,
        include_static=data_cfg.get("include_static", False),
    )
    test_dataset = INCLUDEFeatureDataset(
        split_csv=test_csv,
        feature_root=feature_root,
        include_static=data_cfg.get("include_static", False),
    )

    test_paths = set(test_dataset.frame["video_path"])
    train_paths = set(train_dataset.frame["video_path"])
    val_paths = set(val_dataset.frame["video_path"])
    if train_paths & test_paths:
        raise ValueError("CRITICAL LEAKAGE: Training set contains test data!")
    if val_paths & test_paths:
        raise ValueError("CRITICAL LEAKAGE: Validation set contains test data!")

    model = ControlledSTGCNClassifier(
        num_classes=num_classes,
        in_channels=model_cfg.get("in_channels", 5),
        num_nodes=model_cfg.get("num_nodes", 115),
        channel_progression=model_cfg.get("channel_progression", [64, 128, 256, 256]),
        dropout=model_cfg.get("dropout", 0.3),
        classifier_hidden=model_cfg.get("classifier_hidden", 256),
        classifier_dropout=model_cfg.get("classifier_dropout", 0.2),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())

    print("\n========================================================", flush=True)
    print("CONTROLLED ST-GCN TRAINING PRE-FLIGHT VERIFICATION", flush=True)
    print("========================================================", flush=True)
    print("EXACT CONFIGURATION:", flush=True)
    print(yaml.dump(cfg, default_flow_style=False), flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Random Seed: {train_cfg['seed']}", flush=True)
    print(f"Target Controlled Classes: {num_classes}", flush=True)
    print(f"Train Samples: {len(train_dataset)}", flush=True)
    print(f"Validation Samples: {len(val_dataset)}", flush=True)
    print(f"Test Samples (Held-out): {len(test_dataset)}", flush=True)
    print(f"Model Parameter Count: {param_count:,}", flush=True)
    print(f"Test Set Isolation Verified: True (Zero overlap with train/val loaders)", flush=True)
    print("========================================================\n", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    checkpoint_dir = resolve_path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Controlled ST-GCN initialized ({num_classes} classes, {param_count:,} params).", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    best_val_top1 = -1.0
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 7)
    patience_counter = 0
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            if train_cfg.get("gradient_clipping"):
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clipping"])
            optimizer.step()

            batch_size = y.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_top1, val_top3, val_macro_f1 = evaluate_loader(
            model, val_loader, criterion, device, num_classes=num_classes
        )

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top3": val_top3,
            "val_macro_f1": val_macro_f1,
            "lr": current_lr,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{train_cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Top-1: {val_top1:.4f} Top-3: {val_top3:.4f} Macro-F1: {val_macro_f1:.4f} | "
            f"LR: {current_lr:.1e} | Best Epoch: {best_epoch if best_val_top1 > -1 else epoch} | "
            f"Patience: {patience_counter}/{patience}",
            flush=True,
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "num_classes": num_classes,
                    "vocabulary_path": str(vocab_path),
                },
                checkpoint_dir / "controlled_stgcn_best.pt",
            )
            print(f"  --> Saved new best checkpoint at epoch {epoch} (Val Top-1: {val_top1*100:.2f}%)", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.", flush=True)
                break

        gc.collect()

    torch.save(
        {
            "epoch": len(history),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_top1": best_val_top1,
            "num_classes": num_classes,
            "vocabulary_path": str(vocab_path),
        },
        checkpoint_dir / "controlled_stgcn_last.pt",
    )

    report_out = resolve_path(train_cfg["report_out"])
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_cfg["name"],
                "num_classes": num_classes,
                "best_epoch": best_epoch,
                "best_val_top1": best_val_top1,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved training history to {report_out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled ST-GCN experimental model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/controlled_stgcn.yaml"),
        help="Path to training config YAML",
    )
    args = parser.parse_args()
    try:
        run_training(args.config)
    except Exception as err:
        import traceback
        tb = traceback.format_exc()
        print("FATAL ERROR IN RUN_TRAINING:\n", tb, flush=True)
        Path("train_crash.txt").write_text(tb, encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.include_dataset import INCLUDEFeatureDataset
from src.models.stgcn import STGCNBackbone


SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 262

TRAIN_CSV = "data/processed/include/splits/train.csv"
VAL_CSV = "data/processed/include/splits/validation.csv"
FEATURE_ROOT = "data/processed/include/features"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class STGCNClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = STGCNBackbone(
            in_channels=5,
            channel_progression=[64, 128, 256, 256],
            num_nodes=115,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, mask=None):
        # x: [B, 5, T, V]
        features = self.backbone(x)

        # Controlled baseline: global mean over time and graph nodes.
        embedding = features.mean(dim=(2, 3))

        return self.classifier(embedding)


def unpack_batch(batch):
    if isinstance(batch, dict):
        x = batch["dynamic"]
        y = batch["label"]
        return x, y

    x, y = batch
    return x, y


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    top1 = 0
    top3 = 0

    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

            top1 += (logits.argmax(dim=1) == y).sum().item()

            k = min(3, logits.size(1))
            top3_pred = logits.topk(k, dim=1).indices
            top3 += (top3_pred == y.unsqueeze(1)).any(dim=1).sum().item()

    return (
        total_loss / total,
        top1 / total,
        top3 / total,
    )


def main():
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = INCLUDEFeatureDataset(
        split_csv=TRAIN_CSV,
        feature_root=FEATURE_ROOT,
        include_static=False,
    )
    val_dataset = INCLUDEFeatureDataset(
        split_csv=VAL_CSV,
        feature_root=FEATURE_ROOT,
        include_static=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = STGCNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {EPOCHS}")
    print()

    history = []
    checkpoint_dir = Path("models/checkpoints/stgcn_baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_top1 = -1.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_top1, val_top3 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_top1": val_top1,
            "val_top3": val_top3,
        }
        history.append(record)

        # Save the best model using validation Top-1 accuracy.
        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_top1": val_top1,
                    "val_top3": val_top3,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                },
                checkpoint_dir / "stgcn_best.pt",
            )
            print(f"  -> New best checkpoint: epoch {epoch}")

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_top1={val_top1:.4f} "
            f"val_top3={val_top3:.4f}"
        )

    torch.save(
        {
            "epoch": EPOCHS,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_top1": best_val_top1,
            "num_classes": NUM_CLASSES,
            "seed": SEED,
        },
        checkpoint_dir / "stgcn_last.pt",
    )

    output = Path("reports/training/stgcn_baseline.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "weight_decay": WEIGHT_DECAY,
                "num_classes": NUM_CLASSES,
                "device": str(device),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_top1": best_val_top1,
            },
            indent=2,
        )
    )

    print()
    print(f"Saved results to: {output}")


if __name__ == "__main__":
    main()

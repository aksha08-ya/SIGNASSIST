"""Training loop for the SIGNASSIST dynamic ST-GCN + Transformer classifier."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from src.data.include_dataset import INCLUDEFeatureDataset
from src.models.dynamic_classifier import DynamicSignClassifier
from src.utils.metrics import MetricsEvaluator


@dataclass
class TrainerConfig:
    """Configuration for one dynamic-classifier training run."""

    train_csv: str = "data/processed/include/splits/train.csv"
    validation_csv: str = "data/processed/include/splits/validation.csv"
    feature_root: str = "data/processed/include/features"

    checkpoint_dir: str = "models/checkpoints/dynamic_classifier"
    history_path: str = "reports/training/dynamic_training_history.json"

    epochs: int = 10
    batch_size: int = 4
    num_workers: int = 0

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0

    class_weight_cap: float = 5.0

    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

    seed: int = 42
    device: str = "auto"

    log_every: int = 25


@dataclass
class EpochResult:
    """Metrics recorded for one epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    top1_accuracy: float
    top3_accuracy: float
    macro_f1: float
    learning_rate: float
    epoch_seconds: float


def set_seed(seed: int) -> None:
    """Set deterministic random seeds where supported."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferable for the controlled baseline run.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic training device."""

    requested = requested.lower().strip()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")

    return torch.device(requested)


def build_class_weights(
    labels: np.ndarray,
    num_classes: int,
    cap: float,
) -> torch.Tensor:
    """Build capped inverse-frequency class weights.

    The cap prevents the rarest classes from dominating the loss because
    the INCLUDE training distribution is strongly imbalanced.
    """

    labels = np.asarray(labels, dtype=np.int64)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)

    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(
            f"Training split has no samples for classes: {missing[:20]}"
            + ("..." if len(missing) > 20 else "")
        )

    # Median-frequency balancing:
    #   weight_i = median(counts) / count_i
    # This is less aggressive than raw inverse frequency.
    median_count = float(np.median(counts))
    weights = median_count / counts
    weights = np.minimum(weights, float(cap))

    # Normalize around 1 so the overall loss scale remains stable.
    weights /= float(np.mean(weights))

    return torch.tensor(weights, dtype=torch.float32)


class DynamicTrainer:
    """Train and validate the dynamic 262-class SIGNASSIST model."""

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config = config or TrainerConfig()
        set_seed(self.config.seed)

        self.device = resolve_device(self.config.device)

        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history_path = Path(self.config.history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self.train_dataset = INCLUDEFeatureDataset(
            self.config.train_csv,
            feature_root=self.config.feature_root,
            include_static=False,
            strict=True,
        )
        self.validation_dataset = INCLUDEFeatureDataset(
            self.config.validation_csv,
            feature_root=self.config.feature_root,
            include_static=False,
            strict=True,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        self.validation_loader = DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        self.model = DynamicSignClassifier.from_config().to(self.device)

        num_classes = self.model.num_classes

        if num_classes != 262:
            raise ValueError(
                f"Expected 262 dynamic classes, model has {num_classes}."
            )

        self.class_weights = build_class_weights(
            self.train_dataset.labels,
            num_classes=num_classes,
            cap=self.config.class_weight_cap,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.metrics_evaluator = MetricsEvaluator(zero_division=0.0)

        self.start_epoch = 1
        self.best_macro_f1 = -math.inf
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []

    def _move_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move model inputs and labels to the active device."""

        dynamic = batch["dynamic"].to(self.device, non_blocking=True)
        mask = batch["dynamic_mask"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)

        return dynamic, mask, labels

    def train_one_epoch(
        self,
        epoch: int,
        max_batches: Optional[int] = None,
    ) -> float:
        """Run one training epoch, optionally capped for smoke tests."""

        self.model.train()

        running_loss = 0.0
        sample_count = 0

        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(self.train_loader, start=1):
            dynamic, mask, labels = self._move_batch(batch)

            self.optimizer.zero_grad(set_to_none=True)

            output = self.model(dynamic, mask=mask)
            loss = self.criterion(output.logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, batch {batch_index}: {loss}"
                )

            loss.backward()

            clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm,
            )

            self.optimizer.step()

            batch_size = labels.shape[0]
            running_loss += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size

            if max_batches is not None and batch_index >= max_batches:
                break

            if self.config.log_every > 0 and batch_index % self.config.log_every == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"  train batch {batch_index}/{len(self.train_loader)}"
                    f" | loss={running_loss / sample_count:.4f}"
                    f" | elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if sample_count == 0:
            raise RuntimeError("Training loader produced zero samples.")

        return running_loss / sample_count

    @torch.no_grad()
    def validate(
        self,
        max_batches: Optional[int] = None,
    ) -> tuple[float, float, float, float]:
        """Run validation, optionally capped for smoke tests."""

        self.model.eval()

        running_loss = 0.0
        sample_count = 0

        all_true: list[int] = []
        all_pred: list[int] = []

        top1_correct = 0
        top3_correct = 0

        for batch_index, batch in enumerate(self.validation_loader, start=1):
            dynamic, mask, labels = self._move_batch(batch)

            output = self.model(dynamic, mask=mask)
            logits = output.logits

            loss = self.criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite validation loss: {loss}")

            batch_size = labels.shape[0]
            running_loss += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size

            top1 = logits.argmax(dim=1)
            top1_correct += int((top1 == labels).sum().item())

            top3 = torch.topk(logits, k=3, dim=1).indices
            top3_correct += int(
                (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()
            )

            all_true.extend(labels.detach().cpu().tolist())
            all_pred.extend(top1.detach().cpu().tolist())

            if max_batches is not None and batch_index >= max_batches:
                break

        if sample_count == 0:
            raise RuntimeError("Validation loader produced zero samples.")

        metrics = self.metrics_evaluator.evaluate(
            all_true,
            all_pred,
            num_classes=self.model.num_classes,
        )

        validation_loss = running_loss / sample_count
        top1_accuracy = top1_correct / sample_count
        top3_accuracy = top3_correct / sample_count

        return (
            validation_loss,
            top1_accuracy,
            top3_accuracy,
            metrics.macro_f1,
        )

    def _checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        """Build a resumable checkpoint payload."""

        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_macro_f1": self.best_macro_f1,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "trainer_config": asdict(self.config),
            "model_num_classes": self.model.num_classes,
            "seed": self.config.seed,
        }

    def save_checkpoint(self, path: Path, epoch: int) -> None:
        """Save a checkpoint atomically."""

        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(epoch), temporary)
        temporary.replace(path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint."""

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        required = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "best_macro_f1",
            "best_epoch",
            "epochs_without_improvement",
        }
        missing = required - set(checkpoint)
        if missing:
            raise ValueError(
                f"Checkpoint is missing required fields: {sorted(missing)}"
            )

        if int(checkpoint.get("model_num_classes", -1)) != self.model.num_classes:
            raise ValueError("Checkpoint class count does not match current model.")

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_macro_f1 = float(checkpoint["best_macro_f1"])
        self.best_epoch = int(checkpoint["best_epoch"])
        self.epochs_without_improvement = int(
            checkpoint["epochs_without_improvement"]
        )

        print(
            f"Resumed from {path} | next epoch={self.start_epoch}"
            f" | best macro-F1={self.best_macro_f1:.4f}",
            flush=True,
        )

    def _save_history(self) -> None:
        """Persist training history and run metadata."""

        payload = {
            "trainer_config": asdict(self.config),
            "device": str(self.device),
            "train_samples": len(self.train_dataset),
            "validation_samples": len(self.validation_dataset),
            "num_classes": self.model.num_classes,
            "class_weights": self.class_weights.detach().cpu().tolist(),
            "best_epoch": self.best_epoch,
            "best_macro_f1": self.best_macro_f1,
            "history": self.history,
        }

        temporary = self.history_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.history_path)

    def fit(self, resume_from: Optional[str | Path] = None) -> list[dict[str, Any]]:
        """Train the model for the configured number of epochs."""

        if resume_from is not None:
            self.load_checkpoint(resume_from)

        print("=" * 72)
        print("SIGNASSIST Dynamic Classifier Training")
        print("=" * 72)
        print(f"Device:              {self.device}")
        print(f"Train samples:       {len(self.train_dataset)}")
        print(f"Validation samples:  {len(self.validation_dataset)}")
        print(f"Classes:             {self.model.num_classes}")
        print(f"Batch size:          {self.config.batch_size}")
        print(f"Epochs:              {self.config.epochs}")
        print(f"Learning rate:       {self.config.learning_rate}")
        print(f"Weight decay:        {self.config.weight_decay}")
        print(f"Gradient clip:       {self.config.gradient_clip_norm}")
        print(f"Checkpoint dir:      {self.checkpoint_dir}")
        print("=" * 72)

        for epoch in range(self.start_epoch, self.config.epochs + 1):
            epoch_start = time.perf_counter()

            train_loss = self.train_one_epoch(epoch)
            (
                validation_loss,
                top1_accuracy,
                top3_accuracy,
                macro_f1,
            ) = self.validate()

            epoch_seconds = time.perf_counter() - epoch_start

            current_lr = float(self.optimizer.param_groups[0]["lr"])

            result = EpochResult(
                epoch=epoch,
                train_loss=float(train_loss),
                validation_loss=float(validation_loss),
                top1_accuracy=float(top1_accuracy),
                top3_accuracy=float(top3_accuracy),
                macro_f1=float(macro_f1),
                learning_rate=current_lr,
                epoch_seconds=float(epoch_seconds),
            )

            self.history.append(asdict(result))

            improved = macro_f1 > self.best_macro_f1 + self.config.early_stopping_min_delta

            if improved:
                self.best_macro_f1 = macro_f1
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                self.save_checkpoint(
                    self.checkpoint_dir / "dynamic_classifier_best.pt",
                    epoch,
                )
            else:
                self.epochs_without_improvement += 1

            self.save_checkpoint(
                self.checkpoint_dir / "dynamic_classifier_last.pt",
                epoch,
            )

            self._save_history()

            print(
                f"Epoch {epoch:02d}/{self.config.epochs}"
                f" | train_loss={train_loss:.4f}"
                f" | val_loss={validation_loss:.4f}"
                f" | top1={top1_accuracy:.4f}"
                f" | top3={top3_accuracy:.4f}"
                f" | macro_f1={macro_f1:.4f}"
                f" | time={epoch_seconds:.1f}s"
                f"{' | BEST' if improved else ''}",
                flush=True,
            )

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(
                    f"Early stopping after epoch {epoch}; "
                    f"best epoch={self.best_epoch}, "
                    f"best macro-F1={self.best_macro_f1:.4f}",
                    flush=True,
                )
                break

        return self.history


def main() -> None:
    """Run a default training job."""

    trainer = DynamicTrainer()
    trainer.fit()


if __name__ == "__main__":
    main()

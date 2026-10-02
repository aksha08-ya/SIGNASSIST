"""Temperature scaling calibration for Controlled ST-GCN model on validation split."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize

from src.data.include_dataset import INCLUDEFeatureDataset
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier, unpack_batch


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def main() -> None:
    ckpt_path = Path("models/checkpoints/experimental/controlled_stgcn_best.pt")
    val_csv = Path("data/processed/include/controlled/validation.csv")
    vocab_path = Path("data/processed/include/vocabulary_initial.json")
    feature_root = Path("data/processed/include/features")
    out_calibration = Path("models/checkpoints/experimental/calibration.json")

    print("========================================================")
    print("PHASE 1: EXPERIMENTAL ST-GCN TEMPERATURE CALIBRATION")
    print("========================================================")
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"Validation dataset: {val_csv} (Strictly validation split only)")
    print(f"Vocabulary file:    {vocab_path}")

    # Load vocab and model
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    num_classes = vocab["num_controlled_classes"]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ControlledSTGCNClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load validation dataset ONLY
    val_dataset = INCLUDEFeatureDataset(
        split_csv=val_csv,
        feature_root=feature_root,
        include_static=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in val_loader:
            x, y = unpack_batch(batch)
            logits = model(x)
            logits_list.append(logits)
            labels_list.append(y)

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Initial metrics at T = 1.0
    criterion = nn.CrossEntropyLoss()
    nll_before = criterion(all_logits, all_labels).item()
    probs_before = torch.softmax(all_logits, dim=1).numpy()
    labels_np = all_labels.numpy()
    ece_before = compute_ece(probs_before, labels_np, n_bins=10)
    preds_before = probs_before.argmax(axis=1)
    top1_before = float(np.mean(preds_before == labels_np))

    # Optimize Temperature T > 0 using scipy L-BFGS-B
    def loss_func(t_val: np.ndarray) -> float:
        t = t_val[0]
        scaled_logits = all_logits / t
        return criterion(scaled_logits, all_labels).item()

    res = minimize(loss_func, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
    optimal_t = float(res.x[0])

    # Calibrated metrics at T = optimal_t
    scaled_logits = all_logits / optimal_t
    nll_after = criterion(scaled_logits, all_labels).item()
    probs_after = torch.softmax(scaled_logits, dim=1).numpy()
    ece_after = compute_ece(probs_after, labels_np, n_bins=10)
    preds_after = probs_after.argmax(axis=1)
    top1_after = float(np.mean(preds_after == labels_np))

    # Save calibration metadata
    calib_data = {
        "temperature": optimal_t,
        "num_classes": num_classes,
        "val_nll_before": nll_before,
        "val_nll_after": nll_after,
        "val_ece_before": ece_before,
        "val_ece_after": ece_after,
        "val_top1_before": top1_before,
        "val_top1_after": top1_after,
        "accessed_test_csv": False,
    }

    out_calibration.parent.mkdir(parents=True, exist_ok=True)
    with open(out_calibration, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=2)

    print("\n--- CALIBRATION RESULTS ---")
    print(f"Learned Temperature (T):        {optimal_t:.4f}")
    print(f"Validation NLL:                  {nll_before:.4f} -> {nll_after:.4f}")
    print(f"Validation ECE:                  {ece_before:.4f} -> {ece_after:.4f}")
    print(f"Validation Top-1 Accuracy:       {top1_before*100:.2f}% -> {top1_after*100:.2f}%")
    print(f"Saved calibration metadata to:   {out_calibration}")
    print("Held-out test split (test.csv) accessed: False (CONFIRMED)")
    print("========================================================\n")


if __name__ == "__main__":
    main()

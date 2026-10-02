"""Evaluation Script for Saved ST-GCN Baseline Best Checkpoint.

Evaluates models/checkpoints/stgcn_baseline/stgcn_best.pt directly on all 615 validation NPZ samples.
Verifies strict=True loading.
Reports Top-1, Top-3, Number Correct, Unique Classes, Class Prediction Frequency, Mean/Max Confidence.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.include_dataset import INCLUDEFeatureDataset
from src.training.stgcn_baseline import STGCNClassifier


def main():
    print("=" * 80)
    print("EVALUATION OF REPRODUCED ST-GCN BASELINE BEST CHECKPOINT")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = repo_root / "models" / "checkpoints" / "stgcn_baseline" / "stgcn_best.pt"
    val_csv = repo_root / "data" / "processed" / "include" / "splits" / "validation.csv"
    features_dir = repo_root / "data" / "processed" / "include" / "features"
    mapping_path = repo_root / "data" / "processed" / "include" / "class_mapping.json"

    print(f"Checkpoint Path: {ckpt_path}")
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    id_to_class = mapping_data.get("id_to_class", {})

    # 1. Verify strict=True loading
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Loaded checkpoint dictionary. Best Epoch in ckpt: {ckpt.get('epoch', 'N/A')}")
    print(f"Checkpoint val_top1: {ckpt.get('val_top1', 'N/A')}, val_top3: {ckpt.get('val_top3', 'N/A')}")

    model = STGCNClassifier(num_classes=262)
    state_dict = ckpt["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("VERIFIED: Saved stgcn_best.pt loaded successfully with strict=True.")

    # 2. Evaluate directly on all 615 validation samples
    val_dataset = INCLUDEFeatureDataset(split_csv=val_csv, feature_root=features_dir)
    print(f"Evaluating over all {len(val_dataset)} validation feature samples...")

    top1_hits = 0
    top3_hits = 0
    predictions = []
    confidences = []
    pred_counts = {}

    with torch.no_grad():
        for i in range(len(val_dataset)):
            item = val_dataset[i]
            true_cls = int(item["label"])
            d_feat = item["dynamic"].unsqueeze(0).float()  # [1, 5, 60, 115]

            logits = model(d_feat)
            probs = torch.softmax(logits[0], dim=-1)
            conf, pred_cls = torch.max(probs, dim=-1)
            top3_vals, top3_cls = torch.topk(probs, k=3)

            p_item = pred_cls.item()
            predictions.append(p_item)
            confidences.append(conf.item())
            pred_counts[p_item] = pred_counts.get(p_item, 0) + 1

            if p_item == true_cls:
                top1_hits += 1
            if true_cls in top3_cls.tolist():
                top3_hits += 1

    n_samples = len(predictions)
    top1_acc = top1_hits / n_samples
    top3_acc = top3_hits / n_samples
    unique_preds = len(pred_counts)
    mean_conf = float(np.mean(confidences))
    max_conf = float(np.max(confidences))

    print("\n" + "=" * 80)
    print("REPRODUCED ST-GCN BASELINE BEST CHECKPOINT EVALUATION RESULTS")
    print("=" * 80)
    print(f"Best Epoch: {ckpt.get('epoch')}")
    print(f"Validation Top-1 Accuracy: {top1_acc*100:.2f}% ({top1_hits} / {n_samples})")
    print(f"Validation Top-3 Accuracy: {top3_acc*100:.2f}% ({top3_hits} / {n_samples})")
    print(f"Unique Predicted Classes: {unique_preds} / 262")
    print(f"Mean Raw Softmax Confidence (UNCALIBRATED): {mean_conf:.4f} ({mean_conf*100:.2f}%)")
    print(f"Max Raw Softmax Confidence (UNCALIBRATED): {max_conf:.4f} ({max_conf*100:.2f}%)")
    print("\nTop 15 Most Frequently Predicted Classes:")
    sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
    for c, count in sorted_preds[:15]:
        print(f"   Class {c:3d} ('{id_to_class.get(str(c), 'N/A')}'): {count} samples ({count/n_samples*100:.2f}%)")


if __name__ == "__main__":
    main()

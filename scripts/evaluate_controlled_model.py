"""Evaluation script for models on controlled INCLUDE vocabulary splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.include_dataset import INCLUDEFeatureDataset
from src.models.dynamic_classifier import DynamicSignClassifier
from src.training.train_controlled_stgcn import ControlledSTGCNClassifier, unpack_batch


def evaluate_model_on_split(
    model: nn.Module,
    loader: DataLoader,
    vocab_data: dict[str, Any],
    device: torch.device,
    is_production_model: bool = False,
) -> dict[str, Any]:
    """Evaluate a model on a controlled split and return detailed metrics."""
    model.eval()

    all_targets = []
    all_preds = []
    all_top3 = []
    all_max_probs = []

    # Map original class labels if evaluating production model
    label_to_controlled_id = vocab_data["label_to_controlled_id"]
    controlled_id_to_label = vocab_data["controlled_id_to_label"]
    num_controlled_classes = vocab_data["num_controlled_classes"]

    # Production model uses full 262 class mapping
    class_mapping_path = Path("data/processed/include/class_mapping.json")
    prod_label_to_id = {}
    prod_id_to_label = {}
    if class_mapping_path.exists():
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            prod_label_to_id = c_data.get("class_to_id", {})
            prod_id_to_label = {v: k for k, v in prod_label_to_id.items()}

    with torch.no_grad():
        for batch in loader:
            x, y = unpack_batch(batch)
            x = x.to(device, non_blocking=True)
            
            # y is the controlled_class_id
            target_ids = y.numpy()

            output = model(x)
            if hasattr(output, "logits"):
                logits = output.logits
            elif isinstance(output, dict) and "logits" in output:
                logits = output["logits"]
            else:
                logits = output

            probs = torch.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=-1)

            k = min(3, logits.size(-1))
            top3_indices = torch.topk(logits, k=k, dim=-1).indices.cpu().numpy()

            preds_np = preds.cpu().numpy()
            max_probs_np = max_probs.cpu().numpy()

            if is_production_model:
                # Map 262-class output predictions to controlled_class_ids if possible
                mapped_preds = []
                for p_idx in preds_np:
                    pred_label = prod_id_to_label.get(p_idx, "")
                    mapped_c_id = label_to_controlled_id.get(pred_label, -1)
                    mapped_preds.append(mapped_c_id)
                preds_np = np.array(mapped_preds)

                # Map top3 predictions
                mapped_top3 = []
                for top3_row in top3_indices:
                    row_c_ids = [
                        label_to_controlled_id.get(prod_id_to_label.get(pid, ""), -1)
                        for pid in top3_row
                    ]
                    mapped_top3.append(row_c_ids)
                top3_indices = np.array(mapped_top3)

            all_targets.extend(target_ids)
            all_preds.extend(preds_np)
            all_top3.extend(top3_indices)
            all_max_probs.extend(max_probs_np)

    targets_np = np.array(all_targets)
    preds_np = np.array(all_preds)
    probs_np = np.array(all_max_probs)

    total_samples = len(targets_np)
    if total_samples == 0:
        raise ValueError("No samples evaluated!")

    # Top-1 Accuracy
    top1_correct = np.sum(targets_np == preds_np)
    top1_acc = float(top1_correct / total_samples)

    # Top-3 Accuracy
    top3_correct = 0
    for i in range(total_samples):
        if targets_np[i] in all_top3[i]:
            top3_correct += 1
    top3_acc = float(top3_correct / total_samples)

    # Per-class metrics & Macro averages
    confusion_mat = np.zeros((num_controlled_classes, num_controlled_classes), dtype=int)
    for t, p in zip(targets_np, preds_np):
        if 0 <= t < num_controlled_classes and 0 <= p < num_controlled_classes:
            confusion_mat[t, p] += 1

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for c_id in range(num_controlled_classes):
        label_name = controlled_id_to_label.get(str(c_id), f"Class_{c_id}")
        tp = confusion_mat[c_id, c_id]
        fp = np.sum(confusion_mat[:, c_id]) - tp
        fn = np.sum(confusion_mat[c_id, :]) - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class_metrics[label_name] = {
            "controlled_class_id": c_id,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": int(np.sum(confusion_mat[c_id, :])),
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Prediction Distribution
    unique_preds, pred_counts = np.unique(preds_np, return_counts=True)
    sorted_pred_indices = np.argsort(-pred_counts)
    pred_distribution = {
        controlled_id_to_label.get(str(unique_preds[i]), f"Unmapped_{unique_preds[i]}"): int(pred_counts[i])
        for i in sorted_pred_indices
    }
    most_common_classes = [
        (controlled_id_to_label.get(str(unique_preds[i]), f"Unmapped_{unique_preds[i]}"), int(pred_counts[i]))
        for i in sorted_pred_indices[:5]
    ]

    # Confidence Distribution
    samples_above_070 = int(np.sum(probs_np >= 0.70))
    confidence_stats = {
        "min": float(np.min(probs_np)),
        "max": float(np.max(probs_np)),
        "mean": float(np.mean(probs_np)),
        "median": float(np.median(probs_np)),
        "std": float(np.std(probs_np)),
        "samples_above_0_70": samples_above_070,
        "percentage_above_0_70": float(samples_above_070 / total_samples),
    }

    report = {
        "total_samples": total_samples,
        "num_controlled_classes": num_controlled_classes,
        "top1_accuracy": top1_acc,
        "top3_accuracy": top3_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "unique_predicted_classes": len(unique_preds),
        "most_common_predicted_classes": most_common_classes,
        "confidence_stats": confidence_stats,
        "prediction_distribution": pred_distribution,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_mat.tolist(),
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on controlled vocabulary split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--model-type",
        choices=["controlled_stgcn", "production_dynamic"],
        default="controlled_stgcn",
        help="Model architecture type",
    )
    parser.add_argument(
        "--vocab-json",
        type=Path,
        default=Path("data/processed/include/vocabulary_initial.json"),
        help="Path to controlled vocabulary JSON",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/processed/include/controlled/test.csv"),
        help="Path to split manifest CSV (test.csv or validation.csv)",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("data/processed/include/features"),
        help="Path to feature root directory",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("reports/evaluations/controlled_eval.json"),
        help="Output path for evaluation report JSON",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating checkpoint {args.checkpoint} on {args.split_csv} using device {device}...")

    with open(args.vocab_json, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    # Load dataset
    dataset = INCLUDEFeatureDataset(
        split_csv=args.split_csv,
        feature_root=args.feature_root,
        include_static=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # Instantiate model
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.model_type == "controlled_stgcn":
        num_classes = vocab_data["num_controlled_classes"]
        model = ControlledSTGCNClassifier(num_classes=num_classes)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        is_production = False
    else:
        # Production ST-GCN + Transformer (262 classes)
        model = DynamicSignClassifier(num_classes=262)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        is_production = True

    model = model.to(device)

    report = evaluate_model_on_split(
        model=model,
        loader=loader,
        vocab_data=vocab_data,
        device=device,
        is_production_model=is_production,
    )

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Evaluation Results ---")
    print(f"Split: {args.split_csv.name} ({report['total_samples']} samples)")
    print(f"Top-1 Accuracy: {report['top1_accuracy'] * 100:.2f}%")
    print(f"Top-3 Accuracy: {report['top3_accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score: {report['macro_f1']:.4f}")
    print(f"Unique Predicted Classes: {report['unique_predicted_classes']} / {report['num_controlled_classes']}")
    print(f"Confidence (min/max/mean): {report['confidence_stats']['min']:.4f} / {report['confidence_stats']['max']:.4f} / {report['confidence_stats']['mean']:.4f}")
    print(f"Report saved to: {args.out_report}")


if __name__ == "__main__":
    main()

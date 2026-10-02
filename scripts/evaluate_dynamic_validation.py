"""Evaluation Script for SIGNASSIST Dynamic-Only Model on INCLUDE Validation Set."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.include_dataset import INCLUDEFeatureDataset
from src.pipeline.recognition import RecognitionPipeline
from src.utils.metrics import MetricsEvaluator


def main() -> None:
    split_csv = "data/processed/include/splits/validation.csv"
    feature_root = "data/processed/include/features"
    class_mapping_path = "data/processed/include/class_mapping.json"
    checkpoint_path = "models/checkpoints/dynamic_classifier/dynamic_classifier_best.pt"
    reports_dir = Path("reports/evaluation")
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_out_path = reports_dir / "dynamic_validation.json"
    csv_out_path = reports_dir / "dynamic_validation_confusion_matrix.csv"

    # 1. Load Dataset
    dataset = INCLUDEFeatureDataset(
        split_csv=split_csv,
        feature_root=feature_root,
        include_static=False,
        strict=True,
    )
    total_samples = len(dataset)
    print(f"Loaded validation dataset: {total_samples} samples.")

    # 2. Load Pipeline in dynamic_only mode
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        dynamic_checkpoint_path=checkpoint_path,
        class_mapping_path=class_mapping_path,
    )
    if not pipeline.is_available():
        raise RuntimeError(f"Pipeline not ready: {pipeline._readiness_reason}")

    # 3. DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []
    all_accepted: list[bool] = []
    all_top3: list[list[int]] = []
    all_top3_match: list[bool] = []
    all_video_paths: list[str] = []

    # Run inference
    pipeline.dynamic_classifier.eval()

    with torch.no_grad():
        for batch in loader:
            dynamic = batch["dynamic"].to(pipeline.device)       # [B, 5, 60, 115]
            mask = batch["dynamic_mask"].to(pipeline.device)     # [B, 60, 115]
            labels = batch["label"].cpu().tolist()               # list of int
            video_paths = batch["video_path"]                    # list of str

            out = pipeline.dynamic_classifier(dynamic, mask=mask)
            conf_out = pipeline.confidence_estimator(out.logits)

            pred_classes = conf_out.predicted_class.cpu().tolist()
            confidences = conf_out.confidence.cpu().tolist()
            accepted_flags = conf_out.is_accepted.cpu().tolist()
            top3_indices = conf_out.top_k_indices.cpu().tolist()

            for true_id, pred_id, conf, is_acc, top3_idxs, v_path in zip(
                labels, pred_classes, confidences, accepted_flags, top3_indices, video_paths
            ):
                all_true.append(true_id)
                all_pred.append(pred_id)
                all_conf.append(float(conf))
                all_accepted.append(bool(is_acc))
                all_top3.append([int(x) for x in top3_idxs])
                all_top3_match.append(true_id in top3_idxs)
                all_video_paths.append(v_path)

    all_true_arr = np.array(all_true, dtype=int)
    all_pred_arr = np.array(all_pred, dtype=int)
    all_conf_arr = np.array(all_conf, dtype=float)
    all_accepted_arr = np.array(all_accepted, dtype=bool)
    all_top3_match_arr = np.array(all_top3_match, dtype=bool)

    # Metrics
    num_classes = pipeline.dynamic_classifier.num_classes  # 262
    top1_accuracy = float(np.mean(all_true_arr == all_pred_arr))
    top3_accuracy = float(np.mean(all_top3_match_arr))

    # Evaluate full 262-class metrics
    evaluator = MetricsEvaluator(zero_division=0.0)
    full_eval = evaluator.evaluate(
        all_true_arr,
        all_pred_arr,
        num_classes=num_classes,
        class_names=[pipeline.get_dynamic_label(i) for i in range(num_classes)],
    )

    accepted_count = int(np.sum(all_accepted_arr))
    rejected_count = int(total_samples - accepted_count)

    mean_confidence = float(np.mean(all_conf_arr))
    median_confidence = float(np.median(all_conf_arr))

    # Prediction class distribution
    unique_pred_classes, pred_counts = np.unique(all_pred_arr, return_counts=True)
    num_unique_predicted = len(unique_pred_classes)

    # Sort top 20 most frequent predictions
    sorted_pred_indices = np.argsort(-pred_counts)
    top_20_pred = []
    for idx in sorted_pred_indices[:20]:
        cls_id = int(unique_pred_classes[idx])
        cnt = int(pred_counts[idx])
        top_20_pred.append({
            "class_id": cls_id,
            "gloss": pipeline.get_dynamic_label(cls_id),
            "count": cnt,
            "percentage": float(cnt / total_samples * 100),
        })

    # Accuracy on classes that actually appear in validation
    present_classes = np.unique(all_true_arr)
    num_present_classes = len(present_classes)

    # Per-class metrics for present classes
    present_class_metrics = {}
    present_precisions = []
    present_recalls = []
    present_f1s = []

    for cls_id in present_classes:
        cls_gloss = pipeline.get_dynamic_label(int(cls_id))
        metrics = full_eval.per_class_metrics.get(cls_gloss, {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0})
        present_class_metrics[cls_gloss] = metrics
        present_precisions.append(metrics["precision"])
        present_recalls.append(metrics["recall"])
        present_f1s.append(metrics["f1"])

    present_macro_precision = float(np.mean(present_precisions))
    present_macro_recall = float(np.mean(present_recalls))
    present_macro_f1 = float(np.mean(present_f1s))

    # Accepted-only accuracy
    if accepted_count > 0:
        accepted_mask = all_accepted_arr
        accepted_top1_acc = float(np.mean(all_true_arr[accepted_mask] == all_pred_arr[accepted_mask]))
    else:
        accepted_top1_acc = None

    # 20 Concrete Examples
    examples = []
    sample_indices = np.linspace(0, total_samples - 1, num=20, dtype=int)
    for idx in sample_indices:
        t_id = int(all_true[idx])
        p_id = int(all_pred[idx])
        examples.append({
            "sample_index": int(idx),
            "video_path": all_video_paths[idx],
            "true_class_id": t_id,
            "true_gloss": pipeline.get_dynamic_label(t_id),
            "predicted_class_id": p_id,
            "predicted_gloss": pipeline.get_dynamic_label(p_id),
            "raw_softmax_confidence": float(all_conf[idx]),
            "confidence_label": "UNCALIBRATED_SOFTMAX",
            "is_accepted": bool(all_accepted[idx]),
        })

    # Build summary payload
    report_data = {
        "evaluation_name": "SIGNASSIST Dynamic-Only Validation Evaluation",
        "dataset_split": "validation",
        "checkpoint": checkpoint_path,
        "total_samples": total_samples,
        "num_vocabulary_classes": num_classes,
        "num_classes_in_validation_split": num_present_classes,
        "metrics": {
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "accepted_only_accuracy": accepted_top1_acc,
            "macro_precision_all_classes": full_eval.macro_precision,
            "macro_recall_all_classes": full_eval.macro_recall,
            "macro_f1_all_classes": full_eval.macro_f1,
            "macro_precision_present_classes": present_macro_precision,
            "macro_recall_present_classes": present_macro_recall,
            "macro_f1_present_classes": present_macro_f1,
        },
        "threshold_analysis": {
            "confidence_threshold": 0.70,
            "accepted_predictions_count": accepted_count,
            "accepted_predictions_percentage": float(accepted_count / total_samples * 100),
            "rejected_predictions_count": rejected_count,
            "rejected_predictions_percentage": float(rejected_count / total_samples * 100),
        },
        "raw_softmax_confidence_distribution": {
            "note": "UNCALIBRATED raw softmax probabilities",
            "mean": mean_confidence,
            "median": median_confidence,
            "min": float(np.min(all_conf_arr)),
            "max": float(np.max(all_conf_arr)),
        },
        "prediction_distribution": {
            "unique_classes_predicted": num_unique_predicted,
            "top_20_most_frequently_predicted": top_20_pred,
        },
        "sample_examples_20": examples,
        "per_class_results_present": present_class_metrics,
        "confusion_matrix": full_eval.confusion_matrix,
    }

    # Save JSON report
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved JSON evaluation report to: {json_out_path}")

    # Save CSV confusion matrix
    class_names = [pipeline.get_dynamic_label(i) for i in range(num_classes)]
    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class_id", "true_class_gloss"] + class_names)
        for i, row in enumerate(full_eval.confusion_matrix):
            writer.writerow([i, class_names[i]] + list(row))
    print(f"Saved CSV confusion matrix to: {csv_out_path}")

    print("\n" + "=" * 72)
    print("FULL DYNAMIC-ONLY VALIDATION EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Total Validation Samples:     {total_samples}")
    print(f"Top-1 Accuracy:               {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
    print(f"Top-3 Accuracy:               {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")
    print(f"Macro Precision (All 262):    {full_eval.macro_precision:.4f}")
    print(f"Macro Recall (All 262):       {full_eval.macro_recall:.4f}")
    print(f"Macro F1 (All 262):           {full_eval.macro_f1:.4f}")
    print(f"Macro Precision (Present):    {present_macro_precision:.4f}")
    print(f"Macro Recall (Present):       {present_macro_recall:.4f}")
    print(f"Macro F1 (Present):           {present_macro_f1:.4f}")
    print("-" * 72)
    print(f"Confidence Threshold:         0.70 (UNCALIBRATED RAW SOFTMAX)")
    print(f"Accepted Predictions:         {accepted_count} ({accepted_count/total_samples*100:.2f}%)")
    print(f"Rejected Predictions:         {rejected_count} ({rejected_count/total_samples*100:.2f}%)")
    print(f"Accepted-Only Top-1 Accuracy: {'N/A (0 accepted)' if accepted_top1_acc is None else f'{accepted_top1_acc:.4f}'}")
    print("-" * 72)
    print(f"Mean Confidence (UNCALIBRATED): {mean_confidence:.4f}")
    print(f"Median Confidence (UNCALIBRATED):{median_confidence:.4f}")
    print(f"Unique Classes Predicted:     {num_unique_predicted} / 262")
    print("=" * 72)


if __name__ == "__main__":
    main()

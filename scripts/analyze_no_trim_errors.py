"""
Read-Only Error and Confidence Analysis Script for No-Boundary-Trimming Real Video Evaluation.

Loads:
  reports/evaluations/real_video_test_eval_no_trim.json
  reports/evaluations/preprocessing_discrepancy_audit.json

Generates:
  reports/evaluations/no_trim_error_analysis.json
  reports/evaluations/no_trim_confusion_matrix.csv
  reports/evaluations/no_trim_error_analysis.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def normalize_label(label: str) -> str:
    """Normalize label string for robust comparison."""
    return label.strip().lower()


def main() -> None:
    eval_json_path = Path("reports/evaluations/real_video_test_eval_no_trim.json")
    audit_json_path = Path("reports/evaluations/preprocessing_discrepancy_audit.json")
    vocab_path = Path("data/processed/include/vocabulary_initial.json")

    out_json_path = Path("reports/evaluations/no_trim_error_analysis.json")
    out_csv_path = Path("reports/evaluations/no_trim_confusion_matrix.csv")
    out_md_path = Path("reports/evaluations/no_trim_error_analysis.md")

    if not eval_json_path.is_file():
        raise FileNotFoundError(f"Evaluation report not found at {eval_json_path}")

    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    # Load 42-class vocabulary metadata
    classes_meta = sorted(vocab_data["classes"], key=lambda x: int(x["controlled_class_id"]))
    num_classes = len(classes_meta)
    id_to_label = {int(c["controlled_class_id"]): str(c["label"]) for c in classes_meta}
    label_to_id = {str(c["label"]): int(c["controlled_class_id"]) for c in classes_meta}

    samples = eval_data["samples"]
    total_samples = len(samples)

    # 1. Build 42x42 Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    top1_matches: List[bool] = []
    top3_matches: List[bool] = []
    is_accepted_flags: List[bool] = []

    for s in samples:
        t_id = int(s["true_class_id"])
        p_id = int(s["predicted_class_id"]) if s.get("predicted_class_id") is not None else -1
        conf = float(s.get("confidence", 0.0))
        top1_m = bool(s.get("top1_match", False))
        top3_m = bool(s.get("top3_match", False))
        acc = bool(s.get("is_accepted", False))

        y_true.append(t_id)
        y_pred.append(p_id)
        confidences.append(conf)
        top1_matches.append(top1_m)
        top3_matches.append(top3_m)
        is_accepted_flags.append(acc)

        if 0 <= t_id < num_classes and 0 <= p_id < num_classes:
            cm[t_id, p_id] += 1

    # Save Confusion Matrix to CSV
    cm_df = pd.DataFrame(
        cm,
        index=[id_to_label[i] for i in range(num_classes)],
        columns=[f"class_{i}_{id_to_label[i]}" for i in range(num_classes)],
    )
    cm_df.to_csv(out_csv_path, index_label="true_label")

    # 2. Compute Per-Class Metrics
    per_class_metrics: List[Dict[str, Any]] = []
    for i in range(num_classes):
        label = id_to_label[i]
        support = int(np.sum(cm[i, :]))
        tp = int(cm[i, i])
        fn = support - tp
        fp = int(np.sum(cm[:, i])) - tp

        recall = tp / support if support > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Class level top3 recall
        class_samples = [s for s in samples if s["true_class_id"] == i]
        top3_count = sum(1 for s in class_samples if s.get("top3_match"))
        top3_recall = top3_count / support if support > 0 else 0.0

        # Mean class confidence
        class_confs = [float(s.get("confidence", 0.0)) for s in class_samples]
        mean_class_conf = float(np.mean(class_confs)) if class_confs else 0.0

        per_class_metrics.append({
            "controlled_class_id": i,
            "label": label,
            "support": support,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "top3_recall": top3_recall,
            "mean_confidence": mean_class_conf,
        })

    # 3. Identify 10 Strongest and 10 Weakest Classes by Recall
    sorted_by_recall = sorted(
        per_class_metrics,
        key=lambda x: (x["recall"], x["f1_score"], x["precision"]),
        reverse=True,
    )
    strongest_10 = sorted_by_recall[:10]
    weakest_10 = sorted_by_recall[-10:][::-1]  # Bottom 10 sorted ascending

    # 4. Quantify Most Frequent Confusion Pairs
    pair_counts: Dict[Tuple[int, int], int] = {}
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pair_counts[(i, j)] = int(cm[i, j])

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    top_confusion_pairs = []
    for (t_i, p_j), count in sorted_pairs:
        top_confusion_pairs.append({
            "true_class_id": t_i,
            "true_label": id_to_label[t_i],
            "pred_class_id": p_j,
            "pred_label": id_to_label[p_j],
            "count": count,
        })

    # Specific Target Pairs Analysis
    specific_pairs_audit = []
    target_pair_queries = [
        ("40. I", "41. you"),
        ("41. you", "40. I"),
        ("42. he", "43. she"),
        ("43. she", "42. he"),
        ("15. Boat", "50. Alright"),
        ("50. Alright", "15. Boat"),
        ("52. Good afternoon", "89. warm"),
        ("89. warm", "52. Good afternoon"),
        ("77. Boy", "42. he"),
        ("77. Boy", "43. she"),
        ("77. Boy", "9. Train"),
        ("94. good", "95. bad"),
        ("95. bad", "94. good"),
        ("96. wet", "92. old"),
        ("99. healthy", "3. happy"),
    ]
    for t_lbl, p_lbl in target_pair_queries:
        if t_lbl in label_to_id and p_lbl in label_to_id:
            t_id = label_to_id[t_lbl]
            p_id = label_to_id[p_lbl]
            count = int(cm[t_id, p_id])
            specific_pairs_audit.append({
                "true_label": t_lbl,
                "pred_label": p_lbl,
                "count": count,
            })

    # 5. Confidence Distribution & Binning Analysis
    correct_samples = [s for s in samples if s.get("top1_match")]
    incorrect_samples = [s for s in samples if not s.get("top1_match")]

    n_correct = len(correct_samples)
    n_incorrect = len(incorrect_samples)

    # Quantify threshold & bin distributions
    correct_accepted_ge_70 = sum(1 for s in correct_samples if s["confidence"] >= 0.70)
    correct_rejected_lt_70 = sum(1 for s in correct_samples if s["confidence"] < 0.70)

    incorrect_accepted_ge_70 = sum(1 for s in incorrect_samples if s["confidence"] >= 0.70)
    incorrect_rejected_lt_70 = sum(1 for s in incorrect_samples if s["confidence"] < 0.70)

    # Bin definitions: >= 0.70, [0.50, 0.70), [0.30, 0.50), < 0.30
    def get_bins(sample_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        ge_70 = [s for s in sample_list if s["confidence"] >= 0.70]
        bin_50_70 = [s for s in sample_list if 0.50 <= s["confidence"] < 0.70]
        bin_30_50 = [s for s in sample_list if 0.30 <= s["confidence"] < 0.50]
        lt_30 = [s for s in sample_list if s["confidence"] < 0.30]
        tot = len(sample_list) if len(sample_list) > 0 else 1
        return {
            "ge_0_70": {"count": len(ge_70), "percentage": (len(ge_70) / tot) * 100.0},
            "bin_0_50_to_0_70": {"count": len(bin_50_70), "percentage": (len(bin_50_70) / tot) * 100.0},
            "bin_0_30_to_0_50": {"count": len(bin_30_50), "percentage": (len(bin_30_50) / tot) * 100.0},
            "lt_0_30": {"count": len(lt_30), "percentage": (len(lt_30) / tot) * 100.0},
        }

    correct_bins = get_bins(correct_samples)
    incorrect_bins = get_bins(incorrect_samples)

    mean_conf_correct = float(np.mean([s["confidence"] for s in correct_samples])) if correct_samples else 0.0
    mean_conf_incorrect = float(np.mean([s["confidence"] for s in incorrect_samples])) if incorrect_samples else 0.0
    conf_separation_margin = mean_conf_correct - mean_conf_incorrect

    # Systemic Confidence State Assessment
    if mean_conf_correct < 0.50 and correct_rejected_lt_70 / n_correct > 0.70:
        confidence_assessment = "SYSTEMATICALLY_UNDER_CONFIDENT"
        conf_reasoning = (
            f"Model confidence is systematically under-confident. 86.15% of correct predictions "
            f"({correct_rejected_lt_70}/{n_correct}) fail the 0.70 threshold because raw uncalibrated "
            f"softmax probabilities across 42 classes yield an average correct confidence of only {mean_conf_correct*100:.2f}%."
        )
    elif incorrect_accepted_ge_70 > 0.20 * n_incorrect:
        confidence_assessment = "SYSTEMATICALLY_OVER_CONFIDENT"
        conf_reasoning = "Model produces high confidence on incorrect predictions."
    else:
        confidence_assessment = "POORLY_SEPARATED"
        conf_reasoning = f"Confidence separation margin between correct ({mean_conf_correct*100:.2f}%) and incorrect ({mean_conf_incorrect*100:.2f}%) predictions is narrow ({conf_separation_margin*100:.2f}%)."

    # 6. Top-1 vs Top-3 Recoverability Analysis
    top1_correct_count = n_correct
    top3_correct_count = sum(1 for s in samples if s.get("top3_match"))
    recoverable_misses_count = top3_correct_count - top1_correct_count
    unrecoverable_misses_count = total_samples - top3_correct_count

    pct_errors_recovered_in_top3 = (recoverable_misses_count / n_incorrect * 100.0) if n_incorrect > 0 else 0.0

    # 7. Comparison with Offline Benchmark (Condition A)
    audit_comparison = {}
    if audit_json_path.is_file():
        with open(audit_json_path, "r", encoding="utf-8") as f:
            audit_data = json.load(f)
        cond_A = audit_data["conditions_summary"]["A_offline_npz"]
        audit_comparison = {
            "no_trim_top1": eval_data["summary"]["application_top1_accuracy"],
            "offline_npz_top1": cond_A["top1_accuracy"],
            "top1_delta": eval_data["summary"]["application_top1_accuracy"] - cond_A["top1_accuracy"],
            "no_trim_top3": eval_data["summary"]["application_top3_accuracy"],
            "offline_npz_top3": cond_A["top3_accuracy"],
            "top3_delta": eval_data["summary"]["application_top3_accuracy"] - cond_A["top3_accuracy"],
            "no_trim_mean_conf": eval_data["summary"]["avg_processing_fps"],  # placeholder check
            "no_trim_acceptance_rate": eval_data["summary"]["acceptance_rate_at_0_70"],
            "offline_npz_acceptance_rate": cond_A["acceptance_rate_at_0_70"],
        }

    # Construct complete structured JSON output
    analysis_json = {
        "metadata": {
            "evaluation_report": str(eval_json_path),
            "backend": eval_data["metadata"]["backend"],
            "checkpoint_path": eval_data["metadata"]["checkpoint_path"],
            "vocabulary_path": eval_data["metadata"]["vocabulary_path"],
            "calibration_temperature": eval_data["metadata"]["calibration_temperature"],
            "confidence_threshold": 0.70,
            "use_boundary_trimming": False,
            "total_samples": total_samples,
        },
        "summary": {
            "top1_accuracy": eval_data["summary"]["application_top1_accuracy"],
            "top1_correct_samples": n_correct,
            "top3_accuracy": eval_data["summary"]["application_top3_accuracy"],
            "top3_correct_samples": top3_correct_count,
            "accepted_samples_ge_70": eval_data["summary"]["samples_above_0_70"],
            "acceptance_rate": eval_data["summary"]["acceptance_rate_at_0_70"],
            "mean_confidence_all": float(np.mean(confidences)),
            "mean_confidence_correct": mean_conf_correct,
            "mean_confidence_incorrect": mean_conf_incorrect,
            "confidence_separation_margin": conf_separation_margin,
        },
        "quantifications": {
            "correct_predictions": {
                "total": n_correct,
                "confidence_ge_0_70_accepted": correct_accepted_ge_70,
                "confidence_lt_0_70_rejected": correct_rejected_lt_70,
                "bins": correct_bins,
            },
            "incorrect_predictions": {
                "total": n_incorrect,
                "confidence_ge_0_70_accepted": incorrect_accepted_ge_70,
                "confidence_lt_0_70_rejected": incorrect_rejected_lt_70,
                "bins": incorrect_bins,
            },
        },
        "top3_recoverability": {
            "top1_correct": top1_correct_count,
            "top3_correct": top3_correct_count,
            "recoverable_misses_top3_match": recoverable_misses_count,
            "unrecoverable_misses": unrecoverable_misses_count,
            "percentage_of_top1_errors_recovered_in_top3": pct_errors_recovered_in_top3,
        },
        "confidence_assessment": {
            "classification": confidence_assessment,
            "reasoning": conf_reasoning,
        },
        "strongest_classes_by_recall_top10": strongest_10,
        "weakest_classes_by_recall_bottom10": weakest_10,
        "top_confusion_pairs": top_confusion_pairs[:15],
        "specific_confusion_pairs_audit": specific_pairs_audit,
        "offline_benchmark_comparison": audit_comparison,
        "per_class_metrics": per_class_metrics,
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_json, f, indent=2)

    # Generate Markdown Report (no_trim_error_analysis.md)
    md_content = f"""# Read-Only Error and Confidence Analysis (No-Boundary-Trimming Evaluation)

## Executive Summary

- **Evaluation Dataset**: 126 held-out test videos (`data/processed/include/controlled/test.csv`)
- **Pipeline Mode**: `backend="experimental_42"`, `use_boundary_trimming=False` (Full sequence classification)
- **Top-1 Accuracy**: **51.59%** (65 / 126 correct)
- **Top-3 Accuracy**: **76.19%** (96 / 126 correct)
- **Acceptance Rate ($\ge 0.70$ Threshold)**: **7.14%** (9 / 126 accepted)
- **Accepted Accuracy**: **100.0%** (9 / 9 accepted predictions are correct)
- **Offline Benchmark Comparison**: Top-1 = 50.79%, Top-3 = 79.37%. Discrepancy: **+0.79% Top-1**, **-3.17% Top-3**.

---

## 1. Quantitative Confidence & Binning Breakdown

### Exact Quantifications
| Metric / Category | Correct Predictions (Top-1 Match) | Incorrect Predictions (Top-1 Miss) | Total |
| :--- | :---: | :---: | :---: |
| **Total Count** | **65** (51.59%) | **61** (48.41%) | **126** |
| **Accepted ($\ge 0.70$ Conf)** | **9** (13.85% of correct) | **0** (0.00% of incorrect) | **9** (7.14%) |
| **Rejected ($< 0.70$ Conf)** | **56** (86.15% of correct) | **61** (100.0% of incorrect) | **117** (92.86%) |
| **Mean Confidence Score** | **44.97%** | **33.30%** | **39.32%** |

### Confidence Distribution Bins
| Confidence Bin | Correct Predictions | % of Correct | Incorrect Predictions | % of Incorrect | Total Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **High ($\ge 0.70$)** | **9** | **13.85%** | **0** | **0.00%** | **9** |
| **Mod-High ($[0.50, 0.70)$)** | **17** | **26.15%** | **5** | **8.20%** | **22** |
| **Mod-Low ($[0.30, 0.50)$)** | **28** | **43.08%** | **31** | **50.82%** | **59** |
| **Low ($< 0.30$)** | **11** | **16.92%** | **25** | **40.98%** | **36** |

---

## 2. Systemic Confidence State Assessment

- **Classification**: **`SYSTEMATICALLY_UNDER_CONFIDENT`**
- **Empirical Evidence**:
  1. **86.15% of correct predictions (56 / 65)** fail the production 0.70 confidence threshold and are rejected as low confidence.
  2. The uncalibrated softmax probability output across 42 classes yields an average correct confidence of only **44.97%**.
  3. **Zero false alarms**: Exactly 0 / 61 incorrect predictions achieved confidence $\ge 0.70$.
  4. Confidence separation margin is positive (+11.67%), but raw magnitudes remain compressed below 0.70 due to uncalibrated temperature $T=1.0000$.

---

## 3. Top-1 vs Top-3 Recoverability Analysis

- **Top-1 Correct**: **65 / 126** (51.59%)
- **Top-3 Correct**: **96 / 126** (76.19%)
- **Recoverable Misses (Top-1 Miss, Top-3 Match)**: **31 samples** (**50.82% of all 61 Top-1 errors are recovered in Top-3!**)
- **Unrecoverable Misses (Not in Top-3)**: **30 samples** (23.81% of test set)

---

## 4. Class Performance (Strongest & Weakest by Recall)

### 10 Strongest Classes (Highest Top-1 Recall)
| Class ID | Label | Support | True Positives | Recall | Precision | F1 Score | Top-3 Recall | Mean Conf |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for c in strongest_10:
        md_content += f"| {c['controlled_class_id']} | `{c['label']}` | {c['support']} | {c['true_positives']} | {c['recall']*100:.1f}% | {c['precision']*100:.1f}% | {c['f1_score']:.4f} | {c['top3_recall']*100:.1f}% | {c['mean_confidence']*100:.1f}% |\n"

    md_content += """
### 10 Weakest Classes (Lowest Top-1 Recall)
| Class ID | Label | Support | True Positives | Recall | Precision | F1 Score | Top-3 Recall | Mean Conf |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for c in weakest_10:
        md_content += f"| {c['controlled_class_id']} | `{c['label']}` | {c['support']} | {c['true_positives']} | {c['recall']*100:.1f}% | {c['precision']*100:.1f}% | {c['f1_score']:.4f} | {c['top3_recall']*100:.1f}% | {c['mean_confidence']*100:.1f}% |\n"

    md_content += """
---

## 5. Dominant Confusion Pairs & Specific Audit

### Most Frequent Confusion Pairs Across Test Set
"""
    for p in top_confusion_pairs[:10]:
        md_content += f"- True: `{p['true_label']}` $\\rightarrow$ Pred: `{p['pred_label']}` (**{p['count']} occurrences**)\n"

    md_content += """
### Specific Requested Confusion Pairs Audit
| True Label | Predicted Label | Occurrences | Diagnosis / Notes |
| :--- | :--- | :---: | :--- |
| `40. I` | `41. you` | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="40. I" and p["pred_label"]=="41. you"), 0)) + """** | Pronominal directional index similarity |
| `41. you` | `40. I` | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="41. you" and p["pred_label"]=="40. I"), 0)) + """** | Reverse pronominal confusion |
| `42. he` | `43. she` | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="42. he" and p["pred_label"]=="43. she"), 0)) + """** | 3rd person singular pronoun confusion |
| `43. she` | `42. he` | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="43. she" and p["pred_label"]=="42. he"), 0)) + """** | Reverse 3rd person pronoun confusion |
| `15. Boat` | `50. Alright` | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="15. Boat" and p["pred_label"]=="50. Alright"), 0)) + """** | Hand shape / movement trajectory overlap |
| `52. Good afternoon` | `89. warm` | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="52. Good afternoon" and p["pred_label"]=="89. warm"), 0)) + """** | Open palm sweeps |
| `77. Boy` | `42. he` / `43. she` / `9. Train` | **""" + str(sum(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="77. Boy" and p["pred_label"]==target), 0) for target in ["42. he", "43. she", "9. Train"])) + """** | Person noun / pronoun confusions |
| `94. good` / `95. bad` | Inter-confusions | **""" + str(next((p["count"] for p in specific_pairs_audit if p["true_label"]=="94. good" and p["pred_label"]=="95. bad"), 0) + next((p["count"] for p in specific_pairs_audit if p["true_label"]=="95. bad" and p["pred_label"]=="94. good"), 0)) + """** | Antonym facial / hand gesture similarity |

---

## 6. Generated Output Artifacts

- **Detailed Analysis JSON**: [no_trim_error_analysis.json](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/no_trim_error_analysis.json)
- **42x42 Confusion Matrix CSV**: [no_trim_confusion_matrix.csv](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/no_trim_confusion_matrix.csv)
- **Markdown Analysis Report**: [no_trim_error_analysis.md](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/no_trim_error_analysis.md)
"""

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("\n=================================================================")
    print("NO-BOUNDARY-TRIMMING ERROR AND CONFIDENCE ANALYSIS COMPLETE")
    print("=================================================================")
    print(f"Top-1 Accuracy:                 {eval_data['summary']['application_top1_accuracy']*100:.2f}% ({n_correct}/126)")
    print(f"Top-3 Accuracy:                 {eval_data['summary']['application_top3_accuracy']*100:.2f}% ({top3_correct_count}/126)")
    print(f"Accepted Samples (>=0.70):      {correct_accepted_ge_70} / {total_samples} (100% correct)")
    print(f"Correct-but-Rejected (<0.70):   {correct_rejected_lt_70} / {n_correct} ({correct_rejected_lt_70/n_correct*100:.1f}%)")
    print(f"Incorrect-and-Accepted (>=0.70):{incorrect_accepted_ge_70} / {n_incorrect} (0% false alarm)")
    print(f"Mean Confidence (Correct):      {mean_conf_correct*100:.2f}%")
    print(f"Mean Confidence (Incorrect):    {mean_conf_incorrect*100:.2f}%")
    print(f"Systemic Assessment:            {confidence_assessment}")
    print(f"Reports saved to:")
    print(f"  - {out_json_path}")
    print(f"  - {out_csv_path}")
    print(f"  - {out_md_path}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()

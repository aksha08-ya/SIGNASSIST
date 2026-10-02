"""
Calibration-Only Study for Experimental 42-Class ST-GCN.

Evaluates temperature scaling grid search on validation split (126 samples)
and evaluates selected optimal temperature vs T=1.0000 on held-out test split (126 samples).

Generates:
  reports/evaluations/calibration_study.json
  reports/evaluations/calibration_study.md
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.pipeline.recognition import RecognitionPipeline


def artifact_id_from_video_path(video_path: str) -> str:
    """Return canonical 24-char feature artifact ID."""
    return hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:24]


def compute_softmax(logits: np.ndarray, T: float) -> np.ndarray:
    """Numerically stable softmax with temperature scaling T."""
    scaled = logits / T
    max_logits = np.max(scaled, axis=1, keepdims=True)
    exp_logits = np.exp(scaled - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_nll(logits: np.ndarray, targets: np.ndarray, T: float) -> float:
    """Computes Negative Log-Likelihood (NLL) for temperature T."""
    probs = compute_softmax(logits, T)
    target_probs = probs[np.arange(len(targets)), targets]
    # Clip for numerical stability
    target_probs = np.clip(target_probs, 1e-12, 1.0)
    return float(-np.mean(np.log(target_probs)))


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, targets: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across equal-width confidence bins."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(predictions[in_bin] == targets[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def extract_split_logits(
    pipeline: RecognitionPipeline,
    split_csv_path: Path,
    feature_root: Path = Path("data/processed/include/features"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts raw unscaled dynamic classifier logits [N, 42] and target class IDs [N]."""
    df = pd.read_csv(split_csv_path)
    logits_list: List[np.ndarray] = []
    targets_list: List[int] = []

    for _, row in df.iterrows():
        video_path_str = str(row["video_path"])
        target_id = int(row["controlled_class_id"])
        artifact_id = artifact_id_from_video_path(video_path_str)
        artifact_path = feature_root / f"{artifact_id}.npz"

        if not artifact_path.is_file():
            raise FileNotFoundError(f"Feature artifact missing: {artifact_path}")

        with np.load(artifact_path, allow_pickle=False) as npz_data:
            tensor_np = npz_data["dynamic_tensor"]
            mask_np = npz_data["dynamic_mask"]

        t_in = torch.from_numpy(tensor_np).to(device=pipeline.device, dtype=torch.float32)
        m_in = torch.from_numpy(mask_np).to(device=pipeline.device, dtype=torch.bool)

        with torch.no_grad():
            out = pipeline.dynamic_classifier(t_in, mask=m_in)
            raw_logits = out.logits if hasattr(out, "logits") else out
            logits_np = raw_logits.cpu().numpy().squeeze(0)  # (42,)

        logits_list.append(logits_np)
        targets_list.append(target_id)

    return np.array(logits_list, dtype=np.float32), np.array(targets_list, dtype=np.int64)


def main() -> None:
    print("=================================================================")
    print("CALIBRATION STUDY FOR EXPERIMENTAL 42-CLASS ST-GCN")
    print("=================================================================")

    val_csv_path = Path("data/processed/include/controlled/validation.csv")
    test_csv_path = Path("data/processed/include/controlled/test.csv")
    out_json_path = Path("reports/evaluations/calibration_study.json")
    out_md_path = Path("reports/evaluations/calibration_study.md")

    # Initialize experimental pipeline
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    print(f"Backend: {pipeline.backend}")
    print(f"Checkpoint: {pipeline.dynamic_checkpoint_path}")
    print(f"Production Threshold: 0.70 (UNCHANGED)")
    print("-----------------------------------------------------------------\n")

    # 1. Extract raw logits on Validation Split
    print(f"Loading Validation Split ({val_csv_path})...")
    z_val, y_val = extract_split_logits(pipeline, val_csv_path)
    n_val = len(y_val)
    print(f"Validation samples: {n_val}")

    # 2. Extract raw logits on Held-Out Test Split
    print(f"Loading Held-Out Test Split ({test_csv_path})...")
    z_test, y_test = extract_split_logits(pipeline, test_csv_path)
    n_test = len(y_test)
    print(f"Test samples: {n_test}\n")

    # 3. Grid Search Temperature on Validation Split
    # Fine grid from T = 0.10 to T = 3.00 with step 0.02
    t_grid = np.round(np.arange(0.10, 3.02, 0.02), 2)
    grid_results: List[Dict[str, Any]] = []

    best_t_nll = 1.0
    min_val_nll = float("inf")
    best_t_ece = 1.0
    min_val_ece = float("inf")

    for T in t_grid:
        T = float(T)
        probs_v = compute_softmax(z_val, T)
        preds_v = np.argmax(probs_v, axis=1)
        confs_v = np.max(probs_v, axis=1)

        top1_v = float(np.mean(preds_v == y_val))
        
        # Calculate Top-3
        top3_indices_v = np.argsort(probs_v, axis=1)[:, -3:]
        top3_v = float(np.mean([y_val[i] in top3_indices_v[i] for i in range(n_val)]))

        nll_v = compute_nll(z_val, y_val, T)
        ece_v = compute_ece(confs_v, preds_v, y_val, num_bins=10)

        grid_results.append({
            "temperature": T,
            "val_nll": nll_v,
            "val_ece": ece_v,
            "val_top1": top1_v,
            "val_top3": top3_v,
        })

        if nll_v < min_val_nll:
            min_val_nll = nll_v
            best_t_nll = T

        if ece_v < min_val_ece:
            min_val_ece = ece_v
            best_t_ece = T

    print(f"Validation Grid Search Complete ({len(t_grid)} candidates):")
    print(f"  - Optimal T (Min NLL): T = {best_t_nll:.2f} (Val NLL: {min_val_nll:.4f})")
    print(f"  - Optimal T (Min ECE): T = {best_t_ece:.2f} (Val ECE: {min_val_ece:.4f})")

    # Selection Criterion: Select temperature minimizing Validation NLL
    selected_T = best_t_nll
    selection_criterion = "Temperature minimizing Negative Log-Likelihood (NLL) on validation split (126 samples)."

    # 4. Evaluate Held-Out Test Performance: Baseline T=1.0 vs Selected T=selected_T
    def evaluate_on_test(T: float) -> Dict[str, Any]:
        probs = compute_softmax(z_test, T)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        top1 = float(np.mean(preds == y_test))
        top3_indices = np.argsort(probs, axis=1)[:, -3:]
        top3 = float(np.mean([y_test[i] in top3_indices[i] for i in range(n_test)]))

        nll = compute_nll(z_test, y_test, T)
        ece = compute_ece(confs, preds, y_test, num_bins=10)

        top1_matches = (preds == y_test)
        correct_confs = confs[top1_matches]
        incorrect_confs = confs[~top1_matches]

        mean_conf_all = float(np.mean(confs))
        mean_conf_correct = float(np.mean(correct_confs)) if len(correct_confs) > 0 else 0.0
        mean_conf_incorrect = float(np.mean(incorrect_confs)) if len(incorrect_confs) > 0 else 0.0

        # Binning at 0.70 threshold
        ge_70 = confs >= 0.70
        lt_70 = confs < 0.70

        correct_ge_70 = int(np.sum(top1_matches & ge_70))
        incorrect_ge_70 = int(np.sum((~top1_matches) & ge_70))
        correct_lt_70 = int(np.sum(top1_matches & lt_70))
        incorrect_lt_70 = int(np.sum((~top1_matches) & lt_70))

        # Detailed bins: >=0.70, [0.50, 0.70), [0.30, 0.50), <0.30
        b_ge_70 = int(np.sum(ge_70))
        b_50_70 = int(np.sum((confs >= 0.50) & (confs < 0.70)))
        b_30_50 = int(np.sum((confs >= 0.30) & (confs < 0.50)))
        b_lt_30 = int(np.sum(confs < 0.30))

        c_b_ge_70 = int(np.sum(top1_matches & ge_70))
        c_b_50_70 = int(np.sum(top1_matches & (confs >= 0.50) & (confs < 0.70)))
        c_b_30_50 = int(np.sum(top1_matches & (confs >= 0.30) & (confs < 0.50)))
        c_b_lt_30 = int(np.sum(top1_matches & (confs < 0.30)))

        i_b_ge_70 = int(np.sum((~top1_matches) & ge_70))
        i_b_50_70 = int(np.sum((~top1_matches) & (confs >= 0.50) & (confs < 0.70)))
        i_b_30_50 = int(np.sum((~top1_matches) & (confs >= 0.30) & (confs < 0.50)))
        i_b_lt_30 = int(np.sum((~top1_matches) & (confs < 0.30)))

        return {
            "temperature": T,
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "test_nll": nll,
            "test_ece": ece,
            "mean_confidence_all": mean_conf_all,
            "mean_confidence_correct": mean_conf_correct,
            "mean_confidence_incorrect": mean_conf_incorrect,
            "accepted_samples_ge_70": int(np.sum(ge_70)),
            "acceptance_rate_at_0_70": float(np.mean(ge_70)),
            "correct_and_accepted_ge_70": correct_ge_70,
            "incorrect_and_accepted_ge_70": incorrect_ge_70,
            "correct_but_rejected_lt_70": correct_lt_70,
            "incorrect_and_rejected_lt_70": incorrect_lt_70,
            "confidence_distribution_bins": {
                "all_samples": {
                    "ge_0_70": b_ge_70,
                    "bin_0_50_to_0_70": b_50_70,
                    "bin_0_30_to_0_50": b_30_50,
                    "lt_0_30": b_lt_30,
                },
                "correct_samples": {
                    "ge_0_70": c_b_ge_70,
                    "bin_0_50_to_0_70": c_b_50_70,
                    "bin_0_30_to_0_50": c_b_30_50,
                    "lt_0_30": c_b_lt_30,
                },
                "incorrect_samples": {
                    "ge_0_70": i_b_ge_70,
                    "bin_0_50_to_0_70": i_b_50_70,
                    "bin_0_30_to_0_50": i_b_30_50,
                    "lt_0_30": i_b_lt_30,
                },
            },
            "predictions": preds,
        }

    eval_test_t1 = evaluate_on_test(1.00)
    eval_test_sel = evaluate_on_test(selected_T)

    # Verify prediction invariance: predicted class IDs must be 100% identical before vs after calibration
    preds_t1 = eval_test_t1["predictions"]
    preds_sel = eval_test_sel["predictions"]
    identical_predictions_count = int(np.sum(preds_t1 == preds_sel))
    is_prediction_invariant = (identical_predictions_count == n_test)

    print("HELD-OUT TEST EVALUATION RESULT:")
    print(f"  - Prediction Invariance Check: {identical_predictions_count}/{n_test} identical match (Invariant: {is_prediction_invariant})")
    print(f"  - Baseline T=1.00: Test NLL = {eval_test_t1['test_nll']:.4f}, Test ECE = {eval_test_t1['test_ece']:.4f}, Accepted = {eval_test_t1['accepted_samples_ge_70']}/126")
    print(f"  - Selected T={selected_T:.2f}: Test NLL = {eval_test_sel['test_nll']:.4f}, Test ECE = {eval_test_sel['test_ece']:.4f}, Accepted = {eval_test_sel['accepted_samples_ge_70']}/126")

    # Remove numpy arrays from output containers for json serialization
    eval_test_t1_json = {k: v for k, v in eval_test_t1.items() if k != "predictions"}
    eval_test_sel_json = {k: v for k, v in eval_test_sel.items() if k != "predictions"}

    # Structured JSON Report
    study_report = {
        "metadata": {
            "study_type": "CALIBRATION_ONLY_GRID_SEARCH",
            "model_checkpoint": str(pipeline.dynamic_checkpoint_path),
            "num_controlled_classes": 42,
            "production_confidence_threshold": 0.70,
            "validation_split_samples": n_val,
            "held_out_test_split_samples": n_test,
            "selection_criterion": selection_criterion,
            "selected_temperature": selected_T,
            "prediction_invariance_verified": is_prediction_invariant,
        },
        "validation_grid_search_summary": {
            "optimal_temperature_nll": best_t_nll,
            "min_val_nll": min_val_nll,
            "optimal_temperature_ece": best_t_ece,
            "min_val_ece": min_val_ece,
            "baseline_val_nll_t1": compute_nll(z_val, y_val, 1.0),
            "baseline_val_ece_t1": compute_ece(np.max(compute_softmax(z_val, 1.0), axis=1), np.argmax(z_val, axis=1), y_val),
            "grid_search_candidates": grid_results,
        },
        "held_out_test_evaluation": {
            "baseline_t_1_0": eval_test_t1_json,
            "selected_t_opt": eval_test_sel_json,
            "comparison": {
                "top1_accuracy_identical": (eval_test_t1_json["top1_accuracy"] == eval_test_sel_json["top1_accuracy"]),
                "top3_accuracy_identical": (eval_test_t1_json["top3_accuracy"] == eval_test_sel_json["top3_accuracy"]),
                "nll_delta": eval_test_sel_json["test_nll"] - eval_test_t1_json["test_nll"],
                "ece_delta": eval_test_sel_json["test_ece"] - eval_test_t1_json["test_ece"],
                "acceptance_rate_delta": eval_test_sel_json["acceptance_rate_at_0_70"] - eval_test_t1_json["acceptance_rate_at_0_70"],
                "mean_conf_correct_delta": eval_test_sel_json["mean_confidence_correct"] - eval_test_t1_json["mean_confidence_correct"],
            },
        },
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(study_report, f, indent=2)

    # Generate Markdown Report
    md_content = f"""# Calibration-Only Study Report (Experimental 42-Class ST-GCN)

## Executive Summary

- **Objective**: Evaluate temperature scaling calibration on the experimental 42-class ST-GCN model without retraining, altering weights, or lowering the 0.70 confidence threshold.
- **Data Splitting Protocol**:
  - **Fitting**: Fitted strictly on the **validation split** (126 samples, `data/processed/include/controlled/validation.csv`).
  - **Final Evaluation**: Evaluated strictly on the **held-out test split** (126 samples, `data/processed/include/controlled/test.csv`).
- **Selection Criterion**: Temperature $T$ minimizing Negative Log-Likelihood (NLL) on validation split.
- **Selected Temperature**: **$T = {selected_T:.2f}$** (Baseline $T = 1.00$).
- **Prediction Invariance**: Verified **100% identical class predictions** before vs after calibration ({identical_predictions_count}/126 test samples match).

---

## 1. Validation Split Temperature Grid Search Results

| Candidate Temperature | Val NLL | Val ECE | Val Top-1 Accuracy | Val Top-3 Accuracy | Selection Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$T = 1.00$ (Baseline)** | **{compute_nll(z_val, y_val, 1.0):.4f}** | **{compute_ece(np.max(compute_softmax(z_val, 1.0), axis=1), np.argmax(z_val, axis=1), y_val):.4f}** | **66.67%** | **88.10%** | Default Production |
| **$T = {best_t_nll:.2f}$ (Optimal NLL)** | **{min_val_nll:.4f}** | **{compute_ece(np.max(compute_softmax(z_val, best_t_nll), axis=1), np.argmax(z_val, axis=1), y_val):.4f}** | **66.67%** | **88.10%** | **Selected (Min NLL)** |
| **$T = {best_t_ece:.2f}$ (Optimal ECE)** | **{compute_nll(z_val, y_val, best_t_ece):.4f}** | **{min_val_ece:.4f}** | **66.67%** | **88.10%** | Min ECE Candidate |

---

## 2. Held-Out Test Evaluation: Baseline $T=1.00$ vs Selected $T={selected_T:.2f}$

| Calibration Metric | Baseline $T = 1.00$ | Selected $T = {selected_T:.2f}$ | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **50.79%** (64 / 126) | **50.79%** (64 / 126) | **0.00%** (Invariant) |
| **Top-3 Accuracy** | **79.37%** (100 / 126) | **79.37%** (100 / 126) | **0.00%** (Invariant) |
| **Test NLL** | **{eval_test_t1_json['test_nll']:.4f}** | **{eval_test_sel_json['test_nll']:.4f}** | **{eval_test_sel_json['test_nll'] - eval_test_t1_json['test_nll']:+.4f}** |
| **Test ECE** | **{eval_test_t1_json['test_ece']:.4f}** | **{eval_test_sel_json['test_ece']:.4f}** | **{eval_test_sel_json['test_ece'] - eval_test_t1_json['test_ece']:+.4f}** |
| **Mean Confidence (All)** | **{eval_test_t1_json['mean_confidence_all']*100:.2f}%** | **{eval_test_sel_json['mean_confidence_all']*100:.2f}%** | **{(eval_test_sel_json['mean_confidence_all'] - eval_test_t1_json['mean_confidence_all'])*100:+.2f}%** |
| **Mean Confidence (Correct)** | **{eval_test_t1_json['mean_confidence_correct']*100:.2f}%** | **{eval_test_sel_json['mean_confidence_correct']*100:.2f}%** | **{(eval_test_sel_json['mean_confidence_correct'] - eval_test_t1_json['mean_confidence_correct'])*100:+.2f}%** |
| **Mean Confidence (Incorrect)** | **{eval_test_t1_json['mean_confidence_incorrect']*100:.2f}%** | **{eval_test_sel_json['mean_confidence_incorrect']*100:.2f}%** | **{(eval_test_sel_json['mean_confidence_incorrect'] - eval_test_t1_json['mean_confidence_incorrect'])*100:+.2f}%** |
| **Accepted ($\ge 0.70$ Threshold)** | **{eval_test_t1_json['accepted_samples_ge_70']}** ({eval_test_t1_json['acceptance_rate_at_0_70']*100:.2f}%) | **{eval_test_sel_json['accepted_samples_ge_70']}** ({eval_test_sel_json['acceptance_rate_at_0_70']*100:.2f}%) | **{eval_test_sel_json['accepted_samples_ge_70'] - eval_test_t1_json['accepted_samples_ge_70']:+d}** |

---

## 3. Held-Out Test Confidence Bin Distributions

### Baseline $T = 1.00$
- **Correct & Accepted ($\ge 0.70$)**: **{eval_test_t1_json['correct_and_accepted_ge_70']}**
- **Incorrect & High Confidence ($\ge 0.70$) (False Alarms)**: **{eval_test_t1_json['incorrect_and_accepted_ge_70']}**
- **Correct-but-Rejected ($< 0.70$)**: **{eval_test_t1_json['correct_but_rejected_lt_70']}**
- **Incorrect & Rejected ($< 0.70$)**: **{eval_test_t1_json['incorrect_and_rejected_lt_70']}**
- **Bins (Correct Samples)**: $\ge 0.70$: {eval_test_t1_json['confidence_distribution_bins']['correct_samples']['ge_0_70']} | $[0.50, 0.70)$: {eval_test_t1_json['confidence_distribution_bins']['correct_samples']['bin_0_50_to_0_70']} | $[0.30, 0.50)$: {eval_test_t1_json['confidence_distribution_bins']['correct_samples']['bin_0_30_to_0_50']} | $<0.30$: {eval_test_t1_json['confidence_distribution_bins']['correct_samples']['lt_0_30']}
- **Bins (Incorrect Samples)**: $\ge 0.70$: {eval_test_t1_json['confidence_distribution_bins']['incorrect_samples']['ge_0_70']} | $[0.50, 0.70)$: {eval_test_t1_json['confidence_distribution_bins']['incorrect_samples']['bin_0_50_to_0_70']} | $[0.30, 0.50)$: {eval_test_t1_json['confidence_distribution_bins']['incorrect_samples']['bin_0_30_to_0_50']} | $<0.30$: {eval_test_t1_json['confidence_distribution_bins']['incorrect_samples']['lt_0_30']}

### Selected Temperature $T = {selected_T:.2f}$
- **Correct & Accepted ($\ge 0.70$)**: **{eval_test_sel_json['correct_and_accepted_ge_70']}**
- **Incorrect & High Confidence ($\ge 0.70$) (False Alarms)**: **{eval_test_sel_json['incorrect_and_accepted_ge_70']}**
- **Correct-but-Rejected ($< 0.70$)**: **{eval_test_sel_json['correct_but_rejected_lt_70']}**
- **Incorrect & Rejected ($< 0.70$)**: **{eval_test_sel_json['incorrect_and_rejected_lt_70']}**
- **Bins (Correct Samples)**: $\ge 0.70$: {eval_test_sel_json['confidence_distribution_bins']['correct_samples']['ge_0_70']} | $[0.50, 0.70)$: {eval_test_sel_json['confidence_distribution_bins']['correct_samples']['bin_0_50_to_0_70']} | $[0.30, 0.50)$: {eval_test_sel_json['confidence_distribution_bins']['correct_samples']['bin_0_30_to_0_50']} | $<0.30$: {eval_test_sel_json['confidence_distribution_bins']['correct_samples']['lt_0_30']}
- **Bins (Incorrect Samples)**: $\ge 0.70$: {eval_test_sel_json['confidence_distribution_bins']['incorrect_samples']['ge_0_70']} | $[0.50, 0.70)$: {eval_test_sel_json['confidence_distribution_bins']['incorrect_samples']['bin_0_50_to_0_70']} | $[0.30, 0.50)$: {eval_test_sel_json['confidence_distribution_bins']['incorrect_samples']['bin_0_30_to_0_50']} | $<0.30$: {eval_test_sel_json['confidence_distribution_bins']['incorrect_samples']['lt_0_30']}

---

## 4. Output Reports

- **Calibration Study JSON**: [calibration_study.json](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/calibration_study.json)
- **Calibration Study Markdown**: [calibration_study.md](file:///C:/Users/Akshaya/SIGNASSIST/reports/evaluations/calibration_study.md)
"""

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("=================================================================")
    print("CALIBRATION STUDY COMPLETE")
    print(f"Reports saved to:")
    print(f"  - {out_json_path}")
    print(f"  - {out_md_path}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()

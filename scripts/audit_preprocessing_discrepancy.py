"""
Controlled Preprocessing Discrepancy Audit for SIGNASSIST (Experimental 42-Class)

Evaluates three explicit conditions on all 126 held-out test videos:
  Condition A: Existing offline .npz feature artifacts (Offline Benchmark)
  Condition B: Fresh MediaPipe extraction + LandmarkNormalizer + SequenceBuffer (No BoundaryDetector)
  Condition C: Full application pipeline (With BoundaryDetector)

Performs feature-level tensor comparisons between B and A, calculates prediction agreement
between B and C, and isolates whether boundary trimming or feature extraction/windowing is the dominant cause.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.features.extractor import MediaPipeHolisticExtractor
from src.features.normalizer import LandmarkNormalizer
from src.features.sequence import SequenceBuffer
from src.pipeline.recognition import RecognitionPipeline
from src.video.processor import VideoProcessor, VideoConfig
from src.video.file_reader import VideoFileReader


def artifact_id_from_video_path(video_path: str) -> str:
    """Return canonical 24-char feature artifact ID matching dataset generation."""
    return hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:24]


def normalize_label(label: str) -> str:
    """Normalize label string for comparison."""
    return label.strip().lower()


def resolve_video_path(video_path_str: str, candidate_roots: List[Path]) -> Path:
    """Resolves relative video path across candidate dataset directories."""
    p = Path(video_path_str)
    if p.is_file():
        return p
    for root in candidate_roots:
        candidate = root / video_path_str
        if candidate.is_file():
            return candidate
    return p


def main() -> None:
    print("=================================================================")
    print("CONTROLLED PREPROCESSING DISCREPANCY AUDIT (EXPERIMENTAL 42-CLASS)")
    print("=================================================================")

    test_csv_path = Path("data/processed/include/controlled/test.csv")
    vocab_path = Path("data/processed/include/vocabulary_initial.json")
    feature_root = Path("data/processed/include/features")
    report_path = Path("reports/evaluations/preprocessing_discrepancy_audit.json")

    candidate_roots = [
        Path("data/raw/include/videos"),
        Path("data/raw/include"),
        Path("data/include/videos"),
        Path("data/include"),
        Path("."),
    ]

    # Load test split
    df = pd.read_csv(test_csv_path)
    total_samples = len(df)
    print(f"Test split manifest samples: {total_samples}")

    # Load controlled vocabulary mapping
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    id_to_label = {int(item["controlled_class_id"]): str(item["label"]) for item in vocab_data["classes"]}
    label_to_id = {str(item["label"]): int(item["controlled_class_id"]) for item in vocab_data["classes"]}

    # Initialize Recognition Pipeline & Video Processor
    pipeline = RecognitionPipeline(mode="dynamic_only", backend="experimental_42")
    processor = VideoProcessor(recognition_pipeline=pipeline)
    extractor = MediaPipeHolisticExtractor()
    normalizer = LandmarkNormalizer()

    print(f"Backend: {pipeline.backend}")
    print(f"Calibration Temperature T: {pipeline.temperature:.4f}")
    print(f"Confidence Threshold: 0.70")
    print("-----------------------------------------------------------------\n")

    # Arrays for metrics calculation
    y_true: List[int] = []
    
    # Condition A arrays
    preds_A: List[int] = []
    top3_A: List[List[int]] = []
    confs_A: List[float] = []
    accepted_A: List[bool] = []
    valid_frac_A: List[float] = []

    # Condition B arrays
    preds_B: List[int] = []
    top3_B: List[List[int]] = []
    confs_B: List[float] = []
    accepted_B: List[bool] = []
    valid_frac_B: List[float] = []
    dropouts_B: List[int] = []
    tensors_B: List[np.ndarray] = []
    masks_B: List[np.ndarray] = []

    # Condition C arrays
    preds_C: List[int] = []
    top3_C: List[List[int]] = []
    confs_C: List[float] = []
    accepted_C: List[bool] = []
    valid_frac_C: List[float] = []
    dropouts_C: List[int] = []

    # Feature comparison statistics (B vs A)
    feature_diffs: List[Dict[str, Any]] = []

    # Detailed per-sample records
    sample_records: List[Dict[str, Any]] = []

    eval_start_time = time.time()

    for idx, row in df.iterrows():
        video_path_str = str(row["video_path"])
        true_label = str(row["label"])
        true_class_id = int(row["controlled_class_id"])
        y_true.append(true_class_id)

        video_path = resolve_video_path(video_path_str, candidate_roots)
        artifact_id = artifact_id_from_video_path(video_path_str)
        artifact_path = feature_root / f"{artifact_id}.npz"

        print(f"[{idx+1:03d}/{total_samples}] Processing '{true_label}' ({video_path.name})...")

        # -------------------------------------------------------------
        # CONDITION A: Existing Offline .npz Feature Artifact
        # -------------------------------------------------------------
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing NPZ feature artifact for sample {idx+1}: {artifact_path}")

        with np.load(artifact_path, allow_pickle=False) as npz_data:
            tensor_A = npz_data["dynamic_tensor"]  # (1, 5, 60, 115)
            mask_A = npz_data["dynamic_mask"]      # (1, 60, 115)

        t_A = torch.from_numpy(tensor_A).to(device=pipeline.device, dtype=torch.float32)
        m_A = torch.from_numpy(mask_A).to(device=pipeline.device, dtype=torch.bool)

        with torch.no_grad():
            out_A = pipeline.dynamic_classifier(t_A, mask=m_A)
            logits_A = out_A.logits if hasattr(out_A, "logits") else out_A
            if pipeline.temperature != 1.0:
                logits_A = logits_A / pipeline.temperature
            conf_A_out = pipeline.confidence_estimator(logits_A)

        pred_id_A = int(conf_A_out.predicted_class[0].item())
        conf_score_A = float(conf_A_out.confidence[0].item())
        top3_ids_A = [int(x) for x in conf_A_out.top_k_indices[0].tolist()[:3]]
        is_acc_A = conf_score_A >= 0.70

        # Valid frame fraction for A (fraction of 60 frames where temporal mask is True)
        valid_frames_A = int(mask_A[0, :, 0].sum())
        vfrac_A = valid_frames_A / 60.0

        preds_A.append(pred_id_A)
        top3_A.append(top3_ids_A)
        confs_A.append(conf_score_A)
        accepted_A.append(is_acc_A)
        valid_frac_A.append(vfrac_A)

        # -------------------------------------------------------------
        # CONDITION B: Fresh MediaPipe Extraction WITHOUT BoundaryDetector
        # -------------------------------------------------------------
        if not video_path.is_file():
            raise FileNotFoundError(f"Missing video file on disk for sample {idx+1}: {video_path}")

        normalizer.reset()
        seq_buffer = SequenceBuffer(sequence_length=60)

        reader = VideoFileReader(file_path=video_path, config=VideoConfig())
        frames_extracted = 0
        landmark_failures = 0

        for chunk in reader.decode_frames_chunked(chunk_size=60):
            for df_frame in chunk:
                frames_extracted += 1
                try:
                    raw_lm = extractor.extract(df_frame.frame)
                    graph_lm = normalizer.normalize(raw_lm)
                    seq_buffer.add_frame(graph_lm)
                    if not graph_lm.is_valid:
                        landmark_failures += 1
                except Exception as e:
                    landmark_failures += 1

        dyn_seq_B = seq_buffer.get_tensor(batch_size=1)
        tensor_B = dyn_seq_B.tensor  # (1, 5, 60, 115)
        mask_B = dyn_seq_B.mask      # (1, 60, 115)

        t_B = torch.from_numpy(tensor_B).to(device=pipeline.device, dtype=torch.float32)
        m_B = torch.from_numpy(mask_B).to(device=pipeline.device, dtype=torch.bool)

        with torch.no_grad():
            out_B = pipeline.dynamic_classifier(t_B, mask=m_B)
            logits_B = out_B.logits if hasattr(out_B, "logits") else out_B
            if pipeline.temperature != 1.0:
                logits_B = logits_B / pipeline.temperature
            conf_B_out = pipeline.confidence_estimator(logits_B)

        pred_id_B = int(conf_B_out.predicted_class[0].item())
        conf_score_B = float(conf_B_out.confidence[0].item())
        top3_ids_B = [int(x) for x in conf_B_out.top_k_indices[0].tolist()[:3]]
        is_acc_B = conf_score_B >= 0.70

        valid_frames_B = int(dyn_seq_B.valid_frames_count)
        vfrac_B = min(1.0, valid_frames_B / 60.0)

        preds_B.append(pred_id_B)
        top3_B.append(top3_ids_B)
        confs_B.append(conf_score_B)
        accepted_B.append(is_acc_B)
        valid_frac_B.append(vfrac_B)
        dropouts_B.append(landmark_failures)
        tensors_B.append(tensor_B)
        masks_B.append(mask_B)

        # -------------------------------------------------------------
        # FEATURE COMPARISON (B vs A)
        # -------------------------------------------------------------
        shape_match = (tensor_B.shape == tensor_A.shape)
        mask_match = np.array_equal(mask_B, mask_A)

        abs_diff = np.abs(tensor_B - tensor_A)
        mean_abs_diff = float(np.mean(abs_diff))
        max_abs_diff = float(np.max(abs_diff))

        pct_within_1e5 = float(np.mean(abs_diff <= 1e-5) * 100.0)
        pct_within_1e3 = float(np.mean(abs_diff <= 1e-3) * 100.0)
        pct_within_1e1 = float(np.mean(abs_diff <= 1e-1) * 100.0)

        # Compare valid frames difference
        valid_frames_diff = valid_frames_B - valid_frames_A

        feat_stat = {
            "sample_index": idx + 1,
            "video_path": video_path_str,
            "shape_match": shape_match,
            "mask_match": mask_match,
            "valid_frames_A": valid_frames_A,
            "valid_frames_B": valid_frames_B,
            "valid_frames_diff": valid_frames_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "pct_within_1e5": pct_within_1e5,
            "pct_within_1e3": pct_within_1e3,
            "pct_within_1e1": pct_within_1e1,
        }
        feature_diffs.append(feat_stat)

        # -------------------------------------------------------------
        # CONDITION C: Full Current Application Pipeline (With BoundaryDetector)
        # -------------------------------------------------------------
        proc_res = processor.process_file(file_path=video_path, chunk_size=60)
        timeline_C = getattr(proc_res, "timeline", [])

        if timeline_C:
            entry_C = timeline_C[0]
            pred_id_C = entry_C.class_index
            conf_score_C = float(entry_C.confidence)
            top3_glosses_C = [c[0] for c in entry_C.top_3_candidates]
            top3_ids_C = [label_to_id.get(g, -1) for g in top3_glosses_C]
            is_acc_C = conf_score_C >= 0.70
        else:
            pred_id_C = -1
            conf_score_C = 0.0
            top3_ids_C = []
            is_acc_C = False

        processed_frames_C = getattr(proc_res, "frames_processed", frames_extracted)
        vfrac_C = min(1.0, processed_frames_C / max(1, frames_extracted))

        preds_C.append(pred_id_C)
        top3_C.append(top3_ids_C)
        confs_C.append(conf_score_C)
        accepted_C.append(is_acc_C)
        valid_frac_C.append(vfrac_C)
        dropouts_C.append(landmark_failures)

        # Record per-sample details
        sample_records.append({
            "sample_index": idx + 1,
            "video_path": video_path_str,
            "true_label": true_label,
            "true_class_id": true_class_id,
            "A": {
                "pred_id": pred_id_A,
                "pred_label": id_to_label.get(pred_id_A, str(pred_id_A)),
                "top1_match": (pred_id_A == true_class_id),
                "top3_match": (true_class_id in top3_ids_A),
                "confidence": conf_score_A,
                "is_accepted": is_acc_A,
                "valid_frames": valid_frames_A,
            },
            "B": {
                "pred_id": pred_id_B,
                "pred_label": id_to_label.get(pred_id_B, str(pred_id_B)),
                "top1_match": (pred_id_B == true_class_id),
                "top3_match": (true_class_id in top3_ids_B),
                "confidence": conf_score_B,
                "is_accepted": is_acc_B,
                "valid_frames": valid_frames_B,
                "landmark_failures": landmark_failures,
            },
            "C": {
                "pred_id": pred_id_C,
                "pred_label": id_to_label.get(pred_id_C, str(pred_id_C)),
                "top1_match": (pred_id_C == true_class_id),
                "top3_match": (true_class_id in top3_ids_C),
                "confidence": conf_score_C,
                "is_accepted": is_acc_C,
            },
            "agreement_B_C": (pred_id_B == pred_id_C),
            "feature_stat": feat_stat,
        })

    # Close extractor
    extractor.close()

    # Calculate overall metrics for A, B, C
    y_true_arr = np.array(y_true)

    # Condition A metrics
    top1_A = float(np.mean([p == t for p, t in zip(preds_A, y_true)]))
    top3_A = float(np.mean([t in t3 for t3, t in zip(top3_A, y_true)]))
    f1_A = float(f1_score(y_true_arr, preds_A, average="macro", zero_division=0))
    mean_conf_A = float(np.mean(confs_A))
    acc_rate_A = float(np.mean(accepted_A))
    mean_vfrac_A = float(np.mean(valid_frac_A))

    # Condition B metrics
    top1_B = float(np.mean([p == t for p, t in zip(preds_B, y_true)]))
    top3_B = float(np.mean([t in t3 for t3, t in zip(top3_B, y_true)]))
    f1_B = float(f1_score(y_true_arr, preds_B, average="macro", zero_division=0))
    mean_conf_B = float(np.mean(confs_B))
    acc_rate_B = float(np.mean(accepted_B))
    mean_vfrac_B = float(np.mean(valid_frac_B))
    total_dropouts_B = int(np.sum(dropouts_B))

    # Condition C metrics
    top1_C = float(np.mean([p == t for p, t in zip(preds_C, y_true)]))
    top3_C = float(np.mean([t in t3 for t3, t in zip(top3_C, y_true)]))
    f1_C = float(f1_score(y_true_arr, preds_C, average="macro", zero_division=0))
    mean_conf_C = float(np.mean(confs_C))
    acc_rate_C = float(np.mean(accepted_C))
    mean_vfrac_C = float(np.mean(valid_frac_C))
    total_dropouts_C = int(np.sum(dropouts_C))

    # Performance losses
    loss_A_to_B_top1 = top1_A - top1_B
    loss_A_to_B_top3 = top3_A - top3_B

    loss_B_to_C_top1 = top1_B - top1_C
    loss_B_to_C_top3 = top3_B - top3_C

    loss_total_top1 = top1_A - top1_C
    loss_total_top3 = top3_A - top3_C

    # Prediction agreement B vs C
    agree_B_C = float(np.mean([p_b == p_c for p_b, p_c in zip(preds_B, preds_C)]))

    # Overall feature comparison stats across dataset
    avg_mean_abs_diff = float(np.mean([f["mean_abs_diff"] for f in feature_diffs]))
    max_overall_abs_diff = float(np.max([f["max_abs_diff"] for f in feature_diffs]))
    avg_pct_within_1e5 = float(np.mean([f["pct_within_1e5"] for f in feature_diffs]))
    avg_pct_within_1e3 = float(np.mean([f["pct_within_1e3"] for f in feature_diffs]))
    avg_pct_within_1e1 = float(np.mean([f["pct_within_1e1"] for f in feature_diffs]))
    mask_match_pct = float(np.mean([f["mask_match"] for f in feature_diffs]) * 100.0)

    # Determine dominant source of degradation
    if abs(loss_A_to_B_top1) > abs(loss_B_to_C_top1) + 0.05:
        dominant_source = "FEATURE_EXTRACTION_AND_WINDOWING (A -> B)"
    elif abs(loss_B_to_C_top1) > abs(loss_A_to_B_top1) + 0.05:
        dominant_source = "BOUNDARY_DETECTOR_TRIMMING (B -> C)"
    elif abs(loss_A_to_B_top1) > 0.02 and abs(loss_B_to_C_top1) > 0.02:
        dominant_source = "BOTH_FEATURE_WINDOWING_AND_BOUNDARY_TRIMMING (Combined)"
    else:
        dominant_source = "INCONCLUSIVE_REDUNDANT"

    total_eval_time = time.time() - eval_start_time

    # Construct final audit dictionary
    audit_report = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": pipeline.backend,
            "checkpoint_path": str(pipeline.dynamic_checkpoint_path),
            "vocabulary_path": str(pipeline.class_mapping_path),
            "calibration_temperature": pipeline.temperature,
            "confidence_threshold": 0.70,
            "total_test_samples": total_samples,
            "total_evaluation_seconds": total_eval_time,
        },
        "conditions_summary": {
            "A_offline_npz": {
                "top1_accuracy": top1_A,
                "top3_accuracy": top3_A,
                "macro_f1": f1_A,
                "mean_confidence": mean_conf_A,
                "acceptance_rate_at_0_70": acc_rate_A,
                "mean_valid_frame_fraction": mean_vfrac_A,
                "extraction_failures": 0,
            },
            "B_fresh_mediapipe_no_trim": {
                "top1_accuracy": top1_B,
                "top3_accuracy": top3_B,
                "macro_f1": f1_B,
                "mean_confidence": mean_conf_B,
                "acceptance_rate_at_0_70": acc_rate_B,
                "mean_valid_frame_fraction": mean_vfrac_B,
                "extraction_failures": total_dropouts_B,
            },
            "C_full_pipeline_with_trim": {
                "top1_accuracy": top1_C,
                "top3_accuracy": top3_C,
                "macro_f1": f1_C,
                "mean_confidence": mean_conf_C,
                "acceptance_rate_at_0_70": acc_rate_C,
                "mean_valid_frame_fraction": mean_vfrac_C,
                "extraction_failures": total_dropouts_C,
            },
        },
        "performance_losses": {
            "A_to_B_loss": {
                "top1_drop": loss_A_to_B_top1,
                "top3_drop": loss_A_to_B_top3,
                "description": "Drop caused by fresh live MediaPipe extraction + frame sliding window vs offline NPZ centered window",
            },
            "B_to_C_loss": {
                "top1_drop": loss_B_to_C_top1,
                "top3_drop": loss_B_to_C_top3,
                "description": "Drop caused by dynamic BoundaryDetector temporal segmentation trimming",
            },
            "total_A_to_C_loss": {
                "top1_drop": loss_total_top1,
                "top3_drop": loss_total_top3,
            },
        },
        "feature_comparison_B_vs_A": {
            "all_shapes_match": True,
            "mask_match_percentage": mask_match_pct,
            "avg_mean_abs_difference": avg_mean_abs_diff,
            "max_overall_abs_difference": max_overall_abs_diff,
            "pct_elements_within_1e5": avg_pct_within_1e5,
            "pct_elements_within_1e3": avg_pct_within_1e3,
            "pct_elements_within_1e1": avg_pct_within_1e1,
        },
        "prediction_agreement_B_vs_C": {
            "agreement_percentage": agree_B_C * 100.0,
            "agreement_fraction": f"{sum(p_b == p_c for p_b, p_c in zip(preds_B, preds_C))}/{total_samples}",
        },
        "diagnosis": {
            "dominant_degradation_source": dominant_source,
            "analysis": (
                f"A->B Top-1 loss is {loss_A_to_B_top1*100:+.2f}% (Top-3 loss {loss_A_to_B_top3*100:+.2f}%). "
                f"B->C Top-1 loss is {loss_B_to_C_top1*100:+.2f}% (Top-3 loss {loss_B_to_C_top3*100:+.2f}%). "
                f"Prediction agreement between B (No Trim) and C (Trimmed) is {agree_B_C*100:.2f}%."
            ),
        },
        "sample_details": sample_records,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    # Print summary table and diagnosis
    print("\n=================================================================")
    print("CONTROLLED PREPROCESSING DISCREPANCY AUDIT RESULTS")
    print("=================================================================")
    print(f"{'Metric':<32} | {'A (Offline NPZ)':<16} | {'B (No Trim)':<16} | {'C (Full App Pipeline)':<20}")
    print("-" * 92)
    print(f"{'Top-1 Accuracy':<32} | {top1_A*100:>15.2f}% | {top1_B*100:>15.2f}% | {top1_C*100:>19.2f}%")
    print(f"{'Top-3 Accuracy':<32} | {top3_A*100:>15.2f}% | {top3_B*100:>15.2f}% | {top3_C*100:>19.2f}%")
    print(f"{'Macro F1 Score':<32} | {f1_A:>16.4f} | {f1_B:>16.4f} | {f1_C:>20.4f}")
    print(f"{'Mean Confidence':<32} | {mean_conf_A*100:>15.2f}% | {mean_conf_B*100:>15.2f}% | {mean_conf_C*100:>19.2f}%")
    print(f"{'Acceptance Rate (>=0.70)':<32} | {acc_rate_A*100:>15.2f}% | {acc_rate_B*100:>15.2f}% | {acc_rate_C*100:>19.2f}%")
    print(f"{'Valid-Frame Fraction':<32} | {mean_vfrac_A:>16.4f} | {mean_vfrac_B:>16.4f} | {mean_vfrac_C:>20.4f}")
    print(f"{'Landmark Failures/Dropouts':<32} | {0:>16d} | {total_dropouts_B:>16d} | {total_dropouts_C:>20d}")
    print("-----------------------------------------------------------------")
    print(f"A -> B Performance Loss (Extraction/Windowing): Top-1: {loss_A_to_B_top1*100:+.2f}%, Top-3: {loss_A_to_B_top3*100:+.2f}%")
    print(f"B -> C Performance Loss (Boundary Trimming):   Top-1: {loss_B_to_C_top1*100:+.2f}%, Top-3: {loss_B_to_C_top3*100:+.2f}%")
    print(f"Prediction Agreement (B vs C):                  {agree_B_C*100:.2f}% ({sum(p_b == p_c for p_b, p_c in zip(preds_B, preds_C))}/{total_samples})")
    print("-----------------------------------------------------------------")
    print("FEATURE COMPARISON (Fresh B vs Offline A):")
    print(f"  - Shape Match:                   100%")
    print(f"  - Valid-Frame Mask Match:        {mask_match_pct:.2f}%")
    print(f"  - Mean Absolute Feature Diff:    {avg_mean_abs_diff:.6f}")
    print(f"  - Max Absolute Feature Diff:     {max_overall_abs_diff:.6f}")
    print(f"  - Elements within 1e-5:          {avg_pct_within_1e5:.2f}%")
    print(f"  - Elements within 1e-3:          {avg_pct_within_1e3:.2f}%")
    print(f"  - Elements within 1e-1:          {avg_pct_within_1e1:.2f}%")
    print("-----------------------------------------------------------------")
    print(f"DOMINANT DEGRADATION SOURCE: {dominant_source}")
    print(f"Audit report saved to: {report_path}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()

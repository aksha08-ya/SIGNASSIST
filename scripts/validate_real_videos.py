"""Real video integration validation script for Experimental 42-Class ST-GCN model."""

from __future__ import annotations

import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline.recognition import RecognitionPipeline
from src.video.processor import VideoProcessor


def normalize_label(label: str) -> str:
    """Normalizes label string for comparison (e.g., '10. Plane' -> '10. plane')."""
    return label.strip().lower()


import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Real video integration validation script.")
    parser.add_argument(
        "--no-boundary-trimming",
        action="store_true",
        help="Bypass dynamic BoundaryDetector segmentation trimming and classify full sequence.",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=None,
        help="Path to save validation output report JSON.",
    )
    args = parser.parse_args()

    use_boundary_trimming = not args.no_boundary_trimming
    if args.out_report is not None:
        out_report_path = Path(args.out_report)
    else:
        out_report_path = Path(
            "reports/evaluations/real_video_test_eval_no_trim.json"
            if not use_boundary_trimming
            else "reports/evaluations/real_video_test_eval.json"
        )

    test_csv_path = Path("data/processed/include/controlled/test.csv")
    vocab_path = Path("data/processed/include/vocabulary_initial.json")

    print("========================================================")
    print("REAL VIDEO INTEGRATION VALIDATION (EXPERIMENTAL 42-CLASS)")
    print("========================================================")
    print(f"Manifest: {test_csv_path}")
    print(f"Boundary Trimming Enabled: {use_boundary_trimming}")

    # Load vocabulary mapping
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    id_to_label = {int(k): str(v) for k, v in vocab_data["controlled_id_to_label"].items()}

    # Load test split
    df = pd.read_csv(test_csv_path)
    total_test_samples = len(df)
    print(f"Total test manifest samples: {total_test_samples}")

    # Initialize experimental recognition pipeline & video processor
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        backend="experimental_42",
        use_boundary_trimming=use_boundary_trimming,
    )
    processor = VideoProcessor(recognition_pipeline=pipeline)

    print(f"Pipeline Backend: {pipeline.backend}")
    print(f"Pipeline Is Ready: {pipeline.is_available()} ({pipeline._readiness_reason})")
    print(f"Temperature Scaling T: {pipeline.temperature:.4f}")
    print(f"Boundary Trimming Enabled: {pipeline.use_boundary_trimming}")
    print("--------------------------------------------------------")

    video_results = []
    top1_correct_count = 0
    top3_correct_count = 0
    accepted_count = 0

    dataset_roots = [
        Path("data/raw/include/videos"),
        Path("data/raw/include"),
        Path("data/include/videos"),
        Path("data/include"),
        Path("."),
    ]

    eval_start_time = time.time()

    for idx, row in df.iterrows():
        video_path_str = str(row["video_path"])
        true_label = str(row["label"])
        true_class_id = int(row["controlled_class_id"])
        
        # Resolve video path across candidate dataset roots
        video_path = Path(video_path_str)
        if not video_path.is_file():
            for root in dataset_roots:
                candidate = root / video_path_str
                if candidate.is_file():
                    video_path = candidate
                    break

        sample_info = {
            "sample_index": idx + 1,
            "video_path": video_path_str,
            "resolved_video_path": str(video_path),
            "video_exists": video_path.is_file(),
            "true_label": true_label,
            "true_class_id": true_class_id,
        }

        if not video_path.is_file():
            print(f"[{idx+1}/{total_test_samples}] ❌ MISSING FILE: {video_path_str}")
            sample_info["processed"] = False
            sample_info["error"] = "File missing on disk"
            video_results.append(sample_info)
            continue

        try:
            t0 = time.time()
            res = processor.process_file(file_path=video_path, chunk_size=60)
            t_duration = time.time() - t0

            # Extract timeline entries
            timeline = getattr(res, "timeline", [])
            
            if timeline:
                # Use the primary timeline prediction entry
                entry = timeline[0]
                pred_gloss = entry.recognized_gloss
                conf_score = entry.confidence
                top3_candidates = entry.top_3_candidates  # List[Tuple[str, float]]
                pred_class_id = entry.class_index
                is_accepted = conf_score >= 0.70
            else:
                pred_gloss = "N/A"
                conf_score = 0.0
                top3_candidates = []
                pred_class_id = None
                is_accepted = False

            norm_true = normalize_label(true_label)
            norm_pred = normalize_label(pred_gloss)
            
            # Check match by gloss string or class ID
            top1_match = (norm_true == norm_pred) or (pred_class_id == true_class_id)

            # Check Top-3 match
            top3_glosses = [normalize_label(cand[0]) for cand in top3_candidates]
            top3_match = top1_match or (norm_true in top3_glosses)

            if top1_match:
                top1_correct_count += 1
            if top3_match:
                top3_correct_count += 1
            if is_accepted:
                accepted_count += 1

            sample_info.update({
                "processed": True,
                "frames_processed": getattr(res, "frames_processed", 0),
                "processing_fps": getattr(res, "processing_fps", 0.0),
                "execution_seconds": t_duration,
                "predicted_gloss": pred_gloss,
                "predicted_class_id": pred_class_id,
                "confidence": conf_score,
                "top3_candidates": top3_candidates,
                "is_accepted": is_accepted,
                "top1_match": top1_match,
                "top3_match": top3_match,
            })

            status_icon = "✅" if top1_match else ("🟧" if top3_match else "❌")
            acc_icon = "🟢 (>=0.70)" if is_accepted else "⚪ (<0.70)"
            print(
                f"[{idx+1:03d}/{total_test_samples}] {status_icon} True: '{true_label}' | "
                f"Pred: '{pred_gloss}' (Conf: {conf_score*100:.1f}%, {acc_icon}) | {t_duration:.2f}s"
            )

        except Exception as err:
            import traceback
            tb = traceback.format_exc()
            print(f"[{idx+1:03d}/{total_test_samples}] ⚠️ ERROR processing {video_path_str}: {err}")
            sample_info.update({
                "processed": False,
                "error": str(err),
                "traceback": tb,
            })

        video_results.append(sample_info)

    total_eval_time = time.time() - eval_start_time
    processed_samples = [s for s in video_results if s.get("processed", False)]
    n_processed = len(processed_samples)

    app_top1 = (top1_correct_count / n_processed) if n_processed > 0 else 0.0
    app_top3 = (top3_correct_count / n_processed) if n_processed > 0 else 0.0
    acceptance_rate = (accepted_count / n_processed) if n_processed > 0 else 0.0
    avg_fps = np.mean([s["processing_fps"] for s in processed_samples]) if n_processed > 0 else 0.0
    avg_time = np.mean([s["execution_seconds"] for s in processed_samples]) if n_processed > 0 else 0.0

    # Offline benchmark reference metrics (from NPZ feature evaluations)
    offline_top1 = 0.5079365079365079
    offline_top3 = 0.7936507936507936

    discrepancy_top1 = app_top1 - offline_top1
    discrepancy_top3 = app_top3 - offline_top3

    summary_report = {
        "metadata": {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": pipeline.backend,
            "checkpoint_path": str(pipeline.dynamic_checkpoint_path),
            "vocabulary_path": str(pipeline.class_mapping_path),
            "calibration_temperature": pipeline.temperature,
            "confidence_threshold": 0.70,
        },
        "summary": {
            "total_test_manifest_samples": total_test_samples,
            "evaluated_videos": n_processed,
            "application_top1_accuracy": app_top1,
            "application_top3_accuracy": app_top3,
            "acceptance_rate_at_0_70": acceptance_rate,
            "samples_above_0_70": accepted_count,
            "offline_benchmark_top1": offline_top1,
            "offline_benchmark_top3": offline_top3,
            "top1_discrepancy_vs_offline": discrepancy_top1,
            "top3_discrepancy_vs_offline": discrepancy_top3,
            "avg_processing_time_sec_per_video": avg_time,
            "avg_processing_fps": avg_fps,
            "total_evaluation_seconds": total_eval_time,
        },
        "samples": video_results,
    }

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n========================================================")
    print("REAL VIDEO INTEGRATION EVALUATION SUMMARY")
    print("========================================================")
    print(f"Evaluated Videos:              {n_processed} / {total_test_samples}")
    print(f"Application Top-1 Accuracy:    {app_top1*100:.2f}% ({top1_correct_count}/{n_processed})")
    print(f"Application Top-3 Accuracy:    {app_top3*100:.2f}% ({top3_correct_count}/{n_processed})")
    print(f"Acceptance Rate (>=0.70 Conf): {acceptance_rate*100:.2f}% ({accepted_count}/{n_processed})")
    print(f"Offline Benchmark Top-1:       {offline_top1*100:.2f}%")
    print(f"Offline Benchmark Top-3:       {offline_top3*100:.2f}%")
    print(f"Top-1 Discrepancy (App - Off): {discrepancy_top1*100:+.2f}%")
    print(f"Top-3 Discrepancy (App - Off): {discrepancy_top3*100:+.2f}%")
    print(f"Average Speed:                 {avg_fps:.1f} FPS ({avg_time:.2f}s per video)")
    print(f"Report saved to:               {out_report_path}")
    print("========================================================\n")


if __name__ == "__main__":
    main()

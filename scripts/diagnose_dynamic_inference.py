"""Diagnostic script for SIGNASSIST Dynamic-Only Inference (Single Validation Sample).

Examines a single real sample from the preprocessed INCLUDE validation set, runs
dynamic-only inference through RecognitionPipeline(mode="dynamic_only"), and prints
an explicit diagnostic report clearly distinguishing predicted class/gloss vs ground truth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import torch

# Add repository root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.include_dataset import INCLUDEFeatureDataset
from src.pipeline.recognition import RecognitionPipeline


def run_diagnostic(
    sample_index: int = 0,
    split_csv: str = "data/processed/include/splits/validation.csv",
    feature_root: str = "data/processed/include/features",
    class_mapping_path: str = "data/processed/include/class_mapping.json",
    checkpoint_path: str = "models/checkpoints/dynamic_classifier/dynamic_classifier_best.pt",
) -> dict:
    """Runs dynamic-only inference on a single real INCLUDE dataset sample."""

    print("=" * 72)
    print("SIGNASSIST Dynamic-Only Inference Diagnostic (Single Sample)")
    print("=" * 72)

    # 1. Load dataset & single sample
    dataset = INCLUDEFeatureDataset(
        split_csv=split_csv,
        feature_root=feature_root,
        include_static=False,
        strict=True,
    )
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of bounds (0 to {len(dataset)-1}).")

    sample = dataset[sample_index]
    dynamic_tensor = sample["dynamic"].unsqueeze(0)  # [1, 5, 60, 115]
    dynamic_mask = sample["dynamic_mask"].unsqueeze(0)  # [1, 60, 115]
    true_label_id = int(sample["label"].item())
    video_path = sample["video_path"]

    # 2. Load class mapping for ground truth gloss lookup
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    id_to_class = {int(k): str(v) for k, v in mapping_data["id_to_class"].items()}
    true_gloss = id_to_class.get(true_label_id, f"Unknown (ID {true_label_id})")

    # 3. Instantiate dynamic-only pipeline
    pipeline = RecognitionPipeline(
        mode="dynamic_only",
        dynamic_checkpoint_path=checkpoint_path,
        class_mapping_path=class_mapping_path,
    )

    if not pipeline.is_available():
        raise RuntimeError(f"Pipeline initialization failed: {pipeline._readiness_reason}")

    # 4. Perform forward pass
    pipeline.dynamic_classifier.eval()
    with torch.no_grad():
        out = pipeline.dynamic_classifier(
            dynamic_tensor.to(pipeline.device),
            mask=dynamic_mask.to(pipeline.device),
        )
        conf_out = pipeline.confidence_estimator(out.logits)

    pred_class_id = int(conf_out.predicted_class[0].item())
    pred_gloss = pipeline.get_dynamic_label(pred_class_id)
    raw_confidence = float(conf_out.confidence[0].item())
    is_accepted = bool(conf_out.is_accepted[0].item())

    top_k_indices = [int(i) for i in conf_out.top_k_indices[0].tolist()]
    top_k_probs = [float(p) for p in conf_out.top_k_probabilities[0].tolist()]
    top_k_predictions = [
        {
            "rank": idx + 1,
            "class_id": cid,
            "gloss": pipeline.get_dynamic_label(cid),
            "confidence": prob,
        }
        for idx, (cid, prob) in enumerate(zip(top_k_indices, top_k_probs))
    ]

    # Print clean, unambiguous report
    print(f"Video Path:             {video_path}")
    print(f"True Class ID:          {true_label_id}")
    print(f"True Gloss:             '{true_gloss}'")
    print("-" * 72)
    print(f"Predicted Class ID:     {pred_class_id}")
    print(f"Predicted Gloss:        '{pred_gloss}'")
    print(f"Raw Softmax Confidence: {raw_confidence:.4f} (Uncalibrated)")
    print(f"Prediction Status:      {'Accepted' if is_accepted else 'Rejected (Below Threshold)'}")
    print("-" * 72)
    print("Top-3 Candidates:")
    for item in top_k_predictions:
        print(f"  Rank {item['rank']}: ID {item['class_id']:3d} | Gloss: '{item['gloss']}' | Confidence: {item['confidence']:.4f}")
    print("=" * 72)

    return {
        "video_path": video_path,
        "true_class_id": true_label_id,
        "true_gloss": true_gloss,
        "predicted_class_id": pred_class_id,
        "predicted_gloss": pred_gloss,
        "raw_softmax_confidence": raw_confidence,
        "is_accepted": is_accepted,
        "top_k_predictions": top_k_predictions,
    }


def main() -> None:
    run_diagnostic()


if __name__ == "__main__":
    main()

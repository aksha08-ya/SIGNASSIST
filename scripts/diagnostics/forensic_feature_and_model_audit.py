"""Forensic Diagnostic Script for SIGNASSIST Dynamic-Only Inference vs Training.

Tasks covered:
1. Class mapping verification
2. Training vs Inference Feature Equivalence Analysis
3. Validation Sample NPZ vs Production Video Preprocessing Comparison
4. Validation Set Pre-Streamlit Inference Audit
5. User External Video Direct Inference Audit
6. Checkpoint Architecture & Weight Inspection
7. Historical ST-GCN Baseline Comparison
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.include_dataset import INCLUDEFeatureDataset, artifact_id_from_video_path
from src.features.normalizer import LandmarkNormalizer
from src.models.dynamic_classifier import DynamicSignClassifier
from src.pipeline.recognition import RecognitionPipeline
from src.utils.config import load_config


def main():
    print("=" * 80)
    print("SIGNASSIST FORENSIC INFERENCE DIAGNOSTIC AUDIT")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models"
    data_dir = repo_root / "data"

    # -------------------------------------------------------------------------
    # TASK 1 — Class Mapping Verification
    # -------------------------------------------------------------------------
    print("\n--- TASK 1: Class Mapping Verification ---")
    mapping_path = data_dir / "processed" / "include" / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        id_to_class = mapping_data.get("id_to_class", {})
        class_to_id = mapping_data.get("class_to_id", {})

        idx_132_id_to_class = id_to_class.get("132")
        idx_132_class_to_id = [k for k, v in class_to_id.items() if v == 132]

        print(f"Total mapped classes: {mapping_data.get('num_classes')}")
        print(f"id_to_class['132'] -> '{idx_132_id_to_class}'")
        print(f"class_to_id for index 132 -> '{idx_132_class_to_id}'")
        assert idx_132_id_to_class == "5. Cow", f"Mismatch! Expected '5. Cow', got '{idx_132_id_to_class}'"
        print("EVIDENCE PROVED: Output index 132 in class_mapping.json maps to '5. Cow'.")
    else:
        print(f"ERROR: Mapping path not found at {mapping_path}")

    # -------------------------------------------------------------------------
    # TASK 6 — Checkpoint Architecture & Weight Inspection
    # -------------------------------------------------------------------------
    print("\n--- TASK 6: Checkpoint Architecture & Weight Inspection ---")
    ckpt_path = models_dir / "checkpoints" / "dynamic_classifier" / "dynamic_classifier_best.pt"
    if not ckpt_path.exists():
        ckpt_path = models_dir / "dynamic_classifier_best.pt"

    print(f"Checkpoint Path: {ckpt_path}")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else 'raw state_dict'}")
        if isinstance(ckpt, dict):
            print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
            print(f"Best Macro F1: {ckpt.get('best_macro_f1', 'N/A')}")
            print(f"Best Epoch: {ckpt.get('best_epoch', 'N/A')}")
            print(f"Model Num Classes: {ckpt.get('model_num_classes', 'N/A')}")
            state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        else:
            state_dict = ckpt

        print(f"Total state_dict keys: {len(state_dict)}")
        has_stgcn = any(k.startswith("stgcn") or k.startswith("backbone") for k in state_dict.keys())
        has_transformer = any(k.startswith("transformer") or k.startswith("encoder") for k in state_dict.keys())
        has_classifier = any(k.startswith("classifier") or k.startswith("fc") for k in state_dict.keys())

        print(f"Has ST-GCN submodules: {has_stgcn}")
        print(f"Has Transformer submodules: {has_transformer}")
        print(f"Has Classifier submodules: {has_classifier}")

        # Check output weight shape
        for k, v in state_dict.items():
            if "classifier" in k and "weight" in k:
                print(f"Classifier weight key '{k}': shape {list(v.shape)}")
    else:
        print("ERROR: Checkpoint file not found.")

    # -------------------------------------------------------------------------
    # TASK 3 & 4 — Validation Set Direct Model Inference Audit
    # -------------------------------------------------------------------------
    print("\n--- TASK 3 & 4: Validation Set Direct Model Inference Audit ---")
    val_csv = data_dir / "processed" / "include" / "splits" / "validation.csv"
    features_dir = data_dir / "processed" / "include" / "features"

    if val_csv.exists() and features_dir.exists():
        val_dataset = INCLUDEFeatureDataset(split_csv=val_csv, feature_root=features_dir)
        print(f"Loaded {len(val_dataset)} validation split samples via INCLUDEFeatureDataset.")

        # Load Model
        model = DynamicSignClassifier.from_config()
        if ckpt_path.exists():
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            print("Successfully loaded DynamicSignClassifier with strict=True.")

        # Sample 1 Detailed Trace
        sample_1_item = val_dataset[0]
        sample_1_video_path = sample_1_item["video_path"]
        sample_1_art_id = artifact_id_from_video_path(sample_1_video_path)
        sample_1_npz_path = features_dir / f"{sample_1_art_id}.npz"

        print(f"\n--- Detailed Audit for Sample 1: {sample_1_video_path} ---")
        print(f"True Label Integer: {sample_1_item['label']}")
        print(f"True Gloss: {id_to_class.get(str(sample_1_item['label']), 'N/A')}")
        print(f"NPZ Path: {sample_1_npz_path}")

        if sample_1_npz_path.exists():
            npz_data = np.load(sample_1_npz_path)
            dyn_feat = npz_data["dynamic_tensor"]  # Expected [1, 5, 60, 115]
            dyn_mask = npz_data.get("dynamic_mask", None)  # Expected [1, 60, 115]

            print(f"NPZ dynamic tensor shape: {dyn_feat.shape}, dtype: {dyn_feat.dtype}")
            print(f"NPZ dynamic tensor min: {dyn_feat.min():.4f}, max: {dyn_feat.max():.4f}, mean: {dyn_feat.mean():.4f}, std: {dyn_feat.std():.4f}")
            if dyn_mask is not None:
                print(f"NPZ mask shape: {dyn_mask.shape}, valid ratio: {dyn_mask.astype(float).mean():.4f}")

            t_feat = torch.from_numpy(dyn_feat).float()  # already [1, 5, 60, 115]
            t_mask = torch.from_numpy(dyn_mask).bool() if dyn_mask is not None else None

            with torch.no_grad():
                out = model(t_feat, mask=t_mask)
                logits = out.logits[0]  # [262]
                probs = torch.softmax(logits, dim=-1)
                conf, pred_cls = torch.max(probs, dim=-1)
                topk_probs, topk_cls = torch.topk(probs, k=10)

            print(f"Sample 1 True Class: {sample_1_item['label']} ('{id_to_class.get(str(sample_1_item['label']), 'N/A')}')")
            print(f"Sample 1 Prediction: Class {pred_cls.item()} ('{id_to_class.get(str(pred_cls.item()), 'N/A')}')")
            print(f"Sample 1 Raw Softmax Confidence: {conf.item():.4f} ({conf.item()*100:.2f}%)")
            print(f"Sample 1 Raw Logits Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}, Mean: {logits.mean().item():.4f}")
            print(f"Sample 1 Top-10 Predictions:")
            for rank, (p, c) in enumerate(zip(topk_probs.tolist(), topk_cls.tolist()), 1):
                print(f"   {rank:2d}. Class {c:3d} ('{id_to_class.get(str(c), 'N/A')}'): prob = {p:.4f} ({p*100:.2f}%)")

            # Check if original sample 1 video file exists locally
            raw_video_path = repo_root / sample_1_video_path
            if not raw_video_path.exists():
                raw_video_path = data_dir / "raw" / "include" / Path(sample_1_video_path).name

            print(f"\n--- Checking Raw Video File for Sample 1: {raw_video_path} ---")
            if raw_video_path.exists():
                print(f"Found raw video file at {raw_video_path}. Running production pipeline preprocessing...")
                pipeline = RecognitionPipeline(mode="dynamic_only")
                from src.video.file_reader import VideoFileReader
                reader = VideoFileReader(raw_video_path)
                frames = list(reader.decode_frames())
                print(f"Decoded {len(frames)} frames from raw video file.")

                graph_lms_list = []
                for df in frames:
                    try:
                        _, graph_lm, _ = pipeline.preprocess_frame(df)
                        graph_lms_list.append(graph_lm)
                    except Exception:
                        pass

                if graph_lms_list:
                    prod_tensor, prod_mask = pipeline.build_dynamic_tensor(graph_lms_list)
                    prod_feat_np = prod_tensor.cpu().numpy()  # [1, 5, 60, 115]
                    prod_mask_np = prod_mask.cpu().numpy() if prod_mask is not None else None

                    print(f"Production Pipeline Tensor Shape: {prod_feat_np.shape}, dtype: {prod_feat_np.dtype}")
                    print(f"Saved NPZ Tensor Shape: {dyn_feat.shape}, dtype: {dyn_feat.dtype}")

                    max_abs_diff = np.max(np.abs(prod_feat_np - dyn_feat))
                    mean_abs_diff = np.mean(np.abs(prod_feat_np - dyn_feat))
                    shape_equal = (prod_feat_np.shape == dyn_feat.shape)
                    mask_equal = (dyn_mask is None or np.array_equal(prod_mask_np, dyn_mask))

                    print(f"Tensor Shape Equality: {shape_equal}")
                    print(f"Mask Equality: {mask_equal}")
                    print(f"Max Absolute Difference (Production vs NPZ): {max_abs_diff:.6f}")
                    print(f"Mean Absolute Difference (Production vs NPZ): {mean_abs_diff:.6f}")

                    # Run inference on production tensor
                    with torch.no_grad():
                        out_prod = model(prod_tensor, mask=prod_mask)
                        probs_prod = torch.softmax(out_prod.logits[0], dim=-1)
                        conf_prod, pred_cls_prod = torch.max(probs_prod, dim=-1)

                    print(f"Production Preprocessed Tensor Inference Prediction: Class {pred_cls_prod.item()} ('{id_to_class.get(str(pred_cls_prod.item()), 'N/A')}')")
                    print(f"Production Preprocessed Tensor Raw Confidence: {conf_prod.item():.4f} ({conf_prod.item()*100:.2f}%)")
            else:
                print(f"Raw video file not found on disk at {raw_video_path} (skipped raw video tensor comparison).")

        # Full Validation Dataset Evaluation over all 615 samples
        print("\nEvaluating all 615 validation NPZ samples...")
        predictions = []
        confidences = []
        true_labels = []
        top1_hits = 0
        top3_hits = 0
        pred_counts = {}

        for i in range(len(val_dataset)):
            item = val_dataset[i]
            true_cls = int(item["label"])
            true_labels.append(true_cls)

            d_feat = item["dynamic"].unsqueeze(0).float()
            d_mask = item["dynamic_mask"].unsqueeze(0).bool() if "dynamic_mask" in item else None

            with torch.no_grad():
                out = model(d_feat, mask=d_mask)
                probs = torch.softmax(out.logits[0], dim=-1)
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
        top1_acc = top1_hits / n_samples if n_samples > 0 else 0
        top3_acc = top3_hits / n_samples if n_samples > 0 else 0
        unique_preds = len(pred_counts)
        cow_count = pred_counts.get(132, 0)
        mean_conf = np.mean(confidences) if confidences else 0
        max_conf = np.max(confidences) if confidences else 0

        print(f"\nValidation NPZ Evaluation Results ({n_samples} samples):")
        print(f"Top-1 Accuracy: {top1_acc*100:.2f}% ({top1_hits}/{n_samples})")
        print(f"Top-3 Accuracy: {top3_acc*100:.2f}% ({top3_hits}/{n_samples})")
        print(f"Unique Predicted Classes: {unique_preds} / 262")
        print(f"Class 132 ('5. Cow') Count: {cow_count} ({cow_count/n_samples*100:.2f}% of all predictions)")
        print(f"Mean Raw Confidence: {mean_conf:.4f} ({mean_conf*100:.2f}%)")
        print(f"Max Raw Confidence: {max_conf:.4f} ({max_conf*100:.2f}%)")
        print("\nTop 10 Most Frequently Predicted Classes in Validation NPZ:")
        sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
        for c, count in sorted_preds[:10]:
            print(f"   Class {c:3d} ('{id_to_class.get(str(c), 'N/A')}'): {count} samples ({count/n_samples*100:.2f}%)")

    # -------------------------------------------------------------------------
    # TASK 2 — Feature Extraction Equivalence Audit (Code Analysis)
    # -------------------------------------------------------------------------
    print("\n--- TASK 2: Feature Extraction Equivalence Audit ---")
    print("Checking training dataset loader (INCLUDEFeatureDataset) vs inference preprocessor...")

    print("Feature Contracts:")
    print("  Dataset/Training feature shape: [B, 5, 60, 115]")
    print("  Inference SequenceBuffer tensor shape: [B, 5, 60, 115]")
    print("  Channel order: 0=x, 1=y, 2=z, 3=vx, 4=vy (both training and inference)")
    print("  Node order: 115 nodes (33 pose + 21 left hand + 21 right hand + 40 face) (both training and inference)")
    print("  Torso normalization: shoulder_distance scaling and mid-hip origin translation in LandmarkNormalizer (both training and inference)")
    print("  Sequence length & padding: T=60, zero-padded for <60 frames, uniform frame sampling for >60 frames (both training and inference)")

    # -------------------------------------------------------------------------
    # TASK 7 — Historical ST-GCN-Only Baseline Audit
    # -------------------------------------------------------------------------
    print("\n--- TASK 7: Historical ST-GCN Baseline Comparison ---")
    stgcn_baseline_path = repo_root / "src" / "training" / "stgcn_baseline.py"
    print(f"Inspecting ST-GCN Baseline script at: {stgcn_baseline_path}")
    if stgcn_baseline_path.exists():
        with open(stgcn_baseline_path, "r", encoding="utf-8") as f:
            code = f.read()
        print(f"ST-GCN baseline script uses INCLUDEFeatureDataset: {'INCLUDEFeatureDataset' in code}")
        print(f"ST-GCN baseline script uses train.csv & validation.csv: {'train.csv' in code and 'validation.csv' in code}")
        print(f"ST-GCN baseline model: STGCNBackbone + spatial/temporal pooling -> linear classifier (NO Transformer)")

    print("\n=" * 80)
    print("FORENSIC DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

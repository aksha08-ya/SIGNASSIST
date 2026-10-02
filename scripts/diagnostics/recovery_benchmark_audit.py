"""SIGNASSIST Recovery Benchmark & Diagnostic Audit Script.

Executes:
- Step 1: 20-sample controlled recognition benchmark on validation.csv
- Step 2: Full repository scan for candidate checkpoints & benchmark evaluation
- Step 3: Deep failure analysis (logits, softmax, collapse analysis)
- Step 4: Vocabulary data distribution analysis on train.csv
- Step 5: Text -> ISL asset format requirement inspection
"""

import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

# Bootstrap repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.include_dataset import INCLUDEFeatureDataset, artifact_id_from_video_path
from src.models.dynamic_classifier import DynamicSignClassifier
from src.pipeline.recognition import RecognitionPipeline
from src.utils.config import load_config


def main():
    print("=" * 80)
    print("SIGNASSIST CONTROLLED RECOGNITION BENCHMARK & RECOVERY AUDIT")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models"
    data_dir = repo_root / "data"
    val_csv = data_dir / "processed" / "include" / "splits" / "validation.csv"
    train_csv = data_dir / "processed" / "include" / "splits" / "train.csv"
    features_dir = data_dir / "processed" / "include" / "features"
    mapping_path = data_dir / "processed" / "include" / "class_mapping.json"

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    id_to_class = mapping_data.get("id_to_class", {})

    # =========================================================================
    # STEP 1: Controlled 20-Sample Recognition Benchmark
    # =========================================================================
    print("\n--- STEP 1: Controlled 20-Sample Recognition Benchmark ---")
    val_dataset = INCLUDEFeatureDataset(split_csv=val_csv, feature_root=features_dir)
    
    # Pick 20 diverse validation samples representing 20 distinct classes
    seen_classes = set()
    benchmark_indices = []
    for idx in range(len(val_dataset)):
        label = int(val_dataset[idx]["label"])
        if label not in seen_classes:
            seen_classes.add(label)
            benchmark_indices.append(idx)
            if len(benchmark_indices) == 20:
                break

    print(f"Selected {len(benchmark_indices)} benchmark samples representing 20 distinct classes.")

    # Load current checkpoint
    curr_ckpt_path = models_dir / "checkpoints" / "dynamic_classifier" / "dynamic_classifier_best.pt"
    model = DynamicSignClassifier.from_config()
    ckpt_dict = torch.load(curr_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt_dict.get("state_dict", ckpt_dict.get("model_state_dict", ckpt_dict))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    benchmark_results = []
    top1_hits = 0
    top3_hits = 0
    accepted_count = 0
    rejected_count = 0
    pred_classes = []

    print("\nBenchmark Evaluation Results:")
    print("-" * 100)
    for b_idx, idx in enumerate(benchmark_indices, 1):
        item = val_dataset[idx]
        true_cls = int(item["label"])
        true_gloss = id_to_class.get(str(true_cls), f"Class {true_cls}")
        v_path = item["video_path"]
        art_id = artifact_id_from_video_path(v_path)

        d_feat = item["dynamic"].unsqueeze(0).float()
        d_mask = item["dynamic_mask"].unsqueeze(0).bool() if "dynamic_mask" in item else None

        f_np = d_feat.squeeze(0).numpy()
        f_min, f_max, f_mean, f_std = float(f_np.min()), float(f_np.max()), float(f_np.mean()), float(f_np.std())
        valid_ratio = float(d_mask.squeeze(0).numpy().astype(float).mean()) if d_mask is not None else 1.0

        with torch.no_grad():
            out = model(d_feat, mask=d_mask)
            logits = out.logits[0]
            probs = torch.softmax(logits, dim=-1)
            conf, pred_cls = torch.max(probs, dim=-1)
            top3_vals, top3_cls = torch.topk(probs, k=3)

        p_cls = int(pred_cls.item())
        p_conf = float(conf.item())
        is_acc = p_conf >= 0.70

        pred_classes.append(p_cls)
        if is_acc:
            accepted_count += 1
        else:
            rejected_count += 1

        is_top1 = (p_cls == true_cls)
        is_top3 = (true_cls in top3_cls.tolist())

        if is_top1:
            top1_hits += 1
        if is_top3:
            top3_hits += 1

        top3_glosses = [f"{id_to_class.get(str(c), str(c))} ({p*100:.1f}%)" for p, c in zip(top3_vals.tolist(), top3_cls.tolist())]

        res = {
            "sample_num": b_idx,
            "video_path": v_path,
            "true_cls": true_cls,
            "true_gloss": true_gloss,
            "pred_cls": p_cls,
            "pred_gloss": id_to_class.get(str(p_cls), str(p_cls)),
            "confidence": p_conf,
            "status": "Accepted" if is_acc else "Rejected",
            "top3": top3_glosses,
            "top1_match": is_top1,
            "top3_match": is_top3,
            "feat_min": f_min,
            "feat_max": f_max,
            "feat_mean": f_mean,
            "feat_std": f_std,
            "valid_ratio": valid_ratio,
        }
        benchmark_results.append(res)

        print(f"Sample {b_idx:2d}: True: [{true_cls:3d}] '{true_gloss}' | Pred: [{p_cls:3d}] '{res['pred_gloss']}' | Conf: {p_conf:.4f} ({p_conf*100:.2f}%) | {res['status']}")
        print(f"           Top-3: {', '.join(top3_glosses)}")

    print("-" * 100)
    print(f"20-Sample Benchmark Metrics:")
    print(f"  • Top-1 Accuracy: {top1_hits / 20 * 100:.2f}% ({top1_hits} / 20)")
    print(f"  • Top-3 Accuracy: {top3_hits / 20 * 100:.2f}% ({top3_hits} / 20)")
    print(f"  • Accepted Count (>= 0.70): {accepted_count} / 20")
    print(f"  • Rejected Count (< 0.70): {rejected_count} / 20")
    print(f"  • Unique Predicted Classes: {len(set(pred_classes))} / 20")
    print(f"  • Pred Class Distribution: {dict((c, pred_classes.count(c)) for c in set(pred_classes))}")

    # =========================================================================
    # STEP 2: Candidate Checkpoints Repository Scan
    # =========================================================================
    print("\n--- STEP 2: Scanning Repository for All Candidate Checkpoints ---")
    all_ckpts = list(repo_root.glob("**/*.pt")) + list(repo_root.glob("**/*.pth")) + list(repo_root.glob("**/*.bin")) + list(repo_root.glob("**/*.ckpt"))
    
    print(f"Found {len(all_ckpts)} total checkpoint file(s) across entire repository:")
    for c_file in all_ckpts:
        rel_path = c_file.relative_to(repo_root)
        c_size = c_file.stat().st_size / (1024 * 1024)
        print(f"\n  • Candidate: {rel_path} ({c_size:.2f} MB)")
        try:
            c_data = torch.load(c_file, map_location="cpu", weights_only=False)
            if isinstance(c_data, dict):
                ep = c_data.get("epoch", "N/A")
                n_cls = c_data.get("model_num_classes", c_data.get("num_classes", "N/A"))
                print(f"    - Epoch: {ep}, Num Classes: {n_cls}")
                c_state = c_data.get("state_dict", c_data.get("model_state_dict", c_data))
            else:
                c_state = c_data
            
            # Test loading into DynamicSignClassifier
            m_cand = DynamicSignClassifier.from_config()
            try:
                m_cand.load_state_dict(c_state, strict=True)
                m_cand.eval()
                print("    - Loads into DynamicSignClassifier (strict=True): YES")
                
                # Evaluate on 20 benchmark samples
                c_top1 = 0
                c_top3 = 0
                c_preds = []
                with torch.no_grad():
                    for idx in benchmark_indices:
                        item = val_dataset[idx]
                        t_c = int(item["label"])
                        d_f = item["dynamic"].unsqueeze(0).float()
                        d_m = item["dynamic_mask"].unsqueeze(0).bool() if "dynamic_mask" in item else None
                        logits_c = m_cand(d_f, mask=d_m).logits[0]
                        probs_c = torch.softmax(logits_c, dim=-1)
                        p_c = int(torch.argmax(probs_c).item())
                        t3_c = torch.topk(probs_c, k=3).indices.tolist()
                        c_preds.append(p_c)
                        if p_c == t_c:
                            c_top1 += 1
                        if t_c in t3_c:
                            c_top3 += 1
                print(f"    - Benchmark 20 Eval: Top-1 = {c_top1/20*100:.2f}%, Top-3 = {c_top3/20*100:.2f}%, Unique Preds = {len(set(c_preds))}")
            except Exception as le:
                print(f"    - DynamicSignClassifier load: NO ({le})")
        except Exception as err:
            print(f"    - Failed to load checkpoint: {err}")

    # =========================================================================
    # STEP 3: Failure Diagnostic Analysis
    # =========================================================================
    print("\n--- STEP 3: Deep Failure Diagnostic Analysis ---")
    # Evaluate logits distribution over benchmark
    all_logits = []
    all_probs = []
    with torch.no_grad():
        for idx in benchmark_indices:
            item = val_dataset[idx]
            d_f = item["dynamic"].unsqueeze(0).float()
            d_m = item["dynamic_mask"].unsqueeze(0).bool() if "dynamic_mask" in item else None
            l_raw = model(d_f, mask=d_m).logits[0]
            p_raw = torch.softmax(l_raw, dim=-1)
            all_logits.append(l_raw.numpy())
            all_probs.append(p_raw.numpy())

    all_logits_np = np.stack(all_logits)  # [20, 262]
    all_probs_np = np.stack(all_probs)    # [20, 262]

    print(f"Logits Statistics (20 samples):")
    print(f"  • Min Logit: {all_logits_np.min():.4f}")
    print(f"  • Max Logit: {all_logits_np.max():.4f}")
    print(f"  • Mean Logit: {all_logits_np.mean():.4f}")
    print(f"  • Std Logit: {all_logits_np.std():.4f}")

    print(f"Softmax Probabilities Statistics (20 samples):")
    print(f"  • Min Softmax Prob: {all_probs_np.min():.6f} (uniform 1/262 = {1/262:.6f})")
    print(f"  • Max Softmax Prob: {all_probs_np.max():.6f}")
    print(f"  • Mean Softmax Peak: {all_probs_np.max(axis=1).mean():.4f}")

    print(f"Saved Checkpoint Training Metadata:")
    print(f"  • Epoch: {ckpt_dict.get('epoch')}")
    print(f"  • Best Epoch: {ckpt_dict.get('best_epoch')}")
    print(f"  • Best Macro F1: {ckpt_dict.get('best_macro_f1')}")

    # =========================================================================
    # STEP 4: Dataset & Vocabulary Distribution Analysis
    # =========================================================================
    print("\n--- STEP 4: Clean INCLUDE Dataset & Vocabulary Distribution ---")
    if train_csv.exists():
        import pandas as pd
        df_train = pd.read_csv(train_csv)
        counts_per_class = df_train["label"].value_counts()
        
        print(f"Training Set Class Count Distribution ({len(df_train)} samples across {len(counts_per_class)} classes):")
        print(f"  • Min samples per class: {counts_per_class.min()}")
        print(f"  • Max samples per class: {counts_per_class.max()}")
        print(f"  • Median samples per class: {counts_per_class.median():.1f}")
        print(f"  • Mean samples per class: {counts_per_class.mean():.2f}")
        
        c_ge_5 = (counts_per_class >= 5).sum()
        c_ge_10 = (counts_per_class >= 10).sum()
        c_ge_15 = (counts_per_class >= 15).sum()
        c_ge_20 = (counts_per_class >= 20).sum()

        print(f"  • Classes with >= 5 train samples: {c_ge_5} / {len(counts_per_class)}")
        print(f"  • Classes with >= 10 train samples: {c_ge_10} / {len(counts_per_class)}")
        print(f"  • Classes with >= 15 train samples: {c_ge_15} / {len(counts_per_class)}")
        print(f"  • Classes with >= 20 train samples: {c_ge_20} / {len(counts_per_class)}")

    # =========================================================================
    # STEP 5: Text -> ISL Asset Format & Code Inspection
    # =========================================================================
    print("\n--- STEP 5: Text -> ISL Asset Format & Code Inspection ---")
    from src.nlp.isl_dictionary import ISLDictionaryIndex
    from src.video.isl_stitcher import ISLVideoStitcher
    
    print("Expected Asset Format & Schema:")
    print("  • Index File: JSON object with 'entries': list of dicts")
    print("  • Dict schema: {'word': str, 'video_path': str, 'category': str, 'attributes': dict}")
    print("  • Video File format: H.264 / AAC MP4 video clips, 30 FPS, preferred 640x480 resolution.")
    print("  • Video Directory: data/external/isl_dictionary/videos/")
    
    # Check if download scripts or configs exist in repo
    dl_scripts = list(repo_root.glob("**/download*.py")) + list(repo_root.glob("**/fetch*.py")) + list(repo_root.glob("**/pull*.py"))
    print(f"  • Download/Fetch scripts found in repo: {[str(s.relative_to(repo_root)) for s in dl_scripts]}")

    print("\n=" * 80)
    print("RECOVERY BENCHMARK & DIAGNOSTIC AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

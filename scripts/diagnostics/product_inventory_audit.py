"""SIGNASSIST Product Functionality Audit Script.

Performs empirical inspection of:
1. Model checkpoints on disk
2. Feature dataset & CSV splits
3. Text -> ISL dictionary index & actual MP4/video files on disk
4. End-to-end recognition test on real validation samples using RecognitionPipeline
5. Text -> ISL execution for Hello, Thank you, Yes, No, Good
6. TTS synthesis execution
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
from src.nlp.isl_dictionary import ISLDictionaryIndex
from src.nlp.text_to_isl import TextToISLConverter
from src.pipeline.recognition import RecognitionPipeline
from src.tts.tts_engine import TTSEngine
from src.utils.config import load_config
from src.video.isl_stitcher import ISLVideoStitcher


def main():
    print("=" * 80)
    print("SIGNASSIST PRODUCT FUNCTIONALITY & ASSET AUDIT")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models"
    data_dir = repo_root / "data"
    assets_dir = repo_root / "assets"

    # -------------------------------------------------------------------------
    # PART 1.1: Checkpoints Inventory
    # -------------------------------------------------------------------------
    print("\n--- PART 1.1: Model Checkpoints Inventory ---")
    ckpt_files = list(models_dir.glob("**/*.pt")) + list(models_dir.glob("**/*.pth"))
    print(f"Found {len(ckpt_files)} checkpoint file(s) in {models_dir}:")
    for f in ckpt_files:
        rel_p = f.relative_to(repo_root)
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"\n  • Checkpoint Path: {rel_p} ({size_mb:.2f} MB)")
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                print(f"    - Keys: {list(ckpt.keys())[:8]}")
                print(f"    - Epoch: {ckpt.get('epoch', 'N/A')}")
                print(f"    - Num Classes: {ckpt.get('model_num_classes', ckpt.get('num_classes', 'N/A'))}")
                state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
            else:
                state = ckpt
            
            # Test loading into DynamicSignClassifier
            from src.models.dynamic_classifier import DynamicSignClassifier
            m = DynamicSignClassifier.from_config()
            try:
                m.load_state_dict(state, strict=True)
                print("    - Loads successfully with strict=True: YES")
            except Exception as le:
                print(f"    - Loads with strict=True: NO ({le})")
        except Exception as err:
            print(f"    - Error loading checkpoint: {err}")

    # -------------------------------------------------------------------------
    # PART 1.2: Feature Dataset Inventory
    # -------------------------------------------------------------------------
    print("\n--- PART 1.2: Dataset & Preprocessed Feature Inventory ---")
    raw_vids = list(data_dir.glob("**/*.MOV")) + list(data_dir.glob("**/*.mp4")) + list(data_dir.glob("**/*.avi"))
    feat_files = list((data_dir / "processed" / "include" / "features").glob("*.npz"))
    
    val_csv = data_dir / "processed" / "include" / "splits" / "validation.csv"
    train_csv = data_dir / "processed" / "include" / "splits" / "train.csv"
    test_csv = data_dir / "processed" / "include" / "splits" / "test.csv"

    print(f"  • Raw video files on disk: {len(raw_vids)}")
    print(f"  • Preprocessed NPZ feature artifacts on disk: {len(feat_files)}")

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        import pandas as pd
        df_tr = pd.read_csv(train_csv)
        df_va = pd.read_csv(val_csv)
        df_te = pd.read_csv(test_csv)
        print(f"  • Split CSVs found:")
        print(f"    - Train: {len(df_tr)} samples")
        print(f"    - Validation: {len(df_va)} samples")
        print(f"    - Test: {len(df_te)} samples")
        print(f"    - Total Split Samples: {len(df_tr) + len(df_va) + len(df_te)}")

    mapping_path = data_dir / "processed" / "include" / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as mf:
            mdata = json.load(mf)
        print(f"  • INCLUDE Class Count: {mdata.get('num_classes')} classes in class_mapping.json")

    # -------------------------------------------------------------------------
    # PART 1.3: Text -> ISL Dictionary & Video Asset Inventory
    # -------------------------------------------------------------------------
    print("\n--- PART 1.3: Text -> ISL Dictionary & Video Asset Inventory ---")
    dict_candidates = [
        data_dir / "processed" / "isl_dictionary" / "index.json",
        data_dir / "isl_dictionary.json",
        assets_dir / "isl_dictionary.json",
        repo_root / "data" / "nlp" / "dictionary.json",
    ]
    
    found_dict = None
    for cand in dict_candidates:
        if cand.exists():
            found_dict = cand
            break
            
    if found_dict is None:
        # Search anywhere in data or assets
        all_json = list(data_dir.glob("**/*.json")) + list(assets_dir.glob("**/*.json"))
        for j in all_json:
            if "dict" in j.name.lower() or "index" in j.name.lower():
                found_dict = j
                break

    print(f"  • Dictionary Index File: {found_dict if found_dict else 'NONE FOUND'}")
    
    if found_dict and found_dict.exists():
        with open(found_dict, "r", encoding="utf-8") as df:
            dict_raw = json.load(df)
        
        entries = dict_raw.get("entries", dict_raw) if isinstance(dict_raw, dict) else dict_raw
        print(f"  • Total Dictionary Index Entries: {len(entries)}")

        video_dir = found_dict.parent / "videos"
        if not video_dir.exists():
            video_dir = data_dir / "isl_dictionary" / "videos"

        print(f"  • Video Asset Directory Search: {video_dir}")
        
        valid_video_count = 0
        missing_video_count = 0
        
        for entry in entries:
            word = entry.get("word", entry.get("gloss", ""))
            v_path = entry.get("video_path", entry.get("path", ""))
            
            # Check if video file actually exists
            real_v = None
            if v_path:
                p1 = Path(v_path)
                p2 = repo_root / v_path
                p3 = video_dir / Path(v_path).name
                if p1.exists():
                    real_v = p1
                elif p2.exists():
                    real_v = p2
                elif p3.exists():
                    real_v = p3

            if real_v and real_v.exists():
                valid_video_count += 1
            else:
                missing_video_count += 1

        print(f"  • Entries with ACTUAL VIDEO FILES ON DISK: {valid_video_count}")
        print(f"  • Entries MISSING VIDEO FILES ON DISK: {missing_video_count}")

        # Specifically check target words
        target_words = ["Hello", "Hi", "Thank you", "Yes", "No", "Good", "Bad", "Please", "Sorry"]
        print("\n  • Specific Target Word Asset Verification:")
        for tw in target_words:
            matched_entry = None
            for e in entries:
                w = e.get("word", e.get("gloss", "")).strip().lower()
                if w == tw.lower():
                    matched_entry = e
                    break
            
            if matched_entry:
                v_rel = matched_entry.get("video_path", matched_entry.get("path", ""))
                v_file_exists = False
                if v_rel:
                    p1 = Path(v_rel)
                    p2 = repo_root / v_rel
                    p3 = video_dir / Path(v_rel).name
                    v_file_exists = p1.exists() or p2.exists() or p3.exists()
                
                status_str = "✅ VIDEO ASSET EXISTS ON DISK" if v_file_exists else "❌ INDEXED BUT VIDEO FILE MISSING ON DISK"
                print(f"    - '{tw}': Match found in index ('{matched_entry.get('word')}'), {status_str}")
            else:
                print(f"    - '{tw}': ❌ NOT FOUND IN DICTIONARY INDEX")

    # -------------------------------------------------------------------------
    # PART 2: Real End-to-End Recognition Test on Validation Samples
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Real End-to-End Recognition Test on Validation Samples ---")
    pipeline = RecognitionPipeline(mode="dynamic_only")
    print(f"  • RecognitionPipeline(mode='dynamic_only') available: {pipeline.is_available()}")

    if pipeline.is_available() and val_csv.exists():
        val_ds = INCLUDEFeatureDataset(split_csv=val_csv, feature_root=data_dir / "processed" / "include" / "features")
        with open(mapping_path, "r", encoding="utf-8") as f:
            id_to_class = json.load(f).get("id_to_class", {})

        print("\n  Evaluating representative validation samples through production RecognitionPipeline...")
        
        # Test Sample 1 (Top-1 hit if any), Sample 2, Sample 3
        correct_samples = []
        incorrect_samples = []
        
        for i in range(len(val_ds)):
            item = val_ds[i]
            true_cls = int(item["label"])
            true_gloss = id_to_class.get(str(true_cls), f"Class {true_cls}")
            v_path = item["video_path"]

            d_feat = item["dynamic"].unsqueeze(0).float()
            d_mask = item["dynamic_mask"].unsqueeze(0).bool() if "dynamic_mask" in item else None

            with torch.no_grad():
                out = pipeline.dynamic_classifier(d_feat, mask=d_mask)
                conf_out = pipeline.confidence_estimator(out.logits)
                p_cls = int(conf_out.predicted_class[0].item())
                p_conf = float(conf_out.confidence[0].item())
                is_acc = bool(conf_out.is_accepted[0].item())
                p_gloss = pipeline.get_dynamic_label(p_cls)

            res_tuple = (v_path, true_gloss, p_gloss, p_conf, "Accepted" if is_acc else "Rejected")

            if p_cls == true_cls:
                correct_samples.append(res_tuple)
            else:
                incorrect_samples.append(res_tuple)

        print("\n  • SAMPLE RECOGNITION TEST RESULTS:")
        print("  A) Correct Prediction Example(s) (if any):")
        if correct_samples:
            for s in correct_samples[:3]:
                print(f"     Video: {s[0]}")
                print(f"     True Gloss: '{s[1]}'")
                print(f"     Predicted Gloss: '{s[2]}'")
                print(f"     Confidence: {s[3]:.4f} ({s[3]*100:.2f}%)")
                print(f"     Status: {s[4]}")
        else:
            print("     None found in sample evaluation.")

        print("\n  B) Incorrect Prediction Examples:")
        for s in incorrect_samples[:3]:
            print(f"     Video: {s[0]}")
            print(f"     True Gloss: '{s[1]}'")
            print(f"     Predicted Gloss: '{s[2]}'")
            print(f"     Confidence: {s[3]:.4f} ({s[3]*100:.2f}%)")
            print(f"     Status: {s[4]}")

    # -------------------------------------------------------------------------
    # PART 3: Real Text -> ISL Execution Test
    # -------------------------------------------------------------------------
    print("\n--- PART 3: Real Text -> ISL Execution Test ---")
    test_words = ["Hello", "Thank you", "Yes", "No", "Good"]

    dict_idx = None
    if found_dict and found_dict.exists():
        dict_idx = ISLDictionaryIndex.load(found_dict)

    converter = TextToISLConverter(dictionary=dict_idx)
    stitcher = ISLVideoStitcher()

    print("Running TextToISLConverter & ISLVideoStitcher on target words:")
    for w in test_words:
        print(f"\n  • Word: '{w}'")
        seq = converter.convert(w)
        d_count = seq.dictionary_count
        f_count = seq.fingerspell_count
        u_count = seq.unknown_count

        dict_match = (d_count > 0)
        fingerspell = (f_count > 0)
        
        # Check actual video asset
        has_video = False
        if seq.demonstrations:
            demo = seq.demonstrations[0]
            if demo.video_path and Path(demo.video_path).exists():
                has_video = True

        out_mp4 = repo_root / "scratch" / f"test_stitch_{w}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        stitch_res = None
        playable_output = False
        try:
            stitch_res = stitcher.stitch_sequence(seq, output_path=out_mp4)
            if stitch_res.total_clips > 0 and out_mp4.exists():
                playable_output = True
        except Exception as se:
            pass

        print(f"    - Dictionary Match: {'YES' if dict_match else 'NO'}")
        print(f"    - Actual Video Asset Exists: {'YES' if has_video else 'NO (BROKEN/UNAVAILABLE)'}")
        print(f"    - Fingerspelling Fallback Used: {'YES' if fingerspell else 'NO'}")
        print(f"    - Generated Sequence: {[d.token for d in seq.demonstrations]}")
        print(f"    - Stitched Video Created: {'YES' if (stitch_res and stitch_res.total_clips > 0) else 'NO'}")
        print(f"    - Playable Output File: {'YES (' + str(out_mp4) + ')' if playable_output else 'NO (UNAVAILABLE)'}")

    # -------------------------------------------------------------------------
    # PART 6: TTS Engine Direct Execution Test
    # -------------------------------------------------------------------------
    print("\n--- PART 6: TTS Engine Direct Execution Test ---")
    tts_engine = TTSEngine()
    print("Testing direct TTS synthesis from manually supplied text ('Hello welcome to SIGNASSIST'):")
    try:
        tts_out = tts_engine.synthesize("Hello welcome to SIGNASSIST")
        print(f"  • Direct TTS Engine Status: ✅ WORKING")
        print(f"  • Engine Used: {tts_out.engine_used}")
        print(f"  • Audio Duration: {tts_out.duration_sec:.2f} s")
        print(f"  • Audio Bytes Size: {len(tts_out.audio_bytes)} bytes")
        print(f"  • MIME Type: {tts_out.mime_type}")
    except Exception as te:
        print(f"  • Direct TTS Engine Status: ❌ FAILED ({te})")

    print("\n" + "=" * 80)
    print("PRODUCT FUNCTIONALITY AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

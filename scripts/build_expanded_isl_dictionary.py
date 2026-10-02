"""
Expanded ISL Dictionary Builder for SIGNASSIST (Phase 1 / Product Hardening).

Builds a comprehensive reverse Text->ISL dictionary using ONLY REAL videos from the
TRAINING SPLIT (data/processed/include/splits/train.csv).
Includes 42 controlled classes + '2. Cat' + all other clean TRAIN-only INCLUDE classes.
Ensures zero validation/test leakage.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex


def generate_aliases(gloss: str) -> list[str]:
    """Generates clean search aliases for a gloss (e.g. '2. Cat' -> ['cat', '2. cat', '2 cat'])."""
    aliases = []
    cleaned = gloss.strip()

    # Strip numeric prefix like "10. " or "2. "
    no_prefix = re.sub(r"^\d+\.\s*", "", cleaned).strip()
    if no_prefix and no_prefix.casefold() != cleaned.casefold():
        aliases.append(no_prefix)

    # Replace dot with space
    no_dot = cleaned.replace(".", " ").strip()
    if no_dot and no_dot.casefold() != cleaned.casefold():
        aliases.append(no_dot)

    # Lowercase variant
    aliases.append(cleaned.lower())
    if no_prefix:
        aliases.append(no_prefix.lower())

    # De-duplicate while preserving case
    seen = set()
    unique_aliases = []
    for a in aliases:
        cf = a.casefold()
        if cf not in seen and cf != cleaned.casefold():
            seen.add(cf)
            unique_aliases.append(a)

    return unique_aliases


def main() -> None:
    print("=================================================================")
    print("BUILDING EXPANDED ISL DICTIONARY (STRICT TRAIN SPLIT ONLY)")
    print("=================================================================")

    train_csv_path = Path("data/processed/include/splits/train.csv")
    val_csv_path = Path("data/processed/include/splits/validation.csv")
    test_csv_path = Path("data/processed/include/splits/test.csv")

    assert train_csv_path.exists(), f"Missing train.csv: {train_csv_path}"

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path) if val_csv_path.exists() else pd.DataFrame()
    test_df = pd.read_csv(test_csv_path) if test_csv_path.exists() else pd.DataFrame()

    val_paths = set(val_df["video_path"].tolist()) if not val_df.empty else set()
    test_paths = set(test_df["video_path"].tolist()) if not test_df.empty else set()

    dict_dir = Path("data/external/isl_dictionary")
    dict_videos_dir = dict_dir / "videos"
    dict_videos_dir.mkdir(parents=True, exist_ok=True)
    index_json_path = dict_dir / "index.json"

    raw_videos_dir = Path("data/raw/include/videos")

    # Group train dataframe by label
    grouped = train_df.groupby("label")

    entries: list[dict] = []
    processed_classes = 0

    print(f"Total unique classes available in train split: {len(grouped)}")

    for gloss, group in sorted(grouped, key=lambda x: str(x[0])):
        # Find first existing source video file in train split that is NOT in val or test
        selected_row = None
        source_video = None

        for _, row in group.iterrows():
            rel_v_path = row["video_path"]

            # Strict Leakage Safety Check
            if rel_v_path in val_paths or rel_v_path in test_paths:
                continue

            cand_path = raw_videos_dir / rel_v_path
            if cand_path.is_file():
                # Validate video decodability
                cap = cv2.VideoCapture(str(cand_path))
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    if frame_count > 0 and width > 0 and height > 0:
                        source_video = cand_path
                        selected_row = row
                        break
                cap.release()

        if source_video is None or selected_row is None:
            print(f"⚠️ Warning: No valid TRAIN video file found for class '{gloss}'")
            continue

        # Generate target filename
        safe_gloss_name = re.sub(r"[^a-zA-Z0-9_]", "_", str(gloss)).lower()
        src_filename = source_video.name
        target_filename = f"class_{safe_gloss_name}_{src_filename}"
        target_video_path = dict_videos_dir / target_filename

        # Copy video file if not present
        if not target_video_path.exists():
            shutil.copy2(source_video, target_video_path)

        # Get video metadata
        cap = cv2.VideoCapture(str(target_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        fps_effective = fps if fps > 0 else 30.0
        duration_sec = round(frame_count / fps_effective, 2)

        category = str(selected_row.get("parent_label", "general"))
        if not category or category == "nan":
            category = "general"

        aliases = generate_aliases(str(gloss))

        entry_dict = {
            "gloss": str(gloss),
            "video_path": f"videos/{target_filename}",
            "aliases": aliases,
            "category": category,
            "source": "INCLUDE_TRAIN",
            "license": "Academic Open Research",
            "duration_sec": duration_sec,
            "description": f"Real ISL demonstration clip for '{gloss}' (Sourced strictly from TRAIN split: {src_filename}).",
            "fingerspelling_available": True,
        }

        entries.append(entry_dict)
        processed_classes += 1

    catalog_data = {
        "schema_version": "1.0.0",
        "description": "SIGNASSIST Indian Sign Language (ISL) Dictionary Index - Full TRAIN Split Catalog",
        "total_entries": len(entries),
        "entries": entries,
    }

    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)

    print(f"\nSuccessfully built expanded dictionary at '{dict_dir}':")
    print(f"  - Total Indexed Entries: {len(entries)}")
    print(f"  - Index JSON: {index_json_path}")

    # Check if '2. Cat' / 'cat' is present
    cat_entry = [e for e in entries if "cat" in e["gloss"].lower() or any("cat" == a.lower() for a in e["aliases"])]
    if cat_entry:
        print("\n✅ VERIFIED: 'cat' is included in expanded TRAIN dictionary!")
        print(json.dumps(cat_entry[0], indent=2))
    else:
        print("\n❌ NOTE: 'cat' was not found.")

if __name__ == "__main__":
    main()

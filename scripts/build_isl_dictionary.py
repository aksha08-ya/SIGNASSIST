"""
Deterministic ISL Dictionary Asset Builder for SIGNASSIST (Phase 15 / Audit Fix).

Selects 1 real source video strictly from the TRAINING SPLIT (data/processed/include/controlled/train.csv)
for each of the 42 controlled vocabulary classes, validates the video clip via OpenCV,
copies the clip into data/external/isl_dictionary/videos/, and builds data/external/isl_dictionary/index.json.
"""

from pathlib import Path
import json
import shutil
import re
import cv2
import pandas as pd

from src.nlp.isl_dictionary import ISLDictionaryEntry, ISLDictionaryIndex


def generate_aliases(gloss: str) -> list[str]:
    """Generates deterministic clean search aliases for a class gloss (e.g. '10. Plane' -> ['plane', '10. plane', '10 plane'])."""
    aliases = []
    cleaned = gloss.strip()
    
    # Strip numeric prefix like "10. " or "5. "
    no_prefix = re.sub(r"^\d+\.\s*", "", cleaned).strip()
    if no_prefix and no_prefix.casefold() != cleaned.casefold():
        aliases.append(no_prefix)

    # Replace dot with space if present
    no_dot = cleaned.replace(".", " ").strip()
    if no_dot and no_dot.casefold() != cleaned.casefold():
        aliases.append(no_dot)

    return aliases


def main() -> None:
    print("=================================================================")
    print("SIGNASSIST ISL DICTIONARY ASSET BUILDER (TRAIN SPLIT ISOLATION)")
    print("=================================================================")

    train_csv_path = Path("data/processed/include/controlled/train.csv")
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Training split manifest missing at '{train_csv_path}'")

    train_df = pd.read_csv(train_csv_path)

    vocab_path = Path("data/processed/include/vocabulary_initial.json")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary mapping file missing at '{vocab_path}'")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    id_to_label = vocab.get("controlled_id_to_label", {})
    if not id_to_label:
        raise ValueError(f"No controlled_id_to_label dict found in '{vocab_path}'")

    dict_dir = Path("data/external/isl_dictionary")
    dict_videos_dir = dict_dir / "videos"
    dict_videos_dir.mkdir(parents=True, exist_ok=True)
    index_json_path = dict_dir / "index.json"

    entries: list[ISLDictionaryEntry] = []

    print(f"Building dictionary assets from TRAIN split for {len(id_to_label)} controlled classes...")

    for class_id_str, gloss in sorted(id_to_label.items(), key=lambda x: int(x[0])):
        class_id = int(class_id_str)
        
        # Filter training split manifest for this class ID
        class_train_df = train_df[train_df["controlled_class_id"] == class_id]
        if class_train_df.empty:
            # Try filtering by exact label string
            class_train_df = train_df[train_df["label"] == gloss]

        if class_train_df.empty:
            raise FileNotFoundError(f"No training split entries found for class '{gloss}' (ID {class_id}) in '{train_csv_path}'")

        # Find first existing source video file in train split
        source_video = None
        selected_row = None

        for _, row in class_train_df.iterrows():
            rel_v_path = row["video_path"]
            cand_path = Path("data/raw/include/videos") / rel_v_path
            if cand_path.is_file():
                source_video = cand_path
                selected_row = row
                break

        if source_video is None or selected_row is None:
            raise FileNotFoundError(f"No valid video file exists on disk for class '{gloss}' from training split manifest")

        category = selected_row.get("parent_label", "general")
        if not isinstance(category, str) or not category.strip():
            category = "general"

        # Validate video using OpenCV
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Corrupt or unreadable video file: '{source_video}'")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1.0
        cap.release()

        duration_sec = float(frame_count / fps)

        # Destination file path
        dest_filename = f"class_{class_id_str.zfill(3)}_{source_video.name}"
        dest_path = dict_videos_dir / dest_filename

        # Copy source video clip into dictionary videos folder
        shutil.copy2(source_video, dest_path)

        # Generate alias variations for lookup flexibility
        aliases = generate_aliases(gloss)

        entry = ISLDictionaryEntry(
            gloss=gloss,
            video_path=f"videos/{dest_filename}",
            aliases=aliases,
            category=category,
            source="INCLUDE_TRAIN",
            license="Academic Open Research",
            duration_sec=duration_sec,
            description=f"Controlled 42-class ISL demonstration clip for '{gloss}' (Sourced strictly from TRAIN split: {source_video.name}).",
            fingerspelling_available=True,
        )
        entries.append(entry)

    # Build and validate ISLDictionaryIndex
    index = ISLDictionaryIndex(entries=entries)

    # Serialize to index.json
    raw_entries = []
    for e in index.entries():
        raw_entries.append({
            "gloss": e.gloss,
            "video_path": e.video_path,
            "aliases": e.aliases,
            "category": e.category,
            "source": e.source,
            "license": e.license,
            "duration_sec": e.duration_sec,
            "description": e.description,
            "fingerspelling_available": e.fingerspelling_available,
        })

    out_data = {
        "schema_version": "1.0.0",
        "description": "SIGNASSIST Indian Sign Language (ISL) Dictionary Index - 42 Controlled Classes (STRICT TRAIN SPLIT ONLY)",
        "entries": raw_entries,
    }

    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print("=================================================================")
    print(f"SUCCESSFULLY BUILT ISL DICTIONARY ({len(entries)} TRAIN-SPLIT VALIDATED CLIPS)")
    print(f"Index saved to: {index_json_path}")
    print("=================================================================")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

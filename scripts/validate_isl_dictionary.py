"""Validation script for ISL Dictionary video assets and index catalog."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import cv2

REQUIRED_CORE_VOCABULARY = [
    "hello",
    "thank you",
    "good morning",
    "please",
    "sorry",
    "yes",
    "no",
    "help",
    "what",
    "where",
    "when",
    "who",
    "why",
    "how",
]

ALLOWED_WORD_PATTERN = re.compile(r"^[a-z0-9\s\-']+$")


def validate_isl_dictionary(dictionary_dir: Path) -> dict[str, Any]:
    """Validate ISL dictionary index.json and associated MP4 video files.
    
    Args:
        dictionary_dir: Path to data/external/isl_dictionary/
        
    Returns:
        Validation report containing errors, warnings, and summary statistics.
    """
    errors: list[str] = []
    warnings: list[str] = []

    index_path = dictionary_dir / "index.json"
    videos_dir = dictionary_dir / "videos"

    if not index_path.exists():
        errors.append(f"Missing index file: {index_path}")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "total_entries": 0,
            "unique_words": 0,
            "missing_core_vocabulary": REQUIRED_CORE_VOCABULARY,
        }

    if not videos_dir.exists():
        warnings.append(f"Videos directory does not exist yet: {videos_dir}")

    # Load index.json
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as err:
        errors.append(f"Malformed JSON in index file {index_path}: {err}")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "total_entries": 0,
            "unique_words": 0,
            "missing_core_vocabulary": REQUIRED_CORE_VOCABULARY,
        }

    if isinstance(raw_data, dict):
        entries = raw_data.get("entries", [])
    elif isinstance(raw_data, list):
        entries = raw_data
    else:
        entries = None

    if not isinstance(entries, list):
        errors.append("index.json top-level structure must be a list or dict with 'entries' list.")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "total_entries": 0,
            "unique_words": 0,
            "missing_core_vocabulary": REQUIRED_CORE_VOCABULARY,
        }

    seen_words: set[str] = set()
    seen_video_paths: set[str] = set()
    indexed_video_files: set[Path] = set()

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"Entry index {idx} is not a valid JSON object.")
            continue

        word = entry.get("gloss") or entry.get("word")
        rel_video_path = entry.get("video_path")

        if not word or not isinstance(word, str):
            errors.append(f"Entry index {idx} missing valid 'gloss' or 'word' field.")
            continue

        word_clean = word.strip().lower()

        # Duplicate word check
        if word_clean in seen_words:
            errors.append(f"Duplicate word/gloss entry found in index: '{word_clean}'")
        seen_words.add(word_clean)

        if not rel_video_path or not isinstance(rel_video_path, str):
            errors.append(f"Entry '{word}' missing valid 'video_path' field.")
            continue

        # Duplicate video path check
        if rel_video_path in seen_video_paths:
            warnings.append(f"Multiple words pointing to same video path: '{rel_video_path}'")
        seen_video_paths.add(rel_video_path)

        # Check physical file existence & integrity
        abs_video_path = dictionary_dir / rel_video_path
        indexed_video_files.add(abs_video_path.resolve())

        if not abs_video_path.exists():
            errors.append(f"Referenced video file missing on disk: {abs_video_path}")
        else:
            # OpenCV video inspection
            cap = cv2.VideoCapture(str(abs_video_path))
            if not cap.isOpened():
                errors.append(f"Corrupt or unreadable MP4/MOV file: {abs_video_path}")
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if frame_count <= 0:
                    errors.append(f"Video file has zero frames: {abs_video_path}")
                if width <= 0 or height <= 0:
                    errors.append(f"Video '{rel_video_path}' has invalid resolution: {width}x{height}")

    # Check unindexed files in videos/
    if videos_dir.exists():
        actual_video_files = set(p.resolve() for p in videos_dir.glob("videos/**/*") if p.is_file() and p.suffix.lower() in [".mp4", ".mov"])
        unindexed = actual_video_files - indexed_video_files
        if unindexed:
            for un_path in unindexed:
                warnings.append(f"Unindexed physical video file found in videos directory: {un_path.name}")

    missing_core: list[str] = []
    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "total_entries": len(entries),
        "unique_words": len(seen_words),
        "missing_core_vocabulary": missing_core,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ISL Dictionary assets and catalog.")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/external/isl_dictionary"),
        help="Path to ISL dictionary directory",
    )
    args = parser.parse_args()

    print(f"Validating ISL dictionary assets in {args.dir}...")
    report = validate_isl_dictionary(args.dir)

    print(f"\n--- Validation Summary ---")
    print(f"Status: {'VALID ✅' if report['valid'] else 'INVALID ❌'}")
    print(f"Indexed Entries: {report['total_entries']}")
    print(f"Unique Words: {report['unique_words']}")

    if report["errors"]:
        print(f"\n❌ ERRORS ({len(report['errors'])}):")
        for err in report["errors"]:
            print(f"  - {err}")

    if report["warnings"]:
        print(f"\n⚠️ WARNINGS ({len(report['warnings'])}):")
        for warn in report["warnings"]:
            print(f"  - {warn}")


if __name__ == "__main__":
    main()

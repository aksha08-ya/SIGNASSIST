"""Build a clean, auditable manifest from INCLUDE metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.include_manifest import IncludeRecord, write_manifest


DATA_DIR = Path("data/raw/include/data")
OUTPUT_DIR = Path("data/processed/include")


SPLITS = {
    "train-00000-of-00001.parquet": "train",
    "val-00000-of-00001.parquet": "validation",
    "test-00000-of-00001.parquet": "test",
}


def load_metadata() -> pd.DataFrame:
    frames = []

    for filename, split in SPLITS.items():
        path = DATA_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing INCLUDE metadata: {path}")

        frame = pd.read_parquet(path).copy()
        frame["original_split"] = split
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    required = {
        "parent_label",
        "label",
        "video_path",
        "include_50",
        "original_split",
    }

    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return combined


def build_records(frame: pd.DataFrame) -> tuple[list[IncludeRecord], list[IncludeRecord]]:
    # Remove exact duplicate metadata rows while preserving provenance.
    deduplicated = frame.drop_duplicates(
        subset=[
            "video_path",
            "label",
            "parent_label",
            "original_split",
            "include_50",
        ]
    ).copy()

    # A physical video path is considered clean only when it has one label
    # and one parent category across the complete metadata collection.
    grouped = deduplicated.groupby("video_path", sort=True)

    conflicting_paths = {
        video_path
        for video_path, group in grouped
        if group["label"].nunique() > 1
        or group["parent_label"].nunique() > 1
    }

    clean_records: list[IncludeRecord] = []
    quarantine_records: list[IncludeRecord] = []

    for video_path, group in grouped:
        target = quarantine_records if video_path in conflicting_paths else clean_records

        for _, row in group.sort_values("original_split").iterrows():
            target.append(
                IncludeRecord(
                    video_path=str(row["video_path"]),
                    label=str(row["label"]),
                    parent_label=str(row["parent_label"]),
                    original_split=str(row["original_split"]),
                    include_50=bool(row["include_50"]),
                    status="quarantined" if video_path in conflicting_paths else "clean",
                )
            )

    return clean_records, quarantine_records


def main() -> None:
    frame = load_metadata()
    clean_records, quarantine_records = build_records(frame)

    write_manifest(
        clean_records,
        OUTPUT_DIR / "manifest.csv",
    )
    write_manifest(
        quarantine_records,
        OUTPUT_DIR / "quarantine.csv",
    )

    clean_paths = {record.video_path for record in clean_records}
    quarantine_paths = {
        record.video_path for record in quarantine_records
    }

    print(f"Metadata rows: {len(frame)}")
    print(f"Clean unique videos: {len(clean_paths)}")
    print(f"Quarantined unique videos: {len(quarantine_paths)}")
    print(f"Clean classes: {len({record.label for record in clean_records})}")
    print(f"Quarantined classes: {len({record.label for record in quarantine_records})}")
    print(f"Manifest: {OUTPUT_DIR / 'manifest.csv'}")
    print(f"Quarantine: {OUTPUT_DIR / 'quarantine.csv'}")


if __name__ == "__main__":
    main()

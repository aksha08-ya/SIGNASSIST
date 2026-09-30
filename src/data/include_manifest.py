"""Canonical manifest utilities for the INCLUDE ISL dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


@dataclass(frozen=True)
class IncludeRecord:
    video_path: str
    label: str
    parent_label: str
    original_split: str
    include_50: bool
    status: str


MANIFEST_COLUMNS = (
    "video_path",
    "label",
    "parent_label",
    "original_split",
    "include_50",
    "status",
)


def write_manifest(records: list[IncludeRecord], output_path: Path) -> None:
    """Write canonical INCLUDE records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "video_path": record.video_path,
                    "label": record.label,
                    "parent_label": record.parent_label,
                    "original_split": record.original_split,
                    "include_50": str(record.include_50).lower(),
                    "status": record.status,
                }
            )

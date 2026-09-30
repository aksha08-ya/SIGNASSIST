import csv
from pathlib import Path

from src.data.build_include_manifest import build_records, load_metadata


def test_include_manifest_counts():
    frame = load_metadata()
    clean, quarantine = build_records(frame)

    clean_paths = {record.video_path for record in clean}
    quarantine_paths = {record.video_path for record in quarantine}

    assert len(clean_paths) == 4210
    assert len(quarantine_paths) == 47
    assert len(clean_paths & quarantine_paths) == 0
    assert len({record.label for record in clean}) == 262


def test_clean_records_have_single_label_per_video():
    frame = load_metadata()
    clean, _ = build_records(frame)

    labels_by_path = {}

    for record in clean:
        labels_by_path.setdefault(record.video_path, set()).add(record.label)

    assert all(len(labels) == 1 for labels in labels_by_path.values())


def test_quarantine_contains_conflicting_paths():
    frame = load_metadata()
    _, quarantine = build_records(frame)

    labels_by_path = {}
    parents_by_path = {}

    for record in quarantine:
        labels_by_path.setdefault(record.video_path, set()).add(record.label)
        parents_by_path.setdefault(record.video_path, set()).add(record.parent_label)

    assert labels_by_path

    for video_path in labels_by_path:
        assert (
            len(labels_by_path[video_path]) > 1
            or len(parents_by_path[video_path]) > 1
        )


def test_manifest_files_exist_and_have_expected_columns():
    manifest = Path("data/processed/include/manifest.csv")
    quarantine = Path("data/processed/include/quarantine.csv")

    assert manifest.exists()
    assert quarantine.exists()

    expected = {
        "video_path",
        "label",
        "parent_label",
        "original_split",
        "include_50",
        "status",
    }

    with manifest.open(encoding="utf-8") as handle:
        assert set(csv.DictReader(handle).fieldnames) == expected

    with quarantine.open(encoding="utf-8") as handle:
        assert set(csv.DictReader(handle).fieldnames) == expected

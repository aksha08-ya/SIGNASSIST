import json
from pathlib import Path
import pandas as pd
import pytest


def test_demo_assets_manifest_integrity():
    """Verify that demo_assets/index.json exists, contains valid metadata,

    and strictly uses videos from the TRAIN split with zero test/validation leakage.
    """
    manifest_path = Path("data/processed/include/demo_assets/index.json")
    assert manifest_path.exists(), f"Demo asset manifest missing at '{manifest_path}'"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assets = data.get("demo_assets", [])
    assert len(assets) > 0, "Demo asset manifest is empty"

    train_df = pd.read_csv("data/processed/include/controlled/train.csv")
    val_df = pd.read_csv("data/processed/include/controlled/validation.csv")
    test_df = pd.read_csv("data/processed/include/controlled/test.csv")

    train_videos = set(train_df["video_path"])
    val_videos = set(val_df["video_path"])
    test_videos = set(test_df["video_path"])

    for asset in assets:
        assert asset["split"] == "train", f"Asset {asset['label']} split is {asset['split']}, expected 'train'"
        assert "checksum" in asset and len(asset["checksum"]) == 64, f"Invalid SHA256 checksum in asset {asset['label']}"
        assert asset["duration_seconds"] > 0, f"Invalid duration in asset {asset['label']}"
        assert asset["fps"] > 0, f"Invalid FPS in asset {asset['label']}"
        assert "quality_rating" in asset, f"Missing quality_rating in asset {asset['label']}"

        video_file = Path("data/processed/include") / asset["demo_video_path"]
        assert video_file.exists(), f"Demo video file missing on disk: '{video_file}'"

        orig_raw = asset.get("source_raw_path", "")
        for v_path in val_videos:
            assert Path(v_path).name not in orig_raw, f"Validation video leak in demo asset {asset['label']}: {orig_raw}"
        for v_path in test_videos:
            assert Path(v_path).name not in orig_raw, f"Test video leak in demo asset {asset['label']}: {orig_raw}"

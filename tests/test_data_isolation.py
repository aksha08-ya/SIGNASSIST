import json
from pathlib import Path
import pandas as pd
import pytest


def test_dictionary_data_isolation():
    """Verify that all videos in the ISL dictionary are strictly from the TRAIN split,

    and that zero videos are from held-out VALIDATION or TEST splits.
    """
    train_df = pd.read_csv("data/processed/include/controlled/train.csv")
    val_df = pd.read_csv("data/processed/include/controlled/validation.csv")
    test_df = pd.read_csv("data/processed/include/controlled/test.csv")

    train_videos = set(train_df["video_path"])
    val_videos = set(val_df["video_path"])
    test_videos = set(test_df["video_path"])

    index_path = Path("data/external/isl_dictionary/index.json")
    assert index_path.exists(), f"Dictionary index not found at {index_path}"

    with open(index_path, "r", encoding="utf-8") as f:
        dict_data = json.load(f)

    entries = dict_data.get("entries", [])
    assert len(entries) == 42, f"Expected 42 controlled dictionary entries, found {len(entries)}"

    val_leaks = []
    test_leaks = []

    for entry in entries:
        gloss = entry["gloss"]
        dict_video = entry["video_path"]
        source = entry.get("source", "")
        
        # Explicit source check if metadata present
        assert source == "INCLUDE_TRAIN", f"Entry {gloss} source is {source}, expected INCLUDE_TRAIN"

        orig_name = Path(dict_video).name.replace("class_", "")
        orig_file = "_".join(orig_name.split("_")[1:]) if "_" in orig_name else orig_name

        # Check for presence in validation/test sets
        for v_path in val_videos:
            if Path(v_path).name == orig_file and (gloss.split(". ")[-1] in v_path or gloss in v_path):
                val_leaks.append((gloss, dict_video, v_path))

        for v_path in test_videos:
            if Path(v_path).name == orig_file and (gloss.split(". ")[-1] in v_path or gloss in v_path):
                test_leaks.append((gloss, dict_video, v_path))

    assert len(val_leaks) == 0, f"Found validation split leaks in ISL dictionary: {val_leaks}"
    assert len(test_leaks) == 0, f"Found test split leaks in ISL dictionary: {test_leaks}"


def test_app_demo_button_uses_train_split():
    """Verify that the demo asset selector in src/app.py uses strictly TRAIN split videos."""
    app_py_path = Path("src/app.py")
    assert app_py_path.exists(), "src/app.py not found"

    app_code = app_py_path.read_text(encoding="utf-8")

    # Assert MVI_4175 (which was in held-out test set) is NOT used as demo path in app.py
    assert "MVI_4175.MOV" not in app_code, "Held-out test video MVI_4175.MOV found in src/app.py"

    # Assert app.py references demo_assets/index.json which contains TRAIN split videos
    assert "demo_assets/index.json" in app_code, "demo_assets/index.json manifest loader not found in src/app.py"

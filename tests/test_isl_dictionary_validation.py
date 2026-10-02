"""Tests for ISL dictionary asset validation module."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.validate_isl_dictionary import validate_isl_dictionary


def test_validate_missing_dictionary_dir(tmp_path: Path) -> None:
    report = validate_isl_dictionary(tmp_path / "non_existent")
    assert report["valid"] is False
    assert len(report["errors"]) > 0
    assert report["unique_words"] == 0


def test_validate_malformed_index_json(tmp_path: Path) -> None:
    idx_file = tmp_path / "index.json"
    idx_file.write_text("invalid json content", encoding="utf-8")

    report = validate_isl_dictionary(tmp_path)
    assert report["valid"] is False
    assert any("Malformed JSON" in err for err in report["errors"])


def test_validate_valid_schema_missing_video(tmp_path: Path) -> None:
    idx_file = tmp_path / "index.json"
    entries = [
        {
            "word": "hello",
            "video_path": "videos/hello.mp4",
        }
    ]
    idx_file.write_text(json.dumps(entries), encoding="utf-8")

    report = validate_isl_dictionary(tmp_path)
    assert report["valid"] is False
    assert any("missing on disk" in err for err in report["errors"])

"""PyTorch datasets for preprocessed INCLUDE features."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EXPECTED_DYNAMIC_SHAPE = (1, 5, 60, 115)
EXPECTED_STATIC_SHAPE = (225,)
EXPECTED_DYNAMIC_MASK_SHAPE = (1, 60, 115)
EXPECTED_STATIC_MASK_SHAPE = (75,)


def artifact_id_from_video_path(video_path: str) -> str:
    """Return the canonical 24-character feature artifact ID."""
    return hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:24]


class INCLUDEFeatureDataset(Dataset):
    """Dataset backed by precomputed INCLUDE feature artifacts."""

    def __init__(
        self,
        split_csv: str | Path,
        feature_root: str | Path = "data/processed/include/features",
        include_static: bool = False,
        strict: bool = True,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.feature_root = Path(feature_root)
        self.include_static = include_static
        self.strict = strict

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")

        if not self.feature_root.exists():
            raise FileNotFoundError(
                f"Feature directory not found: {self.feature_root}"
            )

        self.frame = pd.read_csv(self.split_csv)

        required = {"video_path", "label", "split"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(
                f"Split CSV missing required columns: {sorted(missing)}"
            )

        self.frame = self.frame.reset_index(drop=True)

        if self.frame["video_path"].duplicated().any():
            raise ValueError("Duplicate video_path values found in split.")

        self.artifacts: list[Path] = []
        missing_artifacts: list[str] = []

        for video_path in self.frame["video_path"]:
            artifact = self.feature_root / (
                artifact_id_from_video_path(str(video_path)) + ".npz"
            )
            if not artifact.exists():
                missing_artifacts.append(str(video_path))
            self.artifacts.append(artifact)

        if missing_artifacts and self.strict:
            raise FileNotFoundError(
                f"{len(missing_artifacts)} feature artifacts are missing. "
                f"First examples: {missing_artifacts[:5]}"
            )

        self.labels = self._load_labels()

    def _load_labels(self) -> np.ndarray:
        """Load label IDs from manifest column 'controlled_class_id' if available, else from feature artifacts."""
        if "controlled_class_id" in self.frame.columns:
            return self.frame["controlled_class_id"].to_numpy(dtype=np.int64)

        labels = []
        for artifact in self.artifacts:
            if not artifact.exists():
                labels.append(-1)
                continue

            with np.load(artifact, allow_pickle=False) as data:
                if "label_id" not in data:
                    raise ValueError(f"Missing label_id in {artifact}")
                labels.append(int(data["label_id"]))

        return np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        artifact = self.artifacts[index]

        if not artifact.exists():
            raise FileNotFoundError(f"Missing feature artifact: {artifact}")

        with np.load(artifact, allow_pickle=False) as data:
            required = {
                "dynamic_tensor",
                "dynamic_mask",
                "static_features",
                "static_mask",
                "label_id",
            }
            missing = required - set(data.files)
            if missing:
                raise ValueError(
                    f"Artifact {artifact.name} missing keys: {sorted(missing)}"
                )

            dynamic = data["dynamic_tensor"]
            dynamic_mask = data["dynamic_mask"]
            static = data["static_features"]
            static_mask = data["static_mask"]
            npz_label_id = int(data["label_id"])

        if dynamic.shape != EXPECTED_DYNAMIC_SHAPE:
            raise ValueError(
                f"{artifact.name}: dynamic shape {dynamic.shape}, "
                f"expected {EXPECTED_DYNAMIC_SHAPE}"
            )

        if dynamic_mask.shape != EXPECTED_DYNAMIC_MASK_SHAPE:
            raise ValueError(
                f"{artifact.name}: dynamic mask shape {dynamic_mask.shape}, "
                f"expected {EXPECTED_DYNAMIC_MASK_SHAPE}"
            )

        if static.shape != EXPECTED_STATIC_SHAPE:
            raise ValueError(
                f"{artifact.name}: static shape {static.shape}, "
                f"expected {EXPECTED_STATIC_SHAPE}"
            )

        if static_mask.shape != EXPECTED_STATIC_MASK_SHAPE:
            raise ValueError(
                f"{artifact.name}: static mask shape {static_mask.shape}, "
                f"expected {EXPECTED_STATIC_MASK_SHAPE}"
            )

        if not np.isfinite(dynamic).all():
            raise ValueError(f"{artifact.name}: dynamic tensor contains NaN/Inf")

        if not np.isfinite(static).all():
            raise ValueError(f"{artifact.name}: static features contain NaN/Inf")

        target_label = int(self.labels[index])
        if "controlled_class_id" not in self.frame.columns:
            if npz_label_id != target_label:
                raise ValueError(
                    f"{artifact.name}: inconsistent label_id "
                    f"{npz_label_id} != {target_label}"
                )

        sample: dict[str, Any] = {
            "dynamic": torch.from_numpy(dynamic[0].astype(np.float32)),
            "dynamic_mask": torch.from_numpy(dynamic_mask[0].astype(bool)),
            "label": torch.tensor(target_label, dtype=torch.long),
            "video_path": str(self.frame.iloc[index]["video_path"]),
        }

        if self.include_static:
            sample["static"] = torch.from_numpy(static.astype(np.float32))
            sample["static_mask"] = torch.from_numpy(static_mask.astype(bool))

        return sample

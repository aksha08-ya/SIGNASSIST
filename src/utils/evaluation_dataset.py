"""Research dataset abstraction and Leave-One-Signer-Out (LOSO) split engine for SIGNASSIST Phase 16.

Provides model-agnostic EvaluationSample, EvaluationDataset, and LOSOFold abstractions
for static and dynamic sign evaluation without third-party ML framework dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set
import numpy as np


@dataclass
class EvaluationSample:
    """Dataclass representing a single evaluation sample in a research dataset.

    Attributes:
        sample_id: Non-empty unique string identifier for the sample.
        label: Non-negative integer target class label (0, 1, 2, ...).
        signer_id: Non-empty string identifier for the signer / subject.
        features: NumPy ndarray containing static or dynamic feature representations.
        metadata: Optional dictionary containing arbitrary key-value metadata.
    """

    sample_id: str
    label: int
    signer_id: str
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        self.sample_id = self.sample_id.strip()

        if not isinstance(self.signer_id, str) or not self.signer_id.strip():
            raise ValueError("signer_id must be a non-empty string")
        self.signer_id = self.signer_id.strip()

        if isinstance(self.label, (float, np.floating)):
            if not float(self.label).is_integer():
                raise ValueError("label must be an integer or integer-valued float")
            self.label = int(self.label)
        elif not isinstance(self.label, (int, np.integer)):
            raise ValueError("label must be a non-negative integer")

        self.label = int(self.label)
        if self.label < 0:
            raise ValueError(f"label must be a non-negative integer, got {self.label}")

        if not isinstance(self.features, np.ndarray):
            raise TypeError(f"features must be a NumPy ndarray, got {type(self.features)}")

        if self.metadata is None:
            self.metadata = {}
        elif not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)


@dataclass
class LOSOFold:
    """Dataclass representing a Leave-One-Signer-Out (LOSO) cross-validation fold.

    Attributes:
        held_out_signer: Signer ID isolated in the test set.
        train_indices: 0-indexed integer list of training set sample indices.
        test_indices: 0-indexed integer list of test set sample indices.
        train_samples: List of EvaluationSample objects in the training set.
        test_samples: List of EvaluationSample objects in the test set.
    """

    held_out_signer: str
    train_indices: List[int]
    test_indices: List[int]
    train_samples: List[EvaluationSample] = field(default_factory=list)
    test_samples: List[EvaluationSample] = field(default_factory=list)


class EvaluationDataset:
    """Model-agnostic research dataset container supporting Leave-One-Signer-Out (LOSO) splits."""

    def __init__(self, samples: Sequence[EvaluationSample]) -> None:
        if not samples:
            raise ValueError("EvaluationDataset samples sequence cannot be empty")

        validated_samples: List[EvaluationSample] = []
        seen_sample_ids: Set[str] = set()

        for idx, s in enumerate(samples):
            if not isinstance(s, EvaluationSample):
                raise TypeError(
                    f"Sample at index {idx} is not an EvaluationSample instance: {type(s)}"
                )
            if s.sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id detected: '{s.sample_id}'")
            seen_sample_ids.add(s.sample_id)
            validated_samples.append(s)

        self._samples = validated_samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> EvaluationSample:
        return self._samples[idx]

    @property
    def samples(self) -> List[EvaluationSample]:
        """Returns ordered list of all samples in the dataset."""
        return list(self._samples)

    @property
    def labels(self) -> List[int]:
        """Returns list of integer labels for all samples in insertion order."""
        return [s.label for s in self._samples]

    def labels_array(self) -> np.ndarray:
        """Returns 1D int64 NumPy array of labels for all samples."""
        return np.array(self.labels, dtype=np.int64)

    @property
    def signer_ids(self) -> List[str]:
        """Returns list of signer IDs for all samples in insertion order."""
        return [s.signer_id for s in self._samples]

    def signer_ids_list(self) -> List[str]:
        """Returns list of signer IDs for all samples in insertion order."""
        return self.signer_ids

    def sample_ids_list(self) -> List[str]:
        """Returns list of sample IDs for all samples in insertion order."""
        return [s.sample_id for s in self._samples]

    def features_list(self) -> List[np.ndarray]:
        """Returns list of NumPy feature ndarrays for all samples."""
        return [s.features for s in self._samples]

    @property
    def unique_signers(self) -> List[str]:
        """Returns sorted list of unique signer IDs in deterministic alphabetical order."""
        return sorted(list(set(self.signer_ids)))

    def get_loso_folds(self) -> List[LOSOFold]:
        """Generates Leave-One-Signer-Out (LOSO) cross-validation folds.

        One fold is generated for each unique signer in unique_signers.
        The test set contains ONLY the held-out signer's samples;
        the training set contains ALL other signers' samples.

        Returns:
            List of LOSOFold objects in deterministic order of unique_signers.

        Raises:
            ValueError: If dataset contains fewer than 2 unique signers.
        """
        unique_signers_list = self.unique_signers
        if len(unique_signers_list) < 2:
            raise ValueError(
                f"LOSO split requires at least 2 unique signers, found {len(unique_signers_list)}"
            )

        folds: List[LOSOFold] = []
        for held_out_signer in unique_signers_list:
            train_indices: List[int] = []
            test_indices: List[int] = []
            train_samples: List[EvaluationSample] = []
            test_samples: List[EvaluationSample] = []

            for idx, sample in enumerate(self._samples):
                if sample.signer_id == held_out_signer:
                    test_indices.append(idx)
                    test_samples.append(sample)
                else:
                    train_indices.append(idx)
                    train_samples.append(sample)

            fold = LOSOFold(
                held_out_signer=held_out_signer,
                train_indices=train_indices,
                test_indices=test_indices,
                train_samples=train_samples,
                test_samples=test_samples,
            )
            folds.append(fold)

        return folds

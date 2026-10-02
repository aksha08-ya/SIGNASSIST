"""Script to generate controlled vocabulary and filtered train/validation/test manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.vocabulary import (
    filter_split_manifest,
    save_vocabulary,
    select_vocabulary_from_train,
    verify_controlled_manifests,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled vocabulary and manifests.")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/processed/include/splits/train.csv"),
        help="Path to training manifest",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/processed/include/splits/validation.csv"),
        help="Path to validation manifest",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/processed/include/splits/test.csv"),
        help="Path to test manifest",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        default=Path("data/processed/include/class_mapping.json"),
        help="Path to full class mapping JSON",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=15,
        help="Minimum training sample threshold per class",
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("data/processed/include/vocabulary_initial.json"),
        help="Output JSON path for vocabulary",
    )
    parser.add_argument(
        "--controlled-dir",
        type=Path,
        default=Path("data/processed/include/controlled"),
        help="Output directory for controlled manifests",
    )

    args = parser.parse_args()

    print(f"Selecting controlled vocabulary from {args.train_csv} (min_samples >= {args.min_samples})...")
    vocab_data = select_vocabulary_from_train(
        train_csv_path=args.train_csv,
        min_samples=args.min_samples,
        class_mapping_path=args.class_mapping,
    )

    save_vocabulary(vocab_data, args.vocab_out)
    print(f"Saved vocabulary ({vocab_data['num_controlled_classes']} classes) to {args.vocab_out}")

    print("Filtering manifests using selected TRAIN vocabulary...")
    filter_split_manifest(args.train_csv, vocab_data, args.controlled_dir / "train.csv")
    filter_split_manifest(args.val_csv, vocab_data, args.controlled_dir / "validation.csv")
    filter_split_manifest(args.test_csv, vocab_data, args.controlled_dir / "test.csv")

    print("Verifying controlled manifests...")
    stats = verify_controlled_manifests(args.controlled_dir, vocab_data)

    print("\n--- Controlled Manifest Generation Complete ---")
    print(f"Classes: {stats['num_controlled_classes']}")
    print(f"Train samples: {stats['train_samples']}")
    print(f"Validation samples: {stats['validation_samples']}")
    print(f"Test samples: {stats['test_samples']}")
    print(f"Total samples: {stats['total_controlled_samples']}")
    print(f"Train class balance (min/max/mean): {stats['train_class_counts']['min']} / {stats['train_class_counts']['max']} / {stats['train_class_counts']['mean']:.2f}")
    print(f"Leakage free: {stats['leakage_free']}")
    print(f"Quarantine free: {stats['quarantine_free']}")


if __name__ == "__main__":
    main()

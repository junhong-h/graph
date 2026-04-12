"""Import Locomo dataset into Graphmemory's ProcessedDatasetStore.

Usage:
    python scripts/import_locomo.py \
        --data data/locomo/locomo10.json \
        --repo runs/repo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from graphmemory.dataset import load_locomo
from graphmemory.storage.dataset_store import ProcessedDatasetStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to locomo10.json")
    parser.add_argument("--repo", required=True, help="Repository root directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    store = ProcessedDatasetStore(args.repo)

    # Register raw file (copy-once; skip if already registered)
    raw_dest = Path(args.repo) / "raw" / "locomo" / data_path.name
    if not raw_dest.exists():
        dest = store.register_raw_file("locomo", data_path)
        print(f"Registered raw file: {dest}")
    else:
        print(f"Raw file already registered, skipping copy.")

    # Load and import processed samples
    print("Loading Locomo samples...")
    samples = list(load_locomo(data_path))
    print(f"  {len(samples)} turns loaded")

    manifest = store.import_processed_samples(
        dataset_name="locomo",
        samples=samples,
        processing_script="scripts/import_locomo.py",
    )
    print(f"Imported version: {manifest['version']}")
    print(f"Sample count: {manifest['sample_count']}")
    print(f"Stored at: {Path(args.repo) / 'processed' / 'locomo' / manifest['version']}")


if __name__ == "__main__":
    main()

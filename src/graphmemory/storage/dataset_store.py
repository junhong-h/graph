from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pyarrow as pa

from graphmemory.models import ProcessedSample
from graphmemory.naming import iso_now, make_version_id, validate_snake_case
from graphmemory.storage.layout import RepositoryLayout
from graphmemory.storage.parquet_io import write_json, write_rows

PROCESSED_SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("text", pa.string()),
        ("timestamp", pa.string()),
        ("source_doc_id", pa.string()),
        ("speaker", pa.string()),
        ("metadata_json", pa.string()),
    ]
)


class ProcessedDatasetStore:
    def __init__(self, root: str | Path):
        self.layout = RepositoryLayout(root)
        self.layout.ensure_scaffold()

    def register_raw_file(
        self,
        dataset_name: str,
        source_path: str | Path,
        destination_name: str | None = None,
    ) -> Path:
        validate_snake_case(dataset_name, "dataset_name")
        source = Path(source_path).resolve()
        if not source.exists():
            raise FileNotFoundError(source)

        raw_dir = self.layout.raw_dataset_dir(dataset_name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        destination = raw_dir / (destination_name or source.name)
        if destination.exists():
            raise FileExistsError(f"raw file already exists: {destination}")
        shutil.copy2(source, destination)
        return destination

    def import_processed_samples(
        self,
        dataset_name: str,
        samples: Iterable[ProcessedSample],
        processing_script: str,
        field_descriptions: dict[str, str] | None = None,
        version: str | None = None,
    ) -> dict[str, object]:
        validate_snake_case(dataset_name, "dataset_name")
        version = version or make_version_id()
        version_dir = self.layout.processed_dataset_dir(dataset_name, version)
        if version_dir.exists():
            raise FileExistsError(f"processed dataset version already exists: {version_dir}")

        rows = [sample.to_row() for sample in samples]
        write_rows(version_dir / "samples.parquet", rows, PROCESSED_SAMPLE_SCHEMA)
        manifest = {
            "dataset_name": dataset_name,
            "version": version,
            "created_at": iso_now(),
            "processing_script": processing_script,
            "sample_count": len(rows),
            "fields": field_descriptions
            or {
                "sample_id": "Stable processed sample id.",
                "text": "Normalized sample text.",
                "timestamp": "Original event or utterance timestamp.",
                "source_doc_id": "Source document identifier.",
                "speaker": "Speaker or author name.",
                "metadata_json": "JSON string with extra dataset metadata.",
            },
        }
        write_json(version_dir / "manifest.json", manifest)
        return manifest

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa

from graphmemory.constants import SCHEMA_VERSION
from graphmemory.models import RawFallbackHit, RawMemoryChunk, RawMemoryRecord
from graphmemory.naming import iso_now, make_version_id, validate_snake_case
from graphmemory.storage.layout import RepositoryLayout
from graphmemory.storage.parquet_io import read_json, read_rows, write_json, write_rows

RAW_RECORD_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("dataset_name", pa.string()),
        ("source_doc_id", pa.string()),
        ("turn_id", pa.string()),
        ("text", pa.string()),
        ("timestamp", pa.string()),
        ("metadata_json", pa.string()),
        ("processed_sample_id", pa.string()),
    ]
)

RAW_CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("record_id", pa.string()),
        ("chunk_text", pa.string()),
        ("chunk_order", pa.int64()),
        ("token_count", pa.int64()),
        ("embedding_ref", pa.string()),
    ]
)

RAW_FALLBACK_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("record_id", pa.string()),
        ("chunk_id", pa.string()),
        ("hit_time", pa.string()),
        ("reason", pa.string()),
        ("promote_candidate_bool", pa.bool_()),
    ]
)


class RawMemoryStore:
    def __init__(self, root: str | Path):
        self.layout = RepositoryLayout(root)
        self.layout.ensure_scaffold()

    def write_snapshot(
        self,
        memory_name: str,
        dataset_name: str,
        records: Iterable[RawMemoryRecord],
        chunks: Iterable[RawMemoryChunk],
        fallback_hits: Iterable[RawFallbackHit] | None = None,
        version: str | None = None,
        base_version: str | None = None,
    ) -> dict[str, object]:
        validate_snake_case(memory_name, "memory_name")
        validate_snake_case(dataset_name, "dataset_name")
        version = version or make_version_id()
        version_dir = self.layout.raw_memory_dir(memory_name, version)
        if version_dir.exists():
            raise FileExistsError(f"raw memory version already exists: {version_dir}")

        base_records: list[dict[str, object]] = []
        base_chunks: list[dict[str, object]] = []
        base_hits: list[dict[str, object]] = []
        dataset_names = {dataset_name}
        if base_version:
            base_dir = self.layout.raw_memory_dir(memory_name, base_version)
            if not base_dir.exists():
                raise FileNotFoundError(base_dir)
            base_records = read_rows(base_dir / "records.parquet")
            base_chunks = read_rows(base_dir / "chunks.parquet")
            base_hits = read_rows(base_dir / "fallback_hits.parquet")
            base_meta = read_json(base_dir / "meta.json")
            dataset_names.update(base_meta.get("dataset_names", []))

        record_rows = [*base_records, *(record.to_row() for record in records)]
        chunk_rows = [*base_chunks, *(chunk.to_row() for chunk in chunks)]
        fallback_rows = [*base_hits, *((hit.to_row() for hit in fallback_hits) if fallback_hits else [])]

        write_rows(version_dir / "records.parquet", record_rows, RAW_RECORD_SCHEMA)
        write_rows(version_dir / "chunks.parquet", chunk_rows, RAW_CHUNK_SCHEMA)
        write_rows(version_dir / "fallback_hits.parquet", fallback_rows, RAW_FALLBACK_SCHEMA)
        meta = {
            "memory_name": memory_name,
            "version": version,
            "created_at": iso_now(),
            "schema_version": SCHEMA_VERSION,
            "dataset_names": sorted(dataset_names),
            "base_version": base_version,
        }
        write_json(version_dir / "meta.json", meta)
        return meta

    def top_fallback_records(
        self,
        memory_name: str,
        version: str,
        limit: int = 10,
    ) -> list[dict[str, object]]:
        validate_snake_case(memory_name, "memory_name")
        version_dir = self.layout.raw_memory_dir(memory_name, version)
        hits = read_rows(version_dir / "fallback_hits.parquet")
        counts: dict[tuple[str, str], int] = {}
        for hit in hits:
            key = (str(hit["record_id"]), str(hit["chunk_id"]))
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        return [
            {"record_id": record_id, "chunk_id": chunk_id, "hit_count": hit_count}
            for (record_id, chunk_id), hit_count in ranked[:limit]
        ]

    def resolve_chunk(
        self,
        memory_name: str,
        version: str,
        chunk_id: str,
    ) -> dict[str, object] | None:
        version_dir = self.layout.raw_memory_dir(memory_name, version)
        chunks = read_rows(version_dir / "chunks.parquet")
        chunk = next((row for row in chunks if row["chunk_id"] == chunk_id), None)
        if chunk is None:
            return None
        records = read_rows(version_dir / "records.parquet")
        record = next((row for row in records if row["record_id"] == chunk["record_id"]), None)
        return {"record": record, "chunk": chunk}

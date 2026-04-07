# GraphMemory

GraphMemory is a research-oriented implementation scaffold for storage-first memory experiments.

This repository currently focuses on:

- immutable graph snapshots stored as Parquet
- raw memory snapshots stored as Parquet
- run-scoped artifacts and intermediate outputs
- optional DuckDB indexes rebuilt from Parquet facts

## Layout

```text
configs/
data/raw/<dataset_name>/
data/processed/<dataset_name>/<version>/
artifacts/raw_memory/<memory_name>/<version>/
artifacts/graphs/<graph_name>/<version>/
artifacts/indexes/<index_name>/<version>/
runs/<run_id>/
src/
tests/
```

## Quick start

```bash
python3 -m pip install -e '.[dev]'
pytest
```

## Design rules

- Parquet is the source of truth.
- DuckDB is a rebuildable query layer, never the only durable state.
- Graph snapshots are immutable and versioned.
- Run artifacts are isolated by `run_id`.

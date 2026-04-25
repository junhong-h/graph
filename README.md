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

## Local Qwen3-4B via Ollama

This repo can run against a local Ollama endpoint without changing the
OpenAI-style client code.

```bash
brew install ollama
brew services start ollama
ollama pull qwen3:4b
ollama create graphmemory-qwen3:4b-32k -f configs/graphmemory-qwen3-4b-32k.Modelfile
```

The local Qwen configs also force non-thinking mode so Qwen3 does not inject
`<think>` content into JSON-style outputs used by graph construction and judge
evaluation.

Build memory with the local model:

```bash
python scripts/build_memory.py --config configs/build_memory_qwen3_local.yaml
```

Run QA and local LLM judge with the local model:

```bash
python scripts/run_qa.py --config configs/run_qa_qwen3_local.yaml
```

The local QA config points at `runs/build_qwen3_local/graphs`, so it reads the
graphs created by the matching local build config instead of the default API run.

## Design rules

- Parquet is the source of truth.
- DuckDB is a rebuildable query layer, never the only durable state.
- Graph snapshots are immutable and versioned.
- Run artifacts are isolated by `run_id`.

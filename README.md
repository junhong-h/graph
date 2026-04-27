# GraphMemory

GraphMemory is a research-oriented implementation scaffold for graph-memory
experiments on conversational memory benchmarks such as LoCoMo.

This repository currently focuses on:

- building per-conversation memory graphs from raw dialogue
- retrieving answers through graph localization, graph jumps, and raw-text fallback
- saving experiment-scoped graphs, metrics, logs, and analysis artifacts
- keeping formal research runs reproducible under `experiments/`

## Layout

```text
configs/
data/locomo/
experiments/YYYY-MM-DD-NNN-<slug>/
runs/
src/
scripts/
docs/
tests/
```

Formal experiment directories have this shape:

```text
experiments/YYYY-MM-DD-NNN-<slug>/
  config.yaml
  notes.md
  build/
    graphs/
    graph_trajectories_<sample_id>.jsonl
    build.log
    build_stats.json
  qa/
    qa_results.jsonl
    qa_results_eval.jsonl
    qa_metrics.json
    qa_analysis.xlsx
    qa.log
  chroma/
```

Current experiment runs use JSON graph snapshots plus ChromaDB vector indexes.
The older Parquet/DuckDB storage scaffold remains in `src/graphmemory/storage/`
but is not the primary path for the current LoCoMo experiments.

## Quick start

```bash
python3 -m pip install -e '.[dev]'
pytest
```

## Experiment workflow

Formal results must be saved under `experiments/`, not only under `runs/`.

Create a new experiment from the template:

```bash
cp -r experiments/.template experiments/YYYY-MM-DD-NNN-<slug>
```

Fill in `experiments/YYYY-MM-DD-NNN-<slug>/config.yaml` and `notes.md`, then run:

```bash
python scripts/build_memory.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
python scripts/run_qa.py       --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
python scripts/summarize_exp.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>
```

With `--exp-dir`:

- build artifacts are written to `experiments/.../build/`
- graph JSON files are written to `experiments/.../build/graphs/`
- `graph_trajectories_<sample_id>.jsonl` is written to `experiments/.../build/`
- QA artifacts are written to `experiments/.../qa/`
- ChromaDB is written to `experiments/.../chroma/`

The old `--config` interface still works for quick checks:

```bash
python scripts/build_memory.py --config configs/build_memory.yaml --sample-ids conv-26
python scripts/run_qa.py --config configs/run_qa.yaml --sample-ids conv-26
```

Do not reuse an existing experiment directory for a different run. If code,
config, prompts, model, or sample selection changes, create a new experiment
directory.

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

- Formal runs are isolated by experiment directory.
- Config, prompt, model, and evaluation changes must be explicit.
- Save outputs, metrics, logs, and useful intermediate artifacts for each run.
- Treat ChromaDB as rebuildable; track compact graph and metric artifacts in
  `experiments/`.

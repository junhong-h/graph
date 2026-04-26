# GraphMemory Evaluation Protocol

Use LoCoMo Cat1-4 when comparing against the standard answerable-QA protocol.
Cat5 is adversarial/unanswerable and should be reported separately.

## Cat1-4 GraphMemory QA

```bash
python scripts/run_qa.py \
  --config configs/run_qa_qwen3_local.yaml \
  --locomo-cat1-4 \
  --metrics-only
```

Equivalent explicit category filters:

```bash
python scripts/run_qa.py \
  --config configs/run_qa_qwen3_local.yaml \
  --categories 1 2 3 4
```

## Sample Splits

Run a split by sample id:

```bash
python scripts/run_qa.py \
  --config configs/run_qa_qwen3_local.yaml \
  --locomo-cat1-4 \
  --sample-ids conv-26 conv-30
```

Skip the first N samples in dataset order:

```bash
python scripts/run_qa.py \
  --config configs/run_qa_qwen3_local.yaml \
  --locomo-cat1-4 \
  --skip-samples 5
```

Build the matching memory split before QA if graph files do not exist:

```bash
python scripts/build_memory.py \
  --config configs/build_memory_qwen3_local.yaml \
  --skip-samples 5 \
  --limit 5
```

## Cat5

Use a separate run for Cat5:

```bash
python scripts/run_qa.py \
  --config configs/run_qa_qwen3_local.yaml \
  --categories 5
```

## Final Answer Compression

For F1-focused runs, enable the optional final LLM canonicalization pass:

```bash
python scripts/run_qa.py \
  --config configs/run_qa_qwen3_local.yaml \
  --locomo-cat1-4 \
  --compress-final-answer
```

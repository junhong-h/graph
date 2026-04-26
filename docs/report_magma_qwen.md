# MAMGA / MAGMA Qwen3-4B 重跑报告

日期：2026-04-26
工作目录：`/Users/junhong/Projects/forks/MAMGA`
目标协议：LoCoMo Cat1-4，mem-t / MAGMA test split `chat_data[2:]`

## 1. 结论摘要

本轮已完成 MAMGA 方法下的 LoCoMo Cat1-4 正式重跑：samples 2-9，对应 `conv-41, conv-42, conv-43, conv-44, conv-47, conv-48, conv-49, conv-50`，共 1307 题。构建 memory、QA 生成和 LLM judge 均使用 DashScope OpenAI-compatible 的 `qwen3-4b`，embedding 使用 `minilm`，`best-of-n=1`。

总体结果：

| Total | Exact Acc | Avg F1 | BLEU-1 | LLM Judge |
|---:|---:|---:|---:|---:|
| 1307 | 8.26% | 35.11% | 27.41% | 69.17% |

注意：这里的 `Exact Acc` 是 MAMGA 代码里的 exact-match accuracy，不是 GraphMemory 报告中的 LLM judge accuracy；因此它会显著偏低。更适合与 GraphMemory 语义正确率对照的是 `LLM Judge`。

## 2. 本轮代码处理

为支持 Qwen3-4B / DashScope，本轮已在 MAMGA 仓库做兼容修复：

- `utils/llm_compat.py`：新增 OpenAI-compatible 兼容工具，处理 text `response_format`、Qwen3 `<think>...</think>`、JSON fence/prose 容错解析。
- `utils/memory_layer.py`：支持 `OPENAI_BASE_URL`，text 模式下省略 `response_format`，JSON response_format 失败后重试，支持 `MAMGA_DISABLE_THINKING` 和 `MAMGA_MAX_TOKENS`。
- `memory/llm_judge.py` / `memory/evaluator.py`：judge 模型不再硬编码 gpt-4o/gpt-4o-mini，默认跟随 `--model`，也可用 `--judge-model` 或 `MAMGA_JUDGE_MODEL` 覆盖。
- `test_fixed_memory.py`：新增 `--judge-model`、`--no-llm-judge`，并把模型、base_url、类别、样本、cache/output 路径写入结果 JSON。
- `load_dataset.py` / `utils/load_dataset.py`：保留原始 LoCoMo `sample_id`，便于与 mem-t split 对齐。
- `scripts/run_magma_qwen_locomo.sh`：新增可复现正式运行脚本。

## 3. 运行协议

mem-t / MAGMA test split `chat_data[2:]` 对应 MAMGA `data/locomo10.json` 的样本 index 2-9：

| index | sample_id | Cat1-4 题数 |
|---:|---|---:|
| 2 | conv-41 | 152 |
| 3 | conv-42 | 199 |
| 4 | conv-43 | 178 |
| 5 | conv-44 | 123 |
| 6 | conv-47 | 150 |
| 7 | conv-48 | 191 |
| 8 | conv-49 | 156 |
| 9 | conv-50 | 158 |
| **合计** |  | **1307** |

运行参数：

| 参数 | 值 |
|---|---|
| Model | `qwen3-4b` |
| Judge model | `qwen3-4b` |
| API | DashScope OpenAI-compatible |
| Embedding | `minilm` |
| Categories | `1,2,3,4` |
| best-of-n | `1` |
| Cache dir | `runs/magma_qwen_full/cache` |
| Output dir | `results_qwen3_4b` |

复现命令模板：

```bash
cd /Users/junhong/Projects/forks/MAMGA
MODEL=qwen3-4b \
JUDGE_MODEL=qwen3-4b \
CACHE_DIR=runs/magma_qwen_full/cache \
SAMPLES="2 3 4 5 6 7 8 9" \
MAX_QUESTIONS=500 \
BEST_OF_N=1 \
EMBEDDING_MODEL=minilm \
CATEGORIES=1,2,3,4 \
MAMGA_DISABLE_THINKING=1 \
MAMGA_MAX_TOKENS=5000 \
bash scripts/run_magma_qwen_locomo.sh
```

实际逐样本命令记录在：

- `/Users/junhong/Projects/forks/MAMGA/runs/magma_qwen_full/command_samples_2_9.txt`

## 4. 结果路径

逐样本结果：

- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample2.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample3.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample4.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample5.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample6.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample7.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample8.json`
- `/Users/junhong/Projects/forks/MAMGA/results_qwen3_4b/fixed_results_sample9.json`

Cache 和日志：

- `/Users/junhong/Projects/forks/MAMGA/runs/magma_qwen_full/cache/sample2` 到 `sample9`
- `/Users/junhong/Projects/forks/MAMGA/runs/magma_qwen_full/logs/sample2.log` 到 `sample9.log`
- `/Users/junhong/Projects/forks/MAMGA/runs/magma_qwen_full/status.tsv`

所有逐样本 JSON 均包含完整 `results`，且每个文件的 `stats.overall.total` 与预期题数一致。

## 5. 总体结果

| 指标 | 数值 |
|---|---:|
| Total | 1307 |
| Exact correct | 108 |
| Exact Acc | 8.26% |
| Avg F1 | 35.11% |
| BLEU-1 | 27.41% |
| LLM Judge | 69.17% |
| `Information not found` 数量 | 83 |

## 6. 分类别结果

| Cat | Total | Exact Correct | Exact Acc | Avg F1 | BLEU-1 | LLM Judge |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 239 | 14 | 5.86% | 26.00% | 19.00% | 59.45% |
| 2 | 258 | 2 | 0.78% | 35.06% | 22.94% | 61.94% |
| 3 | 83 | 1 | 1.20% | 10.53% | 7.61% | 38.86% |
| 4 | 727 | 91 | 12.52% | 40.92% | 34.03% | 78.39% |

## 7. 分样本结果

| sample | sample_id | Total | Exact Acc | Avg F1 | BLEU-1 | LLM Judge |
|---|---|---:|---:|---:|---:|---:|
| sample2 | conv-41 | 152 | 7.24% | 36.90% | 27.99% | 68.62% |
| sample3 | conv-42 | 199 | 6.53% | 28.32% | 22.65% | 68.52% |
| sample4 | conv-43 | 178 | 8.99% | 38.29% | 31.50% | 70.17% |
| sample5 | conv-44 | 123 | 11.38% | 34.66% | 27.10% | 68.50% |
| sample6 | conv-47 | 150 | 8.00% | 36.01% | 27.36% | 65.89% |
| sample7 | conv-48 | 191 | 8.38% | 35.91% | 28.34% | 70.05% |
| sample8 | conv-49 | 156 | 9.62% | 36.57% | 27.64% | 69.94% |
| sample9 | conv-50 | 158 | 6.96% | 35.39% | 27.20% | 71.20% |

## 8. 与 GraphMemory 当前结果的关系

按同一 mem-t split / Cat1-4 / 1307 题口径，已有 GraphMemory 对齐结果如下：

| 方法 / 结果源 | Avg F1 | BLEU-1 | 备注 |
|---|---:|---:|---|
| GraphMemory GPT-4o run | 50.92% | 44.60% | `runs/qa` |
| GraphMemory Qwen3-4B DashScope | 38.89% | 33.25% | `runs/qa_dashscope` |
| MAMGA Qwen3-4B DashScope | 35.11% | 27.41% | 本报告 |
| Oracle evidence + Qwen3-4B | 57.77% | 51.41% | `runs/qa_oracle` |

在当前统一模型和统一 split 下，MAMGA-Qwen 的 F1 低于 GraphMemory-Qwen 约 3.78 pts，BLEU-1 低约 5.84 pts；但 MAMGA 的 LLM Judge 为 69.17%，说明很多答案语义上部分可接受，exact-match accuracy 不适合作为主要横向指标。

## 9. 运行说明和异常

正式结果已生成完整 sample2-9 文件。运行日志尾部均显示 `Results saved to results_qwen3_4b/fixed_results_sample*.json`，且每个 JSON 的 `stats.overall.total` 与预期题数一致。

`runs/magma_qwen_full/failures.log` 中存在逐样本 `failed; continuing` 记录，但对应日志显示结果已保存，且 `status.tsv` 记录了每个样本完成时间。该 failure 记录更像是外层 wrapper 的退出码记录问题，不影响结果文件完整性；后续如需提交代码，应修复 wrapper 的 rc 捕获逻辑，避免误报。

本轮没有生成 MAMGA 自带聚合 JSON；本报告由 sample2-9 的逐样本结果本地聚合得到。

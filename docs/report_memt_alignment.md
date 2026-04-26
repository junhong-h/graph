# GraphMemory 与 Mem-T 评测协议对齐报告

日期：2026-04-26

## 结论摘要

本报告只使用 Mem-T 的 LoCoMo 评测协议作为参考，没有重跑 Mem-T 模型结果。协议口径为：`chat_data[2:]` test split，即 `conv-41, conv-42, conv-43, conv-44, conv-47, conv-48, conv-49, conv-50`；只统计 Cat1-4，排除 Cat5。对齐集共 1307 题：Cat1=239、Cat2=258、Cat3=83、Cat4=727。

2026-04-26 新增 GraphMemory DashScope F1-fix run：独立目录 `runs/qa_dashscope_f1fix`，复用 `runs/build_dashscope/graphs` 与 `runs/chroma_dashscope`，开启 `final_answer_compression=true` 与 `--compress-final-answer`。该 run 输出显著变短，但 F1/BLEU 没有提升：Overall F1 从 0.3889 降至 0.3559，BLEU-1 从 0.3325 降至 0.3089。主要原因是 prompt 更强的“不可答”约束把大量 Cat1-4 可答题压成 `Not mentioned in the conversation`，F1=0 比例从 30.1% 增加到 41.8%。

Mem-T 自带 evaluator 当前不能直接运行：`llm_judge.py --metrics-only` 仍会在模块导入阶段加载 `llm_api.py`，而 `llm_api.py` 顶层导入 `vllm`，本环境缺少该包，报错 `ModuleNotFoundError: No module named 'vllm'`。因此在 GraphMemory 内新增等价离线脚本 `scripts/recompute_memt_alignment.py`，复刻 Mem-T `llm_judge.py` 的 metrics-only normalization、F1 和 BLEU-1 计算，不调用任何模型。

## 协议差异

| 项目 | GraphMemory 既有报告常用口径 | Mem-T 对齐口径 |
|---|---:|---:|
| 数据 split | 常见为 10 个样本全量，或单样本分析 | `chat_data[0]` train、`chat_data[1]` valid、`chat_data[2:]` test |
| 本次 test 样本 | 既有结果含 `conv-26/30` | 仅 `conv-41/42/43/44/47/48/49/50` |
| Cat5 | DashScope/Oracle 结果含 Cat5 | 排除 Cat5，只算 Cat1-4 |
| 指标 | GraphMemory evaluator 与 LLM judge 混用 | Mem-T metrics-only：token F1、BLEU-1 |
| LLM 调用 | 部分既有 eval 含 judge | 本次无任何模型调用 |

Mem-T 的 F1 归一化规则为：小写、去标点、压缩空白、移除英文冠词 `a/an/the`，再做 token overlap F1；多 gold answer 取最高分。

## 复算结果

| 结果源 | 配置/含义 | Cat1 F1 | Cat2 F1 | Cat3 F1 | Cat4 F1 | Overall F1 | BLEU-1 | F1=0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `runs/qa/qa_results.jsonl` | GPT-4o run，Cat1-4 | 0.3718 | 0.5403 | 0.2805 | 0.5695 | 0.5092 | 0.4460 | 22.9% |
| `runs/qa_dashscope/qa_results.jsonl` | Qwen3-4B DashScope run，过滤 Cat5 | 0.2875 | 0.3093 | 0.1484 | 0.4779 | 0.3889 | 0.3325 | 30.1% |
| `runs/qa_dashscope_f1fix/qa_results.jsonl` | Qwen3-4B DashScope F1-fix，短答案压缩 | 0.2703 | 0.2833 | 0.1365 | 0.4349 | 0.3559 | 0.3089 | 41.8% |
| `runs/qa_oracle/oracle_results.jsonl` | Oracle evidence + Qwen3-4B | 0.5164 | 0.4477 | 0.3251 | 0.6728 | 0.5777 | 0.5141 | 16.1% |

补充观察：

| 结果源 | Exact F1 | Pred/Gold token 比 |
|---|---:|---:|
| GPT-4o run | 24.8% | 1.05x |
| Qwen3-4B DashScope run | 15.7% | 1.77x |
| Qwen3-4B DashScope F1-fix | 15.5% | 1.00x |
| Oracle evidence + Qwen3-4B | 27.7% | 1.22x |

这些数字来自 `artifacts/memt_alignment/memt_alignment_summary.json`。逐题复算结果保存在同目录的 `*_memt_metrics.jsonl`。

F1-fix 相对旧 DashScope 的变化：

| 指标 | 旧 DashScope | F1-fix | 变化 |
|---|---:|---:|---:|
| Overall F1 | 0.3889 | 0.3559 | -0.0330 |
| BLEU-1 | 0.3325 | 0.3089 | -0.0236 |
| F1=0 | 30.1% | 41.8% | +11.6 pts |
| Pred/Gold token 比 | 1.77x | 1.00x | -0.77x |

按类别看，F1-fix 的回退集中在 Cat4：

| 类别 | 旧 DashScope F1 | F1-fix F1 | 变化 |
|---|---:|---:|---:|
| Cat1 | 0.2875 | 0.2703 | -0.0172 |
| Cat2 | 0.3093 | 0.2833 | -0.0259 |
| Cat3 | 0.1484 | 0.1365 | -0.0119 |
| Cat4 | 0.4779 | 0.4349 | -0.0431 |

## F1 偏低原因

1. Token F1 对数字表达不等价。典型例子是 gold=`three`、pred=`3`，或 gold=`two`、pred=`2`，语义正确也会得到 0 分。这类问题在计数题中很常见，直接拉低 Cat1/Cat4。

2. DashScope 旧结果明显更冗长。Qwen3-4B run 的 pred/gold token 比为 1.77x，Cat4 达 2.19x；Mem-T F1 是 precision/recall 的调和平均，答案包含正确片段但附带解释时 precision 会下降。

3. F1-fix 输出变短但召回受损。新 run 的 pred/gold token 比降到 1.00x，说明压缩生效；但 Cat1-4 中 `Not mentioned in the conversation` 增至 344 条，其中 Cat4 167 条。旧 DashScope 在 Cat1-4 的 `Unknown`/`None`/`Not mentioned` 类拒答约 125 条。短答案约束减少了冗余，却把很多可答题变成拒答，导致 F1=0 增加。

4. Cat3 与 token F1 天然不匹配。Cat3 是开放推断题，本次 Qwen3-4B Cat3 F1 只有 0.1484，F1=0 比例 61.4%；GPT-4o Cat3 F1 也只有 0.2805。很多合理表达没有词面重叠，例如 `Presumably not` vs `No`、`likely yes` vs 更完整的解释。

5. 检索/图结构仍有实质损失。Oracle evidence + Qwen3-4B 的 Overall F1 为 0.5777，而旧 Qwen3-4B GraphMemory run 为 0.3889，差距 0.1888；F1-fix run 与 Oracle 的差距扩大到 0.2218。Cat1、Cat3、Cat4 的差距尤其明显，说明除指标问题外，还存在证据定位、聚合和具体事实抽取不足。

6. 时序与日期格式会被部分惩罚。Mem-T normalization 能处理逗号差异，但不能把 `2023-05-07` 与 `7 May 2023` 完全等价，也不能自动把相对时间或月份表述对齐到 gold。

## 后续建议

1. 报 Mem-T 对齐结果时固定使用 `scripts/recompute_memt_alignment.py` 或修复 Mem-T evaluator 的 lazy import；不要把含 Cat5 或 `conv-26/30` 的结果混入 Cat1-4 F1。

2. 不建议直接采用当前 F1-fix prompt 作为默认结果。它解决了冗长问题，但把 Cat1-4 的可答题过度拒答。下一版应把“Not mentioned”规则限定在 Cat5 或显式不可答评测，不应在 Cat1-4 主协议中积极触发。

3. 对 Qwen3-4B 检索回答继续保留短答案约束，但应改为“必须给出 best short answer”，只在 Cat5 或单独 unanswerable run 中允许 `Not mentioned in the conversation`。压缩步骤应只改写已有答案，不应重新判断可答性。

4. 增加后处理 normalization，用于内部分析时单独标注数字词与阿拉伯数字、ISO 日期与自然日期的等价情况；正式 Mem-T F1 仍保留原始指标，避免和论文口径不一致。

5. 对 Cat1/Cat4 继续优先修复具体事实抽取和聚合检索。Oracle 差距显示，当前低 F1 不只是生成格式问题，仍有证据缺失和错误定位。

6. Cat3 不应只看 token F1。若需要判断真实能力，应同步报告 LLM judge 或人工抽样；Mem-T F1 可作为词面一致性指标，但不能单独代表推断质量。

## 复现命令

```bash
python scripts/recompute_memt_alignment.py \
  --dataset data/locomo/locomo10.json \
  --out-dir artifacts/memt_alignment \
  --input graphmemory_default=runs/qa/qa_results.jsonl \
  --input graphmemory_dashscope=runs/qa_dashscope/qa_results.jsonl \
  --input graphmemory_dashscope_f1fix=runs/qa_dashscope_f1fix/qa_results.jsonl \
  --input oracle_dashscope=runs/qa_oracle/oracle_results.jsonl
```

F1-fix GraphMemory QA 命令：

```bash
python scripts/run_qa.py \
  --config configs/run_qa_dashscope_f1fix.yaml \
  --locomo-cat1-4 \
  --skip-samples 2 \
  --compress-final-answer \
  --metrics-only
```

Mem-T evaluator 尝试命令如下，当前因 `vllm` 导入失败不能进入 metrics-only：

```bash
python /Users/junhong/Projects/third-party/mem-t/llm_judge.py \
  --input artifacts/memt_alignment/graphmemory_dashscope_memt_split_cat1_4_input.jsonl \
  --output artifacts/memt_alignment/graphmemory_dashscope_memt_metrics.jsonl \
  --benchmark locomo \
  --metrics-only
```

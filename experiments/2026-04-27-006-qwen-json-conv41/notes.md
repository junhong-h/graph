# 实验 2026-04-27-006：qwen-json-conv41

## 假设
Qwen JSON Mode、显式 JSON object envelope、`top_p: 0.7` 和 `seed: 42` 可以降低构图与检索规划输出的不确定性，减少格式错误和 action/anchor 解析失败，从而让 conv-41 的端到端结果更稳定。

## 与上次的差异

| 维度 | 上次 (2026-04-27-005) | 本次 |
|------|-----------------------|------|
| 代码改动 | current retrieval + refine2 graph | Qwen JSON/output 稳定化 |
| 模型 | qwen3-4b | qwen3-4b |
| temperature | 0.0 | 0.0 |
| top_p | 未显式设置 | 0.7 |
| seed | 未显式设置 | 42 |
| jump_budget | 5 | 5 |
| seed_top_k | 5 | 5 |
| 样本 | conv-41, conv-26 | conv-41 |
| 图 | 复用 004 图 | 重新构图 |

## 改动
- `src/graphmemory/llm_client.py` — JSON Mode 自动补 JSON 指令，传入 `top_p`/`seed`，并在 provider 不支持 seed 时降级重试。
- `src/graphmemory/graph_construction.py` — 构图输出改为 `{"ops": [...]}` JSON object，调用 `json_mode=True`。
- `src/graphmemory/graph_retrieval.py` — anchor 选择和 action planner 调用 `json_mode=True`，保留旧格式 fallback。

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-27-006-qwen-json-conv41
python scripts/run_qa.py       --exp-dir experiments/2026-04-27-006-qwen-json-conv41
python scripts/summarize_exp.py --exp-dir experiments/2026-04-27-006-qwen-json-conv41
```

## 运行环境
- commit: `5321a1c`
- build 耗时: ~13 min
- QA 耗时: ~12 min（193 题）
- 图文件: `experiments/2026-04-27-006-qwen-json-conv41/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | 176 | 243 | 35 | 141 | 228 |

补充统计：
- event-event 边：15
- events with fact/quote：141 / 141
- isolated event：0
- event without entity-event edge：0
- low-degree event：112
- build failed_ops：174
- build log 中 `invalid edge family ('')`：92 次
- build log 中 self-loop prevented：2 次

## QA 结果

| Category | 2026-04-27-005 conv-41 | 本次 | 变化 |
|----------|-------------------------|------|------|
| Cat1     | - | 96.8% (F1=0.466) | - |
| Cat2     | - | 92.6% (F1=0.419) | - |
| Cat3     | - | 75.0% (F1=0.129) | - |
| Cat4     | - | 90.7% (F1=0.506) | - |
| Cat1-4   | 86.2% (F1=0.451) | 91.4% (F1=0.444) | +5.3 / -0.007 |
| Cat5     | 70.7% (F1=0.713) | 65.9% (F1=0.659) | -4.9 / -0.055 |
| Overall  | 82.9% (F1=0.507) | 86.0% (F1=0.504) | +3.1 / -0.003 |

## 分析

### 主要结论

Qwen JSON/output 稳定化对可答题有明显正向影响：conv-41 Cat1-4 从 005 的 86.2% 恢复到 91.4%，也略高于 004 refine2 graph 的 90.8%。这说明结构化输出和 hybrid jump 修复后的主线可答题检索恢复了。

但 Cat5 从 005 的 70.7% 继续下降到 65.9%，距离 004 的 80.5% 更远。问题不是格式解析失败，而是不可答题的拒答策略仍然不稳：模型在 raw fallback 中看到 near-miss 证据后倾向于给具体答案。

### Trace 信号

本次 193 题：
- 有 `raw_fallback` 的题：98 / 193
- 有 `jump` 的题：51 / 193
- 有 forced finish 的题：44 / 193
- QA log 中 `action=raw_fallback`：109 次
- QA log 中 `Retrieval forced finish`：44 次

按类别看 raw fallback：
- Cat4：44 题，其中错 3 题
- Cat5：30 题，其中错 8 题
- Cat1：11 题，其中错 1 题
- Cat2：7 题，其中错 0 题
- Cat3：6 题，其中错 2 题

Cat5 典型错误：
- `conv-41_q169`：问题问 John 为什么最近加入附近教堂，gold 是 `Not mentioned in the conversation`，raw fallback 后回答 “to feel closer to a community and his faith”。
- `conv-41_q190`：问题问 Maria 如何描述和 Max 的露营旅行，gold 是 `Not mentioned in the conversation`，raw fallback 后回答 “a blast and a great experience”。
- `conv-41_q192`：问题问 John 的新小狗适应情况，gold 是 `Not mentioned in the conversation`，raw fallback 后回答 “Adjusting well, learning commands and house training”。

### 构图质量

最终图表面结构指标比 004 更大更完整：
- 004 conv-41：156 nodes / 164 edges / 43 Entity / 113 Event / 148 entity-event edges
- 006 conv-41：176 nodes / 243 edges / 35 Entity / 141 Event / 228 entity-event edges

所有 Event 都有 fact/quote，也没有 isolated event 或 event without entity-event edge。但 build 过程中出现 92 次空 `family` 边被拒绝，说明 JSON Mode 保证了可解析格式，却没有完全保证 schema 语义正确。这个失败不会污染最终图，但会浪费 LLM 输出，并可能暴露 prompt 对 relation family 约束仍不够硬。

## 遗留问题

- Cat5 拒答仍是主要短板，尤其 raw fallback 会把 near-miss 证据转成具体答案。
- 构图 prompt 仍允许模型输出空 `family`，需要更强 schema/enum 约束或执行前 repair。
- GraphStore 统计里 `failed_ops=174` 偏高，需要拆分空 family、自环、重复/无效引用等失败原因。
- LLM judge 仍有宽松案例，例如 `conv-41_q0` pred=`John`、gold=`her mother` 被判 CORRECT，后续需要人工抽查关键翻转。

## 下一步

- 优先修 Cat5：raw fallback 前加入 answerability check，或要求 fallback evidence 必须满足时间/主体/对象约束，否则拒答。
- 对 Link/AddEdge 加硬校验：family 只能是 `entity-event` 或 `event-event`，predicate/family 缺失时尝试 repair，一次失败后丢弃并记录原因。
- 对 006 和 004/005 做逐题翻转表，确认 Cat1-4 提升来自哪些题，Cat5 下降是否集中在 raw fallback near-miss。

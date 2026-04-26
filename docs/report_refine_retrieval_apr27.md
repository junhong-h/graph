# GraphMemory Refinement 报告：Construction 与 Retrieval

**日期**：2026-04-27  
**分支**：`refine-retrieval-construction`  
**评测目录**：`runs/refine_final_conv41/`  
**评测样本**：`conv-41`  
**问题数量**：193  
**评测方式**：`scripts/run_qa.py` 全量 QA + LLM judge

本文档记录本轮 refine 分支已经完成的代码改动、`conv-41` 最终评测结果、运行中观察到的问题，以及下一步建议优先处理的方向。

---

## 1. 本轮已经实现的内容

当前分支包含 6 个阶段性提交：

| Commit | 内容 |
|---|---|
| `651727f` | 修复 QA retrieval baseline 行为，包括恢复 final answer compression 和 answerable refusal 的 raw fallback 修复。 |
| `f435386` | 增加 construction invariants：规范 edge family、加入 construction context、修复 event metadata、自动处理 orphan event linking。 |
| `bc620d5` | 优化 construction ontology prompt，明确 Entity、Concept、Event 的边界。 |
| `c85036e` | 为 recall-heavy 问题加入 multi-seed retrieval localization。 |
| `9ec320e` | 改进 jump 扩展排序、constraint scoring、frontier exhausted fallback 和 raw fallback 修复逻辑。 |
| `2592e5b` | 增加 retrieval trace diagnostics 和汇总脚本。 |

代码已经推送到：

```text
origin/refine-retrieval-construction
```

最终单元测试结果：

```text
108 passed
```

---

## 2. 最终评测结果

本轮在 `conv-41` 上跑完了完整 QA 和 LLM judge，共 193 个问题。

| 类别 | 题数 | Accuracy | Avg F1 | BLEU-1 |
|---|---:|---:|---:|---:|
| Cat1 | 31 | 93.5% | 0.4169 | 0.2929 |
| Cat2 | 27 | 88.9% | 0.3526 | 0.2585 |
| Cat3 | 8 | 62.5% | 0.1705 | 0.1667 |
| Cat4 | 86 | 90.7% | 0.5294 | 0.4667 |
| Cat5 | 41 | 53.7% | 0.4815 | 0.4749 |
| **Overall** | **193** | **81.9%** | **0.4616** | **0.3990** |

关键产物：

```text
runs/refine_final_conv41/qa_results.jsonl
runs/refine_final_conv41/qa_results_eval.jsonl
runs/refine_final_conv41/qa_metrics.json
runs/refine_final_conv41/retrieval_trace_summary.json
```

从结果看，Cat1、Cat2、Cat4 的 LLM judge accuracy 已经比较高，但 Cat3 和 Cat5 仍然是明显短板。尤其 Cat5 只有 53.7%，说明系统在“证据不足时拒答”这件事上还不稳定。

---

## 3. Retrieval Trace 汇总

trace summary 显示，当前问题已经不是单一的 `max_hop` 问题。真实瓶颈混合了以下几类：

- graph navigation 仍然会中途耗尽 frontier；
- raw fallback 很有用，但使用频率偏高；
- final answer synthesis 有时会过度概括或抽错邻近事实；
- Cat5 unanswerable 问题缺少严格的 evidence sufficiency 判断。

整体 trace 指标：

| 指标 | 数值 |
|---|---:|
| 平均 trace steps | 2.79 |
| 触发 forced finish 的样本比例 | 19.2% |
| 触发 frontier exhausted 的样本比例 | 27.5% |
| 使用 raw fallback 的样本比例 | 57.0% |
| `max_hop_exhausted` forced finish 数量 | 19 |
| `frontier_exhausted` forced finish 数量 | 18 |

action 统计：

| Action | Count |
|---|---:|
| `finish` | 156 |
| `raw_fallback` | 154 |
| `jump` | 124 |
| `frontier_exhausted` | 64 |
| `forced_finish` | 37 |
| `answerable_refusal_raw_fallback` | 4 |

按类别看：

| 类别 | 使用 raw fallback 的样本数 | forced finish 样本数 | 主要信号 |
|---|---:|---:|---|
| Cat1 | 14 | 5 | 聚合问题仍然依赖 raw fallback 补证据。 |
| Cat2 | 17 | 4 | 时间问题 judge accuracy 高，但检索路径仍不够直接。 |
| Cat3 | 6 | 5 | 样本少，但 forced finish 比例很高。 |
| Cat4 | 46 | 9 | 很多正确答案仍依赖 fallback 修复。 |
| Cat5 | 27 | 14 | unanswerable 问题经常继续搜索，或者从相似证据里误答。 |

---

## 4. 观察到的主要问题

### 4.1 Max-hop 仍然存在，但不是唯一根因

在这轮改动前，`max_hop` 看起来像 retrieval 的主要失败点。加入 frontier exhausted fallback 和 raw fallback 之后，很多原本卡住的路径可以继续，并且部分问题能答对。

但是最终仍有 19.2% 的样本以 forced finish 结束。这里需要区分三种情况：

- 对 Cat5 来说，forced finish 后正确拒答可能是合理行为；
- 有些 forced finish 是 planner 反复选择低价值 jump 导致的；
- 有些样本没有 forced finish，但仍然因为找到了相似但错误的 event 而答错。

因此下一步不建议简单增加 `max_hops`。更多 hop 很可能引入更多噪声，并进一步拉低 Cat5。

### 4.2 Raw fallback 有收益，但使用过多

本轮 57.0% 的样本使用了 raw fallback。它确实是有效的兜底机制，尤其在 graph navigation 因缺边、event 粒度不准或 frontier exhausted 停住时，可以从原始对话中补回证据。

但 raw fallback 对 Cat5 有副作用：当问题本身不可回答时，它经常会找到“语义相近但约束不满足”的证据。final answer model 随后可能根据这个 near-miss evidence 生成具体答案，而不是拒答。

典型模式是：

```text
问题要求 X 且满足约束 A。
raw fallback 找到了 X 或相似 X，但它满足的是约束 B。
final answer 忽略约束差异，直接回答 X。
```

这说明 raw fallback 后面需要 evidence sufficiency verification，而不是直接交给 final answer。

### 4.3 Cat5 拒答能力最弱

Cat5 accuracy 只有 53.7%，明显低于 Cat1、Cat2、Cat4。当前 pipeline 更偏向“尽量找到相关信息并回答”，但 Cat5 需要的是“判断证据是否真的足够”。

运行中观察到的例子：

| QA | 问题 |
|---|---|
| `conv-41_q154` | 问 John 开始 volunteering 是受谁启发；系统回答 `Maria's aunt`，但目标事实其实没有被提到。 |
| `conv-41_q167` | 问 Maria 和父亲 dinner spread 里的食物；系统从附近 food mention 中回答 `banana split sundae`。 |
| `conv-41_q186` | 问 Maria 组织的 5K charity run 支持什么 cause；系统回答 `veterans and their families`，这是 John 的 5K event 的信息。 |

这些不是单纯的 retrieval miss，而是 retrieved evidence 没有满足问题的完整约束。

### 4.4 相似 event 被混淆

最清楚的例子是多个 `5K charity run`。它们 surface form 很像，但 actor、date、cause 不同。

观察到的错误：

- John 的 5K charity run cause 被回答成 `homeless shelter`，这是另一个 event 的信息；
- Maria 的 5K charity run 被回答成 `veterans and their families`，这是 John 的 event 的信息。

这说明当前 event identity 还太弱。event 不能只靠 activity text 识别，而应该有更完整的 signature：

```text
event_signature = actor + action + object + time + location + purpose/cause
```

retrieval 时也应该优先选择 signature 与问题约束匹配的 event，而不是只选语义相似的 event。

### 4.5 时间问题仍会混淆 session date 和 event date

Cat2 的 judge accuracy 虽然达到 88.9%，但运行中仍观察到一个重要问题：模型有时会把 conversation/session date 当成事件发生时间。

例子：

```text
Question: When did John participate in a 5K charity run?
Gold: first weekend of August 2023
Pred: 2023-04-07
```

这说明 construction 和 retrieval 需要明确区分：

- `turn_time`：对话发生的时间；
- `event_time`：对话中描述的事件发生时间；
- `relative_time_text`：例如 "last weekend"、"the week before"、"later that evening"；
- `time_anchor`：相对时间解析时依赖的 anchor。

目前这些 time-like 信息在 final answer synthesis 时容易混在一起。

### 4.6 Answer synthesis 有时会丢细节

有些问题看起来已经拿到了相关 evidence，但最终答案过度压缩或过于泛化。

例子：

| QA | Gold | Prediction | 问题 |
|---|---|---|---|
| `conv-41_q148` | `doing great - learning commands and house training` | `Adjusting well` | 方向对，但细节丢失。 |
| `conv-41_q125` | `the resilience of the veterans and their inspiring stories` | `John appreciates the veteran's hospital visit` | 复述问题，没有抽取真正答案。 |

这不是纯图结构问题，而是 final answer prompt 需要更明确地要求保留 answer-bearing details。

### 4.7 Cat3 仍然偏弱

Cat3 accuracy 为 62.5%，F1 和 BLEU-1 都较低。虽然该类别在 `conv-41` 中只有 8 题，但 trace profile 仍然值得注意：

```text
Cat3 samples with forced finish: 5 / 8
Cat3 samples with raw fallback: 6 / 8
```

这说明 multi-seed localization 有帮助，但还没有完全解决开放聚合和推断问题。Cat3 需要更强的 evidence grouping 和 answer-time synthesis，而不仅仅是取更多节点。

---

## 5. 总体判断

本轮 refine 之后，retrieval 的结构性行为有明显改善：

- jump 扩展不再按边顺序截断，而是按相关性排序；
- constraint matching 不再是默认全通过；
- frontier exhausted 不再直接停住，而是触发 raw fallback；
- Cat1/Cat3 可以使用多个 localized subgraph 的 union；
- trace diagnostics 让失败路径可观测。

但当前系统仍然存在一个核心张力：

1. answerable 问题需要扩大召回，跨 session 找全证据；
2. unanswerable 问题需要严格约束验证，避免从相似证据里编答案。

raw fallback 帮助了第一个目标，但如果没有 evidence sufficiency verification，会伤害第二个目标。

因此下一步重点不应该是继续盲目扩大检索范围，而应该是在 final answer 前加入证据充分性判断。

---

## 6. 下一步改进建议

### Priority 1：加入 evidence sufficiency verification

在 final answer 前加入一个轻量 verification step：

```text
给定 question + candidate evidence：
1. 从 question 中抽取必须满足的约束；
2. 检查 evidence 是否满足每个约束；
3. 如果任何关键约束缺失或冲突，则标记为 insufficient；
4. 只有 sufficient 时才允许回答，否则返回 Not mentioned in the conversation。
```

建议 verifier 输出结构化结果：

```text
sufficient: true/false
missing_constraints: [...]
supporting_evidence_ids: [...]
reason: ...
```

这一步对 Cat5 和相似 event 混淆问题最关键。

### Priority 2：增强 event signature

每个 event node 尽量保存标准化 slot：

```text
actor
action
object
time
location
purpose/cause
source_turn_ids
speaker
polarity/status
```

retrieval scoring 应该使用这些 slot。对于 surface form 相似的 event，如果 actor、time、purpose 不匹配，就不应该排在前面。

这会直接改善 5K charity run 这类混淆。

### Priority 3：拆分 `turn_time` 和 `event_time`

construction 应该保留：

```text
turn_time: conversation timestamp
event_time: described event timestamp
relative_time_text: original relative time phrase
time_anchor: resolved anchor if available
```

retrieval 和 final answer synthesis 在回答 "when did X happen" 时应优先使用 `event_time`，只有在解析 relative time 时才使用 `turn_time`。

### Priority 4：优化 final answer prompt，保留细节

final answer 阶段应避免把具体 evidence 压缩成泛泛总结。尤其是属性、原因、列表、状态类问题。

建议 prompt 增加规则：

```text
如果 evidence 中包含具体 answer-bearing details，必须保留这些细节。
不要用泛化总结替代具体答案。
```

这会改善 `Adjusting well` 这类过度压缩答案。

### Priority 5：加入 same-name event disambiguation reranking

当多个 event action/name 相似时，rerank 应考虑完整约束匹配：

```text
score = semantic_similarity
      + actor_match
      + time_match
      + object_match
      + purpose_match
      + source_turn_relevance
      - contradiction_penalty
```

这套 scoring 应该同时用于 graph jump candidate 和 raw fallback evidence。

### Priority 6：把 Cat5 当作独立模式处理

Cat5 不应该和 Cat1-Cat4 使用完全相同的“找到相关信息就回答”策略。一个实用规则是：

```text
如果问题包含具体约束，而 retrieved evidence 只匹配 general topic，
则必须拒答。
```

Cat5 还应该单独统计：

- false positive answer rate；
- near-miss evidence rate；
- missing constraint reasons；
- refusal 是因为 no evidence 还是 insufficient evidence。

---

## 7. 建议 Coding Plan

### Stage A：Evidence verifier

可能涉及文件：

```text
src/graphmemory/graph_retrieval.py
src/graphmemory/prompts.py
tests/
```

先实现 prompt-based verifier，再逐步加入 deterministic checks，例如 actor/time/object 字段存在时直接比对。

每个阶段完成后运行测试并 commit。

### Stage B：Event slot schema

可能涉及文件：

```text
src/graphmemory/graph_build.py
src/graphmemory/prompts.py
src/graphmemory/graph_store.py
tests/
```

扩展 construction prompt 和 repair logic，保留 event slots。实现时要兼容已有 graph JSON，避免旧图无法读取。

每个阶段完成后运行测试并 commit。

### Stage C：Constraint-aware reranking

可能涉及文件：

```text
src/graphmemory/graph_retrieval.py
tests/
```

使用 event slots 给 graph candidate 和 raw fallback hit rerank。测试用例要覆盖 same-action events with different actors/purposes。

每个阶段完成后运行测试并 commit。

### Stage D：Time handling

可能涉及文件：

```text
src/graphmemory/graph_build.py
src/graphmemory/graph_retrieval.py
src/graphmemory/prompts.py
tests/
```

拆分 `turn_time`、`event_time`、`relative_time_text`。增加 conversation date 与 event date 不一致的测试。

每个阶段完成后运行测试并 commit。

### Stage E：Focused evaluation + full evaluation

建议评测顺序：

1. 运行单元测试；
2. 跑 targeted QA：
   - 5K charity run actor/cause 问题；
   - Cat5 near-miss food/activity 问题；
   - event date vs session date 问题；
3. 跑完整 `conv-41` QA + LLM judge；
4. 对比：
   - overall accuracy；
   - Cat5 accuracy；
   - forced finish rate；
   - raw fallback rate；
   - same-event confusion examples。

---

## 8. 结论

当前分支已经让 retrieval 更稳、更可观测，但剩余错误说明下一个瓶颈不是“继续扩大图搜索”，而是“验证证据是否真的足够回答问题”。

下一步最高优先级是：

```text
在 final answer 前加入 evidence sufficiency verification，
然后增强 event signature，让 retrieval 能区分相似 event。
```

这两件事应该能同时改善 Cat5 拒答和 same-event contamination，而且不会破坏本轮通过 raw fallback 和 multi-seed localization 得到的 recall 收益。

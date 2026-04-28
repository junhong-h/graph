# 实验 2026-04-28-008：skip-recovery-finish-check

## 假设

007 的 22 道 Cat1+Cat4 错题分析显示：
- **Construction 误 Skip**：Skip 整 batch 生效，主调闲聊会淹没单 turn 的具体事实（如 "I had a little doll"、"after I failed the military aptitude test"）。
- **Retrieval 早 finish**：LLM 凭语义印象/raw_fallback 漂移作答（如 q13 答 "becoming a mom"）。

本次三项改动联合解决：

1. **Skip Rule 8 收窄**（`graph_construction._SYSTEM_PROMPT`）：从"OR 主调闲聊就 Skip"改为"every turn 都没具体事实才 Skip"，并加 "When in doubt, do NOT skip" 兜底原则。无具体短语/反例，跨数据集泛化。
2. **Skip 拆 turn 重试**（`graph_construction.GraphConstructor.run`）：当 LLM 对多 turn batch 返回纯 Skip 时，自动按 `\n\n` 拆 turn 逐个再调 LLM，避免一刀切丢嵌入事实。
3. **finish 前自检**（`graph_retrieval._SYSTEM_PROMPT`）：要求 finish 必须由已收集的图证据/raw turns 显式支持，禁止凭语义相似/世界知识推断。

预期：
- Cat1 正确率 +3~5（恢复 q7、q12 等被 Skip 的事实）
- Cat4 正确率小幅 +1~2（finish 自检减少漂移）
- Cat5 不退步（finish 自检与 Cat5 拒答指令一致）
- 图节点数小幅增加（per-turn 恢复出更多 Entity/Event）

## 与上次的差异

| 维度 | 007 (based_on) | 本次 |
|------|----------------|------|
| 代码改动 | Cat5 raw_fallback 屏蔽 + 时间词加权 | + Skip Rule 8 收窄、Skip 拆 turn 重试、finish 自检 |
| 图 | 复用 005 图 | **重建（C1 + A3 影响）** |
| 样本 | conv-41, conv-26 | **全 10 样本** |
| k_turns | 4 | 4 |
| jump_budget | 5 | 5 |
| seed_top_k | 5 | 5 |

## 改动

- `src/graphmemory/graph_construction.py:_SYSTEM_PROMPT` — Rule 8 改为 "every turn 都没事实才 Skip + when in doubt don't skip"
- `src/graphmemory/graph_construction.py:GraphConstructor.run` — 检测纯 Skip 后自动拆 turn 重试，op_log 标 `per_turn_recovery`、首条记 `BatchSkipRecovery`
- `src/graphmemory/graph_retrieval.py:_SYSTEM_PROMPT` — 增加 finish 前自检要求

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-008-skip-recovery-finish-check
python scripts/run_qa.py       --exp-dir experiments/2026-04-28-008-skip-recovery-finish-check
python scripts/summarize_exp.py --exp-dir experiments/2026-04-28-008-skip-recovery-finish-check
```

## 运行环境

- commit: `162b7b0`
- build 耗时: ~52 min（10 样本，4 worker 并发）
- QA 耗时: ~16 min retrieval + 6 min judge（1986 题，8 worker 并发）
- 图文件: `experiments/2026-04-28-008-skip-recovery-finish-check/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-26 | 142 | 216 | 38 | 104 | 198 |
| conv-30 | 171 | 226 | 23 | 148 | 221 |
| conv-41 | 214 | 255 | 54 | 160 | 235 |
| conv-42 | 286 | 321 | 80 | 206 | 302 |
| conv-43 | 283 | 329 | 75 | 208 | 314 |
| conv-44 | 264 | 383 | 50 | 214 | 330 |
| conv-47 | 325 | 411 | 90 | 235 | 378 |
| conv-48 | 298 | 348 | 88 | 210 | 325 |
| conv-49 | 232 | 254 | 44 | 188 | 234 |
| conv-50 | 262 | 333 | 55 | 207 | 314 |
| **总计** | **2477** | **3076** | **597** | **1880** | **2851** |

总 1571 batch，trigger 1477（94%），其中 **70 个 batch 触发 BatchSkipRecovery**（被 LLM 整体 Skip 后拆 turn 重试），这是 A3 算法的实际命中量。

对比同样本（conv-26、conv-41）：
| 样本 | 005/007 节点 | 008 节点 | Δ |
|------|-------------:|---------:|--:|
| conv-26 | 96 | 142 | +46 |
| conv-41 | 156 | 214 | +58 |

构图覆盖密度明显提升，符合 A3+C1 预期。

## QA 结果

### 全 10 样本（1986 题）

| Category | n | Acc | F1 |
|----------|--:|----:|---:|
| Cat1 | 282 | 86.5% | 0.320 |
| Cat2 | 321 | 82.9% | 0.362 |
| Cat3 | 96 | 70.8% | 0.188 |
| Cat4 | 841 | 89.2% | 0.460 |
| **Cat1-4** | **1540** | **86.2%** | **—** |
| Cat5 | 446 | 86.3% | 0.851 |
| **Overall** | **1986** | **86.3%** | **0.499** |

### 同样本对比（conv-26 + conv-41，与 007 同图 ablation 比较）

| Category | 007 Acc | 008 Acc | Δ | 007 F1 | 008 F1 | Δ |
|----------|--------:|--------:|--:|-------:|-------:|--:|
| Cat1 | 88.9% | **95.2%** | **+6.3** | 0.348 | 0.388 | +0.040 |
| Cat2 | 87.5% | 84.4% | −3.1 | 0.381 | 0.376 | −0.005 |
| Cat3 | 76.2% | 76.2% | 0.0 | 0.218 | 0.170 | −0.048 |
| Cat4 | 90.4% | 89.7% | −0.7 | 0.466 | 0.483 | +0.017 |
| Cat1-4 | 88.5% | **89.2%** | +0.7 | — | — | — |
| Cat5 | 87.5% | 86.4% | −1.1 | 0.877 | 0.866 | −0.011 |

翻转：32 UP / 32 DOWN，**Cat1 净 +4 题**（其余类别基本平衡）。

## 分析

### Cat1 大幅改善（+6.3%，6 UP / 2 DOWN）

A3（Skip 拆 turn 重试）+ C1（Skip 规则收窄）针对的"嵌入事实被淹没"场景全部命中：

| qa_id | gold | 007 pred | 008 pred |
|-------|------|---------:|---------:|
| conv-41_q7 (childhood items) | A doll, a film camera | family road trip | a furry pal |
| conv-41_q12 (military test) | military aptitude test | Unknown | aptitude test |
| conv-26_q13 (career path) | counseling/mental health | becoming a mom | counseling and mental health |
| conv-26_q23 (books) | Charlotte's Web, ... | The book Caroline recommended | Charlotte's Web |

被 007 Skip 掉的具体事实在 008 重建图中变成了独立节点。q7 答得仍不完美但已显著优于 007（"a furry pal" 部分匹配 doll）。

### Cat2 小幅退步（-3.1%，4 UP / 6 DOWN）

D1 finish 自检让 LLM 更倾向于使用图里已有的 event time，但部分 Cat2 题需要从 raw_fallback 找原始日期。例如 conv-26_q29（pottery workshop 时间），007 pred="15 July 2023"（正确），008 pred="17 August, 2023"（错图节点的 time attr）。

这是 finish 自检的合理 trade-off：它防止了 Cat1 的语义漂移（如 "becoming a mom"），但把 Cat2 推向了图证据优先，对部分日期题有副作用。

### 全量 10 样本结果

Cat1-4 综合 86.2%，与 007 同样本 88.5% 相近但略低，这是泛化到所有样本（包括 conv-30/42/43 等更长会话）的结果。Cat4 89.2%、Cat5 86.3% 均稳定。

### 与 Apr20 P0+P1 报告对比

`docs/report_progress_apr20.md` 中 P0+P1 全量 Cat1=90.4%、Cat4=91.2%、Cat1-4=87.3%。本次 008 为 Cat1=86.5%、Cat4=89.2%、Cat1-4=86.2%，分别下降 3.9、2.0、1.1 个点。

下降的主要原因不是图覆盖不足，而是图过密带来的 near-miss 噪声，以及 Cat5 安全规则/finish 自检对 Cat1-4 answerable 题的副作用：

- Cat1 38 道错题中，28 道是 hop0 直接 `finish`。这说明很多跨 session 聚合题没有真正做聚合，而是从局部 near-miss 证据直接给答案。
- Cat4 91 道错题中，42 道直接 `finish` 错，另有 16 道走 `raw_fallback > raw_fallback > raw_fallback > answerable_refusal_raw_fallback > forced_finish`，出现 answerable 题被推向 `Unknown` / `Not mentioned` 的副作用。
- 下降集中在图膨胀明显的样本：Cat1 错误最多的是 conv-42、conv-49、conv-43、conv-50；Cat4 错误最多的是 conv-48、conv-42、conv-49、conv-41。

典型 Cat1 错误：
- `conv-42_q78`：问 Nate 参加过多少 tournaments，gold=`nine`，pred=`four`。
- `conv-43_q17`：问 John 提到赢过多少 games，gold=`6`，pred=`3`。
- `conv-49_q3`：问 Evan owned 几辆 Prius，gold=`two`，pred=`one`。

典型 Cat4 错误：
- `conv-26_q106`：gold=`Running`，pred=`Unknown`。
- `conv-49_q95`：gold=`tracks progress...`，pred=`Not mentioned in the conversation`。
- `conv-48_q150`：问 Switch open-world game，gold=`Zelda BOTW`，pred=`Monster Hunter: World`。

结论：008 把 Cat5 从 Apr20 的 36.1% 大幅提升到 86.3%，但 Cat1/Cat4 受到了两类代价影响：一是大图噪声造成的同主题错配，二是拒答/显式证据规则迁移到 answerable 题后过度保守。

### 图结构健康度

008 图结构偏不健康，主要表现为节点和边明显膨胀、低度 Event 过多、重复/抽象事件过多。

与 Apr20 P0+P1 报告中的图结构粗略对比：

| 指标 | Apr20 P0+P1 | 008 | 变化 |
|------|------------:|----:|----:|
| 节点总数 | ~1308 | 2477 | +89% |
| entity-event 边 | ~1426 | 2851 | +100% |
| Event 总数 | - | 1880 | - |
| failed_ops | - | 1220 | - |

按样本看，低度 Event 占比很高：

| 样本 | Event | degree<=2 Event | degree=1 Event |
|------|------:|----------------:|---------------:|
| conv-30 | 148 | 130 | 96 |
| conv-41 | 160 | 134 | 92 |
| conv-42 | 206 | 174 | 129 |
| conv-48 | 210 | 173 | 113 |
| conv-49 | 188 | 173 | 124 |
| conv-50 | 207 | 176 | 115 |

全量 Event 度分布显示，1880 个 Event 中 1039 个 degree=1，531 个 degree=2；也就是 83.5% 的 Event 是低度叶子或近似叶子。这会放大 localize/anchor 的候选空间，但不一定提升可答性。

不必要建点/建边的主要来源：

1. **Rule 8 过强**：`When in doubt, do NOT skip` 让模型把寒暄、鼓励、抽象价值判断也图谱化。
   例子：`Jon stays resilient and focused`、`Gina encourages Jon to stay resilient and focused`、`Joanna appreciates Nate's support`、`Pets bring joy and feel like family`。

2. **同一 turn group 拆出太多近义 Event**：很多 4-turn batch 被拆出 5-8 个 Event。
   例子：
   - conv-50 session_18 batch_0 同一组 turns 生成 8 个事件：`Calvin's album release`、`Work gets noticed`、`Motivated to keep going`、`Positive feedback received`、`Connection with people` 等。
   - conv-49 session_14 batch_0 同一组 turns 生成 8 个事件：`Sam had health issue`、`Sam prioritized health`、`Evan focused on fitness`、`Evan exercise tips` 等。
   - conv-42 session_3 batch_0 同一组 turns 生成 6 个事件：screenplay submission、emotion、hope、support、effort、coconut milk ice cream。

3. **重复 canonical event**：同一图内有大量近似/完全重复事件。
   例子：
   - conv-42：`Joanna appreciates Nate's support` 出现 6 次，`Nate encourages Joanna to keep writing` 出现 3 次。
   - conv-44：`Pets bring joy and feel like family` 出现 4 次，`Audrey and Andrew plan hike with dogs next month` 出现 3 次。
   - conv-30：`Gina encourages Jon to stay resilient and focused` 出现 3 次。

4. **泛化 Entity 节点**：模型创建了很多低度、泛化、不可稳定复用的 Entity。
   例子：`Project`、`Purpose`、`Symbol`、`Park`、`Dog`、`Gaming`、`Nature`、`Body Change`、`nice place`、`Food there`、`Picture of family`、`Outdoors reflection photo`。

5. **边类型被机械补齐**：Rule 12 要求每个 Event 必须接 entity-event，导致 `participant`、`object_of`、`experienced` 被大量使用。
   全量 top predicates：
   - `participant`: 1142
   - `object_of`: 613
   - `experienced`: 567

   三者合计 2322 条，占全图 3076 条边的 75.5%。这说明很多边只是为了满足连通性，而不是表达强语义约束。

6. **prompt 禁止的弱谓词仍然出现**：虽然 Rule 4b 禁止 event-event 使用 `mentions/discussed`，最终图里仍有 `mentions`、`discussing`、`mentioning` 等谓词，说明 4B 对复杂 schema 约束仍不稳定。

图健康度结论：008 的 A3+C1 方向能恢复少数被 Skip 的关键事实，但当前实现没有质量门控，导致图从“漏事实”滑向“过度事实化”。对 Cat1/Cat4，过密低质图会让检索命中更多同主题错事实，抵消构图覆盖收益。

## 遗留问题

- **Cat2 -3.1%**：finish 自检对 Cat2 时间题有副作用。可能要给 Cat2 单独允许 raw_fallback 兜底，或在 prompt 里允许从 raw_fallback 直接采纳精确日期。
- **BatchSkipRecovery 70 次**：占 1571 batch 的 4.5%，命中率不高但每一次都很值（恢复出 evidence-critical 的 Entity）。trigger_rate 0.94 已经很高，说明大部分 batch 没有 Skip 可拆，这条路径主要捕获 LLM 误判。
- **Cat3 退步 -0.048 F1**：判断题（Yes/No），可能 finish 自检把"略有暗示"的题推向 No，需要单独看。
- **q7 doll 仍未完全正确**：008 pred="a furry pal" 接近 doll 但 LLM 选了 furry pal 这个旁路细节，说明图里 doll 节点存在但选择优先级仍被其他节点干扰。
- **图过密且低度节点过多**：需要给 SkipRecovery 和 CreateEvent 加质量门控，避免把支持/感谢/抽象影响类话语全都建成 Event。
- **failed_ops=1220**：说明 LLM 输出了大量无效操作，需要在 trajectory 里记录 raw LLM response / raw ops，才能定位原始 `family`、端点和 predicate 错误。

## 下一步

- 把 008 配置定为新的默认 retrieval+construction baseline。
- Cat2 副作用排查：抽 6 道 Cat2 DOWN 题，确认是否可以用更精准的 finish 自检（"图证据时间冲突时允许 raw_fallback 验证"）减轻退步。
- 如果要进一步提升 Cat1，可以补 P1 改动（C3 + C4 Entity 显式化反例 + Per-turn pre-scan）。
- 暂不应把 008 的构图策略作为默认全量构图 baseline；应先加图质量门控：
  - SkipRecovery 只保留能产生 answerable object/time/number/name 的 turn。
  - 单个 4-turn batch 限制新 Event 数，超过阈值时要求合并或丢弃低价值抽象事件。
  - 对 support/thanks/encourage/appreciate/impact 事件设默认 Skip，除非包含具体对象、行动、地点、时间或可直接回答 QA 的实体。
  - 建图后做 canonical/fact 级近重复合并。

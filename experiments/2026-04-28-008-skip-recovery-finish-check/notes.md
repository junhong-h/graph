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
- build 耗时: TBD
- QA 耗时: TBD
- 图文件: `experiments/2026-04-28-008-skip-recovery-finish-check/build/graphs/`

## 图统计

待填（10 样本）。

## QA 结果

待填。

## 分析

待填。

## 遗留问题

待填。

## 下一步

待填。

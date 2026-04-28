# 实验 2026-04-28-009：fact-first-conv41

## 假设
把 construction 拆成 fact extraction 和 fact-to-op 两阶段后，LLM 第一阶段只判断“有什么事实”，系统负责 `fact_id`、`source_turn_ids`、Event attrs 和 edge family，可减少空/错 family、无意义节点和过度建边。

## 与上次的差异
对比 `2026-04-27-006-qwen-json-conv41`，模型和图检索参数保持一致，只替换构图流程。此次先只跑 `conv-41` 的 build，不跑 QA。

| 维度 | 上次 (based_on) | 本次 |
|------|----------------|------|
| 代码改动 | direct construction | fact-first construction |
| 模型 | qwen3-4b | qwen3-4b |
| jump_budget | 5 | 5 |
| seed_top_k | 5 | 5 |
| 样本 | conv-41 | conv-41 |

## 改动
- `fact_extraction.py` — Stage 1 只输出 `fact` 和 `dialogue_time`，系统补追踪字段。
- `fact_construction.py` — Stage 2 输出高层 `EnsureEntity/EnsureEvent/Relate`，系统编译落图。
- `fact_construction.py` — compiler repair 未定义的 `E*/V*` ref，避免 LLM 漏写 `Ensure*` 时产生 unresolved edge。
- `graph_builder.py` — `construction.mode: fact_first` 时使用 8-turn chunks 并行抽事实，op/apply 串行。

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-009-fact-first-conv41 --workers 1
```

## 运行环境
- commit: `97c4fb7`
- build 耗时: ~
- QA 耗时: 未跑
- 图文件: `experiments/2026-04-28-009-fact-first-conv41/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | | | | | |

## QA 结果

| Category | Baseline (<based_on>) | 本次 | 变化 |
|----------|-----------------------|------|------|
| Cat1     | | | |
| Cat2     | | | |
| Cat3     | | | |
| Cat4     | | | |
| Cat1-4   | | | |
| Cat5     | | | |
| Avg F1   | | | |

## 分析
待 build 完成后先看图规模、failed ops、facts 数量和 trajectory raw responses。

## 遗留问题
- 这次只验证构图健康度，不验证 QA 准确率。

## 下一步
如果图规模和边类型健康，再跑同样 `conv-41` 的 QA 对照 `2026-04-27-006`。

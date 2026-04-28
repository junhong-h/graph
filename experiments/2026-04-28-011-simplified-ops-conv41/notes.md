# 实验 2026-04-28-011：simplified-ops-conv41

## 假设

与 009（fact-first）对比，新的单阶段简化 schema 应能在 conv-41 上产生更健康的图结构：Entity 占比更高、Event 节点度分布更分散（degree≥3 的节点更多），避免 009 中 99% Event 节点 degree≤2 的碎片化问题。

## 与上次的差异

| 维度 | 009 (fact-first) | 本次 |
|------|-----------------|------|
| 架构 | 两阶段：先抽事实再建图 | 单阶段：直接从对话建图 |
| op schema | CreateEntity/CreateEvent/Link | EnsureEntity/EnsureEvent/Relate |
| Relate family | 手动指定 | 自动从节点类型推断 |
| Event attrs | fact/time/quote/source | fact/time 只需两项 |
| Rule 8 (skip) | 宽泛 | 明确三类 skip |
| 模型 | qwen3-4b | qwen3-4b |
| 样本 | conv-41 | conv-41 |

## 改动

- `graph_construction.py:_SYSTEM_PROMPT` — 新 op schema，精简 prompt，收窄 Rule 8
- `graph_construction.py:_do_relate` — 自动推断 edge family
- `graph_builder.py:_format_batch_text` — 新对话预处理格式

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-011-simplified-ops-conv41
```

## 运行环境
- commit: `748dcf9`
- build 耗时: ~
- QA 耗时: ~（N 题）
- 图文件: `experiments/YYYY-MM-DD-NNN-<name>/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-26 | | | | | |

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
<!-- 结果符合预期吗？哪里超预期/低于预期？根因是什么？ -->

## 遗留问题
<!-- 发现的新问题，或还未解决的问题 -->

## 下一步
<!-- 基于这次结果，下一步做什么 -->

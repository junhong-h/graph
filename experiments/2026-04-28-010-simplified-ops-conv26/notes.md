# 实验 2026-04-28-010：simplified-ops-conv26

## 假设

简化 op schema 后（EnsureEntity/EnsureEvent/Relate 替代 CreateEntity/CreateEvent/Link），配合更干净的对话预处理格式和收窄的 Rule 8（明确三类 skip），LLM 应该更容易理解 op 格式，减少无效操作，同时 Entity/Event 比例更合理（Entity 不过少，Event 不过多）。

## 与上次的差异

| 维度 | 上次 (2026-04-28-008) | 本次 |
|------|----------------------|------|
| op schema | CreateEntity/CreateEvent/Link | EnsureEntity/EnsureEvent/Relate |
| Relate family | 需手动指定 | 自动从节点类型推断 |
| Event attrs | 需要 quote/source | 只需 fact/time |
| 对话输入格式 | per-turn header 格式 | [session_time]\n\nSpeaker: text |
| Rule 8 (skip) | 宽泛（"when in doubt, don't skip"） | 明确三类：social reactions/abstract beliefs/conversational acts |
| 模型 | qwen3-4b | qwen3-4b |
| 样本 | 10 samples | conv-26 单样本 |

## 改动

- `graph_construction.py:_SYSTEM_PROMPT` — 新 op schema，精简 prompt，收窄 Rule 8
- `graph_construction.py:_do_relate` — 新方法，自动推断 edge family
- `graph_construction.py:_dispatch` — 支持 EnsureEntity/EnsureEvent/Relate
- `graph_builder.py:_format_batch_text` — 新对话预处理格式

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-010-simplified-ops-conv26
```

## 运行环境
- commit: `224d1bc`
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

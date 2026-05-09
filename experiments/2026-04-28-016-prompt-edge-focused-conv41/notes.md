# 实验 2026-04-28-016：prompt-edge-focused-conv41

## 假设
放松 fact 完美转述要求后，模型可以保留一些低价值或不够干净的 Event；只要 prompt 明确要求主体 Entity→Event 边、命名对象/伴随者 Entity→Event 边和 planned 谓词，图结构应更适合 retrieval。

## 与上次的差异

| 维度 | 上次 (based_on) | 本次 |
|------|----------------|------|
| 代码改动 | 015 edge guard + LLM log | `src/graphmemory/graph_construction.py` — prompt 收敛到 retrieval-friendly graph，弱化 fact 完美性，强化 Entity→Event 对应 |
| 模型 | qwen3-4b | qwen3-4b |
| jump_budget | 5 | 5 |
| seed_top_k | 5 | 5 |
| 样本 | conv-41 | conv-41 |

## 改动
- `src/graphmemory/graph_construction.py:_SYSTEM_PROMPT` — 移除 concrete few-shot 示例；明确 Event fact 可不完美，但必须包含关键 subject/object 信息。
- `src/graphmemory/graph_construction.py:_SYSTEM_PROMPT` — 强化每个 Event 的主体边、命名对象/伴随者边、planned 谓词和禁止 Entity→Entity。

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-016-prompt-edge-focused-conv41 --log-level INFO
# 本次不跑 QA
```

## 运行环境
- commit: `d54912e+worktree`
- build 耗时: ~5m53s
- QA 耗时: 未运行
- 图文件: `experiments/2026-04-28-016-prompt-edge-focused-conv41/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | 89 | 104 | 11 | 78 | 101 |

## QA 结果
本次只建图，不跑 QA。

## 分析
结构不合格。没有 Entity→Entity 边，但仍有 3 个 Event 没有 Entity 入边，6 个 unused Entity。session30 的 construction 原始输出把已有 Event ID `0b08d85e` 当成 Entity 使用，导致多条 Entity→Event 语义边被执行层按 event-event 拒绝，后续只能依赖 repair 补部分边。

session30 关键问题：
- Coco 事件 fact 写成 `John and Maria adopted Coco two weeks ago`，主体错。
- Camping/reflecting/volunteering 的 subject edge 原始输出使用了 Event ID，执行被拒绝。
- 最终图依赖 repair，缺少完整对象边。

## 遗留问题
prompt 对 existing local subgraph 的 node type 约束不够。模型看到相关 Max 旧 Event 后，把 Event ID 当成 John/Max 的 Entity anchor 使用。

## 下一步
新建 017，只增强 prompt：复用已有 ID 时必须匹配显示的 node type；Event ID 不能作为 Entity→Event 的 src；缺失 subject/object Entity 时必须 EnsureEntity。

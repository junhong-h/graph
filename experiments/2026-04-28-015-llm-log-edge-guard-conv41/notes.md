# 实验 2026-04-28-015：llm-log-edge-guard-conv41

## 假设

014 已消除 `related` 和非法 event-event，但结合原始 context 发现主要剩余问题是 Entity↔Event 错连：听者被连成经历者/对象，repair 在 predicate 被拒后会按 first speaker 盲连。

本次增加 LLM 调用记录，方便回溯 prompt/response；同时收紧 Entity↔Event 写入和 repair：

- 每次 LLM 调用写入 `build/llm_calls.jsonl`，包含 metadata、messages、response。
- Entity↔Event 写边前要求 Entity 名字出现在 Event `canonical_name` 或 `fact`。
- `planned` 作为合法 Entity↔Event predicate。
- repair 不再创建/使用 first speaker 兜底，只在 Event 文本命中已有 Entity 时补边。

本次只验证建图结构，不跑 QA。

## 与上次的差异

| 维度 | 上次 (2026-04-28-014) | 本次 |
|------|----------------------|------|
| LLM 调用记录 | 无完整请求/响应记录 | 写入 `build/llm_calls.jsonl` |
| Entity↔Event 写边 | 只校验 predicate 白名单 | 额外要求 Entity 名字出现在 Event name/fact |
| `planned` predicate | 被拒绝 | 合法 |
| repair | 可能按 first speaker 盲连 | 只按 Event 文本命中已有 Entity 补边 |
| 模型 | qwen3-4b | qwen3-4b |
| k_turns | 999 | 999 |
| 样本 | conv-41 | conv-41 |

## 改动

- `src/graphmemory/llm_client.py` — 支持 `call_log_path`，记录 LLM request/response。
- `scripts/build_memory.py` — build 模式下写 `llm_calls.jsonl`。
- `src/graphmemory/graph_trigger.py` / `src/graphmemory/graph_builder.py` / `src/graphmemory/graph_construction.py` — 为 LLM 调用附加 sample/batch/session/phase metadata。
- `src/graphmemory/graph_construction.py` — 收紧 Entity↔Event 连边，允许 `planned`，repair 不再盲连 first speaker。
- `tests/` — 覆盖 LLM 调用日志、planned predicate、听者错连拒绝、repair 不盲连。

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-015-llm-log-edge-guard-conv41
```

## 运行环境

- commit: `d54912e+worktree`
- build 耗时: ~4m56s
- QA 耗时: 未运行
- 图文件: `experiments/2026-04-28-015-llm-log-edge-guard-conv41/build/graphs/`
- LLM 调用记录: `experiments/2026-04-28-015-llm-log-edge-guard-conv41/build/llm_calls.jsonl`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 | related 边 | 非法 e-ev |
|------|------|-----|--------|-------|---------|------------|-----------|
| conv-41 | 98 | 116 | 8 | 90 | 4 | 0 | 0 |

## QA 结果

未运行。

## 分析

LLM 调用记录已生成：

- `build/llm_calls.jsonl`
- 共 64 条调用：32 条 trigger，32 条 construction。
- 每条包含 `sample_id`, `batch_id`, `session_id`, `phase`, request messages 和 response。

图结构变化：

| 指标 | 014 | 015 |
|------|-----|-----|
| 节点 | 94 | 98 |
| 边 | 150 | 116 |
| Entity | 11 | 8 |
| Event | 83 | 90 |
| entity-event | 142 | 112 |
| event-event | 8 | 4 |
| related | 0 | 0 |
| 非法 event-event | 0 | 0 |

本次确实减少了听者错连：Entity 名字不出现在 Event name/fact 的 Entity↔Event 被拒绝，轨迹中有 25 条 `experienced`、5 条 `attended`、2 条 `object_of` 因此被拒绝。

但 fact 本身仍有明显问题：

- 部分事实主体错：`John Had a Puppy` 的 fact 写成 John had Coco，但原文是 Maria got Coco。
- 部分时间错：`John Shared Max's Photo` / Max 相关事件把 “10 years with family” 写成 “passed away 10 years ago”。
- 部分对话行为被事实化：`John and Maria Talk About...`、`John Shared...Photo` 仍进入 Event。
- 部分概括过度：session 21 中 `Maria and John Received Letter from Shelter Resident Laura` 把 Maria 收到信扩成 Maria and John。

所以当前主要瓶颈已经从“弱边/非法边”转向“fact 抽取本身的主体和时间约束不稳”。

## 遗留问题

- Event fact 必须更严格地绑定 utterance speaker，不应把 listener 写进事实主体。
- 相对时间不能误解，例如 “with family for 10 years” 不能变成 “10 years ago”。
- 对话行为和图片分享类 Event 仍偏多。
- Entity 数从 014 的 11 降到 8，说明严格连边后部分 Entity 没有稳定保留。

## 下一步

- 下一轮应优先修 fact schema：显式区分 `speaker`, `subject`, `object_entities`, `event_time`, `mentioned_at`。
- prompt 中要求每个 Event fact 必须能由具体 turn 原文支持，并保留短 `evidence_quote`。
- 对 construction 输出做后校验：Event canonical/fact 的主语若与 source speaker 或显式实体冲突，拒绝或重试。

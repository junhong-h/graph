# 实验 2026-04-28-014：predicate-guard-conv41

## 假设

013 中 `Relate` 示例缺少 `predicate`，执行层又默认补成 `related`，导致大量低信息边进入图。补全示例 predicate，并在写边前拒绝 `related` 和 family 不匹配的 predicate，应能显著减少弱语义边和错误 event-event 边。

本次只验证建图结构，不跑 QA。

## 与上次的差异

| 维度 | 上次 (2026-04-28-013) | 本次 |
|------|----------------------|------|
| Relate 示例 | 示例无 predicate | 示例显式使用 `experienced` / `object_of` |
| predicate 校验 | 无硬校验，缺省为 `related` | 写边前拒绝 `related`、空 predicate 和非法 family-predicate |
| 模型 | qwen3-4b | qwen3-4b |
| k_turns | 999 | 999 |
| 样本 | conv-41 | conv-41 |

## 改动

- `src/graphmemory/graph_construction.py:_SYSTEM_PROMPT` — 补全 Relate 示例 predicate，避免模型照抄无 predicate 示例。
- `src/graphmemory/graph_construction.py:_is_allowed_predicate` — 新增 predicate 白名单校验。
- `src/graphmemory/graph_construction.py:_do_relate/_do_link/_do_add_edge` — 写边前拒绝非法 predicate。
- `tests/test_graph_construction.py` — 覆盖缺失 predicate 和非法 event-event predicate 被拒绝。

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-014-predicate-guard-conv41
```

## 运行环境

- commit: `d54912e+worktree`
- build 耗时: ~6m23s
- QA 耗时: 未运行
- 图文件: `experiments/2026-04-28-014-predicate-guard-conv41/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 | related 边 |
|------|------|-----|--------|-------|---------|------------|
| conv-41 | 94 | 150 | 11 | 83 | 8 | 0 |

## QA 结果

未运行。

## 分析

与 013 对比：

| 指标 | 013 | 014 | 变化 |
|------|-----|-----|------|
| 节点 | 74 | 94 | +20 |
| 边 | 161 | 150 | -11 |
| Entity | 5 | 11 | +6 |
| Event | 69 | 83 | +14 |
| event-event | 18 | 8 | -10 |
| related 边 | 92 | 0 | -92 |
| 非法 event-event predicate | 17 | 0 | -17 |
| Event degree <= 1 | 33/69 | 44/83 | 低度 Event 仍偏多 |
| Event degree <= 2 | 46/69 | 62/83 | 低度 Event 仍偏多 |

本次改动达成了主要目标：

- `related` 弱语义边被完全消除。
- `event-event` 中的 `related/experienced/participant/object_of` 等非法 predicate 被拒绝，最终非法 event-event 为 0。
- Entity 覆盖有明显改善，新增了 `Coco`, `David`, `Cindy`, `Laura`, `Samuel`, `Organization` 等节点。

但新问题也很明确：

- Event 数从 69 增到 83，说明 prompt 仍倾向拆出较多细粒度 Event。
- event-event 从 18 降到 8，虽然语义更干净，但跨事件连通性更弱。
- 轨迹中有 39 条 event-event `experienced/object_of` 被拒绝，说明模型仍在把 Entity↔Event predicate 用到 Event↔Event 上，prompt 还需要进一步强调 endpoint 类型。
- 仍有少量命名对象没有建 Entity：`little girl`, `instructor`。

## 遗留问题

- Event 粒度偏碎，低 degree Event 比例仍高。
- prompt 仍不能稳定让模型区分 Entity↔Event 和 Event↔Event predicate。
- `focused_on/planned` 这类有用但不在白名单内的 Entity↔Event predicate 被拒绝，后续需要决定是扩展白名单还是要求模型映射到 `experienced/decided/started`。
- 命名 Entity 覆盖仍未完全解决。

## 下一步

- 调整 prompt：明确 Relate 的两端类型决定 predicate 集合，并给出 Event↔Event 正反例。
- 继续修 Entity 覆盖：对 Event fact/canonical_name 中出现但未建节点的 named entity 做构建后补救。
- 继续收紧低价值 Event 过滤和 Event 合并，避免图因过多 degree=1 Event 变碎。

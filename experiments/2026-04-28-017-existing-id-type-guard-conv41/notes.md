# 实验 2026-04-28-017：existing-id-type-guard-conv41

## 假设
016 中模型把 local subgraph 里的已有 Event ID 当成 Entity ID 使用，导致 Entity→Event 连边被执行层拒绝。若 prompt 明确要求复用现有节点时必须匹配 type，且 Event ID 不能作为 Entity→Event 的 src，应能减少错连和 repair 依赖。

## 与上次的差异
<!-- 对比 based_on 实验，本次改动了什么。未列出 = 与上次相同 -->

| 维度 | 上次 (based_on) | 本次 |
|------|----------------|------|
| 代码改动 | 016 prompt-edge-focused | `src/graphmemory/graph_construction.py` — 增加 existing node type guard |
| 模型 | qwen3-4b | qwen3-4b |
| jump_budget | 5 | 5 |
| seed_top_k | 5 | 5 |
| 样本 | conv-41 | conv-41 |

## 改动
- `src/graphmemory/graph_construction.py:_SYSTEM_PROMPT` — 明确 Event ID 不能当 Entity 使用；缺少 John/Maria/Coco/Max 等 Entity 时必须 EnsureEntity。

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-017-existing-id-type-guard-conv41 --log-level INFO
# 本次不跑 QA
```

## 运行环境
<!-- commit hash 务必填写 -->
- commit: `d54912e+worktree`
- build 耗时: ~4m17s
- QA 耗时: 未运行
- 图文件: `experiments/2026-04-28-017-existing-id-type-guard-conv41/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | 95 | 110 | 17 | 78 | 110 |

## QA 结果
本次只建图，不跑 QA。

## 分析
结构层明显改善：没有 Entity→Entity 边、没有无 Entity 入边的 Event，rejected op 从 016 的 81 降到 36，RepairEventLink 从 27 降到 9。说明 existing node type guard 有效，模型较少把 Event ID 当成 Entity ID。

但事实覆盖变差。session30 只输出了 `Maria got a new puppy named Coco two weeks ago`，漏掉 Max camping、John feeling stuck、John exploring volunteering/local organizations。Coco Entity 被创建，但模型先输出了 Maria→Coco 的 entity-entity `object_of`，被拒绝，最终 Coco 没连到 puppy Event。

## 遗留问题
仅靠“ID type guard”会让模型保守输出，减少错连但牺牲关键事实覆盖。还需要在 prompt 中显式要求每个 excerpt 至少检查并覆盖四类 durable facts：acquisition/possession、named trip/activity、current personal state/problem、plan/intention。

## 下一步
下一步应在 prompt 中加入短的 coverage checklist，但不要加入具体 few-shot 内容；重点让模型输出 `EnsureEvent` 后马上输出 subject/object edges，避免先连 Entity→Entity。

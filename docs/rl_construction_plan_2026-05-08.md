## GraphMemory：基于 Operation 的 RL 路线讨论

日期：2026-05-08
范围：从当前 construction operation 体系出发，梳理研究现状、核心问题、对 RL 的影响，以及下一步改动建议。
对象：为后续基于 operation 的 RL 训练做技术决策。

---

## 1. 当前 research 状况（截至 2026-04-28）

### 1.1 系统 = Build + Retrieval，operation 集合已收敛

**Operation 集合**（`src/graphmemory/graph_construction.py`）只剩 6 类：

| Op | 作用 |
|----|------|
| `EnsureEntity` | 创建/复用实体节点（人/物/地点/组织） |
| `EnsureEvent` | 创建事件节点（含 fact / time） |
| `Relate` | 节点间连边，family 由节点 type 自动推断 |
| `AttachAttr` | 给现有节点挂属性 |
| `MergeNode` | 合并同指节点 |
| `Skip` | 当前 batch 无可建图事实 |

旧 ops（`CreateEntity` / `CreateEvent` / `Link` / `AddEdge` / `ReviseAttr` / `DeleteEdge` / `PruneNode` / `KeepSeparate`）已弃用。

**Predicate 白名单**：

| Family | 允许 predicate |
|--------|---------------|
| entity-event | `participant` / `experienced` / `owns` / `attended` / `visited` / `decided` / `started` / `planned` / `achieved` / `object_of` |
| event-event | `before` / `after` / `updates` / `inspired` |
| entity-entity | `same_as` / `family_of` / `friend_of` / `colleague_of` / `owns` |

**执行层 guard**（已经在做 reward-shaping 该做的事）：

- 拒绝 `related` / 空 predicate / family-predicate 不匹配 → `status=rejected`
- 拒绝 Entity 名不在 Event text 的 entity-event 边
- 拒绝 session-container Event（`chat` / `discussion` / `talk to`）
- 拒绝低价值抽象 Event（kindness / positivity / animals comfort 等）
- `_repair_created_events`：补 batch_id / source_turn_ids / time，并在 Event 没有 entity-event 边时补一条

### 1.2 评测现状

Mem-T 对齐口径：`conv-41/42/43/44/47/48/49/50`，Cat1-4，1307 题。

| 配置 | Cat1-4 F1 | 备注 |
|------|----------:|------|
| GPT-4o（默认） | 0.51 | |
| Qwen3-4B | 0.39 | |
| Oracle（gold evidence + Qwen） | 0.58 | 检索/图损失 ~0.19 |
| 008 全量（Qwen, all 10 samples） | Cat1-4 86.2% / Cat5 86.3% | Cat5 修好但图严重膨胀 |

### 1.3 实验脉络

- **2026-04-20** P0+P1：靠 prompt 把 Cat1-4 推到 87.3%
- **004 refine2 proposition**：fact / quote 字段 → Cat5 提升明显
- **007 Cat5 fix**：屏蔽 Cat5 的 `raw_fallback` → Cat5 87.5%
- **008 SkipRecovery**：全 10 样本，但图爆炸（2477 节点 / 1880 Event / 83.5% Event degree≤2）
- **009-017**：从「调检索」转「重做 construction」，每个 prompt 改动顾此失彼

---

## 2. 核心问题

### 2.1 结构层面的"图不健康"

| 病症 | 量化 | 根因 |
|------|------|------|
| Event 叶子化 | 008 conv-41 Event 160 中 92 个 degree≤1 | 子图重复抽事实、缺去重 |
| Entity 过少 / 过泛 | session-level 仅 4 Entity；008 出现 Project / Nature / Gaming 等 | 模型自由度过大 |
| 边机械补齐 | participant / object_of / experienced 占 75.5% | repair 强制补边、prompt 要求每 Event 必有边 |
| Fact 主体错误 | "Maria got Coco" → "John and Maria adopted Coco" | Listener 被错连成 subject |
| 时间错乱 | "with family for 10 years" → "passed away 10 years ago" | duration / event time / 对话时间混淆 |
| Local subgraph 污染 | 016 把 Event ID 当 Entity src | LLM 把旧 Event 当事实源 |

### 2.2 prompt-only 的硬上限

已观测的此消彼长：

- **Type guard 严** → 结构干净但 recall 下降（017）
- **Coverage checklist 强** → recall 上升但重复 / 污染翻倍（008）
- **Skip 收紧** → 图爆炸；**Skip 放松** → 漏关键事实（007 vs 008）
- **session-level** → 调用少但 Entity 不足；**k=4** → recall 好但 batch 间重复

### 2.3 当前 LLM I/O 的设计问题

- **Action 是 mixed**：一次输出里既有 id 创建（`EnsureEntity`）、又有关系绑定（`Relate`）、又有属性挂载（`AttachAttr`）。LLM 必须同时完成「事实抽取 + 实体抽取 + 关系类型分类 + 时态判断 + ID 复用」5 件事。
- **State 是「全量子图 + 原文」**：LLM 看到 local subgraph 后会把旧 Event 当输入抄一遍。
- **Reward 信号稀疏**：只有最终 QA F1，op 层面没有任何对齐信号。

---

## 3. 对 RL 的影响

### 3.1 RL 能解决的

1. **prompt 颠簸的此消彼长**：RL 可以同时优化 recall + structural validity，prompt-only 做不到。
2. **执行层 rejected/ok 是免费的 step-level reward**：rejected predicate / rejected entity-event edge / rejected low-value event 已经在产出二元信号，可以直接用作 dense shaping reward。
3. **Subject / object 边正确性**：可以用 fact 文本 + Entity 命名匹配规则做 verifiable reward。
4. **Skip vs 抽事实的 trade-off**：RL 比 prompt 阈值更适合学这种连续判断。

### 3.2 RL 也解决不了 / 会加剧的

1. **State 污染**：RL 不能修「LLM 抄 local subgraph」这件事 —— 这是 input 层面的设计问题，得改成「Entity table only」。
2. **Mixed action space**：当前 op 把"事实抽取 + ID 复用 + 关系分类"耦合在一起，RL 信用分配会很难。建议先**解耦成两阶段**（fact extraction → deterministic compilation）再 RL。
3. **Outcome reward 太远**：QA F1 距离 op 序列 5+ 步，不做 step-level dense reward，RL 收敛会非常慢。
4. **Reward hacking 风险**：如果只用 `rejected_op` 比例 + Event 数作为 reward，模型会学会大量 Skip 或大量低价值 Event 但不被 reject 的输出。
5. **数据效率**：Qwen3-4B 单 sample build 已经 4-5min，每个 trajectory 包含 30-200 个 batch op，RL rollout 成本极高。

### 3.3 当前 codebase 对 RL 的友好度

**好的部分**：

- `build/llm_calls.jsonl`（015 起）已经记录完整 LLM call：messages / response / metadata，是 SFT 数据集现成来源
- 每个 op 在 `_dispatch` 里产出 `{op, status, error}` 结构化日志，天然能做 step reward
- experiment 目录隔离做得好，可以 reproducible rollout

**差的部分**：

- 没有"gold operation trace"标注，只有最终 QA gold answer
- 没有 fact-level alignment（Event fact ↔ LoCoMo evidence 对齐）
- `GraphConstructor.run()` 是同步 batch 流，要改成可中断 / per-op 步进才能做 online RL
- 现在的 LLM client 不区分 logprob，没准备 RL 训练栈

---

## 4. 下一步改动建议（按优先级）

### Stage 0：封板 prompt-only 路线，记录 baseline（1-2 天）

- 用 015 的 v3 prompt + k=4 + Skip 收窄，跑全 10 样本 → 这是 RL 的 SFT 起点 baseline
- 输出物：`build/llm_calls.jsonl`（已有）+ 每个 op 的 status 标签

### Stage 1：解耦 LLM action（RL 前提，不是 RL 本身）（3-5 天）

把 LLM 输出从「图操作」改为「标准化 fact JSON」：

```json
{
  "facts": [
    {
      "fact": "Maria got a puppy named Coco two weeks ago.",
      "subject": "Maria",
      "predicate": "experienced",
      "objects": [{"name": "Coco", "role": "object_of"}],
      "time": "two weeks ago",
      "source_turn_ids": ["D30:1"]
    }
  ]
}
```

然后 deterministic compiler 把它编译为 `EnsureEntity / EnsureEvent / Relate`。

理由：

- LLM 不再操作图 ID，根除「Event ID 当 Entity」的失败模式
- action 维度从 6 类 op 降到 1 个结构化 schema，RL 信用分配清晰
- 编译器层做 dedup / 名称匹配 / 类型一致性，结构正确性变成代码保证而不是 prompt 押宝

### Stage 2：重做 state，斩断污染源（2-3 天）

把 prompt 中 `[Current local subgraph]` 改为只给：

```text
[Reusable entities]
| id_prefix | canonical_name | aliases |
| 1cf8b8a4  | John           | -       |
| cdf87837  | Maria          | -       |
```

不给 Event 列表，不给边。LLM 只能从这里复用 ID。新事实必须从 `[Input excerpt]` 抽。

### Stage 3：定义 verifiable rewards（2-3 天）

搭三层 reward 信号（用于 RL 而不只是 SFT）：

1. **Step reward（dense）**：来自现有 guard
   - `rejected_op` = -1
   - low-value rejected = -0.5
   - `ok` = +0.1
   - subject 名出现在 Event fact = +0.5
2. **Fact-level reward（mid）**：与 LoCoMo evidence 对齐
   - 每个 Event fact 与 gold evidence 算 ROUGE-L / 命名实体重合
3. **Outcome reward（sparse）**：QA F1 / Cat 分项 acc

### Stage 4：先 SFT 再 RL（1-2 周）

- 从 008 / 015 的 LLM call log 中筛选 status 全 ok 且 fact-level 对齐高的轨迹做 SFT
- SFT 后再上 GRPO / PPO，用 Stage 3 的混合 reward
- 单 sample build 成本高 → 先用 conv-41 + conv-26 两个 sample 做 RL pilot，验证后再扩 10 样本

### Stage 5：Session-level consolidation（独立线，与 RL 并行）

- 不是 RL 的一部分，但是图健康度的关键
- 在每个 session 结束时跑一遍 dedup：同 subject + 高 fact 相似度 + 时间一致 → `MergeNode`
- 这步用启发式比 LLM 划算

---

## 5. 一句话总结

当前 prompt-only 的 op 体系已经收敛到只剩 6 类、有完整 guard 和 status 日志，RL-ready 的部分已经具备；但要真正做 RL 必须先解决 **action 解耦**（fact JSON → deterministic compile）和 **state 污染**（只给 entity table）这两个上游问题，否则 RL 在 mixed-action + 污染 state 上无法收敛。Reward 信号在现有 guard 基础上扩到三层（step / fact / outcome）即可。

---

## 6. 相关资料索引

- 现状讨论：`docs/graphmemory_rebuild_discussion_2026-04-28.md`
- 算法说明：`ALGORITHM.md`
- 实验汇总：`experiments/README.md`
- LLM call log（RL 数据源候选）：`experiments/2026-04-28-015-llm-log-edge-guard-conv41/build/llm_calls.jsonl`
- 当前 op 实现：`src/graphmemory/graph_construction.py`
- 执行层 guard：`_is_allowed_predicate` / `_invalid_entity_event_edge` / `_is_session_container_event` / `_is_low_value_abstract_event`

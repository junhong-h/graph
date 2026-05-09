## Op Schema 重设计：为什么废弃旧 ops

日期：2026-04-28
对应 commit：`224d1bc` "Simplify graph construction: new op schema, clean dialogue input format, tighter Rule 8"
对应实验：`experiments/2026-04-28-010-simplified-ops-conv26` / `experiments/2026-04-28-011-simplified-ops-conv41`
被废弃的旧 ops：`CreateEntity` / `CreateEvent` / `Link` / `AddEdge` / `ReviseAttr` / `DeleteEdge` / `PruneNode` / `KeepSeparate`

---

## 1. 新旧 schema 对照

### 旧 schema（commit 224d1bc 之前）

构建侧（5 类）：

```json
{"op": "CreateEntity", "id": "NEW_<label>", "canonical_name": "...", "aliases": [...], "attrs": {...}}
{"op": "CreateEvent",  "id": "NEW_<label>", "canonical_name": "...", "attrs": {"fact": "...", "quote": "...", "source": [...], "time": "..."}}
{"op": "Link",         "src": "...", "dst": "...", "family": "entity-event|entity-entity|event-event", "predicate": "..."}
{"op": "AttachAttr",   "node": "...", "key": "...", "value": "..."}
{"op": "Skip",         "reason": "..."}
```

更新侧（6 类）：

```json
{"op": "MergeNode",    ...}
{"op": "ReviseAttr",   ...}
{"op": "AddEdge",      ...}
{"op": "DeleteEdge",   ...}
{"op": "PruneNode",    ...}
{"op": "KeepSeparate", ...}
```

### 新 schema（224d1bc 之后）

```json
{"op": "EnsureEntity", "id": "NEW_<label>", "canonical_name": "...", "aliases": []}
{"op": "EnsureEvent",  "id": "NEW_<label>", "canonical_name": "...", "attrs": {"fact": "...", "time": "..."}}
{"op": "Relate",       "src": "...", "dst": "...", "predicate": "..."}
{"op": "AttachAttr",   "node": "...", "key": "...", "value": "..."}
{"op": "MergeNode",    "src": "...", "dst": "..."}
{"op": "Skip",         "reason": "..."}
```

只剩 6 类。`AddEdge / ReviseAttr / DeleteEdge / PruneNode / KeepSeparate` 在 `_dispatch` 里直接标 deprecated 跳过。

---

## 2. 重设计的 5 个动机

### 2.1 `Create*` → `Ensure*`：解决重复创建

**问题**：`Create*` 语义是"无脑新建"，LLM 即使在 local subgraph 里看到 John 也会再 `CreateEntity` 一个 John，导致重复节点。

**新方案**：`Ensure*` 是 idempotent 的 —— 存在则复用，不存在则创建。

010 notes 原话：

> EnsureEntity/EnsureEvent are idempotent

把"复用现有节点"从 prompt 押宝转成代码保证（`_do_create_node` 里 `_find_existing_entity` 命中就复用并返回）。

### 2.2 `Link` → `Relate`：family 不再让 LLM 手填

**问题**（对应 `docs/graphmemory_rebuild_discussion_2026-04-28.md` 第 2.2 节）：

- LLM 经常填空 family
- LLM 经常填错 family（明明 src 是 Entity / dst 是 Event，却写 entity-entity）
- family + predicate 不匹配（family=event-event 但 predicate=participant）
- 008 全量 build 出现 92 次空 family

**新方案**：`Relate` 不接受 family 字段，由 `_do_relate` 用 src/dst 节点 type 自动推断。

010 notes 原话：

> Relate family 需手动指定 → 自动从节点类型推断

LLM 只需输出 predicate，family 错误从根上消灭。

### 2.3 Event attrs 从 `fact + quote + source` 简化为 `fact + time`

**旧要求**：每个 Event 必含 4 个属性

```json
{"attrs": {"fact": "...", "quote": "...", "source": [...], "time": "..."}}
```

**问题**：

- `quote` 经常被 LLM 复制粘贴整段 utterance，意义不大
- `source` 由 batch_id / turn_id 上下文可推得，让 LLM 输出反而出错
- 4 件套增加 schema 复杂度，LLM 经常漏字段被 reject

**新方案**：

```json
{"attrs": {"fact": "...", "time": "..."}}
```

`source_turn_ids` 由 `_repair_created_events` 在执行后从 context 自动补，不再让 LLM 输出。

### 2.4 删除 5 个低使用率的 update ops

旧设计有 8 类 update ops。实际使用情况（从代码 `_dispatch` 看）：

```python
elif name in ("AddEdge", "ReviseAttr", "DeleteEdge", "PruneNode", "KeepSeparate"):
    logger.debug(f"Deprecated op ignored: {name}")
    return {"op": name, "status": "deprecated"}
```

直接标 deprecated 跳过。说明 LLM 几乎从不用这几个 op，留着只是 prompt 噪声。

最终保留：

| 保留 | 理由 |
|------|------|
| `MergeNode` | 同指节点合并，仍有用 |
| `AttachAttr` | 属性补充，仍有用 |

合并：

| 合并去向 | 理由 |
|---------|------|
| `AddEdge` → `Relate` | 两者都是连边，合并避免冗余 |

删除：

| 删除 | 理由 |
|------|------|
| `ReviseAttr` | LLM 几乎不用，与 AttachAttr 功能重叠 |
| `DeleteEdge` | LLM 几乎不用，需要 edge_id 太复杂 |
| `PruneNode` | LLM 几乎不用，删节点风险高 |
| `KeepSeparate` | LLM 几乎不用，意图记录无落图操作 |

### 2.5 Rule 8（Skip）从模糊变明确

**旧 prompt**：

> Output Skip ONLY if every turn in the excerpt contains no concrete personal fact whatsoever. **When in doubt, do NOT skip** — extracting a low-value fact is recoverable, skipping a high-value fact is not.

**问题**：008 图爆炸（83.5% Event degree≤2），因为 LLM 把 "kindness matters" 这种也建图了。

**新 prompt** 明确三类 skip：

> (a) social reactions — thanking / praising / encouraging
> (b) abstract values or beliefs — aspirations / life philosophies
> (c) conversational acts — sharing a photo / mentioning / discussion

---

## 3. 设计哲学

把 LLM 的负担从 **"输出无歧义图操作"** 降为 **"输出最小化语义意图"**，剩下的 family 推断、节点复用、source 补全、predicate 校验全部移到代码层。

| 维度 | 旧设计 LLM 负担 | 新设计 LLM 负担 |
|------|----------------|----------------|
| 节点复用判断 | LLM 决定 Create vs 不 Create | 代码决定（Ensure 自动） |
| Edge family | LLM 输出 | 代码从 type 推断 |
| Event source 字段 | LLM 输出 | 代码从 context 补 |
| Event quote 字段 | LLM 输出 | 删除 |
| Update ops（5 类） | LLM 选择 | 删除 |

---

## 4. 与 RL 路线的关系

224d1bc 已经把"代码能确定性算出的部分"剥离了一层（family 推断、source 补全、节点复用判断）。

这其实是当前 `docs/rl_construction_plan_2026-05-08.md` Stage 1 提到的 **"action 解耦"** 思想的雏形 —— 把这个思路推到底就是：**让 LLM 完全不操作图 ID 和图操作，只输出 fact 结构**，剩下全靠 deterministic compiler。

| 阶段 | LLM 输出粒度 |
|------|-------------|
| 224d1bc 之前 | 完整图操作（含 family / source / quote） |
| 224d1bc 之后（当前） | 简化图操作（只 op + ID + predicate） |
| RL 路线 Stage 1（建议） | 标准化 fact JSON（subject / predicate / objects / time） |

每一步都是把 LLM action space 进一步收窄，把可确定性的部分下沉到代码层。

---

## 5. 引用位置

- Commit：`git show 224d1bc`
- 实验 notes：`experiments/2026-04-28-010-simplified-ops-conv26/notes.md:9-18`
- 实验 notes：`experiments/2026-04-28-011-simplified-ops-conv41/notes.md`
- deprecated 处理：`src/graphmemory/graph_construction.py:319-321`
- family 自动推断：`src/graphmemory/graph_construction.py:_do_relate` 第 380-389 行
- 关联讨论：`docs/graphmemory_rebuild_discussion_2026-04-28.md` 第 2.2 节
- RL 路线：`docs/rl_construction_plan_2026-05-08.md` Stage 1

# GraphMemory 算法逻辑说明

本文档描述当前 GraphMemory 系统的完整算法流程，包括图构建（写路径）和问答检索（读路径）。

---

## 整体架构

系统分为两个独立流程，共用同一套图存储：

```
对话数据 (LoCoMo)
      │
      ▼
┌─────────────────────────────┐
│     图构建 (build_memory)   │  ← 离线批处理
│  Step 1-5: 把对话写入图     │
└─────────────┬───────────────┘
              │  共享 GraphStore + ChromaDB
┌─────────────▼───────────────┐
│    问答检索 (run_qa)        │  ← 在线推理
│  Step 6-11: 从图里找答案    │
└─────────────────────────────┘
```

---

## 存储层

| 存储 | 内容 | 用途 |
|------|------|------|
| `{sample}_graph.json` | 图结构（节点+边） | 所有图操作的主存储 |
| ChromaDB `{sample}_nodes` | 节点向量索引 | Step 3/6 的向量检索 |
| ChromaDB `{sample}_turns` | 对话批次向量索引 | Step 11 的原始文本回退 |

**节点类型：**
- **Entity**：可复用的答案对象（人、地点、物品、技能、证书等）
- **Event**：具体事实（某人在某时做了某事）

**边类型（family）：**
- `entity-event`：实体参与了某事件（`participant`, `achieved`, `owns` 等）
- `entity-entity`：实体间稳定关系（`sibling_of`, `works_at` 等）
- `event-event`：事件间时序/因果链（`before`, `after`, `updates`, `inspired`）

---

## 图构建流程（Steps 1–5）

入口：`scripts/build_memory.py` → `GraphBuilder.build_from_sample()`

对话按 `k_turns=4` 分批，每批依次执行以下步骤：

### Step 1：RawArchive（无条件写入）

```
batch_text → ChromaDB {sample}_turns 集合
```

无论后续是否触发图更新，每批原始对话都存入向量库。作用：
- 为 Step 11 的 raw fallback 提供检索源
- 确保没有信息丢失

### Step 2：GraphTrigger（LLM 决策）

```
batch_text + 当前图规模 → LLM → TRIGGER / SKIP
```

LLM 判断该批对话是否包含值得写入知识图的长期事实。若为 SKIP，流程结束，图不变。

**触发逻辑**（prompt 约束）：
- 含具体事实、状态变化、计划、成就 → TRIGGER
- 仅闲聊、泛泛之论、纯会话容器 → SKIP

### Step 3：Localize（定位相关子图）

```
batch_text → 找到最相关的局部子图
```

三步流程（`GraphLocalizer`）：

**3.1 Seed Retrieval**
- 用 BAAI/bge-m3 对 batch_text 做向量检索
- 返回 top-k 个最相似节点（`seed_top_k=5`）
- 强制加入对话参与者节点（Caroline、Melanie）避免重复创建

**3.2 Neighbourhood Assembly（BFS 展开）**
- 从种子节点做 BFS，沿所有边展开
- 约束：`max_hops=2`，`max_nodes=20`，`max_edges=30`
- 结果：一个候选子图

**3.3 Subgraph Scoring**
- 按四因子规则对候选子图打分
- 返回最高分的一个子图

### Steps 4+5：Construction（合并图编辑，单次 LLM 调用）

```
batch_text + 局部子图 → LLM → JSON 操作序列 → 执行到 GraphStore
```

LLM 输出包含构建和更新两类操作：

**构建操作（新增结构）：**
| 操作 | 作用 |
|------|------|
| `CreateEntity` | 创建实体节点（人/物/地点/技能等） |
| `CreateEvent` | 创建事件节点（必须含 `fact`/`quote`/`source` 三个属性） |
| `Link` | 连接至少一个新节点的边 |
| `AttachAttr` | 给现有节点添加属性 |
| `Skip` | 无需写图 |

**更新操作（对齐现有图）：**
| 操作 | 作用 |
|------|------|
| `MergeNode` | 合并同指不同节点（吸收别名/属性，重定向边） |
| `ReviseAttr` | 修改现有属性 |
| `AddEdge` | 在两个现有节点间添加边 |
| `DeleteEdge` | 删除错误或过时的边 |
| `PruneNode` | 删除低价值冗余节点 |
| `KeepSeparate` | 记录两个相似节点确实是不同对象 |

**关键约束（prompt 硬编码）：**
- Event 必须包含 `fact`（自足句）、`quote`（原文最短引用）、`source`（turn_id 列表）
- 禁止创建"会话容器"事件（如 "A and B chat on date X"）
- 禁止创建泛泛抽象事件（如 "做个好人" 类价值观表达）
- 优先复用现有节点而非重复创建
- Entity-Event 边禁用 `spoke_to`/`discussed`，改写为 `participant`
- 相对时间（"last week"）写入 `fact` 时保留相对形式，不推算绝对日期

**后修复（`_repair_created_events`）：**
- 若新建 Event 没有 entity-event 边，自动链接到说话人
- 补充缺失的 `batch_id`、`time` 等上下文字段

---

## 问答检索流程（Steps 6–11）

入口：`scripts/run_qa.py` → `GraphRetriever.answer()`

### Step 6：Localize（同 Step 3，重用 GraphLocalizer）

根据问题类别选择定位策略：
- Category 1、3（recall-heavy）：`localize_union`，多查询变体聚合，`max_nodes=40`
- 其他 category：标准 `localize`

### Step 7：SelectAnchor（LLM 选锚点）

```
问题 + 局部子图 → LLM → 1-3 个锚点节点 ID
```

- 若子图节点 ≤ 5 个，直接用所有节点作锚点
- 否则 LLM 从子图中挑选最可能含答案或通向答案的 1-3 个节点

### Steps 8–10：Jump 循环（ReAct）

这是核心检索循环，每轮：

1. **Pool**：把当前已访问节点/边 + 原始文本压缩为 evidence_text
2. **LLM 决策**：基于 evidence 选择下一步动作

LLM 可选动作：

| 动作 | 参数 | 效果 |
|------|------|------|
| `jump` | `node_ids`, `relation_family`, `constraint`, `budget` | 沿指定类型的边扩展 frontier |
| `raw_fallback` | `query` | 切换到 ChromaDB 向量检索原始对话 |
| `finish` | `answer` | 停止，返回答案 |

**Jump 候选打分（`_score_jump_candidate`）：**
- 问题词命中节点文本 → +2.0
- edge predicate 命中问题词 → +1.0
- 满足 constraint → +3.0
- 节点为会话容器（含 " chat"/"conversation"） → -2.0
- predicate 为 `spoke_to`/`discussed` → -1.0

**停止条件：**
- LLM 调用 `finish` → 立即停止
- frontier 为空（图已遍历完）→ 自动触发 raw_fallback，继续
- `hop_count >= max_hop=3` → 强制 finish

### Step 11：强制完成（Forced Finish）

达到 max_hop 后：
- 若为可答 category（1/2/3/4）且还没有 raw context → 自动补一次 raw_fallback
- 调用 `_forced_answer()`：明确要求 LLM 从 evidence 给出最佳短答案

### 答案后处理（`_finalize_answer`）

1. `_canonicalize_final_answer()`：清理格式（去 markdown fence、去引导词、多行合并为逗号分隔）
2. 若为可答 category 但答案是拒绝词（"Unknown"/"Not mentioned" 等）→ 自动追加 raw_fallback 再重新生成
3. 若开启 `final_answer_compression` → LLM 二次压缩为最短短语

---

## 答案格式策略（Category-aware）

| Category | 问题类型 | 关键规则 |
|----------|----------|---------|
| 1, 2, 4 | 可直接回答 | 只返回答案短语；日期格式 "15 July, 2023"；相对时间用 "the week before X" |
| 3 | 推理题 | 允许 "likely"；禁止 "Unknown" |
| 5 | 对抗（陷阱）题 | 无直接证据支持则输出 "Not mentioned in the conversation" |

---

## 关键配置参数

```yaml
# build_memory.yaml
memory.k_turns: 4            # 每批包含几轮对话
graph.seed_top_k: 5          # 向量检索种子数
graph.max_hops: 2            # BFS 深度
graph.max_nodes: 20          # 子图节点上限
graph.max_edges: 30          # 子图边上限

# run_qa.yaml
graph.retrieval_max_hop: 3   # ReAct 最大跳转次数
graph.jump_budget: 5         # 每次 jump 最多扩展几个邻居
memory.retrieval_topk: 5     # raw_fallback 返回条数
```

---

## 核心文件索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 流程编排 | `graph_builder.py` | `GraphBuilder._process_batch()` |
| 触发决策 | `graph_trigger.py` | `GraphTrigger.should_trigger()` |
| 子图定位 | `graph_localize.py` | `GraphLocalizer.localize()` |
| 图编辑 | `graph_construction.py` | `GraphConstructor.run()` |
| 图存储 | `graph_store.py` | `GraphStore.add_node/add_edge/merge_nodes()` |
| 向量存储 | `vector_store.py` | `ChromaStore.search()` |
| QA 检索 | `graph_retrieval.py` | `GraphRetriever.answer()` |
| 原始文本库 | `raw_archive.py` | `RawArchive.archive()/search()` |
| 评估 | `evaluator.py` | `f1_score()`, LLM judge |

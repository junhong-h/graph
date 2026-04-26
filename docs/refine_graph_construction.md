# GraphMemory 建图流程 Refine 备忘

日期：2026-04-26

## 背景

当前检索阶段频繁出现 `forced_finish` / `max hop reached`，但具体例子显示，问题往往不是单纯 hop 数不够，而是构建阶段已经把事实节点建成孤点，导致检索时 frontier 很快耗尽。

例子：`conv-41_q15`

```text
Question: What writing classes has Maria taken?
Gold: Poetry, creative writing
Evidence:
- D7:1 Maria took a creative writing class.
- D9:1 Maria has been taking a poetry class.
```

构建阶段确实创建了两个 Event：

```text
Event: Creative Writing Class
Event: Maria took poetry class
```

但两者都没有成功 Link 到 `Maria` 或其它上下文节点。检索阶段只命中 `Maria took poetry class`，随后沿 `entity-event` 边 jump 时没有新节点，最终 forced finish，只答出 `poetry class`。

## 核心结论

GraphMemory 更适合建成以 `Entity/Concept --role--> Event` 为主干的事件图，而不是自由知识图。

大多数有效边应该是 `entity-event`。`event-event` 和 `entity-entity` 只在有明确语义时才建，否则会制造噪声和错误路径。

## Entity 与 Event 的划分

### Entity / Concept

Entity 表示相对稳定、可复用、可被多次提及、可作为答案值的对象或概念。

应该建 Entity 的类型：

- 人：Maria, John
- 地点：Boston, San Francisco
- 组织/机构：church, school, company
- 物品：cross necklace, guitar, car
- 课程/活动类型：creative writing class, poetry class, yoga class
- 作品：book, movie, song
- 疾病/问题：gastritis, car trouble
- 长期项目/计划：veterans support initiative, community project
- 宠物、家庭成员、同事等可被反复引用的对象

### Event

Event 表示某个主体在某个时间发生、做过、经历、计划或表达的事实断言。

Event 应该具备：

- 至少一个参与主体 Entity
- 时间属性 `time`
- 原文证据 `original_text` / `evidence_quote`
- 来源定位 `source_turn_ids` 或 `batch_id`
- 可选的动作、情绪、结果、数量等 attrs

### 例子

原文：

```text
Maria: I took a creative writing class recently, and it was super enlightening.
```

推荐结构：

```text
Entity: Maria
Entity: creative writing class

Event: Maria took creative writing class
attrs:
  time: 2023-02-25
  activity: took a creative writing class
  original_text: I took a creative writing class recently, and it was super enlightening.
  source_turn_ids: [conv-41_conv_session_7_D7:1]

Edges:
  Maria --took/participant--> Maria took creative writing class
  creative writing class --object_of/taken_in--> Maria took creative writing class
```

同理：

```text
Entity: poetry class
Event: Maria took poetry class
Maria --took/participant--> Maria took poetry class
poetry class --object_of/taken_in--> Maria took poetry class
```

这样回答 “What writing classes has Maria taken?” 时，可以从 `Maria` 找到相关 Event，再抽取 class Entity，得到 `poetry class, creative writing class`。

## 边类型规则

### entity-event：主干边

大部分事实都应该落到 `entity-event`。

常见 predicate：

- `participant`
- `experienced`
- `took`
- `attended`
- `visited`
- `started`
- `joined`
- `owned`
- `created`
- `launched`
- `mentioned`
- `object_of`

约束：

- 每个 Event 至少有一条连接主体的 `entity-event` 边。
- 如果 Event 中有答案型名词短语，也应该建 Entity/Concept，并用 `entity-event` 连到 Event。
- 禁止孤立 Event 写入最终图；如果 LLM 没有生成 Link，执行层应自动补最小边。

### event-event：只表达真实事件关系

只在以下情况建 `event-event`：

- 时间顺序：`before` / `after`
- 更新或修正：`updates`
- 因果或触发：`caused` / `led_to` / `inspired`
- 同一长期事项的阶段链：`planned -> attended -> reflected`

不要因为两个事实出现在同一 session、相邻 turn、或互相提到就建 event-event。

不推荐 predicate：

- `spoke_to`
- `discussed`
- `mentions`
- `related_to`
- `participated`

这些边通常没有检索价值，会让 traversal 漫游到噪声节点。

### entity-entity：只表达稳定关系

只在关系本身稳定、可复用时建 `entity-entity`：

- 人际关系：`spouse`, `friend`, `colleague`, `family_member`
- 所属关系：`works_at`, `member_of`, `owns`
- 别名/同一对象：`same_as`
- 地理/包含关系：`located_in`, `part_of`

不要因为两个 Entity 在同一句或同一 Event 中共同出现就直接连 entity-entity。共同出现的信息应通过同一个 Event 表达。

## 当前流程的主要问题

1. `CreateEvent` 后 Link 失败不会触发修复，导致 Event 孤立。
2. LLM 会生成无法 resolve 的临时 ID，例如 `NEW_Maria_and_John_chat`，执行层只记录 error。
3. `CreateEntity` 太保守，答案型概念没有被建成 Entity。
4. `GraphStore.add_edge()` 没有强校验 family，图里出现了 `event-entity`。
5. 一些 session-container Event，例如 `Maria and John chat`，成为低语义高噪声节点。
6. Event 缺少稳定的 `source_turn_ids` 和 `original_text`，后续难以精准回查原文。

## 建议的改进顺序

### P0：执行层 invariant

在执行 graph ops 后做结构校验：

- 所有 Event 必须有至少一条 `entity-event` 边连接主体 Entity。
- `event-entity` 自动 normalize 成 `entity-event`。
- 非法 family 直接拒绝或修正。
- Link 失败时尝试解析 `[8-char-id]`、`NEW_*`、canonical name。
- 如果仍失败，至少连到当前 batch speaker。

### P1：增强 Entity/Concept 抽取

prompt 中明确要求：

- 答案型名词短语要建 Entity/Concept。
- class、book、movie、city、disease、hobby、project、item 等不要只塞进 Event attrs。
- Event 负责“谁在何时做了什么”，Entity 负责“被做的对象/答案值是什么”。

### P2：减少 session-container Event

不要默认创建 `Maria and John chat on ...` 这种容器节点。

如果需要保留 session 级信息，应作为 raw pointer 或低优先级 metadata，不参与主检索图 traversal。

### P3：保留原文证据

每个 Event 至少保存：

```text
time
source_turn_ids
original_text 或 evidence_quote
```

这能提升 Cat4 精确词、Cat2 时间题，以及 raw fallback 的精准回查能力。

### P4：检索侧配套

建图修复后，再做检索优化：

- Cat1 聚合题合并多个 localizer candidate，而不是只取 best subgraph。
- Jump 前对邻居按 question relevance rerank。
- frontier 空时 trace 写成 `frontier_exhausted`，不要统一记为 `max_hop reached`。

## 设计原则

一句话版本：

```text
Entity 是答案和值，Event 是带时间和证据的事实，entity-event 是主干。
event-event 只表达时间/因果/更新链，entity-entity 只表达稳定关系。
```

当前失败不是因为图完全记不住事实，而是事实节点没有稳定接入 `entity-event` 主干。

## 算法流程改造建议

当前 construction 的根本问题不只是 prompt，而是算法上让 LLM 直接编辑图：

```text
batch text + local subgraph -> LLM outputs graph ops -> execute ops
```

这会把 ID 管理、ontology 约束、边类型选择、merge 决策都交给 LLM。一旦 LLM 生成无法解析的 `NEW_*`、漏掉 Link、或者输出非法 family，图结构就会损坏。

更稳的流程应该拆成：

```text
batch text
  -> typed extraction
  -> entity/concept resolution
  -> event normalization
  -> deterministic edge compilation
  -> graph validation / repair
  -> write graph
```

### 1. LLM 只做 typed extraction

LLM 不再直接输出 `CreateEvent` / `Link` / `MergeNode`，而是输出事实抽取结果。

示例：

```json
{
  "entities": [
    {"mention": "Maria", "type": "person"},
    {"mention": "creative writing class", "type": "class"}
  ],
  "events": [
    {
      "predicate": "took",
      "subject": "Maria",
      "object": "creative writing class",
      "time": "2023-02-25",
      "evidence_quote": "I took a creative writing class recently, and it was super enlightening.",
      "source_turn_ids": ["conv-41_conv_session_7_D7:1"]
    }
  ]
}
```

LLM 负责理解文本，程序负责把抽取结果编译成图。

### 2. 程序负责 deterministic graph compiler

compiler 根据 typed extraction 做确定性写图：

1. resolve `Maria` 到已有 Entity。
2. resolve 或创建 `creative writing class` 这个 Concept Entity。
3. 创建 Event：`Maria took creative writing class`。
4. 自动生成主干边：

```text
Maria --took/participant--> Maria took creative writing class
creative writing class --object_of/taken_in--> Maria took creative writing class
```

这样 `entity-event` 边不再依赖 LLM 记得输出。

### 3. Entity resolution 独立成层

Entity resolution 不应该混在图编辑 prompt 里。

建议 resolution 顺序：

1. exact normalized match：`Maria` = `maria`
2. alias match：`Jon` / `Jonathan`
3. phrase normalization：`Poetry Class` / `poetry class`
4. embedding candidate retrieval
5. LLM verify，只用于模糊候选
6. type constraint：person 不与 class、location、disease 等类型合并

Entity 可以相对积极地 merge，因为它表示稳定对象。但每次 merge 需要保留 aliases 和 source provenance。

### 4. Event normalization 保守处理

Event 是证据单元，不应该过度 merge。

推荐 Event key：

```text
source_turn_id + subject + predicate + object
```

或：

```text
subject + predicate + object + time + source
```

Event 可以做轻量去重，但不能把不同时间、不同原文、不同对象的事实合并掉。

### 5. Graph lint / repair 作为必经步骤

每个 batch 写图前或写图后都跑 lint：

- Event 是否有 `time`
- Event 是否有 `source_turn_ids`
- Event 是否有 `original_text` / `evidence_quote`
- Event 是否至少连一个主体 Entity
- 是否出现非法 family，例如 `event-entity`
- 是否有孤立 Event
- 是否有低价值 `event-event` predicate，例如 `spoke_to`, `discussed`, `mentions`
- 是否存在 session-container Event 变成高噪声枢纽

发现问题时：

- 能修复则自动修复。
- 无法修复则拒绝写入该 op，并保留 sample-level error log。
- 不要只写 warning 后继续生成坏图。

### 6. event-event 边单独分类

event-event 边不要由 LLM 在主抽取里随手生成。

可以单独做一个 event relation classifier，只允许输出：

```text
before
after
updates
caused
led_to
inspired
```

如果两个 Event 只是同一 session 中相邻出现，或者只是互相提到，不建 event-event。

### 7. 检索侧配套算法

建图修复后，检索也应从单一 BFS 改成 scored retrieval/traversal。

建议流程：

```text
query
  -> query analysis
  -> multi-seed retrieval
  -> candidate subgraph union
  -> node/edge rerank
  -> evidence assembly
  -> answer
```

特别是 Cat1 聚合题，不应该只取一个 best local subgraph。

例如：

```text
What writing classes has Maria taken?
```

可以生成多个检索视角：

```text
Maria
writing classes
creative writing class
poetry class
taken class
```

然后合并候选节点与邻居，再做 rerank。这样即使 `poetry class` 和 `creative writing class` 分散在两个 session，也更容易同时召回。

### 8. 最小可行改造路径

短期不必一次性推翻现有实现，可以分阶段做：

1. 先保留当前 LLM graph ops，但增加执行层 lint / repair。
2. 修复 Link resolve、family normalize、Event 必须连主体 Entity。
3. 增加答案型 Concept Entity 抽取规则。
4. 新增 typed extraction + compiler 的实验路径，与旧路径并行对比。
5. 如果新路径结构更稳定，再替换旧 graph editor prompt。

最关键的方向：

```text
LLM extract facts, program compiles graph.
```

也就是把图结构正确性从 prompt 中拿出来，放到确定性代码和校验流程里。

## Retrieval 流程改造建议

当前 retrieval 的主要问题不是单一 prompt 问题，而是工具语义和算法能力偏弱：

```text
query
  -> localizer 只返回一个 best subgraph
  -> LLM 选 anchor
  -> jump 按边存储顺序扩展
  -> constraint 基本不生效
  -> frontier 空或 max hop 后 forced answer
```

这会导致 Cat1 聚合、多 session、多证据问题召回不足。`conv-41_q15` 是典型例子：图里有 `poetry class` 和 `creative writing class` 两个事实节点，但 localizer 只保留了 `poetry class`，后续 jump 又因该节点是孤点而提前耗尽。

Retrieval 应从：

```text
single best subgraph + sequential BFS
```

改成：

```text
multi-seed union + relevance-ranked expansion + rule-based raw repair
```

### P0：修 trace 与 stop reason

当前 frontier 空也会进入最终 `forced_finish`，日志容易误判为 `max hop reached`。

建议在 trace 中区分：

```text
frontier_exhausted
max_hop_exhausted
no_evidence
parse_error
```

当 jump 后 `new_nodes` 为空时：

```text
trace: frontier_exhausted
auto raw_fallback(question)
continue one more LLM step with raw evidence
```

不要直接 break 后统一打印 `Max hops reached`。

### P1：Localizer 支持 top-M union

当前 `GraphLocalizer.localize()` 只返回评分最高的一个子图。对 Cat1/Cat3 聚合和推理题不合适。

建议新增 retrieval 专用接口：

```python
localize_ranked(input_text, top_m=3)
localize_union(input_text, top_m=3, max_nodes=40, max_edges=80)
```

改法：

- `_subgraph_scoring()` 返回排序后的 candidates，而不是只返回 best。
- Cat1/Cat3 使用 top-M union。
- Cat4 保持较小子图，因为多为单点事实。
- retrieval scoring 中减弱或去掉 size penalty，避免优先选择小而密的局部。

目标是让聚合问题能同时保留多个 session 的入口证据。

### P2：Query variants / multi-seed retrieval

不要只用原问题做一次 embedding search。

可以先用规则生成多个 query variants：

```text
original question
main entity + target phrase
target phrase only
subject + verb/object phrase
```

例子：

```text
Question: What writing classes has Maria taken?

Variants:
- What writing classes has Maria taken?
- Maria writing classes
- writing classes
- Maria taken class
```

先用规则实现，后续再考虑 LLM query expansion。

### P3：Jump 使用 relevance-ranked expansion

当前 `_execute_jump()` 的行为是：

```text
get all edges of node
iterate by storage order
take first unvisited neighbors until budget
```

这在高出度节点上会截掉相关边。

建议改成：

```text
collect candidate edges/neighbors
score each neighbor by query + constraint + predicate + node attrs
sort by score
take top budget
```

可以先用 lexical score：

```text
+ canonical_name 命中 query term
+ attrs 命中 query 或 constraint
+ predicate 与 query verb/object 相关
+ node type 符合 constraint
- generic chat/session-container node
- predicate 是 spoke_to/discussed/related_to
```

接口上可以把 question 传给 jump：

```python
_execute_jump(..., question=question)
```

### P4：Constraint 改成 soft filter

当前 `_matches_constraint()` 最后默认 `return True`，所以 constraint 基本不生效。

建议将 constraint 实现为 soft filter：

```text
if any candidate matches constraint:
    prioritize matching candidates
else:
    fallback to unconstrained candidates
```

先支持常见格式：

```text
activity=poetry class
time=2023-04
node_type=Event
```

这样既能利用 LLM 给出的约束，又避免约束写错导致空结果。

### P5：规则式 raw repair

不要完全让 LLM 决定何时 raw_fallback。代码层应在高风险场景强制补原文。

建议规则：

```text
frontier exhausted -> raw_fallback(question)
forced finish 前 raw_context 为空 -> raw_fallback(question)
Cat4 -> 默认补 raw top-k
Cat1 聚合题 -> graph evidence 少于 2 个候选事实时补 raw
Cat2 时间题 -> graph evidence 没有明确 time 时补 raw
```

图负责结构召回，raw archive 负责精确短语、时间和原文措辞修复。

### P6：Prompt 与真实工具语义对齐

在 P1-P5 后再改 prompt。重点不是变长，而是对齐真实工具能力。

新增规则可以包括：

```text
- For list or aggregate questions, one evidence item may be partial; use raw_fallback if only one candidate item is found.
- If Jump returns no new evidence, switch to raw_fallback or finish from raw evidence.
- Use concise key=value constraints when possible, e.g. activity=poetry class.
- Do not repeatedly jump from an exhausted frontier.
```

当前 prompt 的主要问题是让模型以为 `jump` 和 `constraint` 是智能检索工具，但实际底层只是顺序 BFS 和近乎无效的自由文本 constraint。

### Retrieval 实验顺序

建议按 ablation 逐步验证：

```text
baseline
+ trace/frontier raw fallback
+ localize_union
+ localize_union + jump rerank
+ localize_union + jump rerank + raw repair
```

除 F1/BLEU 外，应额外记录：

```text
forced_finish rate
frontier_exhausted rate
true max_hop_exhausted rate
raw_fallback rate
avg evidence nodes
Cat1/Cat2/Cat4 F1
```

最关键的 retrieval 方向：

```text
从“单入口 best subgraph + 顺序 BFS”
改成“多入口 union + 相关性排序扩展 + 规则式 raw repair”。
```

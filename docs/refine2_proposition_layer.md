# GraphMemory Refine2：Proposition Layer 与 Evidence Quote 设计

日期：2026-04-27

## 背景

在重建 `conv-41` 图之后，硬结构已经明显改善：

- 非法边为 0
- 孤立 Event 为 0
- session-container Event 为 0
- `event-event` 谓词落在允许集合内
- raw fallback 从上一轮的 57.0% 降到 32.6%

但 Cat4 中仍有 27 个问题触发 raw fallback。逐题检查后发现，Cat5 触发 raw fallback 比较正常，因为 false-premise 问题需要验证不存在；Cat4 的问题不同，通常是：

```text
图里有相关主题节点或 broader event，
但答案需要的精确事实、原因、状态、对象或短语没有被结构化。
```

当前 retrieval 在 raw fallback 前给 LLM 的内容是 `format_subgraph()` 生成的图文本：

```text
Nodes:
  [node_id] Event "canonical_name" | attr1=..., attr2=...
  [node_id] Entity "canonical_name" | aliases=...

Edges:
  "src" --[predicate]--> "dst" (family)
```

raw fallback 后才会追加原始对话：

```text
--- Raw conversation context ---
[turn_id=...; speaker=...; session_time=...]
原始 turn 文本
```

所以 raw 前不是完全没有信息，但图证据通常是“主题摘要 + attrs + edges”。如果精确答案只藏在整段 `original_text` 里，或根本没有作为 attr/event 被抽出，LLM 就容易主动调用 raw fallback。

## 需要避免的方向

这次 refine 不能做成 LOCOMO/Cat4 专项优化。

不应该：

- 按 LOCOMO category 写 construction 逻辑。
- 针对 `started / received / hosted / puppy / shelter / veterans` 等具体词写规则库。
- 针对 `conv-41` 中的错误 case 写 few-shot。
- 根据 gold answer 反向补图。

这些都会降低泛化性。换一个 QA 检索任务、换一个领域或换一个数据集，很可能失效。

真正要解决的问题应该抽象成：

```text
source text 中的可验证事实断言，是否被 graph memory 充分表示？
```

因此，后续设计应从 “Cat4 fine-grained fact” 改成更通用的：

```text
source-grounded proposition layer
```

## 核心设计

GraphMemory 后续应从二层结构：

```text
Entity + Event
```

扩展成三层结构：

```text
Entity layer
Proposition layer
Event / Relation layer
```

### Entity Layer

Entity 表示相对稳定、可复用、可作为主体或客体的对象。

例如：

```text
Maria
John
local dog shelter
university degree
painting
military memorial
```

Entity 仍然是检索图的稳定锚点。

### Proposition Layer

Proposition 是由原文直接支持的最小事实断言。

定义：

```text
A source-grounded minimal factual assertion.
```

中文理解：

```text
一个 atomic proposition 是从原文中抽出的、可以独立判断真假的最小事实断言。
```

它不等同于 LOCOMO 的 Cat4，也不等同于某个具体动词触发的事实。它是通用的信息表示单元。

一个 proposition 应满足：

- 有明确或可由上下文确定的主体。
- 表达一个主要事实关系、动作、状态、经历、意图、评价或时间信息。
- 由原文直接支持，不加入推理。
- 可以被一个问题单独询问或验证。
- 不只是寒暄、附和、泛泛评价，除非它承载了明确事实。
- 保留最短证据片段和 source id。

推荐通用 schema：

```json
{
  "subject": "...",
  "predicate": "...",
  "object": "...",
  "qualifiers": {},
  "modality": "factual | planned | desired | uncertain | negated",
  "time": "...",
  "source_ids": [],
  "evidence_quote": "..."
}
```

其中：

- `subject`：事实主体。
- `predicate`：事实关系或动作，用自然语言短语即可，不强制 ontology。
- `object`：事实客体，可为空。
- `qualifiers`：开放字典，放时间、地点、原因、方式、状态、频率、数量等限制信息。
- `modality`：区分事实、计划、愿望、不确定、否定。
- `time`：事件时间或提及时间；后续可拆成 `event_time` / `mentioned_at`。
- `source_ids`：支持该 proposition 的 turn id / chunk id。
- `evidence_quote`：支持该 proposition 的最短原文片段。

### Event / Relation Layer

Event 仍然有价值，但它更像是组织多个 proposition 的上下文容器，而不是承载所有细节的唯一单元。

例如一个 broader event：

```text
Maria volunteering at shelter
```

可能包含多个 proposition：

```text
Maria volunteered at a shelter.
Maria noticed a little girl sitting alone.
The girl seemed sad.
Maria sat with the girl.
The girl had no other family.
```

Event 可以负责：

- 时间段或场景聚合
- 多个 proposition 的共同上下文
- event-event 的时间、更新、因果关系
- entity-event 的参与关系

Proposition 负责：

- 精确事实
- QA 可直接使用的 answer-bearing assertion
- source-grounded evidence quote

## Evidence Quote 定义

`evidence_quote` 是支持某个 proposition 的最短原文片段。

要求：

- 必须来自原始 source text。
- 尽量短，只保留证明该 proposition 的必要文字。
- 不用模型改写。
- 可以是一句或相邻两句，但不应是整个 turn batch。
- 必须能和 `source_ids` 对齐。

例子：

```json
{
  "subject": "Maria",
  "predicate": "started volunteering at",
  "object": "local dog shelter",
  "qualifiers": {
    "frequency": "once a month",
    "recency": "just started"
  },
  "modality": "factual",
  "source_ids": ["conv-41_conv_session_17_D17:12"],
  "evidence_quote": "I just started volunteering at a local dog shelter once a month."
}
```

这个 quote 的作用不是为了复刻 LOCOMO，而是让任何 QA 检索任务都能从图证据中看到最短可验证证据，而不是被迫回到 raw batch。

## 与当前 Event 表示的对比

当前 Event 常像这样：

```json
{
  "type": "Event",
  "canonical_name": "John and family considering adopting rescue dog",
  "attrs": {
    "reason": "pass on lesson of unconditional love and loyalty",
    "time": "11:51 am on 3 June, 2023",
    "original_text": "... I just started volunteering at a local dog shelter once a month."
  }
}
```

问题：

- 主事件是 John considering adopting rescue dog。
- Maria dog shelter 事实只是藏在 `original_text`。
- 检索到这个 Event 后，LLM 不一定稳定抽出 Maria 的新活动。
- 如果 localizer 没选中这个 Event，就只能靠 raw fallback。

加入 proposition 后可以表示为：

```json
{
  "type": "Event",
  "canonical_name": "Maria started volunteering at local dog shelter",
  "attrs": {
    "proposition": {
      "subject": "Maria",
      "predicate": "started volunteering at",
      "object": "local dog shelter",
      "qualifiers": {
        "frequency": "once a month",
        "recency": "just started"
      },
      "modality": "factual"
    },
    "evidence_quote": "I just started volunteering at a local dog shelter once a month.",
    "source_ids": ["conv-41_conv_session_17_D17:12"],
    "mentioned_at": "3 June, 2023"
  }
}
```

更抽象地说：

```text
当前 Event：这里发生过一个相关主题。
Proposition：这里有一个可验证、可直接问答的事实。
Evidence quote：这个事实由原文哪句话支持。
```

## 泛化性讨论

Proposition layer 不应理解为 LOCOMO 专用优化。它适用于任何需要 source-grounded QA 的检索任务。

例如：

```text
科研论文：Method X improves F1 by 3 points.
医疗记录：Patient reports chest pain since Monday.
客服对话：User wants refund for order A.
代码文档：Function X requires parameter Y.
会议纪要：Alice will send the draft by Friday.
对话记忆：Maria volunteers at a dog shelter monthly.
```

这些领域不同，但都需要：

```text
最小事实断言 + 原文证据 + 来源定位
```

因此，泛化设计原则是：

- 不以问题类别触发。
- 不以具体领域词触发。
- 不以 gold answer 反向补图。
- 从 source text 中抽取 source-grounded propositions。
- 用 QA 结果评估 coverage，而不是指导具体规则。

## Construction 侧改进方向

### 1. Prompt 改为保留 distinct factual assertions

prompt 不应列大量数据集特定例子，而应强调信息完整性原则：

```text
Represent each distinct factual assertion that is directly supported by the source text.
Prefer minimal, source-grounded propositions over broad summaries when the detail could be independently queried or verified.
If a statement contains a main fact and a supporting detail that changes who/what/why/how/when, preserve the detail as a proposition or as structured qualifiers.
Do not create propositions for greetings, generic acknowledgements, or unsupported inferences.
```

### 2. Event 与 Proposition 的关系

短期建议不要新增独立 `Claim` 节点，避免图结构一次性变复杂。

第一阶段可以把 proposition 作为 Event attr：

```json
{
  "type": "Event",
  "canonical_name": "...",
  "attrs": {
    "proposition": {
      "subject": "...",
      "predicate": "...",
      "object": "...",
      "qualifiers": {},
      "modality": "factual"
    },
    "evidence_quote": "...",
    "source_ids": []
  }
}
```

这样兼容当前 `Entity/Event` 图，不需要立刻新增 `Proposition` node type。

如果后续 proposition 数量变多，且检索确实需要更细粒度跳转，再考虑把 Proposition 独立成节点：

```text
Entity --subject_of--> Proposition
Entity --object_of--> Proposition
Proposition --part_of--> Event
```

### 3. Coverage audit，而不是 trigger-word audit

不要写基于具体词表的 audit。

更泛化的 audit 流程：

```text
source text
-> extract source-grounded propositions
-> compare with graph events/propositions
-> classify coverage as covered / partial / missing
```

coverage 判断：

```text
covered:
  图中已有 proposition/event 表达了相同主体、兼容 predicate/object，并保留关键 qualifier。

partial:
  图中有相关 Entity/Event/topic，但缺少关键 predicate/object/qualifier/evidence_quote。

missing:
  图中没有对应事实。
```

这能解释 Cat4 raw fallback：

```text
raw 前图证据 partial，
raw 后原文命中完整 proposition，
说明 construction coverage 不够。
```

但它不是为 Cat4 写的。换成其他 QA 或文档检索任务，也可以用同样方法检查 graph memory 的 source coverage。

## Retrieval 侧改进方向

当前 raw 前 LLM 看到的是：

```text
节点 canonical_name + aliases + attrs
边 predicate/family
```

如果 attrs 中只有 broad summary 或整段 `original_text`，LLM 可能判断证据不够。

加入 proposition 后，raw 前 evidence pool 应更像：

```text
Propositions:
  [node_id] Maria started volunteering at local dog shelter.
    qualifiers: frequency=once a month
    quote: "I just started volunteering at a local dog shelter once a month."
    source: conv-41_conv_session_17_D17:12

Edges:
  Maria --participant--> ...
  local dog shelter --object_of--> ...
```

检索逻辑应从“找相关节点”逐步变成：

```text
question -> information need proposition
candidate graph evidence -> represented propositions
rank by proposition compatibility + source support
```

兼容性可以考虑：

- subject 是否匹配
- predicate/object 是否语义兼容
- qualifiers 是否满足问题约束
- time/source 是否支持
- evidence_quote 是否包含 answer-bearing phrase

但第一阶段不必实现复杂 entailment。先把 proposition/evidence_quote 放进图 evidence，让 LLM 能看到更清楚的事实即可。

## Temporal 表示：把时间关系写进事实句

Temporal 优化也不应该一开始设计复杂 schema。

关键问题不是增加很多字段，而是 construction 生成的事实概括必须 self-contained，尤其要区分：

```text
事实什么时候被提及
事实实际指向什么时间
相对时间表达以什么时间为锚点
```

如果只把 `session_time` 放进 `time` 字段，容易把“对话时间”误当成“事件发生时间”。

### 最小原则

每个 proposition / event 优先保存三类信息：

```text
fact
quote
source
```

其中 `fact` 是 self-contained fact sentence，应该把时间关系自然写进去，而不是拆成大量字段。

例子：

```json
{
  "fact": "On 18 April 2023, John said he and his colleagues had gone to a tech-for-good convention the previous month.",
  "quote": "My colleagues and I went to a convention together last month.",
  "source": ["D12:9"]
}
```

这里不需要先设计 `mentioned_at / time_text / anchor / role` 等字段。`fact` 自己已经表达了：

```text
提及时间：18 April 2023
相对时间：the previous month
事件内容：John and colleagues went to a tech-for-good convention
```

`quote` 用来校验 `fact` 是否忠实于原文。

### Fact sentence 的时间规则

construction prompt 应强调：

```text
Write each factual memory as a self-contained sentence.
If the source uses relative time, include both the mention date and the relative expression in the sentence.
Do not rewrite a relative-time fact as if it happened on the mention date.
Do not infer exact dates unless the source directly states them or a later QA stage explicitly needs deterministic inference.
Preserve the shortest supporting quote.
```

好的事实句应该满足：

```text
不读原始 turn，也能知道“谁在什么时候说了什么，以及这个事实相对哪个时间成立”。
```

### 两类 temporal retrieval

Temporal QA 至少有两个方向。

第一类：根据时间找信息。

```text
time condition -> event/proposition
```

例如：

```text
What did X do on / before / after / around time T?
What happened the week before date T?
What new activity was mentioned on date T?
```

第二类：根据信息找时间。

```text
event/proposition -> time
```

例如：

```text
When did X do Y?
When did John get Max?
When did Maria adopt Coco?
```

这两类不能都靠 `event.time = session_time` 解决。

### Case 分析

#### Case 1：问题给具体日期，证据用相对时间指向该日期

```text
Q: Who did Maria have dinner with on May 3, 2023?
Gold: her mother
Evidence: D13:16
Session time: 4 May, 2023
Quote: My mom and I made some dinner together last night!
```

这里问题问的是 `May 3, 2023`，但证据是在 `4 May, 2023` 提到 `last night`。

不好的表示：

```text
Maria made dinner with mom
time = 4 May, 2023
```

问题是它会让检索误以为 dinner 发生在 4 May。

更好的事实句：

```text
On 4 May 2023, Maria said she and her mother had made dinner together the previous night.
```

这样 retrieval / QA 可以从 `previous night` 与 `4 May 2023` 关联到 `May 3, 2023`。

#### Case 2：问题问月份，证据在下个月用 last month 提到

```text
Q: What did John attend with his colleagues in March 2023?
Gold: a tech-for-good convention
Evidence: D12:9
Session time: 18 April, 2023
Quote: My colleagues and I went to a convention together last month.
```

这是典型的：

```text
问题时间 = March 2023
提及时间 = 18 April, 2023
原文时间 = last month
```

好的事实句：

```text
On 18 April 2023, John said he and his colleagues had gone to a tech-for-good convention the previous month.
```

这种写法既保留原文相对时间，又让下游可以推到 March 2023。

#### Case 3：时间关系用于相似事实消歧

```text
Q: What is the name of Maria's puppy she got two weeks before August 11, 2023?
Gold: Coco
Evidence: D30:1
Session time: 11 August, 2023
Quote: I got a puppy two weeks ago! Her name's Coco.
```

图中还有相似事实：

```text
On 13 August 2023, Maria said she had adopted a puppy named Shadow last week.
```

如果只匹配：

```text
Maria + puppy + name
```

就容易选错 Shadow。

好的事实句：

```text
On 11 August 2023, Maria said she had got a puppy named Coco two weeks earlier.
```

retrieval 可以用问题里的 `two weeks before August 11` 匹配这个 fact，而不是只靠 puppy 语义相似度。

#### Case 4：根据相对时间找事件

```text
Q: What did John do the week before August 3, 2023 involving his kids?
Gold: Had a meaningful experience at a military memorial
Evidence: D27:9
Session time: 3 August, 2023
Quote: Last week, we had a meaningful experience at a military memorial. It really made an impact on my kids.
```

好的事实句：

```text
On 3 August 2023, John said he and his kids had a meaningful experience at a military memorial the previous week.
```

这里问题中的 `the week before August 3` 可以直接对应 fact 中的 `previous week`。

#### Case 5：根据信息推时间

```text
Q: When did John get his dog Max?
Gold: In 2013
Evidence: D17:1
Session time: 3 June, 2023
Quote: Max ... was such an important part of our family for 10 years.
```

证据没有直接说 2013，需要从：

```text
as of 3 June 2023
for 10 years
```

推导。

好的事实句：

```text
On 3 June 2023, John said Max had been part of his family for 10 years.
```

QA 阶段再根据任务需要推：

```text
2023 - 10 = 2013
```

这里不建议 construction 阶段直接把原文改写成：

```text
John got Max in 2013.
```

除非同时保留 quote 和推导说明。否则模型可能把推导事实当成原文直接陈述。

### Retrieval 如何使用这种时间事实句

对于“时间 -> 信息”的问题：

```text
先用问题中的目标时间或相对时间表达匹配 fact sentence，
再匹配 subject / predicate / object。
```

例如：

```text
May 3, 2023
```

可以匹配：

```text
On 4 May 2023 ... previous night
```

对于“信息 -> 时间”的问题：

```text
先匹配 subject / predicate / object，
再从 fact sentence 中读取或推导时间。
```

例如：

```text
John + Max + had/got
```

匹配：

```text
On 3 June 2023, John said Max had been part of his family for 10 years.
```

然后 QA 阶段输出 `2013`。

对于“时间消歧”的问题：

```text
多个语义相似候选都命中时，
优先选择 fact sentence 中时间关系与问题时间约束兼容的候选。
```

例如：

```text
two weeks before August 11
```

优先匹配：

```text
On 11 August 2023 ... two weeks earlier
```

而不是：

```text
On 13 August 2023 ... last week
```

### 设计取舍

这个方案刻意避免把 temporal 设计成重 schema。

不首选：

```text
mentioned_at
time_text
anchor
role
event_time
event_time_granularity
temporal_relation
```

这些字段后续可以作为 derived metadata 或 indexing feature，但不应该成为核心语义表示的第一步。

首选：

```text
self-contained fact sentence + quote + source
```

原因：

- 更轻量。
- 更接近自然语言 QA。
- 更泛化到其他数据集。
- 不会把 LOCOMO 的时间答案格式写进图结构。
- 能同时支持“时间找信息”和“信息找时间”。

后续如果 retrieval 需要更强时间索引，可以从 `fact` 和 `quote` 中派生轻量 metadata，但不能替代 source-grounded fact。

## Raw Fallback 的定位

raw fallback 不应马上删除。

当前 Cat1-4 中 raw fallback 后大多是正确的，说明它在补召回，而不是单纯制造错误。

更合理的定位：

```text
raw fallback = retrieval fallback + construction diagnostic signal
```

如果某个 answerable question 触发 raw fallback 并最终答对，说明：

```text
raw text 中有可回答证据，
但 graph evidence 没有充分表达或没有被检索到。
```

这些 case 应进入 proposition coverage audit，用来评估 construction coverage，而不是直接按 case 写规则。

## 建议的实施步骤

### Stage 1：实现 Proposition Coverage Audit

新增脚本：

```text
scripts/audit_proposition_coverage.py
```

输入：

```text
graph.json
raw archive / source text
可选 qa_results.jsonl，用于聚焦 raw fallback case
```

输出：

```text
runs/.../proposition_coverage_audit.jsonl
runs/.../proposition_coverage_audit.md
```

每条记录包含：

```json
{
  "source_id": "...",
  "proposition": {
    "subject": "...",
    "predicate": "...",
    "object": "...",
    "qualifiers": {},
    "modality": "factual"
  },
  "evidence_quote": "...",
  "matched_graph_nodes": [],
  "coverage": "covered | partial | missing",
  "coverage_reason": "..."
}
```

这一步只生成报告，不改图。

### Stage 2：在 construction 中写入 proposition attrs

更新 construction prompt / parser，使每个有信息量的 Event 尽量包含：

```text
proposition
evidence_quote
source_ids
```

但仍遵守已有 graph invariants：

- 不创建 session/container event。
- Event 必须连接至少一个 Entity。
- `entity-event` 是主干。
- `event-event` 只表达真实时间、更新、因果等关系。

### Stage 3：重建图并做结构对比

重建后检查：

```text
node_count / event_count 是否暴涨
invalid_edge_count 是否仍为 0
isolated_event_count 是否仍为 0
proposition coverage missing/partial 是否下降
raw fallback case 中的 graph evidence 是否更充分
```

如果 Event 数量大幅膨胀，说明 proposition 抽取过细，需要收紧为“可独立验证且有信息量”的断言。

### Stage 4：重新跑 QA

观察：

```text
Cat4 raw fallback 是否下降
Cat4 accuracy 是否保持或提升
Cat1/Cat2 是否被细粒度 proposition 噪声影响
Cat5 是否因更多相似 proposition 而误答增加
avg trace steps 是否下降
frontier_exhausted 是否下降
```

目标不是让 raw fallback 立刻归零，而是让 answerable question 在图证据中有更高 coverage。

### Stage 5：再做 retrieval rerank

等图里有 proposition 后，再改 retrieval：

- localizer / jump 按 proposition compatibility 排序。
- high-degree Entity 不直接展开所有邻居。
- evidence pool 中优先展示 proposition + evidence_quote，而不是整段 `original_text`。
- 如果 graph proposition 已经满足 information need，就减少 raw fallback。

## 风险与控制

### 风险 1：图膨胀

Proposition 太细会导致 Event 数量暴涨。

控制：

- 只保留 source-grounded、可独立验证、有信息量的 proposition。
- 不抽寒暄、重复、泛泛情绪。
- 每个 turn/batch 设置上限。
- audit 阶段先报告，不直接补图。

### 风险 2：过度领域化

如果 prompt 写具体动词或具体场景，会过拟合。

控制：

- prompt 使用 “distinct factual assertion” 这类抽象规则。
- 不写 LOCOMO case few-shot。
- 不按 category 定制 construction。

### 风险 3：Cat5 误答增加

更多 proposition 可能让 false-premise 问题命中相似事实。

控制：

- Retrieval/QA 需要完整约束验证。
- Cat5 或 unanswerable-style 问题必须检查 subject/predicate/object/qualifier 是否同时满足。
- raw fallback 和 proposition retrieval 都不能只凭局部相似就回答。

## 一句话总结

Refine1 解决的是：

```text
图的硬结构是否健康：Entity/Event/Edge family 是否正确。
```

Refine2 要解决的是：

```text
图是否充分表示 source text 中可验证的最小事实断言。
```

最终目标不是为 LOCOMO Cat4 打补丁，而是把 GraphMemory 从“事件主题图”推进到：

```text
Entity + source-grounded Proposition + Event/Relation
```

这样 raw fallback 暴露出的缺口可以被解释为 proposition coverage 问题，并且这个分析框架可以迁移到其他 QA 检索任务。

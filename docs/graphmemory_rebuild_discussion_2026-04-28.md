# GraphMemory 重建讨论材料：当前流程、实验脉络与图结构问题

日期：2026-04-28  
范围：从 2026-04-20 进展报告到 2026-04-28 最新构图实验  
目的：总结目前 GraphMemory 算法流程、实验结果、遇到的问题，以及为什么需要重新思考图结构和构建流程。

---

## 1. 当前研究目标

GraphMemory 的核心目标是为长对话构建一个可检索、可更新、可追溯的长期记忆结构。系统希望比纯 raw retrieval 更强，因为它可以：

- 将长期稳定对象建成 Entity，作为检索 anchor。
- 将发生过的事实、状态变化、计划、活动建成 Event，作为回答证据。
- 用 Entity-Event / Event-Event 边支持跨 session 跳转和多跳推理。
- 保留 raw archive，避免图构建漏抽造成不可恢复的信息损失。

目前遇到的核心矛盾是：

> 图太稀疏会漏检索；图太密会引入大量 near-miss 噪声。  
> 只追求边覆盖会导致低质量 Event 和弱语义边；只追求事实精度又会漏掉关键事实。

因此当前阶段需要重新思考“健康图结构”到底应该是什么，以及 construction 应该如何分阶段完成。

---

## 2. 原始设计思路：GraphMemory Agent

最早的 Word 文档 `docs/GraphMemory Agent(construction & retrieval).docx` 中，系统被设计为维护三类状态：

### 2.1 Raw Archive `R_t`

每条输入无条件进入 raw archive。它不是图化结果，而是原始对话缓存。

作用：

- GraphTrigger 误判时，信息不会丢失。
- Graph construction 漏抽时，retrieval 可以 raw fallback。
- 未来可以做 delayed graphization：某段 raw 经常被 fallback 命中，再重新送入 Construction / Update。

这对应 Mem-T 中的 Raw Memory。后续实验也证明 raw fallback 对 Cat1-4 可答题有帮助，但对 Cat5 对抗不可答题会诱发 near-miss 幻觉。

### 2.2 Graph Memory `G_t=(V_t,E_t,A_t)`

长期结构化记忆。最初设计只保留两类主节点：

| 类型 | 含义 | 例子 |
|------|------|------|
| Entity | 稳定对象，可作为检索锚点 | John, Maria, Coco, Max, school, organization |
| Event | 一次事实、状态变化、计划、活动或 episode | Maria got a puppy; John and Max went camping |

原始设计中，Event 不应该是完整 schema extraction，也不应该把所有属性都升格为节点。一级节点承担三个功能：

- 检索锚点。
- 更新/合并对象。
- 图上跳转中转点。

边的设计原本是：

- 固定三类 edge family：`entity-event`, `entity-entity`, `event-event`
- 少量控制型谓词：`same_as`, `before`, `after`, `updates`
- 其余细粒度语义可以开放表示

后续实验发现，开放谓词对 Qwen3-4B 太难，模型会输出大量 `related`, `discussed`, `mentions`, 空 family 或 family-predicate 不匹配边。因此目前逐步收紧到较小的 predicate set。

### 2.3 Working Buffer `B_t`

短期临时状态，用于当前一步 Localize / Construct / Update。不会进入长期记忆。

### 2.4 原始 Construction / Update 流程

原始设想：

```text
input x_t
  -> RawArchive
  -> GraphTrigger
  -> Localize_write
  -> Construction
  -> Update
  -> G_{t+1}
```

关键原则：

- RawArchive 总是执行。
- GraphTrigger 只决定是否图化，不决定是否保存信息。
- Construction 是 selective graph induction，不是全量信息抽取。
- Update 负责合并、修订、补边、删边、保留分离等图编辑。

### 2.5 原始 Retrieval 流程

```text
query q
  -> Localize_query
  -> SelectAnchor
  -> Jump
  -> Pool
  -> Finish
  -> Raw Fallback if needed
```

Retrieval 目标不是找若干相关节点，而是在图上逐步构造足以支持回答的证据子图。

---

## 3. 当前代码中的算法流程

当前代码基本保留了原始设计，但实现上做了多轮简化和修正。

### 3.1 Build pipeline

入口：`scripts/build_memory.py`

每个 sample 的构建流程：

1. `load_locomo_sessions()` 读取 LoCoMo session 级数据。
2. `GraphBuilder.build_from_sample()` 遍历 session。
3. 每个 session 按 `k_turns` 切 batch。
4. 每个 batch 执行：
   - RawArchive：保存原文 batch。
   - GraphTrigger：LLM 判断是否进入图写入。
   - GraphLocalizer：基于当前 batch + 参与者节点取 local subgraph。
   - GraphConstructor：LLM 输出 graph ops，代码执行。
   - trajectory 记录 batch 级日志。

当前 `GraphBuilder` 的切分逻辑是：

```python
for i in range(0, len(turns), self.k_turns):
    batch = turns[i : i + self.k_turns]
```

因此：

- `k_turns=4`：小窗口分批构图。
- `k_turns=999`：近似整 session 一次构图。

### 3.2 当前 Construction prompt 的目标

当前 prompt 已经从复杂 ontology 收敛为 retrieval-oriented：

- 不追求 fact 完美转述。
- 允许多余 Event。
- 关键要求是：
  - key information 进入 Event。
  - Entity→Event 主体边正确。
  - 命名对象/伴随者连到对应 Event。
  - planned / experienced 谓词大体正确。
  - local subgraph 只能用于复用 ID，不能作为新 fact 来源。

当前主要 op schema：

```json
{"op": "EnsureEntity", "id": "NEW_<label>", "canonical_name": "...", "aliases": []}
{"op": "EnsureEvent",  "id": "NEW_<label>", "canonical_name": "...", "attrs": {"fact": "...", "time": "..."}}
{"op": "Relate", "src": "<id>", "dst": "<id>", "predicate": "experienced|planned|object_of|participant|before|after|updates|inspired"}
{"op": "AttachAttr", "node": "<id>", "key": "...", "value": "..."}
{"op": "MergeNode", "src": "<id>", "dst": "<id>"}
{"op": "Skip", "reason": "..."}
```

当前 prompt 强约束包括：

- Existing Event ID 不能当 Entity ID 使用。
- 缺少 named subject/object 时必须 `EnsureEntity`。
- Event fact 必须包含 subject 名字和重要 named object 名字。
- planned 用于 plan/intention/considering/exploring/researching/joining/volunteering/future action。
- past/current state 用 experienced。
- Skip 只在当前 excerpt 没有 durable memory 时使用。

### 3.3 当前执行层约束

执行层做了几类 guard：

- predicate 白名单，拒绝 `related`、空 predicate、family-predicate 不匹配。
- Entity→Event 连边要求 Entity 名字出现在 Event canonical_name 或 fact 中。
- repair 不再按 first speaker 盲连，只在 Event 文本命中已有 Entity 时补边。
- LLM call 记录写入 `build/llm_calls.jsonl`，包含 messages、response、metadata。

这些 guard 不是最终设计目标，而是为了观察 LLM 输出失败模式、避免污染图。

### 3.4 当前 Retrieval pipeline

入口：`GraphRetriever.answer()`

主要步骤：

1. Localize query：
   - 普通题调用 `localize(question)`。
   - recall-heavy 类别使用多 query variant 的 union localize。
2. SelectAnchor：
   - LLM 从 local subgraph 选 1-3 个 anchor。
3. ReAct-style retrieval：
   - `jump`
   - `raw_fallback`
   - `finish`
4. 对 Cat5 adversarial question：
   - raw_fallback 被屏蔽或强约束。
   - 必须有 explicit statement 才能回答，否则输出 `Not mentioned in the conversation`。
5. max hop 后 forced finish。

### 3.5 当前评估设置

数据集：LoCoMo-10  
主要模型：Qwen3-4B via DashScope  
embedding：BAAI/bge-m3  
主要指标：

- Judge-Acc：LLM judge 判断是否正确。
- Token-level F1：SQuAD 风格词袋 F1。
- Cat5：对抗不可答题，正确答案为 `Not mentioned in the conversation`。

---

## 4. 2026-04-20 基线与主要结论

4/20 报告是后续所有实验的出发点。

### 4.1 GPT-4o 与 Qwen3-4B 初始表现

最初 GPT-4o Cat1-4 全量评估：

| 指标 | GPT-4o 基线 |
|------|-------------|
| Cat1-4 Judge-Acc | 80.7% |
| Avg F1 | 0.508 |

换用 Qwen3-4B 后：

| 类别 | GPT-4o | Qwen3-4B | 变化 |
|------|--------|----------|------|
| Cat1 | 76.8% | 84.4% | +7.6 |
| Cat2 | 75.7% | 74.5% | -1.2 |
| Cat3 | 68.7% | 67.7% | -1.0 |
| Cat4 | 85.1% | 89.3% | +4.2 |
| Cat1-4 Overall | 80.7% | 84.0% | +3.3 |
| Avg F1 | 0.508 | 0.335 | -0.173 |

观察：

- Qwen3-4B Judge-Acc 高于 GPT-4o，但 F1 低很多。
- F1 低的主要原因是 Qwen 输出冗长，pred/gold 长度比约 3.7x。

### 4.2 Oracle 上界

用 gold evidence 直接喂给 Qwen3-4B，绕过图检索：

| 类别 | Qwen3-4B 系统 | Oracle 上界 | 检索损失 |
|------|---------------|-------------|----------|
| Cat1 | 84.4% | 95.7% | -11.3 |
| Cat2 | 74.5% | 92.8% | -18.4 |
| Cat3 | 67.7% | 88.5% | -20.8 |
| Cat4 | 89.3% | 95.5% | -6.2 |
| Overall | 84.0% | 95.1% | -11.1 |

结论：主要瓶颈是检索，而不是 LLM 推理能力。图构建质量和检索路径是核心。

### 4.3 早期图结构问题

以 conv-30 为例：

| 指标 | GPT-4o | Qwen3-4B |
|------|--------|----------|
| entity-event 边数 | 118 | 14 |
| 未连到 Entity 的 Event | 少量 | 88%（67/76） |
| Entity 最大 degree | 60 | 9 |
| event-event 主要谓词 | before/after/updates | discussed/mentions |

问题：

- Event 大量孤立，Entity 出发无法访问。
- event-event 被 `discussed/mentions` 这种对话关系污染。
- 4B 对复杂 prompt 和开放 ontology 遵循不稳定。

### 4.4 P0+P1 改动

4/20 阶段强化了 construction prompt：

- 禁止 session/container event。
- 相对时间解析。
- event-event 谓词限制。
- 有疑问时优先建节点。
- 保留原文词汇，减少泛化。
- 带时间锚点行为独立建 Event。
- 每个 Event 后应该跟 entity-event Link。

全量结果：

| 类别 | Qwen3-4B 旧 Acc | P0+P1 Acc | 变化 |
|------|-----------------|-----------|------|
| Cat1 | 84.4% | 90.4% | +6.0 |
| Cat2 | 74.5% | 78.5% | +4.0 |
| Cat3 | 67.7% | 74.0% | +6.3 |
| Cat4 | 89.3% | 91.2% | +1.9 |
| Cat1-4 | 84.0% | 87.3% | +3.3 |
| Cat5 | 51.9% | 36.1% | -15.8 |

核心收益来自 entity-event 覆盖提升；核心代价是 Cat5 对抗题变差。

---

## 5. 2026-04-27：检索与 proposition layer 实验

### 5.1 实验 001 / 002：refine-jump 与 baseline 对照

`002 baseline-conv26` 使用接近早期版本的 commit `0408882` 跑 conv-26：

| 样本 | 节点 | 边 | Entity | Event | e-ev |
|------|------|-----|--------|-------|------|
| conv-26 | 21 | 30 | 2 | 19 | 21 |

QA：

| 指标 | 002 baseline | 001 refine-jump |
|------|--------------|-----------------|
| Cat1-4 | 84.2% | 86.8% |
| Cat5 | 19.1% | 51.1% |
| Avg F1 | 0.280 | 0.415 |

`001 refine-jump` 的改动：

- jump 候选从关键词打分改为 bge-m3 向量相似度。
- 多 anchor 独立扩展，每个 anchor 分配预算。

观察：

- Cat5 提升很大。
- 但 001 和 002 不只是 retrieval 差异，图规模也差异巨大：21/30 vs 107/120。
- 说明不能把提升简单归因到 jump 算法。

### 5.2 实验 003：Refine1 graph construction

目标：修 Event 孤点和 localizer。

改动：

- 强化 Entity/Event 区分和 Link 规则。
- multi-seed localizer。
- 修复 Jump/raw repair。

conv-41 图：

| 节点 | 边 | Entity | Event | e-ev | event-event |
|------|-----|--------|-------|------|-------------|
| 249 | 404 | 89 | 160 | 273 | 131 |

QA：

| 类别 | Apr20 全量 | 003 conv-41 |
|------|------------|-------------|
| Cat1 | 90.4% | 90.3% |
| Cat2 | 78.5% | 85.2% |
| Cat3 | 74.0% | 62.5% |
| Cat4 | 91.2% | 96.5% |
| Cat1-4 | 87.3% | 91.4% |
| Cat5 | 36.1% | 61.0% |
| Avg F1 | 0.390 | 0.462 |

观察：

- 图连通性提升，Cat5 大幅提升。
- 但图规模很大，Link rejected=153，说明 schema 仍不稳定。

### 5.3 实验 004：Refine2 proposition layer

动机：

- Event 只有 name 不够，检索语义不清。
- 增加 `fact` 和 `quote` 字段，让 node evidence 更接近回答证据。
- 过滤 low-value abstract event。

图统计：

| 样本 | 节点 | 边 | 备注 |
|------|------|-----|------|
| conv-41 | 156 | 164 | 比 003 少 93 节点 |
| conv-26 | 96 | 117 | 比 002 大很多 |

QA：

| 样本 | Cat1-4 | Cat5 | Avg F1 |
|------|--------|------|--------|
| conv-41 | 90.8% | 80.5% | 0.520 |
| conv-26 | 86.8% | 59.6% | 0.420 |

重要结论：

- fact/quote 对 Cat5 帮助明显。
- low-value 过滤让图更小，但可能损失 Cat4。
- Refine2 是目前相对有效的一条线：它让 Event 变成更清楚的证据单元。

### 5.4 实验 005：同图切换 current retrieval

复用 004 图，只切换检索代码。

总体：

| 指标 | 004 retrieval | 005 current retrieval | 变化 |
|------|---------------|-----------------------|------|
| Cat1-4 | 88.8% | 86.8% | -2.0 |
| Cat5 | 69.3% | 67.0% | -2.3 |
| Overall | 84.7% | 82.6% | -2.1 |

conv-41 下降明显：

| 样本 | 指标 | 004 | 005 | 变化 |
|------|------|-----|-----|------|
| conv-41 | Cat1-4 | 90.8% | 86.2% | -4.6 |
| conv-41 | Cat5 | 80.5% | 70.7% | -9.8 |
| conv-26 | Cat1-4 | 86.8% | 87.5% | +0.7 |
| conv-26 | Cat5 | 59.6% | 63.8% | +4.3 |

结论：

- 纯向量 jump + per-anchor budget 有 near-miss 风险。
- 需要 hybrid scoring：向量相似度 + question terms + predicate + constraint + time/object/actor。

### 5.5 实验 006：Qwen JSON Mode

改动：

- construction / retrieval planner 都用 JSON object envelope。
- top_p=0.7, seed=42。
- Qwen DashScope JSON Mode 稳定化。

conv-41 图：

| 节点 | 边 | Entity | Event | e-ev |
|------|-----|--------|-------|------|
| 176 | 243 | 35 | 141 | 228 |

QA：

| 类别 | 006 conv-41 |
|------|-------------|
| Cat1 | 96.8% |
| Cat2 | 92.6% |
| Cat3 | 75.0% |
| Cat4 | 90.7% |
| Cat1-4 | 91.4% |
| Cat5 | 65.9% |
| Overall | 86.0% |

观察：

- JSON Mode 解决了格式稳定性，但不能保证 schema 语义正确。
- build failed_ops=174，空 family 仍有 92 次。
- Cat1-4 高，但 Cat5 仍受 raw fallback near-miss 影响。

---

## 6. 2026-04-28：Cat5、SkipRecovery 与图健康度问题

### 6.1 实验 007：Cat5 fix

动机：

Cat5 的主要错误路径是：

```text
LLM 发起 raw_fallback
  -> 找到话题相关但非答案的 raw turn
  -> 以 near-miss evidence 给出具体答案
```

改动：

- Cat5 raw_fallback 屏蔽。
- Cat5 prompt 强调 explicit statement。
- jump 时间词加权。

同图 QA：

| Category | 005 Acc | 007 Acc | 变化 |
|----------|---------|---------|------|
| Cat1 | 90.5% | 88.9% | -1.6 |
| Cat2 | 89.1% | 87.5% | -1.6 |
| Cat3 | 76.2% | 76.2% | 0 |
| Cat4 | 85.9% | 90.4% | +4.5 |
| Cat1-4 | 86.8% | 88.5% | +1.6 |
| Cat5 | 67.0% | 87.5% | +20.5 |
| Overall | 82.6% | 88.3% | +5.7 |

结论：

- Cat5 raw_fallback 屏蔽非常有效。
- 时间词加权改善 Cat4。
- 但 Cat1/Cat2 小幅退步，说明安全规则和可答题召回存在 trade-off。

### 6.2 实验 008：SkipRecovery + finish check，全 10 样本

动机：

007 错题分析发现：

- Construction 误 Skip：一个 batch 中的具体事实被闲聊淹没。
- Retrieval 早 finish：LLM 凭语义印象或 near-miss 作答。

改动：

1. Skip rule 收窄：只有 every turn 都没有具体事实才 Skip。
2. 如果多 turn batch 被整体 Skip，拆成单 turn 重试。
3. Finish 前自检：答案必须被已有 graph/raw evidence 显式支持。

全量图统计：

| 样本 | 节点 | 边 | Entity | Event | e-ev |
|------|------|-----|--------|-------|------|
| conv-26 | 142 | 216 | 38 | 104 | 198 |
| conv-30 | 171 | 226 | 23 | 148 | 221 |
| conv-41 | 214 | 255 | 54 | 160 | 235 |
| conv-42 | 286 | 321 | 80 | 206 | 302 |
| conv-43 | 283 | 329 | 75 | 208 | 314 |
| conv-44 | 264 | 383 | 50 | 214 | 330 |
| conv-47 | 325 | 411 | 90 | 235 | 378 |
| conv-48 | 298 | 348 | 88 | 210 | 325 |
| conv-49 | 232 | 254 | 44 | 188 | 234 |
| conv-50 | 262 | 333 | 55 | 207 | 314 |
| 总计 | 2477 | 3076 | 597 | 1880 | 2851 |

全量 QA：

| Category | n | Acc | F1 |
|----------|---:|----:|---:|
| Cat1 | 282 | 86.5% | 0.320 |
| Cat2 | 321 | 82.9% | 0.362 |
| Cat3 | 96 | 70.8% | 0.188 |
| Cat4 | 841 | 89.2% | 0.460 |
| Cat1-4 | 1540 | 86.2% | — |
| Cat5 | 446 | 86.3% | 0.851 |
| Overall | 1986 | 86.3% | 0.499 |

重要观察：

- Cat5 稳定在 86.3%，远好于 Apr20 P0+P1 的 36.1%。
- 但 Cat1-4 比 Apr20 P0+P1 的 87.3% 略低。
- 图膨胀明显：节点总数 2477，Event 1880。
- 低度 Event 非常多。全量 Event 中 1039 个 degree=1，531 个 degree=2；83.5% Event 是 degree<=2。

图不健康表现：

1. Rule 8 过强，寒暄/鼓励/抽象价值也被图谱化。
2. 同一 4-turn batch 拆出太多近义 Event。
3. 大量重复 canonical event。
4. 泛化 Entity 节点出现，例如 `Project`, `Purpose`, `Nature`, `Gaming`。
5. 边类型机械补齐，`participant/object_of/experienced` 占 75.5%。
6. prompt 禁止的弱谓词仍然出现。

结论：

> 008 的 A3+C1 恢复了少数关键事实，但没有质量门控，导致图从“漏事实”滑向“过度事实化”。

---

## 7. 2026-04-28：围绕 construction 重建的实验 009-017

从 009 开始，实验重点从 retrieval 转向“如何重新建图”。

### 7.1 实验 009：fact-first construction

假设：

把 construction 拆成两阶段：

1. Stage 1：只抽 fact + dialogue_time。
2. Stage 2：把 fact 编译成 `EnsureEntity/EnsureEvent/Relate`。

期望减少空/错 family、无意义节点、过度建边。

实际 conv-41 图：

| 节点 | 边 | Entity | Event | e-ev | 无 Entity 入边 Event | degree<=1 | degree<=2 |
|------|-----|--------|-------|------|----------------------|-----------|-----------|
| 291 | 275 | 33 | 258 | 275 | 4 | 238 | 257 |

问题：

- Event 爆炸，258 个 Event。
- 绝大多数 Event 是 degree<=1 或 degree<=2。
- fact-first 如果没有 strong dedup / merge，会把事实抽成大量叶子节点。

### 7.2 实验 010 / 011：simplified ops

改动：

- `CreateEntity/CreateEvent/Link` 改成 `EnsureEntity/EnsureEvent/Relate`。
- edge family 自动从节点类型推断。
- Event attrs 简化为 fact/time。
- 对话输入格式更干净。

结果：

| 实验 | 样本 | 节点 | 边 | Entity | Event | e-ev | e-e | ent-ent |
|------|------|------|-----|--------|-------|------|-----|---------|
| 010 | conv-26 | 46 | 236 | 16 | 30 | 119 | 96 | 21 |
| 011 | conv-41 | 77 | 198 | 11 | 66 | 142 | 48 | 8 |

问题：

- 节点数下降，但边数相对过高。
- 出现 Entity-Entity 边和大量 Event-Event 边。
- Event 不再孤立，但图出现“高边密度、语义弱”的趋势。

### 7.3 实验 012：session-level construction

配置：

- `k_turns=999`，近似整 session 一次构图。
- conv-41 从 179 batches 降到 32 batches。

结果：

| 节点 | 边 | Entity | Event | e-ev | e-e | ent-ent |
|------|-----|--------|-------|------|-----|---------|
| 51 | 198 | 4 | 47 | 145 | 52 | 1 |

问题：

- 速度快、batch 少。
- 但 Entity 严重不足，只有 4 个。
- 模型倾向把概念塞进 Event，而不保留 Coco/Max/David 等命名对象。
- 边很多，但 anchor 少，不利于检索。

### 7.4 实验 013：entity-fix

目标：修正 Rule 2/4，让 Max、Coco、David 等命名角色成为独立 Entity。

结果：

| 节点 | 边 | Entity | Event | e-ev | related 边 |
|------|-----|--------|-------|------|------------|
| 74 | 161 | 5 | 69 | 141 | 92 |

问题：

- Entity 仍只有 5。
- 因 Relate 示例缺 predicate，执行层默认补成 `related`，产生 92 条弱语义边。

### 7.5 实验 014：predicate guard

改动：

- Relate 示例显式 predicate。
- 拒绝 `related`、空 predicate、非法 family-predicate。

结果：

| 指标 | 013 | 014 | 变化 |
|------|-----|-----|------|
| 节点 | 74 | 94 | +20 |
| 边 | 161 | 150 | -11 |
| Entity | 5 | 11 | +6 |
| Event | 69 | 83 | +14 |
| event-event | 18 | 8 | -10 |
| related 边 | 92 | 0 | -92 |
| 非法 event-event predicate | 17 | 0 | -17 |

问题：

- `related` 消失。
- 但 Event 仍碎，degree<=2 Event 为 62/83。
- 模型仍会把 Entity-Event predicate 用到 Event-Event 上，只是被执行层拒绝。

### 7.6 实验 015：LLM log + edge guard

改动：

- 记录完整 LLM 调用：`build/llm_calls.jsonl`。
- Entity→Event 连边要求 Entity 名字出现在 Event name/fact 中。
- `planned` 成为合法 predicate。
- repair 不再按 first speaker 盲连。

结果：

| 指标 | 014 | 015 |
|------|-----|-----|
| 节点 | 94 | 98 |
| 边 | 150 | 116 |
| Entity | 11 | 8 |
| Event | 83 | 90 |
| entity-event | 142 | 112 |
| event-event | 8 | 4 |
| related | 0 | 0 |

LLM call log：

- 64 条调用：32 trigger + 32 construction。
- 每条包含 sample_id / batch_id / session_id / phase / request / response。

问题转移：

- 弱边减少。
- 但 fact 本身出错：
  - 主体错：如 John had Coco，但原文是 Maria got Coco。
  - 时间错：`with family for 10 years` 被写成 `passed away 10 years ago`。
  - 对话行为被事实化：shared photo / talked about。
  - listener 被写进事实主体：Maria and John received letter，但原文只有 Maria。

结论：

> 015 后，瓶颈从“边弱/非法”转移到“fact 抽取的主体、时间、对象不稳”。

### 7.7 prompt 单 session 测试：session30

我们专门用 conv-41 session30 做 prompt 迭代。这个 session 的关键事实包括：

- Maria got a puppy named Coco two weeks ago。
- John and Max had a camping trip last summer。
- John felt stuck / questioned decisions and goals。
- John explored joining local organizations or volunteering programs。

重要 prompt 版本观察：

| 版本 | 观察 |
|------|------|
| v3 | 输出好，但有 few-shot 泄漏，不能作为有效测试 |
| v4/v5 | zero-shot 下抽太多 event，包括评论/照片/感谢 |
| v6 | fact 有改善，但实体变泛化：Pet, CampingTrip, NatureExperience |
| v7 | 连边方向改善，但 fact 仍像 quote 拼接 |
| v8 | fact 变成第三人称记忆句，但 Coco/Max 对象边漏掉 |
| v9 | 对象边补上，但又退回 `John said...` |
| v10 | 5 个 Event，关键边基本正确；仍有 photo 低价值 Event |
| v15 actual prompt | 结构正确，能覆盖核心边；但整 session/full build 下仍受 local subgraph 干扰 |

关键结论：

- 不应该过分追求 fact 文本完美。
- 更重要的是 key information 和 Entity→Event 对应关系。
- 但 prompt-only 很脆弱：只要 local subgraph 复杂，模型会把旧 Event ID 当 Entity，或从旧 subgraph 重复抽事实。

### 7.8 实验 016：prompt-edge-focused

目标：

- 放松 fact 完美性。
- 强化主体边、对象边、planned 谓词。
- 移除 concrete few-shot 示例，避免泄漏。

结果：

| 节点 | 边 | Entity | Event | e-ev | 无入边 Event | unused Entity |
|------|-----|--------|-------|------|--------------|---------------|
| 89 | 104 | 11 | 78 | 101 | 3 | 6 |

问题：

- LLM 在 session30 把已有 Event ID `0b08d85e` 当 Entity 使用。
- 多条边被执行层按 event-event 拒绝。
- 最终依赖 repair 补边。

### 7.9 实验 017：existing-id type guard

增加 prompt 规则：

- 复用 existing node 时必须匹配 type。
- Event ID 不能作为 Entity→Event 的 src。
- 缺失 John/Maria/Coco/Max 等 Entity 时必须 EnsureEntity。

结果：

| 指标 | 016 | 017 |
|------|-----|-----|
| 节点 | 89 | 95 |
| 边 | 104 | 110 |
| Entity | 11 | 17 |
| Event | 78 | 78 |
| entity-event | 101 | 110 |
| 无 Entity 入边 Event | 3 | 0 |
| Entity-Entity | 0 | 0 |
| rejected op | 81 | 36 |
| RepairEventLink | 27 | 9 |

改善：

- 结构层明显更干净。
- 没有 Entity-Entity。
- 没有无入边 Event。
- Event ID 当 Entity 的问题明显缓解。

副作用：

- 模型变保守，session30 只抽了 Coco puppy，漏掉 Max camping、stuck、volunteering。
- Coco Entity 被创建，但模型先输出了 Maria→Coco 的 entity-entity `object_of`，被拒绝，最终 Coco 没连到 puppy Event。

结论：

> 单纯增强 type guard 会提高结构正确性，但会降低 key fact recall。

---

## 8. 最新单 session k_turns=4 测试

用户提出：不要整样本，先看单 session，测试 `k_turns=4`。

测试对象：

- sample: `conv-41`
- session: `session_30`
- batch size: `k_turns=4`
- 从空图开始，只跑该 session，不跑 QA。

### 8.1 v1：原 prompt + k=4

路径：`runs/single_session_k4_conv41_s30`

结果：

| 节点 | 边 | Entity | Event |
|------|-----|--------|-------|
| 8 | 5 | 4 | 4 |

覆盖：

- Maria got puppy / Coco：有。
- Max camping：有。
- John stuck：漏。
- volunteering plan：漏。

问题：

- 后半段 batch 被 Skip。
- `exploring options / volunteering` 被误判为 low-value dialogue act。
- subject 有错连，例如 Maria/John 混淆。

### 8.2 v2：补充 plan/volunteering 不可 Skip + subject name 匹配

路径：`runs/single_session_k4_conv41_s30_v2`

结果：

| 节点 | 边 | Entity | Event |
|------|-----|--------|-------|
| 15 | 14 | 4 | 11 |

改善：

- John stuck 被抽出来。
- 后半段不再全 Skip。

问题：

- local subgraph 污染严重。
- 后续 batch 反复重建 camping/goal reflection。
- 仍漏 volunteering。
- 出现重复 Event：多个 `Camping trip with Max`, `Camping trip with Maria`, `Reflection on goals`。

### 8.3 v3：local subgraph 只用于复用 ID，不作为新 fact 来源

路径：`runs/single_session_k4_conv41_s30_v3`

结果：

| 节点 | 边 | Entity | Event | 结构问题 |
|------|-----|--------|-------|----------|
| 11 | 11 | 4 | 7 | 0 |

Entities：

- John
- Maria
- Coco
- Max

Events：

| Event | fact | 入边 |
|-------|------|------|
| Maria got a puppy | Maria got a puppy two weeks ago. Her name's Coco and she's adorable. | Maria experienced; Coco object_of |
| Camping trip with Max | John and Max had a blast on their camping trip last summer... | John planned; Max participant |
| Camping trip with Maria and John | John and Maria talked about a camping trip... | Maria participant/object_of; John experienced |
| John questioning his decisions and goals | John has been questioning his decisions and goals lately... | John experienced |
| Exploring options | John has been exploring options... joining local organizations or volunteering programs | John planned |
| Researching organizations | John is researching local organizations and volunteering programs... | John planned |
| John took photos of the camping trip | John took photos... | John experienced |

v3 的好处：

- 四个关键事实都覆盖了。
- 没有 Entity-Entity 边。
- 没有无 Entity 入边 Event。
- Coco / Max 对象边存在。
- Volunteering plan 被正确抽出，并用 planned。

v3 的问题：

- `Camping trip with Max` 的 John 边用了 `planned`，应该是 `experienced`。
- 出现 `Camping trip with Maria and John` 这类污染/重复 Event。
- `John took photos...` 是低价值 Event。
- per-turn SkipRecovery 仍可能从 local subgraph 带入旧事实。

结论：

> k_turns=4 比整 session 更适合 recall，但必须配合去重、局部事实来源控制、predicate 稳定化和低价值事件过滤。

---

## 9. 当前图结构“不健康”的具体表现

### 9.1 Event 叶子化

多次实验出现大量低度 Event。

| 实验 | 样本 | Event | degree<=1 | degree<=2 |
|------|------|-------|-----------|-----------|
| 008 | conv-41 | 160 | 92 | 134 |
| 009 | conv-41 | 258 | 238 | 257 |
| 014 | conv-41 | 83 | 44 | 62 |
| 015 | conv-41 | 90 | 64 | 86 |
| 017 | conv-41 | 78 | 60 | 70 |

低度 Event 的含义：

- 很多 Event 只有一个 Entity 入边。
- 它们可以被检索到，但几乎不能支持图上跳转。
- 对 Localize 来说，它们增加候选空间。
- 对 Retrieval 来说，它们制造 near-miss。

这说明当前图更像“事实列表 + 弱边”，而不是真正有结构的图。

### 9.2 Entity 过少或过泛

不同实验出现两种极端：

- session-level：Entity 只有 4，Coco/Max 等被埋进 Event。
- 008 大图：出现很多泛化 Entity，如 Project, Purpose, Nature, Gaming。

健康 Entity 应该是稳定检索 anchor，而不是临时 topic label。

当前 prompt 已经加入规则禁止：

- Experience
- Reflection
- Idea
- Positivity
- Nature
- CampingTrip
- Pet
- VolunteeringIdeas

但模型仍可能变体输出泛化节点。

### 9.3 边被机械补齐

为了避免孤立 Event，prompt 曾要求每个 Event 都必须有 Entity-Event 边。但副作用是：

- `participant`, `object_of`, `experienced` 被机械使用。
- listener 被错连成 participant/experienced。
- object_of 被用于对话对象，而不是真正 object。

008 全量 top predicates：

- participant: 1142
- object_of: 613
- experienced: 567

三者合计 2322 条，占 3076 条边的 75.5%。这说明很多边只是为了满足连通性，而不是强语义。

### 9.4 Fact 主体错误

典型例子：

- 原文：Maria got a puppy named Coco。
- 错误 fact：John and Maria adopted Coco。

原因：

- 长 session 中多个 speaker 互相回应，LLM 容易把 listener 也写进事实主体。
- local subgraph 里已有相关 Event，会污染当前 batch 抽取。
- prompt 过度要求覆盖，会让模型把对话参与者都纳入 Event。

### 9.5 时间错误

典型例子：

- 原文：Max was with family for 10 years。
- 错误：Max passed away 10 years ago。

原因：

- 模型把 duration 当成 event time。
- 相对时间/持续时间/对话时间三者混淆。

当前策略：

- prompt 要求 preserve relative time，不要 fabricate exact dates。
- 但模型仍不稳定。

### 9.6 对话行为事实化

常见低价值 Event：

- shared photo
- thanked
- encouraged
- asked
- commented
- talked about
- reflected on

这些有时对 retrieval 有一点帮助，但会显著增加噪声。当前用户判断是：不必过分追求 fact 完美，低价值 Event 可以在 retrieval 过滤，但关键事实必须清楚、边必须正确。

### 9.7 local subgraph 污染

016/017/session k=4 测试都显示：

- Local subgraph 是必要的，因为需要复用已有节点。
- 但 LLM 会把 local subgraph 里的旧 Event 当作当前 input 的事实来源。
- 甚至会把 Event ID 当 Entity ID 使用。

因此 prompt 已加：

- local subgraph only for ID reuse / merge / update。
- new Event fact must be supported by current Input excerpt。
- Event ID cannot be Entity src。

但这仍然不是完全可靠的代码层保证。

### 9.8 Skip 与 recall 的矛盾

Skip 过强：

- 漏掉嵌入事实。
- 007 中 Cat1 错题很多来自 construction 误 Skip。

Skip 过弱：

- 008 图膨胀。
- 抽象支持、鼓励、价值判断被图谱化。

目前看：

- 整 session：LLM 会压缩，漏关键事实。
- k=4：recall 更好，但重复/污染更多。

---

## 10. 当前经验结论

### 10.1 Retrieval 角度

更有帮助的图元素不是越多越好，而是：

- 稳定人物/对象 Entity。
- 事实 Event 必须能被多个查询词命中。
- Event fact 中必须包含 subject 和 named object。
- Entity→Event 边要能让问题从人物或对象跳到事实。
- planned / experienced 区分能帮助过滤未来计划 vs 已发生事实。
- time 字段要保留相对时间，不乱转绝对日期。

不健康元素：

- 泛化 Entity。
- 只有一个边的低价值 Event。
- conversation act Event。
- 重复 paraphrase Event。
- listener 错连。

### 10.2 Construction 角度

当前 prompt-only 方案存在上限：

- 同一条规则强化后，经常会牺牲另一项能力。
- type guard 提高结构正确性，但降低 recall。
- coverage checklist 提高 recall，但增加重复和污染。
- Skip 收窄提高 recall，但造成图爆炸。
- session-level 降调用量，但漏关键事实。
- k=4 提高局部 recall，但 batch 间重复更多。

### 10.3 代码层 guard 的作用

代码层 guard 对避免污染图很有必要：

- 拒绝非法 predicate。
- 拒绝 Entity 名不在 Event fact/name 中的边。
- 记录 llm_calls。
- repair 不盲连。

但用户当前倾向是：

> 不要在代码层做太多“事实语义校验”，先通过 prompt 和流程设计让模型输出正确结构。  
> 低价值 Event 可以存在，retrieval 时排除。关键是 key facts 和连边正确。

---

## 11. 可能的重新建图方向

下面是供讨论的设计方向，不是最终结论。

### 11.1 从“Event schema extraction”转为“Memory fact node”

Event 节点不再承担复杂 schema，只存：

```json
{
  "type": "Event",
  "fact": "Maria got a puppy named Coco two weeks ago.",
  "conversation_time": "12:10 am on 11 August, 2023",
  "source_turn_ids": ["D30:1"]
}
```

不要求：

- event_type
- actor list
- object list
- quote
- complicated attributes

Entity 和 edge 负责提供结构 anchor。

### 11.2 两阶段但更轻：Fact first + deterministic edge proposal?

可考虑：

1. Stage A：小窗口抽 normalized memory facts。
2. Stage B：从 fact 文本中抽 named Entity anchors 和 planned/experienced predicate。
3. Stage C：合并重复 facts。

不同于 009 的 fact-first，关键是必须有 dedup 和质量门控，否则会 Event 爆炸。

### 11.3 小窗口 construction + session-level consolidation

推荐流程：

```text
session
  -> split into k=4 or topic windows
  -> extract candidate facts/events
  -> session-level consolidate/merge
  -> graph write
```

理由：

- k=4 能捕获局部 facts。
- session-level 用来去重、合并、纠正主体，而不是直接抽全图。

### 11.4 local subgraph 只做 ID alignment

Construction 输入可拆成两个区域：

```text
[Current input excerpt]
必须从这里抽新事实。

[Reusable graph nodes]
只用于复用 ID，不允许作为事实来源。
```

甚至可以不提供完整 local subgraph，而只提供候选 Entity table：

| id | type | canonical_name |
|----|------|----------------|
| 1cf8b8a4 | Entity | John |
| cdf87837 | Entity | Maria |
| ... | ... | ... |

这样减少 Event 污染。

### 11.5 先抽取，后落图

把 LLM 输出和图操作分离：

```json
{
  "memory_facts": [
    {
      "fact": "...",
      "source_turn_ids": [...],
      "subject": "John",
      "named_objects": ["Max"],
      "status": "experienced"
    }
  ]
}
```

然后系统 deterministic 编译为：

- EnsureEntity(subject)
- EnsureEntity(named_objects)
- EnsureEvent(fact)
- Relate(subject, Event, status)
- Relate(object, Event, participant/object_of)

优点：

- LLM 不直接操作 graph ID，减少 Event ID 当 Entity ID 的问题。
- 代码可以更容易去重和检查 missing edge。

风险：

- 用户不希望代码层做过多语义校验。
- 需要定义 minimal schema，不能让 LLM 输出太多。

一个折中 schema：

```json
{
  "fact": "John and Max had a camping trip last summer.",
  "time": "last summer",
  "subject": "John",
  "predicate": "experienced",
  "objects": [{"name": "Max", "role": "participant"}],
  "source_turn_ids": ["D30:6"]
}
```

### 11.6 图后处理：dedup / merge 是必须的

无论 prompt 如何，k=4 都会产生重复。

需要至少做：

- 同 subject + 同 named object + 高 fact similarity + 时间一致 => merge Event。
- Canonical name 相近但 source 不同，保留 source_turn_ids union。
- 对 conversation act Event 降权，而不是一定删除。

### 11.7 Retrieval-time precision

如果 construction 选择 recall-first，则 retrieval 必须做 precision：

- 对低价值 Event 降权。
- 对 planned/experienced 按问题时态过滤。
- 对 subject/object/time 约束做 hard match。
- 对 Cat5 做 premise check。
- 对 raw_fallback 的 near-miss 做 answerability check。

---

## 12. 推荐和老师讨论的核心问题

### 12.1 Event 应该是什么？

当前有两种路线：

1. Event = episode / interaction / schema-rich event
2. Event = one memory fact + conversation time

目前实验更支持第二种。原因：

- schema-rich extraction 对 4B 模型不稳定。
- 太多类型/属性会导致错误 schema。
- retrieval 需要的是可命中的事实句，而不是完整 ontology。

### 12.2 Entity 应该多细？

应保留：

- 人名
- 宠物名/动物名
- 具体地点/组织
- 具体物品

应避免：

- Nature
- Experience
- Reflection
- Idea
- CampingTrip
- VolunteeringIdeas
- generic topic

需要讨论：

- local organizations / volunteering programs 这种未命名概念是否建 Entity？
  - 当前倾向：不建 Entity，放在 Event fact 中即可。
  - 如果未来 retrieval 需要概念 anchor，可以另设 `Concept` 或 `Topic`，但不要混入 Entity。

### 12.3 是否还需要 Event-Event 边？

当前 Event-Event 边经常带来噪声。

可能方案：

- construction 阶段暂时不让 LLM 输出 Event-Event。
- 只保留系统可确定的 temporal ordering：
  - same session order
  - explicit before/after
  - updates
- retrieval 主要走 Entity→Event。

### 12.4 construction 应该整 session 还是分 batch？

实验结论：

- 整 session：速度快，但漏关键事实，Entity 少。
- k=4：key fact recall 更好，但重复/污染多。

推荐：

- 用 k=4 或 topic chunk 抽候选 facts。
- 用 session-level consolidation 合并。
- 不建议直接整 session 落图。

### 12.5 prompt-only 是否足够？

目前看 prompt-only 不足以保证：

- ID type 正确。
- subject edge 正确。
- planned/experienced 稳定。
- 不从 local subgraph 抽旧事实。
- 去重。

但代码层语义校验也不宜过度复杂。合理边界可能是：

- 代码负责 structural validity 和 deterministic compilation。
- LLM 负责 fact selection / summarization / subject-object extraction。
- 代码不判断 fact 是否真实，只保证输出结构可落图。

---

## 13. 当前可复现实验路径

正式实验：

- Apr20 报告：`docs/report_progress_apr20.md`
- 实验汇总：`experiments/README.md`
- 001-017 notes：`experiments/*/notes.md`

最新 construction 相关：

- `experiments/2026-04-28-014-predicate-guard-conv41`
- `experiments/2026-04-28-015-llm-log-edge-guard-conv41`
- `experiments/2026-04-28-016-prompt-edge-focused-conv41`
- `experiments/2026-04-28-017-existing-id-type-guard-conv41`

单 session k=4：

- `runs/single_session_k4_conv41_s30`
- `runs/single_session_k4_conv41_s30_v2`
- `runs/single_session_k4_conv41_s30_v3`

关键日志：

- `build/llm_calls.jsonl`
- `build/graph_trajectories_conv-41.jsonl`
- `build/graphs/conv-41_graph.json`

---

## 14. 一句话总结

从 4/20 到现在，GraphMemory 的主要进展是：

- 证明检索瓶颈主要来自图构建，而不是 LLM 推理能力。
- 通过 entity-event 覆盖、fact/quote、JSON mode、Cat5 raw_fallback 策略，显著提升了多项指标。
- 但 construction 逐步暴露出更深层问题：当前图不是“结构化健康记忆图”，而是在“事实召回”和“图噪声”之间摆动。
- 最新单 session k=4 说明小窗口能恢复关键事实，但需要 session-level consolidation、去重和更清晰的 Entity/Event 定义。

因此下一阶段不应继续单纯加 prompt 规则，而应重新设计 construction：

```text
small-window fact recall
  -> minimal fact schema
  -> deterministic graph compilation
  -> session-level merge/dedup
  -> retrieval-time precision filtering
```

这可能比继续让 LLM 直接输出复杂 graph edits 更稳定。


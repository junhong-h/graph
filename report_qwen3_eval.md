# GraphMemory 评测报告 — Qwen3-4B（DashScope）

> 日期：2026-04-16
> 数据集：LoCoMo-10（10 个样本，1986 题，Cat1-5）
> 模型：Qwen3-4B via DashScope API（构建 + 检索 + 评测）
> Prompt 版本：P0（词汇保留）+ P1（时间行为）+ 2b（禁止容器节点）+ 3b（相对时间解析）+ R1（拒绝幻觉）
> 评测指标：Token-level F1、BLEU-1、LLM-as-Judge（Acc）
> 新增：Oracle 基线（直接给证据文本）、Cat5 对抗性问题测试

---

## 1. 整体结果（含 Cat5）

| 类别 | 定义 | 题数 | 系统 Acc | Oracle Acc | 检索损失 |
|------|------|------|---------|-----------|---------|
| Cat1 | 跨Session聚合 | 282 | 84.4% | 95.7% | −11.3 |
| Cat2 | 时序问答 | 321 | 74.5% | 92.8% | −18.4 |
| Cat3 | 推理/常识 | 96 | 67.7% | 88.5% | −20.8 |
| Cat4 | 单Session单事实 | 841 | 89.3% | 95.5% | −6.2 |
| **Cat5** | **对抗性/不可答** | **445** | **51.9%** | — | — |
| **Overall (Cat1-4)** | | **1540** | **84.0%** | **95.1%** | **−11.1** |
| Overall (含Cat5) | | 1986 | 76.8% | — | — |

---

## 2. Cat5 — 对抗性问题测试（幻觉检测）

### 定义

Cat5 问的是**对话中从未发生的事情**，正确行为是拒绝回答或说明"未提及"。数据集提供 `adversarial_answer`（诱骗性答案），用来衡量模型是否被引导产生幻觉。

### 结果对比

| 版本 | 正确拒绝率 | 幻觉率 | 含拒绝关键词 |
|------|-----------|--------|------------|
| 无拒绝规则 | 31.5% | 68.5% | 18.7% |
| **加 R1 规则后** | **51.9%** | **48.1%** | **44.3%** |
| 提升 | +20.4 pts | −20.4 pts | +25.6 pts |

**R1 规则**（加入 `graph_retrieval.py` `_SYSTEM_PROMPT`）：
> "If the evidence does NOT support the question's premise (the described event never happened or is not mentioned anywhere), call finish with answer 'Not mentioned in the conversation' — do NOT fabricate."

### 剩余 48.1% 幻觉根本原因

图中存在**主题相关节点**（如存在 Melanie 节点、adoption 节点），但不存在问题前提所需的**具体事实**。模型看到半相关证据反而更容易幻觉，而非完全无证据时更倾向拒绝。

典型模式：
```
Q: Why did Melanie choose the adoption agency?
→ 图中有 Melanie 和 adoption 节点，但 Melanie 选择某机构的理由从未出现
→ pred: "Caroline chose the adoption agency because they help LGBTQ+ folks"（混淆主体+编造理由）
```

---

## 3. 图结构分析

### 3.1 各样本图统计

| 样本 | 节点 | 边 | Entity | Event | E-Ev边 | Ev-Ev边 | En-En边 | 有时间% | MaxDeg | AvgDeg |
|------|------|-----|--------|-------|--------|--------|--------|---------|--------|--------|
| conv-26 | 96 | 60 | 3 | 93 | 25 | 31 | 0 | 100% | 15 | 3.2 |
| conv-30 | 78 | 31 | 2 | 76 | 14 | 16 | 0 | 97% | 9 | 3.4 |
| conv-41 | 56 | 71 | 2 | 54 | 26 | 43 | 0 | 100% | 17 | 3.4 |
| conv-42 | 116 | 69 | 2 | 114 | 23 | 46 | 0 | 100% | 8 | 1.5 |
| conv-43 | 141 | 188 | 2 | 139 | 104 | 75 | 0 | 100% | 48 | 4.5 |
| conv-44 | 125 | 183 | 2 | 123 | 107 | 64 | 4 | 96% | 86 | 4.9 |
| conv-47 | 138 | 198 | 3 | 135 | 102 | 75 | 0 | 100% | 56 | 4.8 |
| conv-48 | 117 | 81 | 5 | 112 | 49 | 27 | 2 | 100% | 24 | 3.2 |
| conv-49 | 129 | 114 | 4 | 125 | 57 | 54 | 0 | 100% | 21 | 3.6 |
| conv-50 | 156 | 113 | 5 | 151 | 18 | 95 | 0 | 100% | 6 | 2.2 |

### 3.2 关键结构问题

**问题 A：entity-event 边严重稀疏（最核心问题）**

以 conv-30 为例（GPT-4o vs Qwen3-4B）：

| 指标 | GPT-4o | Qwen3-4B |
|------|--------|---------|
| entity-event 边 | **118** | **14** |
| event-event 边 | 26 | 16 |
| 未连接到 Entity 的 Event 节点 | 少量 | **67/76（88%）** |
| Entity 最大 degree | 60 | 9 |

**88% 的 Event 节点没有 entity-event 边**，意味着检索时从 Entity 节点出发，只能 jump 到 9 个事件，其余 67 个事件**完全不可达**。这是 Cat1/Cat2 检索损失的主因。

**问题 B：entity-event 谓词语义退化**

最常见的 entity-event 谓词：
```
spoke_to    (11次) ← 等同于"对话了"，无区分度
experienced  (2次)
planning     (1次)
```

应有的语义丰富谓词（GPT-4o 风格）：`participated`, `visited`, `decided`, `experienced`, `owns`, `launched` 等完全缺失。

**问题 C：entity-entity 边几乎为零**

10 个样本中只有 conv-44（4条）和 conv-48（2条）有 entity-entity 边，其余均为 0。人物关系（family_member、partners、colleagues）未被建模，跨实体查询无法利用关系边。

**问题 D：event-event 谓词语义退化**

最常见：`discussed`（201）、`mentions`（186）、`participated`（118），而应有的时序谓词 `before/after/updates` 总计不到 60 次。大量 event-event 边表达的是"提到了"而非时间顺序，时序链退化为无向关联图。

**问题 E：conv-50 结构异常（AvgDeg 仅 2.2）**

156 个 Event 节点，只有 18 条 entity-event 边，18/95 的 ev-ev/e-ev 比例说明模型对该样本几乎只建了时间链，Entity-Event 锚定完全失效，导致 conv-50 准确率仅 80.4%。

---

## 4. 检索行为分析

### Max hops 达到上限的频率

系统 QA 日志中大量出现 `Max hops (3) reached. Forcing answer.`，说明模型普遍无法在 3 跳内找到答案后主动 finish，而是被迫强答。

原因：Entity-Event 边稀疏 → 第 1 跳从 Entity 只能到达少数节点 → 第 2/3 跳在事件链上漫游 → 找不到目标 → max_hop 强制结束。

---

## 5. 与 GPT-4o 对比

> GPT-4o 使用旧版 prompt（无 P0/P1/2b/3b/R1），conv-26 无 LLM judge；仅供参考。

| 指标 | GPT-4o（旧 prompt） | Qwen3-4B（新 prompt） |
|------|-------------------|---------------------|
| Overall Acc (Cat1-4) | 80.7% | **84.0%** |
| Cat1 | 76.8% | **84.4%** |
| Cat2 | 75.7% | 74.5% |
| Cat3 | 68.7% | 67.7% |
| Cat4 | 85.1% | **89.3%** |
| Avg F1 | **0.508** | 0.335 |
| Cat5 | 未测 | 51.9% |

Qwen3-4B 在 Acc 上以 4B 参数超过 GPT-4o，F1 偏低是输出冗长所致（pred/gold 长度比均值 3.7x）。

---

## 6. 下一步修改方案

### 优先级排序

| 优先级 | 方向 | 改动位置 | 预期收益 |
|--------|------|---------|---------|
| **P0** | 修复 entity-event 边稀疏 | Construction prompt | Cat1+2 Acc +5~8 pts |
| **P1** | 修复 event-event 谓词（时序链） | Construction prompt | Cat2 Acc +2~3 pts |
| **P2** | 多查询 Seed 检索 | graph_localize.py | Cat1 Acc +3~4 pts |
| **P3** | 默认 raw_fallback 补充原文 | graph_retrieval.py | Cat4 F1 +0.05~0.1 |
| **P4** | 简洁性约束（减少冗长） | graph_retrieval.py | F1 +0.05~0.1 |
| **P5** | Cat5 前提验证 | graph_retrieval.py | Cat5 +10~15 pts |

### P0：修复 entity-event 边（最高优先）

**目标**：每个 Event 节点必须至少连接到一个 Entity 节点。

在 `graph_construction.py` `_SYSTEM_PROMPT` 规则末尾追加：

```
12. ENTITY-EVENT LINKING (MANDATORY): Every CreateEvent MUST be followed by at least one
    Link operation connecting it to the relevant Entity node(s) via entity-event family.
    The predicate must describe the relationship specifically:
    Use: experienced / participated / visited / decided / launched / owns / mentioned
    Do NOT use: spoke_to / discussed / related_to (these are too generic)
    Bad:  {"op": "Link", "src": "NEW_Gina", "dst": "NEW_AdCampaign", "family": "entity-event", "predicate": "spoke_to"}
    Good: {"op": "Link", "src": "NEW_Gina", "dst": "NEW_AdCampaign", "family": "entity-event", "predicate": "launched"}
```

### P1：修复 event-event 时序谓词

在规则 4 后追加约束：

```
4b. For event-event edges, ONLY use these predicates: before / after / updates / inspired.
    Do NOT use: discussed / mentions / followed_by / spoke_to / participated.
    "followed_by" → use "before" (A before B) or "after" (B after A) instead.
```

### P2：多查询 Seed 检索

修改 `graph_localize.py` 的 `_seed_retrieval()`，用 LLM 对 query 扩展 2-3 个变体，分别检索后合并去重。对 Cat1 聚合题收益最大。

### P3+P4：检索输出优化

`graph_retrieval.py` 的 answer format 加：

```
"Answer with the shortest possible phrase. Match the style of the expected answer.
If gold is a single date, reply only the date. If gold is a name, reply only the name."
```

### P5：Cat5 前提验证增强

当前 R1 规则依赖 LLM 自我判断，有 48% 漏网。可改为在 SelectAnchor 之后增加一步"前提核查"：

```python
# 伪代码
premise_valid = self._check_premise(question, local_sub)
if not premise_valid:
    return {"answer": "Not mentioned in the conversation", "traces": []}
```

前提核查本身也是一次 LLM 调用，只需问："Based on the subgraph, does the question's premise hold?"

---

## 7. 执行顺序建议

```
本轮（prompt 改动，重建图）：
  P0 + P1 → rebuild conv-30 → QA验证（约20分钟）→ 若提升，全量rebuild

下轮（代码改动，不需重建图）：
  P2（多查询Seed）→ 只跑QA → 验证Cat1提升
  P3+P4（输出简洁）→ 只跑QA → 验证F1提升

长期：
  P5（Cat5前提验证）→ 独立实验
```

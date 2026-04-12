# Graphmemory — 实验报告

**日期**：2026-04-08  
**数据集**：Locomo（conv-26，1 个 sample）  
**模型**：gpt-4o

---

## 一、做了什么

### 系统架构

参考 Mem-T 论文，在 Graphmemory 框架上实现了完整的长对话记忆构建与检索流水线：

```
原始对话 (locomo10.json)
    ↓  dataset.py
Session-grouped 格式
    ↓  memory_builder.py
┌─────────────────────────────────┐
│  Memory Formation (LLM)         │  每 4 轮对话一个 batch
│  → create_fact                  │  提取事实、经验、人物画像、摘要
│  → create_experience            │
│  → update_persona               │
│  → update_summary               │
└──────────┬──────────────────────┘
           ↓
┌─────────────────────────────────┐
│  Memory Update (LLM + ChromaDB) │  每条新 item 检索 top-3 相似条目
│  → ADD / UPDATE / DELETE /      │  决策是否新增、更新或忽略
│     IGNORE                      │
└──────────┬──────────────────────┘
           ↓
      ChromaDB (持久化向量库)
           ↓  memory_retrieval.py
┌─────────────────────────────────┐
│  Memory Retrieval (ReAct 多轮)  │  最多 6 步，工具驱动检索
│  → search_facts                 │
│  → search_experiences           │
│  → search_personas              │
│  → search_summary               │
│  → search_turns                 │
│  → finish                       │
└──────────┬──────────────────────┘
           ↓
      QA 答案 + 评估 (F1, BLEU-1)
```

### 记忆分层设计（同 Mem-T）

| 层级 | 内容 | 说明 |
|------|------|------|
| Facts | 具体可验证的事实陈述 | "Caroline 于 7 May 2023 参加了 LGBTQ 支持小组" |
| Experiences | 可复用的经验/规律 | 较少触发，对话偏生活日常 |
| Personas | 人物完整档案 | 每个说话人一条，持续更新 |
| Summary | 会话摘要 | 每 session 一条，随 batch 累积 |
| Turns | 原始对话批次 | 111 个 batch，用于精确语义检索 |

---

## 二、Memory Build 结果

**样本**：conv-26（444 轮对话，分 19 个 session，共 111 个 4-turn batch）

| 集合 | 条目数 | 说明 |
|------|--------|------|
| `conv-26_facts` | **181** | 事实记忆 |
| `conv-26_experiences` | **2** | 经验记忆（日常对话较少触发） |
| `conv-26_personas` | **2** | Caroline / Melanie 人物档案 |
| `conv-26_summary` | **19** | 每 session 一条摘要 |
| `conv-26_turns` | **111** | 原始对话 batch |

**Memory Update 操作分布**：
- ADD：183 次（98%）——大部分事实是新增的
- UPDATE：4 次（2%）——少量事实被更新合并

---

## 三、QA 检索结果（157 题，全量）

### 整体指标

| 指标 | 值 |
|------|-----|
| Avg F1 | **0.422** |
| Avg BLEU-1 | **0.359** |
| 完全匹配（F1=1.0） | 28 条（18%） |
| 高匹配（F1≥0.5） | 36 条（23%） |
| 部分匹配（0<F1<0.5） | 56 条（36%） |
| 完全不匹配（F1=0） | 37 条（24%） |

### 按 Category 分类

Locomo 的 category 定义：
- **Cat 1**：单跳事实性问题
- **Cat 2**：时间推理（相对/绝对时间）
- **Cat 3**：开放性推断问题
- **Cat 4**：多跳推理问题

| Category | 题数 | Avg F1 | Avg BLEU-1 |
|----------|------|--------|------------|
| Cat 1（单跳事实） | 34 | 0.294 | 0.239 |
| Cat 2（时间推理） | 39 | **0.530** | **0.488** |
| Cat 3（开放推断） | 14 | 0.282 | 0.189 |
| Cat 4（多跳推理） | 70 | 0.452 | 0.380 |

**Cat 2 表现最好**：时间信息在 memory build 阶段被显式提取并存储（`start_time`/`end_time`），检索时直接命中。  
**Cat 1 最差**：单跳事实问题反而低，原因是实体粒度不够细（如国家名被记为"her home country"）。

### 工具调用分布

| 工具 | 调用次数 | 占比 |
|------|----------|------|
| `search_turns` | 75 | 18.8% |
| `search_facts` | 74 | 18.6% |
| `finish` | 150 | 37.6% |
| `search_personas` | 54 | 13.5% |
| `search_experiences` | 26 | 6.5% |
| `search_summary` | 21 | 5.3% |
| `forced_answer` | 5 | 1.3% |

**多数问题 2-3 步内解决**（2步：87题，3步：48题），检索效率较高。  
`search_turns` 与 `search_facts` 使用频率相当，说明原始对话检索与结构化事实检索互补。

### 异常情况

| 类型 | 数量 | 说明 |
|------|------|------|
| 无 tool call（空轨迹） | 2 | LLM 未生成 `<tool_call>` 格式，直接输出答案 |
| 达到 max_steps（6步）| 5 | LLM 连续多步不生成 tool call，触发 forced_answer |

`no tool call` 问题集中在特定问题上（LLM 认为不需要检索直接回答），实际答案质量待 LLM judge 确认。

---

## 四、典型案例分析

### ✅ 成功案例

```
Q: When did Caroline go to the LGBTQ support group?
Gold: 7 May 2023
Pred: 7 May, 2023  (F1=1.0)
路径: search_facts → finish
```

```
Q: What is Caroline's relationship status?
Gold: Single
Pred: Single  (F1=1.0)
路径: search_personas → search_facts → finish
```

```
Q: When did Caroline give a speech at a school?
Gold: The week before 9 June 2023
Pred: the week before 9 June, 2023  (F1=1.0)
路径: search_turns → finish（原文直接命中）
```

### ❌ 失败案例及原因

**原因 1：时间表达形式不一致（F1=0）**
```
Q: When did Melanie paint a sunrise?
Gold: 2022
Pred: last year   ← 记忆中存的是相对时间，未转换为绝对年份
```

**原因 2：检索到相关但不精确的事实（F1=0）**
```
Q: Where did Caroline move from 4 years ago?
Gold: Sweden
Pred: her home country   ← 事实提取时未捕获具体国家名
```

**原因 3：答案冗长，关键词被稀释（F1偏低）**
```
Q: What is Caroline's field?
Gold: Psychology, counseling certification
Pred: Social work, psychology, gender studies, education.  ← 含多余信息
```

**原因 4：finish 被调用但 answer 为空（已修复）**
```
6 条记录 pred 为空，原因是 LLM 调用 finish tool 时未填写 answer 字段
→ 已在 memory_retrieval.py 中添加 fallback 逻辑
```

---

## 五、问题诊断与优化方向

| 问题 | 严重度 | 优化方向 |
|------|--------|----------|
| 时间相对/绝对转换错误 | 中 | Memory build 时强制在 formation prompt 中要求绝对时间 |
| 事实粒度不够细（如国家名丢失） | 中 | 增大 facts topk，或加 search_turns 兜底 |
| 答案冗长降低 F1 | 中 | 在 answer format prompt 中加强"短语"约束 |
| finish 无 answer（已修复） | 低 | 已修复：空 answer 时 fallback forced_answer |
| experiences 几乎为空（2条） | 低 | 日常对话本身较少触发，正常 |
| Cat 1 F1 低于 Cat 2 | 中 | 单跳事实需要更精准的实体提取 |

---

## 六、下一步

1. **运行 LLM-as-Judge**（获取精确 Accuracy）：
   ```bash
   python scripts/run_qa.py --config configs/build_memory.yaml --limit 1
   ```
2. **修复 no tool call 问题**：在 system prompt 中强化格式约束，或对连续无 tool call 情况提前截断
3. **优化 formation prompt**：要求提取绝对时间（解决 Cat 1 相对时间问题）；约束答案为短语（解决冗长问题）
4. **扩展到全部 10 个 sample**：build 剩余 9 个 sample 的 memory，全量评估
5. **图结构扩展**：将提取的 facts 写入 Graphmemory 的 GraphSnapshotStore，实现图查询+向量检索联合检索

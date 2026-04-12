# GraphMemory QA 评估分析报告

**数据集**：LoCoMo（conv-26，两人长期对话）  
**评估题数**：152 题（cat1=32, cat2=37, cat3=13, cat4=70）  
**图规模**：176 节点，307 条边  
**评估指标**：Token-level F1、BLEU-1（词级别精确匹配，无语义理解）

---

## 整体结果

| 类别 | 含义 | 题数 | Avg F1 | BLEU-1 | F1=1.0 | F1=0 |
|------|------|------|--------|--------|--------|------|
| Cat1 | 单跳事实：直接从对话中找答案 | 32 | 0.384 | 0.312 | 3 题 | 8 题 |
| Cat2 | 时序推理：回答"什么时候" | 37 | 0.550 | 0.463 | 10 题 | 6 题 |
| Cat3 | 开放推断：需要综合推理得出结论 | 13 | 0.263 | 0.162 | 0 题 | 3 题 |
| Cat4 | 多跳推理：连接多个事实 | 70 | 0.523 | 0.451 | 15 题 | 11 题 |
| **总计** | | **152** | **0.478** | **0.400** | | |

---

## Cat1：单跳事实（avg F1=0.384）

单跳题理论上最简单——答案来自对话中的某一句话。但实际 F1 反而是四类中最低的，原因分三类：

### 1a. 图里完全没有这条信息

这是最根本的问题：构建阶段遗漏了某些具体细节。

| 问题 | 标准答案 | 模型输出 | 分析 |
|------|---------|---------|------|
| What is Caroline's relationship status? | `Single` | `Surrounded by people who love and accept her, genuine relationships.` | 关系状态未被提取为属性，模型只找到了情感描述 |
| Where did Caroline move from 4 years ago? | `Sweden` | `Caroline's home country` | 具体国家名未存入节点 attrs，只存了模糊概念 |
| How many children does Melanie have? | `3` | `Kids, plural` | 数字类事实几乎从未被提取 |
| What did Melanie paint recently? | `sunset` | `nature-inspired art, another nature-inspired art` | "nature-inspired" 太泛，丢失了 "sunset" 这个精确词 |

**根因**：Construction prompt 倾向于提取宏观事件（"Melanie 画画了"），忽略了隐含细节（"画的是日落"、"来自瑞典"）。

---

### 1b. 语义正确但词重叠为零（指标问题）

这类答案实际上是对的，但 Token F1 看不出来。

| 问题 | 标准答案 | 模型输出 | F1 | 实际情况 |
|------|---------|---------|-----|---------|
| How many times has Melanie gone to the beach in 2023? | `2` | `Twice` | **0.00** | 完全正确，数字 vs 英文单词 |
| What is Caroline's identity? | `Transgender woman` | `Trans woman` | **0.50** | 正确，缩写不匹配 |
| What kind of art does Caroline make? | `abstract art` | `Abstract self-expression, LGBTQ advocacy, flowers, nature` | **0.22** | 包含正确信息但太冗长 |

**根因**：Token F1 不理解同义词、数字表达方式差异。需要 LLM judge 才能正确判断。

---

### 1c. 列表不完整（部分检索）

图里有这些信息，但检索时只遍历了部分子图，遗漏了某些节点。

| 问题 | 标准答案 | 模型输出 | 缺失 |
|------|---------|---------|------|
| What are Melanie's pets' names? | `Oliver, Luna, Bailey` | `Luna, Oliver` | Bailey 漏了 |
| What books has Melanie read? | `"Nothing is Impossible", "Charlotte's Web"` | `Charlotte's Web` | 《Nothing is Impossible》漏了 |
| Where has Melanie camped? | `beach, mountains, forest` | `mountains, forest` | beach 漏了 |
| What has Melanie painted? | `Horse, sunset, sunrise` | `lake sunrise, nature-inspired art, horse` | sunset 漏了，多了一个错误项 |

**根因**：这类"列举所有"问题需要聚合 entity 的所有 entity-event 边，但 SelectAnchor 每次只选 1-3 个锚点，无法一次遍历 entity 的全部关联事件。

---

### Cat1 典型成功案例

| 问题 | 标准答案 | 模型输出 | F1 |
|------|---------|---------|-----|
| What did Caroline research? | `Adoption agencies` | `Adoption agencies` | 1.00 |
| What musical artists/bands has Melanie seen? | `Summer Sounds, Matt Patterson` | `Matt Patterson, Summer Sounds` | 1.00 |
| What book did Melanie read from Caroline's suggestion? | `"Becoming Nicole"` | `"Becoming Nicole"` | 1.00 |

成功的共同点：答案是一个特定名词，图里有对应节点，检索路径短且清晰。

---

## Cat2：时序推理（avg F1=0.550）

回答"什么时候"发生的。这是四类中 F1 最高的，得益于图里大量的时序边（before/after/updates）和 Event 节点的 `time` 属性。

### 2a. 相对时间无法转换（系统性问题）

| 问题 | 标准答案 | 模型输出 | F1 | 分析 |
|------|---------|---------|-----|------|
| When did Melanie paint a sunrise? | `2022` | `last year` | **0.00** | 对话里 Melanie 说"last year I painted a sunrise"，图原样存储，检索时拿到 "last year" 无法还原为 2022 |
| When did Caroline and Melanie go to a pride festival together? | `2022` | `Last year` | **0.00** | 同上 |
| When is Caroline's youth center putting on a talent show? | `September 2023` | `next month` | **0.00** | 对话时间是 2023-08，"next month" 指 September，但图没做时间计算 |
| When did Melanie read "nothing is impossible"? | `2022` | `Unknown` | **0.00** | 图完全没有这条记录 |

**根因**：对话发生在特定日期，"last year"/"next month" 是相对于对话时间的表达。图构建时没有将相对时间结合对话日期转为绝对时间。

---

### 2b. 日期格式不一致（轻微 metric 问题）

| 问题 | 标准答案 | 模型输出 | F1 |
|------|---------|---------|-----|
| When did Caroline go to the LGBTQ support group? | `7 May 2023` | `7 May, 2023` | **1.00** |
| When did Melanie get hurt? | `September 2023` | `September, 2023` | **1.00** |

逗号差异在 normalize 后被吸收，F1=1.0，说明 F1 对日期格式有一定容错。

---

### 2c. 日期定位错误（检索找到了错误的事件）

| 问题 | 标准答案 | 模型输出 | F1 | 分析 |
|------|---------|---------|-----|------|
| When did Caroline go to a pride parade during the summer? | `The week before 3 July 2023` | `26 June, 2023, 11 August, 2023` | **0.18** | 找到了两个 pride 相关事件，给了两个日期，都不完全准确 |
| When did Melanie make a plate in pottery class? | `24 August 2023` | `15 July, 2023` | **0.33** | 找到了 pottery class 节点但日期属性错了 |

**根因**：图里有多个相似 Event 节点（多次 pride 活动、多次 pottery），检索时混淆了不同事件的时间戳。

---

### Cat2 典型成功案例

| 问题 | 标准答案 | 模型输出 | F1 |
|------|---------|---------|-----|
| When did Melanie go to the park? | `27 August 2023` | `27 August, 2023` | 1.00 |
| When did Melanie get hurt? | `September 2023` | `September, 2023` | 1.00 |
| When did Melanie buy the figurines? | `21 October 2023` | `21 October, 2023` | 1.00 |

成功条件：Event 节点有精确的 `time` 属性，且该事件在图中唯一，不与同类事件混淆。

---

## Cat3：开放推断（avg F1=0.263）

需要跨多段对话综合信息，推断出一个开放性结论。**这是四类中最弱的，且 F1=0 完全不公平。**

### 3a. 语义正确但 F1=0（指标严重失效）

| 问题 | 标准答案 | 模型输出 | F1 | 实际情况 |
|------|---------|---------|-----|---------|
| What would Caroline's political leaning likely be? | `Liberal` | `Progressive, LGBTQ rights-focused` | **0.00** | 语义完全一致，Liberal ≈ Progressive，但词不重叠 |
| What personality traits might Melanie say Caroline has? | `Thoughtful, authentic, driven` | `Empathy, understanding, courage, creativity, gratitude` | **0.00** | 合理推断，覆盖了部分正确特质，但词不一样 |
| Would Caroline want to move back to her home country soon? | `No; she's in the process of adopting children.` | `likely not` | **0.00** | 方向正确（No），但缺少理由，F1 极低 |

**根因**：Cat3 是推断题，正确答案本身就没有唯一性。标准答案是某种表达方式，模型用另一种等价表达，Token F1 无法识别。**Cat3 的 F1 指标基本失效，必须用 LLM judge 评估。**

---

### 3b. 确实答错的案例

| 问题 | 标准答案 | 模型输出 | 分析 |
|------|---------|---------|------|
| Would Caroline be considered religious? | `Somewhat, but not extremely religious` | `likely not religious` | 对话里有宗教相关内容但不明显，推断方向有偏差 |
| Would Caroline pursue writing as a career option? | `Likely no; though she likes reading, she wants to be a counselor` | `likely not` | 结论正确，理由不完整 |

---

### 3c. 相对成功的案例

| 问题 | 标准答案 | 模型输出 | F1 |
|------|---------|---------|-----|
| Would Melanie be more interested in going to a national park or a theme park? | `National park; she likes the outdoors` | `national park` | **0.57** |
| Would Caroline still want to pursue counseling if she hadn't received support? | `Likely no` | `likely not` | **0.50** |

成功条件：答案只需选择一个选项，词重叠自然高；不需要复杂推断。

---

## Cat4：多跳推理（avg F1=0.523）

需要连接两个以上事实得出答案。**这是题目最多（70题）的类别，F1 接近 cat2，表现超出预期。**

### 4a. 语义正确但表达方式不同

| 问题 | 标准答案 | 模型输出 | F1 | 分析 |
|------|---------|---------|-----|------|
| What does Melanie think about Caroline's decision to adopt? | `she thinks Caroline is doing something amazing and will be an awesome mom` | `Excited, supportive, admiration for kindness, confidence in parenting abilities.` | **0.00** | 内容正确，但标准答案是句子，模型输出是短语列表，词不重叠 |
| What is Melanie's reason for getting into running? | `To de-stress and clear her mind` | `mental health benefits, mood boost` | **0.00** | 语义相同，"de-stress" vs "mental health benefits" 词不匹配 |
| Why are flowers important to Melanie? | `They remind her to appreciate the small moments and were a part of her wedding decor` | `Joy, growth, beauty, reminder of wedding day.` | **0.19** | 有交集（wedding），但大量信息用不同词表达 |

---

### 4b. 跨 session 多跳成功案例

| 问题 | 标准答案 | 模型输出 | F1 |
|------|---------|---------|-----|
| How long has Melanie been creating art? | `7 years` | `7 years` | **1.00** |
| Which song motivates Caroline to be courageous? | `Brave by Sara Bareilles` | `Brave by Sara Bareilles` | **1.00** |
| What did Caroline make for a local church? | `a stained glass window` | `stained glass window` | **1.00** |

这些成功案例需要跨多个 session 的信息，说明图的连通性和多跳遍历是有效的。

---

### 4c. 部分跨 session 聚合失败

| 问题 | 标准答案 | 模型输出 | F1 | 分析 |
|------|---------|---------|-----|------|
| What did Caroline take away from the book "Becoming Nicole"? | `Lessons on self-acceptance and finding support` | `Connectedness, hope for her own path.` | **0.00** | 图里有"Caroline 读了 Becoming Nicole"的节点，但读后感没有被提取为属性 |
| What kind of pot did Mel and her kids make with clay? | `a cup with a dog face on it` | `pots` | **0.00** | 细节（dog face cup）未提取，只有泛化的"pottery class" |

---

## 综合问题诊断

### 问题一：信息提取粒度不足（影响 Cat1、Cat4）

构建阶段提取的是宏观事件，忽略了具体细节：
- ❌ 未提取：数字（"3 children"、"7 years"）、地名（"Sweden"）、具体物品名（"cup with a dog face"）
- ✅ 已提取：事件发生（"Melanie 上了陶艺课"）、时间戳、大致主题

**改进方向**：Construction prompt 增加"提取所有具体实体属性"的指令，将细节存为 node attrs。

---

### 问题二：相对时间未转绝对时间（影响 Cat2）

对话里的"last year"、"next month"、"yesterday"没有结合对话日期转化：
- 对话日期已知（session metadata 里有 `session_time`）
- 但构建时没有做"last year = 对话年份 - 1"的推算

**改进方向**：构建 prompt 传入当前对话日期，要求 LLM 将相对时间转为绝对日期。

---

### 问题三：指标对 Cat3 完全失效（Cat3 评估无效）

Cat3 的推断题有多种正确表达，Token F1 只认词重叠：
- "Liberal" ≠ "Progressive, LGBTQ rights-focused"（语义相同，F1=0）
- "Thoughtful" ≠ "Empathy, understanding, courage"（语义相似，F1=0）

**改进方向**：Cat3 必须用 LLM judge（GPT 评分），Token F1 不适合评估推断题。

---

### 问题四：列举类问题不完整（影响 Cat1、Cat4）

"列出所有宠物名字"、"列出所有读过的书"需要聚合一个实体的所有关联事件，但检索时只选了 1-3 个锚点，遍历不完整。

**改进方向**：对问题中含有"all"、"what ... has"等列举意图的问题，在检索时扩展到 entity 的全部 entity-event 边。

---

## 模型架构分析

### 当前系统（GraphMemory Agent）流程

```
对话输入
  → GraphTrigger（LLM判断是否触发写图）
  → GraphLocalizer（向量检索找相关子图）
  → GraphConstructor（LLM提取事实写入图）
  → [图存储：176节点，307边]

问题输入
  → GraphLocalizer（找相关子图）
  → SelectAnchor（LLM选1-3个锚点）
  → Jump Loop（最多3跳，遍历邻居）
  → Raw Fallback（不够时搜原始对话）
  → Forced Answer（整合证据生成答案）
```

### 图结构分布

```
节点类型：Entity=21, Event=155
边类型：entity-event=245, event-event=52, entity-entity=10
时序边(before/after/updates)：40条
```

---

## 结论与下一步

| 优先级 | 问题 | 预期收益 | 难度 |
|--------|------|---------|------|
| P0 | 加 LLM judge 评估 Cat3 | Cat3 评估恢复有效性 | 低 |
| P1 | 相对时间 → 绝对时间（构建时） | Cat2 F1 +0.1 估计 | 中 |
| P2 | 提取细节属性（数字、地名、具体物品） | Cat1 F1 +0.05 估计 | 中 |
| P3 | 列举问题全量检索 | Cat1 列表类题 +0.1 估计 | 高 |
| P4 | 全量 build（10 个 sample）+完整评估 | 建立真实基线 | 中 |

---

*生成时间：2026-04-13*  
*评估样本：LoCoMo conv-26（Caroline × Melanie，19个会话，419轮对话）*

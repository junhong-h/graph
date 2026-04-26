# GraphMemory 进展报告

## 一、背景与问题定位

上一份报告（2026-04-13）用 **GPT-4o** 完成了 LoCoMo-10 全量评估（Cat1-4，1540 题），得到初始基线：

| 指标 | GPT-4o 基线 |
|------|------------|
| Cat1-4 Judge-Acc | 80.7% |
| Avg F1 | 0.508 |

该报告的失败案例分析揭示了三类核心问题：

1. **entity-event 边稀疏**：Cat1（跨 Session 聚合）97% 的失败案例与此直接相关
2. **时序信息丢失**：Cat2 时序题 32% 是因图中根本没有时间属性，无从检索
3. **词汇泛化**：LLM 建图时将原文精确表述替换成泛化描述，检索后无法精确匹配

本报告记录从该基线出发做的所有改动及最终结果。

---

## 二、实验扩充与基线建立

### 2.1 换用 Qwen3-4B（DashScope API）

**动机**：后续RL可能需要开源的模型，且为了方便与同类Paper类比。

**改动**：将建图、检索、评测全流程切换至 Qwen3-4B（阿里云 DashScope）。代码层面在 `llm_client.py` 新增 `use_extra_body_thinking` 参数，附加 `extra_body={"enable_thinking": False}`（DashScope 非流式 API 必须关闭 thinking 模式）。

**结果（Qwen3-4B，旧 prompt，全量 10 样本）**：

| 类别 | GPT-4o | Qwen3-4B | 变化 |
|------|--------|---------|------|
| Cat1 跨Session聚合 | 76.8% | 84.4% | +7.6 pts |
| Cat2 时序 | 75.7% | 74.5% | −1.2 pts |
| Cat3 推理 | 68.7% | 67.7% | −1.0 pts |
| Cat4 单事实 | 85.1% | 89.3% | +4.2 pts |
| **Cat1-4 Overall** | **80.7%** | **84.0%** | **+3.3 pts** |
| Avg F1 | 0.508 | 0.335 | −0.173 |

Qwen3-4B 的 Judge-Acc 意外超过 GPT-4o（+3.3 pts）；F1 大幅下降是因为模型输出冗长，pred/gold 长度比约 3.7x。

**发现的图结构问题**（以 conv-30 为例，与 GPT-4o 对比）：

| 指标 | GPT-4o | Qwen3-4B |
|------|--------|---------|
| entity-event 边数 | **118** | **14** |
| 未连到 Entity 的 Event 节点 | 少量 | **88%（67/76）** |
| Entity 最大 degree | 60 | 9 |
| event-event 主要谓词 | before/after/updates | discussed/mentions |

88% 的 Event 节点没有 entity-event 边，意味着从 Entity 出发只能 jump 到 9 个事件，其余 67 个完全不可达——这是 4B 模型遵循复杂 prompt 规则能力不足的直接体现，也是后续 P0+P1 改进的主要攻击目标。

---

### 2.2 新增 Cat5 对抗性问题评测

**动机**：LoCoMo 数据集含 446 道 Cat5 题（问从未发生过的事情，正确行为是拒绝），此前评测完全跳过。

**改动**：
- `dataset.py`：Cat5 题以 `"Not mentioned in the conversation"` 作为 gold answer
- `evaluator.py`：新增专用 judge prompt，评估是否正确拒绝（而非语义匹配）
- `graph_retrieval.py`：新增 R1 规则——"只有在 jump 至少一次后仍无相关证据，才能回答 Not mentioned；不得在第一跳就直接拒绝"

**Cat5 结果**：

| 版本 | 正确拒绝率 | 幻觉率 |
|------|-----------|--------|
| 无 R1 规则 | 31.5% | 68.5% |
| 加 R1 规则后 | **51.9%** | **48.1%** |

R1 规则提升了 20.4 pts，但仍有近一半幻觉。根因在于图中常有话题相关节点（有 Melanie 节点、有 adoption 节点），但缺少问题前提所需的具体事实——模型见到半相关证据反而更易拼凑错误答案。

---

### 2.3 Oracle 基线实验

**动机**：分离"LLM 推理能力"和"图检索能力"的贡献，建立理论上界，量化检索损失。

**做法**：新脚本 `scripts/run_oracle_qa.py`，从数据集的 `evidence` 字段读取标注真实证据，直接喂给 LLM，完全绕过图检索。

**Oracle 结果（全量 10 样本，Cat1-4）**：

| 类别 | Qwen3-4B 系统 | Oracle 上界 | 检索损失 |
|------|-------------|-----------|---------|
| Cat1 跨Session聚合 | 84.4% | 95.7% | −11.3 pts |
| Cat2 时序 | 74.5% | 92.8% | −18.4 pts |
| Cat3 推理/常识 | 67.7% | 88.5% | −20.8 pts |
| Cat4 单Session单事实 | 89.3% | 95.5% | −6.2 pts |
| **Overall** | **84.0%** | **95.1%** | **−11.1 pts** |

关键结论：11.1 pts 的整体差距几乎全部来自检索失败而非推理失败。改善方向明确指向图构建质量，而非 LLM 推理能力。

---

## 三、图构建质量优化（P0+P1）

### 3.1 改动内容

针对 §2.1 诊断出的图结构问题，在 `graph_construction.py` 的构建 prompt 中添加/修改了以下规则：

| 规则 | 内容 | 目标问题 |
|------|------|---------|
| **Rule 2b** | 禁止"容器节点"（如"Jon and Gina chat on Jan 29"），每个 Event 必须是具体事实 | 消除无语义节点 |
| **Rule 3b** | 相对时间解析：结合 session 日期将"next month"转换为绝对日期 | Cat2 时序匹配失败 |
| **Rule 4b** | event-event 谓词限定为 `before/after/updates/inspired`，禁用 `discussed/mentions` | event-event 时序退化 |
| **Rule 8**（修改） | 在有疑问时"优先建节点而非 Skip" | 节点数量过少 |
| **Rule 10** | 词汇保留：情感词、形容词、比喻必须原文复制到 attrs，不得泛化 | F1 因词汇替换下降 |
| **Rule 11** | 带时间锚点的行为必须独立成 Event 节点，不得合并丢失时间 | Cat2 时序信息丢失 |
| **Rule 12**（核心） | 每个 CreateEvent 后 SHOULD 跟一条 entity-event Link，谓词须语义具体（experienced/participated/visited…），禁用 spoke_to/discussed | entity-event 边稀疏 |

### 3.2 图结构变化

新 prompt 下重新建图，entity-event 边数量大幅提升：

| 样本 | 旧节点 | 新节点 | 旧 e-ev 边 | 新 e-ev 边 | Event 已链接% |
|------|--------|--------|-----------|-----------|-------------|
| conv-26 | 96 | 45 | 25 | **100** | 76% |
| conv-30 | 78 | 108 | 14 | **86** | 32% |
| conv-41 | 56 | 125 | 26 | **213** | 66% |
| conv-42 | 116 | 153 | 23 | **90** | 38% |
| conv-43 | 141 | 17 ⚠️→**266** | 104 | **229** | 64% |
| conv-44 | 125 | 92 | 107 | **264** | 68% |
| conv-47 | 138 | 132 | 102 | **134** | 28% |
| conv-48 | 147 | 144 | — | **143** | 39% |
| conv-49 | 127 | 99 | — | **71** | 55% |
| conv-50 | 167 | 144 | — | **96** | 29% |

**注：conv-43 异常**：第一次构建因 Rule 2b（禁止容器节点）与模型理解偏差产生 Skip 爆炸，节点从 141 崩溃至 17。单独重建后恢复至 266 节点。这暴露了 4B 模型在多规则约束下遵循不稳定的问题。

---

### 3.3 QA 评估结果（全量 10 样本，1986 题）

#### 各类别准确率 / F1 / BLEU-1

| 类别 | Qwen3-4B 旧（Acc） | **P0+P1（Acc）** | Acc变化 | P0+P1 F1 | P0+P1 BLEU-1 |
|------|------------------|----------------|---------|----------|--------------|
| Cat1 跨Session聚合 | 84.4% | **90.4%** | +6.0 pts | 0.281 | 0.215 |
| Cat2 时序 | 74.5% | **78.5%** | +4.0 pts | 0.337 | 0.285 |
| Cat3 推理/常识 | 67.7% | **74.0%** | +6.3 pts | 0.151 | 0.117 |
| Cat4 单Session单事实 | 89.3% | **91.2%** | +1.9 pts | 0.473 | 0.416 |
| **Cat1-4 Overall** | **84.0%** | **87.3%** | **+3.3 pts** | **0.390** | **0.333** |
| Cat5 对抗性（拒绝率） | 51.9% | 36.1% | −15.8 pts ⚠️ | — | — |

#### 各样本 Cat1-4 准确率

| 样本 | Qwen3-4B 旧 | P0+P1 新 | 变化 |
|------|-----------|---------|------|
| conv-26 | — | 92.8% | — |
| conv-30 | 84.0% | 90.1% | +6.1 |
| conv-41 | 82.9% | 93.4% | +10.5 |
| conv-42 | 76.4% | 86.4% | +10.0 |
| conv-43 | 84.8% | 92.1% | +7.3 |
| conv-44 | 76.4% | 82.1% | +5.7 |
| conv-47 | 83.3% | 82.0% | −1.3 |
| conv-48 | 77.0% | 84.3% | +7.3 |
| conv-49 | 80.1% | 79.5% | −0.6 |
| conv-50 | 83.5% | 91.1% | +7.6 |

#### 分析

**Cat1 +6.0 pts / Cat3 +6.3 pts（超预期）**：entity-event 覆盖率提升后，更多 Event 节点可从 Entity 出发访问到，跨 Session 聚合路径和多跳推理路径都得到了改善。conv-41（+10.5 pts）和 conv-42（+10.0 pts）受益最明显，对应 e-ev 边分别从 26→213、23→90。

**Cat2 +4.0 pts**：Rule 3b（相对时间解析）+ Rule 11（时态行为独立节点）直接改善了时间属性的保留，使时序检索命中率提升。

**Cat4 +1.9 pts**：单事实题不依赖跨节点聚合，图结构改善对其贡献相对有限。

**Cat5 −15.8 pts**：R1 规则（"先 jump 再拒绝"）与 Cat5 的正确行为存在根本矛盾。R1 防止了 Cat1-4 中的假阴性（没证据就直接说"Not mentioned"），但让系统在找到半相关证据后更容易拼凑错误答案，加剧了 Cat5 幻觉。Cat1-4 和 Cat5 是对立的优化目标，无法用单一检索规则兼顾，需要独立的前提核查机制。

---

## 四、综合结果对比

| 指标 | GPT-4o | Qwen3-4B 旧 | **Qwen3-4B P0+P1** | Oracle 上界 |
|------|--------|------------|-------------------|------------|
| **Cat1 Acc** | 76.8% | 84.4% | **90.4%** | 95.7% |
| **Cat2 Acc** | 75.7% | 74.5% | **78.5%** | 92.8% |
| **Cat3 Acc** | 68.7% | 67.7% | **74.0%** | 88.5% |
| **Cat4 Acc** | 85.1% | 89.3% | **91.2%** | 95.5% |
| **Cat1-4 Overall Acc** | 80.7% | 84.0% | **87.3%** | **95.1%** |
| **Cat1-4 Avg F1** | 0.508 | 0.335 | **0.390** | — |
| **Cat1-4 Avg BLEU-1** | 0.444 | — | **0.333** | — |
| **Cat5 拒绝率** | 未测 | 51.9% | 36.1% | — |
| **检索损失（vs Oracle）** | — | −11.1 pts | **−7.8 pts** | 0 |
| 模型规模 | GPT-4o | Qwen3-4B | Qwen3-4B | Qwen3-4B |
| 题数（Cat1-4） | 1540 | 1540 | 1540 | 1540 |

**纵向总结**：
- Qwen3-4B 旧 prompt 在 Acc 上已超过 GPT-4o，但 F1 差距大（0.335 vs 0.508），说明语义对但表述冗长
- P0+P1 在 Acc 上进一步提升 3.3 pts，F1 也从 0.335 回升至 0.390
- 检索损失从 11.1 pts 收窄至 7.8 pts，改善来自图构建质量提升
- Oracle 95.1% 表明 LLM 推理能力不是瓶颈；剩余 7.8 pts 差距几乎全部来自检索覆盖不足
- Cat5 是独立的挑战，与 Cat1-4 优化存在对立，需要单独设计解决方案

![image-20260420202205814](/Users/junhong/Library/Application Support/typora-user-images/image-20260420202205814.png)

![image-20260420202300131](/Users/junhong/Library/Application Support/typora-user-images/image-20260420202300131.png)

![image-20260420202320565](/Users/junhong/Library/Application Support/typora-user-images/image-20260420202320565.png)

---

## 五、现存问题与下一步

### 现存问题

| 问题 | 严重程度 | 根因 |
|------|---------|------|
| Cat5 幻觉率 64% | 高 | R1 规则与拒绝行为矛盾；半相关证据诱发幻觉 |
| entity-event 覆盖率仍低（均值~45%） | 中 | Rule 12 用 SHOULD 而非 MUST；4B 模型遵循不稳定 |
| F1 偏低（0.390 vs GPT-4o 0.508） | 中 | 模型输出冗长，pred/gold 长度比约 3.7x |
|                                     |          |                                                |

### 下一步优先级

| 优先级 | 方向 | 做法 | 预期收益 |
|--------|------|------|---------|
| P4 | 输出简洁性约束 | answer format 中加入"最短可能短语"限制 | F1 +0.05~0.1 |
| P5 | Cat5 前提核查 | SelectAnchor 后增加一步"前提是否成立"LLM 判断 | Cat5 +15~20 pts |
| P6 | Rule 12 强化 | SHOULD → MUST，增加失败示例 | Cat1 +2~4 pts |
| P7 | 更大模型建图 | 换用 Qwen3-8B/72B 建图，保持 4B 检索 | 图质量提升，成本分析 |

---

## 附录：评测指标计算方法

### 预处理（所有指标共用）

计算前对 pred 和 gold 统一做以下归一化：
1. 转小写
2. 去除标点（替换为空格）
3. 合并多余空格
4. 去除冠词（a / an / the）

### Token-level F1

与 SQuAD 阅读理解评测标准一致，在词袋（bag-of-words）层面计算 pred 和 gold 的重叠。

$$
\text{overlap} = \sum_{t} \min(\text{count}_\text{pred}(t),\ \text{count}_\text{gold}(t))
$$

$$
\text{Precision} = \frac{\text{overlap}}{|\text{pred tokens}|}, \quad
\text{Recall} = \frac{\text{overlap}}{|\text{gold tokens}|}
$$

$$
F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**特点**：对词序不敏感，只看词汇重叠；pred 或 gold 有额外词会同时惩罚 Precision 和 Recall。Qwen3-4B F1 偏低的主因就在于 pred 输出冗长，多出大量非 gold 词汇，导致 Precision 被压低。

### BLEU-1

Unigram BLEU，加入了 Brevity Penalty（BP）以惩罚过短输出。

$$
\text{clipped\_precision} = \frac{\sum_t \min(\text{count}_\text{pred}(t),\ \text{count}_\text{gold}(t))}{|\text{pred tokens}|}
$$

$$
\text{BP} = \begin{cases} 1 & \text{if } |\text{pred}| \geq |\text{gold}| \\ \exp\!\left(1 - \frac{|\text{gold}|}{|\text{pred}|}\right) & \text{otherwise} \end{cases}
$$

$$
\text{BLEU-1} = \text{BP} \times \text{clipped\_precision}
$$

**与 F1 的区别**：
- F1 同时惩罚过长（Precision↓）和过短（Recall↓）
- BLEU-1 只惩罚过短（BP < 1），对过长输出没有额外惩罚（clipped precision 与 F1 的 Precision 计算相同）
- 因此在 pred 普遍偏长的情况下，BLEU-1 ≤ F1（本实验中两者差距约 0.05~0.06）

### LLM-as-a-Judge（Acc）

调用 Qwen3-4B 作为裁判模型，对每道题的 pred 和 gold 输出 `CORRECT` / `WRONG`：

- **Cat1-4**：采用宽松评分标准——"若生成答案触及与标准答案相同的主题则为 CORRECT，忽略细微格式差异"
- **Cat5**：专用 prompt 评估是否正确拒绝——输出含"not mentioned / no evidence / didn't happen"类表述为 CORRECT，给出具体事实性答案为 WRONG

Acc = CORRECT 数 / 总评测题数。F1 和 BLEU-1 是字符串级指标，Acc 是语义级指标，两者衡量不同维度：F1/BLEU 低但 Acc 高，说明答案语义正确但表述冗长。

---

### 实际例子（来自评测结果）

#### ✅ 完全答对（F1=1.0，Judge=CORRECT）

| 问题 | 标准答案 | 系统回答 |
|------|---------|---------|
| Which country was Tim visiting in the second week of November? | UK | UK |
| What was the name of the pet that John had to say goodbye to on 3 June, 2023? | Max | Max |
| Where did Caroline move from 4 years ago? | Sweden | Sweden. |
| How long has Caroline had her current group of friends for? | 4 years | 4 years |
| Has Jolene tried surfing? | no | No. |

---

#### ⚠️ 语义正确但输出过长（Judge=CORRECT，F1 极低）

这是 F1/BLEU 偏低的主要原因：模型答对了但说了很多多余的话。

| 问题 | 标准答案 | 系统回答 | F1 |
|------|---------|---------|-----|
| Do Jon and Gina start businesses out of what they love? | Yes | Yes, Jon and Gina start businesses out of what they love. | 0.167 |
| Did Melanie make the black and white bowl in the photo? | Yes | Yes, Melanie made the black and white bowl in the photo. | 0.200 |
| What is Caroline's relationship status? | Single | Caroline is currently single, focusing on her personal growth and her friends. | 0.182 |
| What did Melanie paint recently? | sunset | A peaceful painting with blue streaks to show tranquility, inspired by the sea. | 0.000 |
| What events has Caroline participated in to help children? | Mentoring program, school speech | Caroline applied to adoption agencies, attended pride festivals, participated in mentoring programs and gave a speech at a school. | 0.114 |

---

#### 🔶 部分答对（Judge=CORRECT，F1 在 0.4–0.8 之间）

| 问题 | 标准答案 | 系统回答 | F1 |
|------|---------|---------|-----|
| What career path has Caroline decided to pursue? | counseling or mental health for Transgender people | Counseling and mental health. | 0.545 |
| What books has Melanie read? | "Nothing is Impossible", "Charlotte's Web" | "Charlotte's Web" and a book recommended by Caroline. | 0.444 |
| When did Melanie run a charity race? | The sunday before 25 May 2023 | 25 May, 2023 | 0.750 |
| What activities does Melanie partake in? | pottery, camping, painting, swimming | Melanie participates in running, reading, playing her violin, camping with her family, roasting marshmallows, swimming, and pottery. | 0.421 |

---

#### ❌ 答案错误（Judge=WRONG，F1=0）

| 问题 | 标准答案 | 系统回答 |
|------|---------|---------|
| How many children does Melanie have? | 3 | 2 |
| Would Caroline be considered religious? | Somewhat, but not extremely religious | No |
| What activity did Caroline used to do with her dad? | Horseback riding | Unknown |
| When did Melanie go to the museum? | 5 July 2023 | Not mentioned in the conversation |
| When did Gina team up with a local artist for cool designs? | February, 2023 | Not mentioned in the conversation |

---

#### 🚫 Cat5：对抗性问题（从未发生的事）

正确行为是拒绝，给出具体答案即为幻觉。

| 问题 | 标准答案 | 系统回答 | 结果 |
|------|---------|---------|------|
| What was grandma's gift to Melanie? | Not mentioned in the conversation | Not mentioned in the conversation | ✅ 正确拒绝 |
| What are Melanie's plans for the summer with respect to adoption? | Not mentioned in the conversation | Melanie's plans for the summer with respect to adoption: None mentioned. | ✅ 正确拒绝 |
| What did Caroline realize after her charity race? | Not mentioned in the conversation | Caroline realized that self-care is really important. | ❌ 幻觉 |
| What is Melanie excited about in her adoption process? | Not mentioned in the conversation | Melanie is excited about the inclusivity and support of the adoption agency Caroline chose. | ❌ 幻觉 |
| Why did Melanie choose the adoption agency? | Not mentioned in the conversation | Because they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me. | ❌ 幻觉 |

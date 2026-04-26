# Graphmemory Report

> 日期：2026-04-13
> 数据集：LoCoMo-10（10个样本，1540题）
> 模型：GPT-4o（构建 + 检索 + 评测）
> 评测指标：Token-level F1、BLEU-1、LLM-as-Judge（Acc）

---

## 1. 整体结果

| 指标 | 数值 |
|------|------|
| 总题数 | 1540 |
| LLM Judge 覆盖 | 1388（conv-26 用旧流程跳过了 judge） |
| **Judge-Acc** | **80.7%** |
| **Avg F1** | **0.508** |
| **BLEU-1** | **0.444** |

---

## 2. 各样本结果

| 样本 | 题数 | 节点数 | 边数 | Judge-Acc | Avg F1 |
|------|------|--------|------|-----------|--------|
| conv-26 | 152 | 176 | 307 | —（无judge） | 0.478 |
| conv-30 | 81 | 78 | 144 | 84.0% | 0.550 |
| conv-41 | 152 | 184 | 346 | 82.9% | 0.571 |
| conv-42 | 199 | 133 | 220 | 76.4% | 0.448 |
| conv-43 | 178 | 196 | 330 | 84.8% | 0.531 |
| conv-44 | 123 | 159 | 234 | 76.4% | 0.462 |
| conv-47 | 150 | 250 | 492 | 83.3% | 0.591 |
| conv-48 | 191 | 147 | 275 | 77.0% | 0.472 |
| conv-49 | 156 | 127 | 220 | 80.1% | 0.529 |
| conv-50 | 158 | 167 | 340 | 83.5% | 0.487 |

**观察**：
- conv-47（250节点/492边）F1最高（0.591），图最大，信息覆盖最全
- conv-42、conv-44、conv-48 准确率偏低（76-77%），与图偏小且信息密集有关

---

## 3. 各类别真实定义与总体结果

### 3.1 各类别证据结构（来自数据集标注）

| 类别 | 题数 | 平均证据条数 | 平均涉及Session数 | 跨Session比例 |
|------|------|------------|-----------------|-------------|
| **Cat1** | 282 | **3.12** | **2.67** | **95%** |
| Cat2 | 321 | 1.17 | 1.10 | 9% |
| Cat3 | 96 | 2.05 | 1.62 | 32% |
| **Cat4** | 841 | **1.06** | **1.00** | **<1%** |

### 3.2 正确定义

| 类别 | 正确定义 | 典型问法 |
|------|---------|---------|
| **Cat4**（841题） | **单Session单事实查找** — 答案来自同一Session的单条对话 | "What is Gina's favorite dance style?" |
| **Cat2**（321题） | **时序问答** — 单条证据，但要求给出精确时间 | "When did Jon lose his job?" |
| **Cat3**（96题） | **推理/常识** — 从对话线索推断隐性结论 | "Does John live near a beach or mountains?" |
| **Cat1**（282题） | **跨Session聚合** — 答案需整合分散在多个Session的信息 | "What activities does Melanie partake in?" |

### 3.3 各类别指标

| 类别 | 题数 | 定义 | Judge-Acc | Avg F1 | BLEU-1 |
|------|------|------|-----------|--------|--------|
| **Cat4** | 841 | 单Session单事实 | **85.1%** | 0.559 | 0.499 |
| Cat2 | 321 | 时序问答 | 75.7% | 0.562 | 0.499 |
| Cat1 | 282 | 跨Session聚合 | 76.8% | 0.375 | 0.297 |
| Cat3 | 96 | 推理/常识 | 68.7% | 0.278 | 0.221 |

**关键发现**：

- **Cat4（单Session单事实）准确率最高**：答案在同一Session就能找到，图一跳即达，系统在此表现强
- **Cat1（跨Session聚合）明显弱于Cat4**：两者差距 8.3 个点，正因为 Cat1 需要跨多个 Session 聚合信息，而 Cat4 不需要
- **Cat1 的 F1（0.375）远低于 Cat4（0.559）**：跨 Session 聚合时，信息不完整导致答案残缺；token-level F1 对列举类答案尤其敏感
- **Cat3 最弱**：推理类问题本质上依赖弱信号综合判断，图结构难以直接支撑

### 3.4 各样本 × 各类别细分

| 样本 | Cat1 跨Session聚合 | Cat2 时序 | Cat3 推理 | Cat4 单Session |
|------|-------------------|----------|----------|---------------|
| conv-26 | 32q/— | 37q/— | 13q/— | 70q/— |
| conv-30 | 11q/82% | 26q/81% | 0q/— | 44q/86% |
| conv-41 | 31q/84% | 27q/70% | 8q/62% | 86q/88% |
| conv-42 | 37q/**65%** | 40q/80% | 11q/**36%** | 111q/83% |
| conv-43 | 31q/74% | 26q/85% | 14q/79% | 107q/**89%** |
| conv-44 | 30q/80% | 24q/**58%** | 7q/43% | 62q/85% |
| conv-47 | 20q/70% | 34q/74% | 13q/77% | 83q/**92%** |
| conv-48 | 21q/71% | 42q/79% | 10q/60% | 118q/79% |
| conv-49 | 37q/81% | 33q/76% | 13q/92% | 73q/79% |
| conv-50 | 32q/84% | 32q/75% | 7q/86% | 87q/86% |

---

## 4. Cat1 — 跨Session信息聚合（282题，76.8% Acc）

### 问题特征

Cat1 要求把散落在不同 Session（不同时间点对话）中的信息整合成完整答案。**95% 的题目涉及多个 Session，平均需要 3.12 条分散证据**。这是对长期记忆系统聚合能力的直接考验。

典型题型：
- 人物整体活动列举（"Melanie 参与了哪些活动？" → 答案横跨 4 个 Session）
- 两人共同点（"Jon 和 Gina 有什么共同点？" → 证据来自不同 Session 的不同事件）
- 成长/演变过程（"Gina 是如何推广她的服装店的？" → 横跨 4 个 Session 的 4 条记录）

### 失败模式分类

数据确认：Cat1 的 58 个失败案例中，**56 个（97%）涉及跨 Session**，说明失败根本原因是跨 Session 聚合不完整，而非单点检索错误。

| 模式 | 数量 | 占比 |
|------|------|------|
| A：聚合不完整——只检索到部分 Session 的信息 | 31 | 53% |
| B：检索到了错误 Session 的相似信息 | 16 | 28% |
| C：精确词汇损失（内容对但表述被泛化） | 8 | 14% |
| D：单证据检索失败 | 3 | 5% |

### 案例分析

#### ✗ 模式A：聚合不完整

**案例 1.A.1 — 只取到了部分维度**
```
样本：conv-30
问题：What do Jon and Gina both have in common?
标准答案：They lost their jobs and decided to start their own businesses.
模型回答：Dance, passion for dancing, stress relief through dance, lifeline=dance.
证据：4条，来自2个Session
F1=0.00

原因分析：
  "两人共同点"有两个维度：①共同的舞蹈爱好 ②共同的失业→创业经历。
  图中舞蹈相关节点密度远高于"失业"事件节点，
  Localizer 的向量搜索把检索入口锚定在舞蹈相关节点上，
  ReAct 在 3 跳内只遍历了舞蹈子图，未连接到失业事件。
  标准答案期望的是跨维度的共性总结，系统只给出了单一维度。
```

**案例 1.A.2 — 列举缺项**
```
样本：conv-41
问题：What activities has Maria done with her church friends?
标准答案：Hiking, picnic, volunteer work
模型回答：Picnic, charades, scavenger hunt.
证据：3条，来自3个Session
F1=0.25

原因分析：
  "picnic"命中了，但"hiking"和"volunteer work"分别记录在
  不同 Session 的 Event 节点中，且与"church friends"的
  关联边强度不足以在 3 跳内被检索到。
  额外混入了不相关活动（charades, scavenger hunt），
  说明检索路径在聚合过程中引入了噪声。
```

**案例 1.A.3 — 地理信息仅捕获部分**
```
样本：conv-41
问题：What European countries has Maria been to?
标准答案：Spain, England
模型回答：Spain
证据：2条，来自2个Session
F1=0.67

原因分析：
  "Spain"在某个 Session 中作为主要话题被明确存储，
  "England"在另一个 Session 中出现时仅是边缘提及，
  图节点未为其建立独立的地名 Entity，导致聚合时漏掉。
```

#### ✗ 模式B：检索到错误Session的相似信息

**案例 1.B.1 — 相似事件混淆**
```
样本：conv-41
问题：Who did John go to yoga with?
标准答案：Rob
模型回答：A colleague
证据：2条，来自2个Session
F1=0.00

原因分析：
  图中 Event 节点记录的是"John went to yoga with a colleague"，
  "colleague"是对话中的泛指，实际人名"Rob"出现在不同 Session 的另一条对话中。
  两者在图中未被连接为同一事件，系统取到了泛指版本，漏掉了精确版本。
```

**案例 1.B.2 — 时间戳相近的事件被混淆**
```
样本：conv-30
问题：How long did it take for Jon to open his studio?
标准答案：six months
模型回答：5 months
证据：2条，来自2个Session
F1=0.50

原因分析：
  "开业时间"分散在两个 Session 中被提及，
  一处说"大约半年"，另一处数字被LLM在摘要时近似为"5个月"，
  聚合时取了错误版本。
```

#### ✓ 成功案例

**案例 1.OK.1 — 多Session聚合成功**
```
样本：conv-30
问题：Which events has Jon participated in to promote his business venture?
标准答案：fair, networking events, dance competition
模型回答：Jon attends networking events, Jon showcases his dance studio at a fair,
         Jon hosts a dance competition, ...
证据：3条，来自3个Session
F1=0.34 → judge=CORRECT

分析：
  三个活动分别在 Session 8/10/16 中被记录为独立 Event 节点，
  且都挂在 Jon 的 participated 边上。
  ReAct 从 Jon 节点出发，三跳内逐一遍历到这些 Event，
  聚合成功。Judge 判为 CORRECT 说明内容语义对，
  但 F1=0.34 揭示了答案表述的冗余（多了额外句子）。
```

**案例 1.OK.2 — 精确枚举成功**
```
样本：conv-30
问题：Which cities has Jon visited?
标准答案：Paris, Rome
模型回答：Paris, Rome
证据：2条，来自2个Session
F1=1.00

分析：
  城市名在图中被存储为独立 Entity 节点，
  均与 Jon 的 visited 边直连，
  向量搜索能精确命中，两跳内枚举完整。
```

### Cat1 核心问题

1. **跨Session检索覆盖不足**：Localizer 的向量搜索基于单次查询，无法保证覆盖所有相关 Session 的入口节点；ReAct 的 max_hop=3 限制了能遍历的节点范围
2. **信息密度不均**：主线事件节点密集、边多，次要事件边少，聚合时系统优先走密集路径，导致稀疏节点被忽略
3. **泛指与具名未关联**："同事/朋友"等泛指与具体人名分散在不同 Session，图中未建立等价连接
4. **聚合过程引入噪声**：多跳路径上的中间节点可能引入不相关信息，污染最终答案

---

## 5. Cat2 — 时序问答（321题，75.7% Acc）

### 问题特征

时序题要求给出具体时间或时序关系（"When did X happen?"）。**91% 为单Session单证据**，本质上是单点检索，难点在于时间信息是否被精确捕获进图。F1（0.562）高但 Judge-Acc（75.7%）偏低，说明答案语义接近但时间格式/精度不匹配。

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：图中无时间信息（输出"Unknown"） | 22 | 32% |
| B：检索到错误时间点 | 29 | 42% |
| C：时间精度偏差（月份正确但日期偏） | 18 | 26% |

### 案例分析

#### ✗ 模式A：时间信息未被图捕获

**案例 2.A.1 — 侧面活动时间丢失**
```
样本：conv-30
问题：When did Jon start to go to the gym?
标准答案：March, 2023
模型回答：Unknown

原因分析：
  "开始去健身房"是对话的侧面信息，重要性低于主线（开舞蹈工作室）。
  GraphTrigger 或将该批次判为 SKIP，或 Construction 时
  该细节被合并进更宏观的 Event 节点而丢失了时间属性。
```

**案例 2.A.2 — 次要行为的时间被忽略**
```
样本：conv-30
问题：When did Jon start reading "The Lean Startup"?
标准答案：May, 2023
模型回答：unknown

原因分析：
  "读书"属于对话中的一句带过的行为，
  未被单独建立 Event 节点，也未在其他节点的属性中保留时间戳。
```

#### ✗ 模式B：检索到错误时间点

**案例 2.B.1 — 同期多事件混淆**
```
样本：conv-41
问题：When did John take a road trip to the Pacific Northwest?
标准答案：2022
模型回答：April, 2023

原因分析：
  John 在 2023 年也有旅行记录，
  ReAct 跳跃时落到了近期旅行节点（time=2023），
  而非更早的太平洋西北旅行（time=2022）。
  时间相近的多个 Event 在向量空间中位置接近，容易混淆。
```

**案例 2.B.2 — 事件内容错配**
```
样本：conv-41
问题：What did John attend with his colleagues in March 2023?
标准答案：a tech-for-good convention
模型回答：John graduated

原因分析：
  March 2023 时间段内有多个 Event 节点，
  "毕业"事件的 time 属性也标注了 March 2023，
  向量搜索时误命中了毕业事件而非 tech convention。
```

#### ✗ 模式C：时间精度偏差

**案例 2.C.1 — 日期被邻近事件污染**
```
样本：conv-30
问题：When did Jon start learning marketing and analytics tools?
标准答案：July, 2023
模型回答：16 June, 2023
F1=0.40

原因分析：
  图中有一个 Event 节点的 time 属性为"June 16"，
  该事件与 marketing 工具相关但不完全一致，
  被检索系统当成了答案。
```

**案例 2.C.2 — 相对时间关系无法表达**
```
样本：conv-41
问题：When did John have a party with veterans?
标准答案：The Friday before 20 May 2023
模型回答：20 May, 2023
F1=0.75

原因分析：
  图中只存储了绝对时间戳"May 20"，
  "之前的周五"这种相对时间关系无法在当前图结构中表示。
```

#### ✓ 成功案例

**案例 2.OK.1 — 核心事件时间精确**
```
样本：conv-30
问题：When Jon has lost his job as a banker?
标准答案：19 January, 2023
模型回答：19 January, 2023
F1=1.00

分析：对话开篇的核心事件，精确日期在 Event 节点的 time 属性中完整保留。
```

**案例 2.OK.2 — 月份级精度匹配**
```
样本：conv-30
问题：When Gina has lost her job at Door Dash?
标准答案：January, 2023
模型回答：January 2023
F1=1.00

分析：月份级精度要求低，图中 time 属性格式稍有差异也能通过 judge。
```

### Cat2 核心问题

1. **次要行为时间未捕获**：非核心事件的时间在构建时被忽略，导致 32% 输出"Unknown"
2. **同期事件竞争**：相同时间段多个 Event 节点在向量空间接近，ReAct 易跳错
3. **相对时间关系无法建模**：图只存储绝对时间戳，"X之前的周五"等表达无法直接回答

---

## 6. Cat3 — 推理/常识问答（96题，68.7% Acc）

### 问题特征

Cat3 不要求直接检索事实，而是从对话线索**推断隐性结论**。**32% 涉及多 Session**，平均需要 2.05 条证据，但核心难点不在于检索量，而在于推理链的完整性。这是最弱的类别（68.7%），失败几乎全是 F1=0 的方向性错误。

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：推断方向相反（结论与线索矛盾） | 10 | 38% |
| B：推断过于笼统或缺关键一步 | 9 | 35% |
| C：所需外部知识图中不存在 | 7 | 27% |

### 案例分析

#### ✗ 模式A：推断方向相反

**案例 3.A.1 — 弱信号权衡失败**
```
样本：conv-41
问题：Does John live close to a beach or the mountains?
标准答案：beach
模型回答：likely mountains

原因分析：
  对话中存在两类线索：登山/远足（→ mountains）、海鲜/海边（→ beach）。
  Localizer 返回的子图偏向了运动/户外节点，
  ReAct 在有限跳数内只聚合到了 mountains 侧的证据。
  正确答案需要全局权衡，而当前系统只能看局部子图。
```

**案例 3.A.2 — 关键节点未连接**
```
样本：conv-41
问题：Would John be open to moving to another country?
标准答案：No，他有参军和竞选公职等明确的美国国内目标。
模型回答：likely open to moving to another country

原因分析：
  需要将"参军""竞选公职"两个 Event 节点与"不愿离美"的推论连接。
  这两个节点在图中存在，但它们的推理含义（"强美国国内绑定"）
  未被建模为显性属性，ReAct 无法在 3 跳内完成这一推理链。
```

#### ✗ 模式B：推断缺关键一步

**案例 3.B.1 — 二步推理链断裂**
```
样本：conv-42
问题：What pets wouldn't cause any discomfort to Joanna?
标准答案：Hairless cats or pigs（无毛动物，因为Joanna对毛发过敏）
模型回答：turtles

原因分析：
  图中存有"Joanna 对毛发过敏"，但推理需要额外一步：
  "对毛发过敏 → 适合无毛动物 → 举例为无毛猫或猪"。
  系统跳过了"无毛动物"这一中间推理步骤，
  自行联想到了"爬行类=无毛"这条错误路径。
```

**案例 3.B.2 — 需要外部知识**
```
样本：conv-42
问题：What Console does Nate own?
标准答案：Nintendo Switch（因为 Xenoblade 2 是 Switch 独占）
模型回答：Likely PC

原因分析：
  图中存有"Nate 喜欢 Xenoblade 2"，但"Xenoblade 2 是 Nintendo Switch 独占"
  是外部游戏知识，图中不存在。
  这类推理题本质上要求 LLM 用世界知识补全推理链，
  超出了当前图内容的覆盖范围。
```

#### ✓ 成功案例

**案例 3.OK.1 — judge 弥补了 F1 的不足**
```
样本：conv-41
问题：Would John be considered a patriotic person?
标准答案：Yes
模型回答：Passion for veterans' rights, desire to serve his country, support for military veterans.
F1=0.00（答案不含"Yes"），但 judge=CORRECT

分析：
  模型输出的是支撑证据而非直接结论，语义正确。
  说明 LLM judge 在 Cat3 中比 token F1 更能准确衡量表现。
```

### Cat3 核心问题

1. **局部子图无法覆盖全局弱信号**：推理题需要权衡多个弱信号，当前子图选择只返回局部视图
2. **隐性推理链未建模**：图只存储显性事实，"参军→强美国绑定→不愿离美"这类推理链不存在
3. **外部知识缺失**：部分推理依赖图外常识（游戏主机知识、无毛动物常识等），当前系统无法补全
4. **judge 与 F1 的分离**：Cat3 中 judge=CORRECT 而 F1=0 的情况较多，F1 低估了系统的实际推理能力

---

## 7. Cat4 — 单Session单事实查找（841题，85.1% Acc）

### 问题特征

Cat4 是题量最大（841题）、准确率最高（85.1%）的类别。**99% 以上的题目只需要单Session的单条对话证据**，平均证据数 1.06。这是最"直接"的检索场景：答案明确存在于某次对话中，系统只需找到那一条。

典型题型：
- 人物偏好/属性（"Gina 最喜欢的舞蹈风格？" → 单条证据）
- 单次事件细节（"慈善赛跑为什么而举办？" → 单条证据）
- 直接引语内容（"Jon 的舞蹈让他感到什么？" → 单条证据）

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：精确词汇丢失——答案被泛化改写（F1<0.1） | 81 | 70% |
| B：回答"Unknown"（信息未被图捕获） | 13 | 11% |
| C：图片/照片相关内容 | 7 | 6% |
| D：部分重叠 | 4 | 3% |

### 案例分析

#### ✗ 模式A：精确词汇在构建时被泛化

这是 Cat4 最主要的失败模式。答案在对话中只出现一次，且是非常具体的一个词，但图构建的摘要过程将其改写为近义描述。

**案例 4.A.1 — 情感词被替换**
```
样本：conv-30
问题：What does Jon's dance make him?
标准答案：happy
模型回答：second nature, living his dream
F1=0.00

对话原文：Jon 说"dance makes me happy"
图中存储："Jon 的舞蹈让他感到充实/梦想成真"
原因分析：LLM 在构建 Event 节点时用了语义更丰富的描述，
         将"happy"这一精确词汇扩展成了更复杂的表述。
         单词"happy"在 token-level 上完全消失。
```

**案例 4.A.2 — 形容词被替换**
```
样本：conv-30
问题：How does Gina describe the feeling that dance brings?
标准答案：magical
模型回答：Stress relief, lifeline.
F1=0.00

原因分析：
  "magical"是对话中 Gina 使用的一个具体形容词，
  图构建时被归纳为"解压/生命线"这类功能性描述，
  原始词汇完全丢失。
```

**案例 4.A.3 — 比喻被替换为字面描述**
```
样本：conv-30
问题：What did Jon and Gina compare their entrepreneurial journeys to?
标准答案：dancing together and supporting each other
模型回答：Scary and rewarding, roller coaster.
F1=0.18

原因分析：
  对话中同时出现了情感描述（scary/rewarding）和比喻（dancing together）。
  LLM 摘要时选择了情感描述，丢掉了比喻。
  检索时问题问的是"比喻"，系统返回了"情感描述"。
```

#### ✗ 模式C：图片/照片相关内容

**案例 4.C.1 — 照片描述未精确存储**
```
样本：conv-30
问题：What does Gina say about the dancers in the photo?
标准答案：They look graceful
模型回答："brings back lots of memories"

原因分析：
  照片相关对话包含两类信息：①对舞者外观的评价（graceful）
  ②对整体情感的表达（memories/happy place）。
  图构建时倾向于捕获情感层面，忽略了外观评价。
  且"graceful"这个精确词汇与模式A一样面临被替换的风险。
```

#### ✓ 成功案例

**案例 4.OK.1 — 专有名词精确检索**
```
样本：conv-30
问题：What is Gina's favorite style of dance?
标准答案：Contemporary
模型回答：Contemporary
F1=1.00

分析：
  "Contemporary"是一个不可被近义词替代的专有名词，
  图构建时被原样存储为 attrs.favorite_dance_style="Contemporary"，
  Localizer 向量搜索精确命中 Gina 节点，一跳取得答案。
```

**案例 4.OK.2 — 专有名词（品牌/材料）精确检索**
```
样本：conv-30
问题：What kind of flooring is Jon looking for in his dance studio?
标准答案：Marley flooring
模型回答：Marley flooring
F1=1.00

分析：
  "Marley flooring"是不可泛化的专有名词，
  在图中被原样保留，检索无损。
```

**小结——为什么 Cat4 成功率高**：
单事实检索只需"找到 + 不改写"，当信息是专有名词（人名、地名、材料名）时系统表现很好。失败主要来自构建时的词汇泛化，而非检索路径问题。

### Cat4 核心问题

1. **构建时词汇精度损失**：LLM 摘要倾向于归纳、改写，而非保留原始精确词汇（"happy"→"fulfilling"，"magical"→"stress relief"）
2. **照片/媒体内容处理薄弱**：照片讨论的细节评价没有专门的存储机制
3. **情感与事实描述竞争**：同一事件的多个描述维度并存，检索时可能选中非期望维度

---

## 8. 跨类别共同问题

### 8.1 词汇精度损失（Cat4 主因，Cat1 次因）

图构建时 LLM 将原始对话改写为摘要，过程中精确词汇（形容词、比喻、专有名词）被近义泛化替代。对 Cat4 影响最大（70% 的失败来源于此）。

**修复方向**：在 Construction 提示中增加"保留对话中的原始关键词"规则；对形容词、比喻单独建立 Quote 节点。

### 8.2 跨Session覆盖不足（Cat1 主因）

Localizer 的向量搜索是基于单次查询的局部检索，无法保证覆盖答案所需的所有 Session 入口。ReAct 的 max_hop=3 进一步限制了跨Session的遍历范围。

**修复方向**：对 Cat1 类型的聚合问题，可引入多轮 Localizer 调用（每个已发现实体再做一次搜索）；或提高 max_hop 至 5。

### 8.3 时间信息捕获不足（Cat2 主因）

次要行为的时间戳（"何时开始读某本书"）在构建时被忽略，导致 32% 的时序题输出"Unknown"。

**修复方向**：降低 GraphTrigger 触发阈值；在 Construction 提示中明确要求每个行为事件必须记录时间属性。

### 8.4 隐性推理链缺失（Cat3 主因）

图只存储显性事实，推理类问题所需的间接推断需要在检索时由 LLM 完成，但 3 跳内无法聚合全部弱信号。

**修复方向**：提高 max_hop 至 5；或在构建阶段显式建立推理边（如"参军 → implies → 强烈美国绑定"）。

---

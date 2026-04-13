# GraphMemory Agent — 全量评测分析报告

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
- conv-47（250节点/492边）F1最高（0.591），图最大，覆盖最全
- conv-42、conv-44、conv-48 准确率偏低（76-77%），与图偏小且信息密集有关
- 图节点数与准确率有一定正相关，但不是绝对

---

## 3. 各类别总体结果

| 类别 | 题数 | 描述 | Judge-Acc | Avg F1 | BLEU-1 |
|------|------|------|-----------|--------|--------|
| **Cat1** | 282 | 单跳事实问答 | 76.8% | 0.375 | 0.297 |
| **Cat2** | 321 | 时序问答 | 75.7% | 0.562 | 0.499 |
| **Cat3** | 96 | 推理/常识 | 68.7% | 0.278 | 0.221 |
| **Cat4** | 841 | 多跳/细节 | **85.1%** | 0.559 | 0.499 |

**关键发现**：
- Cat4（多跳）准确率最高：图的多跳边结构有效支撑跨轮次信息链接
- Cat3（推理）最弱：需要从多个事实推导隐性结论，图存储的是显性事实，难以直接支撑
- Cat2 F1高但 Judge-Acc 偏低：说明答案语义相近但表述不够精确（如 "March 2023" vs "March, 2023"）
- Cat1 F1低于 Cat4：Cat1 虽是单跳，但需要精确匹配实体名，不能泛化

### 各样本 × 各类别细分

| 样本 | Cat1 (n/acc) | Cat2 (n/acc) | Cat3 (n/acc) | Cat4 (n/acc) |
|------|-------------|-------------|-------------|-------------|
| conv-26 | 32q/— | 37q/— | 13q/— | 70q/— |
| conv-30 | 11q/**82%** | 26q/81% | 0q/— | 44q/86% |
| conv-41 | 31q/84% | 27q/70% | 8q/62% | 86q/88% |
| conv-42 | 37q/65% | 40q/80% | 11q/**36%** | 111q/83% |
| conv-43 | 31q/74% | 26q/**85%** | 14q/79% | 107q/**89%** |
| conv-44 | 30q/80% | 24q/58% | 7q/43% | 62q/85% |
| conv-47 | 20q/70% | 34q/74% | 13q/77% | 83q/**92%** |
| conv-48 | 21q/71% | 42q/79% | 10q/60% | 118q/79% |
| conv-49 | 37q/81% | 33q/76% | 13q/**92%** | 73q/79% |
| conv-50 | 32q/84% | 32q/75% | 7q/86% | 87q/86% |

---

## 4. Cat1 — 单跳事实问答（282题，76.8% Acc）

### 问题特征

单跳事实题要求直接查找图中某一节点或边上存储的属性，如人物关系、所做事件、偏好等。失败集中在**实体精度**和**不完整列举**两类。

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：完全检索到错误实体/信息（F1<0.05） | 42 | 72% |
| B：部分正确/列举不完整（F1 0.1–0.6） | 11 | 19% |
| C：高F1但judge判错（精度/范围问题） | 3 | 5% |
| D：回答"Unknown" | 5 | 9% |

### 案例分析

#### ✗ 模式A：检索到了错误信息

**案例 1.A.1 — 混淆相似属性**
```
样本：conv-30
问题：What do Jon and Gina both have in common?
标准答案：They lost their jobs and decided to start their own businesses.
模型回答：Dance, passion for dancing, stress relief through dance, lifeline=dance.
F1=0.00
原因分析：图中存储了大量关于两人舞蹈爱好的边，检索锚点落在这些节点上，
         而"失业→创业"这条共性链接在图中表现为两个独立事件节点，
         未被显式标记为"共同点"，导致ReAct跳跃到错误子图。
```

**案例 1.A.2 — 实体名过于宽泛**
```
样本：conv-41
问题：Who did John go to yoga with?
标准答案：Rob
模型回答：A colleague
F1=0.00
原因分析：图中事件节点可能记录了"John went to yoga with a colleague"，
         但没有将"colleague"解析为具体人名"Rob"。
         构建阶段别名/指代消解不足。
```

**案例 1.A.3 — 地名遗漏**
```
样本：conv-41
问题：What area was hit by a flood?
标准答案：West County
模型回答：John's old area
F1=0.00
原因分析：检索到了相关事件（洪水），但节点属性中未精确存储地名"West County"，
         模型用了更泛化的描述"John's old area"填充。
```

#### ✗ 模式B：部分正确 / 列举不完整

**案例 1.B.1 — 数值偏差**
```
样本：conv-30
问题：How long did it take for Jon to open his studio?
标准答案：six months
模型回答：5 months
F1=0.50
原因分析：数字记录在图中时可能被近似，"6个月"与"5个月"相差一月，
         可能是图节点时间属性精度不足。
```

**案例 1.B.2 — 枚举缺失部分项**
```
样本：conv-41
问题：What activities has Maria done with her church friends?
标准答案：Hiking, picnic, volunteer work
模型回答：Picnic, charades, scavenger hunt.
F1=0.25
原因分析：子图只检索到了"picnic"相关节点，
         "hiking"和"volunteer work"在图中可能分散在不同session节点，
         未被一次跳跃覆盖。另外混入了不相关活动（charades, scavenger hunt）。
```

#### ✓ 成功案例

**案例 1.OK.1 — 精确实体匹配**
```
样本：conv-30
问题：Which cities has Jon visited?
标准答案：Paris, Rome
模型回答：Paris, Rome
F1=1.00
分析：城市名作为Entity节点的canonical_name存储，
      通过"Jon → visited → Paris/Rome"边可直接检索。
```

**案例 1.OK.2 — 技能枚举**
```
样本：conv-41
问题：What martial arts has John done?
标准答案：Kickboxing, Taekwondo
模型回答：taekwondo, kickboxing
F1=1.00
分析：两个运动作为单独Event/Entity节点，与John有明确边，
      ReAct一跳即可枚举完整。
```

### Cat1 核心问题

1. **别名/指代解析不足**：图构建时未将"a colleague"解析为具体人名Rob
2. **分散信息聚合困难**：列举类问题需要从多个节点收集，ReAct跳跃次数有限（max_hop=3）
3. **细节粒度损失**：地名、精确数字在构建摘要时被泛化

---

## 5. Cat2 — 时序问答（321题，75.7% Acc）

### 问题特征

时序题要求给出具体时间或时序关系（"When did X happen?"、"How long after X did Y?"）。F1（0.562）明显高于Cat1（0.375），但Judge-Acc相对偏低，说明答案语义接近但表述格式不精确。

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：图中无时间信息（输出"Unknown"） | 22 | 32% |
| B：检索到错误时间点 | 29 | 42% |
| C：时间精度偏差（月份正确但日期偏） | 18 | 26% |

### 案例分析

#### ✗ 模式A：时间信息未被图捕获

**案例 2.A.1 — 侧面信息丢失**
```
样本：conv-30
问题：When did Jon start to go to the gym?
标准答案：March, 2023
模型回答：Unknown
原因分析："开始去健身房"是对话中一句较轻描述的侧面信息，
         GraphTrigger可能将含该信息的批次判定为SKIP，
         或构建时该事件被归并进更主要事件而时间属性丢失。
```

**案例 2.A.2 — 次要活动被忽略**
```
样本：conv-30
问题：When did Jon start reading "The Lean Startup"?
标准答案：May, 2023
模型回答：unknown
原因分析：书名读书行为属于对话中的次要信息，
         与主线事件（开设舞蹈工作室）相比优先级低，
         在批次摘要时此信息没有被显式存为Event节点。
```

**案例 2.A.3 — 社群活动时间**
```
样本：conv-30
问题：When did Gina go to a dance class with a group of friends?
标准答案：21 July 2023
模型回答：Unknown
原因分析：具体日期"21 July 2023"需要图中Event节点有精确time属性，
         但批次摘要过程可能只记录了活动类型，未保留精确日期。
```

#### ✗ 模式B：检索到错误时间点

**案例 2.B.1 — 时间混淆**
```
样本：conv-41
问题：When did John take a road trip to the Pacific Northwest?
标准答案：2022
模型回答：April, 2023
F1=0.00
原因分析：John在2023年也有旅行记录，ReAct跳跃时选择了错误的Event节点，
         将近期事件的时间戳与太平洋西北旅行混淆。
```

**案例 2.B.2 — 事件内容错配**
```
样本：conv-41
问题：What did John attend with his colleagues in March 2023?
标准答案：a tech-for-good convention
模型回答：John graduated
F1=0.00
原因分析：图中在March 2023附近有多个Event节点，
         检索跳到了"毕业"事件而非"tech convention"，
         说明时间锚点竞争时选择了错误节点。
```

#### ✗ 模式C：时间精度偏差

**案例 2.C.1 — 月份偏移**
```
样本：conv-30
问题：When did Jon start learning marketing and analytics tools?
标准答案：July, 2023
模型回答：16 June, 2023
F1=0.40
原因分析：图中记录了"June 16"某个相关事件的时间，
         但具体"开始学习marketing工具"发生在7月，
         节点时间戳被附近事件污染。
```

**案例 2.C.2 — 日期精度：前/后偏差**
```
样本：conv-41
问题：When did John have a party with veterans?
标准答案：The Friday before 20 May 2023
模型回答：20 May, 2023
F1=0.75
原因分析：图存储了"May 20"作为时间节点，但标准答案要求的是"May 20之前的周五"，
         相对时间关系（before/after某节点）未被图准确建模。
```

#### ✓ 成功案例

**案例 2.OK.1 — 精确日期检索**
```
样本：conv-30
问题：When Jon has lost his job as a banker?
标准答案：19 January, 2023
模型回答：19 January, 2023
F1=1.00
分析：这是对话开篇的核心事件，单次出现且日期明确，
      Event节点的time属性精确记录了"19 January, 2023"。
```

**案例 2.OK.2 — 月份级精度足够**
```
样本：conv-30
问题：When Gina has lost her job at Door Dash?
标准答案：January, 2023
模型回答：January 2023
F1=1.00
分析：月份级时间需求对图的time属性精度要求较低，
      容易满足。
```

### Cat2 核心问题

1. **次要活动时间未捕获**：非核心事件的具体时间在构建时被忽略（32%是"Unknown"）
2. **时间节点竞争**：同一时间段有多个事件时，ReAct易跳错节点（42%错误时间点）
3. **相对时间关系**：图只存储绝对时间戳，"X之前的周五"等相对关系难以表示

---

## 6. Cat3 — 推理/常识问答（96题，68.7% Acc）

### 问题特征

推理题不要求直接检索事实，而是从对话中的线索推断隐性结论（如"他可能住在哪里？"、"她对X会有什么态度？"）。这类问题测试系统能否整合多个弱信号进行推断。**Cat3是最难类别，准确率最低（68.7%）**，且失败几乎全部是F1=0的完全错误推断。

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：推断方向相反 | 10 | 38% |
| B：推断过于笼统/无效 | 9 | 35% |
| C：缺乏足够线索 | 7 | 27% |

### 案例分析

#### ✗ 模式A：推断方向相反

**案例 3.A.1 — 地理位置推断**
```
样本：conv-41
问题：Does John live close to a beach or the mountains?
标准答案：beach
模型回答：likely mountains
原因分析：John的对话中提到户外活动，模型可能从"远足/高地"相关节点推断，
         但实际上对话有关于海边的线索（如海鲜、海滩）被忽视。
         推理类问题需要权衡多个弱信号，图的子图检索只返回局部视角。
```

**案例 3.A.2 — 意愿推断**
```
样本：conv-41
问题：Would John be open to moving to another country?
标准答案：No, he has goals specifically in the U.S. like joining the military and running for office.
模型回答：likely open to moving to another country
原因分析：模型没有连接"参军"+"竞选公职"这两个强烈的美国国内目标，
         推断出了相反结论。这需要跨多节点的综合推理，
         而ReAct在3跳内未能聚合所有相关节点。
```

**案例 3.A.3 — 财务状态推断**
```
样本：conv-41
问题：What might John's financial status be?
标准答案：Middle-class or wealthy
模型回答：Strained, likely improving
原因分析：对话中John提到"有投资"和"资助社区项目"，
         是富裕的正向信号；但同时也有"努力拼搏"的表述。
         模型优先检索到后者，推断出"Strained（拮据）"而非整体偏富裕。
```

#### ✗ 模式B：推断过于笼统

**案例 3.B.1 — 宠物偏好推断**
```
样本：conv-42
问题：What pets wouldn't cause any discomfort to Joanna?
标准答案：Hairless cats or pigs, since they don't have fur, which is one of the main causes of Joanna's allergy.
模型回答：turtles
原因分析：模型知道Joanna对毛发过敏，但未能进一步推理"无毛动物"，
         错误地联想到了爬行类动物。这需要两步推理：
         过敏原因 → 推断无毛动物 → 举例。
```

**案例 3.B.2 — 设备推断**
```
样本：conv-42
问题：What Console does Nate own?（标准答案：A Nintendo Switch, since the game "Xenoblade 2" is made for this console.）
模型回答：Likely PC
原因分析：需要知道"Xenoblade 2是Nintendo Switch专属游戏"这一外部知识，
         再结合图中"Nate喜欢Xenoblade 2"进行推断。
         图只存储了游戏偏好，但未建立"游戏→主机"的外部知识链接。
```

#### ✗ 模式C：线索不足

**案例 3.C.1 — 昵称推断**
```
样本：conv-42
问题：What nickname does Nate use for Joanna?
标准答案：Jo
模型回答：Likely Joanna
原因分析：对话中"Jo"可能只出现一次且非关键信息，
         被图构建时忽略，构建出的Entity节点只有canonical_name "Joanna"，
         别名/昵称"Jo"未被捕获到aliases字段。
```

#### ✓ 成功案例

**案例 3.OK.1 — 简单是/否推理**
```
样本：conv-41
问题：Would John be considered a patriotic person?
标准答案：Yes
模型回答：Passion for veterans' rights, desire to serve his country, support for military veterans.
F1=0.00（但judge判CORRECT）
分析：模型答案方向正确（提供了支持"Yes"的证据），
      judge理解了语义一致性。说明judge在Cat3比F1更准确。
```

**案例 3.OK.2 — 时间节点推断**
```
样本：conv-41
问题：Around which US holiday did Maria get into a car accident?
标准答案：Independence Day
模型回答：Independence Day
F1=1.00
分析：事件时间"July 4附近"在图中有明确的时间戳节点，
      与"Independence Day"的关联可以通过常识+时间推断完成。
```

### Cat3 核心问题

1. **多步推理链断裂**：推理题需要2-3步间接推断，ReAct在max_hop=3的限制下有时无法完整连接
2. **弱信号权衡缺失**：多个弱信号需要综合权衡，但图检索返回的是最近似子图，可能偏向某一方向
3. **外部知识缺失**：部分推理需要图外常识（如"Xenoblade 2是Switch独占游戏"），当前系统只依赖对话内容
4. **细节信息丢失**：昵称、外号等低频细节在构建时被过滤

---

## 7. Cat4 — 多跳/细节问答（841题，85.1% Acc）

### 问题特征

Cat4是题量最大、准确率最高的类别（85.1%），包含需要跨轮次连接信息的问题，如"Jon在哪个时间点谈到了X？"、"对话中两人如何评价Y？"。图的多跳边结构在这里发挥了核心作用。

### 失败模式分类

| 模式 | 数量 | 占比 |
|------|------|------|
| A：检索到语义相近但非精确的描述（F1<0.1） | 81 | 70% |
| B：回答"Unknown" | 13 | 11% |
| C：部分重叠 | 4 | 3% |
| D：照片/图片相关问题 | 7 | 6% |

### 案例分析

#### ✗ 模式A：语义相近但描述不精确

**案例 4.A.1 — 一词之差**
```
样本：conv-30
问题：What does Jon's dance make him?
标准答案：happy
模型回答：second nature, living his dream
F1=0.00
原因分析：图中Event节点记录的是"舞蹈让Jon感到充实/梦想成真"，
         而标准答案要求的精确表述是"happy"。
         两者语义相近，但精确词汇不同。这类问题对token-level F1不友好。
```

**案例 4.A.2 — 情感描述泛化**
```
样本：conv-30
问题：How does Gina describe the feeling that dance brings?
标准答案：magical
模型回答：Stress relief, lifeline.
F1=0.00
原因分析："magical"是对话中的一个具体形容词，
         在图构建时被融入到更宏观的属性描述中（如"带来解压感"），
         原始词汇"magical"未被保留。
```

**案例 4.A.3 — 比喻类答案**
```
样本：conv-30
问题：What did Jon and Gina compare their entrepreneurial journeys to?
标准答案：dancing together and supporting each other
模型回答：Scary and rewarding, roller coaster.
F1=0.18
原因分析：对话中有两种描述：情感描述（scary/rewarding）和比喻（dancing together）。
         图检索返回了前者，而标准答案要求的是比喻部分。
         细粒度的比喻信息在图摘要时丢失。
```

#### ✗ 模式D：照片/图片相关问题

**案例 4.D.1 — 照片内容**
```
样本：conv-30
问题：What do the dancers in the photo represent?
标准答案：They are performing at the festival
模型回答：Expression, happiness, memories.
原因分析：对话中可能包含对照片的讨论，但图只存储了情感相关的属性，
         未存储"照片中舞者正在表演节日活动"这个具体事实。
         照片相关对话的信息密度高，需要细粒度存储。
```

**案例 4.D.2 — 人物评价**
```
样本：conv-30
问题：What does Gina say about the dancers in the photo?
标准答案：They look graceful
模型回答："brings back lots of memories", "dance is still your happy place"
原因分析：图中存储了Gina对舞蹈的整体情感评价，
         但对照片中舞者的具体评语"graceful"未被单独保留为节点属性。
```

#### ✓ 成功案例

**案例 4.OK.1 — 精确属性检索**
```
样本：conv-30
问题：What is Gina's favorite style of dance?
标准答案：Contemporary
模型回答：Contemporary
F1=1.00
分析：舞蹈风格偏好是明确的Entity属性，
      在图中存储为Gina节点的attrs.favorite_dance_style = "Contemporary"，
      一跳即可检索。
```

**案例 4.OK.2 — 物品特征**
```
样本：conv-30
问题：What kind of flooring is Jon looking for in his dance studio?
标准答案：Marley flooring
模型回答：Marley flooring
F1=1.00
分析：具体名词"Marley flooring"在图中作为Event节点的属性存储，
      检索精确。
```

**案例 4.OK.3 — 高F1跨轮次信息连接**
```
样本：conv-43（最高Acc样本之一，89%）
Cat4整体表现最佳：conv-47达到92%，conv-43达到89%
分析：这些样本的对话内容连贯性强，人物事件时序清晰，
      图的多跳边（before/after/updates）有效连接了跨session信息。
```

### Cat4 核心问题

1. **精确词汇丢失**：构建时摘要会用近义词替换原始表达，精确的形容词（"magical"/"graceful"）容易丢失
2. **照片/媒体内容**：照片讨论包含大量细节，图构建未专门处理此类内容
3. **竞争描述共存**：同一事件有多种描述维度（情感/比喻/事实），检索时选择了非标准答案期望的维度

---

## 8. 跨类别共同问题

### 8.1 次要信息丢失（影响Cat2最大）

对话中的主线事件（失业、创业）被完整提取，但侧面信息（何时开始读某本书、何时加入某个群）因为批次触发阈值或摘要压缩而丢失。

**修复方向**：降低GraphTrigger的触发门槛；增加"细粒度活动"的提取规则。

### 8.2 词汇精度损失（影响Cat1/Cat4）

图构建时LLM会将原始表述改写，精确名词（人名、地名、特定词汇）有时被泛化描述替代。

**修复方向**：在Construction提示中增加"保留原始引用词汇"的规则；对直接引语建立Quote节点。

### 8.3 推理上限（影响Cat3）

当前系统只存储显性事实，推理所需的间接推断由检索时的LLM完成。但如果所需线索分散在多个节点且不直接相邻，3跳内无法全部聚合。

**修复方向**：增加推理跳数（max_hop=5）；或在构建阶段添加推理边（如"John支持军事→可能不愿离开美国"）。

### 8.4 相对时间关系（影响Cat2）

图只存储绝对时间戳，"X之前的周五"/"两周后"等相对关系无法直接查询。

**修复方向**：在Event-Event边上存储时间差属性；专门处理相对时间表达。

---

## 9. 改进优先级建议

| 优先级 | 问题 | 预期收益 | 实现难度 |
|--------|------|----------|----------|
| 🔴 高 | 次要活动时间信息提取不足 | Cat2 +5-8% | 中（调整GraphTrigger阈值） |
| 🔴 高 | 精确词汇/名词丢失 | Cat1/Cat4 +3-5% | 中（调整Construction提示） |
| 🟡 中 | 增加推理跳数至5 | Cat3 +3-5% | 低（改配置） |
| 🟡 中 | 昵称/别名提取 | Cat3/Cat1 +2-3% | 中（提示工程） |
| 🟢 低 | 照片/媒体专项处理 | Cat4 +1-2% | 高（需新节点类型） |
| 🟢 低 | 相对时间关系建模 | Cat2 +2-3% | 高（需边属性扩展） |

---

## 10. 结论

GraphMemory系统在全量1540题评测中整体表现良好（Judge-Acc=80.7%，F1=0.508）。

- **优势**：多跳问答（Cat4, 85.1%）表现最强，验证了图结构对长对话信息组织的有效性
- **瓶颈**：推理类问题（Cat3, 68.7%）因需要隐性推断而表现较弱
- **结构性问题**：次要信息丢失和词汇精度损失是两类主要错误来源，均可通过优化构建提示改善
- **下一步优先行动**：优化GraphTrigger触发策略 + 在Construction提示中强化精确词汇保留

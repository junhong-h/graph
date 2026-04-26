# QA Prompt 对比：GraphMemory / mem-t / MAMGA

日期：2026-04-26

本文整理三套方法在 LoCoMo QA 阶段使用的回答格式 prompt，重点比较：

- 是否允许拒答
- 是否强制短答案
- 日期/时间格式
- Yes/No 和列表格式
- Cat3 推理题特殊规则
- Cat5 对抗题处理

## 1. 文件位置

| 方法 | Prompt 位置 | 调用位置 |
|---|---|---|
| GraphMemory | `src/graphmemory/graph_retrieval.py` 的 `_locomo_format()` / `_SYSTEM_PROMPT` | `GraphRetriever.answer()` |
| GraphMemory compression | `src/graphmemory/graph_retrieval.py` 的 `_compress_final_answer()` | `GraphRetriever._finalize_answer()`，需启用 `--compress-final-answer` |
| mem-t | `/Users/junhong/Projects/third-party/mem-t/memory_retrieval.py` 的 `get_final_result_format()` | `FinishTool` / retrieval agent |
| MAMGA | `/Users/junhong/Projects/forks/MAMGA/memory/answer_formatter.py` 的 `build_qa_prompt()` | `/Users/junhong/Projects/forks/MAMGA/memory/test_harness.py` |

## 2. 核心差异表

| 维度 | GraphMemory 当前修正版 | mem-t | MAMGA |
|---|---|---|---|
| Cat1-4 是否允许拒答 | 不允许。明确禁止 `Not mentioned` / `Unknown` / `Information not found` | 不允许。信息不足时继续检索，max steps 必须给答案 | 部分类别允许 `Information not found`，尤其 temporal/default prompt |
| Cat5 是否允许拒答 | 允许，输出 `Not mentioned in the conversation` | mem-t LoCoMo loader 跳过 Cat5 | 允许，输出 `Information not found` |
| 短答案 | 强制 final answer string，通常 1-8 words | 强制 short phrase | 按类别要求 concisely，single-hop 2-15 words typical |
| 日期格式 | 只输出日期/时间短语，如 `15 July, 2023` / `July, 2023` | 详细区分绝对时间和相对时间 | temporal prompt 要求 `D Month YYYY` |
| Yes/No | Yes/No 问题以 Yes/No 开头，非 Yes/No 不以 Yes/No 开头 | 同样要求对齐问题类型 | Cat1 prompt 要求 Yes/No start；Cat3 prompt 要求 `Yes/No, because ...` |
| 列表 | 逗号分隔，不用 bullet / and / prose | 逗号分隔，不用 and | Cat1 prompt 要求 `item1, item2, item3` |
| Cat3 | 短推断标签或 short Yes/No，可用 `likely` | open-domain，可基于模型知识推理，禁止 Unknown | `Yes/No, because [brief reason]` 或列 2-3 traits |
| 后处理 | 轻量 canonicalization；可选 LLM compression | Finish tool 返回答案 | `extract_answer()` + Cat5 validation |

## 3. GraphMemory Prompt

位置：

- `src/graphmemory/graph_retrieval.py`
- `_locomo_format(category)`
- `_SYSTEM_PROMPT`

### 3.1 Answer Format

```text
The Final Result's Format Must Follow These Rules:
1. Return ONLY the final answer string. No explanations, evidence quotes, source mentions, or full supporting sentences.
2. Prefer a short phrase answer: the shortest wording that preserves the facts, usually 1-8 words.
3. Copy exact names, dates, places, adjectives, and short phrases from the conversation whenever possible.
4. Date/time answers: output only the date/time phrase, e.g. '15 July, 2023', 'July, 2023', 'the next day'.
5. Yes/No questions: start with 'Yes' or 'No'. Non-Yes/No questions must NOT start with 'Yes' or 'No'.
6. List answers: comma-separated items only; no bullets, no 'and', no prose.
```

Cat1-4 absence policy：

```text
7. For Cat1-4 answerable questions, NEVER output 'Not mentioned in the conversation', 'Unknown', 'I don't know', or 'Information not found'. If evidence is incomplete, use raw_fallback or give the best short answer supported by available evidence.
```

Cat5 absence policy：

```text
7. This is an adversarial unanswerable question. If the described event/fact is not directly supported, output exactly 'Not mentioned in the conversation'. Do not fabricate.
```

Cat3 extra rule：

```text
8. Inference questions: answer with the inferred label or a short Yes/No answer; use 'likely' only when genuinely uncertain.
```

Max-step rule：

```text
When max steps are reached, MUST call finish with your best short answer.
```

### 3.2 Retrieval System Rules

```text
You are a graph-memory retrieval agent. Answer the user's question by exploring a knowledge graph and retrieving raw conversation turns.

[Available actions — output as a JSON object]

1. Jump: expand your search frontier along graph edges.
   {"action": "jump", "node_ids": ["<8-char-id>", ...], "relation_family": "entity-event|entity-entity|event-event|any", "constraint": "<optional filter, e.g. time range or node type>", "budget": <int 1-5>}

2. Raw Fallback: search raw conversation turns when graph evidence is insufficient.
   {"action": "raw_fallback", "query": "<search query>"}

3. Finish: return the final answer when confident.
   {"action": "finish", "answer": "<concise answer>"}

[Rules]
- Start from the anchor nodes provided, then Jump to explore.
- Jump along entity-event edges to find what happened to people/places.
- Jump along event-event edges (before/after/updates) for temporal chains.
- Jump along entity-entity edges carefully — easiest to over-expand.
- After each Jump, decide: is the evidence enough to finish? If yes, call finish.
- If the graph lacks key details (exact quotes, fine-grained facts), use raw_fallback.
- You may use raw_fallback AND jump in the same session.
- The finish.answer value must follow the Answer format exactly. Do not put evidence,
  reasoning, or quoted conversation snippets in finish.answer.
- Follow the Answer format's absence policy exactly. For Cat1-4 answerable questions,
  do NOT finish with "Not mentioned in the conversation", "Unknown", or
  "Information not found"; use raw_fallback or give the best short answer from evidence.
  For Cat5 adversarial questions, finish with "Not mentioned in the conversation" only
  when the question premise is not directly supported after exploration.
- When max hops are reached you MUST call finish with your best guess.
- Do NOT output any text outside the JSON object.
```

### 3.3 Final Answer Compression Prompt

启用方式：

```bash
python scripts/run_qa.py ... --compress-final-answer
```

Prompt：

```text
Rewrite the draft answer into the shortest LoCoMo-style final answer.
Keep the same factual meaning. Do not add facts. Do not explain.
Do NOT turn a concrete draft answer into 'Not mentioned in the conversation'.

{answer_format}

Question: {question}

Evidence:
{evidence}

Draft answer: {answer}

Final answer:
```

## 4. mem-t Prompt

位置：

- `/Users/junhong/Projects/third-party/mem-t/memory_retrieval.py`
- `get_final_result_format(benchmark_name, category)`

### 4.1 Cat3 Prompt

```text
The Final Result's Format Must Follow These Rules:
1. Provide a short phrase answer, not a full sentence.
2. The question may require you to analyze and infer the answer from the retrieved information.
3. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic numerals in the answer respectively. Numbers are represented by English words by default., eg. prefer **two** not 2.
4. This is an open-domain problem. **When answering this type of question, you can ignore all other requirements that you must be completely confident before responding.** NEVER answer 'I don't know./None./Unknown'. You can perform reasoning based on the retrieved information and your model knowledge. Uncertain inferences can be expressed using 'likely'.
5. When the answer has multiple phrases, connect them with commas don't use 'and'.
6. Ensure your response aligns directly with the question. For instance, start with 'Yes' or 'No' for binary questions, and do not name a province when asked for a country.
7. If the information is not enough, you MUST NOT answer 'Unknown' or 'I don't know.'. Instead, try searching other databases, use different query words or expand the retrieval top-K. Don't call the same tool with same query twice in a row. When reach the max steps, you MUST call the finish tool to give the final answer and MUST NOT say 'Unknown'.
```

### 4.2 Cat1/2/4 Prompt

```text
The Final Result's Format Must Follow These Rules:
1. For questions requiring a date or time, strictly follow the format '15 July, 2023', 'July, 2023'.
2. Pay special attention to relative times like 'yesterday', 'last week', 'last Friday' in the text:
   + Only for last year/ last month/yesterday, calculate the absolute date, precise to year/month/day respectively, eg. 'July, 2023' or '19 July, 2023'.
   + For last week/weekend/Friday/Saturday, or few days ago etc, use 'the week/weekend/Friday before [certain absolute time]' to express **MUST NOT calculate the exact date**, just use the week/weekend/Friday before the certain absolute time, eg. 'the week/weekend/Friday before 15 July, 2023'/ few days before 15 July, 2023;
3. The answer should be the form of a short phrase (roughly a few words) for the following question, not a full sentence.
4. Use exact wording from the original conversation whenever possible.
5. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic numerals in the answer respectively. If it is a quantity or frequency counted by yourself, default to using English word, eg. prefer **two** not 2.
6. When the answer has multiple phrases, connect them with commas don't use 'and'.
7. Ensure your response aligns directly with the question. For instance, start with 'Yes' or 'No' for binary questions, and do not name a province when asked for a country.
8. If the information is not enough, you MUST NOT answer 'Unknown' or 'I don't know.'. Instead, try searching other databases, use different query words or expand the retrieval top-K. Don't call the same tool with same query twice in a row. When reach the max steps, you MUST call the finish tool to give the final answer and MUST NOT say 'Unknown'.
```

## 5. MAMGA Prompt

位置：

- `/Users/junhong/Projects/forks/MAMGA/memory/answer_formatter.py`
- `AnswerFormatter.build_qa_prompt(context, question, category)`
- 调用：`/Users/junhong/Projects/forks/MAMGA/memory/test_harness.py`

### 5.1 Cat1 Multi-hop

```text
Connect facts across the context to answer this question.

{context}

QUESTION: {question}

MULTI-HOP INSTRUCTIONS:
1. Look at KEY FACTS section first if present
2. Connect related information about the same person/topic
3. For "both/all" questions: Find commonalities between people
4. For research/identity: Connect clues (e.g., "researched X" + "chose org for X" = X)
5. Answer format:
   - Lists: "item1, item2, item3"
   - Counts: "Three"
   - Yes/No: Start with "Yes" or "No"

ANSWER:
```

### 5.2 Cat2 Temporal

```text
Extract temporal information from the context.

{context}

QUESTION: {question}

TEMPORAL RULES:
1. For "when" questions: Extract or calculate the date/time
   - Use "Event dates mentioned" for relative dates (e.g., "yesterday", "last week")
   - Format dates as: D Month YYYY (e.g., "7 May 2023" not "07 May 2023")
2. For "how long": Extract the duration mentioned
3. For "which month/year": Extract just the month or year
4. Use event dates NOT conversation timestamps
5. If no date/time found → "Information not found"

ANSWER (only the date/time/duration):
```

### 5.3 Cat4 Single-hop

```text
Find and extract the specific fact requested.

{context}

QUESTION: {question}

INSTRUCTIONS:
- Find the EXACT information requested
- Answer with the specific fact only (2-15 words typical)
- For "what": Extract the specific item/thing/activity
- For "who": Extract the name/person
- For "where": Extract the location
- For "when": Extract the date/time
- For "why": Extract the reason given
- For "how": Extract the method/way described
- Do NOT add explanations, only the fact

ANSWER:
```

### 5.4 Cat5 Adversarial

```text
Verify the EXACT entity exists before answering.

{context}

QUESTION: {question}

CRITICAL RULES:
1. Check if the EXACT person/entity in the question exists in context
2. If question asks about "Person A" but context only has "Person B" → "Information not found"
3. Do NOT make substitutions (e.g., Melanie ≠ Caroline)
4. When uncertain or entity mismatch → "Information not found"

ANSWER (be strict):
```

### 5.5 Cat3 Open-domain / Hypothetical

```text
Make reasonable inferences based on the context.

{context}

QUESTION: {question}

INFERENCE GUIDELINES:
1. For "Would X..." questions: Answer "Yes/No, because [brief reason]"
2. For personality traits: List 2-3 specific traits based on behavior
3. For preferences: Give specific answer based on their interests
4. Make reasonable inferences from available evidence
5. Be confident but base answers on context

ANSWER:
```

### 5.6 Default Prompt

```text
Answer based on the conversation context.

{context}

QUESTION: {question}

INSTRUCTIONS:
1. Check KEY FACTS section first if present
2. Connect related information across memories
3. Verify correct person/entity is mentioned
4. For comparisons: Find commonalities or differences
5. Pay attention to who said what (speaker tags)
6. If information not found → "Information not found"
7. Answer concisely (5-6 words typical)

ANSWER:
```

## 6. 对 GraphMemory F1 的直接启示

1. mem-t 对 Cat1-4 的核心原则是“不拒答”：信息不足时继续搜，max steps 也必须给 best answer。
2. GraphMemory 之前 F1-fix 下降的直接原因是 Cat1-4 过度输出 `Not mentioned in the conversation`；当前已改为 Cat1-4 禁止拒答。
3. MAMGA 在 Cat2/default 中仍允许 `Information not found`，这可能解释其 exact accuracy 和部分 F1 较低的情况。
4. 下一轮 GraphMemory F1 修正实验建议只重跑 Cat1-4/mem-t split，并重点观察：
   - `Not mentioned` / `Unknown` / `Information not found` 在 Cat1-4 中是否降为 0
   - pred/gold token ratio 是否保持接近 1
   - F1 是否从旧 Qwen 的 0.3889 提升

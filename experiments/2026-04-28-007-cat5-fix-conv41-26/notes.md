# 实验 2026-04-28-007：cat5-fix-conv41-26

## 假设

对 005/006 分析发现 Cat5 误判的核心路径是：LLM 发起 raw_fallback → 拉入语义相关但非答案的内容 → 以近似证据给出具体答案。三项改动联合消除这一路径：

1. **Cat5 raw_fallback 屏蔽**：在主循环中拦截 LLM 对 Cat5 题的 raw_fallback 调用，迫使 LLM 仅基于图证据决策。预期 Cat5 Acc 从 006 的 65.9% 恢复到 ≥75%（接近 004 的 80.5%）。
2. **Cat5 prompt 强化**：answer format 和 system prompt 均明确"话题相关不等于答案存在，必须有 EXPLICIT 声明"。配合改动1减少 LLM 误判置信度。
3. **时间词加权**：`_score_jump_candidate` 对 question 中出现的年份/月份在 node.attrs.time 匹配时给额外分，防止同人物不同时间 event 混淆。预期 Cat4 conv-41 改善。

## 与上次的差异

| 维度 | 006 qwen-json-conv41 | 本次 |
|------|----------------------|------|
| 图 | 重新构图（006 conv-41） | 复用 006 图 + 补 conv-26 图 |
| 样本 | conv-41 | conv-41 + conv-26 |
| Cat5 raw_fallback | 允许 | **屏蔽** |
| Cat5 answer format | "not directly supported" | **"EXPLICIT statement"** |
| system prompt Cat5 | "not directly supported after exploration" | **"topically similar NOT enough"** |
| jump 时间词加权 | 无 | **年份+2.0 / 月份+1.0** |

## 改动

- `src/graphmemory/graph_retrieval.py:answer()` — Cat5 raw_fallback 拦截，改 action 为 `cat5_raw_fallback_blocked`
- `src/graphmemory/graph_retrieval.py:_locomo_format()` — Cat5 answer format 要求 EXPLICIT statement
- `src/graphmemory/graph_retrieval.py:_SYSTEM_PROMPT` — Cat5 规则强化 topically similar != evidence
- `src/graphmemory/graph_retrieval.py:_score_jump_candidate()` — 新增 `_YEAR_RE`/`_MONTH_RE`，time attr 匹配加分

## 运行

```bash
# conv-26 图复用 005（与 006 conv-26 图相同来源）
cp experiments/2026-04-27-005-refine2-current-retrieval/build/graphs/conv-26_graph.json \
   experiments/2026-04-28-007-cat5-fix-conv41-26/build/graphs/conv-26_graph.json
cp experiments/2026-04-27-006-qwen-json-conv41/build/graphs/conv-41_graph.json \
   experiments/2026-04-28-007-cat5-fix-conv41-26/build/graphs/conv-41_graph.json

# 复用 chroma（需要两个样本）
cp -r experiments/2026-04-27-005-refine2-current-retrieval/chroma/ \
      experiments/2026-04-28-007-cat5-fix-conv41-26/chroma/

python scripts/run_qa.py       --exp-dir experiments/2026-04-28-007-cat5-fix-conv41-26
python scripts/summarize_exp.py --exp-dir experiments/2026-04-28-007-cat5-fix-conv41-26
```

## 运行环境

- commit: `28ef9b6`
- build 耗时: 不重建
- QA 耗时: ~（392 题，conv-41+conv-26）
- 图文件: `experiments/2026-04-28-007-cat5-fix-conv41-26/build/graphs/`

## 图统计

复用 006 conv-41 图 + 005 conv-26 图。

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | 176 | 243 | 35 | 141 | 228 |
| conv-26 | 96 | 117 | 33 | 63 | 102 |

## QA 结果

| Category | 006 conv-41 | 本次 conv-41 | 005 conv-26 | 本次 conv-26 | 本次合计 |
|----------|------------|-------------|------------|-------------|---------|
| Cat1     | 96.8% | | | | |
| Cat2     | 92.6% | | | | |
| Cat3     | 75.0% | | | | |
| Cat4     | 90.7% | | | | |
| Cat1-4   | 91.4% | | 87.5% | | |
| Cat5     | 65.9% | | 63.8% | | |
| Overall  | 86.0% | | 81.9% | | |

## 分析

（待填）

## 遗留问题

（待填）

## 下一步

（待填）

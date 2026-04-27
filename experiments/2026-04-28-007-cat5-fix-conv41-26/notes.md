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
# 图和 chroma 均复用 005（005 chroma 包含 conv-41(156节点)+conv-26(96节点)，节点ID匹配）
# 006 chroma 只有 conv-41(176节点)，与 005 图不兼容，故不用 006 图
cp experiments/2026-04-27-005-refine2-current-retrieval/build/graphs/conv-26_graph.json \
   experiments/2026-04-28-007-cat5-fix-conv41-26/build/graphs/
cp experiments/2026-04-27-005-refine2-current-retrieval/build/graphs/conv-41_graph.json \
   experiments/2026-04-28-007-cat5-fix-conv41-26/build/graphs/
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

复用 005 的图和 chroma（conv-41 refine2 图，156节点；conv-26，96节点）。

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | 156 | 164 | 43 | 113 | 148 |
| conv-26 | 96 | 117 | 33 | 63 | 102 |

## QA 结果

对比基线 005（同图，旧检索代码）：

| Category | 005 Acc | 007 Acc | 变化 | 005 F1 | 007 F1 | 变化 |
|----------|--------:|--------:|-----:|-------:|-------:|-----:|
| Cat1     | 90.5% | 88.9% | −1.6 | 0.337 | 0.348 | +0.011 |
| Cat2     | 89.1% | 87.5% | −1.6 | 0.408 | 0.381 | −0.027 |
| Cat3     | 76.2% | 76.2% | ±0.0 | 0.196 | 0.218 | +0.022 |
| Cat4     | 85.9% | **90.4%** | **+4.5** | 0.468 | 0.466 | −0.002 |
| Cat1-4   | 86.8% | **88.5%** | **+1.6** | 0.410 | 0.407 | −0.003 |
| Cat5     | 67.0% | **87.5%** | **+20.5** | 0.673 | 0.877 | **+0.204** |
| Overall  | 82.6% | **88.3%** | **+5.7** | 0.469 | — | — |

按样本：

| 样本 | 005 | 007 | 变化 |
|------|----:|----:|-----:|
| conv-26 | 81.9% | 87.9% | +6.0 |
| conv-41 | 82.9% | 88.6% | +5.7 |

翻转：**33 UP / 10 DOWN**，净 +23 题。

| | Cat1 | Cat2 | Cat3 | Cat4 | Cat5 |
|--|-----:|-----:|-----:|-----:|-----:|
| DOWN | 1 | 4 | 1 | 4 | 0 |
| UP | 0 | 3 | 1 | 11 | 18 |

## 分析

### Cat5：完全恢复（+20.5%，18 UP / 0 DOWN）

Cat5 raw_fallback 屏蔽效果显著。005 的 9 道 Cat5 DOWN 全部在 007 中正确拒答；Cat5 Acc 从 67.0% 升至 87.5%，超过 004 的 69.3%，接近 Oracle 水平。18 道 UP 均为"Not mentioned in the conversation"的正确拒答，0 道 DOWN。

### Cat4：时间词加权有效（+4.5%，11 UP / 4 DOWN）

005 中 conv-41 因同人物不同 event 混淆导致的 Cat4 DOWN 题大量翻转为 UP。典型案例：
- `conv-41_q123`：005 回答"join the meeting"，007 正确找到"have dinner with friends from the gym"
- `conv-26_q96`：005 回答"counseling and mental health"（过于模糊），007 定位到"working with trans people, helping them accept themselves"

### Cat1/Cat2 小幅退步（各 −1.6%）

4 道 Cat2 DOWN 集中在 conv-26，需要分析是否与屏蔽 raw_fallback 有关（Cat2 是日期题，部分日期需要从原文精确提取，raw_fallback 对 Cat2 有帮助）。

### 与 004 基线对比

| | 004 | 007 | 变化 |
|--|----:|----:|-----:|
| Cat1-4 Acc | 88.8% | 88.5% | −0.3 |
| Cat5 Acc | 69.3% | 87.5% | +18.2 |
| Overall Acc | 84.7% | 88.3% | +3.6 |

007 的 Cat5 大幅优于 004，Cat1-4 基本持平。综合 Overall 超过 004 约 3.6%。

## 遗留问题

- Cat2 raw_fallback 屏蔽范围过宽：Cat2 日期题有时需要 raw 原文精确定位日期，屏蔽仅限 Cat5 已生效，但需要确认 4 道 Cat2 DOWN 的具体原因
- Cat5 仍有 12.5% 错误率（约 11 题），需要分析是 graph localization 问题还是 LLM 在图证据中推断
- 时间词加权的权重（year +2.0 / month +1.0）未做消融，可能存在过度或不足

## 下一步

- 分析 Cat2 DOWN 的 4 道题，确认是否可以在不影响 Cat5 的情况下有针对性地恢复
- 在全量 10 样本上跑 007 配置，确认结论可泛化
- 考虑把 007 的配置定为新的默认 retrieval baseline

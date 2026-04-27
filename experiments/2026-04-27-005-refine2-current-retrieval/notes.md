# 实验 2026-04-27-005：refine2-current-retrieval

## 假设

复用 004 的 Refine2 图和 Chroma，只切换到当前 HEAD 的检索代码，验证最新检索改动是否导致准确率下降。

重点验证 `0ff3e53` 的影响：
- jump 候选从关键词/constraint 打分改为 bge-m3 向量排序
- 多 anchor 改为 per-anchor 独立分配 budget

预期如果准确率下降集中在 Cat4 或 near-miss 错误，则说明 retrieval reranking/disambiguation 是主因，而不是 Refine2 建图本身。

## 与上次的差异

| 维度 | 004 refine2-prop-conv41-26 | 本次 005 |
|------|----------------------------|----------|
| 图 | Refine2 图 | 复用 004 图 |
| Chroma | 004 chroma | 复制 004 chroma |
| 构图代码 | eaa1ca4 | 不重建 |
| 检索代码 | eaa1ca4 | 36c59ab 当前 HEAD |
| jump 排序 | 旧检索 | 向量排序 + per-anchor budget |
| 样本 | conv-41 + conv-26 | conv-41 + conv-26 |

## 改动

- 不改代码。
- 复制 `experiments/2026-04-27-004-refine2-prop-conv41-26/build/graphs/` 到本实验。
- 复制 `experiments/2026-04-27-004-refine2-prop-conv41-26/chroma/` 到本实验。
- 只运行 QA 和 summarize。

## 运行

```bash
python scripts/run_qa.py --exp-dir experiments/2026-04-27-005-refine2-current-retrieval
python scripts/summarize_exp.py --exp-dir experiments/2026-04-27-005-refine2-current-retrieval
```

## 运行环境

- commit: `36c59ab`
- build 耗时: 不重建，复用 004 图
- QA 耗时: ~6 min（392 题，workers=4）
- 图文件: `experiments/2026-04-27-005-refine2-current-retrieval/build/graphs/`

## 图统计

复用 004 图，理论上应与 004 一致。

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-41 | 156 | 164 | 43 | 113 | 148 |
| conv-26 | 96 | 117 | 33 | 63 | 102 |

## QA 结果

| Category | 004 eaa1ca4 retrieval | 005 current retrieval | 变化 |
|----------|------------------------|-----------------------|------|
| Cat1     | 92.1% (F1=0.385) | 90.5% (F1=0.337) | −1.6 / −0.048 |
| Cat2     | 85.9% (F1=0.360) | 89.1% (F1=0.408) | +3.1 / +0.048 |
| Cat3     | 81.0% (F1=0.193) | 76.2% (F1=0.196) | −4.8 / +0.003 |
| Cat4     | 89.7% (F1=0.458) | 86.5% (F1=0.468) | −3.2 / +0.010 |
| Cat1-4   | 88.8%             | 86.8%             | −2.0 |
| Cat5     | 69.3% (F1=0.695) | 67.0% (F1=0.673) | −2.3 / −0.022 |
| Overall  | 84.7% (F1=0.469) | 82.6% (F1=0.469) | −2.1 / ±0.000 |

## 分析

同图 ablation 支持“最新检索改动导致准确率下降”的判断。

### 分样本结果

| 样本 | 指标 | 004 | 005 | 变化 |
|------|------|-----|-----|------|
| conv-41 | Cat1-4 Acc | 90.8% | 86.2% | −4.6 |
| conv-41 | Cat5 Acc | 80.5% | 70.7% | −9.8 |
| conv-26 | Cat1-4 Acc | 86.8% | 87.5% | +0.7 |
| conv-26 | Cat5 Acc | 59.6% | 63.8% | +4.3 |

总体下降主要来自 conv-41。conv-26 略有改善，说明 current retrieval 不是单向变差，而是在不同图/样本上有较强波动。

### 翻转统计

004 正确但 005 错误：24 题。
004 错误但 005 正确：16 题。

下降集中在：
- Cat4：11 down / 5 up
- Cat5：9 down / 7 up
- conv-41：15 down / 4 up

典型 005 退化：
- `conv-41_q154` Cat5：004 拒答正确，005 从 raw fallback near-miss 中回答 Maria money problems。
- `conv-41_q168` Cat5：004 拒答正确，005 回答 John online group/community service 事实。
- `conv-41_q178` Cat5：004 拒答正确，005 回答 Maria participated in 5K charity run。
- `conv-41_q133` Cat4：gold 是 military memorial experience，005 抽到 rescue dog。
- `conv-41_q137` Cat4：gold 是 John lost job，005 抽到 promotion。

这些错误符合 near-miss / same-topic contamination，而不是图里完全没有答案。

### Trace 信号

005 的 conv-41 trace：
- Cat1 raw fallback 35.5%，forced 16.1%，avg jump 0.55
- Cat5 raw fallback 65.9%，forced 41.5%，avg jump 0.27

相比 004，conv-41 Cat5 raw fallback 从 56.1% 升到 65.9%，说明 current retrieval 更频繁把不可答题带入 raw fallback；这会找到语义相近但约束不满足的证据，诱发具体作答。

### 结论

`0ff3e53` 的方向有局部收益（conv-26、Cat2），但当前实现风险较大：

- `_execute_jump()` 改为纯向量排序后，没有使用 LLM action 中的 `constraint`。
- 原 `_score_jump_candidate()` 中的 constraint、predicate、question-term 加权不再参与排序。
- per-anchor budget 会扩大多锚点探索面，可能把相似但错误的 event 带入 evidence。

因此不建议直接保留 current retrieval 作为默认结果。下一步应实现 hybrid jump scoring，而不是简单回退或继续扩大召回。

## 遗留问题

- 同图 ablation 只比较 retrieval，不比较 construction；结论限于 004 图。
- LLM judge 有少数宽松/不稳定案例，例如 `Unknown` 被判正确，后续分析仍需看逐题内容。
- 需要在 hybrid scoring 后重跑同图 ablation，确认是否能保留 conv-26/Cat2 收益并恢复 conv-41/Cat5。

## 下一步

改造 `_execute_jump()`：

```text
score = vector_similarity
      + question_term_match
      + predicate_match
      + constraint_match
      + actor/time/object/purpose match
      - near_miss_penalty
```

至少先恢复 constraint 与关键词/predicate 信号，再做 006 同图 ablation。

# 实验 2026-04-27-002：baseline-conv26

## 假设
用 commit 0408882（Apr 25，接近 Apr 20 版本）原始代码跑 conv-26，建立与实验 001 refine-jump 的同口径单样本 baseline。
Apr 20 全量报告（docs/report_progress_apr20.md）基于 10 个样本，本次单样本 conv-26 可与 001 直接比较。

## 与上次的差异
本实验为 baseline，001 是相对于本次的改进。

| 维度 | 本次 002 (baseline) | 001 refine-jump |
|------|---------------------|-----------------|
| jump 候选打分 | 关键词词频 `_score_jump_candidate` | 向量相似度 `rank_nodes_by_query` |
| 多锚点扩展 | 所有锚点共享 budget，统一排序 | 每锚点独立 ceil(budget/n)，claimed 防重 |
| 代码版本 | 0408882 | 0ff3e53 |
| 样本 | conv-26 | conv-26 |

## 改动
无代码改动，使用 0408882 原始版本。

## 运行

```bash
# 在 worktree 里执行（旧代码不支持 --exp-dir）
EXP=/Users/junhong/Projects/research/graphmemory/experiments/2026-04-27-002-baseline-conv26

python scripts/build_memory.py --config /tmp/baseline_build.yaml --sample-ids conv-26
python scripts/run_qa.py       --config /tmp/baseline_qa.yaml    --sample-ids conv-26
```

## 运行环境
- commit: `0408882`（worktree `/tmp/gm-baseline`）
- build 耗时: ~6 min（conv-26，单样本）
- QA 耗时: ~25 min（199 题，workers=1）
- 图文件: `experiments/2026-04-27-002-baseline-conv26/build/graphs/conv-26_graph.json`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 | entity-entity | event-event |
|------|------|-----|--------|-------|---------|--------------|-------------|
| conv-26 | 21 | 30 | 2 | 19 | 21 | 0 | 9 |

trigger 率: 100%（111/111 batches），dedup 合并: 1 节点

## QA 结果

| Category | Apr 20 全量 | 002 baseline (conv-26) | 001 refine-jump | 002→001 变化 |
|----------|------------|------------------------|-----------------|-------------|
| Cat1     | 90.4%      | 87.5% (F1=0.245)       | 87.5% (F1=0.261)| 0 / +0.016 |
| Cat2     | 78.5%      | 81.1% (F1=0.356)       | 78.4% (F1=0.418)| −2.7 / +0.062 |
| Cat3     | 74.0%      | 76.9% (F1=0.173)       | 92.3% (F1=0.291)| +15.4 / +0.118 |
| Cat4     | 91.2%      | 84.3% (F1=0.406)       | 90.0% (F1=0.443)| +5.7 / +0.037 |
| Cat1-4   | 87.3%（全量）| 84.2% (F1=0.340)      | 86.8% (F1=0.386)| +2.6 / +0.046 |
| Cat5     | 36.1%      | 19.1% (F1=0.084)       | 51.1% (F1=0.511)| +32.0 / +0.427 |
| Avg F1   | 0.390      | 0.280                  | 0.415           | +0.135 |

## 分析
- 图规模差异极大：002 baseline 只有 21 节点 / 30 边，001 refine-jump 有 107 / 120
  - 旧版代码（0408882）的构建逻辑生成的图明显更小，说明两次实验对比的不只是检索算法，图质量本身也有差异
  - 旧版没有 `low-value abstract event` 过滤，但 `Skip` op 更多（49 次），可能是 prompt 不同导致 LLM 更保守
- Cat5（对抗不可答）差距最大：002 只有 19.1%，001 达到 51.1%（+32 pts）
  - 图太小导致大量题目直接走 raw_fallback，而 raw_fallback 对 Cat5 的答案质量差
- Cat3（推理）002=76.9%，001=92.3%（+15 pts），与图覆盖率相关
- Cat1-4 综合：002=84.2%，001=86.8%（+2.6 pts），改善有限但方向正确
- **结论：两次实验的代码差异不是单纯的检索算法差异，还包含图构建质量的差异，需要控制变量才能准确归因**

## 遗留问题

## 下一步

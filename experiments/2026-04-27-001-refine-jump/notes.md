# 实验 2026-04-27-001：refine-jump

## 假设
jump 候选用向量相似度（bge-m3）排序，比关键词打分更能捕捉语义相关性；
多锚点各自独立扩展、分配 ceil(budget/n) 个槽位后合并，防止一个 anchor 独占 budget。

## 改动
- `graph_store.py`：新增 `rank_nodes_by_query(query, candidate_ids)` — 在候选集内向量排序
- `graph_retrieval.py`：`_execute_jump` 改为 per-anchor 独立扩展，每个 anchor 各取 ceil(budget/n) 个最相关邻居

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-27-001-refine-jump
python scripts/run_qa.py       --exp-dir experiments/2026-04-27-001-refine-jump
```

## 图统计（conv-26）

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-26 | 107 | 120 | 38 | 69 | — |

## QA 结果（conv-26，199 题）

| Category | Baseline (Apr 20 P0+P1，全量) | 本次 (conv-26) | 变化 |
|----------|------------------------------|----------------|------|
| Cat1     | 90.4%                        | 87.5%          | −2.9 |
| Cat2     | 78.5%                        | 78.4%          | −0.1 |
| Cat3     | 74.0%                        | 92.3%          | +18.3 |
| Cat4     | 91.2%                        | 90.0%          | −1.2 |
| Cat1-4   | 87.3%（全量）                | 86.8%          | −0.5 |
| Cat5     | 36.1%                        | 51.1%          | +15.0 |
| Avg F1   | 0.390                        | 0.415          | +0.025 |

## 分析
- Cat3 大幅提升（+18 pts）可能来自向量打分对推理题更友好，也可能是 conv-26 单样本波动
- Cat5 显著提升（+15 pts），整体 F1 +0.025
- Cat1-4 与 baseline 基本持平（−0.5 pts），差距在误差范围内
- 注：baseline 是全量 10 样本，本次只跑 conv-26，对比不完全可靠
- 注：今天重建的图只有 107 节点，Apr 20 的图节点数不详（已被覆盖）

## 遗留问题
- 需要全量 10 样本跑一遍才能得出可靠结论
- `invalid edge family` 警告 28 次（LLM 偶发省略 family 字段，边被丢弃）
- 单样本结果波动大，Cat3 的 +18 pts 需要多样本验证

## 下一步
- 跑全量 10 样本 build + QA，得出可与 Apr 20 baseline 直接对比的数据
- 或先跑 ablation：同一张图，切换 jump 算法前后对比

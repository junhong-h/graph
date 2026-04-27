# 实验 2026-04-27-003：refine1-graph-conv41

## 假设
当前图中频繁出现 Event 孤点（无 entity-event 边），导致检索阶段 frontier 快速耗尽、forced finish。
通过修复构建 prompt（实体覆盖、Link 规则强化）、改进 Localizer（multi-seed）、修复 Jump/raw repair，
应可显著减少孤点，提升 Cat3（推理）和 Cat5（对抗不可答）的表现。

## 与上次的差异

本实验对比 docs/report_progress_apr20.md（Apr 20 全量 10 样本 baseline）。
注意：本次用 conv-41，基准是全量平均值，不是同口径对比。

| 维度 | Apr 20 全量 baseline | 本次 003 |
|------|---------------------|---------|
| 图构建 prompt | 原始 ontology | 强化 Entity/Event 区分 + Link 规则 |
| Localizer | 单 seed | multi-seed（多锚点并发） |
| Jump / raw repair | 基础 | 修复边界条件，加 raw repair |
| 代码版本 | 0408882 | 700b821 |
| 样本 | 全量 10 | conv-41 单样本 |

## 改动

- `graph_construction.py` — 强化 Entity/Event 区分 prompt，修复 Link 必须指向已有节点的规则
- `graph_localize.py` — multi-seed：同时从多个锚点出发扩展，缓解孤点导致的 frontier 为空
- `graph_retrieval.py` — Jump 循环改进；raw_fallback 修复
- `retrieval_trace_summary.py` — 新增检索轨迹诊断工具（辅助分析，不影响结果）

## 运行

```bash
# 原始 runs 目录（非 --exp-dir 模式，事后整理）
python scripts/build_memory.py --config <old_config> --sample-ids conv-41  # runs/refine_build_conv41/
python scripts/run_qa.py       --config <old_config> --sample-ids conv-41  # runs/qa_dashscope_refine_conv41/
```

## 运行环境
- commit: `700b821`（build 开始时 HEAD，Apr 27 01:16）
- build 耗时: ~28 min（conv-41，单样本，01:16–01:44）
- QA 耗时: ~10 min（193 题，workers=4，01:44–01:54）
- 图文件: `experiments/2026-04-27-003-refine1-graph-conv41/build/graphs/conv-41_graph.json`
- 原始 build run: `runs/refine_build_conv41/`
- 原始 QA run: `runs/qa_dashscope_refine_conv41/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 | event-event |
|------|------|-----|--------|-------|---------|-------------|
| conv-41 | 249 | 404 | 89 | 160 | 273 | 131 |

trigger 率: 95.5%（171/179 batches），dedup 合并: 0 节点
主要 ops: CreateEvent=160, CreateEntity=95, Link=419(rejected=153), RepairEventAttrs=160, RepairEventLink=30, AttachAttr=76

## QA 结果

| Category | Apr 20 全量 | 003 refine1 (conv-41) | 说明 |
|----------|------------|-----------------------|------|
| Cat1     | 90.4%      | 90.3% (F1=0.308)      | 持平 |
| Cat2     | 78.5%      | 85.2% (F1=0.377)      | +6.7pt |
| Cat3     | 74.0%      | 62.5% (F1=0.167)      | −11.5pt（仅 8 题，波动大） |
| Cat4     | 91.2%      | 96.5% (F1=0.498)      | +5.3pt |
| Cat1-4   | 87.3%      | **91.4%** (F1=0.389)  | +4.1pt |
| Cat5     | 36.1%      | **61.0%** (F1=0.617)  | +24.9pt |
| Avg F1   | 0.390      | **0.462**             | +0.072 |

## 分析
- Cat1-4 +4.1pt，Cat5 大幅提升 +24.9pt，与假设一致：减少孤点后检索能覆盖更多事实，Cat5 对图覆盖最敏感
- Cat3（推理）-11.5pt，但仅 8 题，统计不可靠；图结构改善不一定立刻改善多步推理
- 图规模大幅提升（conv-41: 249/404 vs 002 baseline conv-26: 21/30），说明新构建代码质量明显更高
- **注意：本次对比 Apr 20 全量均值 vs 单样本 conv-41，不同口径，不可直接归因；需 full-10 run 确认**
- events_with_fact=0 / events_with_quote=0：Refine1 还没有加 proposition layer（fact/quote 字段）
  这说明 Cat5 的提升主要来自 Event 连通性改善（减少孤点），而非 quote 质量

## 遗留问题
- Cat3 是否真的变差还是样本噪声？需要更多样本确认
- Refine1 代码（700b821）尚未包含 Link 数目强化（b2fbe94 "Refine graph construction invariants" 在 build 之后提交）
  Link rejected=153 比例较高（153/572=26.7%），仍有提升空间
- events_with_fact/quote 全为 0 → Refine2（proposition layer）待验证

## 下一步
- 实验 004：Refine2（proposition layer，commit eaa1ca4），conv-41 + conv-26 建图 + QA

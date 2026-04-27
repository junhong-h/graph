# experiments/

每次实验一个目录，命名格式：`YYYY-MM-DD-NNN-<slug>/`

- `YYYY-MM-DD`：实验日期
- `NNN`：当天序号（001, 002, ...）
- `<slug>`：2-4 个词描述改动，用连字符，e.g. `refine-jump`, `per-anchor-budget`, `full-10-samples`

## 目录结构

```
YYYY-MM-DD-NNN-<slug>/
  config.yaml          ← 完整配置（追踪）
  notes.md             ← 假设、改动、结果、分析（追踪）
  build/
    graphs/            ← 图 JSON 文件（追踪，~200KB/样本）
    graph_trajectories_*.jsonl  ← 构建轨迹（不追踪，大文件）
    build.log          ← 构建日志（不追踪）
  qa/
    qa_metrics.json    ← 汇总指标（追踪）
    qa_results.jsonl   ← 完整答案（不追踪，大文件）
    qa_results_eval.jsonl  ← 带 judge 的答案（不追踪）
    qa.log             ← QA 日志（不追踪）
  chroma/              ← 向量库（不追踪，可从 graphs/ 重建）
```

## 运行方式

```bash
# 构建图
python scripts/build_memory.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>

# 跑 QA + LLM judge
python scripts/run_qa.py --exp-dir experiments/YYYY-MM-DD-NNN-<slug>

# 旧方式（仍然支持）
python scripts/build_memory.py --config configs/build_memory_dashscope.yaml --sample-ids conv-26
```

## 实验列表

| ID | 名称 | 说明 | 样本 | Cat1-4 Acc | Cat5 Acc | F1 |
|----|------|------|------|------------|----------|----|
| [2026-04-27-002](2026-04-27-002-baseline-conv26/notes.md) | baseline-conv26 | commit 0408882 原始代码，conv-26 单样本 baseline | conv-26 | 84.2% | 19.1% | 0.280 |
| [2026-04-27-001](2026-04-27-001-refine-jump/notes.md) | refine-jump | 向量打分 + per-anchor 独立扩展 | conv-26 | 86.8% | 51.1% | 0.415 |
| [2026-04-27-003](2026-04-27-003-refine1-graph-conv41/notes.md) | refine1-graph-conv41 | 修复图构建质量：Entity 覆盖、Event isolation、multi-seed localizer | conv-41 | 91.4% | 61.0% | 0.462 |
| [2026-04-27-004](2026-04-27-004-refine2-prop-conv41-26/notes.md) | refine2-prop-conv41-26 | Refine2 proposition layer：fact/quote 字段 + low-value 过滤 | conv-41,26 | 90.8% / 86.8% | 80.5% / 59.6% | 0.520 / 0.420 |
| [2026-04-27-005](2026-04-27-005-refine2-current-retrieval/notes.md) | refine2-current-retrieval | 同图 ablation：复用 004 图，切到 current HEAD 检索 | conv-41,26 | 86.2% / 87.5% | 70.7% / 63.8% | 0.469 |
| [2026-04-27-006](2026-04-27-006-qwen-json-conv41/notes.md) | qwen-json-conv41 | Qwen JSON Mode + top_p/seed 稳定化后端到端重建 conv-41 | conv-41 | 91.4% | 65.9% | 0.504 |
| [2026-04-28-007](2026-04-28-007-cat5-fix-conv41-26/notes.md) | cat5-fix-conv41-26 | Cat5 raw_fallback 屏蔽 + prompt 强化 + 时间词加权，同图 ablation | conv-41,26 | 88.5% | **87.5%** | 0.407 |
| [2026-04-28-008](2026-04-28-008-skip-recovery-finish-check/notes.md) | skip-recovery-finish-check | Skip 拆 turn 重试 + Skip 规则收窄 + finish 自检；**全 10 样本** | all 10 | 86.2% | 86.3% | 0.499 |

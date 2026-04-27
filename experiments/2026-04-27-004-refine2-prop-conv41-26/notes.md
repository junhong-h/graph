# 实验 2026-04-27-004：refine2-prop-conv41-26

## 假设
在 Refine1（003）基础上，为每个 Event 增加 `fact`（自包含事实句）和 `quote`（最短原文引用）字段，
并过滤 low-value abstract event。
- `fact` 字段让检索时节点文本语义更清晰，向量匹配质量提升
- `quote` 字段在 Cat5（对抗不可答）场景下提供原文依据，减少幻觉
- 过滤 low-value event 减少噪声节点，提高图精度
预期 Cat5 进一步提升，Cat1-4 稳定或小幅改善。

## 与上次的差异

| 维度 | 003 refine1-graph-conv41 | 本次 004 |
|------|--------------------------|---------|
| Event attrs | 无 fact/quote | 有 fact + quote + source |
| low-value 过滤 | 无 | 过滤 abstract/generic event |
| Localizer 打分 | 节点 name/original_text | 优先使用 fact 文本 |
| 代码版本 | 700b821 | eaa1ca4 |
| 样本 | conv-41 | conv-41 + conv-26 |

## 改动

- `graph_construction.py` — CreateEvent prompt 新增 fact/quote/source 字段要求；low-value abstract event 过滤
- `graph_store.py` — 节点 evidence text 优先返回 fact/quote；
- `graph_localize.py` — 向量打分优先使用 fact 字段文本

## 运行

```bash
# worktree 运行（eaa1ca4 不支持 --exp-dir，用旧 config 方式）
EXP=/Users/junhong/Projects/research/graphmemory/experiments/2026-04-27-004-refine2-prop-conv41-26
python scripts/build_memory.py --exp-dir $EXP
python scripts/run_qa.py       --exp-dir $EXP
```

## 运行环境
- commit: `eaa1ca4`（worktree `/tmp/gm-refine2`）
- build 耗时: ~15 min（conv-41 + conv-26 并行，20:39–20:51）
- QA 耗时: ~20 min（392 题，workers=4）
- 图文件: `experiments/2026-04-27-004-refine2-prop-conv41-26/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | 对比 003/002 |
|------|------|-----|------------|
| conv-41 | 156 | 164 | 003: 249/404（-93 节点，low-value 过滤效果显著） |
| conv-26 | 96 | 117 | 002: 21/30，001: 107/120（居中） |

trigger 率、ops：待 summarize_exp.py 补充（trajectory 在 build/graphs 目录下）

## QA 结果

### conv-41（对比 003 Refine1）

| Category | 003 refine1 | 004 refine2 | 变化 |
|----------|------------|------------|------|
| Cat1     | 90.3% (F1=0.308) | 93.5% (F1=0.439) | +3.2 / +0.131 |
| Cat2     | 85.2% (F1=0.377) | 96.3% (F1=0.299) | +11.1 / −0.078 |
| Cat3     | 62.5% (F1=0.167) | 75.0% (F1=0.158) | +12.5 / −0.009 |
| Cat4     | 96.5% (F1=0.498) | 89.5% (F1=0.513) | −7.0 / +0.015 |
| Cat1-4   | 91.4%            | 90.8%            | −0.6 |
| Cat5     | 61.0% (F1=0.617) | 80.5% (F1=0.809) | +19.5 / +0.192 |
| Avg F1   | 0.462            | **0.520**        | +0.058 |

### conv-26（对比 002 baseline 和 001 refine-jump）

| Category | 002 baseline | 001 refine-jump | 004 refine2 | vs 002 | vs 001 |
|----------|-------------|----------------|------------|--------|--------|
| Cat1     | 87.5%        | 87.5% (F1=0.261) | 90.6% (F1=0.332) | +3.1 | +3.1 |
| Cat2     | 81.1%        | 78.4% (F1=0.418) | 78.4% (F1=0.405) | −2.7 | ±0 |
| Cat3     | 76.9%        | 92.3% (F1=0.291) | 84.6% (F1=0.214) | +7.7 | −7.7 |
| Cat4     | 84.3%        | 90.0% (F1=0.443) | 90.0% (F1=0.389) | +5.7 | ±0 |
| Cat1-4   | 84.2%        | **86.8%**        | **86.8%**        | +2.6 | ±0 |
| Cat5     | 19.1%        | 51.1% (F1=0.511) | **59.6%** (F1=0.596) | +40.5 | +8.5 |
| Avg F1   | 0.280        | 0.415            | **0.420**        | +0.140 | +0.005 |

## 分析
- **Cat5 大幅提升是最显著结果**：conv-41 +19.5pt（61→80.5%），conv-26 +40.5pt（19.1→59.6%）
  fact/quote 字段让模型在对抗不可答题时有更清晰的原文依据，减少幻觉作答
- **conv-41 Cat2 准确率大幅提升（+11.1pt）但 F1 反而下降（−0.078）**
  可能是 Acc 统计与 F1 标准不一致（judge_label 更宽松），需留意
- **conv-41 Cat4 下降 −7pt**：图节点从 249→156（low-value 过滤移除了 93 个节点），部分事实节点可能被误过滤
- **图规模**: conv-41 从 249→156（−37%），说明 low-value 过滤力度较大；conv-26 从 21→96（+357%），
  说明 Refine2 的 fact/quote 要求让建图更积极（旧代码 002 图太小是因为 LLM 更保守）
- **总体**: avg F1 conv-41=0.520（+0.058 vs 003），conv-26=0.420（+0.140 vs 002，+0.005 vs 001）；
  conv-26 的 Cat1-4 与 001 refine-jump 持平（86.8%），Cat5 超过 001（59.6% vs 51.1%，+8.5pt），
  说明 Refine2 的 fact/quote 图在 Cat5 上优于 refine-jump 检索改进，Cat1-4 效果相当

## 遗留问题
- conv-41 Cat4 下降 7pt：需检查哪些 Cat4 题目丢失，是否与 low-value 过滤阈值过激有关
- Cat2/Cat3 Acc 升但 F1 降的矛盾：judge_label 与 F1 评分标准不完全一致，需分析
- 当前 004 用的是 Refine2 图构建 + 旧检索（关键词打分）；还未测试 Refine2 图 + refine-jump 检索的组合效果

## 下一步
- 实验 005：Refine2 图（eaa1ca4）+ refine-jump 检索（current HEAD），对 conv-41+conv-26
- 或：先跑全量 10 个样本的 Refine2，确认单样本结论的普遍性

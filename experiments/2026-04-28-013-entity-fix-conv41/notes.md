# 实验 2026-04-28-013：entity-fix-conv41

## 假设

修正 Rule 2/4 后，Max、Coco、David 等命名角色应该有独立 Entity 节点，不再被埋进 Event 的 fact 文本里。Entity 总数应从 4 个明显增加。

## 与上次的差异
<!-- 对比 based_on 实验，本次改动了什么。未列出 = 与上次相同 -->

| 维度 | 上次 (based_on) | 本次 |
|------|----------------|------|
| 代码改动 | — | `file.py:fn` — 说明 |
| 模型 | qwen3-4b | qwen3-4b |
| jump_budget | 5 | 5 |
| seed_top_k | 5 | 5 |
| 样本 | conv-26 | conv-26 |

## 改动
<!-- 关键代码改动，格式：`文件:函数` — 说明 -->
-

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/YYYY-MM-DD-NNN-<name>
python scripts/run_qa.py       --exp-dir experiments/YYYY-MM-DD-NNN-<name>
```

## 运行环境
<!-- commit hash 务必填写 -->
- commit: `xxxxxxx`（`git rev-parse --short HEAD`）
- build 耗时: ~
- QA 耗时: ~（N 题）
- 图文件: `experiments/YYYY-MM-DD-NNN-<name>/build/graphs/`

## 图统计

| 样本 | 节点 | 边 | Entity | Event | e-ev 边 |
|------|------|-----|--------|-------|---------|
| conv-26 | | | | | |

## QA 结果

| Category | Baseline (<based_on>) | 本次 | 变化 |
|----------|-----------------------|------|------|
| Cat1     | | | |
| Cat2     | | | |
| Cat3     | | | |
| Cat4     | | | |
| Cat1-4   | | | |
| Cat5     | | | |
| Avg F1   | | | |

## 分析
<!-- 结果符合预期吗？哪里超预期/低于预期？根因是什么？ -->

## 遗留问题
<!-- 发现的新问题，或还未解决的问题 -->

## 下一步
<!-- 基于这次结果，下一步做什么 -->

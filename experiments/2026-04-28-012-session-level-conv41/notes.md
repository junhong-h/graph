# 实验 2026-04-28-012：session-level-conv41

## 假设

session 平均只有 ~700 tokens，完全在 Qwen3-4B 上下文范围内。以整个 session 为单位处理，LLM 有更完整的上下文，entity 跨 turn 解析更准；同时从 179 次 LLM 调用减少到 32 次（~5.6x），speed 大幅提升。

## 与上次的差异

| 维度 | 011 (k_turns=4) | 本次 |
|------|----------------|------|
| k_turns | 4 | 999（等价于全 session） |
| LLM 调用次数 | 179 batches | 32 batches（按 trigger 实际调用） |
| 输入上下文 | 4 turns | 整个 session（14-37 turns） |
| 样本 | conv-41 | conv-41 |
| 模型 | qwen3-4b | qwen3-4b |

## 改动

仅 config 变更，代码无改动：`k_turns: 999`

## 运行

```bash
python scripts/build_memory.py --exp-dir experiments/2026-04-28-012-session-level-conv41
```

## 运行环境
- commit: `85af0c8`
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

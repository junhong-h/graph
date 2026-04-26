"""Summarize GraphMemory QA retrieval traces.

Example:
    python scripts/summarize_retrieval_traces.py \
        --input runs/refine_final_conv41/qa_results.jsonl \
        --output runs/refine_final_conv41/retrieval_trace_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def summarize_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    actions: Counter[str] = Counter()
    sample_actions: Counter[str] = Counter()
    forced_reasons: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    trace_steps = 0

    for record in records:
        total += 1
        category = str(record.get("category", ""))
        traces = record.get("traces") or []
        trace_steps += len(traces)

        seen_actions = {str(trace.get("action", "")) for trace in traces}
        for trace in traces:
            action = str(trace.get("action", ""))
            if not action:
                continue
            actions[action] += 1
            by_category[category][action] += 1
            if action == "forced_finish":
                reason = str(trace.get("reason", "unknown"))
                forced_reasons[reason] += 1
                by_category[category][f"forced_reason:{reason}"] += 1

        if "forced_finish" in seen_actions:
            sample_actions["forced_finish"] += 1
            by_category[category]["samples_with_forced_finish"] += 1
        if "frontier_exhausted" in seen_actions:
            sample_actions["frontier_exhausted"] += 1
            by_category[category]["samples_with_frontier_exhausted"] += 1
        if "raw_fallback" in seen_actions:
            sample_actions["raw_fallback"] += 1
            by_category[category]["samples_with_raw_fallback"] += 1

    return {
        "total": total,
        "avg_trace_steps": trace_steps / total if total else 0.0,
        "actions": dict(actions),
        "forced_reasons": dict(forced_reasons),
        "sample_rates": {
            "forced_finish": _rate(sample_actions.get("forced_finish", 0), total),
            "frontier_exhausted": _rate(sample_actions.get("frontier_exhausted", 0), total),
            "raw_fallback": _rate(sample_actions.get("raw_fallback", 0), total),
        },
        "by_category": {
            category: dict(counter)
            for category, counter in sorted(by_category.items())
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to qa_results.jsonl")
    parser.add_argument("--output", default="", help="Optional summary JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_records(load_jsonl(Path(args.input)))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

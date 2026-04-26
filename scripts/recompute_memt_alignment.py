"""Recompute LoCoMo QA metrics under the Mem-T evaluation protocol.

The Mem-T CLI currently imports vLLM before it reaches --metrics-only in this
environment, so this script mirrors the metrics-only path in mem-t/llm_judge.py:

- LoCoMo split: train=chat_data[0], valid=chat_data[1], test=chat_data[2:].
- Categories: Cat1-4 by default, excluding Cat5.
- Metrics: token-level F1 and BLEU-1 with the same normalization as Mem-T.

No model calls are made.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CATEGORIES = ("1", "2", "3", "4")


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens(s: str) -> list[str]:
    normalized = normalize_text(s)
    return normalized.split() if normalized else []


def f1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks) / len(ptoks))
    else:
        bp = 0.0
    return bp * precision


def parse_named_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--input must be formatted as NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise argparse.ArgumentTypeError("input NAME cannot be empty")
    if not path:
        raise argparse.ArgumentTypeError("input PATH cannot be empty")
    return name, Path(path)


def load_memt_test_ids(dataset_path: Path) -> list[str]:
    with dataset_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [str(sample.get("sample_id", "")) for sample in data[2:]]


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    pred = str(result.get("pred", ""))
    gold = result.get("gold", "")
    if isinstance(gold, list):
        f1_values = [f1_score(pred, str(g)) for g in gold]
        bleu_values = [bleu1_score(pred, str(g)) for g in gold]
        result["f1_score"] = max(f1_values) if f1_values else 0.0
        result["bleu1_score"] = max(bleu_values) if bleu_values else 0.0
    else:
        result["f1_score"] = f1_score(pred, str(gold))
        result["bleu1_score"] = bleu1_score(pred, str(gold))
    result["judge_reasoning"] = "Skipped (Metrics Only Mode)"
    result["judge_label"] = "SKIPPED"
    return result


def empty_stats() -> dict[str, Any]:
    return {
        "total": 0,
        "f1_scores": [],
        "bleu1_scores": [],
        "zero_f1": 0,
        "exact_f1": 0,
        "pred_tokens": [],
        "gold_tokens": [],
    }


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    total = stats["total"]
    f1_scores = stats["f1_scores"]
    bleu1_scores = stats["bleu1_scores"]
    pred_tokens = stats["pred_tokens"]
    gold_tokens = stats["gold_tokens"]
    avg_pred_tokens = sum(pred_tokens) / len(pred_tokens) if pred_tokens else 0.0
    avg_gold_tokens = sum(gold_tokens) / len(gold_tokens) if gold_tokens else 0.0
    length_ratio = avg_pred_tokens / avg_gold_tokens if avg_gold_tokens else 0.0
    return {
        "total": total,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_bleu1": sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        "zero_f1": stats["zero_f1"],
        "zero_f1_rate": stats["zero_f1"] / total if total else 0.0,
        "exact_f1": stats["exact_f1"],
        "exact_f1_rate": stats["exact_f1"] / total if total else 0.0,
        "avg_pred_tokens": avg_pred_tokens,
        "avg_gold_tokens": avg_gold_tokens,
        "pred_gold_token_ratio": length_ratio,
    }


def update_stats(stats: dict[str, Any], record: dict[str, Any]) -> None:
    stats["total"] += 1
    f1 = float(record.get("f1_score", 0.0))
    bleu1 = float(record.get("bleu1_score", 0.0))
    stats["f1_scores"].append(f1)
    stats["bleu1_scores"].append(bleu1)
    if f1 == 0.0:
        stats["zero_f1"] += 1
    if f1 == 1.0:
        stats["exact_f1"] += 1
    pred = str(record.get("pred", ""))
    gold = record.get("gold", "")
    if isinstance(gold, list):
        gold_text = str(gold[0]) if gold else ""
    else:
        gold_text = str(gold)
    stats["pred_tokens"].append(len(tokens(pred)))
    stats["gold_tokens"].append(len(tokens(gold_text)))


def recompute_one(
    name: str,
    input_path: Path,
    out_dir: Path,
    test_ids: set[str],
    categories: set[str],
) -> dict[str, Any]:
    filtered_input_path = out_dir / f"{name}_memt_split_cat1_4_input.jsonl"
    metrics_path = out_dir / f"{name}_memt_metrics.jsonl"

    by_category: dict[str, dict[str, Any]] = {}
    by_sample: dict[str, dict[str, Any]] = {}
    overall = empty_stats()
    source_total = 0
    kept_total = 0

    with (
        input_path.open(encoding="utf-8") as f_in,
        filtered_input_path.open("w", encoding="utf-8") as f_filtered,
        metrics_path.open("w", encoding="utf-8") as f_metrics,
    ):
        for line in f_in:
            if not line.strip():
                continue
            source_total += 1
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            category = str(record.get("category", ""))
            if sample_id not in test_ids or category not in categories:
                continue

            f_filtered.write(json.dumps(record, ensure_ascii=False) + "\n")
            scored = score_record(record)
            f_metrics.write(json.dumps(scored, ensure_ascii=False) + "\n")
            kept_total += 1

            cat_stats = by_category.setdefault(category, empty_stats())
            sample_stats = by_sample.setdefault(sample_id, empty_stats())
            update_stats(cat_stats, scored)
            update_stats(sample_stats, scored)
            update_stats(overall, scored)

    return {
        "input_path": str(input_path),
        "filtered_input_path": str(filtered_input_path),
        "metrics_path": str(metrics_path),
        "source_total": source_total,
        "kept_total": kept_total,
        "overall": finalize_stats(overall),
        "by_category": {
            key: finalize_stats(by_category[key])
            for key in sorted(by_category, key=lambda x: int(x) if x.isdigit() else x)
        },
        "by_sample": {
            key: finalize_stats(by_sample[key])
            for key in sorted(by_sample)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute GraphMemory QA metrics under Mem-T's Locomo protocol."
    )
    parser.add_argument("--dataset", default="data/locomo/locomo10.json")
    parser.add_argument("--out-dir", default="artifacts/memt_alignment")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_named_input,
        help="Input result as NAME=PATH. Repeat for multiple runs.",
    )
    parser.add_argument("--categories", nargs="*", default=list(DEFAULT_CATEGORIES))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_ids = load_memt_test_ids(dataset_path)
    categories = {str(cat) for cat in args.categories}

    summary = {
        "protocol": {
            "dataset_path": str(dataset_path),
            "test_split": "chat_data[2:]",
            "test_sample_ids": test_ids,
            "categories": sorted(categories, key=lambda x: int(x) if x.isdigit() else x),
            "excluded_categories": ["5"],
        },
        "runs": {},
    }

    for name, input_path in args.input:
        summary["runs"][name] = recompute_one(
            name=name,
            input_path=input_path,
            out_dir=out_dir,
            test_ids=set(test_ids),
            categories=categories,
        )

    summary_path = out_dir / "memt_alignment_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {summary_path}")
    for name, run in summary["runs"].items():
        overall = run["overall"]
        print(
            f"{name}: n={overall['total']} "
            f"avg_f1={overall['avg_f1']:.4f} "
            f"avg_bleu1={overall['avg_bleu1']:.4f}"
        )


if __name__ == "__main__":
    main()

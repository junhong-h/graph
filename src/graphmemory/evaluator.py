"""Evaluation utilities: F1, BLEU-1, and LLM-as-a-Judge for QA results."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from graphmemory.llm_client import LLMClient
from graphmemory.qa_filters import (
    normalize_filter_values,
    record_matches_filters,
)


# ---------------------------------------------------------------------------
# String-level metrics
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    return s.strip()


def _tokens(s: str) -> List[str]:
    s = _normalize(s)
    return s.split() if s else []


def f1_score(pred: str, gold: str) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum(min(pc[t], gc[t]) for t in pc)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pt)
    recall    = overlap / len(gt)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    pt, gt = _tokens(pred), _tokens(gold)
    if not pt:
        return 0.0
    pc, gc = Counter(pt), Counter(gt)
    clipped   = sum(min(pc[t], gc[t]) for t in pc)
    precision = clipped / len(pt)
    bp = 1.0 if len(pt) >= len(gt) else math.exp(1 - len(gt) / len(pt))
    return bp * precision


# ---------------------------------------------------------------------------
# LLM judge prompts
# ---------------------------------------------------------------------------

_LOCOMO_SYSTEM = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You are given: (1) a question, (2) a gold answer, (3) a generated answer.

Grading Rules:
1. Be generous. If the generated answer touches on the same topic as the gold answer, mark CORRECT.
2. For time-related questions, if it refers to the same time period, mark CORRECT.
3. Ignore minor formatting differences.

Output ONLY a JSON object with keys "reasoning" and "label".
Example: {"reasoning": "The answer identifies the correct item.", "label": "CORRECT"}\
"""

_LOCOMO_USER = """\
Question: {question}
Gold answer: {gold}
Generated answer: {pred}

Provide one sentence reasoning, then conclude with CORRECT or WRONG.\
"""

# Cat5: adversarial (unanswerable) questions — evaluate refusal, not content
_CAT5_SYSTEM = """\
Your task is to evaluate whether a system correctly handled an UNANSWERABLE question.

The question asks about an event or fact that NEVER appeared in the conversation.
The correct behavior is to REFUSE to answer or say it was not mentioned.

Label as 'CORRECT' if the generated answer:
- Says the event/fact was not mentioned, not found, or didn't happen
- Expresses inability to answer due to lack of evidence
- Uses phrases like "not mentioned", "no evidence", "I don't know", "didn't happen"

Label as 'WRONG' if the generated answer:
- Provides a specific factual answer (hallucination)
- Makes up plausible-sounding content about the question

Output ONLY a JSON object with keys "reasoning" and "label".
Example: {"reasoning": "The answer correctly states the information is not in the conversation.", "label": "CORRECT"}\
"""

_CAT5_USER = """\
Question: {question}
Generated answer: {pred}

Did the system correctly refuse to answer this unanswerable question?
Provide one sentence reasoning, then conclude with CORRECT or WRONG.\
"""


def _parse_judge_output(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    label = "CORRECT" if "CORRECT" in text.upper() and "WRONG" not in text.upper() else "WRONG"
    return {"reasoning": text, "label": label, "parse_error": True}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Compute F1/BLEU-1 and optionally run LLM-as-a-Judge on QA results.

    Input JSONL format (one record per QA pair):
        {"sample_id": ..., "question": ..., "pred": ..., "gold": ..., "category": ...}

    Output JSONL adds: f1_score, bleu1_score, judge_label, judge_reasoning
    """

    def __init__(self, llm: Optional[LLMClient] = None, benchmark: str = "locomo"):
        self.llm       = llm
        self.benchmark = benchmark

    def evaluate_one(self, record: Dict) -> Dict:
        pred = str(record.get("pred", ""))
        gold = record.get("gold", "")

        # Handle multi-reference gold
        if isinstance(gold, list):
            record["f1_score"]   = max((f1_score(pred, str(g))   for g in gold), default=0.0)
            record["bleu1_score"] = max((bleu1_score(pred, str(g)) for g in gold), default=0.0)
            gold_display = " OR ".join(str(g) for g in gold)
        else:
            record["f1_score"]   = f1_score(pred, str(gold))
            record["bleu1_score"] = bleu1_score(pred, str(gold))
            gold_display = str(gold)

        if self.llm is None:
            record["judge_label"]     = "SKIPPED"
            record["judge_reasoning"] = "LLM judge not configured."
            return record

        # Cat5 adversarial questions use a refusal-detection prompt
        is_cat5 = str(record.get("category", "")) == "5"
        if is_cat5:
            messages = [
                {"role": "system", "content": _CAT5_SYSTEM},
                {"role": "user",   "content": _CAT5_USER.format(
                    question=record.get("question", ""),
                    pred=pred,
                )},
            ]
        else:
            messages = [
                {"role": "system", "content": _LOCOMO_SYSTEM},
                {"role": "user",   "content": _LOCOMO_USER.format(
                    question=record.get("question", ""),
                    gold=gold_display,
                    pred=pred,
                )},
            ]
        try:
            response = self.llm.complete(messages, json_mode=True)
            result   = _parse_judge_output(response)
            record["judge_label"]     = result.get("label", "WRONG")
            record["judge_reasoning"] = result.get("reasoning", "")
            if result.get("parse_error"):
                record["judge_parse_error"] = True
        except Exception as exc:
            record["judge_label"] = "ERROR"
            record["judge_error"] = str(exc)

        return record

    def evaluate_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        workers: int = 4,
        sample_ids: Optional[Iterable[Any]] = None,
        include_categories: Optional[Iterable[Any]] = None,
        exclude_categories: Optional[Iterable[Any]] = None,
    ) -> Dict:
        """Evaluate all records in input_path and write results to output_path.

        Returns summary statistics dict.
        """
        input_path  = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_ids = normalize_filter_values(sample_ids)
        include_categories = normalize_filter_values(include_categories)
        exclude_categories = normalize_filter_values(exclude_categories)

        # Resume: skip already-evaluated records
        done_ids: set = set()
        if output_path.exists():
            with output_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        if not record_matches_filters(
                            r,
                            sample_ids=sample_ids,
                            include_categories=include_categories,
                            exclude_categories=exclude_categories,
                        ):
                            continue
                        uid = r.get("qa_id") or r.get("sample_id")
                        if uid:
                            done_ids.add(uid)
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Resuming: {len(done_ids)} records already evaluated.")

        tasks = []
        with input_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not record_matches_filters(
                        r,
                        sample_ids=sample_ids,
                        include_categories=include_categories,
                        exclude_categories=exclude_categories,
                    ):
                        continue
                    uid = r.get("qa_id") or r.get("sample_id")
                    if uid and uid not in done_ids:
                        tasks.append(r)
                except json.JSONDecodeError:
                    continue

        if not tasks:
            logger.info("All records already evaluated.")
        else:
            logger.info(f"Evaluating {len(tasks)} records with {workers} workers…")
            with output_path.open("a", encoding="utf-8") as f_out:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(self.evaluate_one, t): t for t in tasks}
                    for fut in as_completed(futures):
                        try:
                            result = fut.result(timeout=60)
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f_out.flush()
                        except Exception as exc:
                            orig = futures[fut]
                            logger.error(f"Eval failed for {orig.get('qa_id')}: {exc}")

        if not output_path.exists():
            output_path.touch()

        return self.compute_stats(
            output_path,
            sample_ids=sample_ids,
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stats(
        output_path: str | Path,
        *,
        sample_ids: Optional[Iterable[Any]] = None,
        include_categories: Optional[Iterable[Any]] = None,
        exclude_categories: Optional[Iterable[Any]] = None,
    ) -> Dict:
        output_path = Path(output_path)
        sample_ids = normalize_filter_values(sample_ids)
        include_categories = normalize_filter_values(include_categories)
        exclude_categories = normalize_filter_values(exclude_categories)
        by_cat: Dict[str, Dict] = {}

        if not output_path.exists():
            output_path.touch()

        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not record_matches_filters(
                    r,
                    sample_ids=sample_ids,
                    include_categories=include_categories,
                    exclude_categories=exclude_categories,
                ):
                    continue
                cat = str(r.get("category", "unknown"))
                if cat not in by_cat:
                    by_cat[cat] = {"correct": 0, "wrong": 0, "skipped": 0,
                                   "f1": [], "bleu1": [], "total": 0}
                s = by_cat[cat]
                s["total"] += 1
                label = r.get("judge_label", "").upper()
                if label == "CORRECT":
                    s["correct"] += 1
                elif label == "WRONG":
                    s["wrong"] += 1
                else:
                    s["skipped"] += 1
                if "f1_score" in r:
                    s["f1"].append(r["f1_score"])
                if "bleu1_score" in r:
                    s["bleu1"].append(r["bleu1_score"])

        # Print table
        print("\n" + "=" * 72)
        print(f"{'Category':<16} | {'Total':>6} | {'Acc':>8} | {'Avg F1':>8} | {'BLEU-1':>8}")
        print("-" * 72)

        all_f1, all_bleu, all_correct, all_judged = [], [], 0, 0
        for cat in sorted(by_cat, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x)):
            s = by_cat[cat]
            judged = s["correct"] + s["wrong"]
            acc    = f"{s['correct'] / judged * 100:.1f}%" if judged else "N/A"
            avg_f1   = sum(s["f1"])   / len(s["f1"])   if s["f1"]   else 0.0
            avg_bleu = sum(s["bleu1"]) / len(s["bleu1"]) if s["bleu1"] else 0.0
            print(f"{cat:<16} | {s['total']:>6} | {acc:>8} | {avg_f1:>8.4f} | {avg_bleu:>8.4f}")
            all_f1.extend(s["f1"])
            all_bleu.extend(s["bleu1"])
            all_correct += s["correct"]
            all_judged  += judged

        total = sum(s["total"] for s in by_cat.values())
        overall_acc  = f"{all_correct / all_judged * 100:.1f}%" if all_judged else "N/A"
        overall_f1   = sum(all_f1)   / len(all_f1)   if all_f1   else 0.0
        overall_bleu = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
        print("-" * 72)
        print(f"{'Overall':<16} | {total:>6} | {overall_acc:>8} | {overall_f1:>8.4f} | {overall_bleu:>8.4f}")
        print("=" * 72)

        return {
            "total": total, "accuracy": overall_acc,
            "avg_f1": overall_f1, "avg_bleu1": overall_bleu,
            "by_category": by_cat,
        }

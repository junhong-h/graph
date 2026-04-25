"""Oracle QA: answer questions using ground-truth evidence turns directly.

This establishes an upper-bound baseline by bypassing the retrieval pipeline
entirely. For each question, the annotated evidence turns from the dataset are
fed as context to the LLM.

Gap analysis:
  Oracle Acc  - System Acc  =  retrieval loss
  100%        - Oracle Acc  =  reasoning/generation loss

Usage:
    python scripts/run_oracle_qa.py --config configs/run_qa_dashscope.yaml
    python scripts/run_oracle_qa.py --config configs/run_qa_dashscope.yaml --sample-ids conv-30
    python scripts/run_oracle_qa.py --config configs/run_qa_dashscope.yaml --max-qa 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphmemory.config import BuildConfig
from graphmemory.evaluator import Evaluator
from graphmemory.graph_retrieval import get_answer_format
from graphmemory.llm_client import OpenAIClient


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------

def _parse_dia_id(dia_id: str) -> Tuple[int, str]:
    """Parse 'D3:7' → (session_index=3, full_id='D3:7')."""
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if not m:
        raise ValueError(f"Unrecognised dia_id format: {dia_id!r}")
    return int(m.group(1)), dia_id


def _expand_dia_ids(raw_ids: List[str]) -> List[str]:
    """Handle semicolon-joined dia_ids like 'D8:6; D9:17' in a single string."""
    expanded = []
    for item in raw_ids:
        for part in re.split(r"[;,]\s*", item):
            part = part.strip()
            if re.match(r"D\d+:\d+", part):
                expanded.append(part)
    return expanded


def resolve_evidence(conv: Dict, dia_ids: List[str]) -> List[Dict]:
    """Return list of {dia_id, speaker, text, session_time} for each dia_id."""
    results = []
    for dia_id in _expand_dia_ids(dia_ids):
        try:
            session_idx, _ = _parse_dia_id(dia_id)
        except ValueError:
            logger.warning(f"Skipping malformed dia_id: {dia_id!r}")
            continue
        session_key  = f"session_{session_idx}"
        session_time = conv.get(f"session_{session_idx}_date_time", "")
        turns = conv.get(session_key, [])
        found = False
        for turn in turns:
            if turn.get("dia_id") == dia_id:
                results.append({
                    "dia_id":       dia_id,
                    "speaker":      turn.get("speaker", ""),
                    "text":         turn.get("text", ""),
                    "session_time": session_time,
                })
                found = True
                break
        if not found:
            logger.warning(f"dia_id {dia_id!r} not found in conversation.")
    return results


def format_oracle_context(evidence_turns: List[Dict]) -> str:
    """Format resolved turns as a readable context block."""
    if not evidence_turns:
        return "(no evidence)"
    lines = []
    for e in evidence_turns:
        ts = f" [{e['session_time']}]" if e["session_time"] else ""
        lines.append(f"{e['dia_id']}{ts} | {e['speaker']}: {e['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Oracle answering
# ---------------------------------------------------------------------------

_ORACLE_SYSTEM = """\
You are a Memory Assistant. Answer the question based ONLY on the evidence turns provided.
{answer_format}
Do NOT say "I don't know" or "Unknown" — always give your best answer from the evidence.\
"""

_ORACLE_USER = """\
[Evidence]
{context}

[Question]
{question}\
"""


def oracle_answer(
    question: str,
    context: str,
    category: str,
    llm: OpenAIClient,
    benchmark: str = "locomo",
) -> str:
    answer_format = get_answer_format(benchmark, category)
    system = _ORACLE_SYSTEM.format(answer_format=answer_format)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": _ORACLE_USER.format(
            context=context,
            question=question,
        )},
    ]
    return llm.complete(messages).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/run_qa_dashscope.yaml")
    p.add_argument("--sample-ids",  nargs="*", default=None)
    p.add_argument("--max-qa",      type=int, default=None)
    p.add_argument("--metrics-only", action="store_true")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    cfg     = BuildConfig.from_yaml(args.config)
    run_dir = Path("runs/qa_oracle")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "oracle_qa.log", level="DEBUG", rotation="20 MB")

    # Load raw dataset (need original dia_id structure)
    with open(cfg.data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Filter samples
    if args.sample_ids:
        id_set   = set(args.sample_ids)
        raw_data = [s for s in raw_data if s.get("sample_id") in id_set]

    logger.info(f"Running Oracle QA on {len(raw_data)} samples.")

    llm = OpenAIClient(
        model                  = cfg.llm.model,
        api_key                = cfg.llm.api_key or None,
        base_url               = cfg.llm.base_url or None,
        temperature            = cfg.llm.temperature,
        max_retries            = cfg.llm.max_retries,
        disable_thinking       = cfg.llm.disable_thinking,
        use_extra_body_thinking= cfg.llm.use_extra_body_thinking,
    )

    results_path = run_dir / "oracle_results.jsonl"

    # Resume
    done_ids: set = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("qa_id"):
                        done_ids.add(r["qa_id"])
                except json.JSONDecodeError:
                    continue
        if done_ids:
            logger.info(f"Resuming: {len(done_ids)} already answered.")

    with results_path.open("a", encoding="utf-8") as f_out:
        for sample in raw_data:
            conv    = sample["conversation"]
            qa_list = sample.get("qa", [])
            sid     = _get_sample_id(sample)

            if not qa_list:
                continue

            logger.info(f"Sample {sid}: {len(qa_list)} questions")

            questions = qa_list[:args.max_qa] if args.max_qa else qa_list

            for qi, qa in enumerate(questions):
                qa_id = f"{sid}_q{qi}"
                if qa_id in done_ids:
                    continue

                question = qa.get("question", "")
                gold     = qa.get("answer", "")
                category = str(qa.get("category", ""))
                dia_ids  = qa.get("evidence", [])

                # Resolve evidence turns
                evidence_turns = resolve_evidence(conv, dia_ids)
                context        = format_oracle_context(evidence_turns)

                logger.debug(f"  Q{qi} [cat{category}]: {question}")
                logger.debug(f"  Evidence: {dia_ids} → {len(evidence_turns)} turns")

                try:
                    pred = oracle_answer(question, context, category, llm, cfg.dataset_name)
                except Exception as exc:
                    logger.error(f"  Oracle failed for {qa_id}: {exc}", exc_info=True)
                    pred = ""

                logger.debug(f"  → pred: {pred!r}  |  gold: {gold!r}")

                record = {
                    "qa_id":     qa_id,
                    "sample_id": sid,
                    "question":  question,
                    "gold":      gold,
                    "pred":      pred,
                    "category":  category,
                    "evidence":  dia_ids,
                    "context":   context,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

    logger.info(f"Oracle retrieval done. Results: {results_path}")

    eval_path = run_dir / "oracle_results_eval.jsonl"
    judge_llm = None if args.metrics_only else llm
    evaluator = Evaluator(llm=judge_llm, benchmark=cfg.dataset_name)
    evaluator.evaluate_file(results_path, eval_path, workers=4)


def _get_sample_id(sample: Dict) -> str:
    """Return sample_id from raw LoCoMo data (top-level field)."""
    return sample.get("sample_id", "unknown")


if __name__ == "__main__":
    main()

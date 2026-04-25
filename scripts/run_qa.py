"""Run graph-memory retrieval on QA questions and evaluate results.

Usage:
    # First-phase: 5 questions, human inspection
    python scripts/run_qa.py --config configs/run_qa.yaml \
        --sample-ids conv-26 --max-qa 5 --log-level DEBUG

    # Full run, metrics only (no LLM judge)
    python scripts/run_qa.py --config configs/run_qa.yaml \
        --sample-ids conv-26 --metrics-only

    # Full run with LLM judge
    python scripts/run_qa.py --config configs/run_qa.yaml --sample-ids conv-26

Outputs:
    runs/qa/qa_results.jsonl      — one record per QA pair
    runs/qa/qa_results_eval.jsonl — same + F1/BLEU (+ judge if enabled)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphmemory.config import BuildConfig
from graphmemory.dataset import load_locomo_sessions
from graphmemory.evaluator import Evaluator
from graphmemory.graph_localize import GraphLocalizer
from graphmemory.graph_retrieval import GraphRetriever
from graphmemory.graph_store import GraphStore
from graphmemory.llm_client import OpenAIClient
from graphmemory.raw_archive import RawArchive
from graphmemory.vector_store import ChromaStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/run_qa.yaml")
    p.add_argument("--sample-ids",  nargs="*", default=None,
                   help="Only process these sample IDs")
    p.add_argument("--max-qa",      type=int, default=None,
                   help="Max QA questions per sample (for quick testing)")
    p.add_argument("--metrics-only", action="store_true",
                   help="Skip LLM judge; only compute F1/BLEU")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    cfg     = BuildConfig.from_yaml(args.config)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = Path(cfg.graph_dir) if cfg.graph_dir else Path(cfg.run_dir).parent / "build" / "graphs"
    logger.add(run_dir / "qa.log", level="DEBUG", rotation="20 MB")

    # Load dataset
    samples = load_locomo_sessions(cfg.data_path)
    if args.sample_ids:
        id_set  = set(args.sample_ids)
        samples = [s for s in samples
                   if s["conversation"][0]["metadata"].get("sample_id") in id_set]
    logger.info(f"Running QA on {len(samples)} samples.")

    # Shared components
    llm   = OpenAIClient(
        model       = cfg.llm.model,
        api_key     = cfg.llm.api_key  or None,
        base_url    = cfg.llm.base_url or None,
        temperature = cfg.llm.temperature,
        max_retries = cfg.llm.max_retries,
        reasoning_effort = cfg.llm.reasoning_effort,
        disable_thinking = cfg.llm.disable_thinking,
        use_extra_body_thinking = cfg.llm.use_extra_body_thinking,
    )
    store = ChromaStore(path=cfg.vector_store.path, from_scratch=False)

    # Resume: skip already-answered QA
    results_path = run_dir / "qa_results.jsonl"
    done_qa_ids: set = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("qa_id"):
                        done_qa_ids.add(r["qa_id"])
                except json.JSONDecodeError:
                    continue
        if done_qa_ids:
            logger.info(f"Resuming: {len(done_qa_ids)} QA pairs already answered.")

    # Retrieve
    with results_path.open("a", encoding="utf-8") as f_out:
        for idx, sample in enumerate(samples):
            sessions  = sample.get("conversation", [])
            qa_list   = sample.get("qa", [])
            if not sessions or not qa_list:
                continue

            sample_id = sessions[0]["metadata"].get("sample_id", f"sample_{idx}")
            graph_path = graphs_dir / f"{sample_id}_graph.json"

            if not graph_path.exists():
                logger.error(
                    f"Graph not found for {sample_id}: {graph_path}. "
                    "Run build_memory.py first."
                )
                continue

            logger.info(f"[{idx+1}/{len(samples)}] Sample {sample_id}: "
                        f"{len(qa_list)} questions")

            # Per-sample retrieval components
            graph     = GraphStore(graph_path, store, sample_id)
            archive   = RawArchive(store, sample_id)
            localizer = GraphLocalizer(
                graph,
                seed_top_k = cfg.graph.seed_top_k,
                max_hops   = cfg.graph.max_hops,
                max_nodes  = cfg.graph.max_nodes,
                max_edges  = cfg.graph.max_edges,
            )
            retriever = GraphRetriever(
                graph          = graph,
                archive        = archive,
                localizer      = localizer,
                llm            = llm,
                retrieval_topk = cfg.memory.retrieval_topk,
                max_hop        = cfg.graph.retrieval_max_hop,
                jump_budget    = cfg.graph.jump_budget,
                benchmark      = cfg.dataset_name,
            )

            # Limit questions if requested
            questions = qa_list
            if args.max_qa is not None:
                questions = qa_list[: args.max_qa]

            for qi, qa in enumerate(questions):
                qa_id    = f"{sample_id}_q{qi}"
                if qa_id in done_qa_ids:
                    continue

                question = qa.get("question", "")
                gold     = qa.get("answer",   "")
                category = str(qa.get("category", ""))

                logger.debug(f"  Q{qi} [cat{category}]: {question}")
                try:
                    result = retriever.answer(question, category=category)
                    pred   = result["answer"]
                    traces = result["traces"]
                except Exception as exc:
                    logger.error(f"  Retrieval failed for {qa_id}: {exc}", exc_info=True)
                    pred, traces = "", []

                logger.debug(f"  → pred: {pred!r}  |  gold: {gold!r}")

                record = {
                    "qa_id":     qa_id,
                    "sample_id": sample_id,
                    "question":  question,
                    "gold":      gold,
                    "pred":      pred,
                    "category":  category,
                    "evidence":  qa.get("evidence", []),
                    "traces":    traces,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

    logger.info(f"Retrieval done. Results: {results_path}")

    # Evaluate
    eval_path = run_dir / "qa_results_eval.jsonl"
    judge_llm = None if args.metrics_only else llm
    evaluator = Evaluator(llm=judge_llm, benchmark=cfg.dataset_name)
    evaluator.evaluate_file(results_path, eval_path, workers=4)


if __name__ == "__main__":
    main()

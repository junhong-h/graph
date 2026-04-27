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

    # LoCoMo protocol-aligned run: Cat1-4 only, skip first 5 samples
    python scripts/run_qa.py --config configs/run_qa.yaml \
        --locomo-cat1-4 --skip-samples 5

Outputs:
    runs/qa/qa_results.jsonl      — one record per QA pair
    runs/qa/qa_results_eval.jsonl — same + F1/BLEU (+ judge if enabled)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from graphmemory.qa_filters import (
    filter_samples,
    iter_filtered_qa,
    normalize_filter_values,
    resolve_include_categories,
    sample_id_from_sample,
)
from graphmemory.raw_archive import RawArchive
from graphmemory.vector_store import ChromaStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default=None,
                   help="Experiment directory (experiments/<id>-<name>/). "
                        "Loads config.yaml from there and writes outputs under exp-dir/qa/.")
    p.add_argument("--config",      default="configs/run_qa.yaml",
                   help="Ignored when --exp-dir is set.")
    p.add_argument("--run-dir", default=None,
                   help="Override output run directory (ignored when --exp-dir is set)")
    p.add_argument("--sample-ids",  nargs="*", default=None,
                   help="Only process these sample IDs (overrides config sample_ids)")
    p.add_argument("--skip-samples", type=int, default=0,
                   help="Skip the first N selected samples in dataset order")
    p.add_argument("--limit-samples", type=int, default=None,
                   help="Stop after this many selected samples")
    p.add_argument("--categories", nargs="*", default=None,
                   help="Only process these QA categories, e.g. --categories 1 2 3 4")
    p.add_argument("--exclude-categories", nargs="*", default=None,
                   help="Exclude these QA categories, e.g. --exclude-categories 5")
    p.add_argument("--locomo-cat1-4", action="store_true",
                   help="Shortcut for the LoCoMo Cat1-4 evaluation protocol")
    p.add_argument("--max-qa",      type=int, default=None,
                   help="Max filtered QA questions per sample (for quick testing)")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for question retrieval (default: 1)")
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

    if args.exp_dir:
        cfg = BuildConfig.from_exp_dir(args.exp_dir, mode="qa")
    else:
        cfg = BuildConfig.from_yaml(args.config)
        if args.run_dir:
            cfg.run_dir = args.run_dir
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = Path(cfg.graph_dir) if cfg.graph_dir else Path(cfg.run_dir).parent / "build" / "graphs"
    logger.add(run_dir / "qa.log", level="DEBUG", rotation="20 MB")

    # Load dataset
    include_categories = resolve_include_categories(
        args.categories,
        locomo_cat1_4=args.locomo_cat1_4,
    )
    exclude_categories = normalize_filter_values(args.exclude_categories)
    sample_ids = args.sample_ids or cfg.sample_ids or None
    samples = filter_samples(
        load_locomo_sessions(cfg.data_path),
        sample_ids=sample_ids,
        skip_first=args.skip_samples,
        limit=args.limit_samples,
    )
    selected_sample_ids = [sample_id_from_sample(s) for s in samples]
    if not selected_sample_ids:
        logger.warning("No samples selected; nothing to run.")
        return
    logger.info(
        f"Running QA on {len(samples)} samples; "
        f"include_categories={sorted(include_categories) if include_categories else 'all'}, "
        f"exclude_categories={sorted(exclude_categories) if exclude_categories else 'none'}."
    )

    # Shared components
    llm   = OpenAIClient(
        model       = cfg.llm.model,
        api_key     = cfg.llm.api_key  or None,
        base_url    = cfg.llm.base_url or None,
        temperature = cfg.llm.temperature,
        top_p       = cfg.llm.top_p,
        seed        = cfg.llm.seed,
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
    write_lock = threading.Lock()

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

            # Per-sample retrieval components (read-only during retrieval — thread-safe)
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
                final_answer_compression = cfg.graph.final_answer_compression,
            )

            questions = [
                (qi, qa) for qi, qa in iter_filtered_qa(
                    qa_list,
                    include_categories=include_categories,
                    exclude_categories=exclude_categories,
                    max_items=args.max_qa,
                )
                if f"{sample_id}_q{qi}" not in done_qa_ids
            ]
            if not questions:
                logger.info(f"Sample {sample_id}: no QA matched filters (or all done).")
                continue

            logger.info(f"[{idx+1}/{len(samples)}] Sample {sample_id}: "
                        f"{len(questions)} questions (workers={args.workers})")

            def _answer_one(item: tuple) -> dict:
                qi, qa = item
                question = qa.get("question", "")
                gold     = qa.get("answer",   "")
                category = str(qa.get("category", ""))
                qa_id    = f"{sample_id}_q{qi}"
                logger.debug(f"  Q{qi} [cat{category}]: {question}")
                try:
                    result = retriever.answer(question, category=category)
                    pred   = result["answer"]
                    traces = result["traces"]
                except Exception as exc:
                    logger.error(f"  Retrieval failed for {qa_id}: {exc}", exc_info=True)
                    pred, traces = "", []
                logger.debug(f"  → pred: {pred!r}  |  gold: {gold!r}")
                return {
                    "qa_id":     qa_id,
                    "sample_id": sample_id,
                    "question":  question,
                    "gold":      gold,
                    "pred":      pred,
                    "category":  category,
                    "evidence":  qa.get("evidence", []),
                    "traces":    traces,
                }

            if args.workers <= 1:
                for item in questions:
                    record = _answer_one(item)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {executor.submit(_answer_one, item): item for item in questions}
                    done_count = 0
                    for fut in as_completed(futures):
                        try:
                            record = fut.result()
                        except Exception as exc:
                            logger.error(f"Worker failed: {exc}", exc_info=True)
                            continue
                        with write_lock:
                            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            f_out.flush()
                        done_count += 1
                        if done_count % 20 == 0:
                            logger.info(f"  {sample_id}: {done_count}/{len(questions)} done")

    logger.info(f"Retrieval done. Results: {results_path}")

    # Evaluate
    eval_path = run_dir / "qa_results_eval.jsonl"
    judge_llm = None if args.metrics_only else llm
    evaluator = Evaluator(llm=judge_llm, benchmark=cfg.dataset_name)
    summary = evaluator.evaluate_file(
        results_path,
        eval_path,
        workers=4,
        sample_ids=selected_sample_ids,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
    )
    metrics_path = run_dir / "qa_metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Metrics written: {metrics_path}")


if __name__ == "__main__":
    main()

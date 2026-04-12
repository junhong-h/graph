"""Build graph memory for a dataset sample-by-sample.

Usage:
    python scripts/build_memory.py --config configs/build_memory.yaml
    python scripts/build_memory.py --config configs/build_memory.yaml --limit 1
    python scripts/build_memory.py --config configs/build_memory.yaml --sample-ids conv-26
    python scripts/build_memory.py --config configs/build_memory.yaml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphmemory.config import BuildConfig
from graphmemory.dataset import load_locomo_sessions
from graphmemory.graph_builder import GraphBuilder
from graphmemory.graph_construction import GraphConstructor
from graphmemory.graph_localize import GraphLocalizer
from graphmemory.graph_store import GraphStore
from graphmemory.graph_trigger import GraphTrigger
from graphmemory.llm_client import OpenAIClient
from graphmemory.raw_archive import RawArchive
from graphmemory.vector_store import ChromaStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/build_memory.yaml")
    p.add_argument("--sample-ids", nargs="*", default=None,
                   help="Only process these sample IDs")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after this many samples")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    cfg = BuildConfig.from_yaml(args.config)
    run_dir   = Path(cfg.run_dir)
    graphs_dir = run_dir / "graphs"
    run_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(run_dir / "build.log", level="DEBUG", rotation="50 MB")

    logger.info(f"Config: {cfg}")

    # Load dataset
    samples = load_locomo_sessions(cfg.data_path)
    logger.info(f"Loaded {len(samples)} samples.")

    if args.sample_ids:
        id_set  = set(args.sample_ids)
        samples = [s for s in samples
                   if s["conversation"][0]["metadata"].get("sample_id") in id_set]
        logger.info(f"Filtered to {len(samples)} samples.")

    if args.limit is not None:
        samples = samples[: args.limit]
        logger.info(f"Limited to {len(samples)} samples.")

    # from_scratch: wipe trajectory file and all graph.json files before building
    if cfg.vector_store.from_scratch:
        traj_path = run_dir / "graph_trajectories.jsonl"
        if traj_path.exists():
            traj_path.unlink()
            logger.info(f"Deleted trajectory file for fresh build: {traj_path}")
        for sample in samples:
            sid = sample["conversation"][0]["metadata"].get("sample_id", "")
            gp = graphs_dir / f"{sid}_graph.json"
            if gp.exists():
                gp.unlink()
                logger.info(f"Deleted existing graph: {gp}")

    # Shared components (stateless across samples)
    llm   = OpenAIClient(
        model       = cfg.llm.model,
        api_key     = cfg.llm.api_key  or None,
        base_url    = cfg.llm.base_url or None,
        temperature = cfg.llm.temperature,
        max_retries = cfg.llm.max_retries,
    )
    store   = ChromaStore(path=cfg.vector_store.path,
                          from_scratch=cfg.vector_store.from_scratch)
    trigger = GraphTrigger(llm)

    # Process each sample
    for idx, sample in enumerate(samples):
        sample_id = sample["conversation"][0]["metadata"].get("sample_id", f"sample_{idx}")
        logger.info(f"[{idx+1}/{len(samples)}] Sample {sample_id}")

        graph_path = graphs_dir / f"{sample_id}_graph.json"

        # Extract main participants (speaker_a / speaker_b) from first session metadata
        first_meta  = sample["conversation"][0].get("metadata", {})
        participants = [
            n for n in [first_meta.get("speaker_a"), first_meta.get("speaker_b")] if n
        ]

        # Per-sample components
        graph     = GraphStore(graph_path, store, sample_id)
        archive   = RawArchive(store, sample_id)
        localizer = GraphLocalizer(
            graph,
            seed_top_k = cfg.graph.seed_top_k,
            max_hops   = cfg.graph.max_hops,
            max_nodes  = cfg.graph.max_nodes,
            max_edges  = cfg.graph.max_edges,
        )
        constructor = GraphConstructor(llm, graph)
        builder     = GraphBuilder(
            graph, archive, trigger, localizer, constructor, cfg,
            participants=participants,
        )

        try:
            builder.build_from_sample(sample)
            # Post-build: merge duplicate entity nodes
            merges = graph.dedup_entities()
            logger.info(
                f"Sample {sample_id} done: "
                f"{graph.node_count()} nodes, {graph.edge_count()} edges "
                f"(dedup merged {merges} nodes)."
            )
        except Exception as exc:
            logger.error(f"Sample {sample_id} failed: {exc}", exc_info=True)
            continue

    logger.info("Build complete.")


if __name__ == "__main__":
    main()

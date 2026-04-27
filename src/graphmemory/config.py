"""Configuration dataclasses for the memory-building pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class VectorStoreConfig:
    path: str = "runs/chroma"
    from_scratch: bool = False


@dataclass
class MemoryConfig:
    k_turns: int = 4           # turns per batch fed to construction
    retrieval_topk: int = 5    # top-K items retrieved per search tool call


@dataclass
class GraphConfig:
    # Construction (Localize)
    seed_top_k: int = 5        # top-K seed nodes from vector search
    max_hops: int = 2          # BFS depth for neighbourhood assembly
    max_nodes: int = 20        # max nodes in a candidate subgraph
    max_edges: int = 30        # max edges in a candidate subgraph
    # Retrieval (Jump)
    retrieval_max_hop: int = 3 # max hops before forced finish
    jump_budget: int = 5       # max neighbours expanded per jump step
    final_answer_compression: bool = False


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_retries: int = 5
    reasoning_effort: str = ""
    disable_thinking: bool = False
    use_extra_body_thinking: bool = False
    # api_key and base_url read from environment by default
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", ""))


@dataclass
class BuildConfig:
    data_path: str = "data/locomo/locomo10.json"
    dataset_name: str = "locomo"
    run_dir: str = "runs/build"
    graph_dir: str = ""
    sample_ids: list = field(default_factory=list)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        with open(path, encoding="utf-8") as f:
            d: Dict[str, Any] = yaml.safe_load(f) or {}

        # Strip experiment metadata (not part of BuildConfig)
        d.pop("experiment", None)

        vs  = VectorStoreConfig(**d.pop("vector_store", {}))
        mem = MemoryConfig(**d.pop("memory", {}))
        g   = GraphConfig(**d.pop("graph", {}))
        llm = LLMConfig(**d.pop("llm", {}))

        # Flatten nested build: block into top-level fields
        build_block = d.pop("build", {})
        if build_block:
            d.setdefault("data_path", build_block.pop("data_path", d.get("data_path")))
            vs.from_scratch = build_block.pop("from_scratch", vs.from_scratch)
            mem.k_turns     = build_block.pop("k_turns", mem.k_turns)

        return cls(vector_store=vs, memory=mem, graph=g, llm=llm, **d)

    @classmethod
    def from_exp_dir(cls, exp_dir: str | Path, mode: str = "build") -> "BuildConfig":
        """Load config from experiments/<id>/config.yaml, wiring paths automatically.

        mode='build' → run_dir = exp_dir/build, chroma = exp_dir/chroma
        mode='qa'    → run_dir = exp_dir/qa,    chroma = exp_dir/chroma,
                       graph_dir = exp_dir/build/graphs
        """
        exp_dir = Path(exp_dir)
        cfg = cls.from_yaml(exp_dir / "config.yaml")
        cfg.vector_store.path = str(exp_dir / "chroma")
        if mode == "build":
            cfg.run_dir = str(exp_dir / "build")
        else:
            cfg.run_dir   = str(exp_dir / "qa")
            cfg.graph_dir = str(exp_dir / "build" / "graphs")
        return cfg

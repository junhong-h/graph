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
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        with open(path, encoding="utf-8") as f:
            d: Dict[str, Any] = yaml.safe_load(f) or {}

        vs  = VectorStoreConfig(**d.pop("vector_store", {}))
        mem = MemoryConfig(**d.pop("memory", {}))
        g   = GraphConfig(**d.pop("graph", {}))
        llm = LLMConfig(**d.pop("llm", {}))
        return cls(vector_store=vs, memory=mem, graph=g, llm=llm, **d)

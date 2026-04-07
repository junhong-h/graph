from __future__ import annotations

from pathlib import Path

from graphmemory.naming import validate_snake_case


class RepositoryLayout:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def ensure_scaffold(self) -> None:
        for path in (
            self.root / "configs",
            self.root / "data" / "raw",
            self.root / "data" / "processed",
            self.root / "artifacts" / "raw_memory",
            self.root / "artifacts" / "graphs",
            self.root / "artifacts" / "indexes",
            self.root / "runs",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def raw_dataset_dir(self, dataset_name: str) -> Path:
        validate_snake_case(dataset_name, "dataset_name")
        return self.root / "data" / "raw" / dataset_name

    def processed_dataset_dir(self, dataset_name: str, version: str) -> Path:
        validate_snake_case(dataset_name, "dataset_name")
        return self.root / "data" / "processed" / dataset_name / version

    def raw_memory_dir(self, memory_name: str, version: str) -> Path:
        validate_snake_case(memory_name, "memory_name")
        return self.root / "artifacts" / "raw_memory" / memory_name / version

    def graph_dir(self, graph_name: str, version: str) -> Path:
        validate_snake_case(graph_name, "graph_name")
        return self.root / "artifacts" / "graphs" / graph_name / version

    def index_dir(self, index_name: str, version: str) -> Path:
        validate_snake_case(index_name, "index_name")
        return self.root / "artifacts" / "indexes" / index_name / version

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

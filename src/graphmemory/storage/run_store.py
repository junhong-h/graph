from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

import pyarrow as pa

from graphmemory.models import RunManifest
from graphmemory.naming import iso_now, make_run_id
from graphmemory.storage.layout import RepositoryLayout
from graphmemory.storage.parquet_io import append_rows, ensure_parent, write_json, write_rows

RUN_ERROR_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("stage", pa.string()),
        ("error_message", pa.string()),
        ("traceback_text", pa.string()),
        ("logged_at", pa.string()),
        ("extra_json", pa.string()),
    ]
)


class RunStore:
    def __init__(self, root: str | Path):
        self.layout = RepositoryLayout(root)
        self.layout.ensure_scaffold()

    def create_run(
        self,
        tag: str,
        input_dataset: str,
        input_dataset_version: str,
        prior_graph_version: str | None = None,
        config_refs: list[str] | None = None,
    ) -> RunManifest:
        run_id = make_run_id(tag)
        run_dir = self.layout.run_dir(run_id)
        if run_dir.exists():
            raise FileExistsError(f"run already exists: {run_dir}")

        for path in (
            run_dir / "config_snapshot",
            run_dir / "logs",
            run_dir / "intermediate",
            run_dir / "graph",
            run_dir / "retrieval",
        ):
            path.mkdir(parents=True, exist_ok=True)

        manifest = RunManifest(
            run_id=run_id,
            tag=tag,
            created_at=iso_now(),
            input_dataset=input_dataset,
            input_dataset_version=input_dataset_version,
            prior_graph_version=prior_graph_version,
            config_refs=config_refs or [],
        )
        write_json(run_dir / "config_snapshot" / "manifest.json", manifest.to_dict())
        write_json(run_dir / "metrics.json", {})
        return manifest

    def snapshot_config_files(self, run_id: str, config_paths: list[str | Path]) -> list[Path]:
        run_dir = self.layout.run_dir(run_id)
        snapshot_dir = run_dir / "config_snapshot"
        copied_paths: list[Path] = []
        for config_path in config_paths:
            source = Path(config_path).resolve()
            destination = snapshot_dir / source.name
            shutil.copy2(source, destination)
            copied_paths.append(destination)
        return copied_paths

    def write_metrics(self, run_id: str, metrics: Mapping[str, object]) -> None:
        run_dir = self.layout.run_dir(run_id)
        write_json(run_dir / "metrics.json", dict(metrics))

    def write_intermediate_table(
        self,
        run_id: str,
        stage_name: str,
        rows: list[dict[str, object]],
        schema: pa.Schema,
    ) -> Path:
        path = self.layout.run_dir(run_id) / "intermediate" / f"{stage_name}.parquet"
        write_rows(path, rows, schema)
        return path

    def write_retrieval_table(
        self,
        run_id: str,
        name: str,
        rows: list[dict[str, object]],
        schema: pa.Schema,
    ) -> Path:
        path = self.layout.run_dir(run_id) / "retrieval" / f"{name}.parquet"
        write_rows(path, rows, schema)
        return path

    def write_graph_artifact(
        self,
        run_id: str,
        artifact_name: str,
        payload: Mapping[str, object],
    ) -> Path:
        path = self.layout.run_dir(run_id) / "graph" / artifact_name / "summary.json"
        write_json(path, dict(payload))
        return path

    def append_sample_error(
        self,
        run_id: str,
        sample_id: str,
        stage: str,
        error_message: str,
        traceback_text: str = "",
        extra: Mapping[str, object] | None = None,
    ) -> None:
        run_dir = self.layout.run_dir(run_id)
        row = {
            "sample_id": sample_id,
            "stage": stage,
            "error_message": error_message,
            "traceback_text": traceback_text,
            "logged_at": iso_now(),
            "extra_json": json.dumps(dict(extra or {}), sort_keys=True),
        }

        log_path = run_dir / "logs" / "sample_errors.jsonl"
        ensure_parent(log_path)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

        table_path = run_dir / "logs" / "sample_errors.parquet"
        append_rows(table_path, [row], RUN_ERROR_SCHEMA)

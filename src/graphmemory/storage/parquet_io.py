from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import pyarrow.parquet as pq


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_rows(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    ensure_parent(path)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)


def read_rows(path: Path) -> list[dict[str, object]]:
    table = pq.read_table(path)
    return table.to_pylist()


def append_rows(path: Path, rows: list[dict[str, object]], schema: pa.Schema) -> None:
    existing = read_rows(path) if path.exists() else []
    write_rows(path, [*existing, *rows], schema=schema)

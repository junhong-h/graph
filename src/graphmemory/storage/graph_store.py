from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa

from graphmemory.constants import SCHEMA_VERSION
from graphmemory.models import GraphEdge, GraphNode, GraphNodeAttr, ProvenanceRecord
from graphmemory.naming import iso_now, make_version_id, validate_snake_case
from graphmemory.storage.layout import RepositoryLayout
from graphmemory.storage.parquet_io import read_json, read_rows, write_json, write_rows
from graphmemory.storage.raw_memory_store import RawMemoryStore

GRAPH_NODE_SCHEMA = pa.schema(
    [
        ("node_id", pa.string()),
        ("node_type", pa.string()),
        ("canonical_name", pa.string()),
        ("status", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)

GRAPH_EDGE_SCHEMA = pa.schema(
    [
        ("edge_id", pa.string()),
        ("src_id", pa.string()),
        ("dst_id", pa.string()),
        ("family", pa.string()),
        ("predicate", pa.string()),
        ("status", pa.string()),
        ("weight", pa.float64()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ]
)

GRAPH_ATTR_SCHEMA = pa.schema(
    [
        ("node_id", pa.string()),
        ("attr_key", pa.string()),
        ("attr_value", pa.string()),
        ("value_type", pa.string()),
        ("valid_from", pa.string()),
        ("valid_to", pa.string()),
    ]
)

PROVENANCE_SCHEMA = pa.schema(
    [
        ("owner_type", pa.string()),
        ("owner_id", pa.string()),
        ("record_id", pa.string()),
        ("chunk_id", pa.string()),
        ("evidence_span", pa.string()),
        ("confidence", pa.float64()),
    ]
)


class GraphSnapshotStore:
    def __init__(self, root: str | Path):
        self.layout = RepositoryLayout(root)
        self.layout.ensure_scaffold()

    def write_snapshot(
        self,
        graph_name: str,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge],
        node_attrs: Iterable[GraphNodeAttr],
        provenance: Iterable[ProvenanceRecord],
        summary: dict[str, object] | None = None,
        version: str | None = None,
        source_snapshot_version: str | None = None,
        dataset_names: list[str] | None = None,
        base_version: str | None = None,
    ) -> dict[str, object]:
        validate_snake_case(graph_name, "graph_name")
        version = version or make_version_id()
        version_dir = self.layout.graph_dir(graph_name, version)
        if version_dir.exists():
            raise FileExistsError(f"graph version already exists: {version_dir}")

        base_nodes: list[dict[str, object]] = []
        base_edges: list[dict[str, object]] = []
        base_attrs: list[dict[str, object]] = []
        base_provenance: list[dict[str, object]] = []
        if base_version:
            base_dir = self.layout.graph_dir(graph_name, base_version)
            if not base_dir.exists():
                raise FileNotFoundError(base_dir)
            base_nodes = read_rows(base_dir / "nodes.parquet")
            base_edges = read_rows(base_dir / "edges.parquet")
            base_attrs = read_rows(base_dir / "node_attrs.parquet")
            base_provenance = read_rows(base_dir / "provenance.parquet")
            source_snapshot_version = source_snapshot_version or base_version

        node_rows = [*base_nodes, *(node.to_row() for node in nodes)]
        edge_rows = [*base_edges, *(edge.to_row() for edge in edges)]
        attr_rows = [*base_attrs, *(attr.to_row() for attr in node_attrs)]
        provenance_rows = [*base_provenance, *(record.to_row() for record in provenance)]

        write_rows(version_dir / "nodes.parquet", node_rows, GRAPH_NODE_SCHEMA)
        write_rows(version_dir / "edges.parquet", edge_rows, GRAPH_EDGE_SCHEMA)
        write_rows(version_dir / "node_attrs.parquet", attr_rows, GRAPH_ATTR_SCHEMA)
        write_rows(version_dir / "provenance.parquet", provenance_rows, PROVENANCE_SCHEMA)

        meta = {
            "graph_name": graph_name,
            "version": version,
            "created_at": iso_now(),
            "schema_version": SCHEMA_VERSION,
            "dataset_names": sorted(dataset_names or []),
            "source_snapshot_version": source_snapshot_version,
        }
        auto_summary = {
            "node_count": len(node_rows),
            "edge_count": len(edge_rows),
            "node_attr_count": len(attr_rows),
            "provenance_count": len(provenance_rows),
            "merge_count": 0,
            "revise_count": 0,
            "prune_count": 0,
            "fallback_hit_count": 0,
        }
        if summary:
            auto_summary.update(summary)

        write_json(version_dir / "meta.json", meta)
        write_json(version_dir / "summary.json", auto_summary)
        return meta

    def compare_snapshots(
        self,
        graph_name: str,
        left_version: str,
        right_version: str,
    ) -> dict[str, object]:
        validate_snake_case(graph_name, "graph_name")
        left_dir = self.layout.graph_dir(graph_name, left_version)
        right_dir = self.layout.graph_dir(graph_name, right_version)
        return {
            "left_summary": read_json(left_dir / "summary.json"),
            "right_summary": read_json(right_dir / "summary.json"),
            "table_diffs": {
                "nodes": self._table_diff(left_dir / "nodes.parquet", right_dir / "nodes.parquet"),
                "edges": self._table_diff(left_dir / "edges.parquet", right_dir / "edges.parquet"),
                "node_attrs": self._table_diff(left_dir / "node_attrs.parquet", right_dir / "node_attrs.parquet"),
                "provenance": self._table_diff(left_dir / "provenance.parquet", right_dir / "provenance.parquet"),
            },
        }

    def resolve_provenance(
        self,
        graph_name: str,
        version: str,
        owner_type: str,
        owner_id: str,
        raw_memory_store: RawMemoryStore,
        raw_memory_name: str,
        raw_memory_version: str,
    ) -> list[dict[str, object]]:
        version_dir = self.layout.graph_dir(graph_name, version)
        provenance_rows = read_rows(version_dir / "provenance.parquet")
        matches = [
            row
            for row in provenance_rows
            if row["owner_type"] == owner_type and row["owner_id"] == owner_id
        ]
        resolved: list[dict[str, object]] = []
        for row in matches:
            evidence = raw_memory_store.resolve_chunk(
                memory_name=raw_memory_name,
                version=raw_memory_version,
                chunk_id=str(row["chunk_id"]),
            )
            resolved.append({"provenance": row, "evidence": evidence})
        return resolved

    @staticmethod
    def _table_diff(left_path: Path, right_path: Path) -> dict[str, int]:
        left_rows = {json.dumps(row, sort_keys=True) for row in read_rows(left_path)}
        right_rows = {json.dumps(row, sort_keys=True) for row in read_rows(right_path)}
        return {
            "left_count": len(left_rows),
            "right_count": len(right_rows),
            "added_rows": len(right_rows - left_rows),
            "removed_rows": len(left_rows - right_rows),
        }

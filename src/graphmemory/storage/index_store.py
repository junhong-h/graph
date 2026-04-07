from __future__ import annotations

from pathlib import Path

try:
    import duckdb
except ModuleNotFoundError:  # pragma: no cover - covered via fallback behavior
    duckdb = None

from graphmemory.naming import validate_snake_case
from graphmemory.storage.layout import RepositoryLayout
from graphmemory.storage.parquet_io import read_rows


class DuckDBIndexStore:
    def __init__(self, root: str | Path):
        self.layout = RepositoryLayout(root)
        self.layout.ensure_scaffold()

    def build_graph_index(
        self,
        index_name: str,
        version: str,
        graph_name: str,
        graph_version: str,
    ) -> Path:
        validate_snake_case(index_name, "index_name")
        validate_snake_case(graph_name, "graph_name")
        if duckdb is None:
            raise RuntimeError("duckdb is not installed")

        index_dir = self.layout.index_dir(index_name, version)
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "graphmemory.duckdb"
        graph_dir = self.layout.graph_dir(graph_name, graph_version)
        connection = duckdb.connect(str(db_path))
        try:
            node_path = str(graph_dir / "nodes.parquet").replace("'", "''")
            edge_path = str(graph_dir / "edges.parquet").replace("'", "''")
            attr_path = str(graph_dir / "node_attrs.parquet").replace("'", "''")
            provenance_path = str(graph_dir / "provenance.parquet").replace("'", "''")
            connection.execute(
                f"create or replace view nodes as select * from read_parquet('{node_path}')"
            )
            connection.execute(
                f"create or replace view edges as select * from read_parquet('{edge_path}')"
            )
            connection.execute(
                f"create or replace view node_attrs as select * from read_parquet('{attr_path}')"
            )
            connection.execute(
                f"create or replace view provenance as select * from read_parquet('{provenance_path}')"
            )
            connection.execute(
                "create or replace table index_meta as select ? as graph_name, ? as graph_version",
                [graph_name, graph_version],
            )
        finally:
            connection.close()
        return db_path


class GraphQueryService:
    def __init__(self, root: str | Path):
        self.layout = RepositoryLayout(root)
        self.layout.ensure_scaffold()

    def find_nodes(
        self,
        graph_name: str,
        graph_version: str,
        node_type: str | None = None,
        canonical_name_contains: str | None = None,
        index_name: str | None = None,
        index_version: str | None = None,
    ) -> list[dict[str, object]]:
        rows = self._query_with_index(
            graph_name=graph_name,
            graph_version=graph_version,
            node_type=node_type,
            canonical_name_contains=canonical_name_contains,
            index_name=index_name,
            index_version=index_version,
        )
        if rows is not None:
            return rows

        graph_dir = self.layout.graph_dir(graph_name, graph_version)
        rows = read_rows(graph_dir / "nodes.parquet")
        return [
            row
            for row in rows
            if (node_type is None or row["node_type"] == node_type)
            and (
                canonical_name_contains is None
                or canonical_name_contains.lower() in str(row["canonical_name"]).lower()
            )
        ]

    def _query_with_index(
        self,
        graph_name: str,
        graph_version: str,
        node_type: str | None,
        canonical_name_contains: str | None,
        index_name: str | None,
        index_version: str | None,
    ) -> list[dict[str, object]] | None:
        if duckdb is None or not index_name or not index_version:
            return None
        validate_snake_case(index_name, "index_name")
        db_path = self.layout.index_dir(index_name, index_version) / "graphmemory.duckdb"
        if not db_path.exists():
            return None

        connection = duckdb.connect(str(db_path), read_only=True)
        try:
            meta = connection.execute("select graph_name, graph_version from index_meta").fetchone()
            if meta is None or meta[0] != graph_name or meta[1] != graph_version:
                return None

            query = "select * from nodes where 1=1"
            parameters: list[object] = []
            if node_type is not None:
                query += " and node_type = ?"
                parameters.append(node_type)
            if canonical_name_contains is not None:
                query += " and lower(canonical_name) like ?"
                parameters.append(f"%{canonical_name_contains.lower()}%")
            columns = [item[0] for item in connection.execute(query, parameters).description]
            results = connection.execute(query, parameters).fetchall()
            return [dict(zip(columns, row, strict=True)) for row in results]
        finally:
            connection.close()

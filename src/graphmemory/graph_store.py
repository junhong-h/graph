"""JSON-backed graph store with ChromaDB vector index for node retrieval.

Graph JSON schema
-----------------
{
  "nodes": {
    "<node_id>": {
      "type": "Entity" | "Event",
      "canonical_name": "...",
      "aliases": [...],
      "attrs": {"key": "value", ...},
      "created_at": "YYYY-MM-DD HH:MM:SS",
      "updated_at": "YYYY-MM-DD HH:MM:SS"
    }
  },
  "edges": [
    {
      "edge_id": "...",
      "src": "<node_id>",
      "dst": "<node_id>",
      "family": "entity-event" | "entity-entity" | "event-event",
      "predicate": "...",
      "created_at": "YYYY-MM-DD HH:MM:SS"
    }
  ]
}
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from graphmemory.vector_store import VectorStore

NODE_TYPES = {"Entity", "Event"}
EDGE_FAMILIES = {"entity-event", "entity-entity", "event-event"}
CONTROL_PREDICATES = {"same_as", "before", "after", "updates"}


# ---------------------------------------------------------------------------
# GraphStore
# ---------------------------------------------------------------------------

class GraphStore:
    """Maintains a typed graph as a JSON file, with ChromaDB for node vector search."""

    def __init__(self, graph_path: Path, vector_store: VectorStore, sample_id: str):
        self._path = Path(graph_path)
        self._store = vector_store
        self._node_col = f"{sample_id}_nodes"
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            with self._path.open(encoding="utf-8") as f:
                data = json.load(f)
            self._nodes: Dict[str, Dict] = data.get("nodes", {})
            self._edges: List[Dict] = data.get("edges", [])
            logger.info(f"Loaded graph: {len(self._nodes)} nodes, {len(self._edges)} edges")
        else:
            self._nodes = {}
            self._edges = []

        self._store.create_collection(self._node_col, get_or_create=True)

        # Re-sync existing nodes to ChromaDB (handles from_scratch wipe)
        if self._nodes and self._store.count(self._node_col) == 0:
            logger.info(f"Re-syncing {len(self._nodes)} nodes to empty ChromaDB collection…")
            for nid in self._nodes:
                self._sync_embedding(nid)

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_type: str,
        canonical_name: str,
        aliases: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError(f"node_type must be one of {NODE_TYPES}, got: {node_type}")
        node_id = node_id or str(uuid.uuid4())
        now = _now()
        self._nodes[node_id] = {
            "type": node_type,
            "canonical_name": canonical_name,
            "aliases": _coerce_aliases(aliases),
            "attrs": attrs or {},
            "created_at": now,
            "updated_at": now,
        }
        self._sync_embedding(node_id)
        self.save()
        return node_id

    def get_node(self, node_id: str) -> Optional[Dict]:
        node = self._nodes.get(node_id)
        return dict(node) if node else None

    def update_node(
        self,
        node_id: str,
        canonical_name: Optional[str] = None,
        new_aliases: Optional[List[str]] = None,
        attrs_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            logger.warning(f"update_node: node {node_id} not found.")
            return
        if canonical_name is not None:
            node["canonical_name"] = canonical_name
        if new_aliases:
            node["aliases"] = list(set(node["aliases"] + _coerce_aliases(new_aliases)))
        if attrs_update:
            node["attrs"].update(attrs_update)
        node["updated_at"] = _now()
        self._sync_embedding(node_id)
        self.save()

    def merge_nodes(self, src_id: str, dst_id: str) -> None:
        """Merge src into dst: absorb aliases/attrs, redirect edges, delete src."""
        src = self._nodes.get(src_id)
        dst = self._nodes.get(dst_id)
        if not src or not dst:
            logger.warning(f"merge_nodes: node not found ({src_id}, {dst_id}).")
            return

        # Absorb
        dst["aliases"] = list(set(
            _coerce_aliases(dst["aliases"]) +
            _coerce_aliases(src["aliases"]) +
            [src["canonical_name"]]
        ))
        for k, v in src["attrs"].items():
            if k not in dst["attrs"]:
                dst["attrs"][k] = v
        dst["updated_at"] = _now()

        # Redirect edges, drop self-loops and duplicates
        seen: Set[Tuple[str, str, str]] = set()
        new_edges: List[Dict] = []
        for edge in self._edges:
            new_src = dst_id if edge["src"] == src_id else edge["src"]
            new_dst = dst_id if edge["dst"] == src_id else edge["dst"]
            if new_src == new_dst:
                continue  # self-loop
            key = (new_src, new_dst, edge["predicate"])
            if key in seen:
                continue
            seen.add(key)
            e = dict(edge)
            e["src"] = new_src
            e["dst"] = new_dst
            new_edges.append(e)
        self._edges = new_edges

        del self._nodes[src_id]
        try:
            self._store.delete(self._node_col, [src_id])
        except Exception:
            pass
        self._sync_embedding(dst_id)
        self.save()
        logger.info(f"Merged {src_id[:8]} → {dst_id[:8]}")

    def delete_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        del self._nodes[node_id]
        self._edges = [e for e in self._edges if e["src"] != node_id and e["dst"] != node_id]
        try:
            self._store.delete(self._node_col, [node_id])
        except Exception:
            pass
        self.save()

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, src: str, dst: str, family: str, predicate: str) -> str:
        if src == dst:
            logger.warning(f"add_edge: self-loop prevented ({src[:8]} --{predicate}-->)")
            return ""
        if src not in self._nodes or dst not in self._nodes:
            logger.warning(f"add_edge: node(s) not found ({src[:8]}, {dst[:8]})")
            return ""
        src, dst, family = self._normalize_edge(src, dst, family)
        if not family:
            logger.warning(f"add_edge: invalid edge family ({family!r})")
            return ""
        # Dedup
        for e in self._edges:
            if e["src"] == src and e["dst"] == dst and e["predicate"] == predicate:
                return e["edge_id"]
        edge_id = str(uuid.uuid4())
        self._edges.append({
            "edge_id": edge_id,
            "src": src,
            "dst": dst,
            "family": family,
            "predicate": predicate,
            "created_at": _now(),
        })
        self.save()
        return edge_id

    def _normalize_edge(self, src: str, dst: str, family: str) -> Tuple[str, str, str]:
        """Normalize edge family and endpoint order before persistence."""
        family = str(family or "").strip()
        if family == "event-entity":
            family = "entity-event"

        if family not in EDGE_FAMILIES:
            return src, dst, ""

        src_type = self._nodes[src].get("type")
        dst_type = self._nodes[dst].get("type")
        if family == "entity-event":
            if src_type == "Event" and dst_type == "Entity":
                return dst, src, family
            if src_type == "Entity" and dst_type == "Event":
                return src, dst, family
            return src, dst, ""
        if family == "entity-entity" and (src_type != "Entity" or dst_type != "Entity"):
            return src, dst, ""
        if family == "event-event" and (src_type != "Event" or dst_type != "Event"):
            return src, dst, ""

        return src, dst, family

    def delete_edge(self, edge_id: str) -> None:
        self._edges = [e for e in self._edges if e["edge_id"] != edge_id]
        self.save()

    def get_edges(
        self,
        node_id: Optional[str] = None,
        family: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> List[Dict]:
        edges = self._edges
        if node_id:
            edges = [e for e in edges if e["src"] == node_id or e["dst"] == node_id]
        if family:
            edges = [e for e in edges if e["family"] == family]
        if predicate:
            edges = [e for e in edges if e["predicate"] == predicate]
        return [dict(e) for e in edges]

    # ------------------------------------------------------------------
    # Subgraph / neighbourhood
    # ------------------------------------------------------------------

    def get_neighborhood(
        self,
        seed_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 20,
        max_edges: int = 30,
    ) -> Dict[str, Any]:
        """BFS from seed_ids up to max_hops. Returns {"nodes": {...}, "edges": [...]}."""
        visited_nodes: Set[str] = set()
        visited_edge_ids: Set[str] = set()
        frontier = [nid for nid in seed_ids if nid in self._nodes]
        visited_nodes.update(frontier)

        for _ in range(max_hops):
            if not frontier or len(visited_nodes) >= max_nodes:
                break
            next_frontier: List[str] = []
            for nid in frontier:
                for edge in self._edges:
                    if edge["src"] == nid:
                        neighbor = edge["dst"]
                    elif edge["dst"] == nid:
                        neighbor = edge["src"]
                    else:
                        continue
                    if neighbor not in visited_nodes and neighbor in self._nodes:
                        if len(visited_nodes) < max_nodes:
                            visited_nodes.add(neighbor)
                            next_frontier.append(neighbor)
                    visited_edge_ids.add(edge["edge_id"])
            frontier = next_frontier

        subgraph_edges = [
            dict(e) for e in self._edges
            if e["src"] in visited_nodes and e["dst"] in visited_nodes
        ][:max_edges]

        return {
            "nodes": {nid: dict(self._nodes[nid]) for nid in visited_nodes},
            "edges": subgraph_edges,
        }

    def search_nodes(self, query: str, top_k: int = 5) -> List[str]:
        """Return node_ids by embedding similarity, filtered to existing nodes."""
        if self._store.count(self._node_col) == 0:
            return []
        results = self._store.search(self._node_col, query, top_k=top_k)
        ids = results.get("ids", [[]])[0]
        return [nid for nid in ids if nid in self._nodes]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all_nodes(self) -> Dict[str, Dict]:
        return {nid: dict(n) for nid, n in self._nodes.items()}

    def get_all_edges(self) -> List[Dict]:
        return [dict(e) for e in self._edges]

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def dedup_entities(self) -> int:
        """Merge Entity nodes that share the same canonical_name (case-insensitive).

        Within each name group, keeps the node with the most edges as the canonical one
        and merges all others into it. Returns the number of merges performed.
        """
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for nid, node in list(self._nodes.items()):
            if node["type"] == "Entity":
                groups[node["canonical_name"].strip().lower()].append(nid)

        merges = 0
        for name, ids in groups.items():
            if len(ids) < 2:
                continue
            # Pick the node with the most edges as the merge target
            def edge_count_for(nid: str) -> int:
                return sum(1 for e in self._edges if e["src"] == nid or e["dst"] == nid)
            ids_sorted = sorted(ids, key=edge_count_for, reverse=True)
            dst_id = ids_sorted[0]
            for src_id in ids_sorted[1:]:
                if src_id not in self._nodes:
                    continue
                self.merge_nodes(src_id, dst_id)
                merges += 1
                logger.info(f"Dedup: merged '{name}' [{src_id[:8]}] → [{dst_id[:8]}]")
        return merges

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump({"nodes": self._nodes, "edges": self._edges}, f,
                      ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sync_embedding(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if not node:
            return
        text = _node_embedding_text(node)
        try:
            self._store.upsert(
                self._node_col,
                ids=[node_id],
                documents=[text],
                metadatas=[{"node_type": node["type"], "canonical_name": node["canonical_name"]}],
            )
        except Exception as exc:
            logger.warning(f"Embedding sync failed for {node_id[:8]}: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_aliases(aliases) -> list:
    """Ensure aliases is a list of strings (guard against LLM returning list of dicts)."""
    if not aliases:
        return []
    result = []
    for a in aliases:
        if isinstance(a, str):
            result.append(a)
        elif isinstance(a, dict):
            # e.g. {"name": "Alice"} → "Alice"
            result.append(next(iter(a.values()), str(a)))
        else:
            result.append(str(a))
    return result


PROVENANCE_ATTR_KEYS = {
    "original_text",
    "batch_id",
    "source_turn_ids",
    "source",
    "created_at",
    "updated_at",
}


def node_relevance_text(node: Dict) -> str:
    """Return compact node text for embedding and relevance scoring."""
    parts = [f"{node['type']}: {node['canonical_name']}"]
    if node.get("aliases"):
        parts.append("aliases: " + ", ".join(node["aliases"]))
    attrs = node.get("attrs") or {}
    for key in ("fact", "quote", "evidence_quote"):
        if attrs.get(key):
            parts.append(f"{key}: {attrs[key]}")
    compact_attrs = [
        f"{k}={v}"
        for k, v in attrs.items()
        if k not in PROVENANCE_ATTR_KEYS
        and k not in {"fact", "quote", "evidence_quote"}
    ]
    if compact_attrs:
        parts.append(" ".join(compact_attrs))
    return ". ".join(parts)


def _node_embedding_text(node: Dict) -> str:
    return node_relevance_text(node)


def format_subgraph(subgraph: Dict) -> str:
    """Render a subgraph dict as concise readable text for LLM prompts."""
    nodes = subgraph.get("nodes", {})
    edges = subgraph.get("edges", [])

    lines = [f"Nodes ({len(nodes)}):"]
    for nid, node in nodes.items():
        aliases = (f" | aliases: {', '.join(node['aliases'])}" if node.get("aliases") else "")
        lines.append(f"  [{nid[:8]}] {node['type']} \"{node['canonical_name']}\"{aliases}")
        attrs = node.get("attrs") or {}
        for key, label in (("fact", "fact"), ("quote", "quote"), ("evidence_quote", "quote")):
            if attrs.get(key):
                lines.append(f"    {label}: {attrs[key]}")
                if key in {"quote", "evidence_quote"}:
                    break
        source = attrs.get("source") or attrs.get("source_turn_ids")
        if source:
            lines.append(f"    source: {source}")
        compact_attrs = [
            f"{k}={v}"
            for k, v in attrs.items()
            if k not in PROVENANCE_ATTR_KEYS
            and k not in {"fact", "quote", "evidence_quote"}
        ]
        if compact_attrs:
            lines.append(f"    attrs: {', '.join(compact_attrs)}")

    lines.append(f"Edges ({len(edges)}):")
    for edge in edges:
        src_name = nodes.get(edge["src"], {}).get("canonical_name", edge["src"][:8])
        dst_name = nodes.get(edge["dst"], {}).get("canonical_name", edge["dst"][:8])
        lines.append(
            f"  [{edge['edge_id'][:8]}] \"{src_name}\" --[{edge['predicate']}]--> "
            f"\"{dst_name}\" ({edge['family']})"
        )

    return "\n".join(lines)

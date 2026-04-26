"""GraphLocalizer: find the most relevant local subgraph for a given text (Step 3 & Step 6).

Three-step pipeline
-------------------
3.1 Seed Retrieval  — vector search on node embeddings to find entry nodes.
3.2 Neighbourhood Assembly — BFS from seeds along all edges, bounded by hops/nodes/edges.
3.3 Subgraph Scoring — rank candidate subgraphs by a four-factor rule score, return the best.

The same class is used for both write-time localization (goal: find the subgraph most likely
to be affected by the incoming input) and query-time localization (goal: find the subgraph
most likely to contain answer evidence).  The `purpose` parameter selects the scoring bias.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from graphmemory.graph_store import GraphStore, format_subgraph


# ---------------------------------------------------------------------------
# GraphLocalizer
# ---------------------------------------------------------------------------

class GraphLocalizer:
    """
    Parameters
    ----------
    graph       : GraphStore
    seed_top_k  : number of seed nodes retrieved by vector search
    max_hops    : BFS depth for neighbourhood assembly
    max_nodes   : max nodes in a candidate subgraph
    max_edges   : max edges in a candidate subgraph
    """

    def __init__(
        self,
        graph: GraphStore,
        seed_top_k: int = 5,
        max_hops: int = 2,
        max_nodes: int = 20,
        max_edges: int = 30,
    ):
        self.graph       = graph
        self.seed_top_k  = seed_top_k
        self.max_hops    = max_hops
        self.max_nodes   = max_nodes
        self.max_edges   = max_edges

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def localize(
        self,
        input_text: str,
        forced_seed_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Return the best local subgraph as {"nodes": {...}, "edges": [...]}.
        forced_seed_ids are always included in the seed set (e.g. main participants).
        Returns an empty subgraph if the graph is empty.
        """
        ranked = self.localize_ranked(input_text, forced_seed_ids=forced_seed_ids, top_m=1)
        best = ranked[0] if ranked else {"nodes": {}, "edges": []}
        logger.debug(
            f"GraphLocalizer: {len(best.get('nodes', {}))} nodes, "
            f"{len(best.get('edges', []))} edges selected."
        )
        return best

    def localize_ranked(
        self,
        input_text: str,
        forced_seed_ids: List[str] | None = None,
        query_variants: List[str] | None = None,
        top_m: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return top-M candidate subgraphs ranked by relevance score."""
        if self.graph.node_count() == 0:
            return []

        seed_ids = self._collect_seed_ids(input_text, forced_seed_ids, query_variants)
        if not seed_ids:
            logger.debug("GraphLocalizer: no seeds found, returning empty ranked list.")
            return []

        candidates = self._neighbourhood_assembly(seed_ids)
        ranked = self._rank_subgraphs(candidates, input_text, seed_ids)
        return ranked[:max(top_m, 1)]

    def localize_union(
        self,
        input_text: str,
        forced_seed_ids: List[str] | None = None,
        query_variants: List[str] | None = None,
        top_m: int = 3,
        max_nodes: int | None = None,
        max_edges: int | None = None,
    ) -> Dict[str, Any]:
        """Return a union of the top-M localized subgraphs for recall-heavy retrieval."""
        ranked = self.localize_ranked(
            input_text,
            forced_seed_ids=forced_seed_ids,
            query_variants=query_variants,
            top_m=top_m,
        )
        return _merge_subgraphs(
            ranked,
            max_nodes=max_nodes or self.max_nodes,
            max_edges=max_edges or self.max_edges,
        )

    # ------------------------------------------------------------------
    # Step 3.1 — Seed Retrieval
    # ------------------------------------------------------------------

    def _seed_retrieval(self, input_text: str) -> List[str]:
        """Return seed node_ids via vector similarity search."""
        return self.graph.search_nodes(input_text, top_k=self.seed_top_k)

    def _collect_seed_ids(
        self,
        input_text: str,
        forced_seed_ids: List[str] | None,
        query_variants: List[str] | None,
    ) -> List[str]:
        queries = [input_text]
        for q in query_variants or []:
            if q and q not in queries:
                queries.append(q)

        seed_ids: List[str] = []
        if forced_seed_ids:
            seed_ids.extend(forced_seed_ids)
        for query in queries:
            for seed_id in self._seed_retrieval(query):
                if seed_id not in seed_ids:
                    seed_ids.append(seed_id)

        if forced_seed_ids:
            return seed_ids[: self.seed_top_k * len(queries) + len(forced_seed_ids)]
        return seed_ids[: self.seed_top_k * len(queries)]

    # ------------------------------------------------------------------
    # Step 3.2 — Neighbourhood Assembly
    # ------------------------------------------------------------------

    def _neighbourhood_assembly(self, seed_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Build one candidate subgraph per seed by expanding its neighbourhood.
        Returns a list of subgraph dicts so the scorer can pick the best.
        """
        candidates: List[Dict] = []
        seen_seed_sets: List[frozenset] = []  # avoid exact-duplicate subgraphs

        for seed_id in seed_ids:
            sub = self.graph.get_neighborhood(
                [seed_id],
                max_hops=self.max_hops,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
            )
            key = frozenset(sub["nodes"].keys())
            if key not in seen_seed_sets:
                seen_seed_sets.append(key)
                candidates.append(sub)

        # Also add a joint candidate seeded from all seeds together
        if len(seed_ids) > 1:
            joint = self.graph.get_neighborhood(
                seed_ids,
                max_hops=self.max_hops,
                max_nodes=self.max_nodes,
                max_edges=self.max_edges,
            )
            key = frozenset(joint["nodes"].keys())
            if key not in seen_seed_sets:
                candidates.append(joint)

        return candidates

    # ------------------------------------------------------------------
    # Step 3.3 — Subgraph Scoring
    # ------------------------------------------------------------------

    def _subgraph_scoring(
        self,
        candidates: List[Dict],
        input_text: str,
        seed_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Score each candidate by four equal-weight factors and return the best.

        Factors (each normalised to [0, 1]):
          1. Seed coverage   — fraction of seed_ids present in the subgraph
          2. Mention density — fraction of subgraph node names mentioned in input_text
          3. Connectivity    — edge_count / max possible edges (measures cohesion)
          4. Size penalty    — 1 - (node_count / max_nodes)  (prefer smaller, focused graphs)
        """
        if not candidates:
            return {"nodes": {}, "edges": []}

        input_lower = input_text.lower()
        ranked = self._rank_subgraphs(candidates, input_text, seed_ids)
        return ranked[0] if ranked else {"nodes": {}, "edges": []}

    def _rank_subgraphs(
        self,
        candidates: List[Dict],
        input_text: str,
        seed_ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        input_lower = input_text.lower()
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for sub in candidates:
            nodes = sub.get("nodes", {})
            edges = sub.get("edges", [])
            n = len(nodes)
            if n == 0:
                continue

            seed_in_sub = sum(1 for s in seed_ids if s in nodes)
            f1 = seed_in_sub / max(len(seed_ids), 1)
            mentioned = sum(
                1 for node in nodes.values()
                if node["canonical_name"].lower() in input_lower
                or any(a.lower() in input_lower for a in node.get("aliases", []))
            )
            f2 = mentioned / n
            max_edges_possible = n * (n - 1) / 2
            f3 = len(edges) / max_edges_possible if max_edges_possible > 0 else 0.0
            f4 = 1.0 - (n / self.max_nodes)
            score = (f1 + f2 + f3 + f4) / 4.0
            scored.append((score, sub))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [sub for _, sub in scored]


def _merge_subgraphs(
    subgraphs: List[Dict[str, Any]],
    max_nodes: int,
    max_edges: int,
) -> Dict[str, Any]:
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []
    edge_ids: Set[str] = set()

    for sub in subgraphs:
        for nid, node in sub.get("nodes", {}).items():
            if nid not in nodes and len(nodes) < max_nodes:
                nodes[nid] = node
        for edge in sub.get("edges", []):
            edge_id = edge.get("edge_id")
            if edge_id in edge_ids:
                continue
            if edge.get("src") not in nodes or edge.get("dst") not in nodes:
                continue
            if len(edges) >= max_edges:
                break
            edges.append(edge)
            if edge_id:
                edge_ids.add(edge_id)

    return {"nodes": nodes, "edges": edges}

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

from typing import Any, Dict, List, Set

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
        if self.graph.node_count() == 0:
            return {"nodes": {}, "edges": []}

        seed_ids = self._seed_retrieval(input_text)

        # Merge forced seeds, dedup, keep them at front so they survive budget limits
        if forced_seed_ids:
            merged = list(forced_seed_ids)
            for s in seed_ids:
                if s not in merged:
                    merged.append(s)
            seed_ids = merged[: self.seed_top_k + len(forced_seed_ids)]

        if not seed_ids:
            logger.debug("GraphLocalizer: no seeds found, returning empty subgraph.")
            return {"nodes": {}, "edges": []}

        candidates = self._neighbourhood_assembly(seed_ids)
        best = self._subgraph_scoring(candidates, input_text, seed_ids)
        logger.debug(
            f"GraphLocalizer: {len(best.get('nodes', {}))} nodes, "
            f"{len(best.get('edges', []))} edges selected."
        )
        return best

    # ------------------------------------------------------------------
    # Step 3.1 — Seed Retrieval
    # ------------------------------------------------------------------

    def _seed_retrieval(self, input_text: str) -> List[str]:
        """Return seed node_ids via vector similarity search."""
        return self.graph.search_nodes(input_text, top_k=self.seed_top_k)

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
        best_sub, best_score = candidates[0], -1.0

        for sub in candidates:
            nodes = sub.get("nodes", {})
            edges = sub.get("edges", [])
            n = len(nodes)
            if n == 0:
                continue

            # Factor 1: seed coverage
            seed_in_sub = sum(1 for s in seed_ids if s in nodes)
            f1 = seed_in_sub / max(len(seed_ids), 1)

            # Factor 2: mention density
            mentioned = sum(
                1 for node in nodes.values()
                if node["canonical_name"].lower() in input_lower
                or any(a.lower() in input_lower for a in node.get("aliases", []))
            )
            f2 = mentioned / n

            # Factor 3: connectivity
            max_edges_possible = n * (n - 1) / 2
            f3 = len(edges) / max_edges_possible if max_edges_possible > 0 else 0.0

            # Factor 4: size penalty (prefer smaller focused subgraphs)
            f4 = 1.0 - (n / self.max_nodes)

            score = (f1 + f2 + f3 + f4) / 4.0
            if score > best_score:
                best_score = score
                best_sub = sub

        return best_sub

"""Unit tests for GraphLocalizer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphmemory.graph_localize import GraphLocalizer
from graphmemory.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(tmp_path: Path) -> GraphStore:
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0
    mock_vs.search.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_vs.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return GraphStore(tmp_path / "graph.json", mock_vs, sample_id="test")


def _make_localizer(graph: GraphStore, search_returns: list[str] | None = None) -> GraphLocalizer:
    loc = GraphLocalizer(graph, seed_top_k=3, max_hops=2, max_nodes=20, max_edges=30)
    if search_returns is not None:
        graph.search_nodes = MagicMock(return_value=search_returns)
    return loc


# ---------------------------------------------------------------------------
# Empty graph
# ---------------------------------------------------------------------------

def test_localize_empty_graph(tmp_path):
    graph = _make_graph(tmp_path)
    loc = _make_localizer(graph)
    sub = loc.localize("Jon went to a meeting")
    assert sub == {"nodes": {}, "edges": []}


def test_localize_no_seeds_found(tmp_path):
    graph = _make_graph(tmp_path)
    graph.add_node("Entity", "Alice")
    loc = _make_localizer(graph, search_returns=[])
    sub = loc.localize("something unrelated")
    assert sub["nodes"] == {}


# ---------------------------------------------------------------------------
# Seed retrieval
# ---------------------------------------------------------------------------

def test_seed_retrieval_delegates_to_search_nodes(tmp_path):
    graph = _make_graph(tmp_path)
    nid = graph.add_node("Entity", "Jon")
    graph.search_nodes = MagicMock(return_value=[nid])
    loc = GraphLocalizer(graph)
    seeds = loc._seed_retrieval("Jon got a new job")
    graph.search_nodes.assert_called_once()
    assert nid in seeds


# ---------------------------------------------------------------------------
# Neighbourhood assembly
# ---------------------------------------------------------------------------

def test_neighbourhood_assembly_single_seed(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Alice")
    b = graph.add_node("Event", "Meeting")
    graph.add_edge(a, b, "entity-event", "attended")

    loc = _make_localizer(graph, search_returns=[a])
    candidates = loc._neighbourhood_assembly([a])
    assert len(candidates) >= 1
    # Alice and Meeting should both be in the candidate
    assert any(a in c["nodes"] and b in c["nodes"] for c in candidates)


def test_neighbourhood_assembly_joint_candidate(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Alice")
    b = graph.add_node("Entity", "Bob")
    evt = graph.add_node("Event", "Meeting")
    graph.add_edge(a, evt, "entity-event", "attended")
    graph.add_edge(b, evt, "entity-event", "attended")

    loc = _make_localizer(graph, search_returns=[a, b])
    candidates = loc._neighbourhood_assembly([a, b])
    # Should have individual + joint candidates
    node_sets = [frozenset(c["nodes"].keys()) for c in candidates]
    joint_key = frozenset([a, b, evt])
    assert joint_key in node_sets


def test_neighbourhood_assembly_no_duplicates(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Alice")

    loc = _make_localizer(graph, search_returns=[a, a])  # duplicate seed
    candidates = loc._neighbourhood_assembly([a, a])
    node_sets = [frozenset(c["nodes"].keys()) for c in candidates]
    # Should deduplicate
    assert len(node_sets) == len(set(node_sets))


# ---------------------------------------------------------------------------
# Subgraph scoring
# ---------------------------------------------------------------------------

def test_scoring_prefers_more_seeds_covered(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Alice")
    b = graph.add_node("Entity", "Bob")
    c = graph.add_node("Event", "Meeting")

    # Candidate 1: only Alice
    sub1 = {"nodes": {a: graph.get_node(a)}, "edges": []}
    # Candidate 2: Alice + Bob + Meeting (covers more seeds)
    sub2 = {
        "nodes": {a: graph.get_node(a), b: graph.get_node(b), c: graph.get_node(c)},
        "edges": [],
    }

    loc = _make_localizer(graph)
    best = loc._subgraph_scoring([sub1, sub2], "Alice Bob Meeting", [a, b, c])
    # sub2 covers all three seeds, should win
    assert b in best["nodes"] or c in best["nodes"]


def test_scoring_returns_empty_if_no_candidates(tmp_path):
    graph = _make_graph(tmp_path)
    loc = _make_localizer(graph)
    best = loc._subgraph_scoring([], "query", [])
    assert best == {"nodes": {}, "edges": []}


# ---------------------------------------------------------------------------
# End-to-end localize
# ---------------------------------------------------------------------------

def test_localize_returns_connected_subgraph(tmp_path):
    graph = _make_graph(tmp_path)
    jon = graph.add_node("Entity", "Jon")
    meeting = graph.add_node("Event", "Jon's meeting")
    graph.add_edge(jon, meeting, "entity-event", "attended")
    unrelated = graph.add_node("Entity", "Unrelated")

    graph.search_nodes = MagicMock(return_value=[jon])
    loc = GraphLocalizer(graph, max_hops=1)
    sub = loc.localize("Jon attended a meeting")

    assert jon in sub["nodes"]
    assert meeting in sub["nodes"]
    assert unrelated not in sub["nodes"]

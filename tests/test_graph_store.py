"""Unit tests for GraphStore (no LLM, no real ChromaDB)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from graphmemory.graph_store import GraphStore, format_subgraph, node_relevance_text


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> GraphStore:
    """Return a GraphStore backed by a mock VectorStore."""
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0
    mock_vs.search.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_vs.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    graph_path = tmp_path / "graph.json"
    return GraphStore(graph_path, mock_vs, sample_id="test")


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------

def test_add_and_get_node(tmp_path):
    gs = _make_store(tmp_path)
    nid = gs.add_node("Entity", "Jon", aliases=["Jonathan"], attrs={"job": "engineer"})
    node = gs.get_node(nid)
    assert node["canonical_name"] == "Jon"
    assert node["type"] == "Entity"
    assert "Jonathan" in node["aliases"]
    assert node["attrs"]["job"] == "engineer"


def test_add_node_invalid_type(tmp_path):
    gs = _make_store(tmp_path)
    with pytest.raises(ValueError):
        gs.add_node("Person", "Jon")


def test_update_node(tmp_path):
    gs = _make_store(tmp_path)
    nid = gs.add_node("Entity", "Jon")
    gs.update_node(nid, canonical_name="Jonathan", new_aliases=["JJ"], attrs_update={"job": "dev"})
    node = gs.get_node(nid)
    assert node["canonical_name"] == "Jonathan"
    assert "JJ" in node["aliases"]
    assert node["attrs"]["job"] == "dev"


def test_update_node_not_found(tmp_path, caplog):
    gs = _make_store(tmp_path)
    gs.update_node("nonexistent", canonical_name="X")  # should log warning, not crash


def test_delete_node_also_removes_edges(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "Alice")
    b = gs.add_node("Event", "Meeting")
    gs.add_edge(a, b, "entity-event", "attended")
    assert gs.edge_count() == 1

    gs.delete_node(a)
    assert gs.get_node(a) is None
    assert gs.edge_count() == 0


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

def test_merge_nodes_absorbs_aliases_and_attrs(tmp_path):
    gs = _make_store(tmp_path)
    src = gs.add_node("Entity", "Jon", aliases=["J"], attrs={"city": "NYC"})
    dst = gs.add_node("Entity", "Jonathan", aliases=[], attrs={"job": "dev"})
    gs.merge_nodes(src, dst)

    assert gs.get_node(src) is None
    merged = gs.get_node(dst)
    assert "Jon" in merged["aliases"]
    assert "J" in merged["aliases"]
    assert merged["attrs"]["city"] == "NYC"
    assert merged["attrs"]["job"] == "dev"


def test_merge_nodes_redirects_edges(tmp_path):
    gs = _make_store(tmp_path)
    src = gs.add_node("Entity", "Jon")
    dst = gs.add_node("Entity", "Jonathan")
    evt = gs.add_node("Event", "Meeting")
    gs.add_edge(src, evt, "entity-event", "attended")

    gs.merge_nodes(src, dst)
    edges = gs.get_edges(node_id=dst)
    assert any(e["dst"] == evt for e in edges)


def test_merge_nodes_drops_self_loops(tmp_path):
    gs = _make_store(tmp_path)
    src = gs.add_node("Entity", "A")
    dst = gs.add_node("Entity", "B")
    gs.add_edge(src, dst, "entity-entity", "same_as")  # will become self-loop after merge
    gs.merge_nodes(src, dst)
    edges = gs.get_edges(node_id=dst)
    assert all(e["src"] != e["dst"] for e in edges)


def test_merge_nodes_deduplicates_edges(tmp_path):
    gs = _make_store(tmp_path)
    src = gs.add_node("Entity", "A")
    dst = gs.add_node("Entity", "B")
    evt = gs.add_node("Event", "E")
    gs.add_edge(src, evt, "entity-event", "attended")
    gs.add_edge(dst, evt, "entity-event", "attended")  # same predicate to same node

    gs.merge_nodes(src, dst)
    edges = gs.get_edges(node_id=dst, predicate="attended")
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------

def test_add_edge_dedup(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    b = gs.add_node("Event", "B")
    eid1 = gs.add_edge(a, b, "entity-event", "attended")
    eid2 = gs.add_edge(a, b, "entity-event", "attended")
    assert eid1 == eid2
    assert gs.edge_count() == 1


def test_add_edge_self_loop_prevented(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    eid = gs.add_edge(a, a, "entity-entity", "self")
    assert eid == ""
    assert gs.edge_count() == 0


def test_add_edge_missing_node(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    eid = gs.add_edge(a, "nonexistent", "entity-event", "x")
    assert eid == ""


def test_add_edge_normalizes_event_entity_family(tmp_path):
    gs = _make_store(tmp_path)
    event = gs.add_node("Event", "Meeting")
    entity = gs.add_node("Entity", "Alice")

    eid = gs.add_edge(event, entity, "event-entity", "attended")

    assert eid
    edge = gs.get_edges()[0]
    assert edge["family"] == "entity-event"
    assert edge["src"] == entity
    assert edge["dst"] == event


def test_add_edge_infers_blank_family_from_endpoint_types(tmp_path):
    gs = _make_store(tmp_path)
    event = gs.add_node("Event", "Meeting")
    entity = gs.add_node("Entity", "Alice")

    eid = gs.add_edge(event, entity, "", "attended")

    assert eid
    edge = gs.get_edges()[0]
    assert edge["family"] == "entity-event"
    assert edge["src"] == entity
    assert edge["dst"] == event


def test_add_edge_rejects_invalid_family(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    b = gs.add_node("Event", "B")

    eid = gs.add_edge(a, b, "event-location", "x")

    assert eid == ""
    assert gs.edge_count() == 0


def test_add_edge_rejects_family_type_mismatch(tmp_path):
    gs = _make_store(tmp_path)
    alice = gs.add_node("Entity", "Alice")
    donation = gs.add_node("Entity", "Car Donation")
    meeting = gs.add_node("Event", "Meeting")

    assert gs.add_edge(alice, donation, "entity-event", "participant") == ""
    assert gs.add_edge(alice, meeting, "entity-entity", "related") == ""
    assert gs.add_edge(alice, meeting, "event-event", "after") == ""
    assert gs.edge_count() == 0


def test_delete_edge(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    b = gs.add_node("Event", "B")
    eid = gs.add_edge(a, b, "entity-event", "attended")
    gs.delete_edge(eid)
    assert gs.edge_count() == 0


def test_get_edges_filter(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    b = gs.add_node("Event", "B")
    c = gs.add_node("Entity", "C")
    gs.add_edge(a, b, "entity-event", "attended")
    gs.add_edge(a, c, "entity-entity", "knows")

    assert len(gs.get_edges(node_id=a)) == 2
    assert len(gs.get_edges(family="entity-event")) == 1
    assert len(gs.get_edges(predicate="knows")) == 1


# ---------------------------------------------------------------------------
# Neighbourhood tests
# ---------------------------------------------------------------------------

def test_get_neighborhood_1hop(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "A")
    b = gs.add_node("Event", "B")
    c = gs.add_node("Entity", "C")  # not connected
    gs.add_edge(a, b, "entity-event", "attended")

    sub = gs.get_neighborhood([a], max_hops=1)
    assert a in sub["nodes"]
    assert b in sub["nodes"]
    assert c not in sub["nodes"]


def test_get_neighborhood_max_nodes(tmp_path):
    gs = _make_store(tmp_path)
    root = gs.add_node("Entity", "Root")
    for i in range(10):
        n = gs.add_node("Event", f"E{i}")
        gs.add_edge(root, n, "entity-event", "rel")

    sub = gs.get_neighborhood([root], max_hops=1, max_nodes=5)
    assert len(sub["nodes"]) <= 5


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_save_and_reload(tmp_path):
    gs = _make_store(tmp_path)
    nid = gs.add_node("Entity", "Jon")
    gs.add_node("Event", "Meeting")

    mock_vs2 = MagicMock()
    mock_vs2.count.return_value = 0
    gs2 = GraphStore(tmp_path / "graph.json", mock_vs2, sample_id="test")
    assert gs2.node_count() == 2
    assert gs2.get_node(nid)["canonical_name"] == "Jon"


# ---------------------------------------------------------------------------
# format_subgraph
# ---------------------------------------------------------------------------

def test_format_subgraph_output(tmp_path):
    gs = _make_store(tmp_path)
    a = gs.add_node("Entity", "Alice", aliases=["Ali"])
    b = gs.add_node("Event", "Lunch")
    gs.add_edge(a, b, "entity-event", "attended")
    sub = gs.get_neighborhood([a])
    text = format_subgraph(sub)
    assert "Alice" in text
    assert "attended" in text
    assert "Lunch" in text


def test_node_relevance_text_prefers_fact_quote_and_excludes_original_text():
    node = {
        "type": "Event",
        "canonical_name": "John attended convention",
        "aliases": [],
        "attrs": {
            "fact": "On 18 April 2023, John attended a convention the previous month.",
            "quote": "went to a convention together last month",
            "source": ["D12:9"],
            "source_turn_ids": ["D12:9"],
            "batch_id": "batch-1",
            "original_text": "very long batch text that should not be embedded",
            "activity": "convention",
        },
    }

    text = node_relevance_text(node)

    assert "On 18 April 2023" in text
    assert "went to a convention" in text
    assert "activity=convention" in text
    assert "very long batch text" not in text
    assert "batch-1" not in text
    assert "D12:9" not in text


def test_format_subgraph_displays_fact_quote_source_without_original_text(tmp_path):
    gs = _make_store(tmp_path)
    event = gs.add_node(
        "Event",
        "John attended convention",
        attrs={
            "fact": "On 18 April 2023, John attended a convention the previous month.",
            "quote": "went to a convention together last month",
            "source": ["D12:9"],
            "original_text": "very long batch text that should not be shown",
        },
    )

    text = format_subgraph({"nodes": {event: gs.get_node(event)}, "edges": []})

    assert "fact: On 18 April 2023" in text
    assert "quote: went to a convention" in text
    assert "source: ['D12:9']" in text
    assert "very long batch text" not in text

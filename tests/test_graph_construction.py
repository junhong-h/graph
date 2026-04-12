"""Unit tests for GraphConstructor (merged Construction + Update)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphmemory.graph_construction import GraphConstructor, _parse_ops, _resolve
from graphmemory.graph_store import GraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph(tmp_path: Path) -> GraphStore:
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0
    mock_vs.search.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_vs.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return GraphStore(tmp_path / "graph.json", mock_vs, sample_id="test")


def _make_constructor(tmp_path: Path, llm_response: str) -> tuple[GraphConstructor, GraphStore]:
    graph = _make_graph(tmp_path)
    llm = MagicMock()
    llm.complete.return_value = llm_response
    return GraphConstructor(llm, graph), graph


# ---------------------------------------------------------------------------
# _parse_ops
# ---------------------------------------------------------------------------

def test_parse_ops_valid_array():
    response = '[{"op": "Skip", "reason": "nothing"}]'
    ops = _parse_ops(response)
    assert len(ops) == 1
    assert ops[0]["op"] == "Skip"


def test_parse_ops_embedded_in_text():
    response = 'Here are the ops:\n[{"op": "Skip", "reason": "x"}]\nDone.'
    ops = _parse_ops(response)
    assert ops[0]["op"] == "Skip"


def test_parse_ops_empty_on_invalid_json():
    ops = _parse_ops("not json at all")
    assert ops == []


def test_parse_ops_empty_array():
    ops = _parse_ops("[]")
    assert ops == []


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------

def test_resolve_new_label():
    id_map = {"NEW_Jon": "full-uuid-1234"}
    assert _resolve("NEW_Jon", id_map) == "full-uuid-1234"


def test_resolve_8char_prefix():
    id_map = {"abcd1234": "abcd1234-ef56-..."}
    assert _resolve("abcd1234", id_map) == "abcd1234-ef56-..."


def test_resolve_none_on_missing():
    assert _resolve("UNKNOWN", {}) is None


# ---------------------------------------------------------------------------
# CreateEntity / CreateEvent
# ---------------------------------------------------------------------------

def test_create_entity(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEntity",
        "id": "NEW_Jon",
        "canonical_name": "Jon",
        "aliases": ["Jonathan"],
        "attrs": {"job": "engineer"},
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)
    log = gc.run("Jon is an engineer.", {"nodes": {}, "edges": []})

    assert len(log) == 1
    assert log[0]["status"] == "ok"
    # Node should exist in graph
    nodes = graph.get_all_nodes()
    assert any(n["canonical_name"] == "Jon" for n in nodes.values())


def test_create_event(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Lunch",
        "canonical_name": "Jon's lunch meeting",
        "attrs": {"time": "2023-07-15"},
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)
    log = gc.run("Jon had a lunch meeting.", {"nodes": {}, "edges": []})
    assert log[0]["status"] == "ok"
    nodes = graph.get_all_nodes()
    assert any(n["type"] == "Event" for n in nodes.values())


# ---------------------------------------------------------------------------
# Link
# ---------------------------------------------------------------------------

def test_link_new_nodes(tmp_path):
    ops_json = json.dumps([
        {"op": "CreateEntity", "id": "NEW_Jon", "canonical_name": "Jon", "aliases": [], "attrs": {}},
        {"op": "CreateEvent",  "id": "NEW_Mtg", "canonical_name": "Meeting", "aliases": [], "attrs": {}},
        {"op": "Link", "src": "NEW_Jon", "dst": "NEW_Mtg", "family": "entity-event", "predicate": "attended"},
    ])
    gc, graph = _make_constructor(tmp_path, ops_json)
    log = gc.run("Jon attended a meeting.", {"nodes": {}, "edges": []})

    edge_logs = [l for l in log if l["op"] == "Link"]
    assert edge_logs[0]["status"] == "ok"
    assert graph.edge_count() == 1


def test_link_unresolved_id(tmp_path):
    ops_json = json.dumps([
        {"op": "Link", "src": "NONEXIST", "dst": "ALSO_MISSING", "family": "entity-event", "predicate": "x"},
    ])
    gc, graph = _make_constructor(tmp_path, ops_json)
    log = gc.run("...", {"nodes": {}, "edges": []})
    assert log[0]["status"] == "error"


# ---------------------------------------------------------------------------
# AttachAttr / ReviseAttr
# ---------------------------------------------------------------------------

def test_attach_attr_to_existing_node(tmp_path):
    graph = _make_graph(tmp_path)
    nid = graph.add_node("Entity", "Jon")
    prefix = nid[:8]

    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "AttachAttr", "node": prefix, "key": "city", "value": "NYC"},
    ])
    gc = GraphConstructor(llm, graph)
    subgraph = {"nodes": {nid: graph.get_node(nid)}, "edges": []}
    log = gc.run("Jon lives in NYC.", subgraph)

    assert log[0]["status"] == "ok"
    assert graph.get_node(nid)["attrs"]["city"] == "NYC"


def test_revise_attr(tmp_path):
    graph = _make_graph(tmp_path)
    nid = graph.add_node("Entity", "Jon", attrs={"job": "engineer"})
    prefix = nid[:8]

    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "ReviseAttr", "node": prefix, "key": "job", "value": "manager"},
    ])
    gc = GraphConstructor(llm, graph)
    subgraph = {"nodes": {nid: graph.get_node(nid)}, "edges": []}
    log = gc.run("Jon got promoted to manager.", subgraph)

    assert log[0]["status"] == "ok"
    assert graph.get_node(nid)["attrs"]["job"] == "manager"


# ---------------------------------------------------------------------------
# MergeNode
# ---------------------------------------------------------------------------

def test_merge_node(tmp_path):
    graph = _make_graph(tmp_path)
    src = graph.add_node("Entity", "J")
    dst = graph.add_node("Entity", "Jonathan")
    src_p, dst_p = src[:8], dst[:8]

    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "MergeNode", "src": src_p, "dst": dst_p},
    ])
    gc = GraphConstructor(llm, graph)
    subgraph = {"nodes": {src: graph.get_node(src), dst: graph.get_node(dst)}, "edges": []}
    log = gc.run("J is Jonathan.", subgraph)

    assert log[0]["status"] == "ok"
    assert graph.get_node(src) is None  # src merged away
    assert graph.get_node(dst) is not None


# ---------------------------------------------------------------------------
# DeleteEdge
# ---------------------------------------------------------------------------

def test_delete_edge(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "A")
    b = graph.add_node("Event", "B")
    eid = graph.add_edge(a, b, "entity-event", "attended")

    edge_prefix = eid[:8]
    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "DeleteEdge", "edge": edge_prefix},
    ])
    gc = GraphConstructor(llm, graph)
    subgraph = {"nodes": {a: graph.get_node(a), b: graph.get_node(b)},
                "edges": graph.get_edges()}
    log = gc.run("Edge no longer valid.", subgraph)

    assert log[0]["status"] == "ok"
    assert graph.edge_count() == 0


# ---------------------------------------------------------------------------
# PruneNode
# ---------------------------------------------------------------------------

def test_prune_node(tmp_path):
    graph = _make_graph(tmp_path)
    nid = graph.add_node("Entity", "Redundant")
    prefix = nid[:8]

    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "PruneNode", "node": prefix},
    ])
    gc = GraphConstructor(llm, graph)
    subgraph = {"nodes": {nid: graph.get_node(nid)}, "edges": []}
    log = gc.run("...", subgraph)

    assert log[0]["status"] == "ok"
    assert graph.get_node(nid) is None


# ---------------------------------------------------------------------------
# Skip
# ---------------------------------------------------------------------------

def test_skip(tmp_path):
    gc, graph = _make_constructor(tmp_path, '[{"op": "Skip", "reason": "pure small talk"}]')
    log = gc.run("How are you?", {"nodes": {}, "edges": []})
    assert log[0]["op"] == "Skip"
    assert graph.node_count() == 0


# ---------------------------------------------------------------------------
# KeepSeparate
# ---------------------------------------------------------------------------

def test_keep_separate(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Jon Smith")
    b = graph.add_node("Entity", "Jon Jones")
    ap, bp = a[:8], b[:8]

    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "KeepSeparate", "node_a": ap, "node_b": bp, "reason": "different people"},
    ])
    gc = GraphConstructor(llm, graph)
    subgraph = {"nodes": {a: graph.get_node(a), b: graph.get_node(b)}, "edges": []}
    log = gc.run("...", subgraph)
    assert log[0]["status"] == "ok"
    assert graph.node_count() == 2  # no merge happened


# ---------------------------------------------------------------------------
# Malformed LLM output
# ---------------------------------------------------------------------------

def test_malformed_response_returns_empty_log(tmp_path):
    gc, graph = _make_constructor(tmp_path, "I cannot process this.")
    log = gc.run("...", {"nodes": {}, "edges": []})
    assert log == []
    assert graph.node_count() == 0

"""Unit tests for GraphConstructor (merged Construction + Update)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphmemory.graph_construction import (
    ConstructionContext,
    GraphConstructor,
    _first_speaker,
    _parse_ops,
    _resolve,
)
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


def test_resolve_bracketed_prefix_and_canonical_name(tmp_path):
    graph = _make_graph(tmp_path)
    nid = graph.add_node("Entity", "Jon")

    assert _resolve(f"[{nid[:8]}]", {}, graph) == nid
    assert _resolve("Jon", {}, graph) == nid


def test_resolve_new_label_to_existing_entity_name(tmp_path):
    graph = _make_graph(tmp_path)
    nid = graph.add_node("Entity", "Aerial Yoga")

    assert _resolve("NEW_AerialYoga", {}, graph) == nid


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


def test_create_entity_reuses_existing_exact_name(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEntity",
        "id": "NEW_Jon",
        "canonical_name": "Jon",
        "aliases": ["Jonathan"],
        "attrs": {"job": "engineer"},
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)
    existing_id = graph.add_node("Entity", "Jon")

    log = gc.run("Jon is an engineer.", {"nodes": {}, "edges": []})

    assert log[0]["status"] == "ok"
    assert log[0]["node_id"] == existing_id
    assert log[0]["reused"] is True
    assert graph.node_count() == 1
    assert graph.get_node(existing_id)["attrs"]["job"] == "engineer"


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


def test_create_event_rejects_discussion_container(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Discussion",
        "canonical_name": "Maria and John discuss parenting experience",
        "attrs": {"time": "2023-03-06"},
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)

    log = gc.run("...", {"nodes": {}, "edges": []})

    assert log[0]["status"] == "rejected"
    assert graph.node_count() == 0


def test_create_event_rejects_low_value_abstract_opinion(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Positive",
        "canonical_name": "John and Maria spreading positivity",
        "attrs": {
            "fact": "On 1 January 2023, John and Maria agreed on the importance of spreading positivity.",
            "quote": "We should keep spreading positivity.",
            "source": ["D1"],
        },
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)

    log = gc.run("We should keep spreading positivity.", {"nodes": {}, "edges": []})

    assert log[0]["status"] == "rejected"
    assert log[0]["error"] == "low-value abstract event rejected"
    assert graph.node_count() == 0


def test_create_event_keeps_concrete_pet_fact(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Coco",
        "canonical_name": "Maria got a puppy named Coco",
        "attrs": {
            "fact": "On 1 January 2023, Maria got a puppy named Coco.",
            "quote": "I got a puppy named Coco.",
            "source": ["D1"],
        },
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)

    log = gc.run("I got a puppy named Coco.", {"nodes": {}, "edges": []})

    assert log[0]["status"] == "ok"
    assert graph.node_count() == 1


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


def test_link_resolves_bracketed_existing_id(tmp_path):
    graph = _make_graph(tmp_path)
    jon = graph.add_node("Entity", "Jon")
    mtg = graph.add_node("Event", "Meeting")
    llm = MagicMock()
    llm.complete.return_value = json.dumps([
        {"op": "Link", "src": f"[{jon[:8]}]", "dst": f"[{mtg[:8]}]", "family": "entity-event", "predicate": "attended"},
    ])
    gc = GraphConstructor(llm, graph)

    log = gc.run("Jon attended a meeting.", {"nodes": {jon: graph.get_node(jon), mtg: graph.get_node(mtg)}, "edges": []})

    assert log[0]["status"] == "ok"
    assert graph.edge_count() == 1


def test_created_event_repair_adds_source_attrs_and_speaker_link(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Class",
        "canonical_name": "Maria took poetry class",
        "attrs": {},
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)
    context = ConstructionContext(
        batch_id="batch-1",
        batch_turn_ids=["D1"],
        turn_time="9:00 am on 1 January, 2023",
        speaker_a="Maria",
        speaker_b="John",
    )

    log = gc.run(
        "[turn_id=D1; speaker=Maria; listener=John; session_time=9:00 am on 1 January, 2023]\nI took a poetry class.",
        {"nodes": {}, "edges": []},
        context=context,
    )

    event_id = next(item["node_id"] for item in log if item["op"] == "CreateEvent")
    event = graph.get_node(event_id)
    assert event["attrs"]["batch_id"] == "batch-1"
    assert event["attrs"]["source_turn_ids"] == ["D1"]
    assert event["attrs"]["source"] == ["D1"]
    assert event["attrs"]["original_text"].startswith("[turn_id=D1; speaker=Maria")
    assert "quote" not in event["attrs"]
    edges = graph.get_edges(node_id=event_id, family="entity-event")
    assert len(edges) == 1
    assert graph.get_node(edges[0]["src"])["canonical_name"] == "Maria"


def test_created_event_preserves_fact_quote_source_attrs(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Convention",
        "canonical_name": "John attended tech-for-good convention",
        "attrs": {
            "fact": (
                "On 18 April 2023, John said he and his colleagues had gone "
                "to a tech-for-good convention the previous month."
            ),
            "quote": "My colleagues and I went to a convention together last month.",
            "source": ["D12:9"],
        },
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)
    context = ConstructionContext(
        batch_id="batch-1",
        batch_turn_ids=["D12:9"],
        turn_time="7:34 pm on 18 April, 2023",
        speaker_a="John",
        speaker_b="Maria",
    )

    log = gc.run(
        "[turn_id=D12:9; speaker=John; listener=Maria; session_time=7:34 pm on 18 April, 2023]\n"
        "My colleagues and I went to a convention together last month.",
        {"nodes": {}, "edges": []},
        context=context,
    )

    event_id = next(item["node_id"] for item in log if item["op"] == "CreateEvent")
    attrs = graph.get_node(event_id)["attrs"]
    assert attrs["fact"].startswith("On 18 April 2023")
    assert attrs["quote"] == "My colleagues and I went to a convention together last month."
    assert attrs["source"] == ["D12:9"]
    assert attrs["source_turn_ids"] == ["D12:9"]


def test_first_speaker_reads_structured_turn_header():
    text = "[turn_id=D1; speaker=Maria; listener=John; session_time=9:00 am]\nI took a poetry class."

    assert _first_speaker(text) == "Maria"


def test_repair_creates_speaker_instead_of_linking_object_entity(tmp_path):
    ops_json = json.dumps([{
        "op": "CreateEvent",
        "id": "NEW_Donation",
        "canonical_name": "Maria donated old car to homeless shelter",
        "attrs": {},
    }])
    gc, graph = _make_constructor(tmp_path, ops_json)
    graph.add_node("Entity", "Homeless Shelter")
    context = ConstructionContext(
        batch_id="batch-1",
        batch_turn_ids=["D1"],
        turn_time="9:00 am on 1 January, 2023",
        speaker_a="Maria",
        speaker_b="John",
    )

    log = gc.run(
        "[turn_id=D1; speaker=Maria; listener=John; session_time=9:00 am on 1 January, 2023]\nI donated my old car to the homeless shelter.",
        {"nodes": {}, "edges": []},
        context=context,
    )

    event_id = next(item["node_id"] for item in log if item["op"] == "CreateEvent")
    edges = graph.get_edges(node_id=event_id, family="entity-event")
    linked_names = {
        graph.get_node(edge["src"])["canonical_name"]
        if graph.get_node(edge["src"])["type"] == "Entity"
        else graph.get_node(edge["dst"])["canonical_name"]
        for edge in edges
    }
    assert "Maria" in linked_names


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

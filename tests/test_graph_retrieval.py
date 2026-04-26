"""Unit tests for GraphRetriever (Steps 6-11)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graphmemory.graph_retrieval import (
    GraphRetriever,
    _canonicalize_final_answer,
    _parse_action,
    get_answer_format,
)
from graphmemory.graph_store import GraphStore
from graphmemory.raw_archive import RawArchive
from graphmemory.graph_localize import GraphLocalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph(tmp_path: Path) -> GraphStore:
    mock_vs = MagicMock()
    mock_vs.count.return_value = 0
    mock_vs.search.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    mock_vs.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return GraphStore(tmp_path / "graph.json", mock_vs, sample_id="test")


def _make_archive() -> RawArchive:
    store = MagicMock()
    store.count.return_value = 0
    store.search.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return RawArchive(store, sample_id="test")


def _make_localizer(graph: GraphStore, subgraph: dict | None = None) -> GraphLocalizer:
    loc = GraphLocalizer(graph)
    loc.localize = MagicMock(return_value=subgraph or {"nodes": {}, "edges": []})
    return loc


def _make_retriever(
    tmp_path: Path,
    llm_responses: list[str],
    subgraph: dict | None = None,
    archive_hits: list[str] | None = None,
) -> GraphRetriever:
    graph    = _make_graph(tmp_path)
    archive  = _make_archive()
    localizer = _make_localizer(graph, subgraph)
    llm = MagicMock()
    llm.complete.side_effect = llm_responses
    if archive_hits is not None:
        archive.search = MagicMock(return_value=[{"text": t, "meta": {}} for t in archive_hits])
    return GraphRetriever(
        graph=graph, archive=archive, localizer=localizer, llm=llm,
        max_hop=3, jump_budget=3,
    )


# ---------------------------------------------------------------------------
# _parse_action
# ---------------------------------------------------------------------------

def test_parse_action_finish():
    action, args = _parse_action('{"action": "finish", "answer": "Jon"}')
    assert action == "finish"
    assert args["answer"] == "Jon"


def test_parse_action_jump():
    action, args = _parse_action('{"action": "jump", "node_ids": ["abc12345"], "relation_family": "entity-event", "budget": 3}')
    assert action == "jump"
    assert args["relation_family"] == "entity-event"


def test_parse_action_raw_fallback():
    action, args = _parse_action('{"action": "raw_fallback", "query": "Jon job"}')
    assert action == "raw_fallback"
    assert args["query"] == "Jon job"


def test_parse_action_no_json_returns_finish():
    action, args = _parse_action("I cannot find anything.")
    assert action == "finish"


def test_parse_action_malformed_json():
    action, args = _parse_action("{broken json}")
    assert action == "finish"


# ---------------------------------------------------------------------------
# get_answer_format
# ---------------------------------------------------------------------------

def test_answer_format_locomo_cat3():
    fmt = get_answer_format("locomo", "3")
    assert "Yes" in fmt or "No" in fmt
    assert "short phrase" in fmt


def test_answer_format_locomo_default():
    fmt = get_answer_format("locomo", "1")
    assert "July" in fmt or "date" in fmt.lower()
    assert "No explanations" in fmt
    assert "comma-separated" in fmt
    assert "NEVER output 'Not mentioned in the conversation'" in fmt


def test_answer_format_locomo_cat5_allows_refusal():
    fmt = get_answer_format("locomo", "5")
    assert "adversarial unanswerable" in fmt
    assert "output exactly 'Not mentioned in the conversation'" in fmt


# ---------------------------------------------------------------------------
# answer() — finish on first hop
# ---------------------------------------------------------------------------

def test_answer_direct_finish(tmp_path):
    retriever = _make_retriever(
        tmp_path,
        llm_responses=['{"action": "finish", "answer": "engineer"}'],
    )
    result = retriever.answer("What is Jon's job?")
    assert result["answer"] == "engineer"
    assert result["traces"][0]["action"] == "finish"


def test_answer_direct_finish_canonicalizes_answer(tmp_path):
    retriever = _make_retriever(
        tmp_path,
        llm_responses=['{"action": "finish", "answer": "The answer is: engineer."}'],
    )

    result = retriever.answer("What is Jon's job?")

    assert result["answer"] == "engineer"


def test_answer_optional_final_compression(tmp_path):
    retriever = _make_retriever(
        tmp_path,
        llm_responses=[
            '{"action": "finish", "answer": "Jon is an engineer at StarAI."}',
            "engineer",
        ],
    )
    retriever.final_answer_compression = True

    result = retriever.answer("What is Jon's job?")

    assert result["answer"] == "engineer"
    assert retriever.llm.complete.call_count == 2


def test_answerable_refusal_uses_raw_repair(tmp_path):
    retriever = _make_retriever(
        tmp_path,
        llm_responses=[
            '{"action": "finish", "answer": "Not mentioned in the conversation"}',
            "engineer",
        ],
        archive_hits=["Jon is an engineer at StarAI."],
    )

    result = retriever.answer("What is Jon's job?", category="1")

    assert result["answer"] == "engineer"
    assert any(t["action"] == "answerable_refusal_raw_fallback" for t in result["traces"])


def test_cat5_refusal_is_preserved(tmp_path):
    retriever = _make_retriever(
        tmp_path,
        llm_responses=['{"action": "finish", "answer": "Not mentioned in the conversation"}'],
    )

    result = retriever.answer("Did Jon mention a submarine?", category="5")

    assert result["answer"] == "Not mentioned in the conversation"
    assert len(result["traces"]) == 1


def test_answer_cat1_uses_union_localization(tmp_path):
    graph = _make_graph(tmp_path)
    archive = _make_archive()
    localizer = GraphLocalizer(graph)
    localizer.max_nodes = 20
    localizer.max_edges = 30
    localizer.localize = MagicMock(return_value={"nodes": {}, "edges": []})
    localizer.localize_union = MagicMock(return_value={"nodes": {}, "edges": []})
    llm = MagicMock()
    llm.complete.return_value = '{"action": "finish", "answer": "poetry class"}'
    retriever = GraphRetriever(graph=graph, archive=archive, localizer=localizer, llm=llm)

    retriever.answer("What writing classes has Alice taken?", category="1")

    localizer.localize_union.assert_called_once()
    localizer.localize.assert_not_called()


# ---------------------------------------------------------------------------
# answer() — jump then finish
# ---------------------------------------------------------------------------

def test_answer_jump_then_finish(tmp_path):
    graph = _make_graph(tmp_path)
    jon  = graph.add_node("Entity", "Jon")
    mtg  = graph.add_node("Event", "Meeting")
    graph.add_edge(jon, mtg, "entity-event", "attended")

    archive   = _make_archive()
    localizer = _make_localizer(graph, {"nodes": {jon: graph.get_node(jon)}, "edges": []})
    llm = MagicMock()
    llm.complete.side_effect = [
        json.dumps({"action": "jump", "node_ids": [jon[:8]], "relation_family": "entity-event", "budget": 3}),
        json.dumps({"action": "finish", "answer": "Meeting"}),
    ]
    retriever = GraphRetriever(graph=graph, archive=archive, localizer=localizer, llm=llm, max_hop=3)
    result = retriever.answer("What event did Jon attend?")
    assert result["answer"] == "Meeting"
    assert len(result["traces"]) == 2


# ---------------------------------------------------------------------------
# answer() — raw_fallback
# ---------------------------------------------------------------------------

def test_answer_with_raw_fallback(tmp_path):
    retriever = _make_retriever(
        tmp_path,
        llm_responses=[
            '{"action": "raw_fallback", "query": "Jon job"}',
            '{"action": "finish", "answer": "engineer"}',
        ],
        archive_hits=["Jon is an engineer at StarAI."],
    )
    result = retriever.answer("What is Jon's job?")
    assert result["answer"] == "engineer"
    fb_traces = [t for t in result["traces"] if t["action"] == "raw_fallback"]
    assert len(fb_traces) == 1


def test_answer_frontier_exhausted_uses_raw_fallback_then_continues(tmp_path):
    graph = _make_graph(tmp_path)
    isolated = graph.add_node("Event", "Isolated event")
    archive = _make_archive()
    archive.search = MagicMock(return_value=[{"text": "Jon is an engineer at StarAI.", "meta": {}}])
    localizer = _make_localizer(graph, {"nodes": {isolated: graph.get_node(isolated)}, "edges": []})
    llm = MagicMock()
    llm.complete.side_effect = [
        json.dumps({"action": "jump", "node_ids": [isolated[:8]], "relation_family": "any", "budget": 3}),
        json.dumps({"action": "finish", "answer": "engineer"}),
    ]
    retriever = GraphRetriever(graph=graph, archive=archive, localizer=localizer, llm=llm, max_hop=3)

    result = retriever.answer("What is Jon's job?")

    assert result["answer"] == "engineer"
    assert any(t["action"] == "frontier_exhausted" for t in result["traces"])
    assert any(t["action"] == "raw_fallback" and t["args"].get("forced") for t in result["traces"])


# ---------------------------------------------------------------------------
# answer() — max hop forces finish
# ---------------------------------------------------------------------------

def test_answer_max_hop_forces_finish(tmp_path):
    # LLM always returns jump, never finish → should hit max_hop and force
    retriever = _make_retriever(
        tmp_path,
        llm_responses=[
            '{"action": "jump", "node_ids": [], "relation_family": "any", "budget": 1}',
            '{"action": "jump", "node_ids": [], "relation_family": "any", "budget": 1}',
            '{"action": "jump", "node_ids": [], "relation_family": "any", "budget": 1}',
            "best guess answer",  # forced_answer LLM call
        ],
    )
    result = retriever.answer("What is Jon's job?")
    forced = [t for t in result["traces"] if t.get("action") == "forced_finish"]
    assert len(forced) == 1


# ---------------------------------------------------------------------------
# _execute_jump
# ---------------------------------------------------------------------------

def test_execute_jump_expands_neighbors(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Alice")
    b = graph.add_node("Event", "Lunch")
    c = graph.add_node("Entity", "Bob")
    graph.add_edge(a, b, "entity-event", "attended")
    graph.add_edge(a, c, "entity-entity", "knows")

    archive   = _make_archive()
    localizer = _make_localizer(graph)
    llm = MagicMock()
    retriever = GraphRetriever(graph=graph, archive=archive, localizer=localizer, llm=llm)

    new_nodes, new_edges = retriever._execute_jump(
        node_ids=[a[:8]], family="entity-event", constraint="", budget=5, visited={a}
    )
    assert b in new_nodes
    assert c not in new_nodes  # wrong family


def test_execute_jump_skips_visited(tmp_path):
    graph = _make_graph(tmp_path)
    a = graph.add_node("Entity", "Alice")
    b = graph.add_node("Event", "Lunch")
    graph.add_edge(a, b, "entity-event", "attended")

    archive   = _make_archive()
    localizer = _make_localizer(graph)
    llm = MagicMock()
    retriever = GraphRetriever(graph=graph, archive=archive, localizer=localizer, llm=llm)

    new_nodes, _ = retriever._execute_jump(
        node_ids=[a[:8]], family="any", constraint="", budget=5, visited={a, b}
    )
    assert b not in new_nodes  # already visited


def test_execute_jump_respects_budget(tmp_path):
    graph = _make_graph(tmp_path)
    root = graph.add_node("Entity", "Root")
    neighbors = [graph.add_node("Event", f"E{i}") for i in range(10)]
    for n in neighbors:
        graph.add_edge(root, n, "entity-event", "rel")

    archive   = _make_archive()
    localizer = _make_localizer(graph)
    llm = MagicMock()
    retriever = GraphRetriever(graph=graph, archive=archive, localizer=localizer, llm=llm, jump_budget=3)

    new_nodes, _ = retriever._execute_jump(
        node_ids=[root[:8]], family="any", constraint="", budget=3, visited={root}
    )
    assert len(new_nodes) <= 3


def test_execute_jump_reranks_relevant_neighbors(tmp_path):
    graph = _make_graph(tmp_path)
    root = graph.add_node("Entity", "Alice")
    chat = graph.add_node("Event", "Alice and Bob chat", attrs={"activity": "conversation"})
    class_evt = graph.add_node("Event", "Alice took poetry class", attrs={"activity": "poetry class"})
    graph.add_edge(root, chat, "entity-event", "spoke_to")
    graph.add_edge(root, class_evt, "entity-event", "took")
    retriever = _make_retriever(tmp_path, llm_responses=[])
    retriever.graph = graph

    new_nodes, _ = retriever._execute_jump(
        node_ids=[root[:8]],
        family="entity-event",
        constraint="activity=poetry class",
        budget=1,
        visited={root},
        question="What writing classes has Alice taken?",
    )

    assert list(new_nodes) == [class_evt]


# ---------------------------------------------------------------------------
# _pool
# ---------------------------------------------------------------------------

def test_pool_empty_returns_placeholder(tmp_path):
    retriever = _make_retriever(tmp_path, llm_responses=[])
    text = retriever._pool({}, [], [])
    assert "no evidence" in text.lower()


def test_pool_includes_graph_and_raw(tmp_path):
    graph = _make_graph(tmp_path)
    jon = graph.add_node("Entity", "Jon")
    retriever = _make_retriever(tmp_path, llm_responses=[])
    retriever.graph = graph
    text = retriever._pool(
        {jon: graph.get_node(jon)},
        [],
        ["Jon spoke about his job."],
    )
    assert "Jon" in text
    assert "Jon spoke about his job." in text


def test_canonicalize_final_answer_handles_json_and_labels():
    assert _canonicalize_final_answer('{"answer": "Final answer: Alice."}') == "Alice"
    assert _canonicalize_final_answer("- Alice\n- Bob") == "Alice, Bob"

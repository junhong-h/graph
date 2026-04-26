#!/usr/bin/env python
"""Summarize structural quality of a GraphMemory graph JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EDGE_FAMILIES = {"entity-event", "entity-entity", "event-event"}
LOW_VALUE_EVENT_PREDICATES = {
    "spoke_to",
    "discussed",
    "mentions",
    "mentioned",
    "related_to",
    "replied to",
    "linked_with",
    "connected_to",
}
SESSION_CONTAINER_PATTERNS = re.compile(
    r"\b(chat|chats|conversation|conversations|discuss|discusses|discussed|discussion|discussions)\b|"
    r"\b(speak|speaks|spoke|talk|talks|talked|reply|replies|replied)\s+to\b",
    re.IGNORECASE,
)


def load_graph(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("nodes", {}), data.get("edges", [])


def summarize(path: Path) -> dict[str, Any]:
    nodes, edges = load_graph(path)
    incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        incident[edge.get("src", "")].append(edge)
        incident[edge.get("dst", "")].append(edge)

    node_types = Counter(node.get("type") for node in nodes.values())
    edge_families = Counter(edge.get("family") for edge in edges)
    predicates = Counter(edge.get("predicate") for edge in edges)

    event_ids = [nid for nid, node in nodes.items() if node.get("type") == "Event"]
    entity_ids = [nid for nid, node in nodes.items() if node.get("type") == "Entity"]

    def event_has_entity_edge(nid: str) -> bool:
        return any(edge.get("family") == "entity-event" for edge in incident[nid])

    def is_container_event(node: dict[str, Any]) -> bool:
        attrs = node.get("attrs", {})
        name = str(node.get("canonical_name", ""))
        activity = str(attrs.get("activity", ""))
        return bool(SESSION_CONTAINER_PATTERNS.search(name)) or activity.lower() == "conversation"

    def has_source(node: dict[str, Any]) -> bool:
        attrs = node.get("attrs", {})
        return bool(attrs.get("source_turn_ids") or attrs.get("batch_id"))

    def has_text(node: dict[str, Any]) -> bool:
        attrs = node.get("attrs", {})
        return bool(
            attrs.get("original_text")
            or attrs.get("evidence_quote")
            or attrs.get("message")
            or attrs.get("description")
        )

    invalid_edges = [
        edge for edge in edges
        if edge.get("family") not in EDGE_FAMILIES
        or edge.get("src") not in nodes
        or edge.get("dst") not in nodes
        or not _edge_endpoint_types_match(edge, nodes)
    ]
    container_events = [nid for nid in event_ids if is_container_event(nodes[nid])]
    low_value_event_edges = [
        edge for edge in edges
        if edge.get("family") == "event-event"
        and str(edge.get("predicate", "")).lower() in LOW_VALUE_EVENT_PREDICATES
    ]

    summary = {
        "graph_path": str(path),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": dict(node_types),
        "edge_families": dict(edge_families),
        "top_predicates": predicates.most_common(20),
        "entity_count": len(entity_ids),
        "event_count": len(event_ids),
        "entity_event_ratio": (len(entity_ids) / len(event_ids)) if event_ids else None,
        "invalid_edge_count": len(invalid_edges),
        "event_without_entity_event_edge": sum(
            1 for nid in event_ids if not event_has_entity_edge(nid)
        ),
        "isolated_event_count": sum(1 for nid in event_ids if not incident[nid]),
        "low_degree_event_count": sum(1 for nid in event_ids if len(incident[nid]) <= 2),
        "events_missing_time": sum(
            1 for nid in event_ids if not nodes[nid].get("attrs", {}).get("time")
        ),
        "events_missing_source": sum(1 for nid in event_ids if not has_source(nodes[nid])),
        "events_missing_text": sum(1 for nid in event_ids if not has_text(nodes[nid])),
        "session_container_event_count": len(container_events),
        "low_value_event_edge_count": len(low_value_event_edges),
        "sample_entities": [
            {
                "id": nid[:8],
                "name": nodes[nid].get("canonical_name"),
                "degree": len(incident[nid]),
            }
            for nid in sorted(entity_ids, key=lambda node_id: -len(incident[node_id]))[:25]
        ],
        "sample_problem_events": [
            {
                "id": nid[:8],
                "name": nodes[nid].get("canonical_name"),
                "degree": len(incident[nid]),
                "attrs": nodes[nid].get("attrs", {}),
            }
            for nid in event_ids
            if not incident[nid] or not event_has_entity_edge(nid) or is_container_event(nodes[nid])
        ][:25],
    }
    return summary


def _edge_endpoint_types_match(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> bool:
    src = nodes.get(edge.get("src", ""))
    dst = nodes.get(edge.get("dst", ""))
    if not src or not dst:
        return False
    family = edge.get("family")
    src_type = src.get("type")
    dst_type = dst.get("type")
    if family == "entity-event":
        return src_type == "Entity" and dst_type == "Event"
    if family == "entity-entity":
        return src_type == "Entity" and dst_type == "Entity"
    if family == "event-event":
        return src_type == "Event" and dst_type == "Event"
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.graph)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

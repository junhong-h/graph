"""GraphConstructor: merged Construction + Update in a single LLM call (Step 4 + 5).

The LLM receives the input text and the local subgraph, then outputs a sequence of
graph edit operations.  This module both prompts the LLM and executes the operations
against the GraphStore.

Operations
----------
Construction side (propose new structure):
  CreateEntity  — add a new Entity node
  CreateEvent   — add a new Event node
  Link          — add an edge between two nodes (at least one must be new)
  AttachAttr    — attach key-value attribute to an existing node
  Skip          — nothing worth graphizing

Update side (align with existing graph):
  MergeNode     — merge two nodes that refer to the same object (same_as)
  ReviseAttr    — update an existing attribute on an existing node
  AddEdge       — add an edge between two existing nodes
  DeleteEdge    — remove an edge that is now wrong or stale
  PruneNode     — remove a low-value or redundant node
  KeepSeparate  — explicitly record that two similar nodes are distinct

Node ID references in the prompt use the 8-char prefix of the full UUID.
The LLM is instructed to use these prefixes; we resolve them back to full IDs here.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from graphmemory.graph_store import GraphStore, format_subgraph
from graphmemory.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a graph-memory editor. Given a conversation excerpt and the current local subgraph, \
output a sequence of graph edit operations.

[Node ID convention]
Each node in the subgraph is shown as [XXXXXXXX] (first 8 chars of its UUID). \
Use these 8-char prefixes when referencing existing nodes. \
Use NEW_<name> when referencing a node you are about to create (e.g. NEW_Jon, NEW_Meeting1).

[Available Operations — output as a JSON array]
Each operation is a JSON object with an "op" field plus operation-specific fields.

Construction (new structure):
  {"op": "CreateEntity", "id": "NEW_<label>", "canonical_name": "...", "aliases": ["alt name", ...], "attrs": {"key": "value"}}
  {"op": "CreateEvent",  "id": "NEW_<label>", "canonical_name": "...", "attrs": {"time": "YYYY-MM-DD or description", "key": "value"}}
  {"op": "Link",         "src": "<id>", "dst": "<id>", "family": "entity-event|entity-entity|event-event", "predicate": "..."}
  {"op": "AttachAttr",   "node": "<8-char-id>", "key": "...", "value": "..."}
  {"op": "Skip",         "reason": "..."}

Update (align with existing graph):
  {"op": "MergeNode",    "src": "<8-char-id>", "dst": "<8-char-id>"}
  {"op": "ReviseAttr",   "node": "<8-char-id>", "key": "...", "value": "..."}
  {"op": "AddEdge",      "src": "<8-char-id>", "dst": "<8-char-id>", "family": "...", "predicate": "..."}
  {"op": "DeleteEdge",   "edge": "<8-char-edge-id>"}
  {"op": "PruneNode",    "node": "<8-char-id>"}
  {"op": "KeepSeparate", "node_a": "<8-char-id>", "node_b": "<8-char-id>", "reason": "..."}

[Decision rules]
1. ALWAYS reuse existing nodes from the subgraph before creating new ones — check names carefully.
2. Only CreateEntity/CreateEvent if the object is a long-term anchor (will be referenced again).
2b. Do NOT create generic session-container nodes like "Jon and Gina chat on [date]". \
Every CreateEvent MUST represent a SPECIFIC fact, activity, trip, or occurrence — not a session summary. \
Bad:  {"op": "CreateEvent", "canonical_name": "Jon and Gina chat on Jan 29", ...} \
Good: {"op": "CreateEvent", "canonical_name": "Gina launches ad campaign", "attrs": {"time": "2023-01-29", ...}}
3. Every Event node MUST have a "time" attr — use exact date if stated, else "unknown".
3b. When time is expressed relatively ("next month", "last week", "in two weeks") and the session \
date is visible in the input header, resolve it to an absolute date. \
Example: session date = 2023-02-04, "competition next month" → time: "March 2023".
4. For Event-Event edges with chronological order, use predicate "before" or "after". \
   Use "updates" when an event revises a prior one. Use "inspired" only for causal/creative links.
4b. For event-event edges ONLY use these predicates: before / after / updates / inspired. \
    Do NOT use: discussed / mentions / followed_by / spoke_to / participated / related_to. \
    If two events are temporally ordered, always prefer "before" or "after" over any other label.
5. Use MergeNode when two nodes clearly refer to the same real-world object.
6. Use KeepSeparate when nodes are similar but distinct — prevents future erroneous merges.
7. Link/AddEdge: choose the correct family (entity-event / entity-entity / event-event).
8. Output Skip ONLY if the excerpt is entirely pure pleasantries with ZERO factual content. \
   When in doubt between creating nodes and skipping, ALWAYS prefer creating — it is better \
   to have extra nodes than to lose information.
9. Do NOT output explanatory text — output ONLY the JSON array.
10. VOCABULARY PRESERVATION: For emotional words, adjectives, metaphors, similes, and \
direct quotes, copy the EXACT original wording into attrs — do NOT paraphrase or generalize. \
Example: if the text says "makes me happy", store attrs: {"feeling": "happy"}, NOT \
{"feeling": "fulfilling"} or {"feeling": "brings joy"}. \
If the text says "magical", store "magical", not "stress relief" or "uplifting".
11. TEMPORAL BEHAVIORS: Any activity or behavior mentioned with a temporal anchor \
(a date, month, season, or relative marker like "started", "began", "since", "for the first time") \
MUST become its own separate Event node with the time attr set — do NOT merge it into \
a broader Event that would lose the time. \
Example: "Jon started going to the gym in March 2023" → separate Event node, \
attrs: {"time": "March 2023", "activity": "going to the gym"}.
12. ENTITY-EVENT LINKING: Every CreateEvent SHOULD be followed by at least one Link operation \
connecting it to the relevant Entity node(s) via entity-event family. \
Choose a specific predicate that describes the relationship: \
  experienced / participated / visited / decided / launched / started / owns / achieved / attended \
Avoid generic predicates: spoke_to / discussed / related_to (no semantic value). \
If you are unsure which entity to link to, create the event node first and link to the closest entity. \
Bad:  {"op":"Link","src":"NEW_Gina","dst":"NEW_AdCampaign","family":"entity-event","predicate":"spoke_to"} \
Good: {"op":"Link","src":"NEW_Gina","dst":"NEW_AdCampaign","family":"entity-event","predicate":"launched"}

[Output format]
Return a single valid JSON array. Example:
[
  {"op": "CreateEntity", "id": "NEW_Jon", "canonical_name": "Jon", "aliases": ["Jonathan"], "attrs": {"job": "engineer"}},
  {"op": "CreateEvent",  "id": "NEW_Layoff", "canonical_name": "Jon laid off", "attrs": {"time": "July 2023"}},
  {"op": "Link", "src": "NEW_Jon", "dst": "NEW_Layoff", "family": "entity-event", "predicate": "experienced"}
]\
"""

_USER_PROMPT = """\
[Current local subgraph]
{subgraph_text}

[Input excerpt]
{turn_text}

Output the JSON array of graph edit operations:\
"""


# ---------------------------------------------------------------------------
# GraphConstructor
# ---------------------------------------------------------------------------

@dataclass
class ConstructionContext:
    batch_id: str = ""
    batch_turn_ids: List[str] | None = None
    turn_time: str = ""
    speaker_a: str = ""
    speaker_b: str = ""


class GraphConstructor:
    """Calls LLM to propose + execute graph edits in one step."""

    def __init__(self, llm: LLMClient, graph: GraphStore):
        self.llm   = llm
        self.graph = graph

    def run(
        self,
        turn_text: str,
        local_subgraph: Dict[str, Any],
        context: ConstructionContext | None = None,
    ) -> List[Dict]:
        """
        Prompt the LLM with turn_text + local_subgraph, parse the operations,
        execute them against self.graph, and return the executed operation log.
        """
        subgraph_text = format_subgraph(local_subgraph) if local_subgraph.get("nodes") else "(empty)"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT.format(
                subgraph_text=subgraph_text,
                turn_text=turn_text,
            )},
        ]
        response = self.llm.complete(messages)
        ops = _parse_ops(response)
        logger.debug(f"GraphConstructor: {len(ops)} operations parsed.")
        return self._execute_ops(ops, local_subgraph, turn_text, context)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_ops(
        self,
        ops: List[Dict],
        local_subgraph: Dict[str, Any],
        turn_text: str = "",
        context: ConstructionContext | None = None,
    ) -> List[Dict]:
        """
        Execute operations in order. Track NEW_<label> → full node_id mapping
        so later ops can reference just-created nodes.
        """
        id_map: Dict[str, str] = {}  # NEW_label or 8-char prefix → full node_id
        # Pre-populate id_map with existing subgraph nodes
        for full_id in local_subgraph.get("nodes", {}):
            id_map[full_id[:8]] = full_id

        log: List[Dict] = []
        created_event_ids: List[str] = []
        for op in ops:
            result = self._dispatch(op, id_map, local_subgraph)
            log.append(result)
            if result.get("op") == "CreateEvent" and result.get("status") == "ok":
                created_event_ids.append(result["node_id"])
        if context and created_event_ids:
            log.extend(self._repair_created_events(
                created_event_ids,
                turn_text=turn_text,
                context=context,
                id_map=id_map,
            ))
        return log

    def _dispatch(
        self,
        op: Dict,
        id_map: Dict[str, str],
        local_subgraph: Dict[str, Any],
    ) -> Dict:
        name = op.get("op", "")
        try:
            if name == "CreateEntity":
                return self._do_create_node("Entity", op, id_map)
            elif name == "CreateEvent":
                return self._do_create_node("Event", op, id_map)
            elif name == "Link":
                return self._do_link(op, id_map)
            elif name == "AttachAttr":
                return self._do_attach_attr(op, id_map)
            elif name == "Skip":
                return {"op": "Skip", "status": "ok", "reason": op.get("reason", "")}
            elif name == "MergeNode":
                return self._do_merge(op, id_map)
            elif name == "ReviseAttr":
                return self._do_revise_attr(op, id_map)
            elif name == "AddEdge":
                return self._do_add_edge(op, id_map)
            elif name == "DeleteEdge":
                return self._do_delete_edge(op, local_subgraph)
            elif name == "PruneNode":
                return self._do_prune_node(op, id_map)
            elif name == "KeepSeparate":
                return {"op": "KeepSeparate", "status": "ok",
                        "node_a": op.get("node_a"), "node_b": op.get("node_b"),
                        "reason": op.get("reason", "")}
            else:
                logger.warning(f"Unknown op: {name}")
                return {"op": name, "status": "unknown_op"}
        except Exception as exc:
            logger.warning(f"Op {name} failed: {exc}")
            return {"op": name, "status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Individual op handlers
    # ------------------------------------------------------------------

    def _do_create_node(self, node_type: str, op: Dict, id_map: Dict[str, str]) -> Dict:
        label   = op.get("id", "")
        c_name  = op.get("canonical_name", label)
        aliases = op.get("aliases", [])
        attrs   = op.get("attrs", {})
        node_id = self.graph.add_node(node_type, c_name, aliases=aliases, attrs=attrs)
        id_map[label] = node_id         # NEW_Jon → full uuid
        id_map[node_id[:8]] = node_id   # also register 8-char prefix
        logger.debug(f"Created {node_type} '{c_name}' → {node_id[:8]}")
        return {"op": f"Create{node_type}", "status": "ok", "node_id": node_id,
                "canonical_name": c_name}

    def _do_link(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        src = _resolve(op.get("src", ""), id_map, self.graph)
        dst = _resolve(op.get("dst", ""), id_map, self.graph)
        if not src or not dst:
            return {"op": "Link", "status": "error", "error": f"unresolved id: src={op.get('src')} dst={op.get('dst')}"}
        eid = self.graph.add_edge(src, dst, op.get("family", "entity-entity"), op.get("predicate", "related"))
        if not eid:
            return {"op": "Link", "status": "error", "error": "edge rejected"}
        return {"op": "Link", "status": "ok", "edge_id": eid}

    def _do_attach_attr(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        node_id = _resolve(op.get("node", ""), id_map, self.graph)
        if not node_id:
            return {"op": "AttachAttr", "status": "error", "error": f"unresolved node: {op.get('node')}"}
        self.graph.update_node(node_id, attrs_update={op["key"]: op["value"]})
        return {"op": "AttachAttr", "status": "ok", "node_id": node_id,
                "key": op.get("key"), "value": op.get("value")}

    def _do_merge(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        src = _resolve(op.get("src", ""), id_map, self.graph)
        dst = _resolve(op.get("dst", ""), id_map, self.graph)
        if not src or not dst:
            return {"op": "MergeNode", "status": "error", "error": "unresolved ids"}
        self.graph.merge_nodes(src, dst)
        id_map[src[:8]] = dst  # redirect future references
        return {"op": "MergeNode", "status": "ok", "src": src, "dst": dst}

    def _do_revise_attr(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        node_id = _resolve(op.get("node", ""), id_map, self.graph)
        if not node_id:
            return {"op": "ReviseAttr", "status": "error", "error": f"unresolved node: {op.get('node')}"}
        self.graph.update_node(node_id, attrs_update={op["key"]: op["value"]})
        return {"op": "ReviseAttr", "status": "ok", "node_id": node_id,
                "key": op.get("key"), "value": op.get("value")}

    def _do_add_edge(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        src = _resolve(op.get("src", ""), id_map, self.graph)
        dst = _resolve(op.get("dst", ""), id_map, self.graph)
        if not src or not dst:
            return {"op": "AddEdge", "status": "error", "error": "unresolved ids"}
        eid = self.graph.add_edge(src, dst, op.get("family", "entity-entity"), op.get("predicate", "related"))
        if not eid:
            return {"op": "AddEdge", "status": "error", "error": "edge rejected"}
        return {"op": "AddEdge", "status": "ok", "edge_id": eid}

    def _do_delete_edge(self, op: Dict, local_subgraph: Dict) -> Dict:
        prefix = op.get("edge", "")
        # Resolve edge_id from local subgraph edges
        edge_id = _resolve_edge(prefix, local_subgraph.get("edges", []))
        if not edge_id:
            return {"op": "DeleteEdge", "status": "error", "error": f"edge not found: {prefix}"}
        self.graph.delete_edge(edge_id)
        return {"op": "DeleteEdge", "status": "ok", "edge_id": edge_id}

    def _do_prune_node(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        node_id = _resolve(op.get("node", ""), id_map, self.graph)
        if not node_id:
            return {"op": "PruneNode", "status": "error", "error": f"unresolved node: {op.get('node')}"}
        self.graph.delete_node(node_id)
        return {"op": "PruneNode", "status": "ok", "node_id": node_id}

    def _repair_created_events(
        self,
        event_ids: List[str],
        turn_text: str,
        context: ConstructionContext,
        id_map: Dict[str, str],
    ) -> List[Dict]:
        logs: List[Dict] = []
        for event_id in event_ids:
            event = self.graph.get_node(event_id)
            if not event:
                continue

            attrs_update: Dict[str, Any] = {}
            attrs = event.get("attrs", {})
            if context.turn_time and not attrs.get("time"):
                attrs_update["time"] = context.turn_time
            if context.batch_id and not attrs.get("batch_id"):
                attrs_update["batch_id"] = context.batch_id
            if context.batch_turn_ids and not attrs.get("source_turn_ids"):
                attrs_update["source_turn_ids"] = context.batch_turn_ids
            if turn_text and not (attrs.get("original_text") or attrs.get("evidence_quote")):
                attrs_update["original_text"] = turn_text
            if attrs_update:
                self.graph.update_node(event_id, attrs_update=attrs_update)
                logs.append({
                    "op": "RepairEventAttrs",
                    "status": "ok",
                    "node_id": event_id,
                    "attrs": sorted(attrs_update.keys()),
                })

            if self._has_entity_event_edge(event_id):
                continue
            speaker = _first_speaker(turn_text) or context.speaker_a or context.speaker_b
            entity_id = self._find_best_entity_for_event(event_id, speaker)
            if not entity_id and speaker:
                entity_id = self.graph.add_node("Entity", speaker, aliases=[speaker])
                id_map[entity_id[:8]] = entity_id
                logs.append({
                    "op": "RepairCreateEntity",
                    "status": "ok",
                    "node_id": entity_id,
                    "canonical_name": speaker,
                })
            if entity_id:
                eid = self.graph.add_edge(entity_id, event_id, "entity-event", "participant")
                logs.append({
                    "op": "RepairEventLink",
                    "status": "ok" if eid else "error",
                    "edge_id": eid,
                    "src": entity_id,
                    "dst": event_id,
                })
        return logs

    def _has_entity_event_edge(self, event_id: str) -> bool:
        return any(e.get("family") == "entity-event" for e in self.graph.get_edges(node_id=event_id))

    def _find_best_entity_for_event(self, event_id: str, preferred_name: str = "") -> Optional[str]:
        event = self.graph.get_node(event_id) or {}
        haystack = " ".join([
            event.get("canonical_name", ""),
            " ".join(str(v) for v in event.get("attrs", {}).values()),
            preferred_name,
        ]).lower()
        entities = [
            (nid, node)
            for nid, node in self.graph.get_all_nodes().items()
            if node.get("type") == "Entity"
        ]
        if preferred_name:
            for nid, node in entities:
                if node.get("canonical_name", "").lower() == preferred_name.lower():
                    return nid
        for nid, node in entities:
            names = [node.get("canonical_name", "")] + node.get("aliases", [])
            if any(name and name.lower() in haystack for name in names):
                return nid
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_ops(response: str) -> List[Dict]:
    """Extract a JSON array of operations from LLM response.

    Tries JSON first; falls back to ast.literal_eval for Python-dict style output.
    """
    import ast

    # Find the outermost [...] block
    m = re.search(r"\[.*\]", response, re.DOTALL)
    if not m:
        logger.warning("GraphConstructor: no JSON array found in response.")
        return []

    raw = m.group()

    # Attempt 1: strict JSON
    try:
        ops = json.loads(raw)
        if isinstance(ops, list):
            return ops
        logger.warning("GraphConstructor: parsed JSON is not a list.")
        return []
    except json.JSONDecodeError:
        pass

    # Attempt 2: Python literal (single-quoted dicts)
    try:
        ops = ast.literal_eval(raw)
        if isinstance(ops, list):
            logger.debug("GraphConstructor: parsed via ast.literal_eval fallback.")
            return ops
    except (ValueError, SyntaxError, TypeError):
        pass

    logger.warning(f"GraphConstructor: failed to parse ops from response: {raw[:120]!r}")
    return []


def _resolve(ref: str, id_map: Dict[str, str], graph: GraphStore | None = None) -> Optional[str]:
    """Resolve a NEW_label or 8-char prefix to a full node_id."""
    if not ref:
        return None
    ref = _clean_ref(ref)
    if ref in id_map:
        return id_map[ref]
    # Try prefix match in id_map values
    for full_id in id_map.values():
        if full_id.startswith(ref):
            return full_id
    if graph is not None:
        all_nodes = graph.get_all_nodes()
        for full_id in all_nodes:
            if full_id.startswith(ref):
                return full_id
        ref_lower = ref.lower()
        for full_id, node in all_nodes.items():
            names = [node.get("canonical_name", "")] + node.get("aliases", [])
            if any(name and name.lower() == ref_lower for name in names):
                return full_id
    return None


def _clean_ref(ref: str) -> str:
    ref = str(ref or "").strip()
    ref = ref.strip("\"'")
    if ref.startswith("[") and ref.endswith("]"):
        ref = ref[1:-1].strip()
    return ref


def _first_speaker(turn_text: str) -> str:
    for line in str(turn_text or "").splitlines():
        m = re.match(r"\s*([^:\n]+?)\s+speak to\s+[^:]+?:", line)
        if m:
            return m.group(1).strip()
    return ""


def _resolve_edge(prefix: str, edges: List[Dict]) -> Optional[str]:
    """Find a full edge_id by 8-char prefix."""
    for edge in edges:
        if edge.get("edge_id", "").startswith(prefix):
            return edge["edge_id"]
    return None

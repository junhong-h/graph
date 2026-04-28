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
output graph edit operations as a JSON object {"ops": [...]}.

[Node IDs]
Existing nodes are shown as [XXXXXXXX] (8-char UUID prefix) — use these to reference them. \
Use NEW_<label> for nodes you are creating.

[Operations]
{"op": "EnsureEntity", "id": "NEW_<label>", "canonical_name": "...", "aliases": [...]}
{"op": "EnsureEvent",  "id": "NEW_<label>", "canonical_name": "...", "attrs": {"fact": "On <session date>, <speaker> <fact>.", "time": "<event date or period if known>"}}
{"op": "Relate",       "src": "<id>", "dst": "<id>", "predicate": "..."}
{"op": "AttachAttr",   "node": "<8-char-id>", "key": "...", "value": "..."}
{"op": "MergeNode",    "src": "<8-char-id>", "dst": "<8-char-id>"}
{"op": "Skip",         "reason": "..."}

Edge family (entity-event / event-event / entity-entity) is auto-inferred from node types — \
do NOT specify it.

[Rules]
1. Always reuse existing nodes before creating new ones. EnsureEntity/EnsureEvent are idempotent.
2. EnsureEntity for any stable, answerable concept: people, places, organizations, objects, \
activities, pets, projects, credentials — anything that could be a direct answer to a memory \
question. Do NOT create Entity nodes for abstract virtues, slogans, or generic categories.
3. EnsureEvent for concrete personal facts that may need to be recalled: activities, possessions, \
trips, achievements, plans, relationships, health/work/school changes, dated occurrences. \
attrs.fact is required — write a self-contained sentence that includes the session date as context. \
attrs.time is optional — set it when the event has its own specific date or period. \
If the source uses relative time ("last week", "two years ago"), preserve that phrasing in \
attrs.fact; do NOT infer or fabricate exact dates. \
Do NOT create Events for: \
  (a) social reactions — thanking, praising, encouraging, admiring, or reacting to news; \
  (b) abstract values or beliefs — aspirations, life philosophies, general attitudes; \
  (c) conversational acts — sharing a photo, mentioning something, having a discussion.
4. For a fact where a subject acts on an answerable object: create/reuse subject Entity, \
create/reuse object Entity, create one Event, then Relate both Entities to the Event.
5. Every EnsureEvent MUST be followed by at least one Relate to a relevant Entity. \
Entity↔Event predicates: participant / experienced / owns / attended / visited / decided / \
started / achieved / object_of. Event→Event predicates: before / after / updates / inspired.
6. Any activity with a temporal anchor (date, "started", "since", "for the first time") \
MUST be a separate Event with attrs.time set.
7. Use MergeNode when two nodes clearly refer to the same real-world object.
8. Output Skip only when every turn contains only excluded content (a-c above).
9. Output ONLY the JSON object — no explanatory text.

[Output format]
Return a single valid JSON object. Example:
{
  "ops": [
    {"op": "CreateEntity", "id": "NEW_Person", "canonical_name": "<person name>", "aliases": [], "attrs": {}},
    {"op": "CreateEvent",  "id": "NEW_Event", "canonical_name": "<subject action object>", "attrs": {"fact": "<self-contained fact sentence>", "quote": "<short exact source quote>", "source": ["<turn_id>"], "time": "<compatibility time if useful>"}},
    {"op": "Link", "src": "NEW_Person", "dst": "NEW_Event", "family": "entity-event", "predicate": "participant"}
  ]
}\
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

        If the LLM returns a batch-level Skip (all ops are Skip) and the batch
        contains multiple turns, retry per-turn so a single embedded fact does
        not get dropped together with surrounding pleasantries.
        """
        op_log = self._call_llm_and_execute(turn_text, local_subgraph, context)

        if (
            self._is_batch_skip(op_log)
            and context is not None
            and context.batch_turn_ids
            and len(context.batch_turn_ids) > 1
        ):
            turn_blocks = self._split_batch_turns(turn_text)
            if len(turn_blocks) > 1:
                logger.debug(
                    f"GraphConstructor: batch-level Skip detected for "
                    f"{context.batch_id}; retrying per-turn ({len(turn_blocks)} turns)."
                )
                op_log = self._retry_per_turn(turn_blocks, local_subgraph, context, op_log)

        return op_log

    def _call_llm_and_execute(
        self,
        turn_text: str,
        local_subgraph: Dict[str, Any],
        context: ConstructionContext | None,
    ) -> List[Dict]:
        subgraph_text = format_subgraph(local_subgraph) if local_subgraph.get("nodes") else "(empty)"
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT.format(
                subgraph_text=subgraph_text,
                turn_text=turn_text,
            )},
        ]
        response = self.llm.complete(messages, json_mode=True)
        ops = _parse_ops(response)
        logger.debug(f"GraphConstructor: {len(ops)} operations parsed.")
        return self._execute_ops(ops, local_subgraph, turn_text, context)

    @staticmethod
    def _is_batch_skip(op_log: List[Dict]) -> bool:
        """True if op_log contains only Skip ops (batch-level skip)."""
        if not op_log:
            return False
        return all(op.get("op") == "Skip" for op in op_log)

    @staticmethod
    def _split_batch_turns(turn_text: str) -> List[str]:
        """Split a batch_text built by '\\n\\n'.join(turn_blocks) back into
        individual turn blocks. Each block keeps its [turn_id=...] header."""
        return [b for b in turn_text.split("\n\n") if b.strip()]

    def _retry_per_turn(
        self,
        turn_blocks: List[str],
        local_subgraph: Dict[str, Any],
        context: ConstructionContext,
        original_skip_log: List[Dict],
    ) -> List[Dict]:
        """Re-call the LLM independently for each turn in the batch.

        local_subgraph is reused across per-turn calls; downstream graph state
        (self.graph) is shared so later per-turn calls can still observe nodes
        materialized by earlier per-turn calls via _execute_ops side effects.
        """
        merged: List[Dict] = []
        merged.append({
            "op": "BatchSkipRecovery",
            "status": "ok",
            "reason": (original_skip_log[0].get("reason", "") if original_skip_log else ""),
            "n_turns": len(turn_blocks),
        })
        for turn_block in turn_blocks:
            single_log = self._call_llm_and_execute(turn_block, local_subgraph, context)
            for entry in single_log:
                entry["per_turn_recovery"] = True
            merged.extend(single_log)
        return merged

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
            if result.get("op") in ("CreateEvent", "EnsureEvent") and result.get("status") == "ok":
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
            if name in ("EnsureEntity", "CreateEntity"):
                return self._do_create_node("Entity", op, id_map)
            elif name in ("EnsureEvent", "CreateEvent"):
                return self._do_create_node("Event", op, id_map)
            elif name == "Relate":
                return self._do_relate(op, id_map)
            elif name == "Link":                        # backward compat
                return self._do_link(op, id_map)
            elif name == "AttachAttr":
                return self._do_attach_attr(op, id_map)
            elif name == "Skip":
                return {"op": "Skip", "status": "ok", "reason": op.get("reason", "")}
            elif name == "MergeNode":
                return self._do_merge(op, id_map)
            elif name in ("AddEdge", "ReviseAttr", "DeleteEdge", "PruneNode", "KeepSeparate"):
                logger.debug(f"Deprecated op ignored: {name}")
                return {"op": name, "status": "deprecated"}
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
        if node_type == "Event" and _is_session_container_event(c_name, attrs):
            return {
                "op": "CreateEvent",
                "status": "rejected",
                "error": "session-container event rejected",
                "canonical_name": c_name,
            }
        if node_type == "Event" and _is_low_value_abstract_event(c_name, attrs):
            return {
                "op": "CreateEvent",
                "status": "rejected",
                "error": "low-value abstract event rejected",
                "canonical_name": c_name,
            }
        if node_type == "Entity":
            existing_id = _find_existing_entity(c_name, self.graph)
            if existing_id:
                if aliases or attrs:
                    self.graph.update_node(existing_id, new_aliases=aliases, attrs_update=attrs)
                id_map[label] = existing_id
                id_map[existing_id[:8]] = existing_id
                return {
                    "op": "CreateEntity",
                    "status": "ok",
                    "node_id": existing_id,
                    "canonical_name": c_name,
                    "reused": True,
                }
        node_id = self.graph.add_node(node_type, c_name, aliases=aliases, attrs=attrs)
        id_map[label] = node_id         # NEW_<label> → full uuid
        id_map[node_id[:8]] = node_id   # also register 8-char prefix
        logger.debug(f"Created {node_type} '{c_name}' → {node_id[:8]}")
        return {"op": f"Create{node_type}", "status": "ok", "node_id": node_id,
                "canonical_name": c_name}

    def _do_relate(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        """Relate: like Link but auto-infers edge family from node types."""
        src = _resolve(op.get("src", ""), id_map, self.graph)
        dst = _resolve(op.get("dst", ""), id_map, self.graph)
        if not src or not dst:
            return {"op": "Relate", "status": "error",
                    "error": f"unresolved id: src={op.get('src')} dst={op.get('dst')}"}
        src_type = (self.graph.get_node(src) or {}).get("type", "")
        dst_type = (self.graph.get_node(dst) or {}).get("type", "")
        types = {src_type, dst_type}
        if types == {"Event"}:
            family = "event-event"
        elif "Event" in types and "Entity" in types:
            family = "entity-event"
        else:
            family = "entity-entity"
        predicate = _normalize_predicate(family, op.get("predicate", "related"))
        eid = self.graph.add_edge(src, dst, family, predicate)
        if not eid:
            return {"op": "Relate", "status": "error", "error": "edge rejected"}
        return {"op": "Relate", "status": "ok", "edge_id": eid, "family": family}

    def _do_link(self, op: Dict, id_map: Dict[str, str]) -> Dict:
        src = _resolve(op.get("src", ""), id_map, self.graph)
        dst = _resolve(op.get("dst", ""), id_map, self.graph)
        if not src or not dst:
            return {"op": "Link", "status": "error", "error": f"unresolved id: src={op.get('src')} dst={op.get('dst')}"}
        family = op.get("family", "entity-entity")
        predicate = _normalize_predicate(family, op.get("predicate", "related"))
        eid = self.graph.add_edge(src, dst, family, predicate)
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
        family = op.get("family", "entity-entity")
        predicate = _normalize_predicate(family, op.get("predicate", "related"))
        eid = self.graph.add_edge(src, dst, family, predicate)
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
            if context.batch_turn_ids and not attrs.get("source"):
                attrs_update["source"] = context.batch_turn_ids
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
            return None
        for nid, node in entities:
            names = [node.get("canonical_name", "")] + node.get("aliases", [])
            if any(name and name.lower() in haystack for name in names):
                return nid
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_ops(response: str) -> List[Dict]:
    """Extract graph operations from an LLM JSON response.

    Tries JSON first; falls back to ast.literal_eval for Python-dict style output.
    Supports both the current {"ops": [...]} envelope and legacy top-level arrays.
    """
    import ast

    def _coerce_ops(parsed: Any) -> List[Dict] | None:
        if isinstance(parsed, dict) and isinstance(parsed.get("ops"), list):
            return parsed["ops"]
        if isinstance(parsed, list):
            return parsed
        return None

    text = str(response or "").strip()
    if not text:
        logger.warning("GraphConstructor: empty response.")
        return []

    try:
        ops = _coerce_ops(json.loads(text))
        if ops is not None:
            return ops
    except json.JSONDecodeError:
        pass

    # Find an enveloped object first, then a legacy [...] block.
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        raw = m.group()
        try:
            ops = _coerce_ops(json.loads(raw))
            if ops is not None:
                return ops
        except json.JSONDecodeError:
            pass
        try:
            ops = _coerce_ops(ast.literal_eval(raw))
            if ops is not None:
                logger.debug("GraphConstructor: parsed object via ast.literal_eval fallback.")
                return ops
        except (ValueError, SyntaxError, TypeError):
            pass

    m = re.search(r"\[.*\]", response, re.DOTALL)
    if not m:
        logger.warning("GraphConstructor: no JSON ops found in response.")
        return []

    raw = m.group()

    # Attempt 1: strict JSON
    try:
        ops = _coerce_ops(json.loads(raw))
        if ops is not None:
            return ops
    except json.JSONDecodeError:
        pass

    # Attempt 2: Python literal (single-quoted dicts)
    try:
        ops = _coerce_ops(ast.literal_eval(raw))
        if ops is not None:
            logger.debug("GraphConstructor: parsed via ast.literal_eval fallback.")
            return ops
    except (ValueError, SyntaxError, TypeError):
        pass

    logger.warning(f"GraphConstructor: failed to parse ops from response: {raw[:1000]!r}")
    return []


def _is_session_container_event(canonical_name: str, attrs: Dict[str, Any]) -> bool:
    """Reject low-value event nodes that only represent a conversation container."""
    name = str(canonical_name or "").strip().lower()
    activity = str((attrs or {}).get("activity", "")).strip().lower()
    if activity == "conversation":
        return True
    return bool(
        re.search(r"\b(chat|chats|conversation|conversations|discuss|discusses|discussed|discussion|discussions)\b", name)
        or re.search(r"\b(speak|speaks|spoke|talk|talks|talked|reply|replies|replied)\s+to\b", name)
    )


def _is_low_value_abstract_event(canonical_name: str, attrs: Dict[str, Any]) -> bool:
    """Reject generic opinions or value statements that are not durable memory facts."""
    name = str(canonical_name or "").strip().lower()
    attrs = attrs or {}
    fact = str(attrs.get("fact", "") or "").strip().lower()
    quote = str(attrs.get("quote", "") or "").strip().lower()
    text = " ".join([name, fact, quote])

    if not name:
        return False

    generic_subject = re.match(
        r"^(animals|people|kindness|positivity|positive impact|support|self-care|taking care of self)\b",
        name,
    )
    abstract_action = re.search(
        r"\b(spreading positivity|spread kindness|make a difference|working together|community bond|"
        r"animals are|animals comfort|animals bring|animals provide|always there for us|"
        r"taking care of (ourselves|self)|importance of)\b",
        text,
    )
    discussion_only = re.search(
        r"\b(discussed|agreed on|emphasizing|noting)\b.*\b(importance|value|amazing|positive|positivity|kindness)\b",
        text,
    )

    return bool(generic_subject or abstract_action or discussion_only)


def _normalize_predicate(family: str, predicate: str) -> str:
    """Replace low-value traversal predicates with safer graph roles."""
    family = str(family or "")
    pred = str(predicate or "related").strip()
    if family in {"entity-event", "event-entity"} and pred.lower() in {
        "spoke_to",
        "discussed",
        "related_to",
        "mentioned",
        "mentions",
        "replied to",
    }:
        return "participant"
    return pred


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
        ref_names = _candidate_ref_names(ref)
        ref_norms = {_normalize_name(name) for name in ref_names}
        for full_id, node in all_nodes.items():
            names = [node.get("canonical_name", "")] + node.get("aliases", [])
            if any(_normalize_name(name) in ref_norms for name in names if name):
                return full_id
    return None


def _clean_ref(ref: str) -> str:
    ref = str(ref or "").strip()
    ref = ref.strip("\"'")
    if ref.startswith("[") and ref.endswith("]"):
        ref = ref[1:-1].strip()
    return ref


def _candidate_ref_names(ref: str) -> List[str]:
    names = [ref]
    if ref.startswith("NEW_"):
        label = ref[4:]
        if label.lower() not in {"event", "entity", "node"}:
            names.append(label)
            names.append(re.sub(r"(?<!^)(?=[A-Z])", " ", label))
    return names


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def _find_existing_entity(canonical_name: str, graph: GraphStore) -> Optional[str]:
    target = _normalize_name(canonical_name)
    if not target:
        return None
    for node_id, node in graph.get_all_nodes().items():
        if node.get("type") != "Entity":
            continue
        names = [node.get("canonical_name", "")] + node.get("aliases", [])
        if any(_normalize_name(name) == target for name in names if name):
            return node_id
    return None


def _first_speaker(turn_text: str) -> str:
    for line in str(turn_text or "").splitlines():
        header = re.match(r"\s*\[[^\]]*\bspeaker=([^;\]]+)", line)
        if header:
            return header.group(1).strip()
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

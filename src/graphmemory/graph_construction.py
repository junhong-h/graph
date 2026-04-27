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
Use NEW_<name> when referencing a node you are about to create (e.g. NEW_Person, NEW_Event1).

[Available Operations — output as JSON]
Output a single JSON object with key "ops". "ops" is an array of operation objects.
Each operation object has an "op" field plus operation-specific fields.

Construction (new structure):
  {"op": "CreateEntity", "id": "NEW_<label>", "canonical_name": "...", "aliases": ["alt name", ...], "attrs": {"key": "value"}}
  {"op": "CreateEvent",  "id": "NEW_<label>", "canonical_name": "...", "attrs": {"fact": "self-contained factual sentence", "quote": "short source quote", "source": ["turn_id"], "time": "YYYY-MM-DD or description", "key": "value"}}
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
0. INPUT FORMAT: Each turn may be formatted as:
   [turn_id=...; speaker=...; listener=...; session_time=...]
   utterance text
   The bracketed header is metadata only. Do NOT create Events for the header, speaker/listener,
   or the act of talking. Use speaker as the default subject for first-person utterances, and use
   session_time only as a time anchor/default when the utterance gives no better event time.
   Extract graph facts only from the utterance text after the header.
1. ALWAYS reuse existing nodes from the subgraph before creating new ones — check names carefully.
2. CreateEntity for stable answerable objects or concepts, not only people. \
Examples include classes, books, songs, cities, diseases, hobbies, projects, pets, family members, \
organizations, causes, places, foods, certificates/degrees, and important objects. \
If a phrase could be the direct answer to a QA question, you MUST create/reuse an Entity for it. \
This includes named activities, named places, named pets, specific causes, specific items, and \
specific credentials or awards.
Do NOT create Entity nodes for generic categories, slogans, virtues, or verb phrases unless the \
conversation names them as a specific reusable thing.
2b. Do NOT create generic session-container nodes like "<Person A> and <Person B> chat on <date>". \
Every CreateEvent MUST represent a SPECIFIC fact, activity, trip, or occurrence — not a session summary. \
Bad:  {"op": "CreateEvent", "canonical_name": "<Person A> and <Person B> chat", ...} \
Bad:  {"op": "CreateEvent", "canonical_name": "<date> conversation", ...} \
Bad:  {"op": "CreateEvent", "canonical_name": "<Person A> and <Person B> discuss <topic>", ...} \
Bad:  {"op": "CreateEvent", "canonical_name": "<Person A> speaks to <Person B>", ...} \
Good: {"op": "CreateEvent", "canonical_name": "<subject> <action> <object>", "attrs": {"time": "<event time>", ...}} \
If a turn only says that one speaker talked to another speaker, do NOT create an Event. If the turn \
contains facts, extract those underlying facts as separate Events.
2c. Event nodes represent "who did/experienced/said/planned what, when". \
Entity nodes represent reusable answer values. \
Create an Event only when the utterance contains a concrete personal fact, state change, plan, \
achievement, preference, relationship, possession, trip, activity, or incident that may later need \
to be recalled. Do NOT create Events for generic opinions, inspirational slogans, broad values, \
or abstract agreement without a concrete answerable detail.
For a statement where a subject does an activity involving an answerable object, create/reuse \
the subject Entity, create/reuse the object Entity, then create one Event linking both.
2d. NEVER store answerable activities/objects as comma-separated attrs on a person Entity. \
Bad: AttachAttr <person> activity="<activity 1>, <activity 2>, <activity 3>". \
Good: create/reuse each answerable activity/object as an Entity, create a specific Event for each \
fact, and link the subject and object Entities to that Event.
3. Every Event node MUST have attrs.fact, attrs.quote, and attrs.source. \
fact is a self-contained factual sentence that can be understood without reading the original turn. \
quote is the shortest exact source text that supports the fact. source is the supporting turn_id list.
If the input header has session_time, attrs.fact MUST include that mention time as context, usually \
as "On <session date/time>, <speaker> said ...". \
3b. If the source uses relative time ("last night", "last month", "two weeks ago", "for 10 years"), \
write the relation into attrs.fact with the mention date from the turn header. Do NOT rewrite a \
relative-time fact as if it happened on the mention date. Do NOT infer exact dates unless directly \
stated by the source. Keep any legacy "time" attr as a rough compatibility field only.
4. For Event-Event edges with chronological order, use predicate "before" or "after". \
   Use "updates" when an event revises a prior one. Use "inspired" only for causal/creative links.
4b. For event-event edges ONLY use these predicates: before / after / updates / inspired. \
    Do NOT use: discussed / mentions / followed_by / spoke_to / participated / related_to. \
    If two events are temporally ordered, always prefer "before" or "after" over any other label.
5. Use MergeNode when two nodes clearly refer to the same real-world object.
6. Use KeepSeparate when nodes are similar but distinct — prevents future erroneous merges.
7. Link/AddEdge: choose the correct family (entity-event / entity-entity / event-event). \
Most factual links should be entity-event. Use entity-entity only for stable relationships, \
and event-event only for real temporal, update, or causal links.
8. Output Skip if the excerpt is pure pleasantries OR only generic opinion/value statements with \
   no concrete personal fact to remember. Preserve factual details, but do not graph broad \
   affirmations like "kindness matters", "animals are amazing", or "we should stay positive" \
   unless the utterance also states a specific person, object, event, decision, or preference.
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
Example pattern: "<subject> started <activity> in <time>" → separate Event node with \
attrs: {"time": "<time>", "activity": "<activity>"}.
12. ENTITY-EVENT LINKING: Every CreateEvent MUST be followed by at least one Link operation \
connecting it to the relevant Entity node(s) via entity-event family. \
Choose a specific predicate that describes the relationship: \
  participant / experienced / took / attended / visited / decided / launched / started / owns / achieved / object_of \
Avoid generic predicates: spoke_to / discussed / related_to (no semantic value). \
If an Event has an object that could be an answer value, link that object Entity to the Event too. \
If you are unsure which entity to link to, create the event node first and link to the closest speaker/entity. \
Bad:  {"op":"Link","src":"NEW_Person","dst":"NEW_Event","family":"entity-event","predicate":"spoke_to"} \
Good: {"op":"Link","src":"NEW_Person","dst":"NEW_Event","family":"entity-event","predicate":"participant"} \
Good: {"op":"Link","src":"NEW_Object","dst":"NEW_Event","family":"entity-event","predicate":"object_of"}

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
        """
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

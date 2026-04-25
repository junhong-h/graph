"""GraphRetriever: answer questions via graph traversal (Steps 6-11).

Retrieval pipeline
------------------
Step 6   Localize     — find query-relevant starting area (reuses GraphLocalizer)
Step 7   SelectAnchor — choose initial anchor nodes from the localized subgraph
Step 8   Jump         — LLM-driven controlled graph expansion (frontier-based)
Step 9   Pool         — compress visited subgraph to natural-language evidence
Step 10  Finish       — LLM explicit decision to stop and answer
Step 11  Raw Fallback — fall back to raw archive if graph evidence is insufficient

Jump is a ReAct loop where the LLM calls one of:
  jump(node_ids, relation_family, constraint, budget)  — expand frontier
  finish(answer)                                        — stop, return answer
  raw_fallback(query)                                   — search raw turns

Stopping rules (Q3 answer: both-combined):
  - LLM calls finish → stop immediately
  - hop count reaches max_hop → force finish regardless of LLM decision
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from graphmemory.graph_localize import GraphLocalizer
from graphmemory.graph_store import GraphStore, format_subgraph
from graphmemory.llm_client import LLMClient
from graphmemory.raw_archive import RawArchive


# ---------------------------------------------------------------------------
# Answer format (same as before for LoCoMo compatibility)
# ---------------------------------------------------------------------------

def _locomo_format(category: str) -> str:
    if str(category) == "3":
        return (
            "The Final Result's Format Must Follow These Rules:\n"
            "1. Provide a short phrase answer, not a full sentence.\n"
            "2. Answer directly — do NOT start with 'Yes' or 'No' unless the question is literally a yes/no question.\n"
            "3. NEVER answer 'I don't know' or 'Unknown'. Uncertain inferences → use 'likely'.\n"
            "4. Multiple phrases: connect with commas, not 'and'.\n"
            "5. When max steps reached, MUST call finish with your best guess."
        )
    return (
        "The Final Result's Format Must Follow These Rules:\n"
        "1. Answer directly — do NOT start with 'Yes' or 'No' unless the question literally asks for yes/no.\n"
        "2. Date/time answers: use '15 July, 2023' or 'July, 2023'.\n"
        "3. Short phrase, not a full sentence.\n"
        "4. Exact wording from the original conversation when possible.\n"
        "5. Multiple phrases: connect with commas, not 'and'.\n"
        "6. NEVER answer 'Unknown' or 'I don't know'. When max steps reached, "
        "MUST call finish with best guess."
    )


def get_answer_format(benchmark: str = "locomo", category: str = "") -> str:
    if benchmark == "locomo":
        return _locomo_format(category)
    return "Answer with a short phrase. NEVER say 'Unknown' or 'I don't know'."


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a graph-memory retrieval agent. Answer the user's question by exploring a \
knowledge graph and retrieving raw conversation turns.

[Available actions — output as a JSON object]

1. Jump: expand your search frontier along graph edges.
   {{"action": "jump", "node_ids": ["<8-char-id>", ...], "relation_family": "entity-event|entity-entity|event-event|any", "constraint": "<optional filter, e.g. time range or node type>", "budget": <int 1-5>}}

2. Raw Fallback: search raw conversation turns when graph evidence is insufficient.
   {{"action": "raw_fallback", "query": "<search query>"}}

3. Finish: return the final answer when confident.
   {{"action": "finish", "answer": "<concise answer>"}}

[Rules]
- Start from the anchor nodes provided, then Jump to explore.
- Jump along entity-event edges to find what happened to people/places.
- Jump along event-event edges (before/after/updates) for temporal chains.
- Jump along entity-entity edges carefully — easiest to over-expand.
- After each Jump, decide: is the evidence enough to finish? If yes, call finish.
- If the graph lacks key details (exact quotes, fine-grained facts), use raw_fallback.
- You may use raw_fallback AND jump in the same session.
- Only after you have jumped at least once and still find NO relevant evidence anywhere in the \
graph, call finish with answer "Not mentioned in the conversation". \
Do NOT say "Not mentioned" on the first hop — always Jump first to explore before concluding \
the information is absent.
- When max hops are reached you MUST call finish with your best guess.
- Do NOT output any text outside the JSON object.

[Answer format]
{answer_format}

[Current evidence]
{evidence}

Respond with ONLY a single JSON object.\
"""

_USER_PROMPT = "Question: {question}"


# ---------------------------------------------------------------------------
# GraphRetriever
# ---------------------------------------------------------------------------

class GraphRetriever:
    """Runs Steps 6-11 to answer a question from graph + raw archive."""

    def __init__(
        self,
        graph: GraphStore,
        archive: RawArchive,
        localizer: GraphLocalizer,
        llm: LLMClient,
        retrieval_topk: int = 5,
        max_hop: int = 3,
        jump_budget: int = 5,
        benchmark: str = "locomo",
    ):
        self.graph          = graph
        self.archive        = archive
        self.localizer      = localizer
        self.llm            = llm
        self.retrieval_topk = retrieval_topk
        self.max_hop        = max_hop
        self.jump_budget    = jump_budget
        self.benchmark      = benchmark

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def answer(self, question: str, category: str = "") -> Dict[str, Any]:
        """Run the full retrieval pipeline and return {"answer": ..., "traces": [...]}."""
        answer_format = get_answer_format(self.benchmark, category)

        # Step 6: Localize
        local_sub = self.localizer.localize(question)

        # Step 7: SelectAnchor — LLM selects 1-3 anchors from local subgraph
        local_nodes = local_sub.get("nodes", {})
        anchor_ids = self._select_anchor(question, local_sub) if local_nodes else []
        frontier: Set[str] = set(anchor_ids)
        visited:  Set[str] = set(frontier)
        # Evidence starts with full local subgraph (not just anchors)
        evidence_nodes: Dict[str, Dict] = dict(local_nodes)
        evidence_edges: List[Dict]      = list(local_sub.get("edges", []))
        raw_context: List[str]          = []
        traces:      List[Dict]         = []

        logger.debug(f"Retrieval anchors: {len(frontier)} nodes (selected from {len(local_nodes)}).")

        # Steps 8-10: Jump loop
        for hop in range(self.max_hop):
            evidence_text = self._pool(evidence_nodes, evidence_edges, raw_context)
            response      = self._call_llm(question, evidence_text, answer_format)
            action, args  = _parse_action(response)

            trace = {
                "hop":    hop,
                "op_id":  str(uuid.uuid4()),
                "action": action,
                "args":   args,
            }
            traces.append(trace)
            logger.debug(f"Hop {hop}: action={action}, args={args}")

            if action == "finish":
                return {"answer": args.get("answer", "").strip(), "traces": traces}

            elif action == "jump":
                new_nodes, new_edges = self._execute_jump(
                    node_ids=args.get("node_ids", list(frontier)),
                    family=args.get("relation_family", "any"),
                    constraint=args.get("constraint", ""),
                    budget=min(int(args.get("budget", self.jump_budget)), self.jump_budget),
                    visited=visited,
                )
                evidence_nodes.update(new_nodes)
                evidence_edges.extend(new_edges)
                frontier = set(new_nodes.keys())
                visited.update(frontier)

            elif action == "raw_fallback":
                hits = self.archive.search(args.get("query", question), top_k=self.retrieval_topk)
                raw_context.extend(h["text"] for h in hits if h["text"] not in raw_context)

            else:
                logger.warning(f"Unknown action: {action!r} — treated as no-op.")

            # If frontier is empty, no more graph expansion is possible
            if not frontier and action == "jump":
                logger.debug("Frontier exhausted; forcing raw_fallback + finish.")
                hits = self.archive.search(question, top_k=self.retrieval_topk)
                raw_context.extend(h["text"] for h in hits if h["text"] not in raw_context)
                break

        # Step 10: Max hops reached — forced finish
        logger.warning(f"Max hops ({self.max_hop}) reached. Forcing answer.")
        evidence_text = self._pool(evidence_nodes, evidence_edges, raw_context)
        answer = self._forced_answer(question, evidence_text, answer_format)
        traces.append({"hop": self.max_hop, "action": "forced_finish"})
        return {"answer": answer, "traces": traces}

    # ------------------------------------------------------------------
    # Step 7: SelectAnchor
    # ------------------------------------------------------------------

    def _select_anchor(self, question: str, local_sub: Dict[str, Any]) -> List[str]:
        """Use LLM to pick 1-3 anchor node_ids from local_sub most relevant to the question.

        Falls back to all nodes if LLM fails or subgraph is tiny.
        """
        nodes = local_sub.get("nodes", {})
        if len(nodes) <= 5:
            return list(nodes.keys())

        from graphmemory.graph_store import format_subgraph
        subgraph_text = format_subgraph(local_sub)

        prompt = (
            "Given the question and the candidate subgraph below, pick the 1-3 nodes "
            "that are most likely to contain or lead to the answer. "
            "Return ONLY a JSON array of 8-char node IDs, e.g. [\"ab12cd34\", \"ef56gh78\"]. "
            "Do not include any other text.\n\n"
            f"Question: {question}\n\n"
            f"Candidate subgraph:\n{subgraph_text}"
        )
        try:
            response = self.llm.complete([{"role": "user", "content": prompt}])
            m = re.search(r"\[.*?\]", response, re.DOTALL)
            if m:
                prefixes = json.loads(m.group())
                if isinstance(prefixes, list) and prefixes:
                    # Resolve 8-char prefixes to full node_ids
                    resolved = []
                    for prefix in prefixes:
                        for full_id in nodes:
                            if full_id.startswith(str(prefix)):
                                resolved.append(full_id)
                                break
                    if resolved:
                        logger.debug(f"SelectAnchor: {len(resolved)} anchors chosen.")
                        return resolved
        except Exception as exc:
            logger.warning(f"SelectAnchor failed: {exc}")

        # Fallback: return all node IDs
        return list(nodes.keys())

    # ------------------------------------------------------------------
    # Step 8: Jump
    # ------------------------------------------------------------------

    def _execute_jump(
        self,
        node_ids: List[str],
        family: str,
        constraint: str,
        budget: int,
        visited: Set[str],
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """Expand frontier from node_ids along edges matching family, return new nodes/edges."""
        new_nodes: Dict[str, Dict] = {}
        new_edges: List[Dict]      = []

        for nid in node_ids:
            # Resolve 8-char prefix to full node_id if needed
            full_nid = self._resolve_node_id(nid)
            if not full_nid:
                continue

            edges = self.graph.get_edges(node_id=full_nid)
            if family != "any":
                edges = [e for e in edges if e["family"] == family]

            count = 0
            for edge in edges:
                if count >= budget:
                    break
                neighbor = edge["dst"] if edge["src"] == full_nid else edge["src"]
                if neighbor in visited:
                    continue
                node = self.graph.get_node(neighbor)
                if node is None:
                    continue
                if constraint and not _matches_constraint(node, constraint):
                    continue
                new_nodes[neighbor] = node
                new_edges.append(edge)
                count += 1

        return new_nodes, new_edges

    def _resolve_node_id(self, ref: str) -> Optional[str]:
        """Accept full UUID or 8-char prefix."""
        if self.graph.get_node(ref):
            return ref
        for nid in self.graph.get_all_nodes():
            if nid.startswith(ref):
                return nid
        return None

    # ------------------------------------------------------------------
    # Step 9: Pool
    # ------------------------------------------------------------------

    def _pool(
        self,
        nodes: Dict[str, Dict],
        edges: List[Dict],
        raw_context: List[str],
    ) -> str:
        """Compress visited subgraph + raw turns into evidence text for the LLM."""
        parts: List[str] = []

        if nodes:
            parts.append(format_subgraph({"nodes": nodes, "edges": edges}))

        if raw_context:
            parts.append("--- Raw conversation context ---")
            parts.extend(raw_context[-10:])  # keep last 10 to bound context size

        return "\n\n".join(parts) if parts else "(no evidence retrieved yet)"

    # ------------------------------------------------------------------
    # Step 11: Raw Fallback (forced)
    # ------------------------------------------------------------------

    def _forced_answer(self, question: str, evidence: str, answer_format: str) -> str:
        prompt = (
            f"You are a Memory Assistant. Answer the question based ONLY on the evidence below.\n"
            f"{answer_format}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}"
        )
        return self.llm.complete([{"role": "user", "content": prompt}]).strip()

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, question: str, evidence: str, answer_format: str) -> str:
        system = _SYSTEM_PROMPT.format(answer_format=answer_format, evidence=evidence)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": _USER_PROMPT.format(question=question)},
        ]
        return self.llm.complete(messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_action(response: str) -> Tuple[str, Dict]:
    """Extract {"action": ..., ...} from LLM response."""
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        logger.warning(f"No JSON object in retrieval response: {response[:80]!r}")
        return "finish", {"answer": response.strip()}
    try:
        data = json.loads(m.group())
        action = data.pop("action", "finish")
        return action, data
    except json.JSONDecodeError:
        logger.warning("Failed to parse retrieval action JSON.")
        return "finish", {"answer": ""}


def _matches_constraint(node: Dict, constraint: str) -> bool:
    """
    Simple text-based constraint matching.
    Constraint is a free-text string; we check if it appears in node text.
    """
    if not constraint:
        return True
    constraint_lower = constraint.lower()
    # Check node type
    if node.get("type", "").lower() in constraint_lower:
        return True
    # Check canonical name / aliases
    if node.get("canonical_name", "").lower() in constraint_lower:
        return True
    # Check attrs
    for v in node.get("attrs", {}).values():
        if str(v).lower() in constraint_lower:
            return True
    return True  # permissive by default — let the LLM filter further

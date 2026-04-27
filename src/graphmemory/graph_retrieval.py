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


_ANSWERABLE_CATEGORIES = {"1", "2", "3", "4"}


# ---------------------------------------------------------------------------
# Answer format  (mirrors Mem-T get_final_result_format exactly)
# ---------------------------------------------------------------------------

def _locomo_format(category: str) -> str:
    cat = str(category)
    if cat == "3":
        return (
            "The Final Result's Format Must Follow These Rules:\n"
            "1. Output ONLY the answer itself — no subject, no verb, no explanation. "
            "No explanations, evidence quotes, source mentions, or full supporting sentences. "
            "Prefer a short phrase answer. "
            "e.g. Q: 'What would Melanie prefer?' → 'national park', NOT 'Melanie would prefer a national park'.\n"
            "2. The question may require you to analyze and infer the answer from the retrieved information.\n"
            "3. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic "
            "numerals in the answer respectively. Numbers are represented by English words by default, "
            "e.g. prefer two not 2.\n"
            "4. This is an open-domain problem. NEVER answer 'I don't know./None./Unknown'. You can perform "
            "reasoning based on the retrieved information and your model knowledge. "
            "Uncertain inferences can be expressed using 'likely'.\n"
            "5. When the answer has multiple phrases, connect them with commas, don't use 'and'.\n"
            "6. Ensure your response aligns directly with the question. "
            "For instance, start with 'Yes' or 'No' for binary questions.\n"
            "7. If the information is not enough, you MUST NOT answer 'Unknown', 'I don't know', "
            "or 'Not mentioned in the conversation'. "
            "Instead, try raw_fallback with different query words. "
            "When reaching max steps, you MUST call finish to give the final answer — "
            "never say 'Unknown' or 'Not mentioned'."
        )
    if cat == "5":
        return (
            "The Final Result's Format Must Follow These Rules:\n"
            "1. Provide a short phrase answer, not a full sentence.\n"
            "2. This is an adversarial unanswerable question. If the described event/fact is not directly "
            "supported by the retrieved evidence, output exactly 'Not mentioned in the conversation'. "
            "Do not fabricate.\n"
            "3. Use exact wording from the original conversation whenever possible."
        )
    # Cat 1, 2, 4 — and default for unknown categories
    return (
        "The Final Result's Format Must Follow These Rules:\n"
        "1. Return ONLY the final answer string. No explanations, evidence quotes, source mentions, "
        "or full supporting sentences. Prefer a short phrase answer.\n"
        "2. For questions requiring a date or time, strictly follow the format '15 July, 2023', 'July, 2023'.\n"
        "3. Pay special attention to relative times like 'yesterday', 'last week', 'last Friday' in the text:\n"
        "   + Only for last year/last month/yesterday, calculate the absolute date precise to year/month/day, "
        "e.g. 'July, 2023' or '19 July, 2023'.\n"
        "   + For last week/weekend/Friday/Saturday, or few days ago etc, use "
        "'the week/weekend/Friday before [certain absolute time]' — MUST NOT calculate the exact date, "
        "e.g. 'the week before 15 July, 2023'.\n"
        "4. Output ONLY the answer itself — no subject, no verb, no explanation, no surrounding context. "
        "e.g. Q: 'Where did Dave go?' → 'San Francisco', NOT 'Dave went to San Francisco'.\n"
        "5. Use exact wording from the original conversation for the answer entity/phrase itself, "
        "but do NOT copy the surrounding sentence or conversation text.\n"
        "6. For quantities, if English/Arabic numerals are used in the original text, use English/Arabic "
        "numerals in the answer respectively. If it is a quantity or frequency counted by yourself, "
        "default to using English words, e.g. prefer two not 2.\n"
        "7. When the answer has multiple phrases, use comma-separated items only; don't use 'and'.\n"
        "8. Ensure your response aligns directly with the question. For instance, start with 'Yes' or 'No' "
        "for binary questions, and do not name a province when asked for a country.\n"
        "9. For Cat1-4 answerable questions, NEVER output 'Not mentioned in the conversation', "
        "'Unknown', 'I don't know', or 'Information not found'. "
        "Instead, try raw_fallback with different query words. "
        "When reaching max steps, you MUST call finish with your best answer — "
        "never say 'Unknown' or 'Not mentioned'."
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
   {{"action": "finish", "answer": "<direct answer phrase — no subject, no verb, no explanation>"}}

[Rules]
- Start from the anchor nodes provided, then Jump to explore.
- Jump along entity-event edges to find what happened to people/places.
- Jump along event-event edges (before/after/updates) for temporal chains.
- Jump along entity-entity edges carefully — easiest to over-expand.
- After each Jump, decide: is the evidence enough to finish? If yes, call finish.
- If the graph lacks key details (exact quotes, fine-grained facts), use raw_fallback.
- You may use raw_fallback AND jump in the same session.
- finish.answer must be the answer phrase ONLY. Do NOT include the question subject,
  verb, reasoning, explanation, or raw conversation text. Extract just the fact.
- Follow the Answer format's absence policy exactly. For Cat1-4 answerable questions,
  do NOT finish with "Not mentioned in the conversation", "Unknown", or
  "Information not found"; use raw_fallback or give the best short answer from evidence.
  For Cat5 adversarial questions, finish with "Not mentioned in the conversation" only
  when the question premise is not directly supported after exploration.
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
        final_answer_compression: bool = False,
    ):
        self.graph          = graph
        self.archive        = archive
        self.localizer      = localizer
        self.llm            = llm
        self.retrieval_topk = retrieval_topk
        self.max_hop        = max_hop
        self.jump_budget    = jump_budget
        self.benchmark      = benchmark
        self.final_answer_compression = final_answer_compression

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def answer(self, question: str, category: str = "") -> Dict[str, Any]:
        """Run the full retrieval pipeline and return {"answer": ..., "traces": [...]}."""
        answer_format = get_answer_format(self.benchmark, category)

        # Step 6: Localize
        if _uses_recall_heavy_localization(category):
            local_sub = self.localizer.localize_union(
                question,
                query_variants=_query_variants(question),
                top_m=3,
                max_nodes=max(self.localizer.max_nodes, 40),
                max_edges=max(self.localizer.max_edges, 80),
            )
        else:
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
        stop_reason = ""

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
                answer = self._finalize_answer(
                    question=question,
                    answer=args.get("answer", ""),
                    evidence_nodes=evidence_nodes,
                    evidence_edges=evidence_edges,
                    raw_context=raw_context,
                    answer_format=answer_format,
                    category=category,
                    traces=traces,
                )
                return {"answer": answer, "traces": traces}

            elif action == "jump":
                new_nodes, new_edges = self._execute_jump(
                    node_ids=args.get("node_ids", list(frontier)),
                    family=args.get("relation_family", "any"),
                    constraint=args.get("constraint", ""),
                    budget=min(int(args.get("budget", self.jump_budget)), self.jump_budget),
                    visited=visited,
                    question=question,
                )
                evidence_nodes.update(new_nodes)
                evidence_edges.extend(new_edges)
                frontier = set(new_nodes.keys())
                visited.update(frontier)

            elif action == "raw_fallback":
                added = self._add_raw_context(args.get("query", question), raw_context)
                trace["args"]["hits_added"] = added

            else:
                logger.warning(f"Unknown action: {action!r} — treated as no-op.")

            # If frontier is empty, no more graph expansion is possible
            if not frontier and action == "jump":
                stop_reason = "frontier_exhausted"
                traces.append({
                    "hop": hop,
                    "op_id": str(uuid.uuid4()),
                    "action": "frontier_exhausted",
                    "args": {"query": question},
                })
                added = self._add_raw_context(question, raw_context)
                traces.append({
                    "hop": hop,
                    "op_id": str(uuid.uuid4()),
                    "action": "raw_fallback",
                    "args": {"query": question, "forced": True, "hits_added": added},
                })
                logger.debug("Frontier exhausted; added raw fallback context.")

        # Step 10: Max hops reached — forced finish
        if not stop_reason:
            stop_reason = "no_evidence" if not evidence_nodes and not raw_context else "max_hop_exhausted"
        if _is_answerable_category(category) and not raw_context:
            added = self._add_raw_context(question, raw_context)
            traces.append({
                "hop": self.max_hop,
                "op_id": str(uuid.uuid4()),
                "action": "raw_fallback",
                "args": {"query": question, "forced": True, "reason": "pre_forced_answer", "hits_added": added},
            })
        logger.warning(f"Retrieval forced finish ({stop_reason}).")
        evidence_text = self._pool(evidence_nodes, evidence_edges, raw_context)
        answer = self._finalize_answer(
            question=question,
            answer=self._forced_answer(question, evidence_text, answer_format, category=category),
            evidence_nodes=evidence_nodes,
            evidence_edges=evidence_edges,
            raw_context=raw_context,
            answer_format=answer_format,
            category=category,
            traces=traces,
        )
        traces.append({"hop": self.max_hop, "action": "forced_finish", "reason": stop_reason})
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
        question: str = "",
    ) -> Tuple[Dict[str, Dict], List[Dict]]:
        """Expand from each node_id independently, merge results.

        Each anchor gets ceil(budget / n_anchors) slots so no single anchor
        can crowd out the others. Nodes claimed by an earlier anchor are
        excluded from later ones to avoid duplicates.
        """
        resolved = [self._resolve_node_id(nid) for nid in node_ids]
        resolved = [nid for nid in resolved if nid]
        if not resolved:
            return {}, []

        budget_each = max(1, -(-budget // len(resolved)))  # ceil division

        allowed_families = _parse_relation_families(family)
        new_nodes: Dict[str, Dict] = {}
        new_edges: List[Dict]      = []
        claimed: Set[str]          = set(visited)  # grows as anchors are processed

        for full_nid in resolved:
            edges = self.graph.get_edges(node_id=full_nid)
            if allowed_families is not None:
                edges = [e for e in edges if e.get("family") in allowed_families]

            candidates: Dict[str, Tuple[Dict, Dict]] = {}
            for edge in edges:
                neighbor = edge["dst"] if edge["src"] == full_nid else edge["src"]
                if neighbor in claimed or neighbor in candidates:
                    continue
                node = self.graph.get_node(neighbor)
                if node is None:
                    continue
                candidates[neighbor] = (node, edge)

            if not candidates:
                continue

            ranked = self._rank_jump_candidates(
                question=question,
                constraint=constraint,
                candidates=candidates,
            )
            added = 0
            for neighbor in ranked:
                if added >= budget_each:
                    break
                node, edge = candidates[neighbor]
                new_nodes[neighbor] = node
                new_edges.append(edge)
                claimed.add(neighbor)
                added += 1

        return new_nodes, new_edges

    def _rank_jump_candidates(
        self,
        question: str,
        constraint: str,
        candidates: Dict[str, Tuple[Dict, Dict]],
    ) -> List[str]:
        """Rank jump candidates with structural constraints plus vector similarity."""
        candidate_ids = list(candidates)
        vector_ranked = self.graph.rank_nodes_by_query(question, candidate_ids)
        vector_pos = {nid: idx for idx, nid in enumerate(vector_ranked)}
        total = max(len(candidate_ids), 1)

        scored: List[Tuple[float, int, int, str]] = []
        for order, node_id in enumerate(candidate_ids):
            node, edge = candidates[node_id]
            structural_score = _score_jump_candidate(question, constraint, node, edge)
            rank = vector_pos.get(node_id, total)
            # Keep vector similarity as a tie-breaker/soft signal, but do not let it
            # override hard structural evidence such as constraint and predicate match.
            vector_bonus = (total - rank) / total
            scored.append((structural_score + vector_bonus, -rank, -order, node_id))

        scored.sort(reverse=True)
        return [node_id for _, _, _, node_id in scored]

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

    def _add_raw_context(self, query: str, raw_context: List[str]) -> int:
        hits = self.archive.search(query, top_k=self.retrieval_topk)
        added = 0
        for hit in hits:
            text = hit.get("text", "")
            if text and text not in raw_context:
                raw_context.append(text)
                added += 1
        return added

    def _finalize_answer(
        self,
        question: str,
        answer: str,
        evidence_nodes: Dict[str, Dict],
        evidence_edges: List[Dict],
        raw_context: List[str],
        answer_format: str,
        category: str,
        traces: List[Dict],
    ) -> str:
        """Apply deterministic cleanup and optional repair/compression before returning."""
        final = _canonicalize_final_answer(answer)

        if _is_answerable_category(category) and _is_refusal_answer(final):
            hits = self.archive.search(question, top_k=self.retrieval_topk)
            added = [h["text"] for h in hits if h["text"] not in raw_context]
            raw_context.extend(added)
            traces.append({
                "hop": len(traces),
                "op_id": str(uuid.uuid4()),
                "action": "answerable_refusal_raw_fallback",
                "args": {"query": question, "hits": len(added)},
            })
            evidence_text = self._pool(evidence_nodes, evidence_edges, raw_context)
            final = _canonicalize_final_answer(
                self._forced_answer(question, evidence_text, answer_format, category=category)
            )

        if self.final_answer_compression and final and not _is_refusal_answer(final):
            final = _canonicalize_final_answer(
                self._compress_final_answer(question, final, answer_format, category)
            )

        return final

    def _compress_final_answer(
        self,
        question: str,
        answer: str,
        answer_format: str,
        category: str,
    ) -> str:
        """Use the LLM only to shorten an already-produced answer, not to re-decide retrieval."""
        prompt = (
            "Rewrite the draft answer into the shortest answer phrase that preserves the facts. "
            "Do not add new facts, do not explain, and do not answer Unknown.\n"
            f"{answer_format}\n\n"
            f"Question: {question}\n"
            f"Category: {category}\n"
            f"Draft answer: {answer}\n\n"
            "Compressed answer:"
        )
        return self.llm.complete([{"role": "user", "content": prompt}]).strip()

    # ------------------------------------------------------------------
    # Step 11: Raw Fallback (forced)
    # ------------------------------------------------------------------

    def _forced_answer(
        self,
        question: str,
        evidence: str,
        answer_format: str,
        category: str = "",
    ) -> str:
        cat = str(category).strip()
        cat_rule = ""
        if cat in _ANSWERABLE_CATEGORIES:
            cat_rule = (
                "MUST NOT answer 'Unknown', 'I don't know', or 'Not mentioned in the conversation'. "
                "Give your best short answer from the evidence.\n"
            )
            if cat == "3":
                cat_rule += (
                    "For inference questions, combine evidence with common-sense reasoning.\n"
                )
        prompt = (
            f"You are a Memory Assistant. Answer the question based ONLY on the evidence below.\n"
            f"{cat_rule}"
            f"{answer_format}\n\n"
            f"Evidence:\n{evidence}\n\n"
            f"Question: {question}\n\n"
            "Answer (the answer phrase only — no subject, no verb, no explanation):"
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


def _parse_relation_families(family: str) -> Optional[Set[str]]:
    """Parse LLM relation_family values into a family allow-list.

    The planner often emits values such as ``entity-event|event-event`` even
    though persisted edges store one family string. ``None`` means "any".
    """
    raw = str(family or "any").strip().lower()
    if not raw or raw == "any":
        return None

    parts = [
        part.strip()
        for part in re.split(r"[|,/]+|\bor\b", raw)
        if part.strip()
    ]
    aliases = {
        "event-entity": "entity-event",
        "entity event": "entity-event",
        "entity to event": "entity-event",
        "entity-entity": "entity-entity",
        "entity entity": "entity-entity",
        "event-event": "event-event",
        "event event": "event-event",
    }
    families = {
        aliases.get(part, part)
        for part in parts
        if aliases.get(part, part) in {"entity-event", "entity-entity", "event-event"}
    }
    return families or None


def _is_answerable_category(category: str) -> bool:
    return str(category).strip() in _ANSWERABLE_CATEGORIES


def _uses_recall_heavy_localization(category: str) -> bool:
    return str(category).strip() in {"1", "3"}


def _query_variants(question: str) -> List[str]:
    text = str(question or "").strip()
    if not text:
        return []

    variants: List[str] = [text]
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*|\d+", text)
    lower_stop = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "did", "does", "do", "has", "have", "had", "is", "are", "was", "were",
        "the", "a", "an", "of", "to", "in", "on", "for", "with", "and",
    }
    content = [w for w in words if w.lower() not in lower_stop]

    entities = [
        w for w in words
        if w[:1].isupper() and w.lower() not in lower_stop
    ]
    if entities and content:
        variants.append(" ".join([entities[0]] + [w for w in content if w != entities[0]][:4]))
    if content:
        variants.append(" ".join(content[:5]))
    if len(content) > 2:
        variants.append(" ".join(content[-4:]))

    deduped: List[str] = []
    for variant in variants:
        variant = variant.strip(" ?.")
        if variant and variant not in deduped:
            deduped.append(variant)
    return deduped


def _is_refusal_answer(answer: str) -> bool:
    text = str(answer or "").strip().lower()
    return text in {
        "not mentioned in the conversation",
        "unknown",
        "i don't know",
        "information not found",
        "none",
    }


def _canonicalize_final_answer(answer: str) -> str:
    """Light deterministic cleanup for answers that should be short strings."""
    text = str(answer or "").strip()
    if not text:
        return ""

    fence = re.search(r"```(?:json|text)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("answer") is not None:
                text = str(data["answer"]).strip()
        except json.JSONDecodeError:
            pass

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        lines.append(line)
    if len(lines) > 1:
        text = ", ".join(lines)
    elif lines:
        text = lines[0]

    text = text.strip().strip("\"'").strip()
    text = re.sub(
        r"^(?:final answer|answer|the answer is|the final answer is)\s*[:\-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:according to (?:the )?(?:evidence|conversation|context),?\s*)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip().strip("\"'").rstrip(" .;").strip()


def _score_jump_candidate(question: str, constraint: str, node: Dict, edge: Dict) -> float:
    node_text = _node_text(node)
    question_terms = _content_terms(question)
    score = 0.0

    for term in question_terms:
        if term in node_text:
            score += 2.0

    predicate = str(edge.get("predicate", "")).lower()
    if predicate and predicate in str(question or "").lower():
        score += 1.0

    score += _constraint_score(node, constraint) * 3.0

    canonical = str(node.get("canonical_name", "")).lower()
    if any(marker in canonical for marker in (" chat", "conversation", "session")):
        score -= 2.0
    if predicate in {"spoke_to", "speak to", "discussed", "related_to"}:
        score -= 1.0

    return score


def _content_terms(text: str) -> List[str]:
    stop = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "did", "does", "do", "has", "have", "had", "is", "are", "was", "were",
        "the", "a", "an", "of", "to", "in", "on", "for", "with", "and",
    }
    return [
        t.lower()
        for t in re.findall(r"[A-Za-z][A-Za-z'’-]*|\d+", str(text or ""))
        if len(t) > 2 and t.lower() not in stop
    ]


def _node_text(node: Dict) -> str:
    parts = [
        str(node.get("type", "")),
        str(node.get("canonical_name", "")),
        " ".join(str(a) for a in node.get("aliases", [])),
    ]
    parts.extend(str(v) for v in node.get("attrs", {}).values())
    return " ".join(parts).lower()


def _constraint_score(node: Dict, constraint: str) -> float:
    constraint = str(constraint or "").strip()
    if not constraint:
        return 0.0

    node_text = _node_text(node)
    lower = constraint.lower()
    if "=" in lower:
        key, value = [part.strip() for part in lower.split("=", 1)]
        if not value:
            return 0.0
        if key in {"type", "node_type"}:
            return 1.0 if str(node.get("type", "")).lower() == value.lower() else 0.0
        attrs = {str(k).lower(): str(v).lower() for k, v in node.get("attrs", {}).items()}
        if key in attrs and value in attrs[key]:
            return 1.0
        return 0.5 if value in node_text else 0.0

    return 1.0 if lower in node_text else 0.0


def _matches_constraint(node: Dict, constraint: str) -> bool:
    """
    Simple text-based constraint matching.
    Constraint is a free-text string; we check if it appears in node text.
    """
    return _constraint_score(node, constraint) > 0

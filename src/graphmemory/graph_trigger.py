"""GraphTrigger: LLM-based gate deciding whether to run graph construction (Step 2).

Trigger is NOT a global filter — it only decides whether to run the graph write path.
Even when trigger=False, the raw archive has already been written by Step 1.

The LLM is asked one yes/no question:
  "Does this input likely cause long-term graph changes?"

Positive signals (more likely to trigger):
  - New entities / events not yet in the graph
  - State changes to existing objects
  - Relationships that may warrant merge / add-edge
  - Long-term query value
"""

from __future__ import annotations

import re
from typing import List, Dict

from loguru import logger

from graphmemory.llm_client import LLMClient


_SYSTEM_PROMPT = """\
You are a graph-memory gatekeeper. Decide whether the following conversation excerpt \
contains any information worth storing in a long-term knowledge graph.

Answer ONLY with one of:
  TRIGGER
  SKIP

Answer TRIGGER if the excerpt contains ANY of the following — even just one sentence:
- A person, place, object, or organisation (new or existing)
- An event, activity, experience, or plan (past, present, or future)
- A preference, hobby, skill, opinion, or personal fact
- A date, time, or temporal reference tied to an event
- A relationship or connection between people or things
- An update, correction, or new detail about something already known

Answer SKIP ONLY when the excerpt is entirely:
- Pure pleasantries / sign-off with zero factual content (e.g. "Great talking to you! Bye!")
- Repetition of the exact same fact already in the graph summary with no new detail

When in doubt, answer TRIGGER. It is better to process unnecessary content than to miss real facts.

Respond with a single word: TRIGGER or SKIP.\
"""

_USER_PROMPT = """\
[Graph summary]
{graph_summary}

[Input excerpt]
{turn_text}\
"""


class GraphTrigger:
    """Calls the LLM to decide if a turn batch warrants graph construction."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def should_trigger(self, turn_text: str, graph_summary: str = "") -> bool:
        """Return True if the input should enter the graph write path."""
        messages = self._build_messages(turn_text, graph_summary)
        response = self.llm.complete(messages).strip().upper()
        logger.debug(f"GraphTrigger response: {response!r}")
        triggered = self._parse(response)
        logger.info(f"GraphTrigger → {'TRIGGER' if triggered else 'SKIP'}")
        return triggered

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_messages(self, turn_text: str, graph_summary: str) -> List[Dict]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_PROMPT.format(
                graph_summary=graph_summary or "(empty graph)",
                turn_text=turn_text,
            )},
        ]

    @staticmethod
    def _parse(response: str) -> bool:
        """Return True if response contains TRIGGER, False if SKIP or ambiguous."""
        if "TRIGGER" in response:
            return True
        if "SKIP" in response:
            return False
        # Fallback: treat ambiguous response as TRIGGER to avoid false negatives
        logger.warning(f"Ambiguous GraphTrigger response: {response!r}. Defaulting to TRIGGER.")
        return True

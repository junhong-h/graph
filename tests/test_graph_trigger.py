"""Unit tests for GraphTrigger."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphmemory.graph_trigger import GraphTrigger


def _trigger(llm_response: str) -> bool:
    llm = MagicMock()
    llm.complete.return_value = llm_response
    return GraphTrigger(llm).should_trigger("Some turn text", "Empty graph")


def test_trigger_on_TRIGGER():
    assert _trigger("TRIGGER") is True


def test_skip_on_SKIP():
    assert _trigger("SKIP") is False


def test_trigger_case_insensitive():
    # LLM might lowercase; our code uppercases before checking
    assert _trigger("trigger") is True
    assert _trigger("skip") is False


def test_trigger_with_extra_text():
    assert _trigger("After analysis: TRIGGER") is True
    assert _trigger("Decision: SKIP — pure small talk") is False


def test_ambiguous_defaults_to_trigger():
    # Unknown response → default TRIGGER (conservative: avoid false negatives)
    assert _trigger("I am not sure.") is True


def test_build_messages_includes_turn_text():
    llm = MagicMock()
    llm.complete.return_value = "TRIGGER"
    t = GraphTrigger(llm)
    t.should_trigger("Jon got a new job", "2 nodes")
    messages = llm.complete.call_args[0][0]
    full_text = " ".join(m["content"] for m in messages)
    assert "Jon got a new job" in full_text
    assert "2 nodes" in full_text

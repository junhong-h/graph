from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_retrieval_traces.py"
    spec = importlib.util.spec_from_file_location("summarize_retrieval_traces", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_summarize_records_counts_actions_and_reasons():
    mod = _load_module()
    summary = mod.summarize_records([
        {
            "category": "1",
            "traces": [
                {"action": "jump"},
                {"action": "frontier_exhausted"},
                {"action": "raw_fallback", "args": {"forced": True}},
                {"action": "forced_finish", "reason": "frontier_exhausted"},
            ],
        },
        {
            "category": "4",
            "traces": [
                {"action": "raw_fallback"},
                {"action": "finish"},
            ],
        },
    ])

    assert summary["total"] == 2
    assert summary["actions"]["raw_fallback"] == 2
    assert summary["forced_reasons"]["frontier_exhausted"] == 1
    assert summary["by_category"]["1"]["samples_with_frontier_exhausted"] == 1

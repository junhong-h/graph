"""Tests for QA/category/sample split filtering."""

from __future__ import annotations

import json
from pathlib import Path

from graphmemory.evaluator import Evaluator
from graphmemory.qa_filters import (
    filter_samples,
    iter_filtered_qa,
    record_matches_filters,
    resolve_include_categories,
)


def _session_sample(sample_id: str) -> dict:
    return {"conversation": [{"metadata": {"sample_id": sample_id}}], "qa": []}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_filter_samples_supports_sample_ids_skip_and_limit() -> None:
    samples = [_session_sample(f"conv-{i}") for i in range(5)]

    selected = filter_samples(
        samples,
        sample_ids=["conv-1", "conv-2", "conv-3", "conv-4"],
        skip_first=1,
        limit=2,
    )

    assert [s["conversation"][0]["metadata"]["sample_id"] for s in selected] == [
        "conv-2",
        "conv-3",
    ]


def test_iter_filtered_qa_preserves_original_question_index() -> None:
    qa_items = [
        {"question": "q0", "category": 5},
        {"question": "q1", "category": 1},
        {"question": "q2", "category": 4},
        {"question": "q3", "category": 5},
    ]

    filtered = list(iter_filtered_qa(qa_items, exclude_categories={"5"}, max_items=1))

    assert filtered == [(1, {"question": "q1", "category": 1})]


def test_resolve_include_categories_locomo_cat1_4_shortcut() -> None:
    assert resolve_include_categories(None, locomo_cat1_4=True) == {"1", "2", "3", "4"}
    assert resolve_include_categories(["2", "5"], locomo_cat1_4=True) == {"2"}


def test_record_matches_filters_checks_sample_and_category() -> None:
    record = {"sample_id": "conv-2", "category": "4"}

    assert record_matches_filters(record, sample_ids={"conv-2"}, include_categories={"4"})
    assert not record_matches_filters(record, sample_ids={"conv-1"}, include_categories={"4"})
    assert not record_matches_filters(record, sample_ids={"conv-2"}, exclude_categories={"4"})


def test_evaluator_evaluate_file_filters_records(tmp_path: Path) -> None:
    input_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        input_path,
        [
            {"qa_id": "conv-1_q0", "sample_id": "conv-1", "category": 1, "pred": "Alice", "gold": "Alice"},
            {"qa_id": "conv-1_q1", "sample_id": "conv-1", "category": 5, "pred": "Paris", "gold": "Not mentioned"},
            {"qa_id": "conv-2_q0", "sample_id": "conv-2", "category": 4, "pred": "Bob", "gold": "Bob"},
        ],
    )

    summary = Evaluator(llm=None).evaluate_file(
        input_path,
        output_path,
        workers=1,
        sample_ids={"conv-1"},
        include_categories={"1", "2", "3", "4"},
    )

    rows = _read_jsonl(output_path)
    assert [r["qa_id"] for r in rows] == ["conv-1_q0"]
    assert summary["total"] == 1
    assert set(summary["by_category"]) == {"1"}


def test_evaluator_compute_stats_filters_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "eval.jsonl"
    _write_jsonl(
        output_path,
        [
            {"qa_id": "a", "sample_id": "conv-1", "category": 1, "f1_score": 1.0, "bleu1_score": 1.0, "judge_label": "SKIPPED"},
            {"qa_id": "b", "sample_id": "conv-1", "category": 5, "f1_score": 0.0, "bleu1_score": 0.0, "judge_label": "SKIPPED"},
        ],
    )

    summary = Evaluator.compute_stats(
        output_path,
        include_categories={"1", "2", "3", "4"},
    )

    assert summary["total"] == 1
    assert summary["avg_f1"] == 1.0

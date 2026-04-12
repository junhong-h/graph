"""Tests for dataset.py loaders."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from graphmemory.dataset import load_locomo
from graphmemory.models import ProcessedSample


LOCOMO_FIXTURE = [
    {
        "sample_id": "s001",
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_0": [
                {"dia_id": "0", "speaker": "Alice", "text": "Hi Bob!"},
                {"dia_id": "1", "speaker": "Bob", "text": "Hey Alice!"},
            ],
            "session_0_date_time": "1 Jan, 2024",
            "session_1": [
                {"dia_id": "2", "speaker": "Alice", "text": "How are you?"},
            ],
            "session_1_date_time": "2 Jan, 2024",
        },
        "qa": [
            {"question": "What did Alice say first?", "answer": "Hi Bob!", "evidence": ["0"], "category": 1},
            {"question": "Skip me", "answer": "N/A", "evidence": [], "category": 5},
        ],
    }
]


@pytest.fixture()
def locomo_file(tmp_path: Path) -> Path:
    p = tmp_path / "locomo_test.json"
    p.write_text(json.dumps(LOCOMO_FIXTURE), encoding="utf-8")
    return p


def test_load_locomo_turn_count(locomo_file: Path) -> None:
    samples = list(load_locomo(locomo_file))
    # 2 turns in session_0 + 1 turn in session_1 = 3
    assert len(samples) == 3


def test_load_locomo_types(locomo_file: Path) -> None:
    samples = list(load_locomo(locomo_file))
    for s in samples:
        assert isinstance(s, ProcessedSample)


def test_load_locomo_fields(locomo_file: Path) -> None:
    samples = list(load_locomo(locomo_file))
    first = samples[0]
    assert first.text == "Hi Bob!"
    assert first.speaker == "Alice"
    assert first.timestamp == "1 Jan, 2024"
    assert "s001_conv_session_0" in first.source_doc_id


def test_load_locomo_metadata_has_turn_id(locomo_file: Path) -> None:
    samples = list(load_locomo(locomo_file))
    meta = json.loads(samples[0].metadata_json)
    assert meta["turn_id"] == "s001_conv_session_0_0"


def test_load_locomo_qa_on_last_turn_only(locomo_file: Path) -> None:
    samples = list(load_locomo(locomo_file))
    # QA should only be on the last turn of the last session
    last = samples[-1]
    meta = json.loads(last.metadata_json)
    assert len(meta["qa"]) == 1  # category-5 question filtered out
    assert meta["qa"][0]["answer"] == "Hi Bob!"

    # All other turns should have empty qa list
    for s in samples[:-1]:
        meta = json.loads(s.metadata_json)
        assert meta["qa"] == []


def test_load_locomo_evidence_resolved(locomo_file: Path) -> None:
    samples = list(load_locomo(locomo_file))
    last = samples[-1]
    meta = json.loads(last.metadata_json)
    evidence = meta["qa"][0]["evidence_turn_ids"]
    assert evidence == ["s001_conv_session_0_0"]


def test_load_locomo_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        list(load_locomo("/nonexistent/path.json"))

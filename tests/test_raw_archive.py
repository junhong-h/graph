"""Unit tests for RawArchive."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphmemory.raw_archive import RawArchive


def _make_archive() -> RawArchive:
    store = MagicMock()
    store.count.return_value = 0
    store.search.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
    store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return RawArchive(store, sample_id="test")


def test_archive_calls_upsert():
    archive = _make_archive()
    archive.archive("b1", "Alice: hello", "s1", ["t1"], "2023-01-01")
    archive._store.upsert.assert_called_once()
    call_kwargs = archive._store.upsert.call_args
    assert "b1" in call_kwargs.kwargs.get("ids", call_kwargs.args[1] if len(call_kwargs.args) > 1 else [])


def test_search_returns_empty_when_no_data():
    archive = _make_archive()
    results = archive.search("some query")
    assert results == []


def test_search_returns_results():
    archive = _make_archive()
    archive._store.count.return_value = 2
    archive._store.search.return_value = {
        "ids": [["b1", "b2"]],
        "documents": [["Alice: hello", "Bob: hi"]],
        "metadatas": [[{"session_id": "s1"}, {"session_id": "s1"}]],
    }
    results = archive.search("hello")
    assert len(results) == 2
    assert results[0]["text"] == "Alice: hello"
    assert results[0]["meta"]["session_id"] == "s1"


def test_get_all_returns_empty():
    archive = _make_archive()
    assert archive.get_all() == []


def test_count_delegates_to_store():
    archive = _make_archive()
    archive._store.count.return_value = 7
    assert archive.count() == 7

"""Raw archive: unconditionally store every input turn batch (Step 1).

Backed by ChromaDB via VectorStore. Serves as:
1. Fallback source when graph retrieval is insufficient.
2. Source for delayed graphization when a raw batch is repeatedly hit.
"""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from graphmemory.vector_store import VectorStore


class RawArchive:
    """Append-only store for raw turn batches."""

    def __init__(self, store: VectorStore, sample_id: str):
        self._store = store
        self._col = f"{sample_id}_turns"
        self._store.create_collection(self._col, get_or_create=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def archive(
        self,
        batch_id: str,
        batch_text: str,
        session_id: str,
        turn_ids: List[str],
        turn_time: str,
    ) -> None:
        """Unconditionally save a turn batch. Idempotent (upsert)."""
        self._store.upsert(
            self._col,
            ids=[batch_id],
            documents=[batch_text],
            metadatas=[{
                "session_id":  session_id,
                "turn_time":   turn_time,
                "turn_ids":    turn_ids,
                "batch_id":    batch_id,
            }],
        )
        logger.debug(f"RawArchive: stored batch {batch_id}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k matching batches as list of {text, meta} dicts."""
        if self._store.count(self._col) == 0:
            return []
        results = self._store.search(self._col, query, top_k=top_k)
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        out = []
        for doc, meta in zip(docs, metas):
            out.append({"text": doc, "meta": meta or {}})
        return out

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all archived batches as list of {text, meta} dicts."""
        if self._store.count(self._col) == 0:
            return []
        results = self._store.get(self._col, ids=[])
        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])
        return [{"text": d, "meta": m or {}} for d, m in zip(docs, metas) if d]

    def count(self) -> int:
        return self._store.count(self._col)

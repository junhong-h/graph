"""ChromaDB vector store wrapper for Graphmemory."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger


class VectorStore(ABC):
    @abstractmethod
    def create_collection(self, name: str, get_or_create: bool = True) -> None: ...

    @abstractmethod
    def delete_collection(self, name: str) -> None: ...

    @abstractmethod
    def add(self, collection: str, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None: ...

    @abstractmethod
    def upsert(self, collection: str, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None: ...

    @abstractmethod
    def search(self, collection: str, query: str, top_k: int = 5) -> Dict[str, Any]: ...

    @abstractmethod
    def get(self, collection: str, ids: List[str]) -> Dict[str, Any]: ...

    @abstractmethod
    def delete(self, collection: str, ids: List[str]) -> None: ...

    @abstractmethod
    def count(self, collection: str) -> int: ...


class ChromaStore(VectorStore):
    """Persistent ChromaDB store with BAAI/bge-m3 embeddings."""

    def __init__(self, path: str, from_scratch: bool = False):
        import chromadb
        from chromadb.utils import embedding_functions

        self._path = path
        self._from_scratch = from_scratch
        self._client = chromadb.PersistentClient(path=path)

        logger.info("Loading embedding model BAAI/bge-m3…")
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-m3"
        )
        logger.info("Embedding model ready.")

    # ------------------------------------------------------------------
    # Metadata serialisation: ChromaDB only accepts scalar values
    # ------------------------------------------------------------------

    @staticmethod
    def _ser(metadatas: Optional[List[Dict]]) -> Optional[List[Dict]]:
        if not metadatas:
            return metadatas
        out = []
        for meta in metadatas:
            if meta is None:
                out.append(None)
                continue
            out.append(
                {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                 for k, v in meta.items()}
            )
        return out

    @staticmethod
    def _deser(metadatas: Optional[List[Dict]]) -> Optional[List[Dict]]:
        if not metadatas:
            return metadatas
        out = []
        for meta in metadatas:
            if meta is None:
                out.append(None)
                continue
            new_meta = {}
            for k, v in meta.items():
                if isinstance(v, str) and (
                    (v.startswith("[") and v.endswith("]"))
                    or (v.startswith("{") and v.endswith("}"))
                ):
                    try:
                        parsed = json.loads(v)
                        new_meta[k] = parsed if isinstance(parsed, (list, dict)) else v
                    except (json.JSONDecodeError, TypeError):
                        new_meta[k] = v
                else:
                    new_meta[k] = v
            out.append(new_meta)
        return out

    def _col(self, name: str):
        return self._client.get_collection(name=name, embedding_function=self._embed_fn)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_collection(self, name: str, get_or_create: bool = True) -> None:
        if self._from_scratch:
            self.delete_collection(name)
        self._client.get_or_create_collection(name=name, embedding_function=self._embed_fn)
        logger.debug(f"Collection ready: {name}")

    def delete_collection(self, name: str) -> None:
        existing = [c.name for c in self._client.list_collections()]
        if name in existing:
            self._client.delete_collection(name)
            logger.debug(f"Deleted collection: {name}")

    def add(self, collection: str, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
        self._col(collection).add(
            ids=ids, documents=documents, metadatas=self._ser(metadatas)
        )

    def upsert(self, collection: str, ids: List[str], documents: List[str], metadatas: List[Dict]) -> None:
        self._col(collection).upsert(
            ids=ids, documents=documents, metadatas=self._ser(metadatas)
        )

    def search(self, collection: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        col = self._col(collection)
        if col.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        results = col.query(query_texts=[query], n_results=top_k,
                            include=["documents", "metadatas"])
        if results.get("metadatas"):
            results["metadatas"] = [self._deser(batch) for batch in results["metadatas"]]
        return results

    def get(self, collection: str, ids: List[str]) -> Dict[str, Any]:
        col = self._col(collection)
        results = col.get(ids=ids, include=["documents", "metadatas"])
        if results.get("metadatas"):
            results["metadatas"] = self._deser(results["metadatas"])
        return results

    def delete(self, collection: str, ids: List[str]) -> None:
        self._col(collection).delete(ids=ids)

    def count(self, collection: str) -> int:
        return self._col(collection).count()

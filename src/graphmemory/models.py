from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProcessedSample:
    sample_id: str
    text: str
    timestamp: str
    source_doc_id: str
    speaker: str
    metadata_json: str = "{}"

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawMemoryRecord:
    record_id: str
    dataset_name: str
    source_doc_id: str
    turn_id: str
    text: str
    timestamp: str
    metadata_json: str
    processed_sample_id: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawMemoryChunk:
    chunk_id: str
    record_id: str
    chunk_text: str
    chunk_order: int
    token_count: int
    embedding_ref: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RawFallbackHit:
    query_id: str
    record_id: str
    chunk_id: str
    hit_time: str
    reason: str
    promote_candidate_bool: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: str
    canonical_name: str
    status: str
    created_at: str
    updated_at: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphEdge:
    edge_id: str
    src_id: str
    dst_id: str
    family: str
    predicate: str
    status: str
    weight: float
    created_at: str
    updated_at: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphNodeAttr:
    node_id: str
    attr_key: str
    attr_value: str
    value_type: str
    valid_from: str
    valid_to: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProvenanceRecord:
    owner_type: str
    owner_id: str
    record_id: str
    chunk_id: str
    evidence_span: str
    confidence: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GraphSnapshotMeta:
    graph_name: str
    version: str
    created_at: str
    schema_version: str
    dataset_names: list[str] = field(default_factory=list)
    source_snapshot_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    tag: str
    created_at: str
    input_dataset: str
    input_dataset_version: str
    prior_graph_version: str | None = None
    config_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

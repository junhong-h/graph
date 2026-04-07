from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from graphmemory.models import (
    GraphEdge,
    GraphNode,
    GraphNodeAttr,
    ProcessedSample,
    ProvenanceRecord,
    RawFallbackHit,
    RawMemoryChunk,
    RawMemoryRecord,
)
from graphmemory.storage import (
    DuckDBIndexStore,
    GraphQueryService,
    GraphSnapshotStore,
    ProcessedDatasetStore,
    RawMemoryStore,
    RunStore,
)
from graphmemory.storage.parquet_io import read_json, read_rows


def test_processed_dataset_import_creates_versioned_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    dataset_store = ProcessedDatasetStore(repo_root)

    source_file = tmp_path / "download.jsonl"
    source_file.write_text('{"text": "hello"}\n', encoding="utf-8")
    copied_path = dataset_store.register_raw_file("demo_dataset", source_file)
    assert copied_path == repo_root / "data" / "raw" / "demo_dataset" / "download.jsonl"
    assert copied_path.read_text(encoding="utf-8") == '{"text": "hello"}\n'

    manifest = dataset_store.import_processed_samples(
        dataset_name="demo_dataset",
        version="v20260407_120000",
        processing_script="scripts/normalize_demo.py",
        samples=[
            ProcessedSample(
                sample_id="sample-1",
                text="Jon joined the team.",
                timestamp="2026-04-07T12:00:00+00:00",
                source_doc_id="doc-1",
                speaker="user",
                metadata_json='{"split": "train"}',
            )
        ],
    )
    manifest_path = repo_root / "data" / "processed" / "demo_dataset" / "v20260407_120000" / "manifest.json"
    sample_path = manifest_path.with_name("samples.parquet")

    assert manifest["sample_count"] == 1
    assert manifest_path.exists()
    assert sample_path.exists()
    assert read_rows(sample_path)[0]["sample_id"] == "sample-1"

    with pytest.raises(FileExistsError):
        dataset_store.import_processed_samples(
            dataset_name="demo_dataset",
            version="v20260407_120000",
            processing_script="scripts/normalize_demo.py",
            samples=[],
        )


def test_raw_memory_snapshot_extends_and_tracks_fallback_hits(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    raw_store = RawMemoryStore(repo_root)

    raw_store.write_snapshot(
        memory_name="conversation_memory",
        dataset_name="demo_dataset",
        version="v20260407_120100",
        records=[
            RawMemoryRecord(
                record_id="record-1",
                dataset_name="demo_dataset",
                source_doc_id="doc-1",
                turn_id="turn-1",
                text="Jon joined the team yesterday.",
                timestamp="2026-04-07T12:01:00+00:00",
                metadata_json="{}",
                processed_sample_id="sample-1",
            )
        ],
        chunks=[
            RawMemoryChunk(
                chunk_id="chunk-1",
                record_id="record-1",
                chunk_text="Jon joined the team yesterday.",
                chunk_order=0,
                token_count=6,
            )
        ],
    )

    meta = raw_store.write_snapshot(
        memory_name="conversation_memory",
        dataset_name="demo_dataset",
        version="v20260407_120200",
        base_version="v20260407_120100",
        records=[],
        chunks=[],
        fallback_hits=[
            RawFallbackHit(
                query_id="query-1",
                record_id="record-1",
                chunk_id="chunk-1",
                hit_time="2026-04-07T12:02:00+00:00",
                reason="graph_missing_detail",
                promote_candidate_bool=True,
            ),
            RawFallbackHit(
                query_id="query-2",
                record_id="record-1",
                chunk_id="chunk-1",
                hit_time="2026-04-07T12:03:00+00:00",
                reason="exact_quote",
                promote_candidate_bool=True,
            ),
        ],
    )

    snapshot_dir = repo_root / "artifacts" / "raw_memory" / "conversation_memory" / "v20260407_120200"
    assert meta["base_version"] == "v20260407_120100"
    assert read_rows(snapshot_dir / "records.parquet")[0]["record_id"] == "record-1"
    assert read_rows(snapshot_dir / "fallback_hits.parquet")[1]["query_id"] == "query-2"
    assert raw_store.top_fallback_records("conversation_memory", "v20260407_120200") == [
        {"record_id": "record-1", "chunk_id": "chunk-1", "hit_count": 2}
    ]

    resolved = raw_store.resolve_chunk("conversation_memory", "v20260407_120200", "chunk-1")
    assert resolved is not None
    assert resolved["record"]["record_id"] == "record-1"
    assert resolved["chunk"]["chunk_text"] == "Jon joined the team yesterday."


def test_graph_snapshot_compare_and_provenance_resolution(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    raw_store = RawMemoryStore(repo_root)
    graph_store = GraphSnapshotStore(repo_root)

    raw_store.write_snapshot(
        memory_name="conversation_memory",
        dataset_name="demo_dataset",
        version="v20260407_120100",
        records=[
            RawMemoryRecord(
                record_id="record-1",
                dataset_name="demo_dataset",
                source_doc_id="doc-1",
                turn_id="turn-1",
                text="Jon joined the team yesterday.",
                timestamp="2026-04-07T12:01:00+00:00",
                metadata_json="{}",
                processed_sample_id="sample-1",
            )
        ],
        chunks=[
            RawMemoryChunk(
                chunk_id="chunk-1",
                record_id="record-1",
                chunk_text="Jon joined the team yesterday.",
                chunk_order=0,
                token_count=6,
            )
        ],
    )

    graph_store.write_snapshot(
        graph_name="team_memory",
        version="v20260407_120300",
        dataset_names=["demo_dataset"],
        nodes=[
            GraphNode(
                node_id="entity-1",
                node_type="Entity",
                canonical_name="Jon",
                status="active",
                created_at="2026-04-07T12:03:00+00:00",
                updated_at="2026-04-07T12:03:00+00:00",
            )
        ],
        edges=[],
        node_attrs=[],
        provenance=[],
    )

    graph_store.write_snapshot(
        graph_name="team_memory",
        version="v20260407_120400",
        base_version="v20260407_120300",
        dataset_names=["demo_dataset"],
        nodes=[
            GraphNode(
                node_id="event-1",
                node_type="Event",
                canonical_name="Jon joined the team",
                status="active",
                created_at="2026-04-07T12:04:00+00:00",
                updated_at="2026-04-07T12:04:00+00:00",
            )
        ],
        edges=[
            GraphEdge(
                edge_id="edge-1",
                src_id="entity-1",
                dst_id="event-1",
                family="entity_event",
                predicate="participant_in",
                status="active",
                weight=1.0,
                created_at="2026-04-07T12:04:00+00:00",
                updated_at="2026-04-07T12:04:00+00:00",
            )
        ],
        node_attrs=[
            GraphNodeAttr(
                node_id="event-1",
                attr_key="event_type",
                attr_value="employment_change",
                value_type="string",
                valid_from="2026-04-07T12:04:00+00:00",
                valid_to="",
            )
        ],
        provenance=[
            ProvenanceRecord(
                owner_type="node",
                owner_id="event-1",
                record_id="record-1",
                chunk_id="chunk-1",
                evidence_span="0:30",
                confidence=0.92,
            )
        ],
        summary={"merge_count": 1},
    )

    comparison = graph_store.compare_snapshots(
        graph_name="team_memory",
        left_version="v20260407_120300",
        right_version="v20260407_120400",
    )
    assert comparison["table_diffs"]["nodes"]["added_rows"] == 1
    assert comparison["right_summary"]["merge_count"] == 1

    evidence = graph_store.resolve_provenance(
        graph_name="team_memory",
        version="v20260407_120400",
        owner_type="node",
        owner_id="event-1",
        raw_memory_store=raw_store,
        raw_memory_name="conversation_memory",
        raw_memory_version="v20260407_120100",
    )
    assert evidence[0]["evidence"]["chunk"]["chunk_text"] == "Jon joined the team yesterday."


def test_run_store_persists_manifest_metrics_intermediate_and_errors(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_store = RunStore(repo_root)

    manifest = run_store.create_run(
        tag="demo run",
        input_dataset="demo_dataset",
        input_dataset_version="v20260407_120000",
        prior_graph_version="v20260407_120300",
        config_refs=["configs/demo.yaml"],
    )

    run_store.write_metrics(manifest.run_id, {"retrieval_f1": 0.8})
    schema = pa.schema([("sample_id", pa.string()), ("trigger", pa.bool_())])
    run_store.write_intermediate_table(
        manifest.run_id,
        "trigger_outputs",
        [{"sample_id": "sample-1", "trigger": True}],
        schema,
    )
    run_store.write_retrieval_table(
        manifest.run_id,
        "retrieval_trace",
        [{"sample_id": "sample-1", "trigger": True}],
        schema,
    )
    run_store.write_graph_artifact(
        manifest.run_id,
        "candidate_graph",
        {"node_count": 1, "edge_count": 0},
    )
    run_store.append_sample_error(
        manifest.run_id,
        sample_id="sample-2",
        stage="construction",
        error_message="LLM output parse error",
        traceback_text="trace",
        extra={"retryable": True},
    )
    run_store.append_sample_error(
        manifest.run_id,
        sample_id="sample-3",
        stage="update",
        error_message="missing node alignment",
    )

    run_dir = repo_root / "runs" / manifest.run_id
    manifest_path = run_dir / "config_snapshot" / "manifest.json"
    metrics_path = run_dir / "metrics.json"
    errors_jsonl = run_dir / "logs" / "sample_errors.jsonl"
    errors_parquet = run_dir / "logs" / "sample_errors.parquet"

    assert read_json(manifest_path)["input_dataset"] == "demo_dataset"
    assert read_json(metrics_path)["retrieval_f1"] == 0.8
    assert (run_dir / "intermediate" / "trigger_outputs.parquet").exists()
    assert (run_dir / "retrieval" / "retrieval_trace.parquet").exists()
    assert (run_dir / "graph" / "candidate_graph" / "summary.json").exists()
    assert len(errors_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert len(read_rows(errors_parquet)) == 2


def test_graph_query_service_falls_back_to_parquet_when_index_missing(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    graph_store = GraphSnapshotStore(repo_root)
    query_service = GraphQueryService(repo_root)

    graph_store.write_snapshot(
        graph_name="team_memory",
        version="v20260407_120500",
        nodes=[
            GraphNode(
                node_id="entity-1",
                node_type="Entity",
                canonical_name="Jon",
                status="active",
                created_at="2026-04-07T12:05:00+00:00",
                updated_at="2026-04-07T12:05:00+00:00",
            )
        ],
        edges=[],
        node_attrs=[],
        provenance=[],
    )

    rows = query_service.find_nodes(
        graph_name="team_memory",
        graph_version="v20260407_120500",
        node_type="Entity",
        canonical_name_contains="jon",
        index_name="graph_lookup",
        index_version="v20260407_120500",
    )
    assert rows == [
        {
            "node_id": "entity-1",
            "node_type": "Entity",
            "canonical_name": "Jon",
            "status": "active",
            "created_at": "2026-04-07T12:05:00+00:00",
            "updated_at": "2026-04-07T12:05:00+00:00",
        }
    ]


def test_duckdb_index_can_be_built_and_queried(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    graph_store = GraphSnapshotStore(repo_root)
    index_store = DuckDBIndexStore(repo_root)
    query_service = GraphQueryService(repo_root)

    graph_store.write_snapshot(
        graph_name="team_memory",
        version="v20260407_120600",
        nodes=[
            GraphNode(
                node_id="entity-1",
                node_type="Entity",
                canonical_name="Gina",
                status="active",
                created_at="2026-04-07T12:06:00+00:00",
                updated_at="2026-04-07T12:06:00+00:00",
            )
        ],
        edges=[],
        node_attrs=[],
        provenance=[],
    )

    db_path = index_store.build_graph_index(
        index_name="graph_lookup",
        version="v20260407_120600",
        graph_name="team_memory",
        graph_version="v20260407_120600",
    )

    rows = query_service.find_nodes(
        graph_name="team_memory",
        graph_version="v20260407_120600",
        node_type="Entity",
        canonical_name_contains="gin",
        index_name="graph_lookup",
        index_version="v20260407_120600",
    )

    assert db_path.exists()
    assert rows[0]["canonical_name"] == "Gina"

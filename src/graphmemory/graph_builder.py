"""GraphBuilder: orchestrates Steps 1-5 for each input turn batch.

Pipeline per batch
------------------
Step 1  RawArchive   — unconditionally write raw turns
Step 2  GraphTrigger — LLM decides whether to enter graph write path
Step 3  Localize     — find the most relevant local subgraph
Step 4+5 Construction — LLM proposes + executes graph edits (merged)
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from tqdm import tqdm

from graphmemory.config import BuildConfig
from graphmemory.graph_construction import GraphConstructor
from graphmemory.graph_localize import GraphLocalizer
from graphmemory.graph_store import GraphStore
from graphmemory.graph_trigger import GraphTrigger
from graphmemory.raw_archive import RawArchive


class GraphBuilder:
    """Processes one dataset sample through the full construction pipeline."""

    def __init__(
        self,
        graph: GraphStore,
        archive: RawArchive,
        trigger: GraphTrigger,
        localizer: GraphLocalizer,
        constructor: GraphConstructor,
        config: BuildConfig,
    ):
        self.graph       = graph
        self.archive     = archive
        self.trigger     = trigger
        self.localizer   = localizer
        self.constructor = constructor
        self.k_turns     = config.memory.k_turns
        self.traj_path   = Path(config.run_dir) / "graph_trajectories.jsonl"
        self.traj_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build_from_sample(self, sample: Dict[str, Any]) -> None:
        sessions: List[Dict] = sample.get("conversation", [])
        if not sessions:
            logger.warning("Sample has no sessions, skipping.")
            return

        sample_id: str = sessions[0].get("metadata", {}).get("sample_id", "unknown")
        done_batches = self._load_done_batches()

        total_batches = sum(
            (len(s.get("session_turns", [])) + self.k_turns - 1) // self.k_turns
            for s in sessions
        )
        logger.info(f"Building graph for sample {sample_id} ({total_batches} batches)…")

        with tqdm(total=total_batches, desc=f"Sample {sample_id}", unit="batch",
                  initial=len(done_batches)) as pbar:
            for session in sessions:
                self._process_session(session, sample_id, pbar, done_batches)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_session(
        self,
        session: Dict,
        sample_id: str,
        pbar: tqdm,
        done_batches: set,
    ) -> None:
        session_id    = session["session_id"]
        turns         = session["session_turns"]
        meta          = session.get("metadata", {})
        turn_datetime = meta.get("session_time", "")
        speaker_a     = meta.get("speaker_a", "SpeakerA")
        speaker_b     = meta.get("speaker_b", "SpeakerB")

        for i in range(0, len(turns), self.k_turns):
            batch_id = f"{session_id}_batch_{i // self.k_turns}"
            if batch_id in done_batches:
                pbar.update(1)
                continue

            batch = turns[i : i + self.k_turns]
            batch_turn_ids = [t["turn_id"] for t in batch]

            # Format batch text
            lines: List[str] = []
            for turn in batch:
                spk   = turn.get("speaker", "Unknown")
                other = speaker_b if spk == speaker_a else speaker_a
                lines.append(f"{spk} speak to {other} at {turn_datetime}: {turn.get('text', '')}")
            batch_text = "\n".join(lines)

            self._process_batch(
                batch_id=batch_id,
                batch_text=batch_text,
                session_id=session_id,
                batch_turn_ids=batch_turn_ids,
                turn_time=turn_datetime,
            )
            pbar.update(1)

    def _process_batch(
        self,
        batch_id: str,
        batch_text: str,
        session_id: str,
        batch_turn_ids: List[str],
        turn_time: str,
    ) -> None:
        op_id = str(uuid.uuid4())

        # ── Step 1: RawArchive ────────────────────────────────────────
        self.archive.archive(
            batch_id=batch_id,
            batch_text=batch_text,
            session_id=session_id,
            turn_ids=batch_turn_ids,
            turn_time=turn_time,
        )

        # ── Step 2: GraphTrigger ──────────────────────────────────────
        graph_summary = f"{self.graph.node_count()} nodes, {self.graph.edge_count()} edges"
        triggered = self.trigger.should_trigger(batch_text, graph_summary)
        self._log(batch_id, session_id, "trigger", op_id,
                  extra={"triggered": triggered, "graph_summary": graph_summary})

        if not triggered:
            logger.debug(f"Batch {batch_id}: trigger=SKIP, graph unchanged.")
            return

        # ── Step 3: Localize ──────────────────────────────────────────
        local_subgraph = self.localizer.localize(batch_text)
        self._log(batch_id, session_id, "localize", op_id,
                  extra={
                      "subgraph_nodes": len(local_subgraph.get("nodes", {})),
                      "subgraph_edges": len(local_subgraph.get("edges", [])),
                  })

        # ── Steps 4+5: Construction (merged) ─────────────────────────
        op_log = self.constructor.run(batch_text, local_subgraph)
        self._log(batch_id, session_id, "construction", op_id,
                  extra={"op_log": op_log, "batch_turn_ids": batch_turn_ids})
        logger.info(
            f"Batch {batch_id}: {len(op_log)} ops executed. "
            f"Graph: {self.graph.node_count()} nodes, {self.graph.edge_count()} edges."
        )

    def _load_done_batches(self) -> set:
        """Return batch_ids already logged to the trajectory file."""
        done: set = set()
        if self.traj_path.exists():
            with self.traj_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("phase") == "construction":
                        bid = rec.get("batch_id")
                        if bid:
                            done.add(bid)
        return done

    def _log(
        self,
        batch_id: str,
        session_id: str,
        phase: str,
        op_id: str,
        extra: Dict | None = None,
    ) -> None:
        record: Dict = {
            "ts":         time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_id":   batch_id,
            "session_id": session_id,
            "phase":      phase,
            "op_id":      op_id,
        }
        if extra:
            record.update(extra)
        with self.traj_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

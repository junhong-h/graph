"""Dataset loaders for Graphmemory.

Each loader parses a raw benchmark file and yields ProcessedSample objects,
one per conversation turn, ready for ProcessedDatasetStore.import_processed_samples().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

from graphmemory.models import ProcessedSample


# ---------------------------------------------------------------------------
# Locomo
# ---------------------------------------------------------------------------

def load_locomo(path: str | Path) -> Iterator[ProcessedSample]:
    """Parse locomo10.json and yield one ProcessedSample per conversation turn.

    sample_id  : "<locomo_sample_id>_<session_idx>_<dia_id>"
    text       : raw utterance text
    timestamp  : session date string (e.g. "15 July, 2023")
    source_doc_id : session_id ("<locomo_sample_id>_conv_session_<N>")
    speaker    : speaker name
    metadata_json : JSON with locomo_sample_id, session_id, turn_id, qa list
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        locomo_sample_id: str = sample.get("sample_id", "")
        conversation: dict = sample.get("conversation", {})
        qa_list: list = sample.get("qa", [])

        if not conversation or not isinstance(conversation, dict):
            continue

        speaker_a = conversation.get("speaker_a", "SpeakerA")
        speaker_b = conversation.get("speaker_b", "SpeakerB")
        conversation_id = f"{locomo_sample_id}_conv"

        session_keys = sorted(
            [k for k in conversation if re.match(r"session_\d+$", k)],
            key=lambda x: int(x.split("_")[1]),
        )

        # Build dia_id → turn_id map for QA evidence linking
        dia_id_to_turn_id: dict[str, str] = {}
        for session_key in session_keys:
            session_id = f"{conversation_id}_{session_key}"
            for i, turn in enumerate(conversation[session_key]):
                dia_id = turn.get("dia_id", "")
                turn_id = f"{session_id}_{dia_id}" if dia_id else f"{session_id}_turn_{i}"
                if dia_id:
                    dia_id_to_turn_id[dia_id] = turn_id

        # Format QA list (filter category 5, resolve evidence turn ids)
        formatted_qa = []
        for qa_item in qa_list:
            if qa_item.get("category") == 5:
                continue
            evidence_turn_ids = [
                dia_id_to_turn_id[d]
                for d in (qa_item.get("evidence") or [])
                if d in dia_id_to_turn_id
            ]
            formatted_qa.append({
                "question": qa_item.get("question"),
                "answer": qa_item.get("answer", "I don't know."),
                "evidence_turn_ids": evidence_turn_ids,
                "category": qa_item.get("category"),
            })

        # Yield one ProcessedSample per turn
        for session_key in session_keys:
            session_id = f"{conversation_id}_{session_key}"
            session_datetime = conversation.get(f"{session_key}_date_time", "")
            turns: list = conversation[session_key]
            if not isinstance(turns, list):
                continue

            for i, turn in enumerate(turns):
                dia_id = turn.get("dia_id", "")
                turn_id = (
                    f"{session_id}_{dia_id}" if dia_id else f"{session_id}_turn_{i}"
                )
                speaker_raw = turn.get("speaker", "")
                # Resolve coded speaker names to real names
                if speaker_raw == speaker_a:
                    speaker = speaker_a
                elif speaker_raw == speaker_b:
                    speaker = speaker_b
                else:
                    speaker = speaker_raw

                metadata = {
                    "locomo_sample_id": locomo_sample_id,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "turn_id": turn_id,
                    "dia_id": dia_id,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    # QA only on the last turn of the sample to avoid duplication;
                    # stored here for provenance, retrieval uses the graph layer.
                    "qa": formatted_qa if (
                        session_key == session_keys[-1]
                        and i == len(turns) - 1
                    ) else [],
                }

                yield ProcessedSample(
                    sample_id=turn_id,
                    text=turn.get("text", ""),
                    timestamp=session_datetime,
                    source_doc_id=session_id,
                    speaker=speaker,
                    metadata_json=json.dumps(metadata, ensure_ascii=False),
                )


# ---------------------------------------------------------------------------
# Locomo – session-grouped format for MemoryBuilder
# ---------------------------------------------------------------------------

def load_locomo_sessions(path: str | Path) -> List[Dict[str, Any]]:
    """Parse locomo10.json and return a list of samples in session-grouped format.

    Each item: {"conversation": [<session>, ...], "qa": [<qa_item>, ...]}
    Each session: {"sample_id", "session_id", "session_turns": [...], "metadata": {...}}
    Each turn:    {"turn_id", "speaker", "text"}
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict[str, Any]] = []

    for sample in data:
        locomo_sample_id: str = sample.get("sample_id", "")
        conversation: dict = sample.get("conversation", {})
        qa_list: list = sample.get("qa", [])

        if not conversation or not isinstance(conversation, dict):
            continue

        speaker_a = conversation.get("speaker_a", "SpeakerA")
        speaker_b = conversation.get("speaker_b", "SpeakerB")
        conversation_id = f"{locomo_sample_id}_conv"

        session_keys = sorted(
            [k for k in conversation if re.match(r"session_\d+$", k)],
            key=lambda x: int(x.split("_")[1]),
        )

        # Build dia_id → turn_id map for QA evidence linking
        dia_id_to_turn_id: dict[str, str] = {}
        for session_key in session_keys:
            session_id = f"{conversation_id}_{session_key}"
            for i, turn in enumerate(conversation[session_key]):
                dia_id = turn.get("dia_id", "")
                turn_id = f"{session_id}_{dia_id}" if dia_id else f"{session_id}_turn_{i}"
                if dia_id:
                    dia_id_to_turn_id[dia_id] = turn_id

        # Format sessions
        formatted_sessions: List[Dict[str, Any]] = []
        for session_key in session_keys:
            session_id = f"{conversation_id}_{session_key}"
            session_datetime = conversation.get(f"{session_key}_date_time", "")
            turns: list = conversation[session_key]
            if not isinstance(turns, list):
                continue

            formatted_turns = []
            for i, turn in enumerate(turns):
                dia_id = turn.get("dia_id", "")
                turn_id = f"{session_id}_{dia_id}" if dia_id else f"{session_id}_turn_{i}"
                formatted_turns.append({
                    "turn_id": turn_id,
                    "speaker": turn.get("speaker", ""),
                    "text": turn.get("text", ""),
                })

            formatted_sessions.append({
                "sample_id": locomo_sample_id,
                "session_id": session_id,
                "session_turns": formatted_turns,
                "metadata": {
                    "sample_id": locomo_sample_id,
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "session_time": session_datetime,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                },
            })

        # Format QA (filter category 5, resolve evidence turn ids)
        formatted_qa: List[Dict[str, Any]] = []
        for qa_item in qa_list:
            if qa_item.get("category") == 5:
                continue
            evidence_turn_ids = [
                dia_id_to_turn_id[d]
                for d in (qa_item.get("evidence") or [])
                if d in dia_id_to_turn_id
            ]
            formatted_qa.append({
                "sample_id": locomo_sample_id,
                "question": qa_item.get("question"),
                "answer": qa_item.get("answer", "I don't know."),
                "evidence": evidence_turn_ids,
                "category": qa_item.get("category"),
            })

        samples.append({"conversation": formatted_sessions, "qa": formatted_qa})

    return samples

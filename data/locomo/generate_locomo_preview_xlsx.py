from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "locomo10.json"
OUT_DIR = ROOT / "xlsx_preview"


HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=14)
NOTE_FILL = PatternFill("solid", fgColor="EAF2F8")


def session_sort_key(name: str) -> int:
    match = re.search(r"session_(\d+)", name)
    return int(match.group(1)) if match else 10**9


def session_number_from_key(name: str) -> int | None:
    match = re.search(r"session_(\d+)", name)
    return int(match.group(1)) if match else None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def split_evidence_refs(raw: Any) -> list[str]:
    if raw is None:
        return []
    refs: list[str] = []
    values = raw if isinstance(raw, list) else [raw]
    for item in values:
        text = str(item).strip()
        if not text:
            continue
        found = re.findall(r"D\d+:\d+", text)
        if found:
            refs.extend(found)
        else:
            refs.append(text)
    return refs


def add_sheet(wb: Workbook, title: str, rows: list[list[Any]], freeze: str = "A2") -> None:
    ws = wb.create_sheet(title)
    if not rows:
        rows = [["说明"], ["无数据"]]
    for row in rows:
        ws.append(row)

    max_row = ws.max_row
    max_col = ws.max_column
    if max_row >= 1:
        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    ws.freeze_panes = freeze
    ws.auto_filter.ref = ws.dimensions
    if max_row >= 2 and max_col >= 1:
        ref = f"A1:{get_column_letter(max_col)}{max_row}"
        table_name = re.sub(r"[^A-Za-z0-9_]", "_", title)[:20] + "_tbl"
        table = Table(displayName=table_name, ref=ref)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        ws.add_table(table)

    set_widths(ws)


def set_widths(ws) -> None:
    for col_idx in range(1, ws.max_column + 1):
        letter = get_column_letter(col_idx)
        header = str(ws.cell(1, col_idx).value or "")
        if header in {"text", "question", "answer", "adversarial_answer", "session_summary", "event", "observation", "evidence_text"}:
            ws.column_dimensions[letter].width = 58
        elif header in {"img_url", "all_image_urls", "evidence_refs"}:
            ws.column_dimensions[letter].width = 42
        elif header in {"sample_id", "session_datetime", "date"}:
            ws.column_dimensions[letter].width = 18
        else:
            ws.column_dimensions[letter].width = min(max(len(header) + 4, 12), 26)


def build_turn_index(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    conv = sample["conversation"]
    index: dict[str, dict[str, Any]] = {}
    for key in sorted([k for k in conv if re.fullmatch(r"session_\d+", k)], key=session_sort_key):
        session_no = session_number_from_key(key)
        dt = conv.get(f"{key}_date_time", "")
        for turn_idx, turn in enumerate(conv.get(key, []), start=1):
            dia_id = turn.get("dia_id", "")
            if not dia_id:
                continue
            index[dia_id] = {
                "session_no": session_no,
                "session_key": key,
                "session_datetime": dt,
                "turn_index": turn_idx,
                "dia_id": dia_id,
                "speaker": turn.get("speaker", ""),
                "text": turn.get("text", ""),
                "img_url": "; ".join(turn.get("img_url", []) or []),
                "blip_caption": turn.get("blip_caption", ""),
                "query": turn.get("query", ""),
            }
    return index


def sample_rows(sample: dict[str, Any]) -> dict[str, list[list[Any]]]:
    sample_id = sample["sample_id"]
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "")
    speaker_b = conv.get("speaker_b", "")
    session_keys = sorted([k for k in conv if re.fullmatch(r"session_\d+", k)], key=session_sort_key)
    turn_index = build_turn_index(sample)

    conversation_rows = [[
        "sample_id", "session_no", "session_datetime", "turn_index", "dia_id",
        "speaker", "text", "has_image", "img_url", "blip_caption", "query",
    ]]
    image_rows = [[
        "sample_id", "session_no", "session_datetime", "turn_index", "dia_id",
        "speaker", "text", "img_url", "blip_caption", "query",
    ]]
    session_rows = [[
        "sample_id", "session_no", "session_datetime", "speaker_a", "speaker_b",
        "turn_count", "image_turn_count", "qa_count_using_session_evidence",
        "session_summary", "event_count", "observation_count",
    ]]
    summary_rows = [["sample_id", "session_no", "session_datetime", "session_summary"]]

    qa_by_session: Counter[int] = Counter()
    for qa in sample.get("qa", []):
        for ref in split_evidence_refs(qa.get("evidence")):
            turn = turn_index.get(ref)
            if turn and turn.get("session_no") is not None:
                qa_by_session[int(turn["session_no"])] += 1

    for key in session_keys:
        session_no = session_number_from_key(key)
        dt = conv.get(f"{key}_date_time", "")
        turns = conv.get(key, [])
        image_count = 0
        for turn_idx, turn in enumerate(turns, start=1):
            img_urls = "; ".join(turn.get("img_url", []) or [])
            has_image = bool(img_urls)
            image_count += int(has_image)
            row = [
                sample_id, session_no, dt, turn_idx, turn.get("dia_id", ""),
                turn.get("speaker", ""), turn.get("text", ""), has_image,
                img_urls, turn.get("blip_caption", ""), turn.get("query", ""),
            ]
            conversation_rows.append(row)
            if has_image:
                image_rows.append([
                    sample_id, session_no, dt, turn_idx, turn.get("dia_id", ""),
                    turn.get("speaker", ""), turn.get("text", ""), img_urls,
                    turn.get("blip_caption", ""), turn.get("query", ""),
                ])

        summary = sample.get("session_summary", {}).get(f"{key}_summary", "")
        event_obj = sample.get("event_summary", {}).get(f"events_{key}", {})
        obs_obj = sample.get("observation", {}).get(f"{key}_observation", {})
        event_count = sum(len(v) for k, v in event_obj.items() if k != "date" and isinstance(v, list))
        obs_count = sum(len(v) for v in obs_obj.values() if isinstance(v, list))
        session_rows.append([
            sample_id, session_no, dt, speaker_a, speaker_b,
            len(turns), image_count, qa_by_session.get(session_no or -1, 0),
            summary, event_count, obs_count,
        ])
        summary_rows.append([sample_id, session_no, dt, summary])

    qa_rows = [[
        "sample_id", "qa_id", "qa_index", "category", "is_adversarial_cat5",
        "question", "answer", "adversarial_answer", "evidence_count",
        "evidence_refs", "evidence_sessions", "evidence_texts",
    ]]
    evidence_rows = [[
        "sample_id", "qa_id", "qa_index", "category", "evidence_index",
        "raw_evidence", "parsed_ref", "session_no", "session_datetime",
        "dia_id", "speaker", "evidence_text", "reference_status",
    ]]
    for qa_idx, qa in enumerate(sample.get("qa", []), start=1):
        qa_id = f"{sample_id}_q{qa_idx:03d}"
        refs = split_evidence_refs(qa.get("evidence"))
        evidence_sessions = []
        evidence_texts = []
        for ref in refs:
            turn = turn_index.get(ref)
            if turn:
                evidence_sessions.append(str(turn.get("session_no", "")))
                evidence_texts.append(
                    f"{ref} | {turn.get('speaker', '')}: {turn.get('text', '')}"
                )
            else:
                evidence_sessions.append("")
                evidence_texts.append(f"{ref} | [not_found]")
        qa_rows.append([
            sample_id, qa_id, qa_idx, qa.get("category", ""), str(qa.get("category", "")) == "5",
            qa.get("question", ""), qa.get("answer", ""), qa.get("adversarial_answer", ""),
            len(refs), "\n".join(refs),
            "\n".join(evidence_sessions),
            "\n".join(evidence_texts),
        ])
        raw_values = qa.get("evidence", [])
        if not isinstance(raw_values, list):
            raw_values = [raw_values]
        if not refs and not raw_values:
            evidence_rows.append([
                sample_id, qa_id, qa_idx, qa.get("category", ""), "", "", "",
                "", "", "", "", "", "no_evidence",
            ])
        for raw_idx, raw in enumerate(raw_values, start=1):
            parsed_refs = split_evidence_refs([raw])
            if not parsed_refs:
                evidence_rows.append([
                    sample_id, qa_id, qa_idx, qa.get("category", ""), raw_idx, raw, "",
                    "", "", "", "", "", "unparsed",
                ])
            for ref in parsed_refs:
                turn = turn_index.get(ref)
                evidence_rows.append([
                    sample_id, qa_id, qa_idx, qa.get("category", ""), raw_idx, raw, ref,
                    turn.get("session_no", "") if turn else "",
                    turn.get("session_datetime", "") if turn else "",
                    turn.get("dia_id", "") if turn else "",
                    turn.get("speaker", "") if turn else "",
                    turn.get("text", "") if turn else "",
                    "matched" if turn else "not_found",
                ])

    event_rows = [["sample_id", "session_no", "date", "person", "event_index", "event"]]
    for key, value in sorted(sample.get("event_summary", {}).items(), key=lambda kv: session_sort_key(kv[0])):
        session_no = session_number_from_key(key)
        date = value.get("date", "") if isinstance(value, dict) else ""
        if isinstance(value, dict):
            for person, events in value.items():
                if person == "date" or not isinstance(events, list):
                    continue
                for idx, event in enumerate(events, start=1):
                    event_rows.append([sample_id, session_no, date, person, idx, event])

    observation_rows = [[
        "sample_id", "session_no", "person", "observation_index",
        "observation", "evidence_ref", "evidence_text",
    ]]
    for key, value in sorted(sample.get("observation", {}).items(), key=lambda kv: session_sort_key(kv[0])):
        session_no = session_number_from_key(key)
        if not isinstance(value, dict):
            continue
        for person, observations in value.items():
            if not isinstance(observations, list):
                continue
            for idx, item in enumerate(observations, start=1):
                observation = item[0] if isinstance(item, list) and item else normalize_text(item)
                raw_ref = item[1] if isinstance(item, list) and len(item) > 1 else ""
                refs = split_evidence_refs(raw_ref)
                ref = "; ".join(refs)
                turn = turn_index.get(refs[0]) if refs else None
                observation_rows.append([
                    sample_id, session_no, person, idx, observation, ref,
                    turn.get("text", "") if turn else "",
                ])

    qa_category_counts = Counter(str(qa.get("category", "")) for qa in sample.get("qa", []))
    stats_rows = [
        ["metric", "value"],
        ["sample_id", sample_id],
        ["speaker_a", speaker_a],
        ["speaker_b", speaker_b],
        ["session_count", len(session_keys)],
        ["turn_count", len(conversation_rows) - 1],
        ["image_turn_count", len(image_rows) - 1],
        ["qa_count", len(sample.get("qa", []))],
        ["qa_cat1_count", qa_category_counts.get("1", 0)],
        ["qa_cat2_count", qa_category_counts.get("2", 0)],
        ["qa_cat3_count", qa_category_counts.get("3", 0)],
        ["qa_cat4_count", qa_category_counts.get("4", 0)],
        ["qa_cat5_adversarial_count", qa_category_counts.get("5", 0)],
        ["evidence_link_rows", len(evidence_rows) - 1],
        ["event_rows", len(event_rows) - 1],
        ["observation_rows", len(observation_rows) - 1],
    ]

    return {
        "stats": stats_rows,
        "sessions": session_rows,
        "conversation": conversation_rows,
        "qa": qa_rows,
        "qa_evidence": evidence_rows,
        "summaries": summary_rows,
        "events": event_rows,
        "observations": observation_rows,
        "images": image_rows,
    }


def add_readme(wb: Workbook, sample: dict[str, Any]) -> None:
    ws = wb.active
    ws.title = "README"
    conv = sample["conversation"]
    rows = [
        ["LoCoMo sample preview"],
        ["sample_id", sample["sample_id"]],
        ["speaker_a", conv.get("speaker_a", "")],
        ["speaker_b", conv.get("speaker_b", "")],
        ["source", DATA_PATH.name],
        [],
        ["Sheet", "用途"],
        ["Stats", "样本级统计，快速了解会话数、轮次、QA 类别分布。"],
        ["Sessions", "每个会话一行，包含时间、轮次数、摘要、事件/观察数量。"],
        ["Conversation", "逐轮原始对话，含 dia_id、speaker、text、图片 URL 和 caption。"],
        ["QA", "每个问题一行，含类别、答案、adversarial_answer、证据引用和第一条证据文本。"],
        ["QA_Evidence", "把 QA evidence 拆成可筛选的证据行，并回填对应原文。"],
        ["Summaries", "session_summary 原文。"],
        ["Events", "event_summary 按 person/event 展开。"],
        ["Observations", "observation 按 person/observation 展开，并关联证据原文。"],
        ["Images", "只列出带图片的对话轮次。"],
    ]
    for row in rows:
        ws.append(row)
    ws["A1"].font = TITLE_FONT
    ws["A1"].fill = NOTE_FILL
    for cell in ws[7]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 96
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def build_sample_workbook(sample: dict[str, Any], output_path: Path) -> None:
    wb = Workbook()
    add_readme(wb, sample)
    rows_by_sheet = sample_rows(sample)
    sheet_map = [
        ("Stats", rows_by_sheet["stats"]),
        ("Sessions", rows_by_sheet["sessions"]),
        ("Conversation", rows_by_sheet["conversation"]),
        ("QA", rows_by_sheet["qa"]),
        ("QA_Evidence", rows_by_sheet["qa_evidence"]),
        ("Summaries", rows_by_sheet["summaries"]),
        ("Events", rows_by_sheet["events"]),
        ("Observations", rows_by_sheet["observations"]),
        ("Images", rows_by_sheet["images"]),
    ]
    for title, rows in sheet_map:
        add_sheet(wb, title, rows)
    wb.save(output_path)


def build_index_workbook(samples: list[dict[str, Any]], output_path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Index"
    rows = [[
        "sample_id", "speaker_a", "speaker_b", "session_count", "turn_count",
        "image_turn_count", "qa_count", "cat1", "cat2", "cat3", "cat4", "cat5",
        "output_file",
    ]]
    category_summary: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        conv = sample["conversation"]
        session_keys = [k for k in conv if re.fullmatch(r"session_\d+", k)]
        turn_count = sum(len(conv.get(k, [])) for k in session_keys)
        image_turn_count = sum(
            1
            for k in session_keys
            for turn in conv.get(k, [])
            if turn.get("img_url")
        )
        cats = Counter(str(qa.get("category", "")) for qa in sample.get("qa", []))
        category_summary["all"].update(cats)
        rows.append([
            sample["sample_id"], conv.get("speaker_a", ""), conv.get("speaker_b", ""),
            len(session_keys), turn_count, image_turn_count, len(sample.get("qa", [])),
            cats.get("1", 0), cats.get("2", 0), cats.get("3", 0), cats.get("4", 0), cats.get("5", 0),
            f"{sample['sample_id']}_preview.xlsx",
        ])
    for row in rows:
        ws.append(row)
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    set_widths(ws)

    total = wb.create_sheet("Dataset_Stats")
    total_rows = [
        ["metric", "value"],
        ["sample_count", len(samples)],
        ["session_count", sum(row[3] for row in rows[1:])],
        ["turn_count", sum(row[4] for row in rows[1:])],
        ["image_turn_count", sum(row[5] for row in rows[1:])],
        ["qa_count", sum(row[6] for row in rows[1:])],
        ["qa_cat1_count", category_summary["all"].get("1", 0)],
        ["qa_cat2_count", category_summary["all"].get("2", 0)],
        ["qa_cat3_count", category_summary["all"].get("3", 0)],
        ["qa_cat4_count", category_summary["all"].get("4", 0)],
        ["qa_cat5_adversarial_count", category_summary["all"].get("5", 0)],
    ]
    for row in total_rows:
        total.append(row)
    for cell in total[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    set_widths(total)
    wb.save(output_path)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    samples = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    for sample in samples:
        out_path = OUT_DIR / f"{sample['sample_id']}_preview.xlsx"
        build_sample_workbook(sample, out_path)
    build_index_workbook(samples, OUT_DIR / "locomo10_preview_index.xlsx")
    print(f"wrote {len(samples) + 1} xlsx files to {OUT_DIR}")


if __name__ == "__main__":
    main()

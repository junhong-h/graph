from __future__ import annotations

import re
from datetime import UTC, datetime

SNAKE_CASE_RE = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def validate_snake_case(value: str, label: str) -> str:
    if not SNAKE_CASE_RE.fullmatch(value):
        raise ValueError(f"{label} must be snake_case, got: {value!r}")
    return value


def sanitize_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", tag.strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("run tag cannot be empty after sanitization")
    return cleaned


def make_version_id(current_time: datetime | None = None) -> str:
    current_time = current_time or utc_now()
    return "v" + current_time.strftime("%Y%m%d_%H%M%S")


def make_run_id(tag: str, current_time: datetime | None = None) -> str:
    current_time = current_time or utc_now()
    safe_tag = sanitize_tag(tag)
    return current_time.strftime("%Y%m%d_%H%M%S") + "_" + safe_tag

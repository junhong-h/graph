"""Shared filtering helpers for QA runs and evaluation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


LOCOMO_CAT1_4 = {"1", "2", "3", "4"}
NO_CATEGORY_MATCH = "__graphmemory_no_category_match__"


def normalize_filter_values(values: Optional[Iterable[Any]]) -> Optional[set[str]]:
    """Normalize CLI/config filter values to a non-empty string set."""
    if values is None:
        return None
    normalized = {str(v).strip() for v in values if str(v).strip()}
    return normalized or None


def resolve_include_categories(
    categories: Optional[Iterable[Any]],
    *,
    locomo_cat1_4: bool = False,
) -> Optional[set[str]]:
    """Resolve category CLI options, with a shortcut for the LoCoMo Cat1-4 protocol."""
    include = normalize_filter_values(categories)
    if not locomo_cat1_4:
        return include
    if include is None:
        return set(LOCOMO_CAT1_4)
    return include & LOCOMO_CAT1_4 or {NO_CATEGORY_MATCH}


def sample_id_from_sample(sample: Dict[str, Any]) -> str:
    """Return the top-level LoCoMo sample id from raw or session-grouped samples."""
    if sample.get("sample_id"):
        return str(sample["sample_id"])

    conversation = sample.get("conversation") or []
    if conversation:
        metadata = conversation[0].get("metadata", {})
        if metadata.get("sample_id"):
            return str(metadata["sample_id"])
        if conversation[0].get("sample_id"):
            return str(conversation[0]["sample_id"])

    return ""


def filter_samples(
    samples: Iterable[Dict[str, Any]],
    *,
    sample_ids: Optional[Iterable[Any]] = None,
    skip_first: int = 0,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filter samples by id, then apply split-style skip/limit in dataset order."""
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    id_set = normalize_filter_values(sample_ids)
    selected = list(samples)
    if id_set is not None:
        selected = [s for s in selected if sample_id_from_sample(s) in id_set]
    if skip_first:
        selected = selected[skip_first:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def category_allowed(
    category: Any,
    *,
    include_categories: Optional[Iterable[Any]] = None,
    exclude_categories: Optional[Iterable[Any]] = None,
) -> bool:
    """Return whether a QA category passes include/exclude filters."""
    cat = str(category).strip()
    include = normalize_filter_values(include_categories)
    exclude = normalize_filter_values(exclude_categories)
    if include is not None and cat not in include:
        return False
    if exclude is not None and cat in exclude:
        return False
    return True


def iter_filtered_qa(
    qa_items: Iterable[Dict[str, Any]],
    *,
    include_categories: Optional[Iterable[Any]] = None,
    exclude_categories: Optional[Iterable[Any]] = None,
    max_items: Optional[int] = None,
) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield ``(original_index, qa)`` for QA items passing category filters."""
    if max_items is not None and max_items < 0:
        raise ValueError("max_items must be >= 0")

    emitted = 0
    for idx, qa in enumerate(qa_items):
        if not category_allowed(
            qa.get("category", ""),
            include_categories=include_categories,
            exclude_categories=exclude_categories,
        ):
            continue
        if max_items is not None and emitted >= max_items:
            break
        yield idx, qa
        emitted += 1


def record_matches_filters(
    record: Dict[str, Any],
    *,
    sample_ids: Optional[Iterable[Any]] = None,
    include_categories: Optional[Iterable[Any]] = None,
    exclude_categories: Optional[Iterable[Any]] = None,
) -> bool:
    """Return whether an evaluated/result record belongs in the requested split."""
    id_set = normalize_filter_values(sample_ids)
    if id_set is not None and str(record.get("sample_id", "")) not in id_set:
        return False
    return category_allowed(
        record.get("category", ""),
        include_categories=include_categories,
        exclude_categories=exclude_categories,
    )

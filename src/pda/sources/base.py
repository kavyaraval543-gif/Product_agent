"""The contract every feedback source implements.

A source is intentionally dumb: it fetches pages and parses them. It never touches the
database and never decides when to stop. That keeps incremental logic in one place
(pipeline.py) and makes adding a source a ~100-line job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

from pda.models import RawReview


@dataclass
class SourcePage:
    reviews: list[RawReview]
    page_number: int
    has_next: bool
    # True only if this page (and its boundary with the previous page) is verifiably newest-first.
    # The pipeline may stop early on a page of already-known reviews ONLY when this is True.
    ordered_newest_first: bool
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count (deleted, developer, invalid...)


@dataclass
class SourceSnapshot:
    """Source-reported totals, kept so the dashboard can show sampling context honestly."""

    total_text_reviews: int | None
    grouped_ratings: dict[str, int] | None  # star -> count across ALL ratings, with or without text


class Source(Protocol):
    name: str
    snapshot: SourceSnapshot | None

    def iter_pages(self, max_pages: int) -> Iterator[SourcePage]:
        """Yield pages, newest reviews first where the source allows."""
        ...

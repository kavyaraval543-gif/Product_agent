"""Mozilla Add-ons (AMO) review collector.

Endpoint:  GET https://addons.mozilla.org/api/v5/ratings/rating/
Docs:      https://mozilla.github.io/addons-server/topics/api/ratings.html

What was VERIFIED against the live service on 2026-10-17 (see docs/FEASIBILITY.md):
  * robots.txt allows /api/ (only /firefox/downloads/ and locale search/collection paths are disallowed)
  * `filter=without_empty_body` returns only ratings that contain text
  * `page_size` is silently capped at 50
  * results are ordered newest-first across pages (undocumented - so checked on every page)
  * `show_grouped_ratings=true` returns star counts for ALL ratings
  * review permalinks of the form /en-US/firefox/addon/<slug>/reviews/<id>/ return 200 (404 for fake ids)
  * responses carry `Cache-Control: max-age=360`, so brand-new reviews can lag a few minutes

Known limits: the v5 API is documented as "not frozen"; no rate limit is published (we use 1 req/s);
only each user's LATEST rating per add-on is returned; reviewer name/id are discarded on purpose.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pda.errors import SourceFormatError
from pda.http import PoliteClient
from pda.models import RawReview
from pda.sources.base import SourcePage, SourceSnapshot

log = logging.getLogger(__name__)

AMO_BASE = "https://addons.mozilla.org"


class _AmoVersion(BaseModel):
    model_config = ConfigDict(extra="ignore")
    version: str | None = None


class _AmoRating(BaseModel):
    model_config = ConfigDict(extra="ignore")  # reviewer identity fields exist in the payload and are ignored
    id: int
    score: int = Field(ge=1, le=5)
    body: str | None = None
    created: datetime
    is_deleted: bool = False
    is_developer_reply: bool = False
    version: _AmoVersion | None = None


class AmoSource:
    name = "amo"

    def __init__(self, client: PoliteClient, addon_slug: str, page_size: int = 50, base_url: str = AMO_BASE):
        self.client = client
        self.slug = addon_slug
        self.page_size = min(page_size, 50)
        self.base_url = base_url.rstrip("/")
        self.snapshot: SourceSnapshot | None = None

    def permalink(self, rating_id: int) -> str:
        return f"{self.base_url}/en-US/firefox/addon/{self.slug}/reviews/{rating_id}/"

    def iter_pages(self, max_pages: int) -> Iterator[SourcePage]:
        endpoint = f"{self.base_url}/api/v5/ratings/rating/"
        newest_seen_boundary: datetime | None = None  # oldest date on the previous page

        for page in range(1, max_pages + 1):
            params = {"addon": self.slug, "filter": "without_empty_body", "page_size": self.page_size, "page": page}
            if page == 1:
                params["show_grouped_ratings"] = "true"
            data = self.client.get_json(endpoint, params)

            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise SourceFormatError(f"Unexpected AMO response shape on page {page}: missing 'results' list")

            if page == 1:
                self.snapshot = self._parse_snapshot(data)

            reviews: list[RawReview] = []
            skipped: dict[str, int] = {}
            created_dates: list[datetime] = []
            for item in data["results"]:
                try:
                    r = _AmoRating.model_validate(item)
                except ValidationError:
                    skipped["invalid"] = skipped.get("invalid", 0) + 1
                    continue
                created = r.created.astimezone(timezone.utc)
                created_dates.append(created)
                if r.is_deleted:
                    skipped["deleted"] = skipped.get("deleted", 0) + 1
                elif r.is_developer_reply:
                    skipped["developer"] = skipped.get("developer", 0) + 1
                elif not (r.body and r.body.strip()):
                    skipped["no_text"] = skipped.get("no_text", 0) + 1
                else:
                    reviews.append(
                        RawReview(
                            source=self.name,
                            source_review_id=str(r.id),
                            rating=r.score,
                            body=r.body,
                            review_date=created,
                            app_version=r.version.version if r.version else None,
                            url=self.permalink(r.id),
                        )
                    )

            ordered = all(a >= b for a, b in zip(created_dates, created_dates[1:]))
            if ordered and created_dates and newest_seen_boundary is not None:
                ordered = created_dates[0] <= newest_seen_boundary
            if created_dates:
                newest_seen_boundary = min(created_dates)
            if not ordered:
                log.warning("AMO page %d is NOT newest-first; early-stop disabled for this page", page)

            has_next = bool(data.get("next"))
            yield SourcePage(reviews=reviews, page_number=page, has_next=has_next, ordered_newest_first=ordered, skipped=skipped)
            if not has_next:
                return

    @staticmethod
    def _parse_snapshot(data: dict) -> SourceSnapshot:
        count = data.get("count")
        grouped = data.get("grouped_ratings")
        if isinstance(grouped, dict):
            try:
                grouped = {str(k): int(v) for k, v in grouped.items()}
            except (TypeError, ValueError):
                grouped = None
        else:
            grouped = None
        return SourceSnapshot(total_text_reviews=count if isinstance(count, int) else None, grouped_ratings=grouped)

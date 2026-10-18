"""Source registry. To add a source: implement `Source` (see base.py) and register it here."""

from __future__ import annotations

from pda.config import Settings
from pda.http import PoliteClient
from pda.sources.amo import AmoSource
from pda.sources.base import Source


def build_client(settings: Settings) -> PoliteClient:
    c = settings.collector
    return PoliteClient(
        user_agent=settings.user_agent,
        min_interval_s=c.min_request_interval_s,
        timeout_s=c.request_timeout_s,
        max_retries=c.max_retries,
    )


def build_source(settings: Settings, client: PoliteClient) -> Source:
    if settings.app.source == "amo":
        return AmoSource(client, settings.app.slug, page_size=settings.collector.page_size)
    raise ValueError(f"Unknown source: {settings.app.source}")

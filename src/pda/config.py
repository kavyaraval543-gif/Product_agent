"""Typed settings. Defaults work out of the box; config.toml overrides them."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos in config.toml should fail loudly


class AppConfig(_Section):
    source: Literal["amo"] = "amo"
    slug: str = "bitwarden-password-manager"
    name: str = "Bitwarden Password Manager"


class CollectorConfig(_Section):
    contact: str = "https://github.com/YOUR-USERNAME/product-discovery-agent"
    min_request_interval_s: float = Field(1.0, ge=0.5)  # floor keeps us polite even if misconfigured
    page_size: int = Field(50, ge=1, le=50)
    initial_backfill_limit: int = Field(300, ge=1)
    max_pages_per_run: int = Field(20, ge=1)
    request_timeout_s: float = Field(20.0, gt=0)
    max_retries: int = Field(3, ge=0, le=6)


class CleaningConfig(_Section):
    min_words: int = Field(4, ge=1)
    languages: list[str] = ["en"]
    duplicate_min_words: int = Field(12, ge=1)


class LLMConfig(_Section):
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    num_ctx: int = Field(4096, ge=1024)
    timeout_s: float = Field(180.0, gt=0)
    max_retries: int = Field(2, ge=0, le=5)
    embedding_model: str = "nomic-embed-text"  # ~274 MB; only used for grouping similar complaints


class GroupingConfig(_Section):
    """How similar complaints are merged into issue groups."""

    method: Literal["embeddings", "tfidf"] = "embeddings"
    # Cosine distance on mean-centred embeddings. Lower = finer, more groups. See docs/MVP_RESULTS.md for the sweep.
    embedding_distance_threshold: float = Field(0.75, gt=0.0, lt=1.5)


class StorageConfig(_Section):
    db_path: str = "data/pda.sqlite3"


class Settings(_Section):
    app: AppConfig = AppConfig()
    collector: CollectorConfig = CollectorConfig()
    cleaning: CleaningConfig = CleaningConfig()
    llm: LLMConfig = LLMConfig()
    grouping: GroupingConfig = GroupingConfig()
    storage: StorageConfig = StorageConfig()

    @property
    def user_agent(self) -> str:
        from pda import __version__

        return f"ProductDiscoveryAgent/{__version__} (+{self.collector.contact}; read-only research project)"


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    """Load settings from `path`, $PDA_CONFIG, or ./config.toml; fall back to defaults."""
    candidate = path or os.environ.get("PDA_CONFIG") or "config.toml"
    p = Path(candidate)
    if not p.exists():
        if path or os.environ.get("PDA_CONFIG"):
            raise FileNotFoundError(f"Config file not found: {p}")
        return Settings()
    with p.open("rb") as fh:
        return Settings.model_validate(tomllib.load(fh))

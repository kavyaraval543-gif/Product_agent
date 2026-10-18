"""SQLite persistence. One file, no server, migration-ready (PRAGMA user_version).

Design notes
  * reviews are unique on (source, source_review_id): re-collecting never duplicates.
  * Content duplicates (same long text under different ids) are kept but flagged and excluded.
  * review_analysis is keyed by (review_id, model, prompt_version): changing the model or the
    prompt re-analyses; nothing else does. This is what makes runs incremental.
  * Reviewer identity (name, user id) is deliberately NOT stored (data minimisation).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pda.models import CleanedReview, ReviewAnalysisOut

SCHEMA_V1 = """
CREATE TABLE apps (
    id         INTEGER PRIMARY KEY,
    source     TEXT NOT NULL,
    slug       TEXT NOT NULL,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (source, slug)
);

CREATE TABLE runs (
    id          INTEGER PRIMARY KEY,
    app_id      INTEGER NOT NULL REFERENCES apps(id),
    kind        TEXT NOT NULL,          -- collect | analyze | pipeline
    trigger     TEXT NOT NULL,          -- manual | scheduled | agent
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL,          -- running | success | partial | failed
    stats_json  TEXT,
    error       TEXT
);

CREATE TABLE run_events (
    id        INTEGER PRIMARY KEY,
    run_id    INTEGER NOT NULL REFERENCES runs(id),
    ts        TEXT NOT NULL,
    step      TEXT NOT NULL,
    level     TEXT NOT NULL,            -- info | warning | error
    message   TEXT NOT NULL,
    data_json TEXT
);

CREATE TABLE reviews (
    id               INTEGER PRIMARY KEY,
    app_id           INTEGER NOT NULL REFERENCES apps(id),
    source           TEXT NOT NULL,
    source_review_id TEXT NOT NULL,
    rating           INTEGER,
    body_raw         TEXT,
    body_clean       TEXT NOT NULL,
    language         TEXT,
    app_version      TEXT,
    review_date      TEXT NOT NULL,     -- when the reviewer wrote it (UTC ISO-8601)
    url              TEXT NOT NULL,     -- permalink to the original
    collected_at     TEXT NOT NULL,     -- when WE first saw it (UTC ISO-8601)
    collection_run_id INTEGER REFERENCES runs(id),
    content_hash     TEXT NOT NULL,
    word_count       INTEGER NOT NULL,
    usable           INTEGER NOT NULL,
    unusable_reason  TEXT,
    duplicate_of     INTEGER REFERENCES reviews(id),
    UNIQUE (source, source_review_id)
);
CREATE INDEX idx_reviews_app_date ON reviews (app_id, review_date);
CREATE INDEX idx_reviews_hash ON reviews (app_id, content_hash);

CREATE TABLE review_analysis (
    id               INTEGER PRIMARY KEY,
    review_id        INTEGER NOT NULL REFERENCES reviews(id),
    model            TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    analyzed_at      TEXT NOT NULL,
    status           TEXT NOT NULL,     -- ok | failed
    error            TEXT,
    relevance        TEXT,
    sentiment        TEXT,
    theme            TEXT,
    severity         TEXT,
    issue_label      TEXT,
    problem_statement TEXT,
    feature_request  TEXT,
    quote            TEXT,              -- the model's quote, ONLY if verified verbatim in the review
    quote_verified   INTEGER,
    latency_s        REAL,
    eval_tokens      INTEGER,
    raw_json         TEXT,
    UNIQUE (review_id, model, prompt_version)
);

CREATE TABLE source_snapshots (
    id                  INTEGER PRIMARY KEY,
    app_id              INTEGER NOT NULL REFERENCES apps(id),
    run_id              INTEGER REFERENCES runs(id),
    captured_at         TEXT NOT NULL,
    total_text_reviews  INTEGER,
    grouped_ratings_json TEXT           -- {"1": n, ..., "5": n} over ALL ratings, with or without text
);
"""

SCHEMA_V2 = """
CREATE TABLE review_embeddings (
    review_id   INTEGER NOT NULL REFERENCES reviews(id),
    model       TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,          -- hash of the embedded text; a changed analysis => a new embedding
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,          -- float32, little-endian
    created_at  TEXT NOT NULL,
    PRIMARY KEY (review_id, model, text_sha256)
);
"""

SCHEMA_V3 = "ALTER TABLE review_analysis ADD COLUMN overrides TEXT;"

MIGRATIONS = [SCHEMA_V1, SCHEMA_V2, SCHEMA_V3]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class InsertStats:
    inserted: int = 0
    already_known: int = 0
    usable: int = 0
    unusable: int = 0
    duplicate_content: int = 0
    unusable_reasons: dict[str, int] = field(default_factory=dict)

    def merge(self, other: "InsertStats") -> None:
        self.inserted += other.inserted
        self.already_known += other.already_known
        self.usable += other.usable
        self.unusable += other.unusable
        self.duplicate_content += other.duplicate_content
        for k, v in other.unusable_reasons.items():
            self.unusable_reasons[k] = self.unusable_reasons.get(k, 0) + v

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        self.conn.close()

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for i, script in enumerate(MIGRATIONS, start=1):
            if i > version:
                with self.conn:
                    self.conn.executescript(script)
                    self.conn.execute(f"PRAGMA user_version = {i}")

    # ------------------------------------------------------------------ apps & runs

    def upsert_app(self, source: str, slug: str, name: str) -> int:
        with self.conn:
            self.conn.execute(
                "INSERT INTO apps (source, slug, name, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(source, slug) DO UPDATE SET name = excluded.name",
                (source, slug, name, utcnow_iso()),
            )
        row = self.conn.execute("SELECT id FROM apps WHERE source=? AND slug=?", (source, slug)).fetchone()
        return int(row["id"])

    def start_run(self, app_id: int, kind: str, trigger: str = "manual") -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO runs (app_id, kind, trigger, started_at, status) VALUES (?,?,?,?, 'running')",
                (app_id, kind, trigger, utcnow_iso()),
            )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, stats: dict[str, Any] | None = None, error: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET finished_at=?, status=?, stats_json=?, error=? WHERE id=?",
                (utcnow_iso(), status, json.dumps(stats or {}), error, run_id),
            )

    def log_event(self, run_id: int, step: str, message: str, level: str = "info", data: dict[str, Any] | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO run_events (run_id, ts, step, level, message, data_json) VALUES (?,?,?,?,?,?)",
                (run_id, utcnow_iso(), step, level, message, json.dumps(data) if data else None),
            )

    def recent_runs(self, limit: int = 10) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    def run_events(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM run_events WHERE run_id=? ORDER BY id", (run_id,)).fetchall()

    def last_successful_run(self, kind: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE kind=? AND status='success' ORDER BY id DESC LIMIT 1", (kind,)
        ).fetchone()

    # ------------------------------------------------------------------ reviews

    def existing_source_ids(self, source: str, ids: list[str]) -> set[str]:
        if not ids:
            return set()
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT source_review_id FROM reviews WHERE source=? AND source_review_id IN ({marks})", [source, *ids]
        ).fetchall()
        return {r["source_review_id"] for r in rows}

    def insert_reviews(self, app_id: int, run_id: int | None, cleaned: list[CleanedReview], duplicate_min_words: int = 12) -> InsertStats:
        """Insert cleaned reviews idempotently. Safe to call repeatedly with overlapping data."""
        stats = InsertStats()
        collected_at = utcnow_iso()
        # Oldest first, so the earliest copy of a text is the "original" and later copies are flagged.
        for item in sorted(cleaned, key=lambda c: c.raw.review_date):
            raw = item.raw
            usable, reason, dup_of = item.usable, item.unusable_reason, None
            if usable and item.word_count >= duplicate_min_words:
                row = self.conn.execute(
                    "SELECT id FROM reviews WHERE app_id=? AND content_hash=? AND source_review_id<>? ORDER BY review_date, id LIMIT 1",
                    (app_id, item.content_hash, raw.source_review_id),
                ).fetchone()
                if row is not None:
                    usable, reason, dup_of = False, "duplicate_content", int(row["id"])
            with self.conn:
                cur = self.conn.execute(
                    "INSERT OR IGNORE INTO reviews (app_id, source, source_review_id, rating, body_raw, body_clean, language,"
                    " app_version, review_date, url, collected_at, collection_run_id, content_hash, word_count, usable,"
                    " unusable_reason, duplicate_of) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        app_id, raw.source, raw.source_review_id, raw.rating, raw.body, item.body_clean, item.language,
                        raw.app_version, raw.review_date.isoformat(), raw.url, collected_at, run_id, item.content_hash,
                        item.word_count, int(usable), reason, dup_of,
                    ),
                )
            if cur.rowcount == 0:
                stats.already_known += 1
                continue
            stats.inserted += 1
            if usable:
                stats.usable += 1
            else:
                stats.unusable += 1
                stats.unusable_reasons[reason or "unknown"] = stats.unusable_reasons.get(reason or "unknown", 0) + 1
                if reason == "duplicate_content":
                    stats.duplicate_content += 1
        return stats

    def stored_reviews_oldest_first(self, app_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM reviews WHERE app_id=? ORDER BY review_date, id", (app_id,)
        ).fetchall()

    def update_cleaning(self, review_id: int, item: CleanedReview, usable: bool, reason: str | None, duplicate_of: int | None) -> None:
        """Overwrite cleaning results for one stored review. Analyses are untouched."""
        with self.conn:
            self.conn.execute(
                "UPDATE reviews SET body_clean=?, language=?, content_hash=?, word_count=?, usable=?, unusable_reason=?, duplicate_of=?"
                " WHERE id=?",
                (item.body_clean, item.language, item.content_hash, item.word_count, int(usable), reason, duplicate_of, review_id),
            )

    def save_snapshot(self, app_id: int, run_id: int | None, total_text_reviews: int | None, grouped_ratings: dict[str, int] | None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO source_snapshots (app_id, run_id, captured_at, total_text_reviews, grouped_ratings_json) VALUES (?,?,?,?,?)",
                (app_id, run_id, utcnow_iso(), total_text_reviews, json.dumps(grouped_ratings) if grouped_ratings else None),
            )

    def latest_snapshot(self, app_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM source_snapshots WHERE app_id=? ORDER BY id DESC LIMIT 1", (app_id,)
        ).fetchone()

    def review_counts(self, app_id: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, SUM(usable) AS usable, MIN(review_date) AS oldest, MAX(review_date) AS newest "
            "FROM reviews WHERE app_id=?",
            (app_id,),
        ).fetchone()
        reasons = self.conn.execute(
            "SELECT unusable_reason AS reason, COUNT(*) AS n FROM reviews WHERE app_id=? AND usable=0 GROUP BY 1 ORDER BY 2 DESC",
            (app_id,),
        ).fetchall()
        return {
            "total": row["total"] or 0,
            "usable": row["usable"] or 0,
            "oldest": row["oldest"],
            "newest": row["newest"],
            "unusable_reasons": {r["reason"]: r["n"] for r in reasons},
        }

    # ------------------------------------------------------------------ analysis

    def pending_for_analysis(
        self, app_id: int, model: str, prompt_version: str, limit: int | None = None, only_source_ids: set[str] | None = None
    ) -> list[sqlite3.Row]:
        """Usable reviews with NO analysis row (ok or failed) for this model + prompt. Newest first.

        `only_source_ids` restricts to specific reviews (used to try a prompt on a labelled subset only).
        """
        sql = (
            "SELECT r.* FROM reviews r WHERE r.app_id=? AND r.usable=1 AND NOT EXISTS ("
            " SELECT 1 FROM review_analysis a WHERE a.review_id=r.id AND a.model=? AND a.prompt_version=?)"
        )
        params: list[Any] = [app_id, model, prompt_version]
        if only_source_ids is not None:
            sql += " AND r.source_review_id IN (%s)" % ",".join("?" * len(only_source_ids))
            params.extend(sorted(only_source_ids))
        sql += " ORDER BY r.review_date DESC, r.id DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    def save_analysis(
        self,
        review_id: int,
        model: str,
        prompt_version: str,
        out: ReviewAnalysisOut,
        quote_verified: bool,
        latency_s: float,
        eval_tokens: int | None,
        raw: ReviewAnalysisOut | None = None,
        overrides: list[str] | None = None,
    ) -> None:
        """`out` = what is used downstream; `raw` = exactly what the model said (kept for audit)."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO review_analysis (review_id, model, prompt_version, analyzed_at, status, relevance,"
                " sentiment, theme, severity, issue_label, problem_statement, feature_request, quote, quote_verified,"
                " latency_s, eval_tokens, raw_json, overrides) VALUES (?,?,?,?, 'ok', ?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    review_id, model, prompt_version, utcnow_iso(), out.relevance, out.sentiment, out.theme, out.severity,
                    out.issue_label, out.problem_statement, out.feature_request,
                    out.supporting_quote if quote_verified else None, int(quote_verified), latency_s, eval_tokens,
                    (raw or out).model_dump_json(), ",".join(overrides) if overrides else None,
                ),
            )

    def save_analysis_failure(self, review_id: int, model: str, prompt_version: str, error: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO review_analysis (review_id, model, prompt_version, analyzed_at, status, error)"
                " VALUES (?,?,?,?, 'failed', ?)",
                (review_id, model, prompt_version, utcnow_iso(), error[:500]),
            )

    def delete_failed_analyses(self, app_id: int, model: str, prompt_version: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM review_analysis WHERE status='failed' AND model=? AND prompt_version=? AND review_id IN"
                " (SELECT id FROM reviews WHERE app_id=?)",
                (model, prompt_version, app_id),
            )
        return cur.rowcount

    def cached_embeddings(self, model: str, wanted: dict[int, str]) -> dict[int, "np.ndarray"]:
        """{review_id: vector} for the (review_id -> text_sha256) pairs already embedded with `model`."""
        if not wanted:
            return {}
        rows = self.conn.execute(
            "SELECT review_id, text_sha256, vector FROM review_embeddings WHERE model=? AND review_id IN (%s)"
            % ",".join("?" * len(wanted)),
            [model, *wanted],
        ).fetchall()
        return {r["review_id"]: np.frombuffer(r["vector"], dtype=np.float32) for r in rows if wanted[r["review_id"]] == r["text_sha256"]}

    def save_embeddings(self, model: str, items: list[tuple[int, str, "np.ndarray"]]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO review_embeddings (review_id, model, text_sha256, dim, vector, created_at) VALUES (?,?,?,?,?,?)",
                [(rid, model, sha, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes(), utcnow_iso()) for rid, sha, v in items],
            )

    def analysis_counts(self, app_id: int, model: str, prompt_version: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT a.status AS status, COUNT(*) AS n FROM review_analysis a JOIN reviews r ON r.id=a.review_id "
            "WHERE r.app_id=? AND a.model=? AND a.prompt_version=? GROUP BY 1",
            (app_id, model, prompt_version),
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def analysed_df(self, app_id: int, model: str, prompt_version: str) -> pd.DataFrame:
        """One row per successfully analysed review, joined with its measured fields."""
        return pd.read_sql_query(
            "SELECT r.id AS review_id, r.source_review_id, r.rating, r.body_clean, r.review_date, r.url, r.app_version,"
            " a.relevance, a.sentiment, a.theme, a.severity, a.issue_label, a.problem_statement, a.feature_request,"
            " a.quote, a.quote_verified"
            " FROM reviews r JOIN review_analysis a ON a.review_id=r.id"
            " WHERE r.app_id=? AND r.usable=1 AND a.status='ok' AND a.model=? AND a.prompt_version=?",
            self.conn,
            params=(app_id, model, prompt_version),
            parse_dates=["review_date"],
        )

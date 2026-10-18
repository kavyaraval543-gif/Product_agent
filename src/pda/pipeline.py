"""Orchestration: Discover -> Clean -> Understand, as incremental, recorded runs.

Every run is a row in `runs` with an event trail in `run_events`, so "what did the agent do,
and did it work?" is always answerable from the database (this feeds the Agent Activity page).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from pda.analyzer import PROMPT_VERSION, ReviewAnalyzer
from pda.cleaning import clean_review
from pda.config import Settings
from pda.db import Database, InsertStats
from pda.errors import HttpError, LLMModelMissing, LLMOutputError, LLMUnavailable, RobotsDisallowed, SourceFormatError
from pda.llm import OllamaClient
from pda.themes import display_label, embedding_text
from pda.models import RawReview
from pda.sources import build_client, build_source
from pda.sources.base import Source

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_ANALYSIS_FAILURES = 5


@dataclass
class CollectResult:
    run_id: int
    status: str  # success | partial | failed
    stop_reason: str
    pages_fetched: int
    requests_made: int
    fetched_reviews: int
    inserts: InsertStats
    skipped_by_source: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "pages_fetched": self.pages_fetched,
            "requests_made": self.requests_made,
            "fetched_reviews": self.fetched_reviews,
            "skipped_by_source": self.skipped_by_source,
            **{f"insert_{k}": v for k, v in self.inserts.as_dict().items()},
        }


@dataclass
class AnalyzeResult:
    run_id: int
    status: str
    attempted: int
    succeeded: int
    failed: int
    quote_verified: int
    overridden: int
    mean_latency_s: float | None
    effective_tokens_per_s: float | None  # output tokens / wall time (includes prompt processing)
    stop_reason: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("run_id")
        return d


# ----------------------------------------------------------------------------- collect


def collect(
    settings: Settings,
    db: Database,
    *,
    source: Source | None = None,
    trigger: str = "manual",
    full: bool = False,
    limit: int | None = None,
    max_pages: int | None = None,
) -> CollectResult:
    """Fetch new feedback.

    * First run (nothing stored): backfill up to `limit` / initial_backfill_limit new reviews.
    * Later runs: incremental - stop at the first fully-known page (when ordering is verified).
    * `full=True`: never stop on known pages; keep going until `limit` new reviews, the end, or max pages.
    """
    app_cfg = settings.app
    app_id = db.upsert_app(app_cfg.source, app_cfg.slug, app_cfg.name)
    run_id = db.start_run(app_id, "collect", trigger)
    client = None
    if source is None:
        client = build_client(settings)
        source = build_source(settings, client)

    first_run = db.review_counts(app_id)["total"] == 0
    cap = limit if limit else (settings.collector.initial_backfill_limit if (first_run or full) else None)
    pages_allowed = max_pages or settings.collector.max_pages_per_run
    db.log_event(
        run_id, "plan", f"Collecting {app_cfg.name} from {source.name}",
        data={"first_run": first_run, "full": full, "new_review_cap": cap, "max_pages": pages_allowed},
    )

    inserts = InsertStats()
    skipped: Counter[str] = Counter()
    pages = fetched = 0
    stop_reason, status, error = "end_of_results", "success", None

    try:
        for page in source.iter_pages(pages_allowed):
            pages += 1
            fetched += len(page.reviews)
            skipped.update(page.skipped)
            known = db.existing_source_ids(source.name, [r.source_review_id for r in page.reviews])
            fresh = [r for r in page.reviews if r.source_review_id not in known]
            stats = db.insert_reviews(
                app_id, run_id, [clean_review(r, settings.cleaning) for r in fresh], settings.cleaning.duplicate_min_words
            )
            stats.already_known += len(known)
            inserts.merge(stats)
            db.log_event(
                run_id, "fetch_page", f"page {page.page_number}: {len(page.reviews)} reviews, {len(fresh)} new",
                data={"page": page.page_number, "ordered_newest_first": page.ordered_newest_first, **stats.as_dict()},
            )
            if cap and inserts.inserted >= cap:
                stop_reason = "backfill_limit_reached"
                break
            if not full and page.ordered_newest_first and not fresh:
                stop_reason = "caught_up"  # a whole page of already-known reviews: everything newer is stored
                break
            if not page.has_next:
                stop_reason = "end_of_results"
                break
        else:
            stop_reason = "max_pages_reached"
    except RobotsDisallowed as exc:
        status, stop_reason, error = ("partial" if pages else "failed"), "robots_disallowed", str(exc)
        db.log_event(run_id, "blocked", str(exc), level="error")
    except (HttpError, SourceFormatError) as exc:
        status, stop_reason, error = ("partial" if pages else "failed"), "source_error", str(exc)
        db.log_event(run_id, "source_error", str(exc), level="error")

    snap = getattr(source, "snapshot", None)
    if snap is not None:
        db.save_snapshot(app_id, run_id, snap.total_text_reviews, snap.grouped_ratings)

    requests_made = client.request_count if client else 0
    result = CollectResult(
        run_id=run_id, status=status, stop_reason=stop_reason, pages_fetched=pages, requests_made=requests_made,
        fetched_reviews=fetched, inserts=inserts, skipped_by_source=dict(skipped), error=error,
    )
    db.log_event(run_id, "done", f"collect finished: {status} ({stop_reason})", data=result.as_dict())
    db.finish_run(run_id, status, result.as_dict(), error)
    return result


# ----------------------------------------------------------------------------- embeddings


def compute_embeddings(settings: Settings, db: Database, llm: OllamaClient, df) -> dict[int, "np.ndarray"]:
    """{review_id: embedding} for the labelled problem reviews in `df`, computing only what is not cached.

    Raises LLMUnavailable / LLMModelMissing / LLMOutputError; callers fall back to lexical grouping.
    """
    import hashlib

    import numpy as np

    model = settings.llm.embedding_model
    texts: dict[int, str] = {}
    for r in df.itertuples():
        if r.relevance in ("genuine_feedback", "ambiguous") and r.theme != "no_issue" and display_label(r.issue_label, r.problem_statement):
            texts[int(r.review_id)] = embedding_text(r.issue_label, r.problem_statement)
    shas = {rid: hashlib.sha256(t.encode()).hexdigest()[:16] for rid, t in texts.items()}
    have = db.cached_embeddings(model, shas)
    missing = [rid for rid in texts if rid not in have]
    if missing:
        vecs = llm.embed([texts[rid] for rid in missing], model=model)
        fresh = [(rid, shas[rid], np.asarray(v, dtype=np.float32)) for rid, v in zip(missing, vecs)]
        db.save_embeddings(model, fresh)
        have.update({rid: v for rid, _, v in fresh})
    return have


# ----------------------------------------------------------------------------- reclean


@dataclass
class RecleanResult:
    run_id: int
    reviews: int
    changed: int
    became_usable: int
    became_unusable: int


def reclean(settings: Settings, db: Database, *, trigger: str = "manual") -> RecleanResult:
    """Re-apply the CURRENT cleaning rules to stored raw text (no network access).

    Cleaning rules evolve (e.g. a language-detection fix); this lets existing data benefit without
    re-fetching. Analyses are kept; reviews that become usable show up as pending for `analyze`.
    """
    from datetime import datetime

    app_cfg = settings.app
    app_id = db.upsert_app(app_cfg.source, app_cfg.slug, app_cfg.name)
    run_id = db.start_run(app_id, "reclean", trigger)
    first_by_hash: dict[str, int] = {}
    changed = up = down = 0
    rows = db.stored_reviews_oldest_first(app_id)
    for row in rows:
        raw = RawReview(
            source=row["source"], source_review_id=row["source_review_id"], rating=row["rating"], body=row["body_raw"],
            review_date=datetime.fromisoformat(row["review_date"]), app_version=row["app_version"], url=row["url"],
        )
        item = clean_review(raw, settings.cleaning)
        usable, reason, dup_of = item.usable, item.unusable_reason, None
        if usable and item.word_count >= settings.cleaning.duplicate_min_words:
            if item.content_hash in first_by_hash:
                usable, reason, dup_of = False, "duplicate_content", first_by_hash[item.content_hash]
            else:
                first_by_hash[item.content_hash] = int(row["id"])
        if (bool(row["usable"]), row["unusable_reason"], row["body_clean"], row["language"], row["duplicate_of"]) != (
            usable, reason, item.body_clean, item.language, dup_of,
        ):
            changed += 1
            up += int(usable and not row["usable"])
            down += int(not usable and bool(row["usable"]))
            db.update_cleaning(int(row["id"]), item, usable, reason, dup_of)
    res = RecleanResult(run_id, len(rows), changed, up, down)
    db.log_event(run_id, "done", f"reclean: {changed}/{len(rows)} reviews changed", data=res.__dict__)
    db.finish_run(run_id, "success", {k: v for k, v in res.__dict__.items() if k != "run_id"})
    return res


# ----------------------------------------------------------------------------- analyze


def analyze(
    settings: Settings,
    db: Database,
    llm: OllamaClient,
    *,
    trigger: str = "manual",
    limit: int | None = None,
    retry_failed: bool = False,
    only_source_ids: set[str] | None = None,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> AnalyzeResult:
    """Analyse usable reviews that have no analysis for the current model + prompt version."""
    app_cfg = settings.app
    app_id = db.upsert_app(app_cfg.source, app_cfg.slug, app_cfg.name)
    run_id = db.start_run(app_id, "analyze", trigger)

    def finish(status: str, reason: str, attempted=0, ok=0, failed=0, verified=0, lat=None, tps=None, error=None, overridden=0) -> AnalyzeResult:
        res = AnalyzeResult(run_id, status, attempted, ok, failed, verified, overridden, lat, tps, reason, error)
        db.log_event(run_id, "done", f"analyze finished: {status} ({reason})", data=res.as_dict())
        db.finish_run(run_id, status, res.as_dict(), error)
        return res

    try:
        version = llm.ensure_ready()
    except (LLMUnavailable, LLMModelMissing) as exc:
        db.log_event(run_id, "preflight", str(exc), level="error")
        return finish("failed", "llm_not_ready", error=str(exc))

    if retry_failed:
        n = db.delete_failed_analyses(app_id, llm.model, PROMPT_VERSION)
        db.log_event(run_id, "retry_failed", f"cleared {n} failed analyses for retry")

    pending = db.pending_for_analysis(app_id, llm.model, PROMPT_VERSION, limit, only_source_ids)
    db.log_event(
        run_id, "plan", f"{len(pending)} reviews to analyse with {llm.model} (Ollama {version}, prompt {PROMPT_VERSION})",
        data={"pending": len(pending), "model": llm.model, "prompt_version": PROMPT_VERSION},
    )
    analyzer = ReviewAnalyzer(llm, app_cfg.name)

    ok = failed = verified = consecutive = overridden = 0
    latencies: list[float] = []
    tokens = 0.0
    eval_seconds = 0.0
    reason = "all_pending_done"
    error: str | None = None
    status = "success"

    for i, row in enumerate(pending, start=1):
        try:
            res = analyzer.analyze(row)
        except LLMUnavailable as exc:
            status, reason, error = "partial", "llm_unavailable", str(exc)
            db.log_event(run_id, "llm_down", str(exc), level="error")
            break
        except LLMModelMissing as exc:
            status, reason, error = "failed", "llm_model_missing", str(exc)
            db.log_event(run_id, "llm_model_missing", str(exc), level="error")
            break
        except LLMOutputError as exc:
            failed += 1
            consecutive += 1
            db.save_analysis_failure(row["id"], llm.model, PROMPT_VERSION, str(exc))
            db.log_event(run_id, "analysis_failed", f"review {row['id']}: {exc}", level="warning")
            if consecutive >= MAX_CONSECUTIVE_ANALYSIS_FAILURES:
                status, reason = "partial", "too_many_consecutive_failures"
                error = f"{consecutive} consecutive failures; stopping to avoid wasting time. Check the model/logs."
                db.log_event(run_id, "abort", error, level="error")
                break
            continue

        consecutive = 0
        ok += 1
        verified += int(res.quote_verified)
        overridden += int(bool(res.overrides))
        latencies.append(res.meta.latency_s)
        if res.meta.eval_tokens:
            tokens += res.meta.eval_tokens
            eval_seconds += res.meta.latency_s
        db.save_analysis(row["id"], llm.model, PROMPT_VERSION, res.out, res.quote_verified, res.meta.latency_s, res.meta.eval_tokens,
                         raw=res.raw, overrides=res.overrides)
        if progress:
            progress(i, len(pending), {"review_id": row["id"], "theme": res.out.theme, "label": res.out.issue_label, "verified": res.quote_verified})

    if failed and status == "success":
        status = "partial"
    mean_lat = round(sum(latencies) / len(latencies), 2) if latencies else None
    tps = round(tokens / eval_seconds, 2) if eval_seconds else None
    return finish(status, reason, ok + failed, ok, failed, verified, mean_lat, tps, error, overridden)

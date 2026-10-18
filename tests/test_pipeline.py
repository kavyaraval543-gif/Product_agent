"""End-to-end workflow tests with mocked source and LLM (no network, no Ollama)."""

import pytest

from helpers import FakeLLM, FakeSource, default_responder, make_review
from pda.analyzer import PROMPT_VERSION
from pda.config import Settings
from pda.db import Database
from pda.errors import HttpError, LLMModelMissing, LLMOutputError, LLMUnavailable, RobotsDisallowed
from pda.models import ReviewAnalysisOut
from pda.pipeline import MAX_CONSECUTIVE_ANALYSIS_FAILURES, analyze, collect


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def settings():
    s = Settings()
    s.collector.initial_backfill_limit = 12
    return s


def reviews_newest_first(n, start=1):
    return [make_review(i) for i in range(start + n - 1, start - 1, -1)]  # ids n..1, newest first


# ----------------------------------------------------------------------------- collect


def test_first_run_backfills_up_to_the_limit_and_stops(db, settings):
    src = FakeSource(reviews_newest_first(30), page_size=5)
    res = collect(settings, db, source=src)
    assert res.status == "success" and res.stop_reason == "backfill_limit_reached"
    assert res.inserts.inserted == 15  # three pages of 5: the cap (12) is checked after each page
    assert db.review_counts(1)["total"] == 15
    assert src.pages_served == 3


def test_second_run_is_incremental_and_stops_at_the_first_fully_known_page(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(10), page_size=5))
    # 3 brand-new reviews (ids 11-13) appear on top of the 10 already stored
    src = FakeSource(reviews_newest_first(13), page_size=5)
    res = collect(settings, db, source=src)
    assert res.stop_reason == "caught_up"
    assert res.inserts.inserted == 3
    assert src.pages_served == 2  # page 1 = ids 13..9 (3 new), page 2 = ids 8..4 (all known) -> stop
    assert db.review_counts(1)["total"] == 13


def test_run_with_nothing_new_inserts_nothing_and_costs_one_page(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(10), page_size=5))
    src = FakeSource(reviews_newest_first(10), page_size=5)
    res = collect(settings, db, source=src)
    assert res.inserts.inserted == 0 and res.inserts.already_known == 5
    assert res.stop_reason == "caught_up" and src.pages_served == 1


def test_unverified_ordering_disables_early_stop(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(10), page_size=5))
    src = FakeSource(reviews_newest_first(10), page_size=5, ordered=False)
    res = collect(settings, db, source=src)
    assert src.pages_served == 2 and res.stop_reason == "end_of_results"  # had to read everything


def test_full_mode_never_stops_on_known_pages_and_reaches_older_history(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(10, start=21), page_size=5))  # ids 21..30 stored
    src = FakeSource(reviews_newest_first(30), page_size=5)  # source also has older ids 1..20
    res = collect(settings, db, source=src, full=True, limit=7)
    assert res.inserts.inserted >= 7 and res.stop_reason == "backfill_limit_reached"
    assert db.review_counts(1)["total"] > 10


def test_robots_block_marks_run_failed_and_records_why(db, settings):
    src = FakeSource([], fail_on_page=1, fail_with=RobotsDisallowed("https://x/api", "/api"))
    src.reviews = reviews_newest_first(5)
    res = collect(settings, db, source=src)
    assert res.status == "failed" and res.stop_reason == "robots_disallowed"
    run = db.recent_runs(1)[0]
    assert run["status"] == "failed" and "robots.txt disallows" in run["error"]
    assert any(e["level"] == "error" for e in db.run_events(run["id"]))


def test_mid_run_network_failure_is_partial_and_keeps_earlier_pages(db, settings):
    src = FakeSource(reviews_newest_first(20), page_size=5, fail_on_page=3, fail_with=HttpError("boom", 503))
    settings.collector.initial_backfill_limit = 100
    res = collect(settings, db, source=src)
    assert res.status == "partial" and res.stop_reason == "source_error"
    assert db.review_counts(1)["total"] == 10  # pages 1-2 were committed


def test_failure_before_any_page_is_failed(db, settings):
    src = FakeSource(reviews_newest_first(5), fail_on_page=1, fail_with=HttpError("down"))
    assert collect(settings, db, source=src).status == "failed"


def test_snapshot_is_recorded_and_unusable_reviews_are_kept_but_counted(db, settings):
    revs = reviews_newest_first(4)
    revs.append(make_review(99, body="ok"))
    res = collect(settings, db, source=FakeSource(revs, page_size=10))
    assert res.inserts.unusable == 1 and res.inserts.usable == 4
    snap = db.latest_snapshot(1)
    assert snap["total_text_reviews"] == 100


def test_every_run_is_recorded_with_events_and_stats(db, settings):
    res = collect(settings, db, source=FakeSource(reviews_newest_first(5)), trigger="scheduled")
    run = db.recent_runs(1)[0]
    assert (run["kind"], run["trigger"], run["status"]) == ("collect", "scheduled", "success")
    steps = [e["step"] for e in db.run_events(res.run_id)]
    assert steps[0] == "plan" and "fetch_page" in steps and steps[-1] == "done"


def test_content_duplicates_are_flagged_during_collection(db, settings):
    body = "The vault export feature has been broken for weeks and support has not answered my tickets at all"
    revs = [make_review(2, body=body), make_review(1, body=body)]
    res = collect(settings, db, source=FakeSource(revs, page_size=10))
    assert res.inserts.duplicate_content == 1 and res.inserts.usable == 1


# ----------------------------------------------------------------------------- analyze


def seeded(db, settings, n=6):
    collect(settings, db, source=FakeSource(reviews_newest_first(n), page_size=10))
    return FakeLLM()


def test_analyze_processes_each_pending_review_once_and_verifies_quotes(db, settings):
    llm = seeded(db, settings)
    res = analyze(settings, db, llm)
    assert (res.status, res.attempted, res.succeeded, res.failed) == ("success", 6, 6, 0)
    assert res.quote_verified == 6
    assert len(llm.calls) == 6


def test_second_analyze_run_does_no_work_for_already_analysed_reviews(db, settings):
    llm = seeded(db, settings)
    analyze(settings, db, llm)
    calls_before = len(llm.calls)
    res = analyze(settings, db, llm)
    assert res.attempted == 0 and len(llm.calls) == calls_before


def test_only_new_reviews_are_analysed_after_new_collection(db, settings):
    llm = seeded(db, settings, n=4)
    analyze(settings, db, llm)
    collect(settings, db, source=FakeSource(reviews_newest_first(6), page_size=10))
    before = len(llm.calls)
    res = analyze(settings, db, llm)
    assert res.attempted == 2 and len(llm.calls) - before == 2


def test_limit_caps_work_and_leaves_the_rest_pending(db, settings):
    llm = seeded(db, settings)
    res = analyze(settings, db, llm, limit=2)
    assert res.attempted == 2
    assert len(db.pending_for_analysis(1, llm.model, PROMPT_VERSION)) == 4


def test_changing_the_model_triggers_reanalysis(db, settings):
    analyze(settings, db, seeded(db, settings))
    other = FakeLLM(model="another-model:7b")
    assert analyze(settings, db, other).attempted == 6


def test_unverifiable_quotes_are_stored_as_unverified(db, settings):
    def liar(user: str) -> ReviewAnalysisOut:
        return default_responder(user).model_copy(update={"supporting_quote": "words that were never written"})

    llm = FakeLLM(liar)
    collect(settings, db, source=FakeSource(reviews_newest_first(3)))
    res = analyze(settings, db, llm)
    assert res.succeeded == 3 and res.quote_verified == 0
    assert db.conn.execute("SELECT COUNT(*) FROM review_analysis WHERE quote IS NOT NULL").fetchone()[0] == 0


def test_output_errors_are_recorded_per_review_and_do_not_stop_the_run(db, settings):
    calls = {"n": 0}

    def flaky(user: str):
        calls["n"] += 1
        if calls["n"] == 2:
            raise LLMOutputError("model produced garbage")
        return default_responder(user)

    collect(settings, db, source=FakeSource(reviews_newest_first(4)))
    res = analyze(settings, db, FakeLLM(flaky))
    assert (res.status, res.succeeded, res.failed) == ("partial", 3, 1)
    assert db.analysis_counts(1, "fake-model:1b", PROMPT_VERSION) == {"ok": 3, "failed": 1}


def test_failed_reviews_are_not_retried_unless_asked(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(2)))
    analyze(settings, db, FakeLLM(lambda u: (_ for _ in ()).throw(LLMOutputError("bad"))))
    llm = FakeLLM()
    assert analyze(settings, db, llm).attempted == 0
    res = analyze(settings, db, llm, retry_failed=True)
    assert res.succeeded == 2


def test_agent_aborts_after_too_many_consecutive_failures(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(15), page_size=15))
    llm = FakeLLM(lambda u: (_ for _ in ()).throw(LLMOutputError("bad")))
    res = analyze(settings, db, llm)
    assert res.status == "partial" and res.stop_reason == "too_many_consecutive_failures"
    assert res.failed == MAX_CONSECUTIVE_ANALYSIS_FAILURES
    assert len(db.pending_for_analysis(1, llm.model, PROMPT_VERSION)) == 15 - MAX_CONSECUTIVE_ANALYSIS_FAILURES


def test_llm_not_running_fails_fast_without_touching_reviews(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(3)))
    llm = FakeLLM()
    llm.ready_error = LLMUnavailable("Cannot reach Ollama")
    res = analyze(settings, db, llm)
    assert res.status == "failed" and res.stop_reason == "llm_not_ready" and llm.calls == []
    assert "Cannot reach Ollama" in db.recent_runs(1)[0]["error"]


def test_missing_model_is_reported_with_the_fix(db, settings):
    collect(settings, db, source=FakeSource(reviews_newest_first(3)))
    llm = FakeLLM()
    llm.ready_error = LLMModelMissing("Model 'x' is not installed. Run: ollama pull x")
    res = analyze(settings, db, llm)
    assert res.status == "failed" and "ollama pull x" in res.error


def test_ollama_dying_mid_run_keeps_progress_and_stops(db, settings):
    calls = {"n": 0}

    def dies(user: str):
        calls["n"] += 1
        if calls["n"] == 3:
            raise LLMUnavailable("connection lost")
        return default_responder(user)

    collect(settings, db, source=FakeSource(reviews_newest_first(6)))
    res = analyze(settings, db, FakeLLM(dies))
    assert res.status == "partial" and res.stop_reason == "llm_unavailable" and res.succeeded == 2


def test_progress_callback_receives_each_result(db, settings):
    seen = []
    collect(settings, db, source=FakeSource(reviews_newest_first(3)))
    analyze(settings, db, FakeLLM(), progress=lambda i, n, info: seen.append((i, n, info["theme"])))
    assert seen == [(1, 3, "bugs_reliability"), (2, 3, "bugs_reliability"), (3, 3, "bugs_reliability")]


def test_throughput_metrics_are_reported(db, settings):
    res = analyze(settings, db, seeded(db, settings))
    assert res.mean_latency_s == 0.5
    assert res.effective_tokens_per_s == 100.0  # 50 tokens / 0.5 s from the fake meta


# ----------------------------------------------------------------------------- reclean


def test_reclean_applies_new_rules_without_refetching_and_keeps_analyses(db, settings):
    from pda.pipeline import reclean

    revs = [make_review(3, body="Great app"), make_review(2), make_review(1)]  # "Great app" = 2 words
    collect(settings, db, source=FakeSource(revs, page_size=10))
    assert db.review_counts(1)["usable"] == 2
    llm = FakeLLM()
    analyze(settings, db, llm)
    calls_before = len(llm.calls)

    settings.cleaning.min_words = 2  # a rule change: two-word reviews are now acceptable
    res = reclean(settings, db)
    assert (res.reviews, res.changed, res.became_usable, res.became_unusable) == (3, 1, 1, 0)
    assert db.review_counts(1)["usable"] == 3
    assert len(db.pending_for_analysis(1, llm.model, PROMPT_VERSION)) == 1  # only the newly usable one
    assert len(llm.calls) == calls_before  # reclean never calls the model
    assert db.analysis_counts(1, llm.model, PROMPT_VERSION) == {"ok": 2}  # earlier analyses preserved


def test_reclean_is_idempotent_and_redetects_duplicates_in_date_order(db, settings):
    from pda.pipeline import reclean

    body = "The vault export feature has been broken for weeks and support has not answered my tickets at all"
    collect(settings, db, source=FakeSource([make_review(2, body=body), make_review(1, body=body)], page_size=10))
    first = reclean(settings, db)
    second = reclean(settings, db)
    assert first.changed == 0 and second.changed == 0
    dup = db.conn.execute("SELECT source_review_id FROM reviews WHERE unusable_reason='duplicate_content'").fetchone()
    assert dup["source_review_id"] == "2"  # the later one is the copy


def test_analysis_run_counts_and_persists_overrides(db, settings):
    def praise_with_invented_problem(user: str) -> ReviewAnalysisOut:
        return default_responder(user).model_copy(update={"reports_problem": False, "sentiment": "positive"})

    collect(settings, db, source=FakeSource(reviews_newest_first(3)))
    llm = FakeLLM(praise_with_invented_problem)
    res = analyze(settings, db, llm)
    assert res.succeeded == 3 and res.overridden == 3
    rows = db.conn.execute("SELECT theme, severity, overrides FROM review_analysis").fetchall()
    assert all((r["theme"], r["severity"]) == ("no_issue", "none") and r["overrides"] for r in rows)
    df = db.analysed_df(1, llm.model, PROMPT_VERSION)
    assert set(df["theme"]) == {"no_issue"}  # downstream aggregation sees the corrected values

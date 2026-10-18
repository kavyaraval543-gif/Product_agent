import pytest

from helpers import make_review
from pda.cleaning import clean_review
from pda.config import CleaningConfig
from pda.db import Database
from pda.models import ReviewAnalysisOut

CFG = CleaningConfig()
LONG = "The autofill feature stopped working on most of the websites that I use every single day after the update"


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def app_id(db):
    return db.upsert_app("fake", "demo", "Demo App")


def cleaned(*reviews):
    return [clean_review(r, CFG) for r in reviews]


def out(**kw) -> ReviewAnalysisOut:
    base = dict(relevance="genuine_feedback", sentiment="negative", reports_problem=True, theme="bugs_reliability", severity="medium",
                issue_label="autofill broken", problem_statement="x", feature_request="", supporting_quote="Autofill")
    return ReviewAnalysisOut(**{**base, **kw})


def test_upsert_app_is_idempotent_and_updates_name(db):
    a = db.upsert_app("amo", "x", "Old")
    b = db.upsert_app("amo", "x", "New")
    assert a == b
    assert db.conn.execute("SELECT name FROM apps WHERE id=?", (a,)).fetchone()[0] == "New"


def test_insert_is_idempotent_and_counts_known_reviews(db, app_id):
    batch = cleaned(make_review(1), make_review(2))
    first = db.insert_reviews(app_id, None, batch)
    second = db.insert_reviews(app_id, None, batch)
    assert (first.inserted, first.already_known) == (2, 0)
    assert (second.inserted, second.already_known) == (0, 2)
    assert db.review_counts(app_id)["total"] == 2


def test_existing_source_ids(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1), make_review(2)))
    assert db.existing_source_ids("fake", ["1", "2", "3"]) == {"1", "2"}
    assert db.existing_source_ids("other-source", ["1"]) == set()
    assert db.existing_source_ids("fake", []) == set()


def test_long_identical_text_with_different_ids_is_flagged_duplicate_of_the_earliest(db, app_id):
    original, copy = make_review(1, body=LONG), make_review(2, body=LONG.upper())
    stats = db.insert_reviews(app_id, None, cleaned(copy, original))  # given out of order on purpose
    assert stats.duplicate_content == 1 and stats.usable == 1
    rows = {r["source_review_id"]: r for r in db.conn.execute("SELECT * FROM reviews")}
    assert rows["1"]["usable"] == 1 and rows["1"]["duplicate_of"] is None
    assert rows["2"]["usable"] == 0 and rows["2"]["unusable_reason"] == "duplicate_content"
    assert rows["2"]["duplicate_of"] == rows["1"]["id"]


def test_duplicates_are_detected_across_separate_runs(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1, body=LONG)))
    stats = db.insert_reviews(app_id, None, cleaned(make_review(9, body=LONG)))
    assert stats.duplicate_content == 1


def test_short_identical_reviews_are_not_flagged(db, app_id):
    body = "Does not work for me at all"
    stats = db.insert_reviews(app_id, None, cleaned(make_review(1, body=body), make_review(2, body=body)))
    assert stats.duplicate_content == 0 and stats.usable == 2


def test_unusable_reviews_are_stored_with_reason_and_counted(db, app_id):
    stats = db.insert_reviews(app_id, None, cleaned(make_review(1, body="ok fine"), make_review(2)))
    assert stats.unusable == 1 and stats.unusable_reasons == {"too_short": 1}
    c = db.review_counts(app_id)
    assert c["total"] == 2 and c["usable"] == 1 and c["unusable_reasons"] == {"too_short": 1}


def test_reviewer_identity_is_not_a_column(db):
    cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(reviews)")}
    assert not ({"user", "username", "author", "reviewer", "user_id", "name"} & cols)


def test_pending_analysis_is_incremental_and_keyed_by_model_and_prompt(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1), make_review(2), make_review(3, body="meh")))
    pending = db.pending_for_analysis(app_id, "m1", "v1")
    assert [r["source_review_id"] for r in pending] == ["2", "1"]  # newest first, unusable excluded
    db.save_analysis(pending[0]["id"], "m1", "v1", out(), True, 1.0, 40)
    assert [r["source_review_id"] for r in db.pending_for_analysis(app_id, "m1", "v1")] == ["1"]
    assert len(db.pending_for_analysis(app_id, "m2", "v1")) == 2  # new model -> re-analyse
    assert len(db.pending_for_analysis(app_id, "m1", "v2")) == 2  # new prompt -> re-analyse
    assert len(db.pending_for_analysis(app_id, "m1", "v1", limit=1)) == 1


def test_unverified_quote_is_not_stored_as_evidence(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1)))
    rid = db.pending_for_analysis(app_id, "m", "v")[0]["id"]
    db.save_analysis(rid, "m", "v", out(supporting_quote="invented words"), False, 1.0, 10)
    row = db.conn.execute("SELECT quote, quote_verified, raw_json FROM review_analysis").fetchone()
    assert row["quote"] is None and row["quote_verified"] == 0
    assert "invented words" in row["raw_json"]  # kept for audit, but never shown as evidence


def test_failed_analyses_are_not_retried_automatically_but_can_be_cleared(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1)))
    rid = db.pending_for_analysis(app_id, "m", "v")[0]["id"]
    db.save_analysis_failure(rid, "m", "v", "bad json")
    assert db.pending_for_analysis(app_id, "m", "v") == []
    assert db.analysis_counts(app_id, "m", "v") == {"failed": 1}
    assert db.delete_failed_analyses(app_id, "m", "v") == 1
    assert len(db.pending_for_analysis(app_id, "m", "v")) == 1


def test_analysed_df_contains_measured_and_ai_fields_for_ok_rows_only(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1, rating=1), make_review(2, rating=5)))
    rows = db.pending_for_analysis(app_id, "m", "v")
    db.save_analysis(rows[0]["id"], "m", "v", out(), True, 1.0, 10)
    db.save_analysis_failure(rows[1]["id"], "m", "v", "x")
    df = db.analysed_df(app_id, "m", "v")
    assert len(df) == 1
    assert {"rating", "url", "review_date", "theme", "severity", "quote_verified"} <= set(df.columns)


def test_run_lifecycle_and_events(db, app_id):
    run = db.start_run(app_id, "collect", "scheduled")
    db.log_event(run, "fetch_page", "page 1", data={"n": 5})
    db.finish_run(run, "success", {"inserted": 5})
    r = db.recent_runs(1)[0]
    assert (r["kind"], r["trigger"], r["status"]) == ("collect", "scheduled", "success")
    assert r["finished_at"] is not None
    assert db.run_events(run)[0]["message"] == "page 1"
    assert db.last_successful_run("collect")["id"] == run
    assert db.last_successful_run("analyze") is None


def test_snapshot_roundtrip(db, app_id):
    db.save_snapshot(app_id, None, 3144, {"1": 5, "5": 95})
    snap = db.latest_snapshot(app_id)
    assert snap["total_text_reviews"] == 3144 and '"5": 95' in snap["grouped_ratings_json"]


def test_schema_version_is_recorded(db):
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_pending_can_be_restricted_to_specific_reviews(db, app_id):
    db.insert_reviews(app_id, None, cleaned(make_review(1), make_review(2), make_review(3)))
    got = db.pending_for_analysis(app_id, "m", "v", only_source_ids={"1", "3"})
    assert sorted(r["source_review_id"] for r in got) == ["1", "3"]
    assert db.pending_for_analysis(app_id, "m", "v", only_source_ids=set()) == []


def test_existing_v2_database_migrates_in_place_without_losing_data(tmp_path):
    import sqlite3

    from pda.db import SCHEMA_V1, SCHEMA_V2

    path = tmp_path / "old.sqlite3"
    raw = sqlite3.connect(path)
    raw.executescript(SCHEMA_V1 + SCHEMA_V2)
    raw.execute("PRAGMA user_version = 2")
    raw.execute("INSERT INTO apps (source, slug, name, created_at) VALUES ('amo','x','X','2026-01-01')")
    raw.execute("INSERT INTO reviews (app_id, source, source_review_id, body_clean, review_date, url, collected_at, content_hash,"
                " word_count, usable) VALUES (1,'amo','7','kept text','2026-01-01','u','2026-01-02','h',2,1)")
    raw.execute("INSERT INTO review_analysis (review_id, model, prompt_version, analyzed_at, status, theme) VALUES (1,'m','v2','t','ok','bugs_reliability')")
    raw.commit(); raw.close()

    db = Database(path)  # opening applies migration 3
    assert db.conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert db.conn.execute("SELECT body_clean FROM reviews").fetchone()[0] == "kept text"
    row = db.conn.execute("SELECT theme, overrides FROM review_analysis").fetchone()
    assert row["theme"] == "bugs_reliability" and row["overrides"] is None  # old rows are untouched, new column is empty


def test_overrides_and_the_models_raw_answer_are_both_stored(db, app_id):
    raw = out(reports_problem=False, theme="bugs_reliability", severity="high", issue_label="crashes")
    final, fixes = raw.normalised()
    db.insert_reviews(app_id, None, cleaned(make_review(1)))
    rid = db.pending_for_analysis(app_id, "m", "v")[0]["id"]
    db.save_analysis(rid, "m", "v", final, True, 1.0, 10, raw=raw, overrides=fixes)
    row = db.conn.execute("SELECT theme, severity, issue_label, overrides, raw_json FROM review_analysis").fetchone()
    assert (row["theme"], row["severity"], row["issue_label"]) == ("no_issue", "none", "")  # what downstream code sees
    assert '"theme":"bugs_reliability"' in row["raw_json"] and '"issue_label":"crashes"' in row["raw_json"]  # what the model said
    assert "no_problem:theme" in row["overrides"] and "no_problem:issue_label" in row["overrides"]

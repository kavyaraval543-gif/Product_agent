import json

import pytest

from helpers import make_review
from pda.cleaning import clean_review
from pda.config import CleaningConfig
from pda.db import Database
from pda.evaluation import GoldItem, evaluate, load_gold, render_markdown, wilson_interval
from pda.models import ReviewAnalysisOut

import hashlib


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def out(theme, relevance="genuine_feedback", sentiment="negative", feature="", quote="x") -> ReviewAnalysisOut:
    return ReviewAnalysisOut(relevance=relevance, sentiment=sentiment, reports_problem=theme != "no_issue", problem_statement="p", issue_label="l", theme=theme,
                             severity="medium", feature_request=feature, supporting_quote=quote)


@pytest.fixture
def setup():
    db = Database(":memory:")
    app = db.upsert_app("fake", "demo", "Demo")
    bodies = {i: f"This is review number {i} about something that is not working properly for me" for i in range(1, 7)}
    db.insert_reviews(app, None, [clean_review(make_review(i, body=b), CleaningConfig()) for i, b in bodies.items()])
    ids = {r["source_review_id"]: r["id"] for r in db.conn.execute("SELECT id, source_review_id FROM reviews")}
    return db, app, bodies, ids


def gold(i, bodies, theme, alts=(), rel="genuine_feedback", sent="negative", fr=False, contaminated=False, sha_override=None):
    return GoldItem("fake", str(i), sha_override or sha(bodies[i]), rel, [], sent, theme, list(alts), fr, contaminated)


def analyse(db, ids, i, o):
    db.save_analysis(ids[str(i)], "m", "v", o, True, 1.0, 10)


def test_wilson_interval_properties():
    lo, hi = wilson_interval(20, 40)
    assert lo < 0.5 < hi and 0.34 < lo < 0.37 and 0.63 < hi < 0.66  # known value for 20/40
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
    narrow, wide = wilson_interval(500, 1000), wilson_interval(5, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])  # more data -> tighter


def test_strict_and_lenient_theme_scoring_and_confusion(setup):
    db, app, bodies, ids = setup
    items = [
        gold(1, bodies, "bugs_reliability"),                                  # model: exact match
        gold(2, bodies, "bugs_reliability", alts=["compatibility"]),          # model: acceptable alternative
        gold(3, bodies, "performance"),                                       # model: wrong
        gold(4, bodies, "no_issue", sent="positive"),                         # model: exact match
    ]
    analyse(db, ids, 1, out("bugs_reliability")); analyse(db, ids, 2, out("compatibility"))
    analyse(db, ids, 3, out("usability_ux")); analyse(db, ids, 4, out("no_issue", sentiment="positive"))
    res = evaluate(db, app, "m", "v", {"labeler": "test"}, items)
    by = {mt.name: mt for mt in res.metrics}
    assert (by["theme (strict: matches primary label)"].correct, res.n_scored) == (2, 4)
    assert by["theme (lenient: matches primary or an acceptable alternative)"].correct == 3
    assert res.confusion["performance"] == {"usability_ux": 1}
    assert [d["source_review_id"] for d in res.disagreements] == ["3"]


def test_precision_recall_for_problem_and_feature_detection(setup):
    db, app, bodies, ids = setup
    items = [
        gold(1, bodies, "bugs_reliability", fr=True),   # real problem, real request
        gold(2, bodies, "bugs_reliability"),            # real problem, model says no issue  -> FN
        gold(3, bodies, "no_issue", sent="positive"),   # no problem, model invents one      -> FP
        gold(4, bodies, "no_issue", sent="positive"),   # no problem, correct
    ]
    analyse(db, ids, 1, out("bugs_reliability", feature="add export"))
    analyse(db, ids, 2, out("no_issue"))
    analyse(db, ids, 3, out("performance", feature="make it faster"))  # request the labeller did not see -> FP
    analyse(db, ids, 4, out("no_issue"))
    pr = evaluate(db, app, "m", "v", {}, items).precision_recall
    assert (pr["has_problem"]["tp"], pr["has_problem"]["fp"], pr["has_problem"]["fn"]) == (1, 1, 1)
    assert pr["has_problem"]["precision"] == 0.5 and pr["has_problem"]["recall"] == 0.5
    assert (pr["feature_request"]["tp"], pr["feature_request"]["fp"], pr["feature_request"]["fn"]) == (1, 1, 0)


def test_items_that_cannot_be_scored_are_excluded_and_explained(setup):
    db, app, bodies, ids = setup
    items = [
        gold(1, bodies, "bugs_reliability"),                                   # scored
        gold(2, bodies, "bugs_reliability", contaminated=True),                # labeller saw the answer
        gold(3, bodies, "bugs_reliability", sha_override="0" * 16),            # review text changed
        gold(4, bodies, "bugs_reliability"),                                   # not analysed
        GoldItem("fake", "99999", "0" * 16, "genuine_feedback", [], "negative", "other", [], False),  # unknown review
    ]
    analyse(db, ids, 1, out("bugs_reliability")); analyse(db, ids, 2, out("bugs_reliability")); analyse(db, ids, 3, out("bugs_reliability"))
    res = evaluate(db, app, "m", "v", {}, items)
    assert res.n_scored == 1 and res.n_gold == 5
    assert sum(res.skipped.values()) == 4 and len(res.skipped) == 4


def test_scores_use_only_the_requested_model_and_prompt(setup):
    db, app, bodies, ids = setup
    db.save_analysis(ids["1"], "other-model", "v", out("performance"), True, 1.0, 10)
    analyse(db, ids, 1, out("bugs_reliability"))
    res = evaluate(db, app, "m", "v", {}, [gold(1, bodies, "bugs_reliability")])
    assert res.metrics[3].correct == 1


def test_gold_file_roundtrip_and_report_rendering(setup, tmp_path):
    db, app, bodies, ids = setup
    f = tmp_path / "gold.jsonl"
    rows = [{"_meta": {"labeler": "unit test"}},
            {"source": "fake", "source_review_id": "1", "text_sha256": sha(bodies[1]), "relevance": "genuine_feedback",
             "acceptable_relevance": [], "sentiment": "negative", "theme": "bugs_reliability", "acceptable_themes": [],
             "feature_request": False}]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    meta, items = load_gold(f)
    assert meta["labeler"] == "unit test" and len(items) == 1 and not items[0].contaminated
    analyse(db, ids, 1, out("performance"))
    md = render_markdown(evaluate(db, app, "m", "v", meta, items))
    assert "unit test" in md and "95% CI" in md and "Disagreements (1)" in md and "does **not** prove" in md
    assert "text:" in md and "text:" not in render_markdown(evaluate(db, app, "m", "v", meta, items), include_text=False)


def test_no_details_hides_per_item_errors_for_the_held_out_set(setup):
    db, app, bodies, ids = setup
    analyse(db, ids, 1, out("performance"))
    res = evaluate(db, app, "m", "v", {}, [gold(1, bodies, "bugs_reliability")])
    full, hidden = render_markdown(res), render_markdown(res, details=False)
    assert "Disagreements (1)" in full and "performance" in full
    assert "Disagreements" not in hidden and "Theme confusion" not in hidden and "held out" in hidden
    assert "95% CI" in hidden  # the scores themselves are still reported

import pandas as pd
import pytest

from pda.themes import group_labels, overview, summarise_issues, summarise_themes


def row(i, *, theme="bugs_reliability", label="autofill fails on login pages", rating=2, severity="medium", sentiment="negative",
        relevance="genuine_feedback", quote=None, verified=0, feature="", date="2026-10-01"):
    return {
        "review_id": i, "source_review_id": str(1000 + i), "rating": rating, "body_clean": f"full text of review {i} " * 6,
        "review_date": pd.Timestamp(date), "url": f"https://example.test/r/{i}", "app_version": "1.0", "relevance": relevance,
        "sentiment": sentiment, "theme": theme, "severity": severity, "issue_label": label, "problem_statement": "",
        "feature_request": feature, "quote": quote, "quote_verified": verified,
    }


def frame(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def test_wording_variants_cluster_and_distinct_problems_stay_apart():
    labels = [
        "autofill fails on login pages", "autofill failing on login page",   # 0, 1: same problem, different wording
        "slow startup time", "startup is slow",                              # 2, 3
        "cannot export vault", "vault export missing",                       # 4, 5
        "login fails on mobile app",                                         # 6: shares words with 0/1 but is a different problem
    ]
    as_sets = sorted(sorted(c) for c in group_labels(labels))
    assert as_sets == [[0, 1], [2, 3], [4, 5], [6]]


@pytest.mark.xfail(strict=True, reason="Known limitation: lexical grouping cannot match synonyms ('fails' vs 'not working'). "
                                       "A semantic embedding step is the planned upgrade; see docs/FEASIBILITY.md.")
def test_synonym_labels_are_not_merged_yet():
    assert sorted(sorted(c) for c in group_labels(["autofill fails on login pages", "autofill not working on login page"])) == [[0, 1]]


def test_grouping_is_deterministic():
    labels = ["autofill fails", "autofill broken", "slow sync", "sync is slow", "crash on launch"]
    assert group_labels(labels) == group_labels(list(labels))


def test_grouping_edge_cases():
    assert group_labels([]) == []
    assert group_labels(["a"]) == [[0]]
    assert group_labels(["the", "of"]) == [[0], [1]]  # nothing but stopwords: no vocabulary, no merging


def test_theme_counts_shares_and_ratings_are_exact():
    df = frame(
        row(1, rating=1), row(2, rating=3), row(3, rating=5),
        row(4, theme="performance", label="slow startup", rating=2),
        row(5, theme="no_issue", label="", rating=5, sentiment="positive", severity="none"),  # praise: not a problem
        row(6, relevance="irrelevant", rating=1),  # excluded entirely
        row(7, relevance="ambiguous", rating=1),   # kept, flagged low-confidence
    )
    themes = {t.theme: t for t in summarise_themes(df)}
    assert set(themes) == {"bugs_reliability", "performance"}
    bug = themes["bugs_reliability"]
    assert bug.review_count == 4 and bug.low_confidence_count == 1
    assert bug.avg_rating == 2.5  # (1 + 3 + 5 + 1) / 4
    relevant = 6  # reviews 1-5 and 7; review 6 is irrelevant
    assert bug.share_of_relevant == round(4 / relevant, 3)
    assert themes["performance"].share_of_relevant == round(1 / relevant, 3)
    assert [t.theme for t in summarise_themes(df)][0] == "bugs_reliability"  # sorted by count


def test_severity_and_feature_request_counts():
    df = frame(
        row(1, severity="high", feature="add export"),
        row(2, severity="high"), row(3, severity="low", feature="  "),
    )
    (t,) = summarise_themes(df)
    assert t.severity_counts == {"high": 2, "low": 1}
    assert t.feature_request_count == 1


def test_issue_group_counts_and_most_common_label_wins():
    df = frame(
        row(1, label="autofill fails on login pages"), row(2, label="autofill fails on login pages"),
        row(3, label="autofill fails on login page"), row(4, label="vault export missing", theme="bugs_reliability"),
    )
    groups = summarise_issues(df)
    top = groups[0]
    assert top.review_count == 3 and top.label == "autofill fails on login pages"
    assert len(groups) == 2
    assert top.share_of_relevant == round(3 / 4, 3)


def test_groups_span_themes_and_report_how_the_model_filed_them():
    # The SAME problem filed under two themes (what the real model did for autofill) must be ONE group.
    df = frame(
        row(1, theme="usability_ux", label="autofill fails on login pages"),
        row(2, theme="usability_ux", label="autofill fails on login pages"),
        row(3, theme="bugs_reliability", label="autofill fails on login page"),
        row(4, theme="performance", label="slow startup time"),
    )
    groups = summarise_issues(df)
    auto = next(g for g in groups if g.review_count == 3)
    assert auto.theme_mix == {"usability_ux": 2, "bugs_reliability": 1}
    assert {g.label for g in groups} == {"autofill fails on login pages", "slow startup time"}


def test_reviews_without_any_description_are_in_theme_counts_but_not_in_groups():
    df = frame(row(1, label=""), row(2, label="autofill fails"))  # review 1 has neither a label nor a problem statement
    (t,) = summarise_themes(df)
    assert t.review_count == 2
    assert sum(g.review_count for g in summarise_issues(df)) == 1


def test_evidence_prefers_verified_quotes_then_severity_then_recency_and_links_to_real_reviews():
    df = frame(
        row(1, quote="the exact words", verified=1, severity="low", date="2026-10-01"),
        row(2, quote=None, verified=0, severity="critical", date="2026-10-09"),
        row(3, quote="newer exact words", verified=1, severity="low", date="2026-10-05"),
        row(4, quote="most severe exact words", verified=1, severity="high", date="2026-09-01"),
    )
    (g,) = summarise_issues(df)
    ev = g.evidence
    assert [e.review_id for e in ev] == [4, 3, 1]  # verified first; then severity; then recency; k=3
    assert all(e.url == f"https://example.test/r/{e.review_id}" for e in ev)
    assert all(e.is_verbatim_quote for e in ev)


def test_unverified_evidence_falls_back_to_a_plain_excerpt_of_the_real_review():
    df = frame(row(1, quote="invented", verified=0))
    (g,) = summarise_issues(df)
    e = g.evidence[0]
    assert not e.is_verbatim_quote and e.text != "invented"
    assert e.text.startswith("full text of review 1") and len(e.text) <= 221


def test_overview_numbers():
    df = frame(
        row(1, rating=1, verified=1), row(2, rating=5, theme="no_issue", sentiment="positive", verified=1),
        row(3, relevance="ambiguous", rating=3), row(4, relevance="irrelevant", rating=1),
    )
    o = overview(df)
    assert (o.analysed, o.genuine, o.ambiguous, o.irrelevant, o.relevant, o.problem_reviews) == (4, 2, 1, 1, 3, 2)
    assert o.avg_rating_genuine == 3.0  # (1 + 5 + 3) / 3: relevant = genuine + ambiguous
    assert o.quote_verified_rate == 0.5
    assert o.sentiment_counts == {"negative": 2, "positive": 1}  # relevant reviews only: 1, 2 and the ambiguous 3


def test_empty_inputs_are_safe():
    empty = pd.DataFrame(columns=row(1).keys())
    assert summarise_themes(empty) == [] and summarise_issues(empty) == []
    assert overview(empty).analysed == 0
    only_praise = frame(row(1, theme="no_issue", label=""))
    assert summarise_themes(only_praise) == [] and summarise_issues(only_praise) == []


def test_label_helpers():
    from pda.themes import display_label, informative_label

    assert informative_label("Bugs_Reliability ") == "" and informative_label("no issue") == "" and informative_label("performance") == ""
    assert informative_label("Autofill  fails.") == "autofill fails"
    assert display_label("", "Passkey popup never appears on macOS.") == "Passkey popup never appears on macOS"
    long = "word " * 40
    shown = display_label("no issue", long)
    assert len(shown) <= 70 and shown.endswith("\u2026") and display_label(None, None) == ""
    assert display_label("", "x" * 100).endswith("\u2026")  # no spaces to cut at: still bounded and marked

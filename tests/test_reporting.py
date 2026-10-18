import pandas as pd

import re

from pda.reporting import _MD_SPECIAL, md_escape, render_markdown, star_stats
from pda.themes import Evidence, IssueGroup, Overview, ThemeSummary, overview, summarise_issues, summarise_themes

HOSTILE = "![x](http://tracker.example/p.png) [click](javascript:alert(1)) <img src=x onerror=alert(1)> **bold** `code`"


def active_specials(text: str) -> set[str]:
    """Special characters left ACTIVE after removing every backslash-escaped pair."""
    return set(re.sub(r"\\.", "", text)) & _MD_SPECIAL


def test_md_escape_leaves_no_active_markdown_or_html_control_characters():
    out = md_escape(HOSTILE)
    assert active_specials(out) == set()
    assert "tracker.example/p.png" in out  # content is preserved, only neutralised


def test_md_escape_flattens_newlines_so_text_cannot_start_new_blocks():
    assert "\n" not in md_escape("line one\n\n# heading\n- item")


def _issue_with(label: str, quote: str, mix=None) -> IssueGroup:
    ev = Evidence(1, "100", "https://example.test/r/1", "2026-10-01", 1, quote, True)
    return IssueGroup(label=label, review_count=1, share_of_relevant=1.0, avg_rating=1.0, mean_severity=3.0,
                      theme_mix=mix or {"bugs_reliability": 1}, low_confidence_count=0, member_labels=[label], evidence=[ev])


def _render(issue: IssueGroup) -> str:
    ov = Overview(1, 1, 0, 0, 1, 1, {"negative": 1}, 1.0, 1.0, "2026-10-01", "2026-10-01")
    theme = ThemeSummary(theme="bugs_reliability", review_count=1, share_of_relevant=1.0, low_confidence_count=0, avg_rating=1.0,
                         severity_counts={"high": 1}, feature_request_count=0)
    return render_markdown("Demo", ov, [theme], [issue], model="m", prompt_version="v2", snapshot_total_text=10,
                           snapshot_grouped={"1": 1, "5": 9}, collected_total=12, collected_usable=10)


def test_report_never_emits_live_markup_from_reviews_or_model_labels():
    md = _render(_issue_with(label=HOSTILE, quote=HOSTILE))
    label_line = next(ln for ln in md.splitlines() if ln.startswith("### 1. ") and "tracker.example" in ln)
    quote_line = next(ln for ln in md.splitlines() if ln.startswith("> \"") and "tracker.example" in ln)
    inner_label = label_line[len("### 1. "):]  # strip OUR OWN heading prefix, keep the model-written label
    inner_quote = quote_line[3:].rstrip().rstrip("\"").rstrip()  # strip OUR OWN blockquote marker and quotes
    assert active_specials(inner_label) == set()
    assert active_specials(inner_quote) == set()
    assert "[source](https://example.test/r/1)" in md  # our own link is intact


def test_report_separates_measured_from_ai_and_states_caveats():
    md = _render(_issue_with("autofill fails", "it broke", mix={"usability_ux": 2, "bugs_reliability": 1}))
    assert "[measured]" in md and "[AI]" in md
    assert "not a representative sample" in md and "not** a measure" in md
    assert "https://example.test/r/1" in md  # evidence links to the real review
    assert "Source-wide rating (all 10 ratings" in md and "4.60" in md  # (1*1 + 9*5) / 10
    assert "filed by the model under: Bugs & reliability 1" in md or "Usability & UX 2, Bugs & reliability 1" in md
    assert "Treat theme shares as indicative only" in md


def test_star_stats_is_plain_arithmetic():
    assert star_stats({"1": 2, "5": 2}) == (4, 3.0)
    assert star_stats({}) is None and star_stats(None) is None and star_stats({"1": 0}) is None


def test_report_from_real_shaped_dataframe_roundtrip():
    df = pd.DataFrame([{
        "review_id": 1, "source_review_id": "9", "rating": 1, "body_clean": "autofill is broken", "review_date": pd.Timestamp("2026-10-01"),
        "url": "https://example.test/r/9", "app_version": "1", "relevance": "genuine_feedback", "sentiment": "negative",
        "theme": "bugs_reliability", "severity": "high", "issue_label": "autofill broken", "problem_statement": "",
        "feature_request": "", "quote": "autofill is broken", "quote_verified": 1,
    }])
    md = render_markdown("Demo", overview(df), summarise_themes(df), summarise_issues(df), model="m", prompt_version="v2", snapshot_total_text=None,
                         snapshot_grouped=None, collected_total=1, collected_usable=1)
    assert "autofill broken" in md and "100%" in md

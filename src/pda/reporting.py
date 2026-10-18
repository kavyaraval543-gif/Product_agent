"""Markdown report. Measured facts and AI interpretations are labelled separately on purpose."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from pda.themes import IssueGroup, Overview, ThemeSummary

THEME_TITLES = {
    "bugs_reliability": "Bugs & reliability",
    "performance": "Performance",
    "usability_ux": "Usability & UX",
    "missing_feature": "Missing features",
    "compatibility": "Compatibility",
    "account_sync": "Account access & sync",
    "onboarding_setup": "Onboarding & setup",
    "pricing_value": "Pricing & value",
    "privacy_security": "Privacy & security",
    "customer_support": "Customer support",
    "other": "Other",
}

SEVERITY_ORDER = ["critical", "high", "medium", "low", "none"]


_MD_SPECIAL = set("\\`*_{}[]()<>#+!|~")


def md_escape(text: str) -> str:
    """Neutralise Markdown/HTML in UNTRUSTED text (review quotes, model-written labels).

    Without this a reviewer could publish `![](http://tracker/x.png)` or `[click](javascript:...)` and have
    it render as a live image/link in a shared report.
    """
    flat = " ".join(str(text).split())
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in flat)


def theme_title(theme: str) -> str:
    return THEME_TITLES.get(theme, theme.replace("_", " ").title())


def star_stats(grouped: dict[str, int] | None) -> tuple[int, float] | None:
    """(total ratings, mean stars) across ALL ratings at the source. Deterministic arithmetic."""
    if not grouped:
        return None
    total = sum(grouped.values())
    if not total:
        return None
    return total, sum(int(k) * v for k, v in grouped.items()) / total


def _sev_mix(counts: dict[str, int]) -> str:
    parts = [f"{counts[s]} {s}" for s in SEVERITY_ORDER if counts.get(s)]
    return ", ".join(parts) or "-"


def render_markdown(
    app_name: str,
    overview: Overview,
    themes: list[ThemeSummary],
    issues: list[IssueGroup],
    *,
    model: str,
    prompt_version: str,
    snapshot_total_text: int | None,
    snapshot_grouped: dict[str, int] | None,
    collected_total: int,
    collected_usable: int,
    max_issues: int = 10,
    grouping_note: str = "TF-IDF lexical similarity",
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    L += [f"# Customer problems: {app_name}", "", f"_Generated {now} by Product Discovery Agent._", ""]
    L += [
        "> **How to read this report.** Items marked **[measured]** are counts, ratings, dates and links computed "
        "directly from collected reviews. Items marked **[AI]** are interpretations by a small local language model "
        "(sentiment, severity, theme, issue label) and can be wrong. Reviews come from people who chose to write one, "
        "so they are **not a representative sample of all users**, and review counts are **not** a measure of how many "
        "customers are affected or of business impact.",
        "",
    ]

    L += ["## 1. Data scope  [measured]", ""]
    L += [f"- Reviews collected: **{collected_total}**, of which **{collected_usable}** passed cleaning (usable text)."]
    L += [f"- Reviews analysed by the model: **{overview.analysed}** ({overview.date_min} to {overview.date_max})."]
    if snapshot_total_text is not None:
        L += [f"- Reviews with text reported by the source at last collection: **{snapshot_total_text}**."]
    stats = star_stats(snapshot_grouped)
    if stats:
        L += [f"- Source-wide rating (all {stats[0]} ratings, with or without text): **{stats[1]:.2f}** / 5."]
    if overview.avg_rating_genuine is not None:
        L += [f"- Mean rating of the analysed text reviews: **{overview.avg_rating_genuine:.2f}** / 5 "
              "(text reviewers skew more negative than silent raters, so this is not comparable to the source-wide rating)."]
    L += [""]

    L += ["## 2. Model screening  [AI]", ""]
    L += [
        f"- Genuine feedback: **{overview.genuine}** | ambiguous: **{overview.ambiguous}** | irrelevant (excluded): **{overview.irrelevant}**",
        "- _Ambiguous_ reviews are kept and shown as low-confidence: on real data the model often called short but specific "
        "bug reports ambiguous, and dropping them would have hidden real problems.",
        f"- Relevant reviews reporting a problem or request (theme other than _no issue_): **{overview.problem_reviews}** of {overview.relevant}",
    ]
    if overview.quote_verified_rate is not None:
        L += [f"- Evidence check [measured]: **{overview.quote_verified_rate:.0%}** of the model's quotes were found verbatim "
              "in the review. Unverified quotes are replaced by a plain excerpt of the review."]
    L += [""]

    L += ["## 3. Most-mentioned problems  [grouped across themes]", "",
          "Reviews whose model-written problem descriptions are similar are grouped, regardless of the theme the model "
          "assigned. Counts and ratings are **[measured]**; the group label, severity and theme mix are **[AI]**.", ""]
    if not issues:
        L += ["_No describable problems found._", ""]
    for rank, g in enumerate(issues[:max_issues], start=1):
        rating = f"{g.avg_rating:.2f}" if g.avg_rating is not None else "n/a"
        mix = ", ".join(f"{theme_title(t)} {n}" for t, n in sorted(g.theme_mix.items(), key=lambda kv: (-kv[1], kv[0])))
        low = f" · {g.low_confidence_count} low-confidence" if g.low_confidence_count else ""
        L += [f"### {rank}. {md_escape(g.label)}", "",
              f"**{g.review_count} review(s)** ({g.share_of_relevant:.0%} of relevant) · mean rating {rating} · "
              f"mean severity {g.mean_severity:.1f}/4 [AI] · filed by the model under: {mix}{low}", ""]
        for e in g.evidence:
            kind = "quote" if e.is_verbatim_quote else "excerpt"
            stars = f"{e.rating}★" if e.rating else "unrated"
            L += [f"> \"{md_escape(e.text)}\"  ", f"> — {kind}, {stars}, {e.review_date}, [source]({e.url})", ""]
    if len(issues) > max_issues:
        rest = issues[max_issues:]
        L += [f"_{len(rest)} further groups (covering {sum(g.review_count for g in rest)} reviews, "
              f"{sum(1 for g in rest if g.review_count == 1)} of them single-review) are not shown._", ""]

    L += ["## 4. Themes  [AI, coarse]", "",
          "> **Treat theme shares as indicative only.** On held-out reviews the model's theme agreed with a rater about 6 times in 10 "
          "(strictly), and the same reviews moved from 55% to 21% \"bugs\" when only the prompt changed (see `docs/MVP_RESULTS.md`). "
          "The same problem is often filed under several themes (see the _filed under_ line above).", "",
          "| Theme [AI] | Reviews [measured] | Share of relevant [measured] | Mean rating [measured] | Severity mix [AI] | Feature requests [AI] | Low-confidence [AI] |",
          "|---|---:|---:|---:|---|---:|---:|"]
    for t in themes:
        rating = f"{t.avg_rating:.2f}" if t.avg_rating is not None else "-"
        L += [f"| {theme_title(t.theme)} | {t.review_count} | {t.share_of_relevant:.0%} | {rating} | {_sev_mix(t.severity_counts)} | {t.feature_request_count} | {t.low_confidence_count} |"]
    L += [""]

    L += ["## 5. Method and limitations", "",
          f"- Model: `{model}` via Ollama, prompt `{prompt_version}`. Temperature 0; output constrained to a JSON schema.",
          f"- Issue groups cluster the model's short problem descriptions using: {grouping_note}. "
          "They are heuristic: similar problems can be split, and different ones merged. Group labels are written by the model.",
          "- Only reviews written in English with at least a few words are analysed; others are stored but marked unusable.",
          "- Reviewer names and ids are not stored.",
          "- Accuracy against labelled reviews is measured separately with `pda eval` (labels are provisional and single-rater). Severity has not been validated.",
          ""]
    return "\n".join(L)


def snapshot_to_grouped(row) -> tuple[int | None, dict[str, int] | None]:
    if row is None:
        return None, None
    grouped = json.loads(row["grouped_ratings_json"]) if row["grouped_ratings_json"] else None
    return row["total_text_reviews"], grouped

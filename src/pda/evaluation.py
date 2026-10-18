"""Evaluate the LLM's per-review judgements against labelled examples.

This measures AGREEMENT WITH THE LABELS, nothing more. If the labels were written by one person (or by an
AI) they are that rater's judgement, not ground truth - the report says who labelled and how many.

Gold files hold ids + labels only (never review text), so they can be committed without redistributing
other people's words. A short hash of the cleaned text detects reviews that changed since labelling.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pda.db import Database


@dataclass
class GoldItem:
    source: str
    source_review_id: str
    text_sha256: str
    relevance: str
    acceptable_relevance: list[str]
    sentiment: str
    theme: str
    acceptable_themes: list[str]
    feature_request: bool
    contaminated: bool = False  # labeller had seen the model's answer first -> excluded from scores


@dataclass
class Metric:
    name: str
    correct: int
    total: int

    @property
    def value(self) -> float | None:
        return self.correct / self.total if self.total else None

    @property
    def ci(self) -> tuple[float, float] | None:
        return wilson_interval(self.correct, self.total) if self.total else None

    def fmt(self) -> str:
        if not self.total:
            return "n/a"
        lo, hi = self.ci  # type: ignore[misc]
        return f"{self.correct}/{self.total} = {self.value:.0%} (95% CI {lo:.0%}-{hi:.0%})"


@dataclass
class EvalResult:
    model: str
    prompt_version: str
    gold_meta: dict[str, Any]
    n_gold: int
    n_scored: int
    skipped: dict[str, int]
    metrics: list[Metric]
    precision_recall: dict[str, dict[str, float | None]]
    confusion: dict[str, dict[str, int]]
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    quote_verified: Metric | None = None


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion - honest about how little n=40 tells you."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_gold(path: str | Path) -> tuple[dict[str, Any], list[GoldItem]]:
    meta: dict[str, Any] = {}
    items: list[GoldItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "_meta" in row:
            meta = row["_meta"]
            continue
        items.append(
            GoldItem(
                source=row["source"], source_review_id=str(row["source_review_id"]), text_sha256=row["text_sha256"],
                relevance=row["relevance"], acceptable_relevance=row.get("acceptable_relevance", []),
                sentiment=row["sentiment"], theme=row["theme"], acceptable_themes=row.get("acceptable_themes", []),
                feature_request=bool(row["feature_request"]), contaminated=bool(row.get("seen_model_output_before_labelling")),
            )
        )
    return meta, items


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision is not None and recall is not None else None)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate(db: Database, app_id: int, model: str, prompt_version: str, meta: dict[str, Any], gold: list[GoldItem]) -> EvalResult:
    skipped: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for g in gold:
        if g.contaminated:
            skipped["contaminated (labeller saw model output)"] += 1
            continue
        review = db.conn.execute(
            "SELECT id, body_clean, rating, url FROM reviews WHERE app_id=? AND source=? AND source_review_id=?",
            (app_id, g.source, g.source_review_id),
        ).fetchone()
        if review is None:
            skipped["review not in database"] += 1
            continue
        if hashlib.sha256(review["body_clean"].encode()).hexdigest()[:16] != g.text_sha256:
            skipped["text changed since labelling"] += 1
            continue
        a = db.conn.execute(
            "SELECT * FROM review_analysis WHERE review_id=? AND model=? AND prompt_version=? AND status='ok'",
            (review["id"], model, prompt_version),
        ).fetchone()
        if a is None:
            skipped["not analysed by this model/prompt yet"] += 1
            continue
        rows.append({"gold": g, "pred": a, "review": review})

    def m(name: str, hits: list[bool]) -> Metric:
        return Metric(name, sum(hits), len(hits))

    metrics = [
        m("relevance (strict)", [r["pred"]["relevance"] == r["gold"].relevance for r in rows]),
        m("relevance (lenient: any acceptable)", [r["pred"]["relevance"] in {r["gold"].relevance, *r["gold"].acceptable_relevance} for r in rows]),
        m("sentiment (strict, 4 classes)", [r["pred"]["sentiment"] == r["gold"].sentiment for r in rows]),
        m("theme (strict: matches primary label)", [r["pred"]["theme"] == r["gold"].theme for r in rows]),
        m("theme (lenient: matches primary or an acceptable alternative)", [r["pred"]["theme"] in {r["gold"].theme, *r["gold"].acceptable_themes} for r in rows]),
    ]

    # "has a problem" = theme != no_issue, judged on primary labels only
    tp = sum(1 for r in rows if r["gold"].theme != "no_issue" and r["pred"]["theme"] != "no_issue")
    fp = sum(1 for r in rows if r["gold"].theme == "no_issue" and r["pred"]["theme"] != "no_issue")
    fn = sum(1 for r in rows if r["gold"].theme != "no_issue" and r["pred"]["theme"] == "no_issue")
    ftp = sum(1 for r in rows if r["gold"].feature_request and (r["pred"]["feature_request"] or "").strip())
    ffp = sum(1 for r in rows if not r["gold"].feature_request and (r["pred"]["feature_request"] or "").strip())
    ffn = sum(1 for r in rows if r["gold"].feature_request and not (r["pred"]["feature_request"] or "").strip())

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    disagreements = []
    for r in rows:
        confusion[r["gold"].theme][r["pred"]["theme"]] += 1
        if r["pred"]["theme"] not in {r["gold"].theme, *r["gold"].acceptable_themes}:
            disagreements.append({
                "source_review_id": r["gold"].source_review_id, "url": r["review"]["url"], "gold_theme": r["gold"].theme,
                "acceptable": r["gold"].acceptable_themes, "predicted_theme": r["pred"]["theme"],
                "issue_label": r["pred"]["issue_label"], "text": r["review"]["body_clean"],
            })

    return EvalResult(
        model=model, prompt_version=prompt_version, gold_meta=meta, n_gold=len(gold), n_scored=len(rows), skipped=dict(skipped),
        metrics=metrics, precision_recall={"has_problem": _prf(tp, fp, fn), "feature_request": _prf(ftp, ffp, ffn)},
        confusion={k: dict(v) for k, v in confusion.items()}, disagreements=disagreements,
        quote_verified=m("model quotes found verbatim in review", [bool(r["pred"]["quote_verified"]) for r in rows]),
    )


def render_markdown(res: EvalResult, *, include_text: bool = True, details: bool = True) -> str:
    L = [f"# Evaluation: `{res.model}` · prompt `{res.prompt_version}`", ""]
    L += [
        "> **What this measures.** Agreement between the model and a labelled sample. It does **not** prove the model is "
        f"\"accurate\" in general. Labeller: _{res.gold_meta.get('labeler', 'unknown')}_.",
        "> Labels use a fixed taxonomy; where several themes are defensible the labeller listed acceptable alternatives "
        "(shown as _lenient_ scores).", "",
        f"- Gold items: **{res.n_gold}** · scored: **{res.n_scored}**",
    ]
    for reason, n in res.skipped.items():
        L.append(f"- Skipped {n}: {reason}")
    L += ["", "## Scores [measured]", "", "| Measure | Result |", "|---|---|"]
    for mt in res.metrics + ([res.quote_verified] if res.quote_verified else []):
        L.append(f"| {mt.name} | {mt.fmt()} |")
    L += ["", "### Binary detection", "", "| Task | Precision | Recall | F1 | TP / FP / FN |", "|---|---|---|---|---|"]
    for name, d in res.precision_recall.items():
        f = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
        L.append(f"| {name} | {f(d['precision'])} | {f(d['recall'])} | {f(d['f1'])} | {d['tp']} / {d['fp']} / {d['fn']} |")
    if details:
        L += ["", "## Theme confusion (rows = label, columns = model)", ""]
        cols = sorted({p for row in res.confusion.values() for p in row})
        L += ["| label \\ model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for g in sorted(res.confusion):
            L.append(f"| **{g}** | " + " | ".join(str(res.confusion[g].get(c, "")) for c in cols) + " |")
        L += ["", f"## Disagreements ({len(res.disagreements)}) - theme outside label + acceptable alternatives", ""]
        for d in res.disagreements:
            L.append(f"- review `{d['source_review_id']}`: label **{d['gold_theme']}** (also ok: {', '.join(d['acceptable']) or '-'}) vs model **{d['predicted_theme']}** "
                     f"/ \"{d['issue_label']}\"")
            if include_text:
                L.append(f"  - text: {' '.join(d['text'].split())[:300]}")
    else:
        L += ["", "_Per-item errors are deliberately hidden (`--no-details`): this set is held out for final scoring, so prompts must not be tuned against its individual mistakes._"]
    L += ["", "## Caveats", "",
          "- Small sample: read the confidence intervals, not just the percentages.",
          "- One rater. Inter-rater agreement has not been measured.",
          "- Labels were fixed before the model's answers were viewed (items flagged otherwise are excluded).",
          "- Severity is not scored: it is subjective and a single rater cannot establish a reliable reference.", ""]
    return "\n".join(L)

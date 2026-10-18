"""Aggregation of analysed reviews into themes and issue groups.

Everything numeric here is computed with pandas/Python - never by the LLM.

Each output field is tagged by provenance so the UI/report can keep them visibly separate:
  MEASURED     review counts, shares, average star rating, dates, links
  AI-INTERPRETED   theme assignment, severity, issue labels (and therefore the grouping built on them)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pda.models import SEVERITY_SCORE, THEMES

# The model's "ambiguous" label is NOT a reliable reason to discard a review: on real data most "ambiguous" verdicts were
# short but specific bug reports ("Latest version breaks self-hosting."). Only "irrelevant" (spam / off-topic / other
# product) is excluded; "ambiguous" stays in, counted separately as low-confidence.
RELEVANT = ("genuine_feedback", "ambiguous")
EVIDENCE_PER_GROUP = 3
ISSUE_SIMILARITY_THRESHOLD = 0.45


@dataclass
class Evidence:
    review_id: int
    source_review_id: str
    url: str
    review_date: str
    rating: int | None
    text: str
    is_verbatim_quote: bool  # False -> `text` is a plain excerpt from the review, not the model's quote


@dataclass
class IssueGroup:
    """A recurring, specific problem: reviews whose model-written problem descriptions are similar."""

    label: str  # AI-interpreted (the member description closest to the cluster centre)
    review_count: int  # measured
    share_of_relevant: float  # measured: review_count / relevant analysed reviews
    avg_rating: float | None  # measured
    mean_severity: float  # AI-interpreted, 0..4
    theme_mix: dict[str, int]  # AI-interpreted. Themes are unstable labels: the SAME problem is often filed under several
    low_confidence_count: int  # AI-interpreted: members the model also called "ambiguous"
    member_labels: list[str]
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class ThemeSummary:
    """Coarse category counts. The theme of a review is the least stable thing the model produces (see docs)."""

    theme: str  # AI-interpreted assignment
    review_count: int  # measured count of reviews assigned to the theme
    share_of_relevant: float  # measured: review_count / relevant (genuine + ambiguous) analysed reviews
    low_confidence_count: int  # AI-interpreted: how many of review_count the model also called "ambiguous"
    avg_rating: float | None  # measured
    severity_counts: dict[str, int]  # AI-interpreted
    feature_request_count: int  # AI-interpreted


@dataclass
class Overview:
    analysed: int
    genuine: int
    ambiguous: int
    irrelevant: int
    relevant: int
    problem_reviews: int
    sentiment_counts: dict[str, int]
    avg_rating_genuine: float | None
    quote_verified_rate: float | None
    date_min: str | None
    date_max: str | None


def _rating_mean(series: pd.Series) -> float | None:
    s = series.dropna()
    return round(float(s.mean()), 2) if len(s) else None


# ----------------------------------------------------------------------------- issue grouping

_SUFFIXES = ("ing", "ed", "es", "s")


def _stem(token: str) -> str:
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def _label_features(label: str) -> list[str]:
    toks = [_stem(t) for t in re.findall(r"[a-z][a-z0-9'-]+", label.lower()) if t not in ENGLISH_STOP_WORDS]
    # Bigrams are order-insensitive so "slow startup" and "startup is slow" share a feature.
    return toks + ["_".join(sorted(pair)) for pair in zip(toks, toks[1:])]


def group_labels(labels: list[str], threshold: float = ISSUE_SIMILARITY_THRESHOLD) -> list[list[int]]:
    """Leader clustering of short issue labels on TF-IDF cosine similarity.

    Deterministic: most frequent labels become cluster leaders first. Returns lists of indices
    into `labels`. Empty/unusable labels each stay on their own.
    """
    n = len(labels)
    if n == 0:
        return []
    try:
        vec = TfidfVectorizer(analyzer=_label_features, sublinear_tf=True)
        matrix = vec.fit_transform(labels)
    except ValueError:  # empty vocabulary
        return [[i] for i in range(n)]
    freq = Counter(labels)
    order = sorted(range(n), key=lambda i: (-freq[labels[i]], len(labels[i]), labels[i], i))
    leaders: list[int] = []
    clusters: dict[int, list[int]] = {}
    for i in order:
        best, best_sim = None, 0.0
        if leaders:
            sims = cosine_similarity(matrix[i], matrix[leaders]).ravel()
            j = int(sims.argmax())
            best, best_sim = leaders[j], float(sims[j])
        if best is not None and best_sim >= threshold:
            clusters[best].append(i)
        else:
            leaders.append(i)
            clusters[i] = [i]
    return [sorted(members) for members in clusters.values()]


# Seen on real data: the small model sometimes "labels" a problem with a schema field name ("bugs reliability",
# "no issue", "performance"). Those say nothing about WHICH problem, so grouping must ignore them.
GENERIC_LABELS = frozenset({t.replace("_", " ") for t in THEMES} | {"no issue", "none", "n/a", "issue", "problem", "bug", "bugs", "error", "errors"})


def informative_label(label: str | None) -> str:
    text = " ".join(str(label or "").lower().replace("_", " ").split()).strip(" .")
    return "" if text in GENERIC_LABELS else text


def display_label(label: str | None, problem_statement: str | None) -> str:
    """What identifies a review's problem: its informative label, else the start of the problem statement, else ''."""
    lab = informative_label(label)
    if lab:
        return lab
    stmt = " ".join(str(problem_statement or "").split()).strip(" .")
    if len(stmt) <= 70:
        return stmt
    return stmt[:69].rsplit(" ", 1)[0].rstrip(" ,;:-") + "\u2026"  # visibly truncated, never a sentence that just stops


def embedding_text(issue_label: str | None, problem_statement: str | None) -> str:
    """The short, model-written description that gets embedded and compared across reviews."""
    return f"{informative_label(issue_label)}. {' '.join(str(problem_statement or '').split())}".strip(" .")


def group_vectors(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage agglomerative clustering on cosine distance.

    `vectors` must already be mean-centred. Why: every complaint says roughly "<product> stopped working", so raw
    embeddings are all ~0.75 similar and collapse into one giant cluster (measured: 151 of 210 reviews). Subtracting
    the mean vector removes that shared component so the actual differences between problems remain.
    """
    n = len(vectors)
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    norms = np.linalg.norm(vectors, axis=1)
    # A centred vector of exactly zero is "the average complaint" (e.g. identical texts in a tiny dataset).
    # sklearn refuses cosine distance for zero vectors, so those rows form one group of their own.
    nonzero = [i for i in range(n) if norms[i] > 1e-9]
    zero = [i for i in range(n) if norms[i] <= 1e-9]
    result: list[list[int]] = [zero] if zero else []
    if len(nonzero) == 1:
        result.append(nonzero)
    elif nonzero:
        unit = vectors[nonzero] / norms[nonzero][:, None]
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=threshold).fit_predict(unit)
        groups: dict[int, list[int]] = defaultdict(list)
        for pos, lab in zip(nonzero, labels):
            groups[int(lab)].append(pos)
        result.extend(sorted(m) for m in groups.values())
    return result


def _medoid_label(labels: list[str], unit_vectors: np.ndarray, members: list[int]) -> str:
    """Label of the member closest to the cluster centre (ties: shorter, then alphabetical)."""
    centre = unit_vectors[members].mean(axis=0)
    scored = sorted(((-float(unit_vectors[i] @ centre), len(labels[i]), labels[i]) for i in members))
    return scored[0][2]


def _excerpt(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _evidence_rows(group: pd.DataFrame, k: int = EVIDENCE_PER_GROUP) -> list[Evidence]:
    ranked = group.assign(
        _sev=group["severity"].map(SEVERITY_SCORE).fillna(0),
        _verified=group["quote_verified"].fillna(0).astype(int),
    ).sort_values(["_verified", "_sev", "review_date"], ascending=[False, False, False])
    out: list[Evidence] = []
    for _, r in ranked.head(k).iterrows():
        verbatim = bool(r["_verified"]) and bool(r["quote"])
        out.append(
            Evidence(
                review_id=int(r["review_id"]),
                source_review_id=str(r["source_review_id"]),
                url=str(r["url"]),
                review_date=pd.Timestamp(r["review_date"]).strftime("%Y-%m-%d"),
                rating=int(r["rating"]) if pd.notna(r["rating"]) else None,
                text=str(r["quote"]) if verbatim else _excerpt(str(r["body_clean"])),
                is_verbatim_quote=verbatim,
            )
        )
    return out


# ----------------------------------------------------------------------------- public API


def overview(df: pd.DataFrame) -> Overview:
    if df.empty:
        return Overview(0, 0, 0, 0, 0, 0, {}, None, None, None, None)
    relevant = df[df["relevance"].isin(RELEVANT)]
    problems = relevant[relevant["theme"] != "no_issue"]
    return Overview(
        analysed=len(df),
        genuine=int((df["relevance"] == "genuine_feedback").sum()),
        ambiguous=int((df["relevance"] == "ambiguous").sum()),
        irrelevant=int((df["relevance"] == "irrelevant").sum()),
        relevant=len(relevant),
        problem_reviews=len(problems),
        sentiment_counts={k: int(v) for k, v in relevant["sentiment"].value_counts().items()},
        avg_rating_genuine=_rating_mean(relevant["rating"]),
        quote_verified_rate=round(float(df["quote_verified"].fillna(0).astype(int).mean()), 3),
        date_min=pd.Timestamp(df["review_date"].min()).strftime("%Y-%m-%d"),
        date_max=pd.Timestamp(df["review_date"].max()).strftime("%Y-%m-%d"),
    )


def _problem_reviews(df: pd.DataFrame) -> pd.DataFrame:
    relevant = df[df["relevance"].isin(RELEVANT)]
    return relevant[relevant["theme"] != "no_issue"]


def summarise_themes(df: pd.DataFrame) -> list[ThemeSummary]:
    """Counts per theme among relevant (genuine + ambiguous) problem reviews, most-mentioned first."""
    relevant = df[df["relevance"].isin(RELEVANT)]
    if relevant.empty:
        return []
    summaries: list[ThemeSummary] = []
    for theme, sub in _problem_reviews(df).groupby("theme"):
        summaries.append(
            ThemeSummary(
                theme=str(theme),
                review_count=len(sub),
                share_of_relevant=round(len(sub) / len(relevant), 3),
                low_confidence_count=int((sub["relevance"] == "ambiguous").sum()),
                avg_rating=_rating_mean(sub["rating"]),
                severity_counts={k: int(v) for k, v in sub["severity"].value_counts().items()},
                feature_request_count=int((sub["feature_request"].fillna("").str.strip() != "").sum()),
            )
        )
    summaries.sort(key=lambda s: (-s.review_count, s.theme))
    return summaries


def summarise_issues(
    df: pd.DataFrame,
    threshold: float = ISSUE_SIMILARITY_THRESHOLD,
    *,
    vectors: dict[int, np.ndarray] | None = None,
    vector_threshold: float = 0.75,
) -> list[IssueGroup]:
    """Recurring problems, grouped ACROSS themes, most-mentioned first.

    Uses `vectors` ({review_id: embedding}) when they cover every describable problem review, otherwise lexical
    TF-IDF grouping with `threshold`. Reviews with neither an informative label nor a problem statement are left out.
    """
    relevant = df[df["relevance"].isin(RELEVANT)]
    problems = _problem_reviews(df).copy()
    if problems.empty:
        return []
    problems["_display"] = [display_label(a, b) for a, b in zip(problems["issue_label"], problems["problem_statement"])]
    frame = problems[problems["_display"] != ""].reset_index(drop=True)
    if frame.empty:
        return []
    labels = frame["_display"].tolist()

    unit = None
    if vectors and all(int(r) in vectors for r in frame["review_id"]):
        raw = np.stack([vectors[int(r)] for r in frame["review_id"]])
        centred = raw - raw.mean(axis=0)  # remove the "<product> stopped working" component shared by every complaint
        unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-9)
        clusters = group_vectors(centred, vector_threshold)
    else:
        clusters = group_labels(labels, threshold)

    groups: list[IssueGroup] = []
    for members in clusters:
        part = frame.iloc[members]
        member_labels = [labels[i] for i in members]
        if unit is not None:
            label = _medoid_label(labels, unit, members)
        else:
            counts = Counter(member_labels)
            label = sorted(counts, key=lambda s: (-counts[s], len(s), s))[0]
        groups.append(
            IssueGroup(
                label=label,
                review_count=len(part),
                share_of_relevant=round(len(part) / len(relevant), 3),
                avg_rating=_rating_mean(part["rating"]),
                mean_severity=round(float(part["severity"].map(SEVERITY_SCORE).fillna(0).mean()), 2),
                theme_mix={k: int(v) for k, v in part["theme"].value_counts().items()},
                low_confidence_count=int((part["relevance"] == "ambiguous").sum()),
                member_labels=sorted(set(member_labels)),
                evidence=_evidence_rows(part),
            )
        )
    groups.sort(key=lambda g: (-g.review_count, -g.mean_severity, g.label))
    return groups

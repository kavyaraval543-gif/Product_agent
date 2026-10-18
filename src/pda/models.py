"""Domain models and the structured-output contract for the LLM.

Two kinds of data flow through this system and they are kept apart on purpose:
  * MEASURED   - counts, ratings, dates, links copied from the source. Deterministic.
  * AI-DERIVED - sentiment, severity, theme, issue label. Produced by the local LLM,
                 therefore an interpretation, never a measurement.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- source side


class RawReview(BaseModel):
    """One review exactly as a source returned it (minus reviewer identity - see docs)."""

    source: str
    source_review_id: str
    rating: int | None = Field(None, ge=1, le=5)
    body: str | None
    review_date: datetime  # timezone-aware, UTC
    app_version: str | None = None
    url: str


class CleanedReview(BaseModel):
    """A RawReview plus deterministic cleaning results; this is what gets stored."""

    raw: RawReview
    body_clean: str
    language: str | None
    usable: bool
    unusable_reason: str | None
    content_hash: str
    word_count: int


# --------------------------------------------------------------------------- LLM contract

Relevance = Literal["genuine_feedback", "ambiguous", "irrelevant"]
Sentiment = Literal["positive", "neutral", "negative", "mixed"]
Severity = Literal["none", "low", "medium", "high", "critical"]
Theme = Literal[
    "bugs_reliability",
    "performance",
    "usability_ux",
    "missing_feature",
    "compatibility",
    "account_sync",
    "onboarding_setup",
    "pricing_value",
    "privacy_security",
    "customer_support",
    "no_issue",
    "other",
]

# Definitions are injected into the prompt, so this dict is part of the prompt contract.
THEMES: dict[str, str] = {
    "bugs_reliability": "something is broken: errors, crashes, features that do not work or stopped working, data loss",
    "performance": "slow, laggy, freezing; high memory, CPU or battery use",
    "usability_ux": "it works, but is confusing, cluttered, ugly or annoying to use; unwanted design or behaviour changes (layout, pop-ups, defaults)",
    "missing_feature": "asks for a capability or option the product does not have",
    "compatibility": "does not work with specific websites, apps, browsers, devices, OS versions or servers",
    "account_sync": "cannot log in, locked out, logged out unexpectedly, or data does not sync, back up or restore",
    "onboarding_setup": "installing, first-time setup, importing data, learning curve",
    "pricing_value": "price, paywalls, subscriptions, whether it is worth the money",
    "privacy_security": "privacy, data handling, security concerns, trust in the company",
    "customer_support": "support responsiveness, communication, company behaviour",
    "no_issue": "praise or a neutral remark only; nothing wrong, missing or annoying",
    "other": "genuine feedback about the product that fits none of the above",
}

SEVERITY_SCORE: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

_LIMITS = {"issue_label": 80, "problem_statement": 300, "feature_request": 200, "supporting_quote": 400}
_NULLISH = {"none", "n/a", "na", "null", "nil", "no", "-", "--", "not applicable", "not specified", "no problem", "no issue"}


class ReviewAnalysisOut(BaseModel):
    """What the LLM must return for ONE review.

    FIELD ORDER IS GENERATION ORDER (the output is grammar-constrained to this schema):
      * `reports_problem` is decided BEFORE the problem fields. Measured on real reviews (prompt v2): when the
        schema forced a problem statement/label/theme, the model invented problems for praise-only reviews.
      * the problem statement and label come BEFORE theme and severity, so the model commits to what the
        problem is before categorising it.
    """

    relevance: Relevance
    sentiment: Sentiment
    reports_problem: bool
    problem_statement: str
    issue_label: str
    theme: Theme
    severity: Severity
    feature_request: str
    supporting_quote: str

    @field_validator("issue_label", "problem_statement", "feature_request", "supporting_quote", mode="before")
    @classmethod
    def _tidy(cls, value: object, info) -> str:
        text = "" if value is None else str(value)
        text = " ".join(text.split())
        if text.lower().strip(" .") in _NULLISH:
            return ""
        if info.field_name == "issue_label":
            # Models often emit snake_case / kebab-case ("passkey_popup_stuck"); normalise to plain words.
            text = " ".join(text.replace("_", " ").replace("-", " ").lower().strip(" .\"'").split())
        return text[: _LIMITS[info.field_name]]

    def normalised(self) -> tuple["ReviewAnalysisOut", list[str]]:
        """Apply deterministic consistency rules. Returns (consistent copy, names of overrides applied).

        The model's raw answer is kept separately for audit; every override is recorded, never silent.
        """
        fixes: list[str] = []
        data = self.model_dump()
        if not self.reports_problem:
            for field, empty in (("theme", "no_issue"), ("severity", "none"), ("issue_label", ""), ("problem_statement", "")):
                if data[field] != empty:
                    fixes.append(f"no_problem:{field}")
                    data[field] = empty
        elif self.theme == "no_issue":
            fixes.append("problem_without_theme:other")
            data["theme"] = "other"
        if data["theme"] == "no_issue" and (data["severity"] != "none" or data["issue_label"]):
            fixes.append("no_issue_cleared")
            data["severity"], data["issue_label"] = "none", ""
        return (ReviewAnalysisOut.model_validate(data), fixes) if fixes else (self, fixes)

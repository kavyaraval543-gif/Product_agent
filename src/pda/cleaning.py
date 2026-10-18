"""Deterministic cleaning. No LLM involved, so every decision here is reproducible and testable.

Principle: when unsure, KEEP the review (precision over recall on "unusable").
The one exception is privacy: obvious personal identifiers are redacted before storage of the
clean text, so downstream analysis, quotes and exports never carry them.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from functools import lru_cache

from pda.config import CleaningConfig
from pda.models import CleanedReview, RawReview

_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f​-‏‪-‮﻿]")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{7,}\d(?![\w])")
_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def normalise_text(text: str) -> str:
    """Unicode-normalise, strip markup and control characters, collapse whitespace."""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'").replace("\u2018", "'")
    text = _CONTROL_RE.sub("", text)
    return " ".join(text.split())


def redact_pii(text: str) -> str:
    """Replace e-mail addresses and phone-number-like digit runs (>= 9 digits)."""
    text = _EMAIL_RE.sub("[email]", text)

    def _phone(match: re.Match[str]) -> str:
        digits = sum(ch.isdigit() for ch in match.group(0))
        return "[phone]" if digits >= 9 else match.group(0)

    return _PHONE_RE.sub(_phone, text)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def content_hash(text: str) -> str:
    """Hash of the lowercase alphanumeric content: robust to spacing/punctuation/case edits."""
    canon = "".join(ch for ch in text.lower() if ch.isalnum())
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- language

# ~30 languages that realistically appear in app reviews. Restricting the candidate set bounds memory
# (lingua loads models lazily, ~15-30 MB each) and avoids odd guesses (e.g. Somali for the word "good").
_CANDIDATES = (
    "ENGLISH SPANISH FRENCH GERMAN ITALIAN PORTUGUESE DUTCH POLISH RUSSIAN UKRAINIAN TURKISH ARABIC HINDI CHINESE "
    "JAPANESE KOREAN INDONESIAN VIETNAMESE THAI SWEDISH DANISH BOKMAL FINNISH CZECH HUNGARIAN ROMANIAN GREEK HEBREW "
    "PERSIAN BENGALI CATALAN"
).split()

# Below this probability for English we call the text non-English. Chosen from an audit of 300 real reviews
# (see docs/FEASIBILITY.md): it keeps short/bilingual English text while rejecting clear foreign text.
EN_MIN_PROBABILITY = 0.10


@lru_cache(maxsize=1)
def _detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(*(getattr(Language, name) for name in _CANDIDATES)).build()


def detect_language(text: str) -> str:
    """ISO 639-1 code of the most likely language, or 'en' when English is plausible or there is no evidence.

    Uses the open-source `lingua` n-gram detector. English is accepted whenever it is the top guess OR has
    probability >= EN_MIN_PROBABILITY, so bilingual and very short English reviews are not discarded
    (wrongly dropping English is worse than keeping a stray foreign review).
    """
    from lingua import Language

    if not any(ch.isalpha() for ch in text):
        return "en"
    confidence = _detector().compute_language_confidence_values(text)
    if not confidence:
        return "en"
    p_en = next((c.value for c in confidence if c.language == Language.ENGLISH), 0.0)
    top = confidence[0]
    if top.language == Language.ENGLISH or p_en >= EN_MIN_PROBABILITY:
        return "en"
    return top.language.iso_code_639_1.name.lower()


# ----------------------------------------------------------------------------- orchestration


def assess_usability(clean: str, language: str, cfg: CleaningConfig) -> tuple[bool, str | None]:
    if not clean:
        return False, "no_text"
    if sum(ch.isalpha() for ch in clean) < 3:
        return False, "no_letters"
    toks = words(clean)
    if len(toks) < cfg.min_words:
        return False, "too_short"
    if language not in cfg.languages:
        return False, f"language_{language}"
    lowered = [t.lower() for t in toks]
    if len(lowered) >= 4 and len(set(lowered)) / len(lowered) < 0.3:
        return False, "low_information"
    return True, None


def clean_review(raw: RawReview, cfg: CleaningConfig) -> CleanedReview:
    body = raw.body or ""
    clean = redact_pii(normalise_text(body))
    language = detect_language(clean) if clean else None
    usable, reason = assess_usability(clean, language or "en", cfg)
    return CleanedReview(
        raw=raw,
        body_clean=clean,
        language=language,
        usable=usable,
        unusable_reason=reason,
        content_hash=content_hash(clean),
        word_count=len(words(clean)),
    )

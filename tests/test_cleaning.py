import pytest

from helpers import make_review
from pda.cleaning import clean_review, content_hash, detect_language, normalise_text, redact_pii
from pda.config import CleaningConfig

CFG = CleaningConfig()


def test_normalise_strips_markup_entities_control_chars_and_collapses_space():
    raw = "  Great&nbsp;app<br>but​ it   crashes\n\n\twhen I’m offline &amp; tired "
    assert normalise_text(raw) == "Great app but it crashes when I'm offline & tired"


def test_redacts_emails_and_phone_numbers_but_not_versions_or_short_numbers():
    text = "mail me at jane.doe+x@example.co.uk or call +91 98765 43210, version 2026.10.3 broke 12 things"
    out = redact_pii(text)
    assert "[email]" in out and "[phone]" in out
    assert "jane.doe" not in out and "98765" not in out
    assert "2026.10.3" in out and "12 things" in out


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Autofill stopped working on half the sites since the last update", "en"),
        ("Excellent password manager, highly recommended", "en"),
        ("Latest version breaks self-hosting.", "en"),  # terse technical English
        ("Slows down some sites immensely", "en"),
        ("Muy buena aplicación, funciona perfecto y también es muy rápida", "es"),
        ("Funktioniert nicht mehr seit dem Update, sehr schade und auch nervig", "de"),
        ("Cette application ne fonctionne plus depuis la mise à jour, très décevant mais utile", "fr"),
        ("Po ostatniej aktualizacji rozszerzenie przestało działać", "pl"),  # missed by the old stopword heuristic
        ("Отличное приложение, но после обновления не работает автозаполнение", "ru"),
        ("这个应用很好用，但是自动填充经常失败", "zh"),
        ("", "en"),
        ("12345 !!!", "en"),
        # Regressions from auditing real reviews (2026-10-17): English words that look foreign to a naive heuristic.
        ("Secure, no nonsense, no ads and a professional password manager", "en"),
        ("The application is great and the plus side is that autofill works", "en"),
        # Bilingual review (English followed by its Russian translation) must stay usable as English.
        ("The autocomplete settings are not being saved after the update. Настройки автозаполнения не сохраняются после обновления.", "en"),
    ],
)
def test_language_detection(text, expected):
    assert detect_language(text) == expected


def test_content_hash_ignores_case_spacing_and_punctuation():
    assert content_hash("Works GREAT, really!") == content_hash("works great really")
    assert content_hash("works great") != content_hash("works badly")


@pytest.mark.parametrize(
    "body,usable,reason",
    [
        ("Autofill keeps failing on my bank website", True, None),
        (None, False, "no_text"),
        ("   ", False, "no_text"),
        ("!!! ??? ...", False, "no_letters"),
        ("Great app", False, "too_short"),
        ("Muy buena aplicación, funciona perfecto y también es muy rápida", False, "language_es"),
        ("good good good good good good", False, "low_information"),
    ],
)
def test_usability_rules(body, usable, reason):
    cleaned = clean_review(make_review(1).model_copy(update={"body": body}), CFG)
    assert cleaned.usable is usable
    assert cleaned.unusable_reason == reason


def test_cleaned_text_is_pii_free_and_raw_is_preserved():
    review = make_review(1, body="Support never answered my mail to help@vendor.com about the broken sync")
    cleaned = clean_review(review, CFG)
    assert "help@vendor.com" not in cleaned.body_clean and "[email]" in cleaned.body_clean
    assert cleaned.raw.body == review.body  # untouched original
    assert cleaned.word_count == len(cleaned.body_clean.split())

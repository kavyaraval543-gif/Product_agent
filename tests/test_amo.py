"""AMO parsing. The payload shape mirrors responses observed live on 2026-10-17 (content is synthetic)."""

import pytest

from pda.errors import SourceFormatError
from pda.sources.amo import AmoSource


def rating(i, *, created="2026-10-10T10:00:00Z", body="Autofill is broken on my bank site", score=2, **extra):
    base = {
        "id": i,
        "addon": {"id": 1, "slug": "demo-addon", "name": {"en-US": "Demo"}},
        "body": body,
        "created": created,
        "is_deleted": False,
        "is_developer_reply": False,
        "is_latest": True,
        "previous_count": 0,
        "reply": None,
        "score": score,
        "user": {"id": 777, "name": "Jane Reviewer", "url": "https://example/user/777", "username": "anonymous-abc"},
        "version": {"id": 9, "version": "2026.9.3"},
    }
    base.update(extra)
    return base


def page(results, *, nxt=None, count=None, grouped=None):
    d = {"page_size": 50, "page_count": 2, "count": count or len(results), "next": nxt, "previous": None, "results": results}
    if grouped is not None:
        d["grouped_ratings"] = grouped
    return d


class ScriptedClient:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        return self.pages.pop(0)


def source(pages, **kw):
    client = ScriptedClient(pages)
    return AmoSource(client, "demo-addon", **kw), client


def test_parses_review_fields_and_builds_verified_permalink_pattern():
    src, _ = source([page([rating(42, created="2026-10-10T10:00:00Z", score=1)])])
    (p,) = list(src.iter_pages(5))
    r = p.reviews[0]
    assert r.source == "amo" and r.source_review_id == "42" and r.rating == 1
    assert r.app_version == "2026.9.3"
    assert r.review_date.isoformat() == "2026-10-10T10:00:00+00:00"
    assert r.url == "https://addons.mozilla.org/en-US/firefox/addon/demo-addon/reviews/42/"


def test_reviewer_identity_never_leaves_the_parser():
    src, _ = source([page([rating(1)])])
    r = list(src.iter_pages(1))[0].reviews[0]
    dumped = r.model_dump_json()
    assert "Jane" not in dumped and "777" not in dumped and "anonymous" not in dumped


def test_skips_deleted_developer_empty_and_invalid_items_and_counts_them():
    items = [
        rating(1),
        rating(2, is_deleted=True),
        rating(3, is_developer_reply=True),
        rating(4, body=None),
        rating(5, body="   "),
        rating(6, score=9),  # invalid score
        {"id": "x"},  # malformed
    ]
    src, _ = source([page(items)])
    (p,) = list(src.iter_pages(1))
    assert [r.source_review_id for r in p.reviews] == ["1"]
    assert p.skipped == {"deleted": 1, "developer": 1, "no_text": 2, "invalid": 2}


def test_requests_text_only_reviews_with_documented_params_and_grouped_ratings_on_page_one_only():
    src, client = source([page([rating(1)], nxt="x"), page([rating(2)])])
    list(src.iter_pages(5))
    (_, p1), (_, p2) = client.calls
    assert p1["filter"] == "without_empty_body" and p1["addon"] == "demo-addon" and p1["page_size"] == 50
    assert p1["page"] == 1 and p1["show_grouped_ratings"] == "true"
    assert p2["page"] == 2 and "show_grouped_ratings" not in p2


def test_page_size_is_capped_at_the_verified_maximum():
    src, client = source([page([rating(1)])], page_size=500)
    list(src.iter_pages(1))
    assert client.calls[0][1]["page_size"] == 50


def test_stops_when_there_is_no_next_page_and_respects_max_pages():
    src, client = source([page([rating(1)], nxt="n"), page([rating(2)], nxt="n"), page([rating(3)])])
    assert len(list(src.iter_pages(2))) == 2
    assert len(client.calls) == 2


def test_snapshot_captures_total_text_reviews_and_star_distribution():
    grouped = {"1": 5, "2": 3, "3": 2, "4": 10, "5": 80}
    src, _ = source([page([rating(1)], count=3144, grouped=grouped)])
    list(src.iter_pages(1))
    assert src.snapshot.total_text_reviews == 3144
    assert src.snapshot.grouped_ratings == grouped


def test_ordering_flag_true_for_newest_first_across_pages():
    p1 = page([rating(3, created="2026-10-10T00:00:00Z"), rating(2, created="2026-10-09T00:00:00Z")], nxt="n")
    p2 = page([rating(1, created="2026-10-08T00:00:00Z")])
    src, _ = source([p1, p2])
    assert [p.ordered_newest_first for p in src.iter_pages(5)] == [True, True]


def test_ordering_flag_false_when_a_page_is_out_of_order():
    p1 = page([rating(1, created="2026-10-01T00:00:00Z"), rating(2, created="2026-10-09T00:00:00Z")])
    src, _ = source([p1])
    assert [p.ordered_newest_first for p in src.iter_pages(1)] == [False]


def test_ordering_flag_false_when_pages_overlap_in_time():
    p1 = page([rating(2, created="2026-10-05T00:00:00Z")], nxt="n")
    p2 = page([rating(1, created="2026-10-09T00:00:00Z")])  # newer than the previous page's oldest
    src, _ = source([p1, p2])
    assert [p.ordered_newest_first for p in src.iter_pages(5)] == [True, False]


@pytest.mark.parametrize("bad", [[], {"detail": "nope"}, {"results": "not-a-list"}])
def test_unexpected_shapes_raise_a_clear_error(bad):
    src, _ = source([bad])
    with pytest.raises(SourceFormatError):
        list(src.iter_pages(1))

"""Embedding-based issue grouping: client, clustering maths, caching, fallback."""

import json

import numpy as np
import pandas as pd
import pytest

from helpers import FakeLLM, ScriptedServer, plain_session
from pda.config import Settings
from pda.db import Database
from pda.errors import LLMModelMissing, LLMOutputError, LLMUnavailable
from pda.llm import OllamaClient
from pda.pipeline import compute_embeddings
from pda.themes import embedding_text, group_vectors, summarise_issues

# ----------------------------------------------------------------------------- Ollama embed client


@pytest.fixture
def server():
    s = ScriptedServer()
    yield s
    s.close()


def client(server) -> OllamaClient:
    return OllamaClient(server.url, "chat-model", session=plain_session())


def test_embed_batches_applies_task_prefix_and_preserves_order(server):
    server.add("/api/embed", body={"embeddings": [[1.0, 0.0]] * 2})
    server.add("/api/embed", body={"embeddings": [[0.0, 1.0]]})
    vecs = client(server).embed(["a", "b", "c"], model="nomic-embed-text", batch_size=2)
    assert vecs == [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    first, second = (json.loads(r["body"]) for r in server.requests)
    assert first["model"] == "nomic-embed-text" and first["input"] == ["clustering: a", "clustering: b"]
    assert second["input"] == ["clustering: c"]


def test_embed_error_handling(server):
    server.add("/api/embed", status=404, body="not found")
    with pytest.raises(LLMModelMissing, match="ollama pull nomic-embed-text"):
        client(server).embed(["x"], model="nomic-embed-text")
    with pytest.raises(LLMUnavailable):
        OllamaClient("http://127.0.0.1:9", "m", session=plain_session()).embed(["x"], model="e")


def test_embed_rejects_wrong_count_or_shape(server):
    server.add("/api/embed", body={"embeddings": [[1.0]]})
    with pytest.raises(LLMOutputError, match="received 1"):
        client(server).embed(["a", "b"], model="e")
    server2 = ScriptedServer()
    try:
        server2.add("/api/embed", body={"unexpected": True})
        with pytest.raises(LLMOutputError, match="shape"):
            client(server2).embed(["a"], model="e")
    finally:
        server2.close()


# ----------------------------------------------------------------------------- clustering maths


def blob(direction: np.ndarray, n: int, rng, noise=0.05) -> np.ndarray:
    return direction + rng.normal(0, noise, size=(n, len(direction)))


def test_centering_is_what_separates_problems_that_share_a_common_component():
    """Every complaint embeds near the same 'product stopped working' vector; only the residual differs."""
    rng = np.random.default_rng(0)
    dim = 16
    shared = np.zeros(dim); shared[0] = 10.0                      # the big common component
    dirs = [np.eye(dim)[i] for i in (1, 2, 3)]                    # three genuinely different problems
    raw = np.vstack([blob(shared + d, 6, rng) for d in dirs])
    truth = [0] * 6 + [1] * 6 + [2] * 6

    uncentred = group_vectors(raw, 0.3)
    assert len(uncentred) == 1  # everything collapses into one cluster (what we measured on real reviews)

    clusters = group_vectors(raw - raw.mean(axis=0), 0.5)
    assert len(clusters) == 3
    for members in clusters:  # each recovered cluster is pure
        assert len({truth[i] for i in members}) == 1


def test_group_vectors_edge_cases_and_determinism():
    assert group_vectors(np.zeros((0, 4)), 0.5) == []
    assert group_vectors(np.ones((1, 4)), 0.5) == [[0]]
    rng = np.random.default_rng(1)
    v = rng.normal(size=(12, 8))
    assert group_vectors(v, 0.9) == group_vectors(v.copy(), 0.9)
    # Regression: sklearn raises on zero vectors with cosine distance. After mean-centring, identical texts become zero
    # vectors (e.g. a tiny dataset), which used to crash report generation.
    assert group_vectors(np.zeros((3, 4)), 0.5) == [[0, 1, 2]]  # identical -> one group
    mixed = np.vstack([np.zeros((2, 4)), np.eye(4)[:2] + 0.01])
    groups = group_vectors(mixed, 0.5)
    assert [0, 1] in groups and sorted(i for g in groups for i in g) == [0, 1, 2, 3]  # every row accounted for exactly once


def test_threshold_controls_granularity():
    rng = np.random.default_rng(2)
    v = np.vstack([blob(np.eye(8)[i], 4, rng, noise=0.2) for i in range(4)])
    v = v - v.mean(axis=0)
    assert len(group_vectors(v, 0.2)) >= len(group_vectors(v, 0.9))


# ----------------------------------------------------------------------------- integration with summarise_themes


def prow(i, label, statement="", theme="bugs_reliability", rating=2, relevance="genuine_feedback"):
    return {
        "review_id": i, "source_review_id": str(1000 + i), "rating": rating, "body_clean": f"full text {i} " * 6,
        "review_date": pd.Timestamp("2026-10-01"), "url": f"https://example.test/r/{i}", "app_version": "1", "relevance": relevance,
        "sentiment": "negative", "theme": theme, "severity": "medium", "issue_label": label, "problem_statement": statement,
        "feature_request": "", "quote": None, "quote_verified": 0,
    }


def test_embeddings_merge_synonyms_that_lexical_grouping_cannot():
    # The documented limitation of TF-IDF: "fails" vs "not working" share no words.
    df = pd.DataFrame([prow(1, "autofill fails on login pages"), prow(2, "autofill not working on login page"), prow(3, "slow startup")])
    assert [g.review_count for g in summarise_issues(df)] == [1, 1, 1]  # lexical: 3 groups

    shared = np.array([5.0, 0, 0, 0])
    vecs = {1: shared + [0, 1, 0, 0], 2: shared + [0, 1.02, 0.01, 0], 3: shared + [0, 0, 1, 0]}  # 1 and 2 are semantically close
    groups = summarise_issues(df, vectors={k: np.array(v, dtype=np.float32) for k, v in vecs.items()}, vector_threshold=0.5)
    assert sorted(g.review_count for g in groups) == [1, 2]
    merged = next(g for g in groups if g.review_count == 2)
    assert set(merged.member_labels) == {"autofill fails on login pages", "autofill not working on login page"}
    assert merged.label in merged.member_labels  # the label is always a real member label, never invented


def test_missing_vectors_fall_back_to_lexical_grouping():
    df = pd.DataFrame([prow(1, "autofill fails on login pages"), prow(2, "autofill failing on login page")])
    only_one = {1: np.array([1.0, 0.0], dtype=np.float32)}  # no vector for review 2 -> must not crash or drop it
    groups = summarise_issues(df, vectors=only_one)
    assert sum(g.review_count for g in groups) == 2 and len(groups) == 1  # lexical grouping merged the wording variants


def test_reviews_with_no_description_are_never_embedded_or_grouped():
    df = pd.DataFrame([prow(1, ""), prow(2, "autofill fails")])  # review 1: no label and no problem statement
    groups = summarise_issues(df, vectors={2: np.array([1.0, 0.0], dtype=np.float32)})
    assert sum(g.review_count for g in groups) == 1


# ----------------------------------------------------------------------------- compute_embeddings + cache


@pytest.fixture
def db_df():
    db = Database(":memory:")
    app = db.upsert_app("fake", "demo", "Demo")
    from helpers import make_review
    from pda.cleaning import clean_review

    db.insert_reviews(app, None, [clean_review(make_review(i), Settings().cleaning) for i in range(1, 6)])
    ids = [r["id"] for r in db.conn.execute("SELECT id FROM reviews ORDER BY id")]
    rows = [
        prow(ids[0], "autofill fails", "Autofill fails."),
        prow(ids[1], "slow startup", "Startup is slow."),
        prow(ids[2], "", ""),                                               # nothing to describe the problem -> never embedded
        prow(ids[3], "praise", "x", theme="no_issue"),                     # not a problem -> never embedded
        prow(ids[4], "spam", "x", relevance="irrelevant"),                  # irrelevant -> never embedded
    ]
    return db, pd.DataFrame(rows), ids


def test_only_labelled_relevant_problem_reviews_are_embedded(db_df):
    db, df, ids = db_df
    llm = FakeLLM(); llm.embedder = lambda t: [float(len(t)), 1.0]
    out = compute_embeddings(Settings(), db, llm, df)
    assert sorted(out) == sorted(ids[:2])
    assert llm.embed_calls == [[embedding_text("autofill fails", "Autofill fails."), embedding_text("slow startup", "Startup is slow.")]]


def test_generic_echo_labels_fall_back_to_the_problem_statement(db_df):
    db, df, ids = db_df
    df = df.copy()
    df.loc[0, "issue_label"] = "bugs reliability"      # model echoed a schema field name (seen on real data)
    df.loc[1, "issue_label"] = "no issue"
    llm = FakeLLM(); llm.embedder = lambda t: [float(len(t)), 1.0]
    out = compute_embeddings(Settings(), db, llm, df)
    assert sorted(out) == sorted(ids[:2])  # still embedded, via the statement
    assert llm.embed_calls == [["Autofill fails", "Startup is slow"]]  # the echo labels contributed nothing


def test_embeddings_are_cached_and_recomputed_only_when_the_text_changes(db_df):
    db, df, ids = db_df
    llm = FakeLLM(); llm.embedder = lambda t: [float(len(t)), 1.0]
    first = compute_embeddings(Settings(), db, llm, df)
    again = compute_embeddings(Settings(), db, llm, df)
    assert len(llm.embed_calls) == 1  # second call served entirely from SQLite
    assert all(np.allclose(first[k], again[k]) for k in first)
    df2 = df.copy(); df2.loc[0, "problem_statement"] = "A different problem statement now."
    compute_embeddings(Settings(), db, llm, df2)
    assert len(llm.embed_calls) == 2 and len(llm.embed_calls[1]) == 1  # only the changed review was re-embedded


def test_missing_embedding_model_raises_actionable_error(db_df):
    db, df, _ = db_df
    with pytest.raises(LLMModelMissing, match="ollama pull nomic-embed-text"):
        compute_embeddings(Settings(), db, FakeLLM(), df)


def test_embedding_cache_roundtrip_is_exact_float32(db_df):
    db, _, ids = db_df
    v = np.array([0.1, -0.25, 3.5], dtype=np.float32)
    db.save_embeddings("m", [(ids[0], "abc", v)])
    got = db.cached_embeddings("m", {ids[0]: "abc"})
    assert np.array_equal(got[ids[0]], v)
    assert db.cached_embeddings("m", {ids[0]: "different-hash"}) == {}
    assert db.cached_embeddings("other-model", {ids[0]: "abc"}) == {}


def test_group_display_label_never_shows_an_echoed_field_name():
    df = pd.DataFrame([prow(1, "bugs reliability", "Passkey popup never appears on macOS."), prow(2, "no issue", "Passkey popup never appears.")])
    groups = summarise_issues(df)
    assert all(g.label not in ("bugs reliability", "no issue") for g in groups)
    assert sum(g.review_count for g in groups) == 2  # both reviews still grouped, described by their statements

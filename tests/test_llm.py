import json

import pytest

from helpers import ScriptedServer, plain_session
from pda.errors import LLMModelMissing, LLMOutputError, LLMUnavailable
from pda.llm import OllamaClient
from pda.models import ReviewAnalysisOut

GOOD = {
    "relevance": "genuine_feedback", "sentiment": "negative", "reports_problem": True, "theme": "bugs_reliability", "severity": "high",
    "issue_label": "autofill fails", "problem_statement": "Autofill fails.", "feature_request": "", "supporting_quote": "it fails",
}


def chat_reply(content) -> dict:
    return {"message": {"role": "assistant", "content": content if isinstance(content, str) else json.dumps(content)},
            "eval_count": 77, "prompt_eval_count": 123}


@pytest.fixture
def server():
    s = ScriptedServer()
    yield s
    s.close()


def client(server, **kw) -> OllamaClient:
    return OllamaClient(server.url, "qwen-test:3b", session=plain_session(), **kw)


def test_returns_validated_object_and_meta(server):
    server.add("/api/chat", body=chat_reply(GOOD))
    out, meta = client(server).chat_structured("sys", "usr", ReviewAnalysisOut)
    assert out.theme == "bugs_reliability" and out.severity == "high"
    assert (meta.eval_tokens, meta.prompt_tokens, meta.attempts) == (77, 123, 1)


def test_request_contains_schema_constraint_messages_and_options(server):
    server.add("/api/chat", body=chat_reply(GOOD))
    client(server, temperature=0.0, num_ctx=2048).chat_structured("SYSTEM", "USER", ReviewAnalysisOut)
    sent = json.loads(server.requests[0]["body"])
    assert sent["model"] == "qwen-test:3b" and sent["stream"] is False
    assert sent["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]
    assert sent["options"] == {"temperature": 0.0, "num_ctx": 2048}
    props = sent["format"]["properties"]
    assert set(props["theme"]["enum"]) >= {"bugs_reliability", "no_issue"}
    # Generation order matters: the model must state the problem BEFORE choosing a theme/severity.
    order = list(props)
    assert order.index("reports_problem") < order.index("problem_statement") < order.index("issue_label") < order.index("theme") < order.index("severity")
    assert set(sent["format"]["required"]) == set(GOOD)  # every key must be produced


def test_invalid_output_is_retried_with_higher_temperature(server):
    server.add("/api/chat", body=chat_reply("this is not json"))
    server.add("/api/chat", body=chat_reply({**GOOD, "theme": "made_up_theme"}))  # fails schema
    server.add("/api/chat", body=chat_reply(GOOD))
    out, meta = client(server, max_retries=2).chat_structured("s", "u", ReviewAnalysisOut)
    assert meta.attempts == 3 and out.theme == "bugs_reliability"
    temps = [json.loads(r["body"])["options"]["temperature"] for r in server.requests]
    assert temps == [0.0, 0.3, 0.6]


def test_gives_up_after_bounded_retries(server):
    server.add("/api/chat", body=chat_reply("{}"))
    with pytest.raises(LLMOutputError):
        client(server, max_retries=1).chat_structured("s", "u", ReviewAnalysisOut)
    assert len(server.requests) == 2


def test_server_error_is_retried(server):
    server.add("/api/chat", status=500, body="boom")
    server.add("/api/chat", body=chat_reply(GOOD))
    _, meta = client(server, max_retries=1).chat_structured("s", "u", ReviewAnalysisOut)
    assert meta.attempts == 2


def test_unknown_model_raises_model_missing_with_pull_command(server):
    server.add("/api/chat", status=404, body='{"error":"model not found"}')
    with pytest.raises(LLMModelMissing, match="ollama pull qwen-test:3b"):
        client(server).chat_structured("s", "u", ReviewAnalysisOut)


def test_ollama_down_raises_unavailable_immediately():
    c = OllamaClient("http://127.0.0.1:9", "m", session=plain_session())
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        c.version()
    with pytest.raises(LLMUnavailable):
        c.chat_structured("s", "u", ReviewAnalysisOut)


def test_ensure_ready_checks_version_and_model_presence(server):
    server.add("/api/version", body={"version": "0.40.5"})
    server.add("/api/tags", body={"models": [{"name": "qwen-test:3b"}, {"name": "llama3.2:latest"}]})
    assert client(server).ensure_ready() == "0.40.5"


def test_ensure_ready_reports_missing_model(server):
    server.add("/api/version", body={"version": "0.40.5"})
    server.add("/api/tags", body={"models": [{"name": "other:7b"}]})
    with pytest.raises(LLMModelMissing, match="ollama pull qwen-test:3b"):
        client(server).ensure_ready()


def test_untagged_model_name_matches_latest(server):
    server.add("/api/version", body={"version": "1"})
    server.add("/api/tags", body={"models": [{"name": "llama3.2:latest"}]})
    assert OllamaClient(server.url, "llama3.2", session=plain_session()).ensure_ready() == "1"


def test_model_output_fields_are_trimmed_and_none_becomes_empty():
    out = ReviewAnalysisOut.model_validate({**GOOD, "issue_label": "  spaced   out  ", "feature_request": None, "supporting_quote": "q" * 900})
    assert out.issue_label == "spaced out" and out.feature_request == "" and len(out.supporting_quote) == 400


@pytest.mark.parametrize("raw,expected", [
    ("passkey_popup_stuck", "passkey popup stuck"),          # snake_case seen from the real model
    ("javascript-crash", "javascript crash"),                # kebab-case seen from the real model
    ("  Autofill   FAILS. ", "autofill fails"),
    ('"vault export missing"', "vault export missing"),
    ("None", ""), ("N/A", ""), ("null", ""), ("no issue", ""),
])
def test_issue_label_is_normalised_to_plain_lowercase_words(raw, expected):
    assert ReviewAnalysisOut.model_validate({**GOOD, "issue_label": raw}).issue_label == expected


def test_literal_none_in_free_text_fields_becomes_empty():  # seen on a real review: feature_request == "None"
    out = ReviewAnalysisOut.model_validate({**GOOD, "feature_request": "None", "problem_statement": "n/a"})
    assert out.feature_request == "" and out.problem_statement == ""


# ----------------------------------------------------------------------------- deterministic consistency rules


def test_praise_only_review_cannot_carry_an_invented_problem():
    # Seen on real reviews with prompt v2: a long praise-only review came back as theme=bugs_reliability, label "crashes".
    raw = ReviewAnalysisOut.model_validate({**GOOD, "reports_problem": False, "sentiment": "positive"})
    final, fixes = raw.normalised()
    assert (final.theme, final.severity, final.issue_label, final.problem_statement) == ("no_issue", "none", "", "")
    assert set(fixes) == {"no_problem:theme", "no_problem:severity", "no_problem:issue_label", "no_problem:problem_statement"}
    assert raw.theme == "bugs_reliability"  # the raw answer object is not mutated


def test_a_reported_problem_is_never_filed_under_no_issue():
    raw = ReviewAnalysisOut.model_validate({**GOOD, "reports_problem": True, "theme": "no_issue"})
    final, fixes = raw.normalised()
    assert final.theme == "other" and fixes == ["problem_without_theme:other"]


def test_consistent_output_passes_through_untouched():
    raw = ReviewAnalysisOut.model_validate(GOOD)
    final, fixes = raw.normalised()
    assert final is raw and fixes == []
    ok_praise = ReviewAnalysisOut.model_validate({**GOOD, "reports_problem": False, "theme": "no_issue", "severity": "none",
                                                  "issue_label": "", "problem_statement": ""})
    assert ok_praise.normalised()[1] == []

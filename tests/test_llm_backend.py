"""
Tests for the local-LLM backend selection.

Written because the failure mode here is silence, twice over:

  - llama-server answers 404 on every `/api/...` path, so code that only knows
    ollama's dialect finds "no LLM" on a machine where one is loaded on the GPU;
  - the reverse — a level mapped to a model that is not installed — is how a
    whole eight-hour build graded itself with the rule-based teacher while the
    log said "hybrid".

Neither raises. Both are measured here with a fake HTTP layer, so no server is
needed to run the suite.

Run with:  python3 -m pytest tests/test_llm_backend.py -v
"""
import io
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dynamic_model import llm_backend as lb


class FakeHTTP:
    """
    Stands in for urllib.request.urlopen. `routes` maps a URL fragment to the
    JSON to answer with; anything unrouted raises, the way an unreachable
    server does. Every POST body is kept so a test can check the dialect.
    """

    def __init__(self, routes):
        self.routes = routes
        self.posts  = []

    def __call__(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if getattr(req, "data", None):
            self.posts.append((url, json.loads(req.data)))
        for fragment, payload in self.routes.items():
            if fragment in url:
                body = json.dumps(payload).encode()
                return _Resp(body)
        raise OSError(f"connection refused: {url}")


class _Resp(io.BytesIO):
    def __enter__(self):  return self
    def __exit__(self, *a): return False


LLAMA_MODELS  = {"/v1/models": {"data": [{"id": "qwen3-vl"}]}}
OLLAMA_TAGS   = {"/api/tags": {"models": [{"name": "physisml:latest"},
                                          {"name": "gemma4:latest"}]}}
LLAMA_REPLY   = {"/v1/chat/completions":
                 {"choices": [{"message": {"content": " ++ "}}]}}
OLLAMA_REPLY  = {"/api/generate": {"response": "+++"}}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """A probe result cached by one test must not leak into the next."""
    lb.reset_cache()
    monkeypatch.setattr(lb, "ENV_MODEL", None)
    monkeypatch.setattr(lb, "ENV_BACKEND", "auto")
    yield
    lb.reset_cache()


def _install(monkeypatch, *route_dicts):
    routes = {}
    for d in route_dicts:
        routes.update(d)
    fake = FakeHTTP(routes)
    monkeypatch.setattr(lb.urllib.request, "urlopen", fake)
    return fake


def test_llama_cpp_wins_when_both_servers_answer(monkeypatch):
    _install(monkeypatch, LLAMA_MODELS, OLLAMA_TAGS)
    b = lb.detect()
    assert b.kind == "llamacpp"
    assert b.models == ["qwen3-vl"]


def test_ollama_is_used_when_llama_cpp_is_silent(monkeypatch):
    _install(monkeypatch, OLLAMA_TAGS)
    b = lb.detect()
    assert b.kind == "ollama"
    assert "gemma4:latest" in b.models


def test_nothing_reachable_is_none_not_an_exception(monkeypatch):
    _install(monkeypatch)          # every URL refuses
    assert lb.detect() is None
    assert lb.generate("anything") == ""


def test_the_backend_can_be_switched_off(monkeypatch):
    _install(monkeypatch, LLAMA_MODELS, OLLAMA_TAGS)
    assert lb.detect(force="off") is None


def test_ollama_refuses_a_model_it_does_not_host(monkeypatch):
    """
    The veto that matters: substituting a different grader would quietly change
    what a training run measures, so an uninstalled name resolves to None and
    the caller falls back to the rule-based teacher.
    """
    _install(monkeypatch, OLLAMA_TAGS)
    b = lb.detect()
    assert lb.resolve_model("llama3.2:latest", b) is None
    assert lb.resolve_model("gemma4", b) == "gemma4:latest"   # tag-insensitive


def test_llama_cpp_uses_what_it_hosts_whatever_was_asked(monkeypatch):
    """One loaded model, no choice: the level mapping is informational there."""
    _install(monkeypatch, LLAMA_MODELS)
    b = lb.detect()
    assert lb.resolve_model("llama3.2:latest", b) == "qwen3-vl"


def test_the_student_is_never_auto_selected_as_its_own_teacher(monkeypatch):
    """`physisml:latest` is first in `ollama list` on the dev machine."""
    _install(monkeypatch, OLLAMA_TAGS)
    b = lb.detect()
    assert lb.resolve_model(None, b) == "gemma4:latest"
    # explicitly asked for, it is still allowed
    assert lb.resolve_model("physisml", b) == "physisml:latest"


def test_each_backend_is_addressed_in_its_own_dialect(monkeypatch):
    fake = _install(monkeypatch, LLAMA_MODELS, LLAMA_REPLY)
    assert lb.generate("q", system="s", max_tokens=8) == "++"
    url, body = fake.posts[-1]
    assert url.endswith("/v1/chat/completions")
    assert body["messages"][0]["role"] == "system"
    assert body["max_tokens"] == 8

    lb.reset_cache()
    fake = _install(monkeypatch, OLLAMA_TAGS, OLLAMA_REPLY)
    assert lb.generate("q", system="s", model="gemma4:latest") == "+++"
    url, body = fake.posts[-1]
    assert url.endswith("/api/generate")
    assert body["prompt"] == "q" and body["system"] == "s"
    assert body["options"]["num_predict"] == 8


def test_a_reasoning_scratchpad_is_stripped(monkeypatch):
    _install(monkeypatch, LLAMA_MODELS,
             {"/v1/chat/completions": {"choices": [{"message": {
                 "content": "<think>the word is there\nmaybe ++</think>\n+"}}]}})
    assert lb.generate("q") == "+"


def test_a_malformed_answer_is_empty_not_a_crash(monkeypatch):
    _install(monkeypatch, LLAMA_MODELS, {"/v1/chat/completions": {"choices": []}})
    assert lb.generate("q") == ""


def test_the_hybrid_teacher_reports_the_server_it_resolved(monkeypatch):
    """The message the old code got wrong: it said only 'ollama not available'."""
    _install(monkeypatch, OLLAMA_TAGS)
    from dynamic_model.hybrid_teacher import LLMEvaluator
    ev = LLMEvaluator("llama3.2:latest")
    assert ev.is_available() is False
    assert "not installed" in ev.status() and "gemma4:latest" in ev.status()

    lb.reset_cache()
    _install(monkeypatch, LLAMA_MODELS)
    ev = LLMEvaluator("llama3.2:latest")
    assert ev.is_available() is True
    assert ev.model == "qwen3-vl"
    assert "llama.cpp" in ev.status()

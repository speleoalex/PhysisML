"""
llm_backend — one way to reach a local LLM, whichever server is running.

Two servers can host the grader (HybridTeacher) and, later, the ontology
oracle of the autonomy loop:

  * llama.cpp's `llama-server` — OpenAI dialect, `/v1/models`,
    `/v1/chat/completions`. Hosts exactly ONE model, the one it was started
    with, and usually keeps it resident on the GPU.
  * ollama — its own dialect, `/api/tags`, `/api/generate`. Hosts many models
    and loads them on demand.

They are not interchangeable at the HTTP level: llama-server answers 404 on
every `/api/...` path. This module hides that difference behind `generate()`
and picks the server by probing, so nothing upstream has to care.

Order of preference in `auto`: llama.cpp first (single resident model, no load
latency), ollama second. Both silent-fail to "" — a grader that cannot reach
its LLM must degrade to the rule-based teacher, never crash a training run.

Environment:
  PHYSISML_LLM_BACKEND   auto (default) | llamacpp | ollama | off
  LLAMA_SERVER_BASE      default http://localhost:8080
  OLLAMA_BASE            default http://localhost:11434
  PHYSISML_LLM_MODEL     force one model regardless of the level mapping
                         (PHYSISML_OLLAMA_MODEL is still read, older name)

Usage:
  from dynamic_model import llm_backend
  b = llm_backend.detect()                       # None when nothing answers
  model = llm_backend.resolve_model("llama3.2:latest", b)
  text  = llm_backend.generate("2+2?", system="Reply with a digit.",
                               model=model, backend=b, max_tokens=4)
"""
import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

LLAMA_BASE  = os.environ.get("LLAMA_SERVER_BASE", "http://localhost:8080").rstrip("/")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
ENV_BACKEND = (os.environ.get("PHYSISML_LLM_BACKEND") or "auto").strip().lower()
ENV_MODEL   = (os.environ.get("PHYSISML_LLM_MODEL")
               or os.environ.get("PHYSISML_OLLAMA_MODEL") or None)

PROBE_TIMEOUT = 3

# Reasoning models wrap their scratchpad in these. The grader wants the symbol
# that follows, not the deliberation that precedes it.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Never auto-selected as a teacher: PhysisML itself is the student. It is the
# first entry in `ollama list` on this machine, so "just take the first model"
# handed the grading job to the 23.6M-param model being graded — it answered
# the probe with "non lo so.". Only an explicit request can select it.
_NEVER_AUTO = ("physisml",)


class Backend:
    """A reachable LLM server: which dialect it speaks and what it hosts."""

    def __init__(self, kind: str, base: str, models: List[str]):
        self.kind   = kind          # "llamacpp" | "ollama"
        self.base   = base
        self.models = models

    def __repr__(self) -> str:
        return f"Backend({self.kind}, {self.base}, {len(self.models)} model(s))"

    def describe(self) -> str:
        where = "llama.cpp" if self.kind == "llamacpp" else "ollama"
        return f"{where} @ {self.base}"


def _get_json(url: str, timeout: int = PROBE_TIMEOUT) -> Optional[dict]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _post_json(url: str, payload: dict, timeout: int) -> Optional[dict]:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def probe_llamacpp(base: str = None) -> Optional[Backend]:
    """llama-server, asked what it is serving. One model, always."""
    base = (base or LLAMA_BASE).rstrip("/")
    data = _get_json(f"{base}/v1/models")
    if not isinstance(data, dict):
        return None
    ids = [m.get("id") or m.get("name") for m in data.get("data", [])
           if isinstance(m, dict)]
    ids = [i for i in ids if i]
    if not ids:                                  # answered, but hosts nothing
        return None
    return Backend("llamacpp", base, ids)


def probe_ollama(base: str = None) -> Optional[Backend]:
    base = (base or OLLAMA_BASE).rstrip("/")
    data = _get_json(f"{base}/api/tags")
    if not isinstance(data, dict) or "models" not in data:
        return None
    names = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
    return Backend("ollama", base, [n for n in names if n])


_CACHED = "unset"   # None is a real answer here ("nothing reachable")


def detect(force: str = None, use_cache: bool = True) -> Optional[Backend]:
    """
    The server to talk to, or None. Probes once per process: the check costs a
    round trip and is called from inside training loops.
    """
    global _CACHED
    want = (force or ENV_BACKEND or "auto").lower()
    if use_cache and force is None and _CACHED != "unset":
        return _CACHED

    if want in ("off", "none", "disabled"):
        found = None
    elif want in ("llamacpp", "llama.cpp", "llama_cpp", "llama"):
        found = probe_llamacpp()
    elif want == "ollama":
        found = probe_ollama()
    else:
        found = probe_llamacpp() or probe_ollama()

    if force is None:
        _CACHED = found
    return found


def reset_cache() -> None:
    """Forget the probe result — for tests, and after starting a server."""
    global _CACHED
    _CACHED = "unset"


def resolve_model(preferred: str = None, backend: Backend = None) -> Optional[str]:
    """
    The model name to send, or None when the request cannot be honoured.

    llama.cpp hosts one model and the caller does not get to choose: a level
    mapping that names something else is informational, not a veto. ollama
    hosts many, so a name that is not installed IS a veto — substituting a
    different grader would silently change what a training run measures.
    """
    backend = backend or detect()
    if backend is None:
        return None
    forced = ENV_MODEL or preferred
    if backend.kind == "llamacpp":
        if forced and forced in backend.models:
            return forced
        return _first_usable(backend.models)
    # ollama: match on the name before the tag, the way the tags list reports it
    if not forced:
        return _first_usable(backend.models)
    stem = forced.split(":")[0]
    for name in backend.models:
        if name == forced or name.split(":")[0] == stem:
            return name
    return None


def _first_usable(models: List[str]) -> Optional[str]:
    """The first hosted model that is not the student itself."""
    for name in models:
        if name.split(":")[0].lower() not in _NEVER_AUTO:
            return name
    return None


def generate(prompt: str, system: str = "", model: str = None,
             timeout: int = 15, max_tokens: int = 8,
             temperature: float = 0.0, backend: Backend = None) -> str:
    """
    One completion, deterministic by default. Returns "" on any failure —
    unreachable server, unknown model, malformed answer.
    """
    backend = backend or detect()
    if backend is None:
        return ""
    model = model or resolve_model(None, backend)
    if not model:
        return ""

    if backend.kind == "llamacpp":
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})
        data = _post_json(f"{backend.base}/v1/chat/completions", {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }, timeout)
        if not isinstance(data, dict):
            return ""
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""
    else:
        data = _post_json(f"{backend.base}/api/generate", {
            "model":   model,
            "prompt":  prompt,
            "system":  system,
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout)
        if not isinstance(data, dict):
            return ""
        text = data.get("response", "") or ""

    return _THINK_RE.sub("", text).strip()


def status_line(preferred: str = None) -> str:
    """One line for a log: what was found, what will be used."""
    b = detect()
    if b is None:
        return "no local LLM reachable (llama.cpp, then ollama, both silent)"
    model = resolve_model(preferred, b)
    if model is None:
        return (f"{b.describe()}: '{preferred}' not installed — "
                f"available: {', '.join(b.models) or 'none'}")
    return f"{b.describe()}: {model}"


if __name__ == "__main__":
    import sys
    print(status_line(sys.argv[1] if len(sys.argv) > 1 else None))
    b = detect()
    if b:
        print(repr(b))
        print("test:", generate("Rispondi solo con una parola: capitale d'Italia?",
                                max_tokens=8) or "(no answer)")

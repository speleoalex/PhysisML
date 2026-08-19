"""
HybridTeacher — combines LocalTeacher (prompt generation) + local LLM (evaluation).

Architecture:
  - LocalTeacher: generates the next prompt and expected (deterministic, free)
  - OllamaEvaluator: rates the model response (intelligent, local, GPU)

The LLM receives a very short prompt (~60 tokens) and replies only with the
feedback symbol (+++, ++, +, = or -). Much faster than also generating the question.

Recommended models per level:
  L3-L4: qwen2:0.5b-instruct-q4_0  (352MB, ~100ms/evaluation)
  L5-L6: llama3.2:latest            (2GB, ~300ms/evaluation)
  L7+:   qwen3:8b                   (5GB, ~500ms/evaluation)

Usage:
  teacher = HybridTeacher(lang='it', level=3)
  result = teacher.turn(last_prompt='di: il cane', last_response='cane la', turn=2)
"""
import json
import os
import re
import urllib.request
import urllib.error
from typing import Optional

from dynamic_model.local_teacher import LocalTeacher

# Level → ollama model mapping (configurable)
LEVEL_TO_MODEL = {
    0: "qwen2:0.5b-instruct-q4_0",
    1: "qwen2:0.5b-instruct-q4_0",
    2: "llama3.2:latest",   # qwen2:0.5b too weak for strict eval
    3: "llama3.2:latest",
    4: "llama3.2:latest",
    5: "llama3.2:latest",
    6: "llama3.2:latest",
    7: "qwen3:8b",
}
DEFAULT_MODEL = "llama3.2:latest"

OLLAMA_URL = "http://localhost:11434/api/generate"

FEEDBACK_MAP = {
    "+++": 1.0, "++": 0.8, "+": 0.5, "=": 0.0, "-": -0.8,
}

# Evaluation system prompt — strict, anchored to expected answer
EVAL_SYSTEM = """You are a STRICT evaluator of Italian responses for a learning AI model.
Your task is to check whether the response contains the EXPECTED WORD OR PHRASE.

SCALE (apply top-down — use the first that matches):
  +++  The response matches the expected almost exactly: target word + correct article + terminator (.!?)
       EXAMPLE: expected="il cane!" response="il cane!" → +++
  ++   The expected CONTENT WORD (noun/verb) is clearly present with the correct article.
       EXAMPLE: expected="il cane!" response="il cane la di!" → ++
  +    The expected CONTENT WORD is present but the article or required structure is missing.
       EXAMPLE: expected="il cane!" response="cane! la ba" → +
       EXAMPLE: expected="la mamma!" response="mamma mamma!" → + (word present, repeated, no article)
       EXAMPLE: expected="il cane!" response="cane cane!" → + (word present even if repeated)
  =    The expected content word is NOT present. Only function words (il, la, di, che, a, e, è).
       EXAMPLE: expected="il cane!" response="il la di il! la" → =
  -    Incomprehensible output: no recognisable Italian word or only noise (ba, da, ta, ra...).
       EXAMPLE: expected="il cane!" response="ba da ta ra!" → -

CRITICAL RULES:
- Function words "il", "la", "di", "che", "a", "e", "è", "un", "una" ALONE do not count.
  A response like "il la di il la di" must receive =, not + or ++.
- Only the presence of the CONTENT WORD (expected noun or verb) qualifies for + or higher.
- Response with more than 10 words: lower the rating by one level (too long for the level).

Reply ONLY with one of the symbols: +++ ++ + = -
Do NOT add text, explanations, or punctuation. Only the symbol."""


def _ollama_generate(model: str, prompt: str, system: str = "",
                     timeout: int = 15) -> str:
    """Call ollama API and return text response."""
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": 0.0,  # deterministic
            "num_predict": 8,    # only need 1-3 chars
        }
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception:
        return ""


def _parse_feedback(text: str) -> str:
    """Extract feedback symbol from LLM response."""
    text = text.strip()
    for sym in ("+++", "++", "+", "=", "-"):
        if sym in text:
            return sym
    return "="  # default neutral if unclear


class OllamaEvaluator:
    """
    Intelligent response evaluator using a local LLM via ollama.
    Only evaluates (+++/++/+/=/−) — does not generate prompts.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model   = model
        self._online = None  # cached connectivity check

    def is_available(self) -> bool:
        """Check if ollama is running and model is available."""
        if self._online is not None:
            return self._online
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                self._online = any(
                    self.model.split(":")[0] in m for m in models
                )
                return self._online
        except Exception:
            self._online = False
            return False

    def evaluate(self, prompt: str, response: str,
                 expected: str = "", level: int = 0) -> dict:
        """
        Evaluate model response and return feedback dict.

        Returns: {"feedback": "++", "feedback_symbol": "++",
                  "commento": "...", "feedback_value": 0.8}
        """
        # Deterministic exact-match shortcut: if the response equals the
        # expected answer (ignoring EOS marker and whitespace), award +++
        # directly. Bypass the LLM which sometimes misreads exact matches
        # as "echo" when the prompt contains the retry prefix.
        _resp_clean = response.replace("<|EOS|>", "").strip().rstrip("!.?").strip().lower()
        _exp_clean  = expected.strip().rstrip("!.?").strip().lower()
        if _exp_clean and _resp_clean == _exp_clean:
            return {
                "feedback_symbol": "+++",
                "feedback":        FEEDBACK_MAP["+++"],
                "commento":        f"Exact match: {expected}",
            }

        # Build compact evaluation prompt
        # Make the target word explicit so the LLM knows exactly what to look for
        lines = [f"Question: {prompt}"]
        if expected:
            lines.append(f"EXPECTED response (the correct one): {expected}")
            # Extract the content word (first non-article noun) as an anchor
            _words = [w.lower().strip("!.?:,") for w in expected.split()
                      if w.lower().strip("!.?:,") not in
                      ("il", "la", "lo", "i", "le", "gli", "un", "una", "di", "del",
                       "della", "dei", "delle", "degli", "a", "e", "è")]
            if _words:
                lines.append(f"Key word to look for: '{_words[0]}'")
        lines.append(f"Model response: {response}")
        lines.append("\nRating (symbol only):")
        eval_prompt = "\n".join(lines)

        raw = _ollama_generate(self.model, eval_prompt, EVAL_SYSTEM, timeout=20)
        symbol = _parse_feedback(raw)

        # Deterministic sanity check: if the target content word is absent from
        # the response, the LLM cannot legitimately give ++ or +++.
        # This catches cases where llama3.2 ignores the key word instruction.
        if expected:
            _stop = {"il","la","lo","le","gli","un","una","di","del","della",
                     "dei","delle","degli","a","e","è","in","per","non","che","si"}
            _kws = [w.lower().strip("!.?:, ") for w in expected.split()
                    if w.lower().strip("!.?:, ") not in _stop and len(w.strip("!.?:, ")) > 1]
            if _kws:
                resp_lower = response.lower()
                target_present = any(kw in resp_lower for kw in _kws)
                if not target_present and symbol in ("+++", "++", "+"):
                    # Downgrade: LLM awarded credit for a word that isn't there
                    symbol = "="
                elif target_present and symbol == "=":
                    # Upgrade: LLM missed the content word that IS present
                    symbol = "+"

        # Brief comment based on symbol
        comments = {
            "+++": f"{expected or 'response'} clearly present.",
            "++":  "Almost correct, good attempt.",
            "+":   "Something relevant present.",
            "=":   "Confused output, some Italian words.",
            "-":   "Unrecognisable.",
        }

        return {
            "feedback_symbol": symbol,
            "feedback":        FEEDBACK_MAP.get(symbol, 0.0),
            "commento":        comments.get(symbol, ""),
        }


class HybridTeacher:
    """
    Teacher that combines:
    - LocalTeacher for prompt generation (rule-based, deterministic)
    - OllamaEvaluator for response evaluation (LLM, intelligent)

    Falls back to LocalTeacher evaluation if ollama is unavailable.
    """

    def __init__(self, lang: str, level: int,
                 ollama_model: str = None):
        self.lang    = lang
        self.level   = level

        # Rule-based prompt generator
        self.local   = LocalTeacher(lang, level)

        # LLM evaluator
        model = ollama_model or LEVEL_TO_MODEL.get(level, DEFAULT_MODEL)
        self.evaluator = OllamaEvaluator(model)

        self._use_llm = self.evaluator.is_available()
        self._last_expected = ""  # stores expected from previous turn for evaluation
        if self._use_llm:
            print(f"  HybridTeacher: {model} for evaluation (online)")
        else:
            print(f"  HybridTeacher: fallback to LocalTeacher (ollama not available)")

    def turn(self, last_prompt: str = "", last_response: str = "",
             turn: int = 1) -> dict:
        """
        Generate next teaching turn.
        - LocalTeacher generates the next prompt and expected
        - OllamaEvaluator rates the previous response using the STORED expected
          (from the previous turn, not the new one)
        """
        # 1. Get the rule-based evaluation of last_response using the CURRENT
        #    target (before advancing to the next prompt).
        #    We use self._last_expected which was stored in the previous turn.
        rule_result = self.local.turn(last_prompt, last_response, turn)

        # 2. Override feedback with LLM evaluation if available
        if turn > 1 and last_response and last_prompt and self._use_llm:
            # Use the expected stored from the PREVIOUS turn (current evaluation target)
            expected_for_eval = getattr(self, '_last_expected', last_prompt)
            llm_eval = self.evaluator.evaluate(
                prompt=last_prompt,
                response=last_response,
                expected=expected_for_eval,
                level=self.level,
            )
            rule_result["feedback_symbol"] = llm_eval["feedback_symbol"]
            rule_result["feedback"]        = llm_eval["feedback"]
            rule_result["commento"]        = llm_eval["commento"]

        # Store this turn's expected for next turn's evaluation
        self._last_expected = rule_result.get("expected", "")

        return rule_result

    # Compatibility helpers
    def has_config(self, lang: str, level: int) -> bool:
        return self.local.has_config(lang, level)

    def __repr__(self) -> str:
        llm_info = f"LLM={self.evaluator.model}" if self._use_llm else "LLM=off"
        return f"HybridTeacher(lang={self.lang}, level={self.level}, {llm_info})"

"""
LocalTeacher — rule-based teacher, no external APIs.

Reads configuration from training_files/{lang}/{level}/local_teacher.json
and implements the same interface as TutorAgent (produces JSON dictionaries
compatible with the teaching loop in train_curriculum.py).

Advantages over Claude:
  - No API cost for simple levels (L0, L1)
  - 100× faster (no network latency)
  - Deterministic and consistent evaluation
  - Configurable for any language by changing only the JSON

Usage:
    teacher = LocalTeacher(lang="it", level=0)
    result  = teacher.turn(last_prompt="di ma", last_response="ma pa", turn=2)
    # result = {"feedback": "++", "commento": "...", "next_prompt": "...",
    #           "expected": "...", "step": "A", "feedback_symbol": "++"}
"""
import json
import os
import re
import random
from typing import Optional


FEEDBACK_MAP = {
    "+++": 1.0,
    "++":  0.8,
    "+":   0.5,
    "=":   0.0,
    "-":  -0.8,
}

# A question the model asks: 'cosa è un ragno?'. Matched loosely on purpose —
# a model that has not yet learned to close a question still asked one.
_QUESTION_RE = re.compile(
    r"\b(?:cosa|cos|che)\s+(?:è|e)\s+(?:un|una|uno|il|la|lo|l)?\s*"
    r"([a-zàèéìòù]{3,})",
    re.IGNORECASE)

# Reward for asking about something genuinely unknown, and the penalty for
# asking about something already explained. The ASYMMETRY is the whole point:
# a model rewarded for every question learns to always ask, which is a tic and
# not curiosity. The gate in AffectModulator only makes the question happen;
# this is what decides whether it was worth asking.
ASK_REWARD  =  1.0
ASK_PENALTY = -0.3


class LocalTeacher:

    def __init__(self, lang: str, level: int,
                 retry_prefix: Optional[str] = None):
        self.lang  = lang
        self.level = level

        # Load config
        config_path = os.path.join(
            "training_files", lang, str(level), "local_teacher.json"
        )
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"local_teacher.json not found: {config_path}\n"
                f"Create it or use --tutor-model sonnet/haiku for this level."
            )
        with open(config_path, encoding="utf-8") as f:
            self.cfg = json.load(f)

        # retry_prefix override: the config value is what the validated build
        # used, but at L0-L3 it doubles the text ('{prompt} {prompt}. '), and
        # at those levels the prompt IS the answer ('di: il cane' -> 'il
        # cane!'). That puts 'il cane il cane' in the corpus as a valid
        # sequence. Exposed so an arm can replace it without editing the
        # shared config. Pass '' to disable the prefix entirely.
        if retry_prefix is not None:
            self.cfg["retry_prefix"] = retry_prefix

        self.steps     = self.cfg["steps"]
        self.step_keys = list(self.steps.keys())   # ["A", "B", "C", ...]
        self.eval_cfg  = self.cfg.get("evaluation", {})

        # State
        self.current_step       = self.step_keys[0]
        self.current_target_idx = 0
        self.consecutive_ok     = 0   # successes on current target
        self.consecutive_fail   = 0   # failures on current target
        self.total_turns        = 0

        # Curiosity mode: only levels whose config asks for it. The names the
        # model may legitimately not know come from the lexicon, and half of
        # them are marked `probe` and withheld even here — those are what
        # scripts/curiosity_rate.py measures on.
        self.curiosity_enabled = bool(self.cfg.get("curiosity", False))
        self.unknown  = {}     # noun -> (article, class), teachable here
        self.known    = {}     # noun -> (article, class), from the main lexicon
        self.explained = set() # nouns already answered in THIS session
        if self.curiosity_enabled:
            self.unknown, self.known = self._load_nouns()

    def _load_nouns(self) -> tuple:
        """(unknown, known), each noun -> (article, class).

        `unknown` are the held-out nouns this level may teach. Probe nouns are
        excluded: they exist so curiosity can be measured on words the model
        was never taught, and teaching them here would turn the measurement
        into a recall test.

        `known` is the ordinary lexicon — needed because a question about a
        name the model already learned is the case this level exists to
        discourage, and it has to be recognised to be penalised.
        """
        path = os.path.join("training_files", self.lang, "lexicon.json")
        if not os.path.exists(path):
            return {}, {}
        try:
            with open(path, encoding="utf-8") as f:
                lex = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}, {}
        unknown = {n["w"]: (n["art"], n["cls"])
                   for n in lex.get("unknown_nouns", [])
                   if not n.get("probe") and n.get("cls")}
        onto = lex.get("ontology", {})
        kind_class = onto.get("kind_class", {})
        known = {}
        for n in lex.get("nouns", []):
            cls = n.get("cls") or kind_class.get(n.get("kind", ""))
            if cls:
                known[n["w"]] = (n["art"], cls)
        return unknown, known

    def _asked_about(self, response: str):
        """The noun the model asked about, or None if it did not ask."""
        if not response:
            return None
        m = _QUESTION_RE.search(response)
        return m.group(1).lower() if m else None

    def _handle_question(self, noun: str) -> dict:
        """Answer a question worth asking, penalise one that was not.

        Asking about an unknown name earns the full reward AND the answer: the
        next prompt becomes the fact, so the question buys the model something.
        Asking about a name it already has — one explained earlier in this
        session, or an ordinary lexicon word — earns a penalty; otherwise the
        cheapest policy is to ask about everything, forever, and that is a tic
        rather than curiosity.
        """
        teachable = noun in self.unknown and noun not in self.explained
        art, cls = (self.unknown.get(noun) or self.known[noun])
        statement = f"{art} {noun} è {cls}."

        if not teachable:
            return {
                "feedback_symbol": "=",
                "feedback":        ASK_PENALTY,
                "commento":        f"{noun}: lo sai già",
                # The definite form, the same shape L11 step A uses
                # ('cosa è la sedia?'), so the retry is a prompt the model has
                # actually been trained on.
                "next_prompt":     f"cosa è {art} {noun}?",
                "expected":        statement,
                "step":            self.current_step,
                "mode":            "curiosity_repeat",
            }

        self.explained.add(noun)
        return {
            "feedback_symbol": "+++",
            "feedback":        ASK_REWARD,
            "commento":        f"bella domanda: {noun}",
            "next_prompt":     f"{art} {noun} è {cls}",
            "expected":        statement,
            "step":            self.current_step,
            "mode":            "curiosity_answer",
        }

    # ------------------------------------------------------------------
    # Main interface (same as TutorAgent)
    # ------------------------------------------------------------------

    def turn(self, last_prompt: str = "", last_response: str = "",
             turn: int = 1) -> dict:
        """
        Evaluate last response and return next teaching turn.
        Returns dict with same keys as TutorAgent:
          feedback, feedback_symbol, commento, next_prompt, expected, step
        """
        self.total_turns += 1

        # The model asked something. Handle it INSTEAD of grading the response
        # against the current target: a question is not a failed attempt at the
        # target, and _evaluate would score it '=' and teach nothing.
        if self.curiosity_enabled and turn > 1:
            asked = self._asked_about(last_response)
            if asked in self.unknown or asked in self.known:
                return self._handle_question(asked)

        step_cfg  = self.steps[self.current_step]
        targets   = step_cfg["targets"]
        target    = targets[self.current_target_idx % len(targets)]

        # ── Evaluate last response ────────────────────────────────────
        if turn == 1:
            # First turn — no evaluation
            fb_symbol = None
            commento  = ""
        else:
            fb_symbol, commento = self._evaluate(last_response, target, step_cfg)
            self._update_state(fb_symbol, step_cfg, targets)

        # ── Build next prompt ─────────────────────────────────────────
        step_cfg_now = self.steps[self.current_step]
        targets_now  = step_cfg_now["targets"]
        target_now   = targets_now[self.current_target_idx % len(targets_now)]

        next_prompt, expected = self._build_prompt(target_now, step_cfg_now,
                                                    fb_symbol)

        result = {
            "feedback_symbol": fb_symbol,
            "feedback":        FEEDBACK_MAP.get(fb_symbol, 0.0),
            "commento":        commento,
            "next_prompt":     next_prompt,
            "expected":        expected,
            "step":            self.current_step,
        }
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, response: str, target, step_cfg: dict) -> tuple:
        """Return (feedback_symbol, commento)."""
        r = response.lower().strip()

        # Determine what to look for
        if isinstance(target, dict):
            noun       = target.get("noun", "")
            article    = target.get("article", "")
            verb       = target.get("verb", "")
            adjective  = target.get("adjective", "")
            obj        = target.get("object", "")
            req_art    = step_cfg.get("require_article", False)
            req_verb   = step_cfg.get("require_verb", False)
            req_adj    = step_cfg.get("require_adjective", False)
            search_word = noun
        else:
            noun = target; article = ""; verb = ""; adjective = ""; obj = ""
            req_art = False; req_verb = False; req_adj = False
            search_word = target

        max_words = step_cfg.get("max_response_words", 6)
        stop_pat  = self.eval_cfg.get("has_terminator_pattern", r"[.!?]")

        word_count  = len(r.split())
        r_compact = re.sub(r'\s+', '', r)

        def _has(word):
            if not word: return True  # not required → always pass
            # Word-boundary check (primary): prevents 'ba' matching inside 'labato'
            if bool(re.search(r'\b' + re.escape(word) + r'\b', r)):
                return True
            # Compact fallback only for words >= 4 chars to avoid single-letter matches
            if len(word) >= 4 and word in r_compact:
                return True
            return False

        has_target  = _has(noun)
        has_stop    = bool(re.search(stop_pat, r))
        has_article = not req_art  or (article and article in r.split()[:5])
        has_verb    = not req_verb or _has(verb)
        has_adj     = not req_adj  or _has(adjective)
        has_obj     = _has(obj) if obj else True
        too_long    = word_count > max_words * 2.5

        all_required = has_target and has_article and has_verb and has_adj and has_obj

        # Feedback logic
        if too_long:
            fb = "-"
        elif all_required and has_stop and word_count <= max_words:
            fb = "+++"
        elif all_required:
            fb = "++"
        elif has_target and has_article:
            fb = "++"  # partial: noun + article, missing verb/adj
        elif has_target:
            fb = "+"
        elif word_count == 0 or (word_count == 1 and r in ("!", ".", "?")):
            fb = "-"
        else:
            fb = "="

        # Build comment
        comment_key = f"comment_{fb}"
        tmpl = self.eval_cfg.get(comment_key, "")
        commento = tmpl.format(noun=noun, article=article, verb=verb,
                               adjective=adjective, target=search_word)
        if not commento:
            commento = {
                "+++": f"{noun} clearly present.",
                "++":  f"{noun} found.",
                "+":   f"Almost! I hear {noun}.",
                "=":   "Confused output.",
                "-":   "Unrecognisable.",
            }.get(fb, "")

        return fb, commento[:40]

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _update_state(self, fb: str, step_cfg: dict, targets: list) -> None:
        """Advance or retry target based on feedback.

        Only strong positives (++, +++) count for advance. The `+` feedback
        (target present but structure incomplete, e.g. noun without article)
        is an encouragement — it resets the failure counter but does NOT
        increment the consecutive-ok counter. This prevents premature
        advancement based on partial correctness.
        """
        if fb in ("+++", "++"):
            self.consecutive_ok   += 1
            self.consecutive_fail  = 0
        elif fb == "+":
            # Soft positive: encourages the model but doesn't count for advance
            self.consecutive_fail  = 0
        elif fb in ("-", "="):
            self.consecutive_fail += 1
            self.consecutive_ok    = 0

        advance_n = step_cfg.get("advance_after_successes", 3)
        retry_n   = step_cfg.get("retry_after_failures", 3)

        if self.consecutive_ok >= advance_n:
            # Move to next target in current step
            self.consecutive_ok    = 0
            self.consecutive_fail  = 0
            next_idx = self.current_target_idx + 1
            if next_idx >= len(targets):
                # Advance to next step
                self._advance_step()
            else:
                self.current_target_idx = next_idx

        elif self.consecutive_fail >= retry_n:
            # Change target (don't get stuck on impossible one)
            self.consecutive_fail = 0
            self.consecutive_ok   = 0
            self.current_target_idx = (self.current_target_idx + 1) % len(targets)

    def _advance_step(self) -> None:
        """Move to the next step (A→B→C→…). Wraps around at the end."""
        idx = self.step_keys.index(self.current_step)
        if idx < len(self.step_keys) - 1:
            self.current_step       = self.step_keys[idx + 1]
            self.current_target_idx = 0
        else:
            # Already at last step — shuffle targets to keep it varied
            self.current_target_idx = 0

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _build_prompt(self, target, step_cfg: dict,
                      last_fb: Optional[str]) -> tuple:
        """Return (next_prompt, expected)."""
        tmpl = step_cfg.get("prompt_template", "di {target}")
        retry_pfx = self.cfg.get("retry_prefix", "")

        if isinstance(target, dict):
            prompt_val   = target.get("prompt", target.get("noun", ""))
            expected_val = target.get("expected", prompt_val + "!")
        else:
            prompt_val   = target
            tmpl_exp     = step_cfg.get("expected_template", "{target}!")
            expected_val = tmpl_exp.format(target=target)

        # Format prompt
        prompt = tmpl.format(target=prompt_val, prompt=prompt_val)

        # Add retry prefix if last attempt failed
        if last_fb in ("-", "=") and retry_pfx:
            pfx    = retry_pfx.format(target=prompt_val, prompt=prompt_val)
            prompt = pfx + prompt

        # Occasionally add encouragement after success
        if last_fb in ("+++", "++") and random.random() < 0.4:
            enc    = random.choice(self.cfg.get("encouragement", []))
            prompt = enc + " " + prompt

        return prompt, expected_val

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def has_config(self, lang: str, level: int) -> bool:
        path = os.path.join("training_files", lang, str(level), "local_teacher.json")
        return os.path.exists(path)

    def __repr__(self) -> str:
        return (f"LocalTeacher(lang={self.lang}, level={self.level}, "
                f"step={self.current_step}, "
                f"target_idx={self.current_target_idx})")

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

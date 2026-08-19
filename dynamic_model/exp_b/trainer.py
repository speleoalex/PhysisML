"""
TrainerB — online training with affective system.

Each interaction:
  prompt + response + feedback → weight update + affective state update

Feedback:
  +1.0  approval      → weights change, pleasure rises
   0.0  observation   → weights do NOT change, vocab may grow
  -1.0  disapproval   → weights change (counter-gradient), pain rises

The affective system modifies logits ONLY during inference (generate()).
During training the backward is normal — no modulation.
This avoids the unstable loop: high fear → flat distribution → ...
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

import numpy as np
import torch
from splx.torch_model import DEVICE  # auto-detected best device
import torch.nn.functional as F
from typing import Optional, List

from splx.torch_model import TorchGPT, TorchAdamOptimizer
from splx.tokenizer   import BPETokenizer

from dynamic_model.exp_b.affect_state import AffectState, AffectSnapshot
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.axioms       import AxiomRegistry


import re as _re

def _clean_response(text: str) -> str:
    """
    Strip common noise artifacts from generated responses.

    Two artifacts observed in training:
      1. Leading '!' (model echoes the terminator it learned as a START token)
         e.g. '!il cane!' → 'il cane!'
      2. Echoed 'di:' from the prompt
         e.g. '!il di: il cane!' → 'il cane!'

    Only the RESPONSE portion (after prompt) should be passed here.
    Stripping is conservative: only leading noise, never content.
    """
    t = text.strip()
    # Strip leading ! and whitespace (can repeat: '!! la cane!' → 'la cane!')
    t = _re.sub(r'^[!\s]+', '', t).strip()
    # Remove 'di:' echo (with colon — always a prompt artifact, never valid Italian).
    # Also remove any preceding spurious article: 'il di: il cane!' → 'il cane!'
    t = _re.sub(r'(?:il|la|lo|le|gli|un|una)\s+di:\s*', '', t, flags=_re.IGNORECASE).strip()
    t = _re.sub(r'di:\s*', '', t, flags=_re.IGNORECASE).strip()
    # Collapse multiple spaces left by the removal
    t = _re.sub(r'\s{2,}', ' ', t).strip()
    # If stripping removed everything meaningful, return original stripped
    if not t:
        return text.strip()
    return t


class TrainerB:

    def __init__(self, model: TorchGPT, tokenizer,
                 optimizer: TorchAdamOptimizer,
                 affect_state: Optional[AffectState] = None,
                 modulator: Optional[AffectModulator] = None,
                 axiom_registry: Optional[AxiomRegistry] = None):

        self.device   = DEVICE
        self.model    = model.to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = optimizer

        # Affective system (optional — if None, behaves like TrainerA)
        self.affect   = affect_state   or AffectState()
        self.mod      = modulator      or AffectModulator(self.affect)
        self.axioms   = axiom_registry or AxiomRegistry()

        # Collega il backward hook per protezione assiomi
        self.axioms.register_hook(model)

        self.step_count   = 0
        self.loss_history = []

    # ------------------------------------------------------------------
    # Main step with feedback
    # ------------------------------------------------------------------

    def step(self, prompt: str, response: str,
             feedback: float) -> dict:
        """
        One complete interaction.

        feedback:
          +1.0 = response approved
           0.0 = passive observation (no weight update)
          -1.0 = response rejected

        Returns: dict with loss, affect snapshot, step
        """
        # Insert a separator at the prompt/response boundary when missing:
        # encode(prompt + response) would otherwise BPE-merge the last prompt
        # word with the first response word ('di: il cane' + 'il cane!' →
        # segment 'caneil'), training on tokens the model can never reproduce.
        sep = ""
        if prompt and response and not prompt[-1].isspace() \
                and not response[0].isspace():
            sep = " "
        ids_np = np.array(
            self.tokenizer.encode(prompt + sep + response), dtype=np.int32)
        # Prompt tokens are conditioning context only — excluded from the loss.
        # The separator is deliberately counted as part of the RESPONSE: the
        # model generates from encode(prompt) alone, so if the separator
        # position were masked it would never learn to emit it and every turn
        # would start from an off-distribution context. Measured at L0 with
        # the separator masked: P('!'|'di ba') = 0.966 — the model treated the
        # prompt's trailing syllable as its own answer and only added the
        # terminator, capping step A at ~27% exact.
        n_prompt = len(self.tokenizer.encode(prompt)) if prompt else 0

        if len(ids_np) < 2:
            return {"loss": None, "affect": self.affect.snapshot(),
                    "step": self.step_count}

        # Truncate to model's max_seq_len — cut from the LEFT so the response
        # (the learning target) is always preserved; long prompts lose their
        # oldest tokens instead of silently dropping the answer.
        max_len = self.model.max_seq_len - 1
        if len(ids_np) > max_len:
            cut = len(ids_np) - max_len
            ids_np   = ids_np[cut:]
            n_prompt = max(0, n_prompt - cut)

        ids = torch.from_numpy(ids_np).long().to(self.device)

        # ---- Update affective state BEFORE the training step ----
        with torch.no_grad():
            logits_eval = self.model.forward(ids)
            self.affect.update_from_logits(logits_eval[-1], self.model.vocab_size)
            self.affect.update_from_weights(self.model.tok_emb.weight)

        loss_val = None

        # ---- Training step (only if feedback != 0 and there is a target) ----
        # len(ids_np) <= n_prompt means the response was truncated away: the
        # masked loss would be the zero guard and opt.step() would apply a
        # pure-momentum update — skip entirely.
        if feedback != 0.0 and len(ids_np) > n_prompt:
            opt = self.optimizer._opt
            opt.zero_grad()
            logits = self.model.forward(ids)
            # Prompt-masked loss: learn only the response given the prompt.
            loss   = self.model.loss(logits, ids, prompt_len=n_prompt)

            # feedback < 0: gradient ascent (penalise the response); the
            # attenuation lives in the LR multiplier below, NOT in the loss —
            # Adam's adaptive normalisation neutralises loss scaling.
            scaled_loss = -loss if feedback < 0 else loss

            # Guard: a dormant-slot target (tokenizer/model desync) yields
            # loss=+inf; skip the update rather than corrupting the weights.
            if torch.isfinite(scaled_loss):
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Feedback modulates the LEARNING RATE, not the loss: with
                # Adam, loss scaling is neutralised by the adaptive
                # normalisation (and by the clip above), so |feedback| had no
                # effect on the update magnitude. The affine floor keeps
                # positive grades in 0.6-1.0x — close to the full-force regime
                # under which L0-L2 were validated — while still graduating
                # the modulation. Negative feedback: attenuated unlearning.
                if feedback < 0:
                    lr_mult = 0.3 * abs(feedback)
                else:
                    lr_mult = 0.5 + 0.5 * abs(feedback)
                base_lrs = [g["lr"] for g in opt.param_groups]
                try:
                    for g in opt.param_groups:
                        g["lr"] = g["lr"] * lr_mult
                    # Axiom hook is already registered on tok_emb.weight
                    opt.step()
                finally:
                    for g, lr0 in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr0
                loss_val = loss.item()
                self.loss_history.append(loss_val)

        # ---- Update affective state with feedback ----
        self.affect.update_from_feedback(feedback)

        # ---- Novelty drive: bonus for new or rarely-seen tokens ----
        # Only for positive feedback — novelty reward is always positive.
        # Decays with encounter count: sqrt(n+1) denominator.
        if feedback > 0 and len(ids_np) > 0:
            # Use only the response tokens (after the prompt, sep included)
            response_ids = ids_np[n_prompt:].tolist()
            self.affect.update_from_novelty(response_ids, feedback)

        # ---- Register tokens in the modulator ----
        if len(ids_np) > 0:
            self.mod.record_feedback(ids_np.tolist(), feedback)

        self.step_count += 1
        snap = self.affect.snapshot()

        return {
            "loss":   loss_val,
            "affect": snap,
            "step":   self.step_count,
        }

    # ------------------------------------------------------------------
    # Generation with affective modulation
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int = 100,
                 base_temperature: float = 0.8,
                 top_k: int = 40,
                 min_tokens: int = 4,
                 stop_after: Optional[int] = None) -> str:
        """
        Generate text with affective modulation applied to each token's
        logits (inference only, does not touch weights).

        min_tokens: EOS is suppressed (and its boost withheld) before this
          many response tokens. Keep it generous — it is what prevents an
          empty answer.
        stop_after: punctuation soft-stop floor; defaults to min_tokens.
          Must be set to about (expected length - 1) so a short answer can
          close on its own final '!' instead of over-generating
          ('ma!' -> 'mamamama!'). Kept separate from min_tokens because the
          two gates pull in opposite directions.
        """
        if stop_after is None:
            stop_after = min_tokens
        from splx.utils import sample_top_k
        ids = self.tokenizer.encode(prompt)
        # Ensure prompt fits within max_seq_len
        max_ctx = self.model.max_seq_len - 1
        ids = ids[-max_ctx:]

        eos_id    = self.tokenizer.get_special_id(self.tokenizer.EOS_TOKEN) \
                    if hasattr(self.tokenizer, 'get_special_id') else None
        prompt_len = len(ids)

        with torch.no_grad():
            for step in range(max_tokens):
                ctx = torch.tensor(ids[-128:], dtype=torch.long, device=self.device)
                logits = self.model.forward(ctx)   # (T, V)
                last   = logits[-1]                # (V,)

                # Update confidence from these logits
                self.affect.update_from_logits(last, self.model.vocab_size)

                # Modulate (adaptive temperature + gate)
                response_len = len(ids) - prompt_len
                modulated = self.mod.modulate(last, base_temperature,
                                              response_len=response_len,
                                              eos_min_len=min_tokens + 1)

                # Hard-suppress EOS before min_tokens. Gating only the EOS
                # *boost* still allowed the model to SAMPLE EOS as its very
                # first token: measured at L0, 43.8% of answers were a bare
                # '<|EOS|>'. Prompt masking concentrates the gradient on the
                # response, where EOS is 25-33% of the tokens of a short
                # target, so EOS-first becomes the highest-probability move.
                if eos_id is not None and response_len < min_tokens \
                        and eos_id < modulated.shape[0]:
                    modulated[eos_id] = float('-inf')

                # Sample
                next_id = sample_top_k(modulated.cpu().numpy(), k=top_k,
                                       temperature=1.0)  # temp already applied
                ids.append(next_id)

                # Stop at EOS token (classical LLM behavior)
                if eos_id is not None and next_id == eos_id and step >= 1:
                    break
                # Also stop at sentence-ending punctuation (soft stop).
                # min_tokens before stopping: avoids cutting off the target word
                # if the model starts with noise punctuation (e.g. ". ! cane!" would
                # stop at the final "!" after "cane", not at the leading ".").
                if step >= max(1, stop_after) and next_id in (33, 46, 63):
                    break   # '!', '.', '?'

        full = self.tokenizer.decode(ids)

        # Clean the response portion: strip leading noise artifacts.
        # The model has a learned habit of prepending '!' or echoing 'di:' from
        # the prompt. Strip these so inference output is usable.
        response_raw = full[len(self.tokenizer.decode(ids[:prompt_len])):]
        response_clean = _clean_response(response_raw)
        prompt_decoded = self.tokenizer.decode(ids[:prompt_len])
        return prompt_decoded + response_clean

    # ------------------------------------------------------------------
    # Corpus training (no feedback — autonomous learning)
    # ------------------------------------------------------------------

    def train_on_text(self, text: str, block_size: int = 128,
                      batch_size: int = 8, log_every: int = 50,
                      total_steps_hint: int = 0) -> list:
        """
        Self-supervised learning from free text.
        No affective feedback — standard cross-entropy only.
        total_steps_hint: if > 0, shows cumulative percentage in the log.
        """
        # Tokenisation (can take minutes for large corpora with 8K vocabulary)
        mb = len(text) / 1024 / 1024
        print(f"  Tokenising corpus ({mb:.1f}MB)...", end="", flush=True)
        ids_all = np.array(self.tokenizer.encode(text), dtype=np.int32)
        starts  = list(range(0, len(ids_all) - block_size, block_size))
        np.random.shuffle(starts)
        total_batches = len(starts)  # per questa chiamata
        print(f" {len(ids_all):,} token  →  {total_batches:,} batch", flush=True)

        losses = []
        i = 0
        opt = self.optimizer._opt
        import time as _time
        t_start = _time.time()
        _last_milestone = -1   # for 10%-progress prints on large runs

        while i < len(starts):
            seqs = [ids_all[s:s + block_size + 1]
                    for s in starts[i:i + batch_size]
                    if s + block_size + 1 <= len(ids_all)]
            i += batch_size
            if not seqs:
                continue

            # 10% milestone for large corpus (dream N1, phase-0 L3+)
            if total_batches >= 500:
                _pct_now = 10 * (i * 10 // max(total_batches, 1))
                if _pct_now > _last_milestone:
                    _last_milestone = _pct_now
                    _el_m = (_time.time() - t_start) / 60
                    print(f"  [{_pct_now:3d}%] batch {i}/{total_batches}  {_el_m:.1f}min",
                          flush=True)

            batch = torch.from_numpy(np.stack(seqs)).long().to(self.device)
            opt.zero_grad()
            logits = self.model.forward(batch)
            loss   = self.model.loss(logits, batch)

            # Guard against NaN/Inf loss (can occur on XPU in early steps)
            if not torch.isfinite(loss):
                losses.append(float('nan'))
                self.step_count += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            self.step_count += 1

            if len(losses) % log_every == 0:
                elapsed = _time.time() - t_start
                pct_epoch = 100 * i // max(total_batches, 1)
                pct_total = f"  {100 * self.step_count // total_steps_hint}%" \
                            if total_steps_hint > 0 else ""
                eta_str = ""
                if pct_epoch > 0:
                    eta = elapsed * (100 - pct_epoch) / pct_epoch
                    eta_str = f"  eta {eta:.0f}s"
                tok_s = int(len(losses) * block_size * batch_size / max(elapsed, 1e-6))
                valid = [l for l in losses[-log_every:] if l == l and l < 1e6]
                loss_str = f"{np.mean(valid):.4f}" if valid else "nan(skip)"
                print(f"  step {self.step_count:5d}  {pct_epoch:3d}%{pct_total}"
                      f"  loss {loss_str}  {tok_s}tok/s{eta_str}"
                      f"  {self.affect}")

        return losses

    # ------------------------------------------------------------------
    # Supervised training on Q&A pairs
    # ------------------------------------------------------------------

    def train_on_qa_pairs(self, path: str, n_epochs: int = 5,
                          feedback: float = 1.0) -> list:
        """
        Supervised fine-tuning on (prompt, correct_response) pairs.

        Each pair is a gold example: the model must learn to produce
        exactly `response` given `prompt`. Equivalent to Signal 4 (imitation)
        in the teaching loop, but in batch and without external APIs.

        File format: JSONL, one pair per line:
            {"prompt": "di: il cane", "response": "il cane!"}

        Used between teaching sessions to reinforce prompt→response
        correlations before the model is retested.

        Returns: list of losses per step.
        """
        import json, random as _rnd
        if not os.path.exists(path):
            return []

        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if not pairs:
            return []

        losses = []
        for epoch in range(n_epochs):
            _rnd.shuffle(pairs)
            for pair in pairs:
                p = pair.get("prompt", "")
                r = pair.get("response", "")
                if p and r:
                    result = self.step(p, r, feedback=feedback)
                    if result.get("loss") is not None:
                        losses.append(result["loss"])

        return losses

    # ------------------------------------------------------------------
    # Axioms
    # ------------------------------------------------------------------

    def add_axiom(self, text: str, is_objective: bool = True,
                  protection: float = 1.0) -> None:
        """
        Register an axiom from text.
        Example: trainer.add_axiom("1+1=2", is_objective=True)
        """
        ids = self.tokenizer.encode(text)
        self.axioms.register(ids, description=text,
                             is_objective=is_objective,
                             protection_level=protection)
        print(f"  [axiom] '{text}' → tokens {ids}  protection={protection}")

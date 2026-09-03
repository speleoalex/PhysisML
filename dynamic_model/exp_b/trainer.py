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
from physisml.torch_model import DEVICE  # auto-detected best device
import torch.nn.functional as F
from typing import Optional, List

from physisml.torch_model import TorchGPT, TorchAdamOptimizer
from physisml.tokenizer   import BPETokenizer

from dynamic_model.exp_b.affect_state import AffectState, AffectSnapshot
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.axioms       import AxiomRegistry


import re as _re
import hashlib as _hashlib


# ── tokenised-corpus cache ───────────────────────────────────────────────────
# The dream re-encodes the SAME corpus on every cycle. Measured on this repo:
# 4.29MB of N1 corpus takes 125s to encode with the 2607-token BPE (15k tok/s,
# pure Python), and the build runs 6 dreams per level over 13 levels — about
# 2.7 hours of the 55-hour build spent re-tokenising byte-identical text.
#
# The cache is keyed on the text AND on the tokenizer's own contents, because a
# single new merge re-segments everything: reusing ids across a vocabulary
# change would silently train on the wrong tokens. Measured on the last build,
# N2-B added tokens in 2 dreams out of 12, so the key holds most of the time —
# and when it does not, the only cost is the encode that would have happened
# anyway.
_TOK_CACHE_DIR  = os.environ.get("PHYSISML_TOK_CACHE",
                                 os.path.join("models", "cache", "tokenized"))
_TOK_CACHE_KEEP = 24          # newest entries kept; one is ~6MB


def _tokenizer_fingerprint(tok) -> str:
    """Hash of everything in the tokenizer that can change an encoding."""
    h = _hashlib.sha256()
    h.update(f"n={len(tok)}\x1e".encode())
    for k, v in sorted((getattr(tok, "vocab", None) or {}).items(),
                       key=lambda kv: repr(kv[0])):
        h.update(f"{k!r}\x1f{v!r}\x1e".encode("utf-8", "surrogatepass"))
    for m in (getattr(tok, "merges", None) or []):
        h.update(f"{m!r}\x1e".encode("utf-8", "surrogatepass"))
    return h.hexdigest()[:16]


def _prune_tok_cache() -> None:
    try:
        files = sorted((os.path.getmtime(os.path.join(_TOK_CACHE_DIR, f)),
                        os.path.join(_TOK_CACHE_DIR, f))
                       for f in os.listdir(_TOK_CACHE_DIR) if f.endswith(".npy"))
        for _, path in files[:-_TOK_CACHE_KEEP]:
            os.remove(path)
    except OSError:
        pass


def encode_cached(tok, text: str) -> tuple:
    """(ids, from_cache) for `text` under `tok`. Never raises on cache trouble.

    A miss, a corrupt file, an unwritable directory: all fall back to encoding,
    because a build must not fail over a cache.
    """
    ids = None
    try:
        key = (_hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:24]
               + "-" + _tokenizer_fingerprint(tok))
        path = os.path.join(_TOK_CACHE_DIR, key + ".npy")
        if os.path.exists(path):
            cached = np.load(path)
            # Cheap insurance on top of the key: ids from a larger vocabulary
            # would index rows this model does not have.
            if cached.size and int(cached.max()) < len(tok):
                os.utime(path, None)          # keep the hot entries alive
                return cached.astype(np.int32, copy=False), True
    except Exception:
        path = None

    ids = np.array(tok.encode(text), dtype=np.int32)
    if path:
        try:
            os.makedirs(_TOK_CACHE_DIR, exist_ok=True)
            # np.save appends '.npy' unless the name already ends in it, so
            # the temp name carries the suffix and os.replace stays a rename.
            tmp = f"{path}.{os.getpid()}.tmp.npy"
            np.save(tmp, ids)
            os.replace(tmp, path)
            _prune_tok_cache()
        except Exception:
            pass
    return ids, False

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


def encode_prompt_response(tokenizer, prompt: str, response: str,
                           max_len: int) -> tuple:
    """Encode a (prompt, response) pair exactly the way step() trains on it.

    Returns (ids_np, n_prompt). Shared between TrainerB.step and
    EWC.estimate_fisher: the Fisher must measure the same loss geometry the
    training steps optimise, and a hand copy of this block would drift.

    Insert a separator at the prompt/response boundary when missing:
    encode(prompt + response) would otherwise BPE-merge the last prompt word
    with the first response word ('di: il cane' + 'il cane!' -> segment
    'caneil'), training on tokens the model can never reproduce.

    The separator is deliberately counted as part of the RESPONSE: the model
    generates from encode(prompt) alone, so if the separator position were
    masked it would never learn to emit it and every turn would start from an
    off-distribution context. Measured at L0 with the separator masked:
    P('!'|'di ba') = 0.966 — the model treated the prompt's trailing syllable
    as its own answer and only added the terminator, capping step A at ~27%
    exact.

    Truncation cuts from the LEFT so the response (the learning target) is
    always preserved; long prompts lose their oldest tokens instead of
    silently dropping the answer.
    """
    sep = ""
    if prompt and response and not prompt[-1].isspace() \
            and not response[0].isspace():
        sep = " "
    ids_np = np.array(
        tokenizer.encode(prompt + sep + response), dtype=np.int32)
    # Prompt tokens are conditioning context only — excluded from the loss.
    n_prompt = len(tokenizer.encode(prompt)) if prompt else 0

    if len(ids_np) > max_len:
        cut = len(ids_np) - max_len
        ids_np   = ids_np[cut:]
        n_prompt = max(0, n_prompt - cut)
    return ids_np, n_prompt


class TrainerB:

    def __init__(self, model: TorchGPT, tokenizer,
                 optimizer: TorchAdamOptimizer,
                 affect_state: Optional[AffectState] = None,
                 modulator: Optional[AffectModulator] = None,
                 axiom_registry: Optional[AxiomRegistry] = None,
                 ewc=None):

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

        # Optional exp_b.ewc.EWC instance (exp_i's --anti-forgetting ewc arm).
        # When set, every backward site adds its quadratic penalty. None (the
        # default) changes nothing — the validated build never sets it.
        self.ewc = ewc
        self._ewc_log_every   = 200   # steps between penalty/grad-ratio logs
        self._ewc_log_counter = 0

        self.step_count   = 0
        self.loss_history = []

    def _ewc_log(self, total_norm) -> None:
        """Every _ewc_log_every backward passes: the penalty value and the
        ratio of the penalty's analytic gradient norm to the pre-clip norm of
        the COMBINED gradient. A ratio sitting near or above 1 means the
        clip_grad_norm_(1.0) that follows every backward is spending its
        budget on the anchor instead of the task — the lambda-too-high
        failure mode the sweep exists to reject."""
        if self.ewc is None:
            return
        self._ewc_log_counter += 1
        if self._ewc_log_counter % self._ewc_log_every:
            return
        with torch.no_grad():
            pen = float(self.ewc.penalty(self.model))
        ratio = self.ewc.penalty_grad_norm(self.model) \
            / max(float(total_norm), 1e-9)
        print(f"  [ewc] penalty {pen:.4f}  grad-ratio {ratio:.3f}", flush=True)

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
        # Separator insertion, prompt masking and left truncation all live in
        # encode_prompt_response (shared with EWC.estimate_fisher — see its
        # docstring for the measured rationale of each choice).
        ids_np, n_prompt = encode_prompt_response(
            self.tokenizer, prompt, response, self.model.max_seq_len - 1)

        if len(ids_np) < 2:
            return {"loss": None, "affect": self.affect.snapshot(),
                    "step": self.step_count}

        ids = torch.from_numpy(ids_np).long().to(self.device)

        # ---- Update affective state BEFORE the training step ----
        with torch.no_grad():
            logits_eval = self.model.forward(ids)
            self.affect.update_from_logits(logits_eval[-1], self.model.vocab_size)
            # Active rows only: the dormant slots are zeroed by
            # construction, so counting them made ignorance a constant
            # (measured 0.717 with 2547 of 9000 slots in use).
            self.affect.update_from_weights(
                self.model.tok_emb.weight,
                active_vocab_size=getattr(self.model,
                                          'active_vocab_size', None))

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
                total_loss = scaled_loss
                if self.ewc is not None:
                    # AFTER the sign flip above: added before it, a
                    # negative-feedback turn would run gradient ASCENT on the
                    # quadratic and actively repel the weights from the
                    # anchor. The affect lr_mult below scales the combined
                    # update, penalty pull included — a 0.3x unlearning step
                    # also pulls 0.3x toward the anchor, which is
                    # directionally correct.
                    total_loss = total_loss + self.ewc.penalty(self.model)
                total_loss.backward()
                _total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0)
                self._ewc_log(_total_norm)
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
            # Curiosity memory, in word forms. Note this registers the TARGET
            # of the step, which at SIGNAL 1 is the teacher's gold: a word
            # counts as taught from the moment it is shown. Separate channel
            # from the novelty counters above — see AffectState.word_ignorance
            # for why words and not token ids.
            try:
                _txt = self.tokenizer.decode(response_ids)
                # Strip special-token literals first: '<|EOS|>' otherwise
                # enters the memory as the word 'eos', because the word regex
                # sees letters and knows nothing about markup.
                for _name in getattr(self.tokenizer, "special_tokens", {}):
                    _txt = _txt.replace(_name, " ")
                self.affect.register_rewarded_words(_txt, feedback)
            except Exception:
                pass

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

    def _undecodable_mask(self):
        """
        Boolean mask over the model's output layer marking ids the tokenizer
        cannot turn back into bytes.

        The model's vocab_size is rounded up past the tokenizer's real size
        (9000 against 8042 at L3), so several hundred output slots decode to
        nothing. Sampling one of them used to raise KeyError in decode(); even
        once decode tolerates it, the token is invisible in the output and
        derails the rest of the answer. Suppressing the slots also stops
        top_k from spending candidate places on them.

        Returns None when every model id is representable. Cached, and
        recomputed when the dynamic tokenizer grows.
        """
        vocab = getattr(self.tokenizer, "vocab", None)
        if not vocab:
            return None
        key = (len(vocab), self.model.vocab_size)
        if getattr(self, "_undecodable_key", None) != key:
            missing = [i for i in range(self.model.vocab_size) if i not in vocab]
            if missing:
                mask = torch.zeros(self.model.vocab_size, dtype=torch.bool,
                                   device=self.device)
                mask[torch.tensor(missing, dtype=torch.long, device=self.device)] = True
            else:
                mask = None
            self._undecodable_key  = key
            self._undecodable_mask_cached = mask
        return self._undecodable_mask_cached

    def generate(self, prompt: str, max_tokens: int = 100,
                 base_temperature: float = 0.8,
                 top_k: int = 40,
                 min_tokens: int = 4,
                 stop_after: Optional[int] = None,
                 banned_ids: Optional[set] = None) -> str:
        """
        Generate text with affective modulation applied to each token's
        logits (inference only, does not touch weights).

        min_tokens: EOS is suppressed (and its boost withheld) before this
          many response tokens. Keep it generous — it is what prevents an
          empty answer.
        banned_ids: token ids suppressed at every step (logit = -inf), forcing
          the answer through whatever other path the model has. Used by the
          vocabulary ablation to measure how much a trained model actually
          relies on a token it was given; None (the default) changes nothing.
        stop_after: punctuation soft-stop floor; defaults to min_tokens.
          Must be set to about (expected length - 1) so a short answer can
          close on its own final '!' instead of over-generating
          ('ma!' -> 'mamamama!'). Kept separate from min_tokens because the
          two gates pull in opposite directions.
        """
        if stop_after is None:
            stop_after = min_tokens
        from physisml.utils import sample_top_k
        ids = self.tokenizer.encode(prompt)
        # Ensure prompt fits within max_seq_len
        max_ctx = self.model.max_seq_len - 1
        ids = ids[-max_ctx:]

        eos_id    = self.tokenizer.get_special_id(self.tokenizer.EOS_TOKEN) \
                    if hasattr(self.tokenizer, 'get_special_id') else None
        prompt_len = len(ids)
        undecodable = self._undecodable_mask()

        # Per-prompt drive to ask, computed on the prompt TEXT. The state's
        # global `curiosity` cannot tell 'questo è un ragno' from 'questo è un
        # gatto', which is the whole distinction the gate exists to make.
        # Skipped entirely when the gate is off, which is the default.
        curiosity = None
        if getattr(self.mod, "ask_gate", False):
            curiosity = self.affect.ask_drive(prompt)

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
                                              eos_min_len=min_tokens + 1,
                                              curiosity=curiosity)

                # Hard-suppress EOS before min_tokens. Gating only the EOS
                # *boost* still allowed the model to SAMPLE EOS as its very
                # first token: measured at L0, 43.8% of answers were a bare
                # '<|EOS|>'. Prompt masking concentrates the gradient on the
                # response, where EOS is 25-33% of the tokens of a short
                # target, so EOS-first becomes the highest-probability move.
                if eos_id is not None and response_len < min_tokens \
                        and eos_id < modulated.shape[0]:
                    modulated[eos_id] = float('-inf')

                # Never sample an id the tokenizer cannot decode.
                if undecodable is not None:
                    modulated[undecodable] = float('-inf')

                # Ablation: suppress the tokens under test.
                if banned_ids:
                    for _bid in banned_ids:
                        if _bid < modulated.shape[0]:
                            modulated[_bid] = float('-inf')

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
                # Stop at sentence-ending punctuation (soft stop). Test the
                # DECODED token, not just the bare '!'/'.'/'?' ids: the 8K
                # vocabulary has multi-char tokens that already carry the
                # terminator ('la!'=2815, 'ta!'=2017, 'no!'=2945). Matching ids
                # only left those answers unterminated, so generation ran on
                # and EVERY target whose gold answer ends in such a token
                # scored 0% exact ('la!','ta!','da!','sa!','no!','lala!',
                # 'tata!') while 'ma!','pa!','mama!' (bare '!' = id 33) passed.
                if step >= stop_after:
                    # Defensive: a tokenizer/model mismatch can yield an id the
                    # tokenizer cannot decode — never crash generation for it.
                    try:
                        _tail = self.tokenizer.decode([next_id]).rstrip()
                    except Exception:
                        _tail = "!" if next_id in (33, 46, 63) else ""
                    if _tail.endswith(('!', '.', '?')):
                        # At the very first token only a content-bearing token
                        # may terminate — a bare '!' there is leading noise.
                        if step >= 1 or len(_tail) > 1:
                            break

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
        ids_all, cached = encode_cached(self.tokenizer, text)
        if cached:
            print(" [cache]", end="", flush=True)
        # Filtered ONCE, so every window below is valid and the batch can be
        # indexed straight off the device without a host-side check.
        starts  = [st for st in range(0, len(ids_all) - block_size, block_size)
                   if st + block_size + 1 <= len(ids_all)]
        np.random.shuffle(starts)
        total_batches = len(starts)  # per questa chiamata
        print(f" {len(ids_all):,} token  →  {total_batches:,} batch", flush=True)

        # The whole corpus goes to the device ONCE and every batch is gathered
        # there. The old loop rebuilt each batch on the host (np.stack, then a
        # host->device copy) which on a GPU is a stall per step: measured on the
        # Arc, the real loop ran at 2,225 tok/s against 6,015 for the same step
        # in isolation. 1.6M int32 is 6.5MB — the corpus is never the thing that
        # does not fit.
        _ids_dev = torch.from_numpy(ids_all.astype(np.int64)).to(self.device)
        _starts_dev = torch.tensor(starts, dtype=torch.long, device=self.device)
        _offsets = torch.arange(block_size + 1, device=self.device)

        losses = []
        # Losses live on the device until drained in one transfer. 64 steps is
        # far below the logging interval and keeps the buffer negligible.
        _DRAIN_EVERY = 64
        _loss_buf = []
        _n_bad = torch.zeros((), dtype=torch.long, device=self.device)
        i = 0
        opt = self.optimizer._opt
        import time as _time
        t_start = _time.time()
        _last_milestone = -1   # for 10%-progress prints on large runs

        while i < len(starts):
            window = _starts_dev[i:i + batch_size]
            i += batch_size
            if window.numel() == 0:
                continue

            # 10% milestone for large corpus (dream N1, phase-0 L3+)
            if total_batches >= 500:
                _pct_now = 10 * (i * 10 // max(total_batches, 1))
                if _pct_now > _last_milestone:
                    _last_milestone = _pct_now
                    _el_m = (_time.time() - t_start) / 60
                    print(f"  [{_pct_now:3d}%] batch {i}/{total_batches}  {_el_m:.1f}min",
                          flush=True)

            # One gather on the device instead of a host stack plus a copy.
            batch = _ids_dev[window.unsqueeze(1) + _offsets]
            opt.zero_grad()
            logits = self.model.forward(batch)
            loss   = self.model.loss(logits, batch)

            # Guard against NaN/Inf loss (it happens on XPU in early steps).
            # Sanitised ON THE DEVICE: the old `if not torch.isfinite(loss)`
            # read a device tensor from Python, which is a full pipeline stall
            # every single step. nan_to_num's backward passes zero where the
            # input was not finite, so a bad step contributes no gradient —
            # the difference from the old code is that the optimizer step still
            # runs, so Adam's moments decay instead of the step being skipped
            # outright. Non-finite steps are counted on the device and reported
            # at the end.
            _n_bad += (~torch.isfinite(loss.detach())).long()
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

            # The EWC penalty (when attached) joins the backward here too:
            # this covers phase 1's autonomous pre-training and the dream's
            # N1 pass — both major sources of drift from the anchor. The
            # logged loss stays the TASK loss so curves compare across arms.
            total_loss = loss if self.ewc is None \
                else loss + self.ewc.penalty(self.model)
            total_loss.backward()
            _total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)
            opt.step()
            self._ewc_log(_total_norm)

            # Kept on the device; drained in one transfer below. .item() here
            # was the second stall per step.
            _loss_buf.append(loss.detach())
            self.step_count += 1

            if len(_loss_buf) >= _DRAIN_EVERY:
                losses.extend(torch.stack(_loss_buf).cpu().tolist())
                _loss_buf.clear()

            if len(losses) % log_every == 0 and losses:
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

        if _loss_buf:
            losses.extend(torch.stack(_loss_buf).cpu().tolist())
            _loss_buf.clear()
        n_bad = int(_n_bad.item())
        if n_bad:
            print(f"  {n_bad} step con loss non finita: gradiente azzerato")
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

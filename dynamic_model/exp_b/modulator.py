"""
AffectModulator — applies affective state to logits.

ONLY during inference, NEVER during training.
This avoids the unstable feedback loop:
  high fear → flat distribution → low confidence → even more fear → divergence

Four modulators applied in sequence:
  1. Adaptive temperature:   logits /= (base_temp * (1 + fear * 2.0))
  2. "don't know" boost:     if ignorance > 0.7: boost DONT_KNOW token
  3. Pain gate:              inhibits tokens associated with negative feedback
  4. Pleasure gate:          amplifies tokens associated with positive feedback
"""
import torch
import numpy as np
from collections import deque
from typing import List, Optional, Set

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

from dynamic_model.exp_b.affect_state import AffectState


# Special tokens — must be registered in the tokenizer before use
DONT_KNOW_TEXT = "<|dont_know|>"
UNCERTAIN_TEXT = "<|uncertain|>"


class AffectModulator:

    # Thresholds
    IGNORANCE_THRESHOLD  = 0.7
    CONFIDENCE_THRESHOLD = 0.15

    # Effect intensities
    DONT_KNOW_BOOST    = 3.0
    PAIN_INHIBITION    = 1.5
    PLEASURE_AMP       = 0.8
    TEMP_FEAR_SCALE    = 2.0

    # Circular buffer for recent tokens with feedback
    BUFFER_SIZE = 50

    # EOS boost: gradually increases probability of stopping as response grows
    EOS_MIN_LEN   = 5    # never boost EOS before this many response tokens
                         # must be enough to produce target word before stopping
    EOS_BOOST_MAX = 3.0  # max additive boost at long responses (reduced from 4.0)

    def __init__(self, affect_state: AffectState,
                 dont_know_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None):
        self.state          = affect_state
        self.dont_know_id   = dont_know_token_id
        self.eos_id         = eos_token_id  # EOS token ID (e.g. 256 = <|EOS|>)

        # Token buffers associated with feedback (last BUFFER_SIZE)
        self.pain_ids:    deque = deque(maxlen=self.BUFFER_SIZE)
        self.pleasure_ids: deque = deque(maxlen=self.BUFFER_SIZE)

    # ------------------------------------------------------------------
    # Modulazione inference
    # ------------------------------------------------------------------

    def modulate(self, logits: torch.Tensor,
                 base_temperature: float = 1.0,
                 response_len: int = 0,
                 eos_min_len: Optional[int] = None) -> torch.Tensor:
        """
        Apply affective modulations to the LAST token's logits.
        Called AFTER model.forward(), BEFORE sampling.
        NOT called during training.

        logits:       (V,) — logits of the last token
        response_len: tokens generated so far (for EOS boost)
        eos_min_len:  override EOS_MIN_LEN — no EOS boost before this many
                      response tokens. Callers that expect long answers
                      (high curriculum levels) must raise this, otherwise
                      the boost pushes the model to stop mid-answer.
        Returns: (V,) — modulated logits
        """
        logits = logits.clone().float()
        s = self.state

        # 1. Adaptive temperature based on fear
        eff_temp = base_temperature * (1.0 + s.fear * self.TEMP_FEAR_SCALE)
        logits = logits / max(eff_temp, 0.1)

        # 2. EOS boost: progressively increase EOS probability as response grows.
        #    After min_len tokens, the boost rises linearly up to EOS_BOOST_MAX.
        #    This mimics classical LLM behavior where short responses are preferred
        #    and long ones are discouraged without hard truncation.
        _eos_min = eos_min_len if eos_min_len is not None else self.EOS_MIN_LEN
        if self.eos_id is not None and self.eos_id < logits.shape[0]:
            if response_len >= _eos_min:
                eos_boost = min(
                    self.EOS_BOOST_MAX,
                    1.5 * (response_len - _eos_min + 1) / 5.0
                )
                logits[self.eos_id] = logits[self.eos_id] + eos_boost

        # 3. Boost "don't know" token
        if self.dont_know_id is not None:
            if s.ignorance > self.IGNORANCE_THRESHOLD or \
               s.confidence < self.CONFIDENCE_THRESHOLD:
                boost = self.DONT_KNOW_BOOST * s.ignorance
                logits[self.dont_know_id] = logits[self.dont_know_id] + boost

        # 3. Pain gate (inibisce token associati a feedback negativo)
        if s.pain > 0.1 and self.pain_ids:
            pain_t = torch.tensor(list(set(self.pain_ids)),
                                  dtype=torch.long, device=logits.device)
            pain_t = pain_t[pain_t < logits.shape[0]]
            if len(pain_t) > 0:
                logits[pain_t] = logits[pain_t] - s.pain * self.PAIN_INHIBITION

        # 4. Pleasure gate (amplifica token associati a feedback positivo)
        if s.pleasure > 0.5 and self.pleasure_ids:
            pleas_t = torch.tensor(list(set(self.pleasure_ids)),
                                   dtype=torch.long, device=logits.device)
            pleas_t = pleas_t[pleas_t < logits.shape[0]]
            if len(pleas_t) > 0:
                logits[pleas_t] = logits[pleas_t] + \
                                  (s.pleasure - 0.5) * self.PLEASURE_AMP

        return logits

    # ------------------------------------------------------------------
    # Registrazione feedback
    # ------------------------------------------------------------------

    def record_feedback(self, token_ids: List[int], feedback: float) -> None:
        """
        Associate tokens with the feedback valence.
        Called after each response with explicit feedback.
        """
        if feedback > 0.1:
            self.pleasure_ids.extend(token_ids)
        elif feedback < -0.1:
            self.pain_ids.extend(token_ids)

    # ------------------------------------------------------------------
    # Penalty da aggiungere alla loss (opzionale, non differenziabile)
    # ------------------------------------------------------------------

    def affect_loss_penalty(self) -> float:
        """
        Additional scalar term to the loss — penalises the model when it is
        very uncertain but does not use the "don't know" token.
        Does not enter the computational graph (regularisation signal only).
        """
        s = self.state
        penalty = 0.0
        if s.confidence < self.CONFIDENCE_THRESHOLD and \
           s.ignorance > self.IGNORANCE_THRESHOLD:
            penalty += 0.05 * (1.0 - s.confidence)
        penalty += 0.02 * s.pain
        return penalty

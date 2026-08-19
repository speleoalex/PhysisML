"""
TrainerA — training online con TorchDynamicGPT, vocabolario dinamico e dream phase.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

import numpy as np
import torch
from collections import deque
from typing import Optional

from dynamic_model.exp_a.expansion_manager  import VocabExpansionManager
from dynamic_model.exp_a.dream_consolidator import DreamConsolidator


class TrainerA:

    def __init__(self, model, tokenizer, optimizer,
                 expansion_manager: VocabExpansionManager,
                 dream_consolidator: DreamConsolidator,
                 batch_size: int = 8):
        self.model      = model
        self.tokenizer  = tokenizer
        self.optimizer  = optimizer
        self.exp_mgr    = expansion_manager
        self.dream      = dream_consolidator
        self.batch_size = batch_size

        self.step_count   = 0
        self.loss_history = []
        self._text_buffer = deque(maxlen=20_000)
        expansion_manager.set_dream_consolidator(dream_consolidator)

    # ------------------------------------------------------------------
    # Singolo step online (batch_size=1, per feedback interattivo)
    # ------------------------------------------------------------------

    def step(self, text_chunk: str) -> dict:
        self._text_buffer.extend(text_chunk)
        self.exp_mgr.add_text(text_chunk)

        ids_np  = np.array(self.tokenizer.encode(text_chunk), dtype=np.int32)
        if len(ids_np) < 2:
            return {"loss": None, "vocab_size": self.model.vocab_size,
                    "step": self.step_count}

        ids_t = torch.from_numpy(ids_np).long()
        loss  = self.model.train_step(ids_t, self.optimizer)

        self.step_count += 1
        self.loss_history.append(loss)

        new_ids = self.exp_mgr.maybe_expand(self.step_count)
        if new_ids:
            raw = "".join(self._text_buffer)
            self.dream.notify_expansion(len(new_ids), raw)

        return {"loss": loss, "vocab_size": self.model.vocab_size,
                "step": self.step_count}

    # ------------------------------------------------------------------
    # Perplexity su testo di validazione
    # ------------------------------------------------------------------

    def eval_perplexity(self, val_text: str) -> float:
        ids_all = np.array(self.tokenizer.encode(val_text), dtype=np.int32)
        if len(ids_all) < 2:
            return float("nan")
        bs = 64
        losses = []
        with torch.no_grad():
            for start in range(0, len(ids_all) - bs, bs):
                chunk = torch.from_numpy(
                    ids_all[start:start + bs + 1]).long()
                logits = self.model.forward(chunk)
                loss   = self.model.loss(logits, chunk)
                losses.append(loss.item())
        return float(np.exp(np.mean(losses))) if losses else float("nan")

    # ------------------------------------------------------------------
    # Training su testo intero con mini-batching
    # ------------------------------------------------------------------

    def train_on_text(self, text: str, block_size: int = 128,
                      log_every: int = 100) -> list:
        """
        Convert text to tokens, then train with mini-batches of size
        self.batch_size. More efficient than single-step for large corpora.
        """
        ids_all = np.array(self.tokenizer.encode(text), dtype=np.int32)
        total   = len(ids_all)
        losses  = []
        B       = self.batch_size

        # Costruisce mini-batch casuali
        starts = list(range(0, total - block_size, block_size))
        np.random.shuffle(starts)

        i = 0
        while i < len(starts):
            batch_starts = starts[i:i + B]
            i += B

            seqs = []
            for s in batch_starts:
                seqs.append(ids_all[s:s + block_size + 1])

            if len(seqs) < 1:
                continue

            batch = torch.from_numpy(np.stack(seqs)).long()  # (≤B, T+1)
            loss  = self.model.train_step(batch, self.optimizer)
            losses.append(loss)

            self.step_count += 1
            self.exp_mgr.add_text(text[i * block_size:i * block_size + block_size])
            new_ids = self.exp_mgr.maybe_expand(self.step_count)
            if new_ids:
                raw = "".join(self._text_buffer)
                self.dream.notify_expansion(len(new_ids), raw)
                # Ri-encodifica con il nuovo vocab per i prossimi step
                ids_all = np.array(self.tokenizer.encode(text), dtype=np.int32)
                total   = len(ids_all)
                starts  = list(range(0, total - block_size, block_size))

            if len(losses) % log_every == 0:
                avg = np.mean(losses[-log_every:])
                print(f"  step {self.step_count:5d}  loss {avg:.4f}  "
                      f"vocab={self.model.vocab_size}")

        return losses

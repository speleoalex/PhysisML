"""
VocabExpansionManager — decides when to expand the vocabulary and keeps the
tokenizer, the model and the optimizer in step, atomically.

Updated for TorchDynamicGPT:
- vocab_expand now takes a batch of init_vecs (torch.Tensor)
- expand_moments() dropped: PyTorch Adam handles the moments on its own
  (new parameters have no history → effective lr ~ lr in the first iterations)
"""
from collections import deque
from typing import List, Optional
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))


class VocabExpansionManager:

    EXPANSION_INTERVAL = 500     # steps between one check and the next
    N_MERGES_PER_GROW  = 5       # max new tokens per expansion
    BUFFER_CHARS       = 50_000  # characters of raw text in the buffer
    EXPANSION_FREEZE_AFTER = 0   # 0 = never freeze; otherwise in chars

    def __init__(self, model, tokenizer, optimizer=None, checkpoint=None):
        self.model      = model
        self.tokenizer  = tokenizer
        self.optimizer  = optimizer   # unused for expand (PyTorch handles it)
        self.checkpoint = checkpoint

        self.recent_text_buffer: deque = deque(maxlen=self.BUFFER_CHARS)
        self.expansion_count  = 0
        self._step_since_last = 0
        self._dream           = None
        self._chars_seen      = 0
        self.frozen           = False

    def add_text(self, text: str) -> None:
        self.recent_text_buffer.extend(text)
        self._chars_seen += len(text)
        if (not self.frozen and self.EXPANSION_FREEZE_AFTER > 0
                and self._chars_seen >= self.EXPANSION_FREEZE_AFTER):
            self.frozen = True
            print(f"  [expand] vocabulary FROZEN at {self._chars_seen:,} chars "
                  f"(vocab={self.model.vocab_size})")

    def maybe_expand(self, step: int) -> Optional[List[int]]:
        if self.frozen:
            return None
        self._step_since_last += 1
        if self._step_since_last < self.EXPANSION_INTERVAL:
            return None
        self._step_since_last = 0
        if len(self.recent_text_buffer) < 100:
            return None
        return self._expand("".join(self.recent_text_buffer))

    def force_expand(self, text: str) -> List[int]:
        return self._expand(text)

    def set_dream_consolidator(self, dream) -> None:
        self._dream = dream

    def _expand(self, text: str) -> Optional[List[int]]:
        new_ids = self.tokenizer.grow(text, n_merges=self.N_MERGES_PER_GROW)
        if not new_ids:
            return None

        if self.checkpoint:
            self.checkpoint.save(reason=f"pre_expansion_{self.expansion_count + 1}")

        # Collect the initialisation vectors into one batch
        W_np = self.model.tok_emb.weight.data.cpu().numpy()
        init_vecs = np.stack([
            self.tokenizer.get_parent_embedding(nid, W_np)
            for nid in new_ids
        ])                                        # (N, d_model)

        # Atomic expansion — a single batched call
        self.model.vocab_expand(init_vecs)
        # PyTorch Adam needs no expand_moments():
        # new parameters have no history → they are initialised automatically

        self.expansion_count += 1

        if self._dream is not None:
            self._dream.register_new_tokens(new_ids)

        if self.checkpoint:
            self.checkpoint.save(reason=f"expansion_{self.expansion_count}")

        print(f"  [expand] +{len(new_ids)} tokens → vocab={self.model.vocab_size}  "
              f"(expansion #{self.expansion_count})")
        return new_ids

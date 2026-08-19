"""
VocabExpansionManager — decide quando espandere il vocabolario e coordina
tokenizer, modello e optimizer in modo atomico.

Aggiornato per TorchDynamicGPT:
- vocab_expand ora riceve un batch di init_vecs (torch.Tensor)
- expand_moments() rimosso: PyTorch Adam gestisce i momenti automaticamente
  (i nuovi parametri non hanno storia → lr effettivo ~ lr nelle prime iterazioni)
"""
from collections import deque
from typing import List, Optional
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))


class VocabExpansionManager:

    EXPANSION_INTERVAL = 500     # step tra un controllo e l'altro
    N_MERGES_PER_GROW  = 5       # max nuovi token per espansione
    BUFFER_CHARS       = 50_000  # caratteri di testo grezzo nel buffer
    EXPANSION_FREEZE_AFTER = 0   # 0 = mai congelare; altrimenti in chars

    def __init__(self, model, tokenizer, optimizer=None, checkpoint=None):
        self.model      = model
        self.tokenizer  = tokenizer
        self.optimizer  = optimizer   # non usato per expand (PyTorch auto-gestisce)
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
            print(f"  [expand] vocabolario CONGELATO a {self._chars_seen:,} chars "
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

        # Raccoglie i vettori di inizializzazione in un batch
        W_np = self.model.tok_emb.weight.data.cpu().numpy()
        init_vecs = np.stack([
            self.tokenizer.get_parent_embedding(nid, W_np)
            for nid in new_ids
        ])                                        # (N, d_model)

        # Espansione atomica — un'unica chiamata batchata
        self.model.vocab_expand(init_vecs)
        # PyTorch Adam non necessita di expand_moments():
        # i parametri nuovi non hanno storia → viene inizializzato automaticamente

        self.expansion_count += 1

        if self._dream is not None:
            self._dream.register_new_tokens(new_ids)

        if self.checkpoint:
            self.checkpoint.save(reason=f"expansion_{self.expansion_count}")

        print(f"  [expand] +{len(new_ids)} token → vocab={self.model.vocab_size}  "
              f"(espansione #{self.expansion_count})")
        return new_ids

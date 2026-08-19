"""
DreamConsolidator — updated dormant phase for TorchDynamicGPT.

All training steps use PyTorch autograd:
  optimizer.zero_grad() → forward → loss → loss.backward() → clip → optimizer.step()
No manual backward code.
"""
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))


class DreamConsolidator:

    DREAM_TRIGGER       = 10    # new tokens before dreaming
    DREAM_STEPS         = 150   # gradient steps during the dream
    PRUNE_GRACE         = 5     # consecutive dormant phases without use → removal
    MIN_AGE_DREAMS      = 4     # minimum dreams before eligible for pruning
    BLOCK_SIZE          = 64    # lunghezza sequenze durante il sogno
    BUFFER_CHARS        = 50_000
    MICRO_REPLAY_EVERY  = 50    # every N normal training steps
    MICRO_REPLAY_STEPS  = 5     # lightweight gradient steps on already-encoded buffer

    def __init__(self, model, tokenizer, optimizer):
        self.model     = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer   # TorchAdamOptimizer

        self.tokens_since_dream = 0
        self.dream_count        = 0
        self._unused_grace: Dict[int, int] = defaultdict(int)
        self._token_birth_dream: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Helper interno: singolo step PyTorch
    # ------------------------------------------------------------------

    def _train_step(self, ids: torch.Tensor, max_norm: float = 1.0) -> float:
        """Full step: zero_grad → forward → loss → backward → clip → step."""
        opt = self.optimizer._opt if hasattr(self.optimizer, '_opt') else self.optimizer
        opt.zero_grad()
        logits = self.model.forward(ids)
        loss   = self.model.loss(logits, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        opt.step()
        return loss.item()

    def _to_tensor(self, ids) -> torch.Tensor:
        if isinstance(ids, np.ndarray):
            return torch.from_numpy(ids).long()
        return ids.long()

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def register_new_tokens(self, new_ids: List[int]) -> None:
        for tid in new_ids:
            self._token_birth_dream[tid] = self.dream_count

    def notify_expansion(self, n_new_tokens: int,
                         text_buffer: str) -> Optional[dict]:
        self.tokens_since_dream += n_new_tokens
        if self.tokens_since_dream >= self.DREAM_TRIGGER:
            stats = self.run(text_buffer)
            self.tokens_since_dream = 0
            return stats
        return None

    def micro_replay(self, encoded_buffer: Optional[List[int]]) -> None:
        """Lightweight autonomous reinforcement (default mode network)."""
        if encoded_buffer is None or len(encoded_buffer) < self.BLOCK_SIZE + 1:
            return
        bs      = self.BLOCK_SIZE
        max_start = len(encoded_buffer) - bs - 1
        if max_start <= 0:
            return
        for _ in range(self.MICRO_REPLAY_STEPS):
            start = int(np.random.randint(0, max_start))
            chunk = self._to_tensor(
                np.array(encoded_buffer[start:start + bs + 1], dtype=np.int32))
            self._train_step(chunk, max_norm=0.5)

    def run(self, raw_text_buffer: str) -> dict:
        self.dream_count += 1
        print(f"\n  [dream #{self.dream_count}] starting dormant phase...")

        # 1. Re-encoding
        reencoded = self.tokenizer.encode(raw_text_buffer)
        print(f"  [dream] re-encoding: {len(raw_text_buffer)} chars → "
              f"{len(reencoded)} token (vocab={self.model.vocab_size})")

        # 2. Consolidation
        dream_loss = self._consolidation(reencoded)
        print(f"  [dream] consolidation loss: {dream_loss:.4f}")

        # 3. Pruning
        tokens_in_buf = set(reencoded)
        pruned = self._pruning(tokens_in_buf)
        if pruned:
            print(f"  [dream] pruned {len(pruned)} unused tokens: {pruned}")

        # 4. Defrag
        new_merges = self._defrag(raw_text_buffer)
        if new_merges:
            print(f"  [dream] defrag: +{len(new_merges)} new merges")

        print(f"  [dream #{self.dream_count}] completed.\n")
        return {"dream_count": self.dream_count, "dream_loss": dream_loss,
                "tokens_pruned": len(pruned), "new_merges": len(new_merges)}

    # ------------------------------------------------------------------
    # Step interni della dream phase
    # ------------------------------------------------------------------

    def _consolidation(self, ids: List[int]) -> float:
        bs    = self.BLOCK_SIZE
        if len(ids) < bs + 1:
            return float("nan")
        steps = min(self.DREAM_STEPS, (len(ids) - bs) // bs)
        starts = np.random.choice(range(0, len(ids) - bs, bs),
                                  size=steps, replace=True)
        losses = []
        for s in starts:
            chunk = self._to_tensor(
                np.array(ids[s:s + bs + 1], dtype=np.int32))
            losses.append(self._train_step(chunk))
        return float(np.mean(losses)) if losses else float("nan")

    def _pruning(self, tokens_in_buffer: set) -> List[int]:
        pruned = []
        base_vocab = 256
        for token_id in list(self.tokenizer.token_parents.keys()):
            if token_id < base_vocab:
                continue
            # Check if the token is already "dead" in the model
            # (TorchGPT uses vocab_size; removed tokens are not directly supported)
            birth = self._token_birth_dream.get(token_id, 0)
            if self.dream_count - birth < self.MIN_AGE_DREAMS:
                continue
            if token_id not in tokens_in_buffer:
                self._unused_grace[token_id] += 1
                if self._unused_grace[token_id] >= self.PRUNE_GRACE:
                    # In TorchGPT there is no "soft prune" as in NumPy;
                    # we only record the token as a candidate for future removal
                    pruned.append(token_id)
            else:
                self._unused_grace[token_id] = 0
        return pruned

    def _defrag(self, raw_text: str) -> List[int]:
        new_ids = self.tokenizer.grow(raw_text, n_merges=3)
        if not new_ids:
            return []
        W_np    = self.model.tok_emb.weight.data.cpu().numpy()
        init_vecs = np.stack([
            self.tokenizer.get_parent_embedding(nid, W_np)
            for nid in new_ids
        ])
        self.model.vocab_expand(init_vecs)
        return new_ids

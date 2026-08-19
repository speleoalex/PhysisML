"""
DynamicBPETokenizer — extends BPETokenizer with incremental vocabulary growth.

New features compared to test_1:
  - token_parents: tracks parents of each merged token
  - token_freq:    cumulative frequency observed in corpus
  - grow():        adds new merges from recent text with adaptive threshold
  - get_parent_embedding(): computes the initial vector for a new token
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from splx.tokenizer import BPETokenizer, _apply_merge_list, _get_pairs


class DynamicBPETokenizer(BPETokenizer):

    def __init__(self):
        super().__init__()
        # new_id -> (parent_a_id, parent_b_id)  — None per i 256 byte base
        self.token_parents: Dict[int, Optional[Tuple[int, int]]] = {}
        # frequenza cumulativa di ogni token nel corpus visto finora
        self.token_freq: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Override train() per popolare token_parents
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int = 2000) -> None:
        super().train(text, vocab_size)
        # Ricostruisci token_parents dai merges
        self.token_parents = {}
        for (a, b, new_id) in self.merges:
            self.token_parents[new_id] = (a, b)

    # ------------------------------------------------------------------
    # Incremental growth
    # ------------------------------------------------------------------

    def grow(self, new_text: str, n_merges: int = 10,
             max_words: int = 0,
             protect: Optional[set] = None) -> List[int]:
        """
        Analyse new_text with the current vocabulary, identify the most
        frequent uncovered pairs, create up to n_merges new tokens.

        Merges are found and applied WITHIN word boundaries only — the same
        segmentation used by BPETokenizer._encode_bpe (\\S+ segments,
        whitespace emitted as raw bytes). The previous implementation merged
        over the whole byte stream, spaces included: ~85% of the tokens it
        created contained whitespace and could NEVER be produced by encode()
        — permanently dead vocabulary slots that still competed in the
        weight-tied softmax. Special-token literals (e.g. '<|EOS|>' pasted
        into decoded logs) are stripped before the analysis for the same
        reason: encode() splits them out, so merges learned from them are
        unreachable too.

        Soglia adattiva: max(MIN_ABS, n_tokens * MIN_REL)
          MIN_ABS = 5      — mai creare un token visto < 5 volte
          MIN_REL = 0.002  — almeno 0.2% del buffer.
          (Era 0.5% quando le coppie venivano contate anche ATTRAVERSO gli
          spazi: la segmentazione per parola azzera la coda ad alta frequenza
          delle coppie cross-word, quindi la soglia relativa va più bassa per
          non produrre zero merge sui buffer grandi dei livelli alti.)

        max_words: kept as a defensive filter (with word-boundary
          segmentation new tokens can no longer span words, so it should
          never trigger).

        protect: set of strings that must NOT become single tokens. Merging a
          string the model is currently being drilled on orphans the learned
          multi-token path for it: measured at L0, 'lala!' dropped from 8/59
          to 0/21 correct the session after it became a cold single token.
          Targets are freed once the level moves on.

        Returns: lista di nuovi token_id creati
        """
        if not self._trained:
            raise RuntimeError("Tokenizer non addestrato. Chiama train() prima.")

        import re as _re

        MIN_ABS = 5
        MIN_REL = 0.002

        # Strip special-token literals: grow() must never learn merges
        # from marker text.
        text = new_text
        if self.special_tokens:
            _pat = "|".join(_re.escape(n) for n in self.special_tokens)
            text = _re.sub(_pat, " ", text)

        # Segment like _encode_bpe: one token list per \S+ word segment.
        # Cache per unique segment — growth text is highly repetitive.
        cache: Dict[bytes, List[int]] = {}
        segments: List[List[int]] = []
        for seg in _re.findall(r"\S+", text):
            sb = seg.encode("utf-8")
            if sb not in cache:
                toks = list(sb)
                for (a, b, new_id) in self.merges:
                    toks = _apply_merge_list(toks, a, b, new_id)
                    if len(toks) <= 1:
                        break
                cache[sb] = toks
            segments.append(list(cache[sb]))

        # Aggiorna token_freq
        n_tokens_total = 0
        for seg_toks in segments:
            n_tokens_total += len(seg_toks)
            for t in seg_toks:
                self.token_freq[t] += 1

        threshold = max(MIN_ABS, int(n_tokens_total * MIN_REL))
        new_ids: List[int] = []
        blacklist: set = set()   # pairs rejected by the filters below

        MAX_TOKEN_CHARS = 20   # longest reasonable Italian word; kills
                               # degenerate babble mega-tokens ('babababa…')

        _protect = {p.strip().lower() for p in (protect or set()) if p.strip()}

        def _is_periodic(s: str) -> bool:
            """True for degenerate repetition loops: 'bababa' ('ba'*3),
            'lalalala' ('la'*4), 'aaa'. A double ('baba' = 'ba'*2) is a
            legitimate word and is allowed."""
            for ul in range(1, 5):
                if len(s) >= 3 * ul and len(s) % ul == 0:
                    if s[:ul] * (len(s) // ul) == s:
                        return True
            return False

        for _ in range(n_merges):
            # Conta coppie adiacenti dentro ogni segmento (mai fra parole)
            pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
            for seg_toks in segments:
                for i in range(len(seg_toks) - 1):
                    pair = (seg_toks[i], seg_toks[i + 1])
                    if pair not in blacklist:
                        pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best_pair] < threshold:
                break

            a, b = best_pair
            new_token_bytes = self.vocab[a] + self.vocab[b]

            # Skip if the merge already exists
            if new_token_bytes in self.token_to_id:
                blacklist.add(best_pair)
                continue

            # Sanity filters on the would-be token
            try:
                s = new_token_bytes.decode("utf-8", errors="replace")
            except Exception:
                s = ""
            if s:
                core = s.strip().strip("!.?,:;")
                # Length cap + periodicity: babble loops ('babababa!…') must
                # never become tokens — rejecting the doubled pair here also
                # breaks the merge chain that built the 48-char monsters.
                if len(s) > MAX_TOKEN_CHARS or _is_periodic(core):
                    blacklist.add(best_pair)
                    continue   # try next-best pair
                # Never tokenise a string the model is being drilled on
                if _protect and (s.strip().lower() in _protect
                                 or core.lower() in _protect):
                    blacklist.add(best_pair)
                    continue   # try next-best pair
                # Word-count filter (defensive — see docstring)
                if max_words > 0:
                    word_count = len([w for w in s.split() if w])
                    if word_count > max_words:
                        blacklist.add(best_pair)
                        continue   # try next-best pair

            next_id = max(self.vocab.keys()) + 1
            self.vocab[next_id] = new_token_bytes
            self.token_to_id[new_token_bytes] = next_id
            self.merges.append((a, b, next_id))
            self.token_parents[next_id] = (a, b)

            segments = [_apply_merge_list(s, a, b, next_id) for s in segments]
            new_ids.append(next_id)

        return new_ids

    # ------------------------------------------------------------------
    # Inizializzazione embedding per nuovo token
    # ------------------------------------------------------------------

    def get_parent_embedding(self, token_id: int,
                              W: "np.ndarray") -> "np.ndarray":
        """
        Calcola il vettore iniziale per un nuovo token come combinazione dei genitori.

        v_new = 0.7 * (W[parent_a] + W[parent_b]) / 2  +  0.3 * randn * 0.02

        70%: eredita semantica dai genitori.
        30%: rumore per permettere differenziazione durante il training.

        Se il token non ha genitori (byte base), ritorna randn * 0.02.
        """
        import numpy as np

        d = W.shape[1]
        parents = self.token_parents.get(token_id)

        if parents is None:
            return np.random.randn(d) * 0.02

        a, b = parents
        # I genitori potrebbero essere stati rimossi (ID sparsi) — fallback a randn
        if a >= W.shape[0] or b >= W.shape[0]:
            return np.random.randn(d) * 0.02

        parent_mean = (W[a] + W[b]) / 2.0
        noise = np.random.randn(d) * 0.02
        # With weight-tied LM head: large vectors for new tokens "steal"
        # probability from the softmax of existing sequences before training.
        # We initialise with reduced magnitude (0.05×) — nearly invisible —
        # and let training bring them to the right scale.
        return (0.7 * parent_mean + 0.3 * noise) * 0.05

    def update_freq(self, token_ids: List[int]) -> None:
        """Aggiorna le frequenze cumulative dopo aver visto una sequenza."""
        for t in token_ids:
            self.token_freq[t] += 1

    def __len__(self) -> int:
        return len(self.vocab)

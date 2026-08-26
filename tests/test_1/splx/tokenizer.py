"""
Byte-Pair Encoding (BPE) tokenizer — pure Python (no NumPy needed).

Algorithm (word-frequency variant — fast)
-----------------------------------------
Instead of working on the full byte sequence (slow: O(corpus × merges)),
we work on a word-frequency table:

1. Tokenise the corpus into "words" (splits on whitespace).
   Count how many times each word appears: {"hello": 1200, "world": 450, ...}
2. Represent each word as a tuple of byte-level token IDs.
3. Count pair frequencies as: sum over all words of (pair_count_in_word × word_freq).
4. Merge the best pair everywhere, update counts INCREMENTALLY (only affected words).
5. Stop after (vocab_size - 256) merges.

Special tokens (e.g. EOS) are registered separately from BPE merges and bypass
the merge rules. They occupy specific IDs (e.g. 256 for EOS) and are handled
before BPE encoding. This follows the same convention as GPT-2 (token 50256)
and LLaMA (</s> token).

Encoding applies the learned merge rules greedily (left to right) to a new text.
"""
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_pairs(word: Tuple[int, ...]) -> Dict[Tuple[int, int], int]:
    """Count adjacent pairs in a token tuple."""
    counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for a, b in zip(word, word[1:]):
        counts[(a, b)] += 1
    return counts


def _merge_word(word: Tuple[int, ...], a: int, b: int, new_id: int) -> Tuple[int, ...]:
    """Replace every (a, b) in a word tuple with new_id."""
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:

    # Well-known special token names
    EOS_TOKEN = "<|EOS|>"

    def __init__(self):
        self.vocab:       Dict[int, bytes] = {}   # id -> bytes
        self.token_to_id: Dict[bytes, int] = {}   # bytes -> id
        self.merges:      List[Tuple]      = []   # [(a, b, new_id), ...]
        self._trained = False
        # Special tokens: name -> id (bypass BPE merge rules)
        self.special_tokens: Dict[str, int] = {}  # name -> id
        self.special_ids:   Dict[int, str]  = {}  # id -> name

    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------

    def register_special_token(self, name: str, token_id: int) -> None:
        """
        Register a special token at a given ID.

        Special tokens bypass BPE merge rules and are handled before encoding.
        They occupy a fixed slot in the vocabulary — use dormant slots (e.g.
        256 for EOS, right after the 256 base byte tokens).

        Example:
            tok.register_special_token("<|EOS|>", 256)
        """
        token_bytes = name.encode("utf-8")
        self.vocab[token_id]          = token_bytes
        self.token_to_id[token_bytes] = token_id
        self.special_tokens[name]     = token_id
        self.special_ids[token_id]    = name

    def get_special_id(self, name: str) -> Optional[int]:
        """Return the token ID for a special token, or None if not registered."""
        return self.special_tokens.get(name)

    def is_special(self, token_id: int) -> bool:
        """Return True if the token ID corresponds to a special token."""
        return token_id in self.special_ids

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    # Punctuation that must stay its own token, never fused to a word.
    _PUNCT = set(b"?!.,;:\"'()[]{}<>")

    @classmethod
    def _merge_allowed(cls, tok: bytes) -> bool:
        """Reject a merge that mixes punctuation with anything else.

        Pure-punctuation runs ('...', '?!') stay allowed: they are a single
        pre-token for every tokenizer and cost one slot, not one per word.
        """
        has_p = any(c in cls._PUNCT for c in tok)
        if not has_p:
            return True
        return all(c in cls._PUNCT for c in tok)


    def train(self, text: str, vocab_size: int = 2000) -> None:
        """
        Learn BPE merge rules from raw text.

        Parameters
        ----------
        text       : plain text corpus
        vocab_size : target vocabulary size (must be > 256)

        Note: special tokens should be registered AFTER training via
        register_special_token(). They occupy IDs that would otherwise
        be the first BPE merges, so train with vocab_size - n_special
        merges if you want a specific final size.
        """
        if vocab_size <= 256:
            raise ValueError("vocab_size must be > 256")

        n_merges = vocab_size - 256

        # ---- Byte-level base vocabulary ----
        self.vocab        = {i: bytes([i]) for i in range(256)}
        self.token_to_id  = {bytes([i]): i for i in range(256)}
        self.merges       = []
        # Preserve any already-registered special tokens
        for name, tid in list(self.special_tokens.items()):
            tok_bytes = name.encode("utf-8")
            self.vocab[tid]          = tok_bytes
            self.token_to_id[tok_bytes] = tid

        # ---- Build word-frequency table ----
        word_freq: Dict[Tuple[int, ...], int] = defaultdict(int)
        for word_str in re.findall(r'\S+|\n', text):
            word_bytes = tuple(word_str.encode("utf-8"))
            word_freq[word_bytes] += 1

        print(f"  Unique words: {len(word_freq):,}  —  "
              f"Total word tokens: {sum(word_freq.values()):,}")

        # ---- Initial pair counts (weighted by word frequency) ----
        pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        pair_to_words: Dict[Tuple[int, int], set] = defaultdict(set)

        for word, freq in word_freq.items():
            for pair, cnt in _get_pairs(word).items():
                pair_counts[pair]      += cnt * freq
                pair_to_words[pair].add(word)

        # Start from next available ID (skip special token slots)
        next_id = max(self.vocab.keys()) + 1

        # ---- Merge loop ----
        for merge_idx in range(n_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            if pair_counts[best_pair] < 2:
                break

            a, b = best_pair
            new_token_bytes = self.vocab[a] + self.vocab[b]

            # Skip if this byte sequence is already a special token
            if new_token_bytes in self.token_to_id:
                pair_counts.pop(best_pair, None)
                pair_to_words.pop(best_pair, None)
                continue

            # Never glue punctuation onto a word.
            #
            # Measured on the 8K base vocabulary: 625 words carried several
            # punctuated variants ('cane', 'cane!', 'cane.', 'cane?'; 'casa'
            # also 'casa,') for 1,783 slots — 14% of the vocabulary spent
            # representing the same word four times, with each variant
            # competing in the softmax and receiving a fraction of the
            # statistics. It also makes the vocabulary unexportable: every
            # GPT-2-style pre-tokenizer, including the one llama.cpp and
            # ollama use, splits punctuation off words, so 'chiami?' is
            # re-segmented into 'chiami' + '?' and the prompt reaches the
            # model in a form it never saw.
            if not self._merge_allowed(new_token_bytes):
                pair_counts.pop(best_pair, None)
                pair_to_words.pop(best_pair, None)
                continue

            self.vocab[next_id]               = new_token_bytes
            self.token_to_id[new_token_bytes] = next_id
            self.merges.append((a, b, next_id))

            # ---- Incremental update ----
            affected_words = list(pair_to_words[best_pair])
            for old_word in affected_words:
                if (a, b) not in zip(old_word, old_word[1:]):
                    continue
                freq = word_freq[old_word]
                new_word = _merge_word(old_word, a, b, next_id)

                for pair, cnt in _get_pairs(old_word).items():
                    pair_counts[pair] -= cnt * freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_words[pair].discard(old_word)

                word_freq[new_word] = word_freq.get(new_word, 0) + freq
                del word_freq[old_word]
                for pair, cnt in _get_pairs(new_word).items():
                    pair_counts[pair]      += cnt * freq
                    pair_to_words[pair].add(new_word)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            next_id += 1

            if (merge_idx + 1) % 100 == 0 or merge_idx == n_merges - 1:
                print(f"  BPE merge {merge_idx+1}/{n_merges}  "
                      f"vocab={next_id}  "
                      f"best=({self.vocab[a]!r}+{self.vocab[b]!r})",
                      end="\r")

        print()
        self._trained = True
        print(f"BPE training done. Vocab size: {len(self.vocab)}")

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.

        Special tokens (e.g. <|EOS|>) are matched FIRST before BPE encoding,
        so they are always represented as their registered single ID.
        """
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        # If no special tokens registered, use fast path
        if not self.special_tokens:
            return self._encode_bpe(text)

        # Split on special token boundaries, encode each segment
        result: List[int] = []
        # Build pattern that matches any special token string
        special_pattern = "|".join(re.escape(name) for name in self.special_tokens)
        parts = re.split(f"({special_pattern})", text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                result.append(self.special_tokens[part])
            else:
                result.extend(self._encode_bpe(part))
        return result

    def _encode_bpe(self, text: str) -> List[int]:
        """BPE encoding for a text segment (no special token handling)."""
        result: List[int] = []
        cache: Dict[Tuple[int, ...], List[int]] = {}
        for segment in re.findall(r'\S+|\s+', text):
            seg_bytes = tuple(segment.encode("utf-8"))
            if segment[0] in ' \t\r\n':
                result.extend(seg_bytes)
            else:
                if seg_bytes not in cache:
                    tokens = list(seg_bytes)
                    for (a, b, new_id) in self.merges:
                        tokens = _apply_merge_list(tokens, a, b, new_id)
                        if len(tokens) <= 1:
                            break
                    cache[seg_bytes] = tokens
                result.extend(cache[seg_bytes])
        return result

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to a string, removing non-printable control chars.

        Unknown ids are skipped rather than raising. A model whose vocab_size
        exceeds the tokenizer's (pruned slots, or a checkpoint paired with the
        wrong tokenizer) can sample an id that was never assigned bytes, and
        losing one token beats killing an interactive session with a KeyError.
        """
        raw = b"".join(self.vocab.get(i, b"") for i in ids).decode("utf-8", errors="replace")
        # Strip control chars (bytes 0–8, 11, 12, 14–31) and U+FFFD replacement char
        return "".join(c for c in raw if (c >= " " or c in "\t\n\r") and c != "\ufffd")

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "vocab":          {str(k): v.hex() for k, v in self.vocab.items()},
            "merges":         self.merges,
            "special_tokens": self.special_tokens,  # persisted for reload
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab        = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.token_to_id  = {bytes.fromhex(v): int(k) for k, v in data["vocab"].items()}
        self.merges       = [tuple(int(x) for x in m) for m in data["merges"]]
        self._trained     = True
        # Restore special tokens if present (backward compatible)
        self.special_tokens = {}
        self.special_ids    = {}
        for name, tid in data.get("special_tokens", {}).items():
            self.special_tokens[name] = tid
            self.special_ids[tid]     = name

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Fast list-level apply (used by encode)
# ---------------------------------------------------------------------------

def _apply_merge_list(tokens: List[int], a: int, b: int, new_id: int) -> List[int]:
    """Replace every (a, b) pair in a flat token list with new_id."""
    result = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

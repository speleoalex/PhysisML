"""The `<|NONE|>` row: live for scoring and training, dormant for decoding.

`<|NONE|>` is the eleventh ontology class — the one that means "I have no
class for this referent". It is a real token with a real embedding row, and
the row is trained: gold answers pull P(NONE) down as the model learns a
noun, which is the decay the mechanism relies on (§13 of
docs_internal/curiosita_meccanismo.md).

That row must never be EMITTED. The frozen-row experiment of 2026-09-04
isolated the cost exactly: with every other parameter bit-identical, adding
the row still cost 1.3 points of retention 0-12 and put the token inside two
probe answers, because the row competes in the output argmax while the model
writes. Masking it at decoding time removes that cost completely, not
approximately — max |delta margin| 0.00e+00 over 125 nouns, 0 argmax flips,
retention identical to the checkpoint that never had the row.

So the token has two contracts, and this module is the single place that
names it:

  * scoring and training  the row is live (`active_vocab_size` includes it),
                          so `class_posterior` can read P(NONE) and the
                          optimizer can still move the row;
  * decoding              the row is at -inf, on every path that samples.

Keeping the row live by default is also what makes it survivable:
`_sync_vocab_rows` (dynamic_model/train_curriculum.py) re-initialises every
row in `[active_vocab_size, max(vocab)+1)` with a random direction, so a
checkpoint saved with the row dormant would have its trained row silently
destroyed by the next vocabulary sync.

`mask_scoring_rows` is therefore the decoding-side contract, and the reason
it lowers the integer instead of banning ids downstream is exactness: the
affective modulator reads the raw logits to update its confidence before any
downstream suppression, so masking anywhere later would leave the row
steering the temperature. Lowering `active_vocab_size` is the same operation
the measurement was made with.

Nothing here imports torch or the trainer: the constant has to be importable
from both the epistemic reading and the generation loop without a cycle.
"""
from __future__ import annotations

import contextlib
from typing import Optional, Set, Tuple

# The eleventh class. Minted by scripts_internal/eleventh_class.py at the
# first free id, which is why the row sits at the top of the active range.
NONE_TOKEN = "<|NONE|>"

# Every special token that may be scored but never generated. One entry
# today; the tuple is the extension point, and `mask_scoring_rows` handles a
# contiguous block at the top of the vocabulary exactly and anything else by
# per-step masking.
SCORING_ONLY_TOKENS: Tuple[str, ...] = (NONE_TOKEN,)


def row_id(tok, token: str = NONE_TOKEN) -> Optional[int]:
    """The token's id in this tokenizer, or None when it was never minted.

    Absence is the normal case: every checkpoint before 2026-09-04 has no
    `<|NONE|>` row, and every caller must keep working unchanged on those.
    """
    table = getattr(tok, "special_tokens", None)
    if isinstance(table, dict) and token in table:
        return int(table[token])
    getter = getattr(tok, "get_special_id", None)
    if callable(getter):
        try:
            got = getter(token)
        except (KeyError, ValueError):
            return None
        if got is not None:
            return int(got)
    return None


def scoring_only_ids(tok) -> Set[int]:
    """Ids of the tokens that may be scored but never sampled."""
    return {i for i in (row_id(tok, t) for t in SCORING_ONLY_TOKENS)
            if i is not None}


def scoring_cut(active: int, ids: Set[int]) -> Optional[int]:
    """The value `active_vocab_size` must take to hide exactly `ids`.

    None when the ids do not form a contiguous block at the top of the active
    range — a trained row sits above one of them, and lowering the integer
    would mask legitimate vocabulary. `active` itself when nothing is live to
    hide. Pure, so an exporter that never builds a model can ask the same
    question of a serialised config and tokenizer.
    """
    live = {i for i in ids if i < active}
    if not live:
        return active
    cut = min(live)
    return cut if set(range(cut, active)) == live else None


@contextlib.contextmanager
def mask_scoring_rows(model, tok):
    """Make every scoring-only row unsamplable for the duration of the block.

    Yields the ids the caller still has to suppress itself, as a tuple:

      * empty (the normal case) when the rows form a contiguous block at the
        top of the active range, because then lowering `active_vocab_size`
        masks exactly them inside `TorchGPT.forward` and the caller has
        nothing left to do;
      * the ids, when a trained row sits above one of them — lowering the
        integer would mask legitimate vocabulary, so the caller must set
        those logits to -inf at every step instead.

    The integer is restored in a `finally`, so an exception inside the block
    cannot leave the model unable to score. Re-entrant by construction: the
    inner block finds the rows already dormant and yields nothing.
    """
    ids = scoring_only_ids(tok)
    active = getattr(model, "active_vocab_size", None)
    if not ids or active is None:
        yield ()
        return

    cut = scoring_cut(active, ids)
    if cut == active:                          # already dormant: nothing to do
        yield ()
    elif cut is not None:
        model.active_vocab_size = cut
        try:
            yield ()
        finally:
            model.active_vocab_size = active
    else:
        # A trained row above a scoring-only one: the integer cannot express
        # this, so hand the ids back to the decoding loop.
        yield tuple(sorted(i for i in ids if i < active))

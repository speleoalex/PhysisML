#!/usr/bin/env python3
"""
Register <|EOS|> in the tokenizer snapshots that lack it.

WHY THIS SCRIPT EXISTS
The model has no way to signal "answer finished". Everything needed for one is
already wired: BPETokenizer.register_special_token() / get_special_id(),
TrainerB.generate() (blocks EOS below min_tokens and stops on it),
train_curriculum.py (appends EOS to every gold in SIGNAL 1/3/4), and
export_gguf.py (declares tokenizer.ggml.eos_token_id and marks the token
CONTROL so llama.cpp stops on it). All of it is inert because
special_tokens is EMPTY: scripts/train_tokenizer.py never registered the
token, so the vocabulary retrained on 2026-08-25 dropped the EOS that the
older tokenizer_base.json still carries at id 256.

The visible symptom is outside Python: in ollama the model answers correctly
and then keeps going, continuing the teaching dialogue on its own
('physisml! perfetto! di: il cane grande e fedele.'), because the stop
condition lives in the Python generator and not in the weights.

WHY THE ID MUST BE max(id)+1 AND NOTHING ELSE
_sync_vocab_rows() in train_curriculum.py activates every embedding row up to
max(tokenizer.vocab)+1. Put EOS higher than that and the rows in between
become activated slots with no token behind them: the sampler can emit one and
decode() raises KeyError on it. So EOS goes immediately after the last real
token — and a snapshot whose maximum id differs from the active tokenizer's is
SKIPPED rather than given a different id, because the same weights get loaded
with different snapshots and row 2548 must mean <|EOS|> in all of them.

The five L0-L4 snapshots stop at id 2546 (the vocabulary grew by one token
during L5's dream), so they are skipped by design: resuming from them needs a
tokenizer retrained with EOS included, which renumbers everything
consistently — see scripts/train_tokenizer.py.

Usage:
    python3 scripts/register_eos.py --dry-run
    python3 scripts/register_eos.py
    python3 scripts/register_eos.py --check     # exit 1 if anything is missing
"""
import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "tests", "test_1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physisml.tokenizer import BPETokenizer                     # noqa: E402

# The tokenizer the live model is loaded with: it decides the EOS id, every
# other snapshot has to agree with it.
REFERENCE = "models/active_tokenizer.json"


def targets() -> list:
    """The reference first, then every file that may share the same vocabulary.

    dynamic_model/data/ is in here because a build started from scratch does
    NOT read models/active_tokenizer.json: train_curriculum picks
    TOKENIZER = data/tokenizer_8k.json (falling back to tokenizer_base.json).
    The first version of this script covered the checkpoint snapshots and
    missed exactly the file a from-zero build seeds itself with, which is the
    only one that matters when the checkpoints are about to be wiped.

    Files holding a DIFFERENT vocabulary are filtered out below by
    same_vocab(), not flagged: tokenizer_base.json (503 tokens, EOS at 256)
    and tokenizer_8k.pre_punct_fix.json (8002 tokens, EOS at 8000) are each
    internally consistent and have nothing to do with the active id.
    """
    out = [REFERENCE, "standalone/tokenizer.json"]
    out += sorted(glob.glob("dynamic_model/data/tokenizer*.json"))
    out += sorted(glob.glob("models/checkpoints/*/level_*/tokenizer.json"))
    seen, uniq = set(), []
    for p in out:
        if p not in seen and os.path.exists(p):
            seen.add(p); uniq.append(p)
    return uniq


def same_vocab(tok: BPETokenizer, ref: BPETokenizer) -> bool:
    """True when two tokenizers are the same vocabulary, EOS aside.

    Compared on the real tokens only, and on their bytes rather than their
    count: two vocabularies of equal size can still be different vocabularies.
    """
    a, b = last_bpe_id(tok), last_bpe_id(ref)
    if a != b:
        return False
    probes = {0, 128, 255, 256, a // 2, a - 1, a} & set(ref.vocab)
    return all(tok.vocab.get(i) == ref.vocab.get(i) for i in probes)


def last_bpe_id(tok: BPETokenizer) -> int:
    """The highest id that is a real token, special ones excluded.

    max(vocab) is the wrong question: register_special_token() writes the
    special token INTO vocab, so once EOS is registered it becomes the
    maximum itself. Using max(vocab) made a second run compute id max+1 = 2549
    and report every already-correct file as misaligned — the script was not
    idempotent, which for a migration is the one property that matters.
    """
    return max(i for i in tok.vocab if not tok.is_special(i))


def max_id(path: str) -> int:
    tok = BPETokenizer()
    tok.load(path)
    return last_bpe_id(tok)


def main() -> int:
    global REFERENCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change, write nothing")
    ap.add_argument("--check", action="store_true",
                    help="Exit 1 if any eligible snapshot lacks EOS")
    ap.add_argument("--files", nargs="*", default=None,
                    help="Override the file list (the reference stays first)")
    ap.add_argument("--reference", default=REFERENCE,
                    help="Tokenizer that decides the EOS id "
                         "(default: %(default)s)")
    a = ap.parse_args()
    REFERENCE = a.reference

    if not os.path.exists(REFERENCE):
        print(f"✗ reference tokenizer missing: {REFERENCE}")
        return 1

    ref = BPETokenizer(); ref.load(REFERENCE)
    eos_id = last_bpe_id(ref) + 1
    print(f"Reference : {REFERENCE}  →  {BPETokenizer.EOS_TOKEN} = id {eos_id}\n")

    files = a.files if a.files else targets()
    n_done = n_skip = n_bad = n_already = 0

    for path in files:
        tok = BPETokenizer()
        tok.load(path)
        have = tok.get_special_id(BPETokenizer.EOS_TOKEN)
        top  = last_bpe_id(tok)

        if path != REFERENCE and not same_vocab(tok, ref):
            print(f"  – {path:<48} skipped: different vocabulary "
                  f"({top + 1} tokens, EOS "
                  f"{'at ' + str(have) if have is not None else 'absent'})")
            n_skip += 1
            continue

        if have is not None:
            if have == eos_id:
                print(f"  = {path:<48} already registered (id {have})")
                n_already += 1
            else:
                print(f"  ✗ {path:<48} EOS at id {have}, not {eos_id} — "
                      f"misaligned, fix by hand")
                n_bad += 1
            continue

        if top + 1 != eos_id:
            print(f"  – {path:<48} skipped: max id {top}, EOS at {eos_id} "
                  f"would leave {eos_id - top - 1} empty slots")
            n_skip += 1
            continue

        if a.check:
            print(f"  ✗ {path:<48} EOS missing")
            n_bad += 1
            continue

        tok.register_special_token(BPETokenizer.EOS_TOKEN, eos_id)
        if not a.dry_run:
            tok.save(path)
        print(f"  {'~' if a.dry_run else '✓'} {path:<48} "
              f"EOS → id {eos_id} (vocab {top + 1} → {len(tok)})")
        n_done += 1

    print(f"\nregistered {n_done}, already present {n_already}, "
          f"skipped {n_skip}, problems {n_bad}")
    if a.dry_run:
        print("(dry-run: no file written)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())

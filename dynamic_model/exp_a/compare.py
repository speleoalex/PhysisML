"""
Comparison of three models — PyTorch CPU backend with mini-batching.

  baseline-501  : TorchGPT fixed, vocab 501 tokens
  exp-a         : TorchDynamicGPT, vocab grows dynamically from 501
  baseline-2000 : TorchGPT fixed, vocab 2000 tokens (test_1)

Recommended config on this hardware (12 threads, MKL):
  d_model=256, n_heads=4, n_layers=4, d_ff=1024, batch_size=8
  → ~59 seq/s  (vs 11 seq/s for NumPy baseline d=128)
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tests', 'test_1'))

import numpy as np
import torch
import time

from physisml.torch_model import TorchGPT, TorchAdamOptimizer, ids_batch
from physisml.tokenizer   import BPETokenizer

from dynamic_model.core.tokenizer           import DynamicBPETokenizer
from dynamic_model.exp_a.transformer        import TorchDynamicGPT
from dynamic_model.exp_a.expansion_manager  import VocabExpansionManager
from dynamic_model.exp_a.dream_consolidator import DreamConsolidator

torch.set_num_threads(12)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

D_MODEL    = 256
N_HEADS    = 4
N_LAYERS   = 4
D_FF       = 1024
BLOCK_SIZE = 128
BATCH_SIZE = 8
EVAL_EVERY = 50_000      # chars between evaluations
# With batch=8 and block=128, each step = 8×128 = 1024 chars.
# EXPANSION_INTERVAL=10 → first expansion at ~10K chars (well before freeze)
EXPANSION_INTERVAL = 10
EXPANSION_FREEZE   = 150_000  # freeze after 150K chars (end of levels 0+1+initial Pinocchio)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_training_text() -> str:
    parts = []
    for level in ["0", "1", "2"]:
        for path in sorted(glob.glob(f"tests/test_1/data/it/{level}/*.txt")):
            with open(path, encoding="utf-8") as f:
                text = f.read()
            parts.append(text)
            print(f"  + [it/{level}] {os.path.basename(path):35s} {len(text):>8,} chars")
    return "\n\n".join(parts)

def load_validation_text() -> str:
    with open("dynamic_model/data/validation.txt", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Eval perplexity (PyTorch, no grad)
# ---------------------------------------------------------------------------

def eval_ppl(model: TorchGPT, tokenizer, val_text: str) -> float:
    ids_np = np.array(tokenizer.encode(val_text), dtype=np.int32)
    if len(ids_np) < 2:
        return float("nan")
    losses = []
    with torch.no_grad():
        for start in range(0, len(ids_np) - BLOCK_SIZE, BLOCK_SIZE):
            chunk = torch.from_numpy(ids_np[start:start + BLOCK_SIZE + 1]).long()
            logits = model.forward(chunk)
            losses.append(model.loss(logits, chunk).item())
    return float(np.exp(np.mean(losses))) if losses else float("nan")

# ---------------------------------------------------------------------------
# Training con mini-batching e checkpoint di perplexity
# ---------------------------------------------------------------------------

def train_one_epoch(model, tokenizer, optimizer, all_ids, text,
                    chars_per_token, step_offset, eval_every,
                    next_eval_chars, chars_seen, ppl_curve, loss_window,
                    val_text, name, chars_processed_offset,
                    expansion_manager=None, dream=None,
                    sequential: bool = False):
    """
    sequential=True: progressive corpus order (required for dynamic vocabulary).
    sequential=False: random shuffle (optimal for fixed vocab).
    """
    opt = optimizer._opt if hasattr(optimizer, '_opt') else optimizer
    step = step_offset
    B    = BATCH_SIZE
    bs   = BLOCK_SIZE
    chars_processed = chars_processed_offset

    starts = list(range(0, len(all_ids) - bs, bs))
    if not sequential:
        np.random.shuffle(starts)

    i = 0
    while i < len(starts):
        batch_s = starts[i:i + B]
        i += B

        seqs = [all_ids[s:s + bs + 1] for s in batch_s
                if s + bs + 1 <= len(all_ids)]
        if not seqs:
            continue

        batch = torch.from_numpy(np.stack(seqs)).long()

        opt.zero_grad()
        logits = model.forward(batch)
        loss   = model.loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_val = loss.item()
        loss_window.append(loss_val)
        step += 1
        chars_processed += len(seqs) * bs  # chars actually processed

        # Dynamic updates (Exp-A only)
        if expansion_manager is not None:
            # In sequential mode batch_s is monotonic → corpus position = chars seen
            # In shuffled mode we use chars_processed as an accurate counter
            cs = int(batch_s[0] * chars_per_token)
            ce = min(int((batch_s[-1] + bs) * chars_per_token) + 1, len(text))
            expansion_manager.add_text(text[cs:ce])
            expansion_manager._chars_seen = chars_processed

            new_ids = expansion_manager.maybe_expand(step)
            if new_ids and dream is not None:
                raw = "".join(list(expansion_manager.recent_text_buffer))
                dream.notify_expansion(len(new_ids), raw)

            if dream is not None and step % dream.MICRO_REPLAY_EVERY == 0:
                enc_buf = list(all_ids[max(0, batch_s[0] - 2000):batch_s[0]])
                dream.micro_replay(enc_buf)

        # Checkpoint based on chars_processed (monotonic)
        if chars_processed >= next_eval_chars:
            ppl      = eval_ppl(model, tokenizer, val_text)
            avg_loss = np.mean(loss_window[-30:]) if loss_window else float("nan")
            chars_seen.append(chars_processed)
            ppl_curve.append(ppl)
            print(f"  [{name}]  chars={chars_processed:7,}  step={step:4d}  "
                  f"loss={avg_loss:.3f}  ppl={ppl:.1f}  vocab={model.vocab_size}")
            next_eval_chars += eval_every

    return step, next_eval_chars, chars_processed


def train_and_eval(name, model, tokenizer, optimizer, text, val_text,
                   eval_every=EVAL_EVERY, n_epochs=2,
                   expansion_manager=None, dream=None):

    chars_seen      = [0]
    ppl_curve       = [eval_ppl(model, tokenizer, val_text)]
    loss_window     = []
    step_offset     = 0
    chars_processed = 0
    next_eval       = eval_every

    for epoch in range(1, n_epochs + 1):
        all_ids         = np.array(tokenizer.encode(text), dtype=np.int32)
        chars_per_token = len(text) / max(len(all_ids), 1)

        print(f"\n  [{name}] epoch {epoch}/{n_epochs}  "
              f"(vocab={model.vocab_size}, tokens={len(all_ids):,})")

        # Dynamic vocabulary → sequential (new tokens appear in the text
        # progressively). Fixed vocab → shuffle (better for convergence).
        seq = (expansion_manager is not None)

        step_offset, next_eval, chars_processed = train_one_epoch(
            model, tokenizer, optimizer, all_ids, text,
            chars_per_token, step_offset, eval_every,
            next_eval, chars_seen, ppl_curve, loss_window, val_text, name,
            chars_processed, expansion_manager, dream,
            sequential=seq
        )

    return chars_seen, ppl_curve, model.vocab_size

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    train_text = load_training_text()
    val_text   = load_validation_text()
    print(f"  Training: {len(train_text):,} chars")
    print(f"  Validation: {len(val_text):,} chars")
    print(f"\nConfig: d={D_MODEL}, L={N_LAYERS}, H={N_HEADS}, batch={BATCH_SIZE}\n")

    tok_501 = DynamicBPETokenizer()
    tok_501.load("dynamic_model/data/tokenizer_base.json")
    tok_2000 = BPETokenizer()
    tok_2000.load("tests/test_1/checkpoints/it-0/tokenizer.json")
    print(f"Tokenizer base: {len(tok_501)} token")
    print(f"Tokenizer 2000: {len(tok_2000)} token\n")

    results = {}

    # ================================================================
    # Baseline-501
    # ================================================================
    print("=" * 58)
    print("BASELINE-501  (TorchGPT fixed, vocab 501)")
    print("=" * 58)
    t0 = time.time()
    m501 = TorchGPT(len(tok_501), D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SIZE+1)
    o501 = TorchAdamOptimizer(m501.parameters(), lr=1e-3)
    cs, pc, vs = train_and_eval("base-501", m501, tok_501, o501,
                                train_text, val_text, n_epochs=2)
    results["base-501"] = {"chars": cs, "ppl": pc, "vocab": vs,
                           "time": time.time()-t0}
    print(f"  → final: ppl={pc[-1]:.1f}  time={results['base-501']['time']:.0f}s\n")

    # ================================================================
    # Experiment A
    # ================================================================
    print("=" * 58)
    print("EXPERIMENT A  (TorchDynamicGPT, vocab from 501)")
    print("=" * 58)
    t0 = time.time()
    tokA = DynamicBPETokenizer()
    tokA.load("dynamic_model/data/tokenizer_base.json")
    mA   = TorchDynamicGPT(len(tokA), D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SIZE+1)
    oA   = TorchAdamOptimizer(mA.parameters(), lr=1e-3)

    exp_mgr = VocabExpansionManager(mA, tokA, oA)
    exp_mgr.EXPANSION_INTERVAL     = EXPANSION_INTERVAL
    exp_mgr.EXPANSION_FREEZE_AFTER = EXPANSION_FREEZE
    # With PyTorch batch training, consolidation happens in the second epoch
    # (re-encoding with updated vocab). Dream phase disabled.
    # dream = DreamConsolidator(mA, tokA, oA)
    # exp_mgr.set_dream_consolidator(dream)

    cs, pc, vs = train_and_eval("exp-a", mA, tokA, oA,
                                train_text, val_text, n_epochs=2,
                                expansion_manager=exp_mgr, dream=None)
    results["exp-a"] = {"chars": cs, "ppl": pc, "vocab": vs,
                        "time": time.time()-t0}
    print(f"  → final: ppl={pc[-1]:.1f}  vocab={vs}  "
          f"time={results['exp-a']['time']:.0f}s\n")

    # ================================================================
    # Baseline-2000
    # ================================================================
    print("=" * 58)
    print("BASELINE-2000  (TorchGPT fixed, vocab 2000)")
    print("=" * 58)
    t0 = time.time()
    m2000 = TorchGPT(len(tok_2000), D_MODEL, N_HEADS, N_LAYERS, D_FF, BLOCK_SIZE+1)
    o2000 = TorchAdamOptimizer(m2000.parameters(), lr=1e-3)
    cs, pc, vs = train_and_eval("base-2000", m2000, tok_2000, o2000,
                                train_text, val_text, n_epochs=2)
    results["base-2000"] = {"chars": cs, "ppl": pc, "vocab": vs,
                            "time": time.time()-t0}
    print(f"  → final: ppl={pc[-1]:.1f}  time={results['base-2000']['time']:.0f}s\n")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 58)
    print("FINAL RESULTS")
    print("=" * 58)
    print(f"{'Model':15s}  {'Vocab':>8s}  {'PPL':>8s}  {'Time':>8s}")
    print("-" * 58)
    for name, r in results.items():
        print(f"  {name:13s}  {r['vocab']:>8d}  {r['ppl'][-1]:>8.1f}  "
              f"{r['time']:>7.0f}s")

    print("\nPerplexity per checkpoint (concatenated epochs):")
    header = f"  {'chars':>8s}" + "".join(f"  {n:>10s}" for n in results)
    print(header)
    max_pts = max(len(r["chars"]) for r in results.values())
    for i in range(max_pts):
        row = f"  {results[list(results)[0]]['chars'][i] if i < len(results[list(results)[0]]['chars']) else 0:>8,}"
        for r in results.values():
            row += f"  {r['ppl'][i]:>10.1f}" if i < len(r["ppl"]) else f"  {'—':>10s}"
        print(row)


if __name__ == "__main__":
    main()

"""
Inference con statistiche token-by-token.

Uso:
  python3 -m dynamic_model.infer --model models/active.pt --prompt "il cane"
  python3 -m dynamic_model.infer --model dynamic_model/checkpoints/level_0/final.pt \
      --prompt "mamma" --max_tokens 40 --temperature 0.7 --top_k 20
"""
import sys, os, argparse
import numpy as np
import torch
import torch.nn.functional as F

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, 'tests', 'test_1')
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splx.torch_model import TorchGPT
from splx.tokenizer   import BPETokenizer
from splx.utils       import sample_top_k, softmax

torch.set_num_threads(12)

TOKENIZER_PATH = "dynamic_model/data/tokenizer_base.json"


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def _top_candidates(logits_np: np.ndarray, tok: BPETokenizer,
                    temperature: float, k: int = 5):
    scaled = logits_np / max(temperature, 1e-8)
    probs  = softmax(scaled)
    top_idx = np.argsort(probs)[::-1][:k]
    return [(tok.decode([int(i)]), float(probs[i])) for i in top_idx]


def generate_with_stats(model: TorchGPT, tok: BPETokenizer,
                        prompt: str, max_tokens: int,
                        temperature: float, top_k: int) -> dict:
    """
    Generate text collecting, for each generated token:
      - chosen token and its probability
      - distribution entropy
      - top-5 candidates with probabilities
      - cumulative log-likelihood
    """
    prompt_ids = tok.encode(prompt)
    ids = list(prompt_ids)

    token_stats = []   # una voce per ogni token generato
    total_log_prob = 0.0

    model.eval()
    with torch.no_grad():
        for step in range(max_tokens):
            ctx    = torch.tensor(ids[-128:], dtype=torch.long)
            logits = model.forward(ctx)          # (T, V)
            last   = logits[-1].numpy()          # (V,)

            # Distribuzione con temperatura
            scaled = last / max(temperature, 1e-8)
            probs  = softmax(scaled)

            # Top-k sampling
            next_id = sample_top_k(last, k=top_k, temperature=temperature)
            ids.append(next_id)

            token_str  = tok.decode([next_id])
            token_prob = float(probs[next_id])
            entropy    = _entropy(probs)
            top5       = _top_candidates(last, tok, temperature, k=5)
            total_log_prob += float(np.log(max(token_prob, 1e-12)))

            token_stats.append({
                "step":       step + 1,
                "token_id":   next_id,
                "token_str":  token_str,
                "prob":       token_prob,
                "entropy":    entropy,
                "top5":       top5,
            })

    generated_ids  = ids[len(prompt_ids):]
    generated_text = tok.decode(generated_ids)
    full_text      = tok.decode(ids)

    n = len(token_stats)
    avg_entropy    = float(np.mean([s["entropy"]  for s in token_stats])) if n else 0.0
    avg_prob       = float(np.mean([s["prob"]     for s in token_stats])) if n else 0.0
    perplexity     = float(np.exp(-total_log_prob / max(n, 1)))

    return {
        "prompt":         prompt,
        "generated_text": generated_text,
        "full_text":      full_text,
        "token_stats":    token_stats,
        "summary": {
            "n_tokens":       n,
            "avg_prob":       avg_prob,
            "avg_entropy":    avg_entropy,
            "perplexity":     perplexity,
            "total_log_prob": total_log_prob,
        },
    }


def print_report(result: dict, verbose: bool) -> None:
    s = result["summary"]

    print("\n" + "=" * 60)
    print(f"  PROMPT:    {result['prompt']!r}")
    print(f"  GENERATED: {result['generated_text']!r}")
    print("=" * 60)
    print(f"  Token generati : {s['n_tokens']}")
    print(f"  Perplexity     : {s['perplexity']:.2f}")
    print(f"  Entropia media : {s['avg_entropy']:.3f} bit")
    print(f"  Prob. media    : {s['avg_prob']:.4f}")
    print(f"  Log-likelihood : {s['total_log_prob']:.3f}")
    print("=" * 60)

    if verbose:
        print(f"\n  {'Step':>4}  {'Token':<12} {'Prob':>6}  {'Entropy':>7}  Top-3")
        print(f"  {'-'*4}  {'-'*12} {'-'*6}  {'-'*7}  {'-'*30}")
        for st in result["token_stats"]:
            top3 = "  ".join(
                f"{tok_str!r}({p:.2f})" for tok_str, p in st["top5"][:3]
            )
            print(f"  {st['step']:>4}  {st['token_str']!r:<12} "
                  f"{st['prob']:>6.4f}  {st['entropy']:>7.3f}  {top3}")
        print()

    # Distribuzione entropia (bassa=sicuro, alta=incerto)
    entropies = [s["entropy"] for s in result["token_stats"]]
    if entropies:
        low  = sum(1 for e in entropies if e < 3)
        mid  = sum(1 for e in entropies if 3 <= e < 6)
        high = sum(1 for e in entropies if e >= 6)
        total = len(entropies)
        print(f"  Sicurezza token:")
        print(f"    Alta   (H<3 bit): {low:3d} / {total}  ({100*low/total:.0f}%)")
        print(f"    Media  (3-6 bit): {mid:3d} / {total}  ({100*mid/total:.0f}%)")
        print(f"    Bassa  (H>6 bit): {high:3d} / {total}  ({100*high/total:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Inference con statistiche token-by-token")
    parser.add_argument("--model",       required=True,           help="Path al checkpoint .pt")
    parser.add_argument("--prompt",      required=True,           help="Testo di input")
    parser.add_argument("--tokenizer",   default=TOKENIZER_PATH)
    parser.add_argument("--max_tokens",  type=int,   default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--verbose",     action="store_true",
                        help="Mostra tabella token-by-token con top-3 candidati")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Errore: modello non trovato: {args.model}", file=sys.stderr)
        sys.exit(1)

    tok = BPETokenizer()
    tok.load(args.tokenizer)

    print(f"Model:    {args.model}")
    model = TorchGPT.load(args.model)
    print(f"Params:   {model.num_params:,}  vocab={model.vocab_size}")

    result = generate_with_stats(
        model, tok,
        prompt      = args.prompt,
        max_tokens  = args.max_tokens,
        temperature = args.temperature,
        top_k       = args.top_k,
    )

    print_report(result, verbose=args.verbose)


if __name__ == "__main__":
    main()

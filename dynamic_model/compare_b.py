"""
Confronto tra due checkpoint exp_b.

Misura:
  - Perplexity su validation.txt
  - Generazione su prompt fissi
  - Delta pesi (quanto i pesi si sono spostati tra i due modelli)

Uso:
  python3 -m dynamic_model.compare_b \
      --model_a dynamic_model/checkpoints/level_0/final.pt \
      --model_b models/active.pt
"""
import sys, os, argparse
import numpy as np
import torch

_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, 'tests', 'test_1')
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from splx.torch_model import TorchGPT
from splx.tokenizer   import BPETokenizer
from splx.utils       import sample_top_k

torch.set_num_threads(12)

TOKENIZER_PATH = "dynamic_model/data/tokenizer_base.json"
VALIDATION_PATH = "dynamic_model/data/validation.txt"
BLOCK_SIZE = 128

PROMPTS = [
    "il cane",
    "mamma",
    "uno più uno",
    "il mare è",
    "la casa",
]


def compute_perplexity(model: TorchGPT, tok: BPETokenizer, text: str) -> float:
    ids_all = np.array(tok.encode(text), dtype=np.int32)
    if len(ids_all) < 2:
        return float("nan")

    total_loss = 0.0
    n_batches = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids_all) - BLOCK_SIZE, BLOCK_SIZE):
            chunk = ids_all[start:start + BLOCK_SIZE + 1]
            if len(chunk) < 2:
                continue
            ids = torch.from_numpy(chunk).long()
            logits = model.forward(ids)
            loss = model.loss(logits, ids)
            total_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        # Short text: single forward
        ids = torch.from_numpy(ids_all).long()
        logits = model.forward(ids)
        loss = model.loss(logits, ids)
        return float(np.exp(loss.item()))

    return float(np.exp(total_loss / n_batches))


def generate(model: TorchGPT, tok: BPETokenizer,
             prompt: str, max_tokens: int = 60,
             temperature: float = 0.8, top_k: int = 40) -> str:
    model.eval()
    ids = tok.encode(prompt)
    with torch.no_grad():
        for _ in range(max_tokens):
            ctx = torch.tensor(ids[-128:], dtype=torch.long)
            logits = model.forward(ctx)
            last = logits[-1].numpy()
            # apply temperature
            last = last / temperature
            next_id = sample_top_k(last, k=top_k, temperature=1.0)
            ids.append(next_id)
    return tok.decode(ids)


def weight_delta(model_a: TorchGPT, model_b: TorchGPT) -> dict:
    sd_a = {k: v.float() for k, v in model_a.state_dict().items()}
    sd_b = {k: v.float() for k, v in model_b.state_dict().items()}

    results = {}
    total_l2 = 0.0
    total_params = 0

    for key in sd_a:
        if key not in sd_b:
            continue
        diff = (sd_b[key] - sd_a[key]).abs()
        l2   = diff.norm().item()
        nparams = diff.numel()
        results[key] = {
            "l2":    l2,
            "mean":  diff.mean().item(),
            "max":   diff.max().item(),
            "n":     nparams,
        }
        total_l2 += l2 ** 2
        total_params += nparams

    results["__total__"] = {
        "l2":    float(np.sqrt(total_l2)),
        "n":     total_params,
        "l2_per_param": float(np.sqrt(total_l2) / total_params),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", default="dynamic_model/checkpoints/level_0/final.pt")
    parser.add_argument("--model_b", default="models/active.pt")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--validation", default=VALIDATION_PATH)
    parser.add_argument("--max_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    tok = BPETokenizer()
    tok.load(args.tokenizer)

    print(f"Loading  A: {args.model_a}")
    model_a = TorchGPT.load(args.model_a)
    print(f"Loading  B: {args.model_b}")
    model_b = TorchGPT.load(args.model_b)
    print(f"Params   A: {model_a.num_params:,}  B: {model_b.num_params:,}\n")

    # --- Perplexity ---
    with open(args.validation, encoding="utf-8") as f:
        val_text = f.read()

    ppl_a = compute_perplexity(model_a, tok, val_text)
    ppl_b = compute_perplexity(model_b, tok, val_text)
    delta_ppl = ppl_b - ppl_a

    print("=" * 55)
    print(f"  PERPLEXITY su {args.validation}")
    print(f"  A ({os.path.basename(args.model_a):>20}): {ppl_a:8.2f}")
    print(f"  B ({os.path.basename(args.model_b):>20}): {ppl_b:8.2f}")
    sign = "+" if delta_ppl >= 0 else ""
    verdict = "PEGGIORATO" if delta_ppl > 0 else "MIGLIORATO" if delta_ppl < 0 else "INVARIATO"
    print(f"  Delta B-A:                          {sign}{delta_ppl:8.2f}  [{verdict}]")
    print()

    # --- Delta pesi ---
    delta = weight_delta(model_a, model_b)
    tot = delta["__total__"]
    print("=" * 55)
    print(f"  WEIGHT DELTA (L2 norm of differences)")
    print(f"  L2 totale:      {tot['l2']:.4f}")
    print(f"  L2 per param:   {tot['l2_per_param']:.6f}")
    print(f"  Totale param:   {tot['n']:,}")
    print()

    # Top 5 most modified layers
    layers = [(k, v["l2"]) for k, v in delta.items() if k != "__total__"]
    layers.sort(key=lambda x: x[1], reverse=True)
    print("  Most modified layers (top 5):")
    for name, l2 in layers[:5]:
        print(f"    {name:<45} L2={l2:.4f}")
    print()

    # --- Generazione ---
    print("=" * 55)
    print(f"  GENERAZIONE (temp={args.temperature}, top_k={args.top_k})")
    print()
    for prompt in PROMPTS:
        out_a = generate(model_a, tok, prompt, args.max_tokens, args.temperature, args.top_k)
        out_b = generate(model_b, tok, prompt, args.max_tokens, args.temperature, args.top_k)
        gen_a = out_a[len(prompt):]
        gen_b = out_b[len(prompt):]
        print(f"  Prompt: '{prompt}'")
        print(f"  A: {gen_a[:120]!r}")
        print(f"  B: {gen_b[:120]!r}")
        print()


if __name__ == "__main__":
    main()

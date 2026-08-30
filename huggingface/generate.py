"""
PhysisML — inference script for the Hugging Face copy of the model.

Loads config.json, model.safetensors and tokenizer.json from the folder this
file sits in.

    pip install torch safetensors numpy

    python3 generate.py "di: cosa mangia il cane?"   # one answer
    python3 generate.py                              # interactive REPL
    python3 generate.py --no-affect "di: il cane"    # plain transformer

This is a port of the generation path used in the training repository
(`dynamic_model/exp_b/trainer.py`), so it keeps the parts that shape the
model's answers:

  * affective modulation — confidence / pleasure / pain / fear shift the logits
    at every step (on by default; `--no-affect` disables it)
  * EOS suppressed before `--min-tokens`, otherwise the model answers with a
    bare end-of-sequence token
  * soft stop on sentence-ending punctuation, tested on the DECODED token
    because the vocabulary has multi-character tokens carrying the terminator
  * output slots the tokenizer cannot decode are masked out

One thing this script cannot reproduce: the published exact-match scores were
measured with the stop threshold set from the length of the gold answer, which
is information a free-form prompt does not have. Expect the model to keep
talking past its answer more often here.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    from physisml.torch_model import TorchGPT
    from physisml.tokenizer   import BPETokenizer
    from physisml.utils       import sample_top_k
except ImportError:
    print("Error: the 'physisml' package is missing next to this script.\n"
          "       Download the whole model repo, not just generate.py:\n"
          "       hf download speleoalex/physisml-it-preview --local-dir physisml-model",
          file=sys.stderr)
    sys.exit(1)


def load_model(folder: str):
    """Build the model from config.json and fill it from model.safetensors."""
    from safetensors.torch import load_file

    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    model = TorchGPT(
        vocab_size        = cfg["vocab_size"],
        d_model           = cfg["d_model"],
        n_heads           = cfg["n_heads"],
        n_layers          = cfg["n_layers"],
        d_ff              = cfg["d_ff"],
        max_seq_len       = cfg["max_seq_len"],
        dropout_p         = 0.0,
        active_vocab_size = cfg.get("active_vocab_size", cfg["vocab_size"]),
    )
    # lm_head.weight is tied to tok_emb.weight and is therefore NOT stored in
    # the file — TorchGPT.__init__ re-ties it. strict=False is required for
    # that one key and for nothing else, so check exactly that.
    missing, unexpected = model.load_state_dict(
        load_file(os.path.join(folder, "model.safetensors")), strict=False)
    tied = set(cfg.get("tied_weights", {}))
    if set(missing) - tied or unexpected:
        raise RuntimeError(
            f"checkpoint does not match config.json — missing {list(missing)}, "
            f"unexpected {list(unexpected)}")
    model.eval()
    return model, cfg


def undecodable_mask(model, tok):
    """
    Mask over the output layer marking ids the tokenizer cannot turn back into
    bytes. vocab_size is rounded up past the tokenizer's real size, so several
    hundred slots decode to nothing; sampling one derails the whole answer.
    """
    vocab = getattr(tok, "vocab", None)
    if not vocab:
        return None
    missing = [i for i in range(model.vocab_size) if i not in vocab]
    if not missing:
        return None
    mask = torch.zeros(model.vocab_size, dtype=torch.bool)
    mask[torch.tensor(missing, dtype=torch.long)] = True
    return mask


def clean_response(text: str) -> str:
    """
    Strip the two noise artifacts the model learned: a leading '!' (it picked
    up the terminator as a start token) and an echoed 'di:' from the prompt.
    Conservative — leading noise only, never content.
    """
    t = re.sub(r"^[!\s]+", "", text.strip()).strip()
    t = re.sub(r"(?:il|la|lo|le|gli|un|una)\s+di:\s*", "", t,
               flags=re.IGNORECASE).strip()
    t = re.sub(r"di:\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t or text.strip()


def generate(model, tok, prompt: str, *, max_tokens: int, temperature: float,
             top_k: int, min_tokens: int, stop_after: int, context: int,
             modulator=None, affect=None, mask=None) -> str:
    ids = tok.encode(prompt)[-(model.max_seq_len - 1):]
    if not ids:
        return ""
    n_prompt = len(ids)
    eos_id   = tok.get_special_id(tok.EOS_TOKEN)

    with torch.no_grad():
        for step in range(max_tokens):
            ctx    = torch.tensor(ids[-context:], dtype=torch.long)
            logits = model.forward(ctx)[-1]

            if modulator is not None:
                affect.update_from_logits(logits, model.vocab_size)
                # The modulator applies the temperature itself, so the sampler
                # is called with 1.0 afterwards. Greedy therefore cannot be
                # expressed as temperature 0 here: it is top_k = 1, exactly as
                # the training repo's own evaluation harness does it. Passing
                # temperature 0 and leaving top_k at 40 sampled instead —
                # 'cos è un cane?' answered animale, persona and luce on three
                # consecutive runs of what claimed to be greedy decoding.
                logits = modulator.modulate(
                    logits, temperature if temperature > 0 else 1.0,
                    response_len = len(ids) - n_prompt,
                    eos_min_len  = min_tokens + 1,
                )
                sample_temperature = 1.0
                sample_k = 1 if temperature <= 0 else top_k
            else:
                sample_temperature = temperature
                sample_k = top_k

            logits = logits.clone()
            if eos_id is not None and (len(ids) - n_prompt) < min_tokens \
                    and eos_id < logits.shape[0]:
                logits[eos_id] = float("-inf")
            if mask is not None:
                logits[mask] = float("-inf")

            if sample_temperature <= 0 or sample_k == 1:
                next_id = int(torch.argmax(logits).item())
            else:
                next_id = int(sample_top_k(logits.numpy(), k=sample_k,
                                           temperature=sample_temperature))
            ids.append(next_id)

            if eos_id is not None and next_id == eos_id and step >= 1:
                break

            # Soft stop on sentence-ending punctuation. Test the decoded token:
            # multi-character tokens ('la!', 'no!') carry the terminator too.
            if step >= stop_after:
                try:
                    tail = tok.decode([next_id]).rstrip()
                except Exception:
                    tail = "!" if next_id in (33, 46, 63) else ""
                if tail.endswith(("!", ".", "?")) and (step >= 1 or len(tail) > 1):
                    break

    return clean_response(tok.decode(ids[n_prompt:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="PhysisML text generation")
    ap.add_argument("prompt", nargs="*", help="prompt (omit for a REPL)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (default). The published scores are greedy")
    ap.add_argument("--top-k", type=int, default=40,
                    help="only used when temperature > 0")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--min-tokens", type=int, default=4,
                    help="EOS is suppressed before this many response tokens")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="allow the punctuation soft stop after this many "
                         "tokens (default: --min-tokens)")
    ap.add_argument("--no-affect", action="store_true",
                    help="disable affective modulation of the logits")
    ap.add_argument("--folder", default=_HERE,
                    help="folder holding config.json / model.safetensors")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)

    model, cfg = load_model(args.folder)
    tok = BPETokenizer()
    tok.load(os.path.join(args.folder, "tokenizer.json"))

    context    = cfg.get("context_window", cfg["max_seq_len"] - 1)
    stop_after = args.min_tokens if args.stop_after is None else args.stop_after
    mask       = undecodable_mask(model, tok)

    modulator = affect = None
    if not args.no_affect:
        try:
            from physisml.affect_state import AffectState
            from physisml.modulator    import AffectModulator
            affect    = AffectState()
            modulator = AffectModulator(
                affect, eos_token_id=tok.get_special_id(tok.EOS_TOKEN))
        except ImportError:
            print("note: affect modules not bundled — running plain.",
                  file=sys.stderr)

    def answer(text: str) -> str:
        return generate(model, tok, text,
                        max_tokens=args.max_tokens, temperature=args.temperature,
                        top_k=args.top_k, min_tokens=args.min_tokens,
                        stop_after=stop_after, context=context,
                        modulator=modulator, affect=affect, mask=mask)

    if args.prompt:
        print(answer(" ".join(args.prompt)))
        return

    print(f"PhysisML — {cfg['n_parameters']/1e6:.1f}M params, "
          f"{cfg['active_vocab_size']} active tokens, Italian only"
          f"{'' if modulator else ', affect off'}.")
    print("Ctrl-D to quit.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text:
            print(f"<<< {answer(text)}\n")


if __name__ == "__main__":
    main()

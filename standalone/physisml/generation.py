"""
Shared generation path for PhysisML checkpoints.

This is the code that produces the answers quoted in the README, extracted so
that the repository's chat script (`standalone/chat.py`) and the script shipped
with the published weights (`huggingface/generate.py`) cannot drift apart. It
is a port of the generation path used during training
(`dynamic_model/exp_b/trainer.py`) and keeps the parts that shape the answers:

  * optional affective modulation — confidence / pleasure / pain / fear shift
    the logits at every step
  * EOS suppressed before `min_tokens`, otherwise the model answers with a bare
    end-of-sequence token
  * soft stop on sentence-ending punctuation, tested on the DECODED token
    because the vocabulary has multi-character tokens carrying the terminator
  * output slots the tokenizer cannot decode are masked out

One thing this path cannot reproduce: the published exact-match scores were
measured with the stop threshold set from the length of the gold answer, which
is information a free-form prompt does not have. Expect the model to keep
talking past its answer more often here.
"""
import re

import torch

from .utils import sample_top_k


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

    # The EOS token that ended the loop is inside ids: decode() renders it as
    # the literal '<|EOS|>', so drop every trailing special slot first.
    out = ids[n_prompt:]
    while out and tok.is_special(out[-1]):
        out.pop()
    return clean_response(tok.decode(out))


def load_affect(tok):
    """
    Build the affective modulator if the modules are available. They live in
    dynamic_model/exp_b/ in the repository and are copied into this package by
    scripts/export_hf.py, so both layouts are tried. Returns (modulator,
    affect), or (None, None) when neither is importable.
    """
    try:
        try:
            from .affect_state import AffectState        # published package
            from .modulator    import AffectModulator
        except ImportError:
            from dynamic_model.exp_b.affect_state import AffectState
            from dynamic_model.exp_b.modulator    import AffectModulator
    except ImportError:
        return None, None
    affect = AffectState()
    return AffectModulator(
        affect, eos_token_id=tok.get_special_id(tok.EOS_TOKEN)), affect

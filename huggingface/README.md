---
license: mit
language:
- it
tags:
- physisml
- curriculum-learning
- from-scratch
- italian
- research
- experimental
pipeline_tag: text-generation
inference: false
---

# PhysisML — first preview (Italian only)

A small language model trained from scratch on a developmental curriculum: it
learns like a child, sounds first, then words, then sentences, guided by a tutor
that adapts the material to what the model is currently failing at.

This is a **research preview of an experiment**, published so the results in the
repository can be checked against real weights. It is not a general-purpose
assistant and it will not behave like one.

- Code, curriculum and full documentation: **https://github.com/speleoalex/PhysisML**
- Architecture: decoder-only transformer, pre-LayerNorm, untied `lm_head`
- 28.2M parameters — `d_model=512`, `n_layers=6`, `n_heads=8`, `d_ff=2048`
- Context window: **128 tokens**
- Tokenizer: byte-level BPE, 9000 slots allocated, **2590 active** (the vocabulary
  can grow during training; unused slots are masked to `-inf` at inference)
- float32, runs on a CPU

**These weights are level 6 of a curriculum that runs to level 12, taken from a
build still in progress.** The model has been through phonemes, syllables,
words, grammar and simple dialogue, and has never seen the literature levels.
That is why this is a first preview and not a release.

## ⚠️ Italian only — nothing else has been tested

**Every result below was measured on Italian, and Italian is the only language
this model has ever been trained or evaluated on.**

The repository contains an English curriculum skeleton (`training_files/en/`),
but no English run has been trained, no English checkpoint exists, and no
English number has been measured. Prompting this model in any language other
than Italian is untested territory — expect noise, not a translation.

The same caveat applies to the curriculum design itself: the level ladder
(phonemes → syllables → words → grammar → dialogue → literature) was built
around Italian phonotactics and Italian morphology. Whether the approach
transfers to another language is an open question, not a claim of this release.

## What it can do

It answers the kind of prompt its curriculum drilled — short Italian questions
about a closed world of familiar nouns and verbs. Every line below is an actual
greedy output of these weights, through the bundled `generate.py`:

```
di: il cane                            → il cane!
di: la mamma                           → la mamma!
di: cosa mangia il cane?               → il cane mangia il pane.
cosa fa il cavallo?                    → il cavallo dorme.
dove dorme il cane?                    → il cane dorme in casa.
perché il cane mangia?                 → il cane mangia perché ha fame.
cosa ha mangiato il cane?              → il cane ha mangiato il pane.
```

And the same checkpoint failing, on the same run:

```
il pane è buono?                       → il buono ma non dolce.
il cane è fedele?                      → il cane è stanco.
chi è più grande, il cane o il gatto?  → il cane è grande!
cosa mangerà il cane domani?           → il cane mangia il pane.
chi sei tu?                            → la mucca perché ha fame.
```

Both halves are the honest picture. Opinion, comparison and future tense are
taught at levels 7 and above, which these weights have not reached.

## What it cannot do

- **No world knowledge.** It was trained on a synthetic teaching curriculum of a
  few megabytes, not on a web corpus. Anything factual it produces is invention.
- **No instruction following** beyond the prompt shapes in the curriculum.
- **Closed vocabulary** — 2590 active tokens. Out-of-curriculum words break it.
- **128-token context**, so no documents, no long conversations.
- **No alignment or safety tuning of any kind.** There is no refusal behaviour,
  no filtering, no RLHF. It is a research artifact, not a product.
- **Repetition and prefix collapse**: prompts sharing an opening tend to
  collapse onto the same answer.

Do not put this in front of users. Use it to study the training method.

## Results

Exact match against the curriculum's gold answers, greedy decoding, 457 graded
prompts across the seven levels these weights have seen:

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | mean |
|---|----|----|----|----|----|----|----|------|
| exact match | 95% | 91% | 74% | 74% | 72% | 83% | 100% | **84.3%** |
| self-repetition | 0% | 0% | 14% | 6% | 1% | 3% | 0% | 3.4% |

This is one checkpoint measured against **every** level it has seen, which is a
harder question than the usual per-level table where each level is scored on its
own snapshot. A curriculum model can score 100% level by level and still retain
almost nothing of the earlier ones.

### What the dream does

The number above is what the model looks like **after** a consolidation cycle
("dream"): a replay pass over every level's material, with no new teaching. The
same weights immediately before that pass, measured identically:

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | mean |
|---|----|----|----|----|----|----|----|------|
| before the dream | 62% | 0% | 0% | 0% | 3% | 1% | 100% | 23.7% |
| after the dream | 95% | 91% | 74% | 74% | 72% | 83% | 100% | **84.3%** |

Teaching level 6 had erased levels 1 to 5 outright. One replay pass brought them
back and cost the current level nothing. Self-repetition fell from 11.1% to 3.4%
in the same step.

The repository documents the completed reference build in more detail, including
the negative results and the point where extra consolidation stops paying:
[technical documentation](https://github.com/speleoalex/PhysisML/blob/main/docs/en/physisml_model.md).
Those figures come from a different, earlier run — not from these weights.

## How to run it

There is no `AutoModel` support: the architecture and the tokenizer are custom,
so the model loads through the small package shipped in this repo.

```bash
pip install torch safetensors numpy
hf download speleoalex/physisml-it-preview --local-dir physisml-model
cd physisml-model

python3 generate.py "di: cosa mangia il cane?"   # one answer, greedy
python3 generate.py                              # interactive REPL
python3 generate.py --no-affect "di: il cane"    # plain transformer
```

Files:

| File | What it is |
|---|---|
| `model.safetensors` | the weights, float32 |
| `config.json` | architecture, active vocabulary size, context window |
| `tokenizer.json` | byte-level BPE vocabulary + merges |
| `physisml/` | inference code: model, tokenizer, sampling, affective modulation |
| `generate.py` | CLI: single prompt or REPL |
| `MANIFEST.json` | which checkpoint each artifact came from, with sha256 |

`generate.py` is a port of the repository's own generation path, so the
affective system (`confidence`, `pleasure`, `pain`, `fear` shifting the logits
at every step) is active by default — `--no-affect` turns it off if you want to
see what the bare transformer does.

One detail it cannot reproduce: the published exact-match scores are measured
with the punctuation stop threshold set from the length of the gold answer,
which a free-form prompt does not know. Expect the model to run past its answer
more often here than the tables suggest.

## How it was trained

Two phases per level, repeated up the ladder:

1. **Text phase** — ordinary self-supervised training on the level's corpus.
2. **Teaching phase** — a tutor poses a prompt, grades the model's answer
   (`+++` … `-`), and the feedback drives the update. The tutor picks the next
   prompt from what the model is currently getting wrong.

Each session ends in a **dream**: the consolidation pass measured above, which
replays every level's question-answer corpus. It is where cross-level retention
comes from.

The tutors are, in cost order: a rule-based deterministic teacher (offline), a
hybrid teacher using a small local LLM for the grading (offline, via ollama),
and optionally the Claude API. **The whole Italian curriculum trains offline** —
every level ships its own local teacher configuration, so no API key is needed
to reproduce this model from scratch (`./build.sh 6`).

Training data is the curriculum in `training_files/it/`: hand-built and
script-expanded target pools, question-answer pairs, and level texts. Parts of
the question-answer material were seeded by earlier teaching sessions that used
the Claude API tutor.

## License and attribution

MIT — Copyright (c) 2026 Alessandro Vernassa. See `LICENSE`.

```bibtex
@software{physisml,
  author = {Vernassa, Alessandro},
  title  = {PhysisML: a language model trained on a developmental curriculum},
  year   = {2026},
  url    = {https://github.com/speleoalex/PhysisML}
}
```

The name is φύσις (*physis*, nature, growth) + ML. The documentation exists in
both English and Italian in the repository.

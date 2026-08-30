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

# PhysisML — Italian curriculum 0-12 (experimental)

A small language model trained from scratch on a developmental curriculum: it
learns like a child, sounds first, then words, then sentences, guided by a tutor
that adapts the material to what the model is currently failing at.

This is a **research preview of an experiment**, published so the results in the
repository can be checked against real weights. It is not a general-purpose
assistant and it will not behave like one.

- Code, curriculum and full documentation: **https://github.com/speleoalex/PhysisML**
- Architecture: decoder-only transformer, pre-LayerNorm, `lm_head` weight-tied
  to the token embedding
- 23.6M parameters — `d_model=512`, `n_layers=6`, `n_heads=8`, `d_ff=2048`
- Context window: **128 tokens**
- Tokenizer: byte-level BPE, 9000 slots allocated, **2590 active** (the vocabulary
  can grow during training; unused slots are masked to `-inf` at inference)
- float32, runs on a CPU

**These weights are the end of the curriculum**: the level-12 checkpoint of a
completed 0-12 build — phonemes, syllables, words, grammar, dialogue,
literature (Rodari, Manzoni, Dante), class membership, and asking about a name
it has never met.

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
cos è un cane?                         → il cane è un animale.
dove dorme il cane?                    → il cane dorme in casa.
chi è più grande, il cane o il gatto?  → il cane è più grande del gatto.
di: cosa mangia il cane?               → il cane mangia il pane.
perché il cane mangia?                 → il cane mangia perché ha fame.
cosa ha mangiato il cane?              → il cane ha mangiato il pane.
cosa mangerà il cane domani?           → domani il cane mangerà il pane.
il pane è buono?                       → secondo me il pane è buono perché è caldo.
il cane è fedele?                      → secondo me il cane è fedele perché aspetta.
commenta il libro                      → il libro insegna, e questo è importante.
l albero è una pianta, questo è un tamburo  → cos è un tamburo?
```

The last line is level 12: shown a name it has never met, the model asks
instead of inventing.

## Results

Exact match against the curriculum's gold answers, greedy decoding, 849 graded
prompts across the thirteen levels.

**The published checkpoint against every level** — one model, asked everything
it was ever taught:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| 100% | 94% | 80% | 84% | 88% | 94% | 92% | 85% | 87% | 77% | 95% | 79% | 100% |

Mean **89.0%**, self-repetition 1.4%.

For scale: in the project's previous build the final checkpoint scored **20%**
on the same question, and ten extra consolidation cycles brought it only to
48%. Here no level of the finished model sits below 77%.

The usual per-level table — each level scored on its own snapshot, which is a
much easier question — averages 94.9% on this run:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| 100% | 89% | 70% | 89% | 100% | 99% | 95% | 100% | 100% | 100% | 97% | 94% | 100% |

Five things changed between the two builds — six dream cycles per level by
default, target pools widened from 227 to 728, a vocabulary retrained without
punctuation glued to words, one gold answer per prompt enforced across levels,
and `<|EOS|>` registered again — so the gain cannot be attributed to any single
one of them.

## Where it fails

Level 11 teaches class membership, and the `cos è X?` form is asked about only
24 nouns. Eighteen more appear in the level's yes/no steps without ever being
asked that way, and `cane` appears only as an *answer* (`fai un esempio di
animale`). The consequences are visible:

```
cos è un cane?             → il cane è un animale.     ✓
fai un esempio di animale  → il cane è un animale.     ✓
il cane è un animale?      → sì, il cane è un animale. ✓
cos è il cane?             → il cane è una cosa.       ✗ (a superordinate, not the class)
```

The shape is right and the class is not: the model has the fact and no drilled
path from that phrasing to it.

Level 12 teaches asking about a name it has never met, and it does that well in
the sentence shape it was taught (`l albero è una pianta, questo è un tamburo`
→ `cos è un tamburo?`). A bare `chi è zibaldone?` was never taught, and there
it confabulates — `il fratello è una persona.`

Beyond that:

- **No world knowledge.** It was trained on a synthetic teaching curriculum of a
  few megabytes, not on a web corpus. Anything factual it produces is invention.
- **No instruction following** beyond the prompt shapes in the curriculum.
- **Closed vocabulary** — 2590 active tokens. Out-of-curriculum words break it.
- **128-token context**, so no documents, no long conversations.
- **No alignment or safety tuning of any kind.** There is no refusal behaviour,
  no filtering, no RLHF. It is a research artifact, not a product.
- **Sampling costs a lot here.** Every example on this page is greedy
  (`--temperature 0`). With sampling on, `cos è un cane?` answers *animale*,
  *persona* or *luce* depending on the draw: at this scale the model has the
  right answer at the top of the distribution, not alone in it.

Do not put this in front of users. Use it to study the training method.

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
| `model.safetensors` | the weights, float32. `lm_head.weight` is absent on purpose — it is tied to `tok_emb.weight` and the loader re-ties it |
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

Each session ends in a **dream**: a consolidation pass that replays every
level's question-answer corpus, with no new teaching. It is where cross-level
retention comes from, and the effect is large enough to measure on a single
cycle. Taken at level 6 of this same build, immediately before and after one
dream, on all seven levels the model had seen:

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | mean |
|---|----|----|----|----|----|----|----|------|
| before | 62% | 0% | 0% | 0% | 3% | 1% | 100% | 23.7% |
| after | 95% | 91% | 74% | 74% | 72% | 83% | 100% | **84.3%** |

Teaching level 6 had erased levels 1 to 5 outright; one replay pass brought them
back and cost the current level nothing. The build runs six such cycles per
level.

The tutors are, in cost order: a rule-based deterministic teacher (offline), a
hybrid teacher using a small local LLM for the grading (offline, via ollama),
and optionally the Claude API. **The whole Italian curriculum trains offline** —
every level ships its own local teacher configuration, so no API key is needed
to reproduce this model from scratch (`./build.sh 12`, about two and a half
days on a 16-core CPU).

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

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

Short Italian questions about a closed world of familiar nouns and verbs. Every
line is an actual greedy output of these weights:

```
di: cosa mangia il cane?               → il cane mangia il pane.
perché il cane mangia?                 → il cane mangia perché ha fame.
cosa mangerà il cane domani?           → domani il cane mangerà il pane.
cos è il cane?                         → il cane è un animale.
il cane è un animale?                  → sì, il cane è un animale.
il pane è buono?                       → secondo me il pane è buono perché è caldo.
commenta il libro                      → il libro insegna, e questo è importante.
cos è un falco?                        → non lo so.
l albero è una pianta, questo è un tamburo  → cos è un tamburo?
```

The last two are level 12: on a name it has never met the model declares
ignorance or asks, rather than guessing. On held-out names it has never seen,
67% of answers are honest that way, against **0%** on nouns it knows — it never
claims ignorance about something it was taught.

## Results

Exact match against the curriculum's gold answers, greedy, 849 graded prompts
across the thirteen levels.

| | mean | worst level |
|---|---|---|
| **this checkpoint, asked about every level** | **88.1%** | 70% (L9) |
| each level scored on its own snapshot | 94.5% | 70% (L2) |

The first row is the interesting one: one model asked everything it was ever
taught, self-repetition 1.3%. The project's previous build scored **20%** on
that same question, and ten extra consolidation cycles took it only to 48%.

Per level: 100 / 97 / 80 / 83 / 93 / 90 / 85 / 93 / 90 / 70 / 85 / 79 / 100.

The lever is the **dream**: a replay pass over every level's material with no
new teaching. Measured at level 6 of this build, immediately before and after
one cycle — 23.7% → **84.3%**, self-repetition 11.1% → 3.4%. Teaching a level
erases the earlier ones; one replay brings them back and costs the current level
nothing. The build runs six per level.

Five things changed between the two builds (six dreams per level by default,
target pools 227 → 728, a vocabulary retrained without punctuation glued to
words, one gold per prompt enforced, `<|EOS|>` written into the corpus), so the
gain cannot be attributed to any single one.

## Where it fails

**Untaught phrasings drift onto the nearest taught pattern.** This is the
sharpest limit of a model this size:

```
cos è il cane?     → il cane è un animale.            ✓ (the drilled form)
cos è un cane?     → un animale è un essere vivente.  ✗ (answers the class question)
chi è zibaldone?   → il tamburo è un oggetto.         ✗ (asks nothing — shape never taught)
```

Level 11 drills `cos è {definite} X?`; the indefinite variant on a concrete noun
is not in the pool, and the model reaches for the class-level question it does
know. Level 12 teaches the ask on `cos è un X?` and the yes/no form, not on
`chi è X?`.

Also:

- **No world knowledge.** A synthetic teaching curriculum of a few megabytes,
  not a web corpus. Anything factual it produces is invention.
- **Closed vocabulary** — 2607 active tokens; out-of-curriculum words break it.
- **128-token context**, single-turn only: it was never trained on
  conversations, and prior turns crowd out the question.
- **No alignment or safety tuning of any kind.** No refusals, no filtering.
- **The margin is thin.** Every example here is greedy. Sampling makes
  `cos è un cane?` answer *animale*, *persona* or *luce* depending on the draw.

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
| `physisml.gguf` + `Modelfile` | the same weights for llama.cpp / ollama |

`generate.py` is a port of the repository's own generation path, so the
affective system (`confidence`, `pleasure`, `pain`, `fear` shifting the logits
at every step) is active by default — `--no-affect` turns it off if you want to
see what the bare transformer does.

### In ollama

```bash
ollama create physisml -f Modelfile && ollama run physisml
```

The model ends its own answers — it emits `<|EOS|>`, which the GGUF declares —
so the Modelfile needs no stop strings. Note that `ollama run` interactively
sends the whole conversation back as context: with a 128-token window and no
multi-turn training, a few exchanges crowd out the question. Use `/clear`,
one-shot `ollama run physisml "..."`, or the API.

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

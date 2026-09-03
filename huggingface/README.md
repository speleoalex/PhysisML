---
license: mit
language:
- it
tags:
- physisml
- curriculum-learning
- continual-learning
- catastrophic-forgetting
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
- Tokenizer: byte-level BPE, 9000 slots allocated, **2593 active** (the vocabulary
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
l albero è una pianta, questo è un tamburo  → cos è un tamburo? il tamburo è un oggetto.
```

The last two are level 12: on a name held out of the curriculum the model
declares ignorance, or asks and then answers itself once told. On the held-out
probe the gap between honest answers on unknown and on taught nouns is
**+75 points** (gate off, greedy), and it never claims ignorance about a noun
it was taught.

**Outside the trained pool that relation does not generalize.** On seven names
no training file has ever mentioned, only 14% of answers are honest — the
model usually confabulates a class, or asks about a *different* noun it knows
(`questa è una lumaca` → `cos è una bussola?`). Widening the honesty pool from
6 to 38 nouns moved the behaviour onto the new nouns without generalizing it;
meanwhile the model's *internal* state separates known from unknown nearly
perfectly (margin over the 10 classes: AUC 0.987 on these weights, replicated
across two independently trained builds). The gap between what the weights
know and what the string binds to is the project's current work front.

## Results

Exact match against the curriculum's gold answers, greedy. The 2026-09-01
rebuild grew the target set (levels 11-12 now teach the honesty relation over
38 nouns in every phrasing, 1369 graded prompts against the previous 849), so
two numbers, each honest about what it compares:

| | |
|---|---|
| **frozen probe — 104 identical prompts, previous published build vs this one** | 84.6% → **90.4%** |
| this checkpoint on every current target (1369 prompts) | 84%, self-repetition 2% |

Per level: 100 / 97 / 79 / 87 / 93 / 97 / 80 / 91 / 95 / 75 / 97 / 92 / 57.

That last value is worth staring at rather than hiding: level 12's own pool
tripled and hardened (asking about the right referent with a distractor in the
prompt, yes/no answers graded on the polarity word), and the model holds 57%
of it. The failures are real and listed below.

The lever is the **dream**: a replay pass over every level's material with no
new teaching. Measured at level 6, one cycle took cross-level exact match from
23.7% to **84.3%**. This build stopped counting dreams by constant and started
measuring them: after each dream the frozen probe is re-scored, and the level
stops dreaming when the marginal gain dies (`scripts/dream_until_plateau.py`).
Level 11's own knee turned out to be 8 dreams, not the 6 the old constant
assumed; level 12 oscillated sawtooth to its best at dream 9. Each level's
curve ships in its checkpoint directory as `dream_curve.json`.

Between the previous build and this one: the honesty pool went from 6 nouns to
38 with declared roles (23 acquirable by the autonomy loop, 8 permanent
reserve, 7 never taught — the generalization probe above), the two-clause
asking lesson covers all of them with a distractor step, yes/no answers are
graded on the polarity word (they never were before — both `sì` and `no`
earned full marks), phase-1 budgets now cover the whole pool, and dreams are
counted by measurement. As before, the gain cannot be attributed to any single
change.

## Where it fails

**Untaught phrasings drift onto the nearest taught pattern.** This is the
sharpest limit of a model this size:

```
cos è il cane?            → il cane è un animale.          ✓ (drilled)
cos è un cane?            → il cane è un animale.          ✓ (fixed this build: both articles now trained)
il giardino è un animale? → sì, il giardino è un luogo.    ✗ (right class, wrong yes)
chi è zibaldone?          → il cane è grande. il cane…     ✗ (shape never taught)
cos è una tegola?         → la penna è un oggetto.         ✗ (never-seen noun → confabulation)
```

The polarity slip is instructive: the grader was blind to the `sì`/`no` word
until this build (both answers scored identically), so the distinction was
never part of the training signal before now, and one build of graded polarity
has not finished repairing it. `chi è X?` is a shape no level teaches. The
last line is the generalization limit measured above.

Also:

- **No world knowledge.** A synthetic teaching curriculum of a few megabytes,
  not a web corpus. Anything factual it produces is invention.
- **Closed vocabulary** — 2593 active tokens; out-of-curriculum words break it.
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
back and cost the current level nothing. The build repeats the cycle until the
probe stops improving (at least six times per level).

Is the dream just experience replay dressed up? Mostly yes — and that is the
finding. Benchmarked head-to-head on the same harness against online EWC
(Schwarz et al. 2018, λ from a sweep, two seeds, levels 0–6), the dream kept
64% of past levels where EWC kept 13% and no protection at all kept 22%: EWC
landed *below* the unprotected floor, because at near-zero training loss the
empirical Fisher measures inter-example variance concentrated on shared
structural tokens, and the anchor freezes the answer-production machinery
instead of the knowledge. The comparison is not compute-matched, so the
relative efficiency of the two methods remains open. Full numbers and the
diagnostic trail are in the repository README (`exp_i`).

The tutors are, in cost order: a rule-based deterministic teacher (offline), a
hybrid teacher using a small local LLM for the grading (offline, via
llama.cpp or ollama),
and optionally the Claude API. **The whole Italian curriculum trains offline** —
every level ships its own local teacher configuration, so no API key is needed
to reproduce this model from scratch (`./build.sh 12`, about two and a half
days on a 16-core CPU, or hours with a small Intel Arc GPU — the build picks
it up by itself).

Training data is the curriculum in `training_files/it/`: hand-built and
script-expanded target pools, question-answer pairs, and level texts. Parts of
the question-answer material were seeded by earlier teaching sessions that used
the Claude API tutor.

## License and attribution

MIT — Copyright (c) 2026 Alessandro Vernassa. See `LICENSE`.

```bibtex
@software{physisml,
  author  = {Vernassa, Alessandro},
  title   = {PhysisML: a language model trained on a developmental curriculum},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.22285423},
  url     = {https://github.com/speleoalex/PhysisML}
}
```

Concept DOI (always the latest version):
[10.5281/zenodo.22285422](https://doi.org/10.5281/zenodo.22285422).

The name is φύσις (*physis*, nature, growth) + ML. The documentation exists in
both English and Italian in the repository.

---
license: mit
language:
- en
tags:
- physisml
- curriculum-learning
- continual-learning
- catastrophic-forgetting
- from-scratch
- english
- research
- experimental
pipeline_tag: text-generation
inference: false
---

# PhysisML — English curriculum 0-12 (experimental)

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
- Tokenizer: byte-level BPE trained on the English curriculum, 9000 slots
  allocated, **2562 active** (`<|EOS|>` at 2516; unused slots are masked to
  `-inf` at inference)
- float32, runs on a CPU

**These weights are the level-12 checkpoint of a 0-12 build**: isolated sounds,
first words, article + noun + verb, short questions, *who* and *where*, the
connectives *and* / *but* / *because*, then the past, the future, comparatives
and preferences, a thesis with its reason, a short motivated comment — and
finally the two ontology levels: **class membership** (`the dog is an animal`)
and **admitting ignorance** (`what is a compass?` → `i do not know.`).

They replace the 0-10 preview published on 2026-09-06 at this same repo. Levels
11 and 12 added 639 graded targets, so the overall figure below is measured on
1359 prompts against that card's 720 — a larger denominator, and a higher score.

## ⚠️ The twelfth rung is reached; the thirteenth is not

The Italian model of this project runs an **autonomy loop** at level 13: it
detects its own ignorance from an internal signal, asks the tutor, and learns
the answer. **This English build does not**, and the reason is measured, not a
matter of time: the epistemic trigger separates known from unknown nouns at
**AUC 0.788** on these weights, where the loop's guardrail is 0.95. Its verdict
is `OVERLAPPING` — 18 of 57 known nouns would still trigger a spurious question.

That 0.788 is the result of acting on the diagnosis the previous card carried.
English declared one interrogative phrasing (`"ask_heads": ["what is"]`) where
Italian declares two, so its ontology levels were trained on roughly half as
many targets. Adding the second head (`what s`) grew the two pools from 254 and
223 targets to **354 and 285**, and re-running levels 11 and 12 on them took the
AUC from 0.638 to 0.788 — the gap to the guardrail halved, and did not close.

Asking about the *right* referent has not moved. On the twelve prompts of the
shape `the box is an object, this is a snail` these weights ask a question every
time and name the noun in front of them 6 times out of 12 (the previous weights,
5) — and all six are names level 12 taught. On the six names the curriculum
never taught, both score **0 of 6**: `the box is an object, this is a snail` →
`what is a bucket?` The model has learned to ask, and to recite a name it was
given; it has not learned to read the name out of the prompt.

The gain also came with a regression on level 11's affirmative confirmations
that is documented in [Where it fails](#where-it-fails) and should be read
before the results table.

There is no retraction ledger for these weights either, and nothing here has
been through an autonomy session.

It is **English only**: the vocabulary is 2562 English BPE tokens and the model
has never seen another language. It shares no weights and no vocabulary with the
Italian preview — it is a second, independent run of the same method, which is
the point of publishing it.

## What it can do

Short English questions inside a closed world of familiar nouns and verbs.
Every line below is an actual greedy output of these weights, taken from the
repository's own level probe:

```
say ma                                  → ma!
say: the cat                            → the cat!
say: the wolf runs                      → the wolf runs!
what are you called?                    → physisml!
what does the bee do?                   → the bee flies.
who cooks?                              → the woman cooks.
what is the sea like?                   → the sea is cold.
what did the boy eat?                   → the boy ate the bread.
what will the boy eat tomorrow?         → tomorrow the boy will eat the bread.
tell me the three days of the dog       → today it runs, yesterday it ran,
                                          tomorrow it will run.
is the cat small?                       → the cat is small but good.
which is better, the bread or the door? → the bread is better, because you eat it.
what does the story say about the dog   → the story says that the dog runs.
comment on the cat                      → the cat is fast, and this is nice.
what is the dog?                        → the dog is an animal.
the bread is a food?                    → yes, the bread is a food.
the cat is a plant?                     → no, the cat is an animal.
what is a compass?                      → i do not know.
the cow is an animal, this is a compass → what is a compass?
the wolf is an animal, this is a drum   → what is a drum?
```

The last four are level 12. On a name the curriculum never taught, the model
declares ignorance or asks about it, instead of guessing a class. Measured on 35
held-out nouns with no gate of any kind (`scripts/curiosity_rate.py --gate off`):
**77% honest** answers against **8%** on the twelve known nouns, which it
otherwise classifies correctly (83%). With the epistemic gate on, 100% and 8%.

The 8% is a cost, not a rounding: one known noun in twelve now draws a spurious
question that the previous weights answered. It is the same over-asking the AUC
verdict reports, and it is the price paid for the honesty on never-seen names
rising from 49% to 77% on the same 35 prompts.

## Results

Exact match against the curriculum's gold answers, greedy, on these weights (the
post-dream level-12 checkpoint), replayed over **every target of every level** —
1359 prompts, `scripts/measure_repetition.py`:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | overall |
|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|---------|
| 100% | 100% | 85% | 97% | 57% | 90% | 89% | 94% | 89% | 96% | 100% | 85% | 98% | **90%** |

Self-repetition 1%. On the build's own frozen probe — 104 prompts, 8 per
level, fingerprint `a173551267247f59` — these weights score **91.3%**,
self-repetition 1.9%.

**Do not read the L11 column as a summary of level 11.** Its 85% averages a
level that answers `no` correctly almost always and `yes` correctly barely more
than a quarter of the time, and on the affirmative half it is a sharp regression
against the weights this release replaces. See [Where it fails](#where-it-fails).

**Read that 91.3% as the top of a band, not as a level.** Consecutive dreams
swing by up to three probe items in either direction (89.4 → 91.3 → 88.5 here),
and the run publishes the best measured state. A re-run of the same recipe
should be expected to land somewhere in 88-91%, not to reproduce 91.3%.

**The two new levels bought retention rather than costing it.** Against the
level-10 card (78.8% on the 720 targets that existed then), the low levels came
*up*: L5 54% → 90%, L6 68% → 89%, L7 77% → 94%, L8 72% → 89%, L9 75% → 96%,
L2 75% → 85%, L1 98% → 100%. Twenty-two dream cycles — seven on level 11,
fifteen on level 12 — over a corpus that now includes L11 and L12 replayed
everything else with it. Level 5 — the regression
the previous card led with — is repaired.

Those L0-L10 pools are byte-identical to the ones the previous card was scored
on, so the gains there are attributable to the extra dreams alone. The L11 and
L12 columns are **not** comparable to the previous card's: those pools grew with
the second interrogative head, and the older weights were never taught it.

Scored instead with **each level's own** checkpoint, the same 1359 prompts give
**97%** (L11 100%, L12 98%). Everything this model gets wrong it once had
right; the 7-point gap between the two rows is forgetting, not a pool it never
learned.

The lever is the **dream**: a replay pass over every level's material with no
new teaching. Each level dreams until the probe stops improving, and the curve
ships in the checkpoint directory as `dream_curve.json`. Level 11 stopped on its
own after 7 dreams:

```
42 → 53 → 59 → 66 → 70 → 72 → 74 %
```

Level 12 stopped at 8 dreams on the build's own ε=2%, then was resumed for up to
10 more with a tighter ε=1% and ran 7 of them:

```
82.7 → 79.8 → 85.6 → 89.4 → 89.4 → 91.3 → 88.5 → 88.5 %    best kept: 91.3%
```

That sawtooth is why the run re-scores the probe after every cycle and restores
the **best** measured state rather than the last one — the weights published
here are dream 13's, not dream 15's.

**The tighter ε is what bought the last six points, and the reason is
arithmetic.** The probe has 104 items, so one item is 0.96%: at ε=2% a level
must gain three items per dream to count as still gaining, and two-item gains
read as a plateau. Level 12 stopped at 82.7% under that rule with six points
still on the table. A negative swing costs a patience strike too — the -2.9% at
dream 9 above is why the run ended at 15 rather than at the cap of 18.

The two ontology levels took **7h41** on an Intel Arc A370M (L11 1h49, L12 2h04
plus 3h48 of resumed dreams), on top of the 13h37 of levels 0-10 — about **21
hours** for the whole ladder, entirely offline.

## Where it fails

**Level 11's 85% hides a regression, and it is the number on this card to read
before the others.** 54 of its 354 prompts are wrong, and 42 of those are
polarity: the model names the right class and then denies it.

```
the cat is an animal?   → no, the cat is an animal.   ✗ (gold: yes, …)
the mom is a person?    → no, the mom is a person.    ✗ (gold: yes, …)
```

Split by the sign of the gold answer, on the same 57 + 57 confirmations, against
the weights this release replaces:

| level-11 confirmations | previous weights | these weights |
|---|---|---|
| gold `yes` (57) | 44 exact | **16 exact** |
| gold `no` (57) | 47 exact | **56 exact** |

It answers `no` to 41 of the 57 questions whose answer is `yes`; the previous
weights answered `no` to 13. The gain on the negative half is exactly what a
blanket `no` would buy, so this is not sharper discrimination — affirmative
confirmation has collapsed into negative. Level 12 shows the same failure at its
own scale: 6 of its 7 misses are that identical flip. Nothing in this release
targeted polarity, and the two changes that could have caused it — the wider
`what s` pool and the seven extra dreams — arrived together, so nothing here
isolates which one did it. The epistemic numbers above and this regression are
from the same weights and cannot be taken apart.

**The two steps that put a *class* where the other steps put a noun are still
unlearned**, six targets each: `give an example of a plant` → `. the bread is a
food.`, and `what is a plant?` → `the spider is an animal.` Asked about an
individual it is nearly perfect; asked about a category it falls back on an
instance.

**One level did not move, and it is now the weakest of the thirteen.** Level 4
teaches the locative question; the twenty-two dream cycles of levels 11 and 12
took it from 54% to 57%, and 24 of its 29 remaining failures are a single
substitution — the locative frame collapsing into level 5's causal one:

```
where does the cat sleep?  → the cat sleeps because it is tired.  ✗ (gold: … in the house.)
where does the fish swim?  → the fish swims because it is fast.   ✗ (gold: … in the sea.)
```

The other five are the ordering step, which loses the question's second noun:
`what comes first, the sun or the moon?` → `the sun is hotter than the moon.`

**Elsewhere the frame is learned and the content word is not.** This remains the
clearest pattern in what is left, and levels 5 and 8 show it plainly — syntax,
agreement and the comparative construction intact, the binding to the question's
noun missing:

```
who is bigger, the horse or the dog?          → the horse is stronger than the dog.       ✗
what is sweeter, the honey or the apple?      → the apple is sweeter than the apple.      ✗
what is more beautiful, the star or the lamp? → the star is more beautifuler than the lamp. ✗
tell me two things about the woman            → the woman works and works.                ✗
why does the bird sing?                       → the bird flies because it is fast.        ✗
```

Seven of level 8's twelve failures are its preference steps (`what do you like
to eat?` → `I like the milk.`, gold `the bread`), where the gold answer is one
the curriculum happened to pick. Exact match scores those as errors; a human
would not. Take the 85% on that level as a lower bound.

**Longer imitation prompts still degrade.** Level 2 asks the model to repeat a
sentence; 16 of its 100 targets miss, and its 12% self-repetition is the highest
of any level:

```
say: the child opens the door  → the child opens the door the door the door the door …
say: the girl finds the ball   → the girl sings the girl sings!
```

Also:

- **No world knowledge.** A synthetic teaching curriculum of a few megabytes,
  plus a handful of public-domain books used only as raw text. Anything
  factual it produces is invention.
- **Closed vocabulary** — 2562 active tokens; out-of-curriculum words break it.
- **128-token context**, single-turn only: it was never trained on
  conversations, and prior turns crowd out the question.
- **No alignment or safety tuning of any kind.** No refusals, no filtering.
- **The margin is thin.** Every example here is greedy. Sampling changes the
  answers.

Do not put this in front of users. Use it to study the training method.

## How to run it

There is no `AutoModel` support: the architecture and the tokenizer are custom,
so the model loads through the small package shipped in this repo.

```bash
pip install torch safetensors numpy
hf download speleoalex/physisml-en-preview --local-dir physisml-en
cd physisml-en

python3 generate.py "what is the cat like?"   # one answer, greedy
python3 generate.py                           # interactive REPL
python3 generate.py --no-affect "the cat"     # plain transformer
```

Files:

| File | What it is |
|---|---|
| `model.safetensors` | the weights, float32. `lm_head.weight` is absent on purpose — it is tied to `tok_emb.weight` and the loader re-ties it |
| `config.json` | architecture, active vocabulary size, context window |
| `tokenizer.json` | byte-level BPE vocabulary + merges, `<|EOS|>` included |
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
ollama create physisml-en -f Modelfile && ollama run physisml-en
```

The model ends its own answers — it emits `<|EOS|>`, which the GGUF declares —
so the Modelfile needs no stop strings. Note that `ollama run` interactively
sends the whole conversation back as context: with a 128-token window and no
multi-turn training, a few exchanges crowd out the question. Use `/clear`,
one-shot `ollama run physisml-en "..."`, or the API.

## How it was trained

Two phases per level, repeated up the ladder:

1. **Text phase** — ordinary self-supervised training on the level's corpus.
2. **Teaching phase** — a tutor poses a prompt, grades the model's answer
   (`+++` … `-`), and the feedback drives the update. The tutor picks the next
   prompt from what the model is currently getting wrong.

Each session ends in a **dream**: a consolidation pass that replays every
level's question-answer corpus, with no new teaching. It is where cross-level
retention comes from — in this build the level-12 dreams alone moved the probe
from 62% to 91.3%, and the two ontology levels' dreams between them pulled
levels 5-9 up by 17 to 36 points each.

**The whole English curriculum trains offline.** Every level ships its own
local teacher configuration, so no API key is needed to reproduce these weights
from scratch:

```bash
./build.sh 12 --lang en
```

The training data is `training_files/en/` — 11 MB of question-answer pairs
(4163 across the thirteen levels) and level texts. The text phases of levels 2-5
use public-domain books as raw material: Shakespeare (L2), *Alice in Wonderland*
and *Oliver Twist* (L3), *Jane Eyre* and *Pride and Prejudice* (L4),
*Moby-Dick* (L5). Levels 6-12 use no books at all — their text phase reads a
hand-written `sentences_levelN.txt`, and levels 11 and 12 generate theirs from
the class lattice in `training_files/en/lexicon.json`. The graded material —
what the tutor actually teaches and scores — is the curriculum's own pairs, not
the books.

## Relationship to the Italian model

Same code, same architecture, same hyper-parameters, different language folder:
`training_files/en/` instead of `training_files/it/`, its own tokenizer, its own
axiom words (`I am`, `you are`, `he is`, `it is`), its own checkpoints. Nothing
about the English run required a change to the training code — that is the
property the repository's language manifests exist to keep true.

Both ladders now reach level 12, but they are not equally furnished. English
used to declare one interrogative phrasing where Italian declares two, and its
ontology levels were trained on roughly half the targets; this build closes that
gap (`"ask_heads": ["what is", "what s"]`, pools of 354 and 285). The epistemic
trigger moved with it, from AUC 0.638 to **0.788** — still short of the 0.95 the
loop needs, and short of the above-0.98 Italian reaches. Italian has also been
through a retraction pass (`--retract`) and two autonomy runs; English has been
through neither.

The comparison to draw between the two is therefore about method, not about
scores: the levels are not equivalent tasks across languages, and the two runs
have not had the same amount of work done on them.

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

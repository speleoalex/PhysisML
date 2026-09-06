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

# PhysisML — English curriculum 0-10 (experimental)

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
  allocated, **2533 active** (`<|EOS|>` at 2516; unused slots are masked to
  `-inf` at inference)
- float32, runs on a CPU

**These weights are the level-10 checkpoint of a 0-10 build**: isolated sounds,
first words, article + noun + verb, short questions, *who* and *where*, the
connectives *and* / *but* / *because*, then the past, the future, comparatives
and preferences, a thesis with its reason, and a short motivated comment.

They replace the 0-5 preview published on 2026-09-05 at this same repo. The
previous card's numbers are not comparable with the ones below — see *Results*.

## ⚠️ Ten rungs of a twelve-rung ladder

The Italian model of this project is trained to level 12. **This English build
stops at level 10** — the equivalent of a ten-year-old in the curriculum's own
metaphor. Two things the Italian preview is interesting for do not exist here:

- no class membership (`the dog is an animal`),
- no admission of ignorance, no asking about an unfamiliar name, no autonomy
  loop and no retraction ledger.

Asking this model for any of those gets a confident sentence built out of level
0-10 patterns, not an answer.

It is also **English only**: the vocabulary is 2533 English BPE tokens and the
model has never seen another language. It shares no weights and no vocabulary
with the Italian preview — it is a second, independent run of the same method,
which is the point of publishing it.

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
is the cat small?                       → I think the cat is small because it is fast.
which is better, the bread or the door? → the bread is better, because you eat it.
what does the story say about the dog   → the story says that the dog runs.
comment on the cat                      → the cat is fast, and this is nice.
```

## Results

Exact match against the curriculum's gold answers, greedy, on these weights (the
post-dream level-10 checkpoint), replayed over **every target of every level** —
720 prompts, `scripts/measure_repetition.py`:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | overall |
|----|----|----|----|----|----|----|----|----|----|-----|---------|
| 100% | 98% | 75% | 96% | 54% | 54% | 68% | 77% | 72% | 78% | 100% | **78.8%** |

Self-repetition 3.9%. On the build's own frozen probe — 88 prompts, 8 per level,
fixed before the level-6 run — these weights score **75.0%**, self-repetition
5.7%.

**This is not the same measurement as the 0-5 card.** That one reported 90.3%
over a 72-prompt sample (`test_model.py --samples 12`, twelve prompts per
level); this one grades every target the curriculum contains, which is a larger
and harder denominator, and is what the Italian model has always been reported
on. The change of number is a change of ruler.

**What did move is level 5.** On the same 418 prompts of levels 0-5, the old
level-5 checkpoint scored 86.4% and this one scores 77.5% — and 33 of the 37
lost prompts are level 5 alone. Levels 0-4 lost 3 prompts out of 346 between
them, and level 2's imitation, which the previous card flagged as broken, is
partly repaired (`say: the happy girl` is now correct).

The lever is the **dream**: a replay pass over every level's material with no
new teaching. Each level dreams until the probe stops improving, and the curve
ships in the checkpoint directory as `dream_curve.json`. Level 10's, one entry
per dream:

```
36 → 45 → 49 → 55 → 64 → 65 → 67 → 69 → 74 → 75 → 75 %
```

Levels 6-10 were built with `MAX_DREAMS=20` and **none of them reached the cap**
(6, 8, 13, 7 and 11 dreams): unlike level 5 in the 0-5 build, every one stopped
on a plateau or on a regression it had a better snapshot for. Those five levels
took **7h44** on an Intel Arc GPU (67/77/126/72/117 minutes each), on top of the
5h53 of levels 0-5 — **13h37** for the whole ladder, entirely offline.

## Where it fails

**A later level's pattern overwrites an earlier one, and the dream does not
always repair it.** Level 5 teaches *because*; levels 6-10 reuse that frame for
other jobs, and level 5 pays for it. The failures keep the right shape and pick
the wrong content:

```
why does the man work?   → the man walks because he is strong.   ✗ (gold: the man works …)
why does the dog drink?  → the dog runs because it is hungry.    ✗ (gold: the dog drinks …)
```

**Level 9 taught the model to open with a polarity word, and it leaks.** Six
answers across levels 4, 5 and 9 begin with a `no,` or `yes,` the question never
asked for:

```
where does the fish swim? → no, the fish swims in the sea.   ✗ (the rest of the answer is the gold)
why does the cat sleep?   → no, the cat sleeps in the house. ✗
```

**The frame is learned, the content word is not.** This is the single clearest
pattern in every failure the model has left, and level 8 shows it plainly: 23 of
its 82 prompts miss, and almost all of them are the right sentence with the
wrong noun or adjective slotted in.

```
who is older, the woman or the baby?  → the woman is younger than the baby.  ✗ (older)
who is stronger, the horse or the child? → the horse is stronger than the cow. ✗ (child)
what do you like to read?             → I like the milk.                     ✗ (the book)
why do you like the cake?             → I like the bread because it is good. ✗ (cake / sweet)
```

Syntax, agreement and the comparative construction are all intact; what is
missing is the binding between the question's noun and the answer's. The model
has learned how the sentence goes before it has learned what goes in it.

**Longer imitation prompts still degrade.** Level 2 asks the model to repeat a
sentence; past four or five words it starts to loop:

```
say: the baby drinks the milk  → the baby drinks the baby drinks the milk!
```

Also:

- **No world knowledge.** A synthetic teaching curriculum of a few megabytes,
  plus a handful of public-domain books used only as raw text. Anything
  factual it produces is invention.
- **Closed vocabulary** — 2533 active tokens; out-of-curriculum words break it.
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
retention comes from — in this build the level-10 dreams alone moved the probe
from 36% to 75%.

**The whole English curriculum trains offline.** Every level ships its own
local teacher configuration, so no API key is needed to reproduce these weights
from scratch:

```bash
./build.sh 10 --lang en
```

The training data is `training_files/en/` — 9.5 MB of question-answer pairs
(3113 across the eleven levels) and level texts. The text phases of levels 2-5
use public-domain books as raw material: Shakespeare (L2), *Alice in Wonderland*
and *Oliver Twist* (L3), *Jane Eyre* and *Pride and Prejudice* (L4),
*Moby-Dick* (L5). Levels 6-10 use no books at all — their text phase reads a
hand-written `sentences_levelN.txt`. The graded material — what the tutor
actually teaches and scores — is the curriculum's own pairs, not the books.

## Relationship to the Italian model

Same code, same architecture, same hyper-parameters, different language folder:
`training_files/en/` instead of `training_files/it/`, its own tokenizer, its own
axiom words (`I am`, `you are`, `he is`, `it is`), its own checkpoints. Nothing
about the English run required a change to the training code — that is the
property the repository's language manifests exist to keep true.

The comparison to draw between the two is about method, not about scores: the
ladders reach different heights (10 vs 12) and the levels are not equivalent
tasks across languages.

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

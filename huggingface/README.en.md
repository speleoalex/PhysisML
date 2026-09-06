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

# PhysisML — English curriculum 0-5 (experimental)

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
  allocated, **2517 active** (`<|EOS|>` at 2516; unused slots are masked to
  `-inf` at inference)
- float32, runs on a CPU

**These weights are the level-5 checkpoint of a 0-5 build**: isolated sounds,
first words, article + noun + verb, short questions, *who* and *where*, and the
connectives *and*, *but*, *because*.

## ⚠️ Half a ladder, and a young one

The Italian model of this project is trained to level 12. **This English build
stops at level 5** — the equivalent of a five-year-old in the curriculum's own
metaphor. Everything the Italian preview is interesting for above that line
simply does not exist here:

- no class membership (`the dog is an animal`),
- no admission of ignorance, no asking about an unfamiliar name,
- no autonomy loop, no retraction ledger.

Asking this model for any of those gets a confident sentence built out of level
0-5 patterns, not an answer.

It is also **English only**: the vocabulary is 2517 English BPE tokens and the
model has never seen another language. It shares no weights and no vocabulary
with the Italian preview — it is a second, independent run of the same method,
which is the point of publishing it.

## What it can do

Short English questions inside a closed world of familiar nouns and verbs.
Every line below is an actual greedy output of these weights, taken from the
repository's own level probe:

```
say ma                              → ma!
say: the cat                        → the cat!
say: the wolf runs                  → the wolf runs!
what are you called?                → physisml!
what does the bee do?               → the bee flies.
who cooks?                          → the woman cooks.
what is the sea like?               → the sea is cold.
tell me two things about the baby   → the baby sleeps and laughs.
why does the cat sleep?             → the cat sleeps because it is tired.
why does the man work?              → the man works because he is strong.
```

## Results

Exact match against the curriculum's gold answers, greedy, on these weights
(the post-dream level-5 checkpoint), replayed over every level it has seen —
12 prompts per level:

| L0 | L1 | L2 | L3 | L4 | L5 | mean |
|----|----|----|----|----|----|------|
| 100% | 100% | 75% | 100% | 67% | 100% | **90.3%** |

On the build's own frozen probe — 48 prompts, 8 per level, fixed before
training — the level-5 checkpoint scores **85.4%**, self-repetition 4.2%.

The lever is the **dream**: a replay pass over every level's material with no
new teaching. Each level dreams until the probe stops improving, and the curve
ships in the checkpoint directory as `dream_curve.json`. Level 5's, one entry
per dream:

```
60 → 73 → 75 → 75 → 77 → 81 → 81 → 83 → 81 → 85 → 85 → 85 %
```

That level stopped at the cap of twelve dreams, not at a plateau: it was still
climbing when it ran out of budget, so the number above is a floor, not a knee.
Levels 3 and 0 stopped on a genuine plateau, levels 1, 2 and 4 on a regression
and kept their best snapshot.

The whole build took **5h53** on an Intel Arc GPU (levels 0-5, all phases
included), entirely offline.

## Where it fails

**A later level's pattern overwrites an earlier one, and the dream does not
always repair it.** The sharpest example is level 4's *where* question, which
these weights answer with level 5's *because* frame:

```
where does the fish swim?   → the fish swims because it is fast.   ✗ (gold: the fish swims in the sea.)
where does the baby sleep?  → the baby sleeps because it is tired. ✗ (gold: the baby sleeps in the house.)
```

Four prompts of that shape are the entire level-4 loss above. Note what the
model got right anyway: the verb agrees, the subject is carried over, and the
clause is well formed — the interference is at the level of *which frame the
question selects*, not of grammar.

**Longer imitation prompts degrade.** Level 2 asks the model to repeat a
sentence; past three or four words it starts to loop:

```
say: the happy girl            → the girl! say: the happy girl the girl!
say: the baby drinks the milk  → say: the girl girl!
```

Also:

- **No world knowledge.** A synthetic teaching curriculum of a few megabytes,
  plus a handful of public-domain books used only as raw text. Anything
  factual it produces is invention.
- **Closed vocabulary** — 2517 active tokens; out-of-curriculum words break it.
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
retention comes from — in this build the level-5 dream alone moved the probe
from 60% to 85%.

**The whole English curriculum trains offline.** Every level ships its own
local teacher configuration, so no API key is needed to reproduce these weights
from scratch:

```bash
./build.sh 5 --lang en
```

The training data is `training_files/en/` — 6.9 MB of question-answer pairs
(about 1650 across the six levels) and level texts. The text phases use
public-domain books as raw material: Shakespeare (L2), *Alice in Wonderland*
and *Oliver Twist* (L3), *Jane Eyre* and *Pride and Prejudice* (L4),
*Moby-Dick* (L5). The graded material — what the tutor actually teaches and
scores — is the curriculum's own pairs, not the books.

## Relationship to the Italian model

Same code, same architecture, same hyper-parameters, different language folder:
`training_files/en/` instead of `training_files/it/`, its own tokenizer, its own
axiom words (`I am`, `you are`, `he is`, `it is`), its own checkpoints. Nothing
about the English run required a change to the training code — that is the
property the repository's language manifests exist to keep true.

The comparison to draw between the two is about method, not about scores: the
ladders reach different heights (5 vs 12) and the levels are not equivalent
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

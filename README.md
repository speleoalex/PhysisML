# PhysisML

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22285422.svg)](https://doi.org/10.5281/zenodo.22285422)
[![tests](https://github.com/speleoalex/PhysisML/actions/workflows/tests.yml/badge.svg)](https://github.com/speleoalex/PhysisML/actions/workflows/tests.yml)

*Read this in: [Italiano](README.it.md)*

A small LLM built from scratch, inspired by biological learning.
The model learns like a child — sounds first, then words, then sentences —
guided by a tutor that adapts the curriculum in real time.

- **Progressive curriculum**: from phonemes to literature and class membership (Italian levels 0–12, English 0–10)
- **Innate affective system**: `confidence`, `pleasure`, `pain`, `fear` modulate the logits during inference
- **Teacher signal**: a free local teacher (or, optionally, a Claude tutor) generates targeted examples on the model's current deficits
- **Tiny footprint**: GPT-2 style transformer, ~23.6M parameters, trains on a CPU or a consumer GPU

![The published weights answering in ollama](docs/physisml_ollama.png)

The published weights in `ollama`, after
[`scripts/export_gguf.py`](scripts/export_gguf.py) — with `/set nohistory`, so
every answer stands on its own with no conversation context. The last one is
the point of the project: `cocomero` is a noun the model was never taught, and
saying so is a thing it was taught to do.

---

## Try it in two minutes

Trained weights are published on the Hugging Face Hub. Nothing has to be
trained to hear the model talk:

```bash
# --depth 1 on purpose: the curriculum corpora are versioned in the repository
# so the results are reproducible, which makes the full history ~400 MB.
git clone --depth 1 https://github.com/speleoalex/PhysisML.git
cd PhysisML
pip install -r requirements.txt

python3 standalone/chat.py "di: cosa mangia il cane?"
```

The first run downloads ~95 MB from
[`speleoalex/physisml-it-preview`](https://huggingface.co/speleoalex/physisml-it-preview)
into `models/hf/` and answers — the whole thing took 74 seconds on a CPU-only
laptop, download included:

```
il cane mangia il pane.
```

Run it with no argument for a REPL. Verbatim, greedy, on the published weights:

```
========================================================
  PhysisML — interactive generation
========================================================
  params     : 23,589,376
  vocab      : 2593 active / 9000 total
  d_model    : 512  n_heads=8  n_layers=6
  context    : 128 tokens
  temperature: 0.0  (0 = greedy)
  affect     : on
========================================================

>>> cos è il cane?
<<< il cane è un animale.

>>> il pane è buono?
<<< secondo me il pane è buono perché è caldo.

>>> cos è un falco?
<<< non lo so.
```

The model speaks Italian: it has only ever been shown an Italian curriculum.
That last answer is the point of the project — `falco` is a noun it was never
taught, and level 12 teaches it to say so instead of inventing. It is a small
model with a small vocabulary, and it will produce nonsense outside the ground
it was taught; [the FAQ](docs/en/faq.md) is blunt about what that does and does
not demonstrate.

On a CPU-only machine, add the PyTorch CPU index to the install to pull a
~200 MB wheel instead of the ~2 GB CUDA one:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

To train one from scratch instead, see [Quick start](#quick-start) below.

---

**Documentation**

| Document | EN | IT |
|---|---|---|
| **FAQ — the five objections** | [docs/en/faq.md](docs/en/faq.md) | [docs/it/faq.md](docs/it/faq.md) |
| Technical and philosophical design | [docs/en/physisml_model.md](docs/en/physisml_model.md) | [docs/it/modello_PhysisML.md](docs/it/modello_PhysisML.md) |
| Background on classic language models | [docs/en/classic_language_models.md](docs/en/classic_language_models.md) | [docs/it/modelli_linguistici_classici.md](docs/it/modelli_linguistici_classici.md) |
| Development setup | [docs/en/setup/development.md](docs/en/setup/development.md) | [docs/it/setup/development.md](docs/it/setup/development.md) |
| Intel Arc GPU setup | [docs/en/setup/gpu_intel_arc.md](docs/en/setup/gpu_intel_arc.md) | [docs/it/setup/gpu_intel_arc.md](docs/it/setup/gpu_intel_arc.md) |
| How to contribute | [CONTRIBUTING.md](CONTRIBUTING.md) | — |

---

## Results

Exact match against the curriculum's gold answers, greedy
(`scripts/measure_repetition.py`), 2026-09-01 build. The target set grew with
this rebuild — levels 11-12 now teach the honesty relation over 38 nouns in
every phrasing, 1369 graded prompts against the previous 849 — so two numbers,
each honest about what it compares:

| | |
|---|---|
| **frozen probe, 104 identical prompts — previous build vs this one** | 84.6% → **90.4%** |
| this checkpoint on every current target (1369 prompts) | 84%, self-repetition 2% |

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|---|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| every level, one model | 100% | 97% | 79% | 87% | 93% | 97% | 80% | 91% | 95% | 75% | 97% | 92% | 57% |

L12's 57% is on its own tripled-and-hardened pool (asking about the right
referent with a distractor in the prompt, yes/no graded on the polarity word —
which no grader checked before this build).

The lever is the **dream** — a replay pass over every level's material, no new
teaching. Measured at level 6, one cycle took the mean across the seven levels
seen from 23.7% to **84.3%**. Dreams are no longer counted by constant: after
each one the frozen probe is re-scored and the level stops dreaming when the
marginal gain dies (`scripts/dream_until_plateau.py`). Measured on this build,
level 11's own knee was 8 dreams — the old `MIN_DREAMS=6` would have stopped
it at 76.9% instead of 82.7% on the probe — and level 12 reached its best at
dream 9 of a sawtooth curve. Each level's curve is recorded in its checkpoint
directory as `dream_curve.json`.

Real answers, greedy:

```
di: cosa mangia il cane?               → il cane mangia il pane.
perché il cane mangia?                 → il cane mangia perché ha fame.
cosa mangerà il cane domani?           → domani il cane mangerà il pane.
cos è il cane?                         → il cane è un animale.
il pane è buono?                       → secondo me il pane è buono perché è caldo.
commenta il libro                      → il libro insegna, e questo è importante.
cos è un falco?                        → non lo so.
l albero è una pianta, questo è un tamburo  → cos è un tamburo?
```

The last two are level 12: on a name it has never met the model declares
ignorance or asks, instead of guessing. Measured on held-out names it has never
seen (`scripts/curiosity_rate.py --gate off`): 67% honest answers on unknown
nouns against **0%** on known ones — it never claims ignorance about something
it knows.

### English (levels 0-10)

A second curriculum, built from scratch on 2026-09-05/06 with its own
vocabulary, its own axioms and no weights shared with the Italian model. It now
reaches level 10 — past, future, comparatives and preferences, a thesis with
its reason, a motivated comment — but it still stops short of the is-a,
ignorance and asking behaviour levels 11-12 teach in Italian.

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|----|----|----|----|----|----|----|----|----|----|-----|
| every level, one model | 100% | 98% | 75% | 96% | 54% | 54% | 68% | 77% | 72% | 78% | 100% |

| | |
|---|---|
| this checkpoint on every current target (720 prompts) | **78.8%**, self-repetition 3.9% |
| its own frozen probe, 88 prompts | 75.0%, self-repetition 5.7% |

That row is measured the same way as the Italian one above: exact match against
*every* gold answer of every level (`scripts/measure_repetition.py`). The 0-5
card reported 90.3% for a 72-prompt sample (`test_model.py --samples 12`) — a
smaller, easier denominator. The two numbers do not compare; the difference is
a change of ruler, not a regression.

What did regress is level 5. On the same 418 prompts of levels 0-5, the level-5
checkpoint scored 86.4% and this one scores 77.5% — and 33 of the 37 lost
prompts are level 5 alone (100% → 54%). Levels 0-4 lost 3 prompts out of 346
between them. Level 5 is where *because* is taught, and levels 6-10 reuse that
frame for other jobs, so the failures keep the right shape and pick the wrong
reason: `why does the dog drink? → the dog runs because it is hungry.`

Real answers, greedy:

```
what did the boy eat?                   -> the boy ate the bread.
what will the boy eat tomorrow?         -> tomorrow the boy will eat the bread.
tell me the three days of the cat       -> today it sleeps, yesterday it slept,
                                           tomorrow it will sleep.
which is better, the bread or the door? -> the bread is better, because you eat it.
is the cat small?                       -> I think the cat is small because it is fast.
two reasons: why do you like the cat?   -> the cat is small and fast, so I like it.
comment on the cat                      -> the cat is fast, and this is nice.
what does the story say about the cat   -> the story says that the cat sleeps.
```

Levels 6-10 were built with `MAX_DREAMS=20` and none of them reached the cap
(6, 8, 13, 7 and 11 dreams): unlike level 5 in the 0-5 build, every one of them
stopped on its own plateau. Level 10's probe went 36% → 75% over its eleven
dreams. The five levels took 7h44 on an Intel Arc GPU (67/77/126/72/117 minutes
each), on top of the 5h53 of levels 0-5.

The card for these weights is [huggingface/README.en.md](huggingface/README.en.md);
they are published at
[`speleoalex/physisml-en-preview`](https://huggingface.co/speleoalex/physisml-en-preview)
and `python3 standalone/chat.py --lang en "say: the cat"` downloads and runs them.

### Is the dream just replay? The EWC control (exp_i)

The dream is experience replay — so the honest question is whether a standard
anti-forgetting method from the literature does the same job without carrying
the corpus around. Benchmark `exp_i` answers it: same network, same curriculum,
same harness, levels 0→6, two seeds, three arms — `dream` (the shipped
mechanism), `ewc` (online EWC, Schwarz et al. 2018: running Fisher
`F ← γ·F_prev + F_new`, γ=0.95, anchor refreshed at each level boundary,
λ=1000 from a preliminary sweep), `none` (the floor, λ=0). The ewc/none arms
keep every within-level consolidation channel and lose only the cross-level
replay; six dreams per level, fixed, in all arms.

| arm | retention (final checkpoint on all levels) | learning (each level on its own checkpoint) |
|---|---|---|
| dream | **64.4%** (65.0 / 63.9) | 80.7% (82.5 / 79.0) |
| ewc | 13.0% (13.6 / 12.5) | 37.1% (35.0 / 39.2) |
| none | 22.0% (20.1 / 23.9) | 77.9% (76.4 / 79.4) |

*(mean across seeds, then seed 1 / seed 2; run-to-run noise is 2.2 points)*

Three verdicts, each replicated on both seeds:

- **Replay is worth +42.5 points** of retention over no protection at all.
  The `none` arm lands at ~20% — the same number the pre-dream builds scored,
  which doubles as an internal validity check of the harness.
- **EWC lands 9 points *below* doing nothing**, and costs ~44 points of
  *current* learning. The per-level detail is stark: `none` forgets the middle
  levels but keeps L0 and the just-learned L6 at ~100%; `ewc` loses even
  those (L6 at 10–20%).
- λ is not the issue: in the sweep, retention rises monotonically with λ
  (33→41% for λ=100→10000) but never approaches the replay arm, while
  current learning falls.

**Why EWC collapses here** — kept as the sequence of failed diagnoses,
because the elimination process is the argument:

1. *Fisher accumulation over the level anchors?* Dead: γ-accumulation
   explains a factor of ~2.9× at most; the measured Fisher mass grows ~70×.
2. *A feedback spiral from estimating the Fisher on a non-converged level?*
   Dead: level-end losses on the very pairs used for estimation are ≈0
   (0.0001–0.06) even in the collapsed arm; Spearman ρ = −0.14 between final
   loss and new Fisher mass.
3. *What survives:* at loss ≈ 0 the empirical Fisher `E[g²]` is the
   **variance of the gradient across examples**, not curvature. With a
   prompt-masked SFT loss and short answers, that variance concentrates on
   the tokens every example shares: 20 embedding rows out of 2,590 carry
   89–93% of the mass — the space character alone 32–43%, then ':',
   articles, '!' — plus another 27–32% on the first attention block. The
   anchor is **anti-selective**: it freezes the machinery that produces *any*
   answer, not the knowledge of past levels. Replicated on both seeds — same
   concentration, same top tokens, ~3× different absolute mass: the damage
   tracks the concentration, not the scale.

The claim this supports is deliberately narrow: **in this near-perfect
per-level memorization curriculum, our implementation of standard online EWC
with an empirical diagonal Fisher substantially underperforms experience
replay — on both retention and current learning — including against the
unregularized baseline; replay pays for it by keeping the corpus around
instead of summary statistics.** It is not "EWC is wrong in general":
normalizing the Fisher per token, or excluding structural tokens, would be a
different algorithm (the Riemannian-Walk family, Chaudhry et al. 2018), and
the comparison is not compute-matched — the dream's N1 replays up to 7
levels per cycle against the ewc arm's one — so the relative *efficiency* of
the two methods remains open (a compute-matched arm is listed as future
work), though budget cannot explain ewc finishing below `none`. Reproduce with
`MODE=sweep|main ./scripts/experiment_ewc.sh --confirm`; per-arm retention
matrices land in `models/exp_i/`.

### Where it still fails

- **L9 (70%) and L2 (80%)** are the weak levels of the final checkpoint; L2's
  errors are repetition (`di: il fratello beve il latte` → `il letto basso! il
  letto basso!`).
- **Untaught phrasings drift onto the nearest taught pattern.** `cos è il cane?`
  is right, `cos è un cane?` (indefinite, never drilled for a concrete noun)
  answers `un animale è un essere vivente` — the class-level question's answer.
  Same for `chi è zibaldone?`, which confabulates: level 12 teaches the ask on
  `cos è un X?` and the yes/no form, not on that one.
- **The margin is thin.** Every example here is greedy. With sampling on,
  `cos è un cane?` answers *animale*, *persona* or *luce* depending on the draw.

## Requirements

```bash
pip install -r requirements.txt

# CPU-only machine: a ~200 MB wheel instead of the ~2 GB CUDA one
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Four packages: `numpy` and `torch` to train, `huggingface_hub` and
`safetensors` to fetch and load the published weights. `requirements-dev.txt`
adds pytest; `requirements-optional.txt` lists what individual scripts ask for
and guard at the import site — the Claude tutor, the corpus builder, the GGUF
exporter.

**No API key is needed to train the Italian curriculum**: every level 0-12
ships a `local_teacher.json`, so `./build.sh` teaches with the offline tutor.

### Tutors

| `--tutor-model` | What grades the answers | Needs |
|---|---|---|
| `local` | rule-based, deterministic | nothing |
| `hybrid` | local prompts + a small local LLM | [llama.cpp](https://github.com/ggml-org/llama.cpp) or [ollama](https://ollama.com) running |
| `claude-haiku-4-5`, `claude-sonnet-4-6` | Claude API | `pip install anthropic` + API key |
| `auto` *(default)* | `hybrid` → `local` when the level has `local_teacher.json`, Claude otherwise | — |

`build.sh` uses `local` for L0-L1 and `hybrid` from L2 up whenever a local LLM
answers, so a full Italian run costs nothing. The grader runs on **llama.cpp or
ollama**, whichever is up — llama.cpp's `llama-server` is preferred when both
are (one resident model, no load latency) — and it can live on another machine:

```bash
LLAMA_SERVER_BASE=http://gpu-box:8080 ./build.sh 4     # llama.cpp
OLLAMA_BASE=http://gpu-box:11434 PHYSISML_LLM_MODEL=qwen3:8b ./build.sh 4
```

With ollama the level → model mapping is a requirement: a model that is not
installed disables the LLM grader instead of substituting another one. With
llama.cpp the server hosts one model and that is the one used.

The Claude tutor stays the better teacher at the higher levels, and it is the
only tutor for the English curriculum (`training_files/en/` has no
`local_teacher.json` yet). It is optional and unlocked by:

```bash
pip install anthropic python-dotenv
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

> Trained weights are **not** included in the repository (`models/`, `*.pt` are gitignored).
> Build a model from scratch with `./build.sh`.

---

## Quick start

```bash
# Train levels 0→1 automatically (text phase + teaching per level, no API key)
./build.sh 1

# Talk to the model
python3 dynamic_model/run.py                       # interactive
python3 dynamic_model/run.py "text" 2>/dev/null    # single reply
```

### Manual sequence for one level

```bash
# Phase 0: text training on training_files/it/0/
python3 dynamic_model/train_curriculum.py --phase 0 --level 0 --epochs-0 10 --lang it

# Phase 1: tutor teaching (repeatable)
./teach.sh 100        # 100 fixed turns
./teach.sh auto       # continue until quality is reached (Ctrl-C is safe)

# Activate and test the checkpoint
./set_model.sh models/checkpoints/it/level_0/final_learned.pt
python3 dynamic_model/run.py "mamma" 2>/dev/null
```

Same sequence for higher levels: `--level 1`, `--level 2`, …
Each level starts from the previous level's `final_learned.pt`.

---

## Main commands

| Command | Purpose |
|---------|---------|
| `./build.sh N [model] [auto] [--resume]` | Auto-train levels 0→N |
| `./build.sh N --lang en` | Same, on another language's curriculum (`PHYSISML_LANG` sets it too). Every script below takes `--lang` |
| `MIN_DREAMS=6 ./build.sh N` | Dream floor per level (default 6, `0` disables). Above the floor the count is measured: dreams continue while the frozen probe still gains (`MAX_DREAMS`, `DREAM_EPSILON`, `DREAM_PATIENCE`) |
| `./teach.sh [turns\|auto] [local\|hybrid\|haiku\|…] [lang] [level]` | Teaching session |
| `./set_model.sh <checkpoint>` | Set the active model (`models/active.pt`) |
| `./reset.sh [--lang L\|all] [--dry-run]` | Backup + reset the model. Without `--lang` only the default language's checkpoints go |
| `python3 dynamic_model/train_curriculum.py` | Text training and/or teaching (see `--help`) |
| `python3 dynamic_model/test_model.py --level N` | Quality statistics for the current model |
| `python3 scripts/measure_repetition.py --ckpt-base models/checkpoints/it --levels 0-12` | Exact match **and** self-repetition rate, greedy |
| `python3 scripts/retention_matrix.py --levels 0-12` | Retention matrix: every checkpoint against every level |
| `ANTI_FORGETTING=dream\|ewc\|none ./build.sh N` | Anti-forgetting arm: `ewc` = online EWC (`EWC_LAMBDA`, `EWC_GAMMA`) with per-level Fisher sidecars (`scripts/compute_fisher.py`); `none` = no cross-level channel |
| `MODE=sweep\|main ./scripts/experiment_ewc.sh --confirm` | The dream-vs-EWC benchmark (exp_i): λ sweep on L0-L2, then 3 arms × 2 seeds on L0-L6 |
| `python3 dynamic_model/run.py` | Interactive session |
| `python3 scripts/download_wikipedia.py --level N` | Download Wikipedia articles for training |
| `python3 scripts/generate_qa_corpus.py --levels 0 1 2` | Build dialogue corpus from QA pairs |
| `python3 scripts/generate_qa_corpus.py --check --levels 0 1 2` | Verify each `qa_corpus.txt` matches its `qa_pairs.jsonl` (exits 1 if stale) |
| `python3 scripts/export_gguf.py` | Export a checkpoint to GGUF, then `ollama create physisml -f Modelfile` |
| `python3 scripts/export_hf.py [--lang en] [--out DIR]` | Build a Hugging Face upload folder (safetensors + card + inference code). Card and folder follow the language |
| `./scripts/build_status.sh` | Where a running build is: level, session, quality, what is running now |
| `python3 scripts/curiosity_rate.py --gate off` | Does it admit ignorance on unknown names and not on known ones |

Key `train_curriculum.py` flags: `--phase 0|1`, `--level N`, `--lang it|en`,
`--epochs-0 N`, `--interactions N|auto`, `--age 0-7+` (virtual age → teaching style),
`--tutor-model auto|local|hybrid|haiku|sonnet`.

**Teachers**: `local_teacher.py` (deterministic, free, offline),
`hybrid_teacher.py` (local prompts + local-LLM evaluation via llama.cpp or
ollama, free and GPU-friendly),
Claude API tutor (optional — see [Tutors](#tutors)).

---

## Project structure

```
PhysisML/
├── training_files/{it,en}/N/   ← text corpus per language and level
│   ├── *.txt                   ← training texts (qa_corpus.txt = dialogue pairs)
│   └── teacher_prompt.md       ← per-level teacher style (optional)
├── models/                     ← checkpoints (gitignored)
│   ├── active.pt               ← test copy, never touched by training
│   └── checkpoints/{lang}/level_N/{final.pt, final_learned.pt}
├── dynamic_model/              ← training, teaching, inference code
├── scripts/                    ← corpus tools, GGUF export, analysis
├── standalone/                 ← self-contained chat (model + tokenizer + REPL)
│   └── webui/                  ← FastAPI + web chat with feedback/training (see its README)
├── docs/it/                    ← in-depth documentation (Italian)
└── build.sh / teach.sh / reset.sh / set_model.sh
```

---

## Curriculum corpora

| Lang | Level | Content |
|------|-------|---------|
| it | 0 | Sounds and syllables (handwritten) |
| it | 1 | Nursery rhymes, songs, short stories, simple dialogues |
| it | 2 | Basic sentences and grammar + Wikipedia (animals) |
| it | 3 | Pinocchio + OpenSubtitles + Wikipedia |
| it | 4 | Aesop's fables + OpenSubtitles + Wikipedia (culture) |
| it | 5 | Songs + De Amicis + OpenSubtitles + Wikipedia |
| it | 6 | 19th-century fiction (Neera, Serao) |
| it | 7 | Rodari + Wikipedia |
| it | 8–9 | I Promessi Sposi (excerpt, then full) |
| it | 10 | Divina Commedia |
| it | 11 | Class membership (is-a), generated from the curated lexicon |
| it | 12 | Asking when a name has never been met |
| en | 0–1 | Sounds and basic grammar (handwritten) |
| en | 2 | Shakespeare |
| en | 3 | Alice in Wonderland + Oliver Twist |
| en | 4 | Jane Eyre + Pride & Prejudice |
| en | 5 | Moby Dick |

Every level also ships a `qa_corpus.txt` (prompt→answer dialogue pairs).

Long narrative texts stay in the repository, under
`training_files/it/N/_reference/`, but take part in **no** phase: not text
training, not teaching, not dream replay, not tokenizer construction. Adult
prose erases the prompt→answer associations the curriculum has just built, and
archaic Italian is not the language of the curriculum. The mechanism is
location, not a list: every loader globs `<level>/*.txt` non-recursively, so a
subdirectory is invisible by construction. See
[training_files/it/_reference_README.md](training_files/it/_reference_README.md).

---

## Languages

A language is a folder, never a branch in the code. `training_files/<lang>/`
holds the corpus and the teacher configs, `models/checkpoints/<lang>/` the
weights, and every script takes `--lang` (or `PHYSISML_LANG`), so two builds
never overwrite each other:

```bash
./build.sh 10 --lang en                         # build the English curriculum
./teach.sh 100 local --lang en --level 3        # one teaching session
./reset.sh --lang en                            # wipes ONLY checkpoints/en/
python3 dynamic_model/test_model.py --level 10 --lang en
python3 scripts/train_tokenizer.py --lang en --vocab-size 3000
python3 scripts/export_hf.py --lang en --out hf_en
python3 standalone/chat.py --lang en "say: the cat"  # the published weights
```

`dynamic_model/run.py` takes no language flag: it recognises the checkpoint's
own vocabulary and prints what it found before the first answer
(`Language: en  (English)  [from the vocabulary]`).

### The manifest

Most of what a language needs follows the repository's conventions and is
derived from the code alone:

| artifact | convention |
|---|---|
| vocabulary | `dynamic_model/data/tokenizer_<lang>.json` |
| frozen probe | `dynamic_model/data/probe_set_<lang>.json` |
| model card | `huggingface/README.<lang>.md` |
| export folder | `hf_upload_<lang>` |

(Italian keeps its historical names — `tokenizer_8k.json`, `probe_set.json`,
`huggingface/README.md`, `hf_upload/` — because every published checkpoint and
Hub revision was made against them.)

What is left cannot be derived from a language code, because it is *words*. It
lives in `training_files/<lang>/language.json`, and every key is optional:

| key | what it decides |
|---|---|
| `axioms` | the words whose embedding rows training protects, with their protection. They must be whole tokens of **this** language's vocabulary: on the English vocabulary the Italian axiom `mamma` encodes to `m\|am\|ma` and freezes three arbitrary subwords |
| `stop_words` | the function words, for everything that separates content from grammar |
| `polarity` | how this language says yes and no, so the grader can tell a right closed answer from its opposite |
| `teacher_fallback` | the tutor prompt used when a level ships no `teacher_prompt.md`, one band per virtual age |
| `hf_repo` | the Hub repo this language publishes to. No convention on purpose: guessing a repo name and pushing to it cannot be undone |
| `name` | the human-readable name, for screen output |

A language that omits one of the word keys gets an empty list and the caller
says so on screen. That is the intended outcome: no axiom protection at all is
better than protection applied to another language's subwords — which is
exactly what the first English build spent six hours doing.

### Adding a third language

No Python file has to change. Create `training_files/<lang>/`, one numbered
folder per level with `qa_pairs.jsonl`, `local_teacher.json` and a level text,
write `language.json`, train the vocabulary, then build:

```bash
python3 scripts/train_tokenizer.py --lang de --vocab-size 3000
python3 scripts/probe_set.py --lang de --write   # freeze the probe
./build.sh 5 --lang de
```

The rule this design enforces, and that `tests/test_language_manifest.py`
checks: **a dict keyed by language code inside a `.py` file is a list of the
languages the code knows about — complete the day it is written, silently
wrong the day a language is added.** The test walks the source looking for
new ones.

---

## Architecture

- **Model**: TorchGPT — GPT-2 style decoder-only transformer, Pre-LayerNorm,
  weight-tied LM head. In use: `d_model=512`, 6 layers, 8 heads, `d_ff=2048`, 128-token context, **23.6M parameters**.
  (The state dict serialises `lm_head.weight` and `tok_emb.weight` separately, but they are the
  same tensor: counting the file's entries gives 28.2M for a 23.6M model.)
- **Tokenizer**: byte-level BPE, 2,590 active slots out of 9,000 allocated. The
  dream phase can grow it; in the 0-12 build it did not — the same 2,590 tokens
  covered every level, which is what the punctuation-free retrain was for.
- **Affective system**: innate state (`confidence`, `pleasure`, `pain`, `fear`)
  that modulates generation and tracks learning progress.
- **Anti-forgetting**: *interleaved* rehearsal on the gold pairs during teaching
  (4 pairs every 5 turns), plus corpus replay in the dream phase. Rehearsal is
  weighted toward the current level; the cross-level work is done by the dream,
  whose N1 replays *every* level's `qa_corpus`. The dream is what turned the
  final checkpoint from 20% across all levels into ~88% — and benchmarked
  head-to-head on the same harness (not compute-matched) it retains far more
  than online EWC, which collapses below the no-protection floor
  (see [the EWC control](#is-the-dream-just-replay-the-ewc-control-exp_i)).
- **Test-then-show didactics**: the model answers *before* seeing the solution.
  The reverse order (show-then-test) measured recall after a hint rather than
  retained knowledge.

A pure-NumPy educational implementation (every layer with handwritten
`forward()`/`backward()`) is the historical base of the project.

---

## Standalone chat and Web UI

```bash
# Terminal REPL, self-contained (auto-installs deps in a local venv)
cd standalone && python3 chat.py

# Web UI: chat + admin feedback + one-click fine-tuning
cd standalone/webui && ./deploy_local.sh start
```

See [standalone/webui/README.md](standalone/webui/README.md).

### GPU (Intel Arc / XPU)

```bash
./run_gpu.sh dynamic_model/train_curriculum.py --phase 0 --level 3
```

Setup: [docs/it/setup/gpu_intel_arc.md](docs/it/setup/gpu_intel_arc.md).

---

## Notes

- `.env` (API key) and `models/` are gitignored — never commit secrets or weights.
- `training_files/` is never touched by `./reset.sh`; backups go to `models/backups/<timestamp>/`.
- Code, comments and file names are in English; Italian appears only in
  training data and Italian documentation.

---

## How to Cite

If you use PhysisML in your research, please cite it via its Zenodo DOI:

> Vernassa, A. (2026). *PhysisML: a language model trained from scratch on a developmental curriculum, with dream consolidation against catastrophic forgetting* (v1.0.0). Zenodo. <https://doi.org/10.5281/zenodo.22285423>

- **Concept DOI** (always resolves to the latest version): [10.5281/zenodo.22285422](https://doi.org/10.5281/zenodo.22285422)
- **Version DOI** (v1.0.0): [10.5281/zenodo.22285423](https://doi.org/10.5281/zenodo.22285423)

Citation metadata is also available in [CITATION.cff](CITATION.cff) (use the "Cite this repository" button on GitHub).

## License

The code in this repository is released under the [MIT License](LICENSE).

The corpora under `training_files/` and `tests/test_1/data/` are third-party
material included for reproducibility and are **not** covered by the MIT
license. They keep the terms of their respective sources — public-domain
literary texts (Project Gutenberg, Liber Liber) and subtitle corpora derived
from OpenSubtitles via the OPUS project, whose terms restrict redistribution
to non-commercial use. Check the source terms before reusing them.

# PhysisML

*Read this in: [Italiano](README.it.md)*

A small LLM built from scratch, inspired by biological learning.
The model learns like a child — sounds first, then words, then sentences —
guided by a tutor that adapts the curriculum in real time.

- **Progressive curriculum**: from phonemes to literature and class membership (Italian levels 0–12, English 0–5)
- **Innate affective system**: `confidence`, `pleasure`, `pain`, `fear` modulate the logits during inference
- **Teacher signal**: a free local teacher (or, optionally, a Claude tutor) generates targeted examples on the model's current deficits
- **Tiny footprint**: GPT-2 style transformer, ~23.6M parameters, trains on a CPU or a consumer GPU

**Documentation**

| Document | EN | IT |
|---|---|---|
| Technical and philosophical design | [docs/en/physisml_model.md](docs/en/physisml_model.md) | [docs/it/modello_PhysisML.md](docs/it/modello_PhysisML.md) |
| Background on classic language models | [docs/en/classic_language_models.md](docs/en/classic_language_models.md) | [docs/it/modelli_linguistici_classici.md](docs/it/modelli_linguistici_classici.md) |
| Development setup | [docs/en/setup/development.md](docs/en/setup/development.md) | [docs/it/setup/development.md](docs/it/setup/development.md) |
| Intel Arc GPU setup | [docs/en/setup/gpu_intel_arc.md](docs/en/setup/gpu_intel_arc.md) | [docs/it/setup/gpu_intel_arc.md](docs/it/setup/gpu_intel_arc.md) |

---

## Results

Exact match against the curriculum's gold answers, greedy, 849 graded prompts
over the thirteen levels (`scripts/measure_repetition.py`).

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 | mean |
|---|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|------|
| **final checkpoint, every level** | 100% | 97% | 80% | 83% | 93% | 90% | 85% | 93% | 90% | 70% | 85% | 79% | 100% | **88.1%** |
| each level on its own snapshot | 100% | 89% | 70% | 89% | 100% | 99% | 95% | 100% | 100% | 100% | 97% | 90% | 100% | 94.5% |

The first row is the one that matters, and the one that changed: a single model
asked everything it was ever taught, with self-repetition at 1.3%. The previous
build scored **20%** there, and ten extra consolidation cycles took it only to
48%. Here nothing needed recovering.

The lever is the **dream** — a replay pass over every level's material, no new
teaching. Measured at level 6 of this build, before and after one cycle:
23.7% → **84.3%** mean across the seven levels seen, self-repetition 11.1% →
3.4%. Teaching a level erases the earlier ones outright; one replay brings them
back and costs the current level nothing. The build runs six cycles per level.

Five things changed between the two builds — six dreams per level by default,
target pools widened from 227 to 728, a vocabulary retrained without
punctuation glued to words, one gold answer per prompt enforced, and `<|EOS|>`
registered and written into the corpus — so the gain cannot be attributed to
any one of them.

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
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

That is the whole hard dependency. **No API key is needed to train the Italian
curriculum**: every level 0-12 ships a `local_teacher.json`, so `./build.sh`
teaches with the offline tutor.

### Tutors

| `--tutor-model` | What grades the answers | Needs |
|---|---|---|
| `local` | rule-based, deterministic | nothing |
| `hybrid` | local prompts + a small local LLM | [ollama](https://ollama.com) running |
| `claude-haiku-4-5`, `claude-sonnet-4-6` | Claude API | `pip install anthropic` + API key |
| `auto` *(default)* | `hybrid` → `local` when the level has `local_teacher.json`, Claude otherwise | — |

`build.sh` uses `local` for L0-L1 and `hybrid` from L2 up whenever ollama
answers, so a full Italian run costs nothing. The hybrid grader can also live
on another machine:

```bash
OLLAMA_BASE=http://gpu-box:11434 PHYSISML_OLLAMA_MODEL=qwen3:8b ./build.sh 4
```

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
| `MIN_DREAMS=6 ./build.sh N` | Same, with a minimum of N dream cycles per level (default 6, `0` disables) |
| `./teach.sh [turns\|auto] [local\|hybrid\|haiku\|…] [lang] [level]` | Teaching session |
| `./set_model.sh <checkpoint>` | Set the active model (`models/active.pt`) |
| `./reset.sh [--dry-run]` | Backup + reset the model |
| `python3 dynamic_model/train_curriculum.py` | Text training and/or teaching (see `--help`) |
| `python3 dynamic_model/test_model.py --level N` | Quality statistics for the current model |
| `python3 scripts/measure_repetition.py --ckpt-base models/checkpoints/it --levels 0-12` | Exact match **and** self-repetition rate, greedy |
| `python3 scripts/retention_matrix.py --levels 0-12` | Retention matrix: every checkpoint against every level |
| `python3 dynamic_model/run.py` | Interactive session |
| `python3 scripts/download_wikipedia.py --level N` | Download Wikipedia articles for training |
| `python3 scripts/generate_qa_corpus.py --levels 0 1 2` | Build dialogue corpus from QA pairs |
| `python3 scripts/generate_qa_corpus.py --check --levels 0 1 2` | Verify each `qa_corpus.txt` matches its `qa_pairs.jsonl` (exits 1 if stale) |
| `python3 scripts/export_gguf.py` | Export a checkpoint to GGUF, then `ollama create physisml -f Modelfile` |
| `python3 scripts/export_hf.py --out hf_upload` | Build a Hugging Face upload folder (safetensors + card + inference code) |
| `./scripts/build_status.sh` | Where a running build is: level, session, quality, what is running now |
| `python3 scripts/curiosity_rate.py --gate off` | Does it admit ignorance on unknown names and not on known ones |

Key `train_curriculum.py` flags: `--phase 0|1`, `--level N`, `--lang it|en`,
`--epochs-0 N`, `--interactions N|auto`, `--age 0-7+` (virtual age → teaching style),
`--tutor-model auto|local|hybrid|haiku|sonnet`.

**Teachers**: `local_teacher.py` (deterministic, free, offline),
`hybrid_teacher.py` (local prompts + ollama evaluation, free and GPU-friendly),
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
  whose N1 replays *every* level's `qa_corpus`. Six dreams per level is what
  turned the final checkpoint from 20% across all levels into 89% —
  see [Results](#results).
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

## License

The code in this repository is released under the [MIT License](LICENSE).

The corpora under `training_files/` and `tests/test_1/data/` are third-party
material included for reproducibility and are **not** covered by the MIT
license. They keep the terms of their respective sources — public-domain
literary texts (Project Gutenberg, Liber Liber) and subtitle corpora derived
from OpenSubtitles via the OPUS project, whose terms restrict redistribution
to non-commercial use. Check the source terms before reusing them.

# PhysisML

*Read this in: [Italiano](README.it.md)*

A small LLM built from scratch, inspired by biological learning.
The model learns like a child — sounds first, then words, then sentences —
guided by a Claude tutor that adapts the curriculum in real time.

- **Progressive curriculum**: from phonemes to literature (Italian levels 0–10, English 0–5)
- **Innate affective system**: `confidence`, `pleasure`, `pain`, `fear` modulate the logits during inference
- **Teacher signal**: a Claude tutor (or a free local teacher) generates targeted examples on the model's current deficits
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

Exact match over the curriculum targets, post-dream checkpoints, greedy
decoding (`python3 dynamic_model/test_model.py --level N --samples 0`):

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|----|----|----|----|----|----|----|----|----|----|-----|
| 100% | 96% | 100% | 82% | 100% | 95% | 100% | 100% | 100% | 88% | 94% |

**Mean across the 11 levels: 96%.** The dream improves ten levels out of eleven
(+53 points at L7, +38 at L9 and L10).

For comparison, the May build before the fixes: L0 4.4%, L1 1.8%,
L2 12.8%, L3 1.0%, **L4 and beyond 0.0%**.

### What the 96% measures

Each level is scored on its **own** checkpoint: eleven snapshots, not a
cumulative capability. The final L10 checkpoint, asked about *every* level's
targets, scores **20%** — it can do L10 and L0, and the rest is buried.

The retention matrix makes the difference visible
(`python3 scripts/retention_matrix.py --levels 0-10`): the 96% is its
diagonal. The one row that retains earlier levels is L4 (100/83/100/71% on
L0-L3), and L4 is the only level that needed ten teaching sessions — every
session ends in a dream, and the dream's N1 replays *every* level's
`qa_corpus`. The other levels ran one to three.

Retention is therefore a function of consolidation cycles, and the damage is
not permanent. Six extra dreams on the finished L10 checkpoint — no new
teaching, only reconsolidation of what the session logs already hold
(`./scripts/experiment_extra_dreams.sh --confirm`):

| dreams | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|--------|---|---|---|---|---|----|----|
| exact across all levels | 20% | 27% | 36% | 43% | 44% | 48% | **48%** |
| answers with repetition | 37% | 25% | 19% | 17% | 18% | 15% | **18%** |

Every level improves and L10 stays at 100%. The curve **saturates between the
sixth and tenth dream**: +3.6 points per dream from 1 to 6, +1.0 from 7 to 12
— and that second slope is below the measured noise between two identical runs
(2.2 points), so it is indistinguishable from zero. The first six dreams do the
work.

But the ceiling is **48%, not 96%**: consolidation recovers about half the gap.
Per level, after twelve dreams, the rest is still 43 to 74 points below the
diagonal. That second half is not recoverable by dreaming — rehearsal during
teaching draws only on the current level's gold pairs, and changing that needs
a rebuild.

Real examples (greedy, post-dream checkpoints):

```
di ma                      → ma!
di: il cane                → il cane!
di: il cane dorme          → il cane dorme!
di: cosa mangia il cane?   → il cane mangia il pane.
perché il cane mangia?     → il cane mangia perché ha fame.
cosa ha mangiato il cane?  → il cane ha mangiato il pane.
cosa mangerà il cane domani?           → domani il cane mangerà il pane.
chi è più grande, il cane o il gatto?  → il cane è più grande del gatto.
il pane è buono?                       → secondo me il pane è buono perché è caldo.
commenta il libro                      → il libro insegna, e questo è importante.
```

The remaining errors fall into two families: targets sharing a prompt prefix
collapse onto the same answer (`di un numero: tre` and `di un colore: rosso`
both yield `due!`), and at levels 9-10 the model repeats the opening of the
answer before completing it (`il cane il cane il cane è fedele`). It is not a
capacity limit — pure SFT on a level's targets reaches 100% in 30 epochs.

Details in [docs/en/physisml_model.md](docs/en/physisml_model.md).

## Requirements

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install anthropic python-dotenv
```

An Anthropic API key is required only for teaching with the Claude tutor:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

> Trained weights are **not** included in the repository (`models/`, `*.pt` are gitignored).
> Build a model from scratch with `./build.sh`.

---

## Quick start

```bash
# Train levels 0→1 automatically (text phase + Claude teaching per level)
./build.sh 1

# Talk to the model
python3 dynamic_model/run.py                       # interactive
python3 dynamic_model/run.py "text" 2>/dev/null    # single reply
```

### Manual sequence for one level

```bash
# Phase 0: text training on training_files/it/0/
python3 dynamic_model/train_curriculum.py --phase 0 --level 0 --epochs-0 10 --lang it

# Phase 1: Claude teaching (repeatable)
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
| `./teach.sh [turns\|auto] [model] [lang] [level]` | Claude teaching session |
| `./set_model.sh <checkpoint>` | Set the active model (`models/active.pt`) |
| `./reset.sh [--dry-run]` | Backup + reset the model |
| `python3 dynamic_model/train_curriculum.py` | Text training and/or teaching (see `--help`) |
| `python3 dynamic_model/test_model.py --level N` | Quality statistics for the current model |
| `python3 scripts/measure_repetition.py --ckpt-base models/checkpoints/it --levels 0-10` | Exact match **and** self-repetition rate, greedy |
| `python3 scripts/retention_matrix.py --levels 0-10` | Retention matrix: every checkpoint against every level |
| `python3 dynamic_model/run.py` | Interactive session |
| `python3 scripts/download_wikipedia.py --level N` | Download Wikipedia articles for training |
| `python3 scripts/generate_qa_corpus.py --levels 0 1 2` | Build dialogue corpus from QA pairs |
| `python3 scripts/export_gguf.py` | Export a checkpoint to GGUF (llama.cpp / ollama) |

Key `train_curriculum.py` flags: `--phase 0|1`, `--level N`, `--lang it|en`,
`--epochs-0 N`, `--interactions N|auto`, `--age 0-7+` (virtual age → teaching style),
`--tutor-model haiku|sonnet|opus`.

**Teachers**: Claude API tutor (default), `local_teacher.py` (deterministic, free),
`hybrid_teacher.py` (local prompts + Ollama evaluation, free and GPU-friendly).

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
- **Tokenizer**: 8,000-token BPE with dormant slots up to 9,000: the vocabulary
  grows during the dream phase (8,002 → 8,083 tokens from L0 to L10).
- **Affective system**: innate state (`confidence`, `pleasure`, `pain`, `fear`)
  that modulates generation and tracks learning progress.
- **Anti-forgetting**: *interleaved* rehearsal on the gold pairs during teaching
  (4 pairs every 5 turns), plus corpus replay in the dream phase. Both are
  weighted toward the current level, and measurement shows this holds *within*
  a level but not *across* levels — see [What the 96% measures](#what-the-96-measures).
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

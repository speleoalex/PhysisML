# PhysisML

*Read this in: [Italiano](README.it.md)*

A small LLM built from scratch, inspired by biological learning.
The model learns like a child — sounds first, then words, then sentences —
guided by a Claude tutor that adapts the curriculum in real time.

- **Progressive curriculum**: from phonemes to literature (Italian levels 0–10, English 0–5)
- **Innate affective system**: `confidence`, `pleasure`, `pain`, `fear` modulate the logits during inference
- **Teacher signal**: a Claude tutor (or a free local teacher) generates targeted examples on the model's current deficits
- **Tiny footprint**: GPT-2 style transformer, ~3.7M parameters, trains on a CPU

**Documentation**

| Document | EN | IT |
|---|---|---|
| Technical and philosophical design | [docs/en/physisml_model.md](docs/en/physisml_model.md) | [docs/it/modello_PhysisML.md](docs/it/modello_PhysisML.md) |
| Background on classic language models | [docs/en/classic_language_models.md](docs/en/classic_language_models.md) | [docs/it/modelli_linguistici_classici.md](docs/it/modelli_linguistici_classici.md) |
| Development setup | [docs/en/setup/development.md](docs/en/setup/development.md) | [docs/it/setup/development.md](docs/it/setup/development.md) |
| Intel Arc GPU setup | [docs/en/setup/gpu_intel_arc.md](docs/en/setup/gpu_intel_arc.md) | [docs/it/setup/gpu_intel_arc.md](docs/it/setup/gpu_intel_arc.md) |

---

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
splx_model/
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

---

## Architecture

- **Model**: TorchGPT — GPT-2 style decoder-only transformer, Pre-LayerNorm,
  weight-tied LM head. Default: `d_model=256`, 4 layers, 4 heads, ~3.7M parameters.
- **Tokenizer**: 501-token BPE trained only on level-0 texts (no contamination
  from advanced texts).
- **Affective system**: innate state (`confidence`, `pleasure`, `pain`, `fear`)
  that modulates generation and tracks learning progress.
- **Anti-forgetting**: rehearsal mini-batches during teaching; show-then-test
  didactics (the model sees the correct answer before being asked).

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

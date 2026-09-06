# PhysisML development environment

*Read this in: [Italiano](../../it/setup/development.md)*

System: KDE Neon 24.04 (Ubuntu), kernel 6.17.x, i7-1360P 16 threads, 32GB RAM

---

## Python dependencies

```bash
# Base installation (CPU, always required)
pip3 install torch==2.8.0 --break-system-packages
pip3 install numpy pytest --break-system-packages

# IPEX CPU (Intel AVX-512/AMX optimizations, +1.5-2× on CPU)
pip3 install intel-extension-for-pytorch==2.8.0 --break-system-packages

# For Intel Arc A370M GPU — see gpu_intel_arc.md
```

No other dependencies. The project only uses standard Python + NumPy + PyTorch.

---

## Project structure

```
PhysisML/
├── docs/
│   ├── en/setup/           ← this folder
│   └── it/setup/           ← Italian version
├── dynamic_model/          ← PhysisML (Experiment B — affective system)
│   ├── exp_b/              ← TrainerB, AffectState, Modulator, Axioms
│   ├── train_curriculum.py ← main pipeline
│   ├── local_teacher.py    ← local teacher (L0-L2, no API)
│   └── run.py              ← interactive chat/training
├── tests/test_1/physisml/  ← core library (TorchGPT, BPETokenizer, ...)
├── training_files/it/      ← corpus per level (0-10)
├── models/                 ← active checkpoints
├── scripts/                ← analyze_log.py, compare_checkpoints.py, ...
├── diario/                 ← daily notes (local only, not published)
└── build.sh                ← automatic build L0→Ln
```

---

## Main commands

```bash
# Automatic build (L0→L2 with local teacher, L3+ with Sonnet)
./build.sh 2 sonnet auto        # levels 0→2
./build.sh 3 sonnet auto        # levels 0→3
./build.sh 3 sonnet auto --resume  # resumes from where it stopped

# Chat with the model
python3 dynamic_model/run.py --chat --max_tokens 20

# Session analysis
python3 scripts/analyze_log.py --all
python3 scripts/compare_checkpoints.py --ppl-timeline

# Between-sessions experiment
./scripts/experiment_between_sessions.sh dream-only

# 8K tokenizer training
python3 scripts/train_tokenizer.py --vocab-size 8000 --sample 50
```

---

## Build configuration (build.sh)

```bash
EPOCHS_0=3              # text training epochs (3 = time/quality trade-off)
MAX_TEACH_TURNS=200     # maximum turns per teaching session
MAX_SESSIONS=4          # maximum sessions per level before advancing
BETWEEN_SESSIONS="dream-only"  # dream-only | standard | none
TUTOR_MODEL="haiku"     # haiku | sonnet | opus (default for Claude)
```

The local teacher is selected automatically whenever
`training_files/{lang}/{level}/local_teacher.json` exists: `local` for L0-L1,
`hybrid` from L2 up if a local LLM answers — llama.cpp on
`$LLAMA_SERVER_BASE` (default `http://localhost:8080`), else ollama on
`$OLLAMA_BASE` (default `http://localhost:11434`). `TUTOR_MODEL` only decides
which Claude model to use for the levels that have no local teacher config.

---

## Claude API Key (optional)

Not required for the Italian curriculum: levels 0-12 all ship a
`local_teacher.json`, so `./build.sh` runs offline with the local/hybrid tutor.

It is required only for a level with no `local_teacher.json` — currently the
English curriculum (`training_files/en/`). Save it in `.env` (already in
`.gitignore`):
```
ANTHROPIC_API_KEY=sk-ant-...
```
and install the optional SDK: `pip install anthropic python-dotenv`.

Compute device — the CPU and the Intel Arc are interchangeable mid-build,
because a checkpoint written on one reads back bit-identically on the other:
```
PHYSISML_DEVICE=auto   # default: the Arc when usable, else the CPU
PHYSISML_DEVICE=cpu    # force the CPU (the GPU is busy, or a run must match an earlier CPU one)
PHYSISML_DEVICE=xpu    # force the Arc
```
A request that cannot be honoured falls back to the CPU and says so on stderr.
Measured at level 6: a dream takes 6 minutes on the Arc against 27 on the CPU,
and the retention it produces is the same within noise. The GPU path needs the
conda env `physisml_gpu` with a `+xpu` torch wheel and oneAPI 2025.3.

Hybrid-tutor overrides — which server hosts the grader, and which model:
```
PHYSISML_LLM_BACKEND=llamacpp          # auto (default) | llamacpp | ollama | off
LLAMA_SERVER_BASE=http://gpu-box:8080  # default http://localhost:8080
OLLAMA_BASE=http://gpu-box:11434       # default http://localhost:11434
PHYSISML_LLM_MODEL=qwen3:8b            # force one grader for every level
```
`auto` probes llama.cpp first, then ollama. To see what was found:
```bash
python3 -m dynamic_model.llm_backend        # or with a model name to check it
```

---

## Tokenizer

```
dynamic_model/data/tokenizer_base.json   ← 503 tokens (it/0 phonemes + EOS@256)
dynamic_model/data/tokenizer_8k.json     ← 2590 tokens, Italian (EOS@2589)
dynamic_model/data/tokenizer_en.json     ← 2517 tokens, English (EOS@2516)
```

One vocabulary per language: `tokenizer_8k.json` for `it` (the name is
historical — `--vocab-size` is only a ceiling, the merge threshold decides the
real size), `tokenizer_<lang>.json` for every other language. The Italian
vocabulary leaves 117 of the 146 English gold words in pieces
(`thirsty` → `t|h|i|r|st|y`) and compresses English at 1.4 chars/token against
2.4 on Italian, so a language without its own file builds badly — phase 0 warns
on screen when it falls back.

Training: `python3 scripts/train_tokenizer.py --lang en --vocab-size 3000`

Everything else a language needs that cannot be derived from its code — axioms,
function words, yes/no spellings, the tutor fallback, the Hub repo — lives in
`training_files/<lang>/language.json` and is read by `dynamic_model/language.py`.
Adding a language means adding files, never editing a module: see
[Languages](../../README.md#languages).

---

## GPU (optional but recommended)

See `gpu_intel_arc.md` for the full procedure.
Once IPEX XPU is installed, the code uses the GPU automatically:
```python
# No changes needed — auto-detection in torch_model.py
from physisml.torch_model import DEVICE  # cpu | xpu | cuda | mps
```

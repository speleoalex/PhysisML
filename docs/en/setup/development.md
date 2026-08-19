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
splx_model/
├── docs/
│   ├── en/setup/        ← this folder
│   └── it/setup/        ← Italian version
├── dynamic_model/       ← PhysisML (Experiment B — affective system)
│   ├── exp_b/           ← TrainerB, AffectState, Modulator, Axioms
│   ├── train_curriculum.py  ← main pipeline
│   ├── local_teacher.py     ← local teacher (L0-L2, no API)
│   └── run.py               ← interactive chat/training
├── tests/test_1/splx/   ← core library (TorchGPT, BPETokenizer, ...)
├── training_files/it/   ← corpus per level (0-10)
├── models/              ← active checkpoints
├── scripts/             ← analyze_log.py, compare_checkpoints.py, ...
├── diario/              ← daily notes (local only, not published)
└── build.sh             ← automatic build L0→Ln
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

The local teacher (L0-L2) is selected automatically if
`training_files/{lang}/{level}/local_teacher.json` exists.

---

## Claude API Key

Required for the levels using Sonnet/Haiku (L3+).
Save it in `.env` (already in `.gitignore`):
```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Tokenizer

```
dynamic_model/data/tokenizer_base.json   ← 503 tokens (it/0 phonemes + EOS@256)
dynamic_model/data/tokenizer_8k.json     ← 8001 tokens (full corpus + EOS@8000)
```

The system automatically uses `tokenizer_8k.json` if present.
Training: `python3 scripts/train_tokenizer.py --vocab-size 8000 --sample 50`

---

## GPU (optional but recommended)

See `gpu_intel_arc.md` for the full procedure.
Once IPEX XPU is installed, the code uses the GPU automatically:
```python
# No changes needed — auto-detection in torch_model.py
from splx.torch_model import DEVICE  # cpu | xpu | cuda | mps
```

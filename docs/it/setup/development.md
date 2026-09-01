# Ambiente di sviluppo PhysisML

*Leggi in: [English](../../en/setup/development.md)*

Sistema: KDE Neon 24.04 (Ubuntu), kernel 6.17.x, i7-1360P 16 thread, 32GB RAM

---

## Dipendenze Python

```bash
# Installazione base (CPU, sempre necessaria)
pip3 install torch==2.8.0 --break-system-packages
pip3 install numpy pytest --break-system-packages

# IPEX CPU (ottimizzazioni Intel AVX-512/AMX, +1.5-2× su CPU)
pip3 install intel-extension-for-pytorch==2.8.0 --break-system-packages

# Per GPU Intel Arc A370M — vedi gpu_intel_arc.md
```

Nessun'altra dipendenza. Il progetto usa solo Python standard + NumPy + PyTorch.

---

## Struttura progetto

```
PhysisML/
├── docs/
│   ├── it/setup/        ← questa cartella
│   └── en/setup/        ← versione inglese
├── dynamic_model/       ← PhysisML (Esperimento B — sistema affettivo)
│   ├── exp_b/           ← TrainerB, AffectState, Modulator, Axioms
│   ├── train_curriculum.py  ← pipeline principale
│   ├── local_teacher.py     ← teacher locale (L0-L2, no API)
│   └── run.py               ← chat/training interattivo
├── tests/test_1/splx/   ← libreria core (TorchGPT, BPETokenizer, ...)
├── training_files/it/   ← corpus per livello (0-10)
├── models/              ← checkpoints attivi
├── scripts/             ← analyze_log.py, compare_checkpoints.py, ...
├── diario/              ← note giornaliere (solo locale, non pubblicate)
└── build.sh             ← build automatico L0→Ln
```

---

## Comandi principali

```bash
# Build automatico (L0→L2 con teacher locale, L3+ con Sonnet)
./build.sh 2 sonnet auto        # livelli 0→2
./build.sh 3 sonnet auto        # livelli 0→3
./build.sh 3 sonnet auto --resume  # riprende da dove si era fermato

# Chat con il modello
python3 dynamic_model/run.py --chat --max_tokens 20

# Analisi sessioni
python3 scripts/analyze_log.py --all
python3 scripts/compare_checkpoints.py --ppl-timeline

# Esperimento between-sessions
./scripts/experiment_between_sessions.sh dream-only

# Training tokenizer 8K
python3 scripts/train_tokenizer.py --vocab-size 8000 --sample 50
```

---

## Configurazione build (build.sh)

```bash
EPOCHS_0=3              # epoche training testuale (3 = bilancio tempo/qualità)
MAX_TEACH_TURNS=200     # turni massimi per sessione teaching
MAX_SESSIONS=4          # sessioni massime per livello prima di avanzare
BETWEEN_SESSIONS="dream-only"  # dream-only | standard | none
TUTOR_MODEL="haiku"     # haiku | sonnet | opus (default per Claude)
```

Il teacher locale viene selezionato automaticamente quando esiste
`training_files/{lang}/{level}/local_teacher.json`: `local` per L0-L1,
`hybrid` da L2 in su se risponde un LLM locale — llama.cpp su
`$LLAMA_SERVER_BASE` (default `http://localhost:8080`), altrimenti ollama su
`$OLLAMA_BASE` (default `http://localhost:11434`). `TUTOR_MODEL` decide solo
quale modello Claude usare per i livelli che non hanno una config locale.

---

## API Key Claude (opzionale)

Non serve per il curriculum italiano: i livelli 0-12 hanno tutti un
`local_teacher.json`, quindi `./build.sh` gira offline col tutor local/hybrid.

Serve solo per un livello privo di `local_teacher.json` — attualmente il
curriculum inglese (`training_files/en/`). Salvare in `.env` (già in
`.gitignore`):
```
ANTHROPIC_API_KEY=sk-ant-...
```
e installare l'SDK opzionale: `pip install anthropic python-dotenv`.

Dispositivo di calcolo — CPU e Intel Arc sono interscambiabili anche a metà
build, perché un checkpoint scritto su uno si rilegge identico sull'altro:
```
PHYSISML_DEVICE=auto   # default: l'Arc quando è usabile, altrimenti la CPU
PHYSISML_DEVICE=cpu    # forza la CPU (GPU occupata, o run da confrontare con una su CPU)
PHYSISML_DEVICE=xpu    # forza l'Arc
```
Una richiesta che non si può soddisfare ricade sulla CPU e lo dice su stderr.
Misurato a L6: un sogno dura 6 minuti sull'Arc contro 27 su CPU, e la ritenzione
che produce è la stessa entro il rumore. Il percorso GPU richiede l'env conda
`physisml_gpu` con una wheel torch `+xpu` e oneAPI 2025.3.

Override per il tutor ibrido — quale server ospita il valutatore e quale modello:
```
PHYSISML_LLM_BACKEND=llamacpp          # auto (default) | llamacpp | ollama | off
LLAMA_SERVER_BASE=http://gpu-box:8080  # default http://localhost:8080
OLLAMA_BASE=http://gpu-box:11434       # default http://localhost:11434
PHYSISML_LLM_MODEL=qwen3:8b            # forza un solo valutatore per ogni livello
```
`auto` prova prima llama.cpp, poi ollama. Per vedere cosa ha trovato:
```bash
python3 -m dynamic_model.llm_backend        # o con un nome di modello da verificare
```

---

## Tokenizer

```
dynamic_model/data/tokenizer_base.json   ← 503 token (fonemi it/0 + EOS@256)
dynamic_model/data/tokenizer_8k.json     ← 8001 token (corpus completo + EOS@8000)
```

Il sistema usa automaticamente `tokenizer_8k.json` se presente.
Training: `python3 scripts/train_tokenizer.py --vocab-size 8000 --sample 50`

---

## GPU (opzionale ma consigliato)

Vedere `gpu_intel_arc.md` per la procedura completa.
Una volta installato IPEX XPU, il codice usa la GPU automaticamente:
```python
# Nessuna modifica necessaria — auto-detection in torch_model.py
from splx.torch_model import DEVICE  # cpu | xpu | cuda | mps
```

# PhysisML

*Leggi in: [English](README.md)*

Un piccolo LLM costruito da zero, ispirato all'apprendimento biologico.
Il modello impara come un bambino — prima i suoni, poi le parole, poi le frasi —
guidato da un tutor Claude che adatta il curriculum in tempo reale.

- **Curriculum progressivo**: da fonemi a letteratura (italiano livelli 0–10, inglese 0–5)
- **Sistema affettivo innato**: `confidence`, `pleasure`, `pain`, `fear` modulano i logits durante l'inference
- **Segnale insegnante**: un tutor Claude (o un insegnante locale gratuito) genera esempi mirati sui deficit correnti del modello
- **Dimensioni minime**: transformer GPT-2 style, ~23.6M parametri, si addestra su CPU o GPU consumer

**Documentazione**

| Documento | IT | EN |
|---|---|---|
| Progetto tecnico e filosofico | [docs/it/modello_PhysisML.md](docs/it/modello_PhysisML.md) | [docs/en/physisml_model.md](docs/en/physisml_model.md) |
| Contesto sui modelli linguistici classici | [docs/it/modelli_linguistici_classici.md](docs/it/modelli_linguistici_classici.md) | [docs/en/classic_language_models.md](docs/en/classic_language_models.md) |
| Setup di sviluppo | [docs/it/setup/development.md](docs/it/setup/development.md) | [docs/en/setup/development.md](docs/en/setup/development.md) |
| Setup GPU Intel Arc | [docs/it/setup/gpu_intel_arc.md](docs/it/setup/gpu_intel_arc.md) | [docs/en/setup/gpu_intel_arc.md](docs/en/setup/gpu_intel_arc.md) |

---

## Risultati

Exact match sugli obiettivi del curriculum, checkpoint post-sogno, decoding
greedy (`python3 dynamic_model/test_model.py --level N --samples 0`):

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7–L10 |
|----|----|----|----|----|----|----|--------|
| 100% | 96% | 100% | 82% | 100% | 95% | 100% | in ricostruzione |

Per confronto, il build di maggio prima delle correzioni: L0 4.4%, L1 1.8%,
L2 12.8%, L3 1.0%, **L4 e oltre 0.0%**.

Esempi reali (greedy, checkpoint post-sogno):

```
di ma                      → ma!
di: il cane                → il cane!
di: il cane dorme          → il cane dorme!
di: cosa mangia il cane?   → il cane mangia il pane.
perché il cane mangia?     → il cane mangia perché ha fame.
cosa ha mangiato il cane?  → il cane ha mangiato il pane.
```

Gli errori residui sono di una sola famiglia: obiettivi che condividono il
prefisso del prompt collassano sulla stessa risposta (`di un numero: tre` e
`di un colore: rosso` producono entrambi `due!`). Non è un limite di capacità —
un SFT puro sugli obiettivi di un livello li porta al 100% in 30 epoche.

Dettagli in [docs/it/modello_PhysisML.md](docs/it/modello_PhysisML.md).

## Requisiti

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install anthropic python-dotenv
```

La chiave API Anthropic serve solo per l'insegnamento con il tutor Claude:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

> I pesi addestrati **non** sono nel repository (`models/`, `*.pt` sono in `.gitignore`).
> Costruisci un modello da zero con `./build.sh`.

---

## Avvio rapido

```bash
# Addestra i livelli 0→1 in automatico (fase testuale + insegnamento Claude per livello)
./build.sh 1

# Parla col modello
python3 dynamic_model/run.py                       # interattivo
python3 dynamic_model/run.py "testo" 2>/dev/null   # risposta singola
```

### Sequenza manuale per un livello

```bash
# Phase 0: training testuale su training_files/it/0/
python3 dynamic_model/train_curriculum.py --phase 0 --level 0 --epochs-0 10 --lang it

# Phase 1: insegnamento Claude (ripetibile)
./teach.sh 100        # 100 turni fissi
./teach.sh auto       # continua finché la qualità è raggiunta (Ctrl-C sicuro)

# Attiva e testa il checkpoint
./set_model.sh models/checkpoints/it/level_0/final_learned.pt
python3 dynamic_model/run.py "mamma" 2>/dev/null
```

Stessa sequenza per i livelli successivi: `--level 1`, `--level 2`, …
Ogni livello parte dal `final_learned.pt` del livello precedente.

---

## Comandi principali

| Comando | Scopo |
|---------|-------|
| `./build.sh N [modello] [auto] [--resume]` | Addestramento automatico livelli 0→N |
| `./teach.sh [turni\|auto] [modello] [lingua] [livello]` | Sessione di insegnamento Claude |
| `./set_model.sh <checkpoint>` | Imposta il modello attivo (`models/active.pt`) |
| `./reset.sh [--dry-run]` | Backup + reset del modello |
| `python3 dynamic_model/train_curriculum.py` | Training testuale e/o insegnamento (vedi `--help`) |
| `python3 dynamic_model/test_model.py --level N` | Statistiche di qualità del modello corrente |
| `python3 dynamic_model/run.py` | Sessione interattiva |
| `python3 scripts/download_wikipedia.py --level N` | Scarica articoli Wikipedia per il training |
| `python3 scripts/generate_qa_corpus.py --levels 0 1 2` | Genera corpus dialogico dalle coppie QA |
| `python3 scripts/export_gguf.py` | Esporta un checkpoint in GGUF (llama.cpp / ollama) |

Flag principali di `train_curriculum.py`: `--phase 0|1`, `--level N`, `--lang it|en`,
`--epochs-0 N`, `--interactions N|auto`, `--age 0-7+` (età virtuale → stile di
insegnamento), `--tutor-model haiku|sonnet|opus`.

**Insegnanti disponibili**: tutor Claude via API (default), `local_teacher.py`
(deterministico, gratuito), `hybrid_teacher.py` (prompt locali + valutazione
Ollama, gratuito e con GPU).

---

## Struttura del progetto

```
PhysisML/
├── training_files/{it,en}/N/   ← corpus testuale per lingua e livello
│   ├── *.txt                   ← testi di training (qa_corpus.txt = coppie dialogiche)
│   └── teacher_prompt.md       ← stile insegnante per livello (opzionale)
├── models/                     ← checkpoint (in .gitignore)
│   ├── active.pt               ← copia per test, mai modificata dal training
│   └── checkpoints/{lang}/level_N/{final.pt, final_learned.pt}
├── dynamic_model/              ← codice di training, insegnamento, inference
├── scripts/                    ← strumenti corpus, export GGUF, analisi
├── standalone/                 ← chat autoconsistente (modello + tokenizer + REPL)
│   └── webui/                  ← FastAPI + chat web con feedback/training (README dedicato)
├── docs/it/                    ← documentazione approfondita
└── build.sh / teach.sh / reset.sh / set_model.sh
```

Per ogni livello ci sono due checkpoint: `final.pt` (solo training testuale) e
`final_learned.pt` (conoscenza appresa nell'insegnamento, aggiornato ad ogni sessione).

---

## Corpus del curriculum

| Lingua | Livello | Contenuto |
|--------|---------|-----------|
| it | 0 | Suoni e sillabe (scritto a mano) |
| it | 1 | Parole singole, famiglia, filastrocche |
| it | 2 | Articolo + sostantivo, frasi base |
| it | 3 | Frasi soggetto+verbo, identità, numeri e colori |
| it | 4 | Soggetto+verbo+oggetto, sequenze prima/poi |
| it | 5 | Connettivi (e / ma / perché), aggettivi |
| it | 6 | Passato prossimo, cause, due frasi collegate |
| it | 7 | Futuro, contrasto fra i tempi, dialogo breve |
| it | 8 | Comparativi, preferenze motivate, descrizioni |
| it | 9 | Tesi + motivo + conclusione, sinonimi |
| it | 10 | Commento motivato, confronto, citazione |
| en | 0–1 | Suoni e grammatica base (scritto a mano) |
| en | 2 | Shakespeare |
| en | 3 | Alice in Wonderland + Oliver Twist |
| en | 4 | Jane Eyre + Pride & Prejudice |
| en | 5 | Moby Dick |

Ogni livello ha un `local_teacher.json` (pool chiuso di obiettivi, deterministico),
un testo curato coerente col livello e un `qa_corpus.txt` generato dalle sessioni
(coppie prompt→risposta). I testi narrativi lunghi restano nel repository ma sono
esclusi dall'addestramento dal livello 3 in su: prosa per adulti cancella le
associazioni prompt→risposta appena costruite.

---

## Architettura

- **Modello**: TorchGPT — transformer decoder-only GPT-2 style, Pre-LayerNorm,
  testa LM con weight tying. Configurazione in uso: `d_model=512`, 6 layer,
  8 head, `d_ff=2048`, contesto 128 token, **23.6M parametri**.
- **Tokenizer**: BPE da 8.000 token, con slot dormienti fino a 9.000: il
  vocabolario cresce durante la fase di sogno (8.002 → 8.079 token da L0 a L10).
- **Sistema affettivo**: stato innato (`confidence`, `pleasure`, `pain`, `fear`)
  che modula la generazione e traccia lo stato di apprendimento.
- **Anti-forgetting**: rehearsal *interleaved* sulle coppie gold durante
  l'insegnamento (4 coppie ogni 5 turni), più il replay del corpus nella fase
  di sogno.
- **Didattica test-then-show**: il modello risponde *prima* di vedere la
  soluzione. L'ordine inverso (show-then-test) misurava il richiamo dopo
  suggerimento invece della conoscenza ritenuta.

L'implementazione originaria in pure NumPy (ogni layer con `forward()`/`backward()`
scritti a mano) resta la base didattica del progetto.

---

## Chat standalone e Web UI

```bash
# REPL da terminale, autoconsistente (installa le dipendenze in una venv locale)
cd standalone && python3 chat.py

# Web UI: chat + feedback admin + fine-tuning con un click
cd standalone/webui && ./deploy_local.sh start
```

Dettagli in [standalone/webui/README.md](standalone/webui/README.md).

### GPU (Intel Arc / XPU)

```bash
./run_gpu.sh dynamic_model/train_curriculum.py --phase 0 --level 3
```

Setup: [docs/it/setup/gpu_intel_arc.md](docs/it/setup/gpu_intel_arc.md).

---

## Note

- `.env` (chiave API) e `models/` sono in `.gitignore` — non committare mai segreti o pesi.
- `training_files/` non viene mai toccato da `./reset.sh`; i backup vanno in
  `models/backups/<timestamp>/`.
- Codice, commenti e nomi di file sono in inglese; l'italiano è usato solo nei
  dati di training e nella documentazione italiana.

---

## Licenza

Il codice di questo repository è rilasciato sotto [licenza MIT](LICENSE).

I corpora in `training_files/` e `tests/test_1/data/` sono materiale di terze
parti, incluso per riproducibilità, e **non** sono coperti dalla licenza MIT.
Mantengono i termini delle rispettive fonti: testi letterari di pubblico
dominio (Project Gutenberg, Liber Liber) e corpora di sottotitoli derivati da
OpenSubtitles tramite il progetto OPUS, i cui termini limitano la
redistribuzione all'uso non commerciale. Verifica i termini della fonte prima
di riutilizzarli.

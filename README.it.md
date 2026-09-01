# PhysisML

*Leggi in: [English](README.md)*

Un piccolo LLM costruito da zero, ispirato all'apprendimento biologico.
Il modello impara come un bambino — prima i suoni, poi le parole, poi le frasi —
guidato da un tutor che adatta il curriculum in tempo reale.

- **Curriculum progressivo**: da fonemi a letteratura e appartenenza a classi (italiano livelli 0–12, inglese 0–5)
- **Sistema affettivo innato**: `confidence`, `pleasure`, `pain`, `fear` modulano i logits durante l'inference
- **Segnale insegnante**: un insegnante locale gratuito (o, opzionalmente, un tutor Claude) genera esempi mirati sui deficit correnti del modello
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

Exact match contro le risposte gold del curriculum, greedy
(`scripts/measure_repetition.py`), build del 2026-09-01. Il set di target è
cresciuto con questo rebuild — i livelli 11-12 ora insegnano la relazione
dell'onestà su 38 nomi in ogni formulazione, 1369 prompt contro gli 849
precedenti — quindi due numeri, ognuno onesto su cosa confronta:

| | |
|---|---|
| **probe congelato, 104 prompt identici — build precedente vs questa** | 84.6% → **90.4%** |
| questo checkpoint su tutti i target attuali (1369 prompt) | 84%, auto-ripetizione 2% |

| | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 | L11 | L12 |
|---|----|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| tutti i livelli, un solo modello | 100% | 97% | 79% | 87% | 93% | 97% | 80% | 91% | 95% | 75% | 97% | 92% | 57% |

Il 57% di L12 è sul suo pool triplicato e più duro (chiedere del referente
giusto con un distrattore nel prompt, sì/no valutato sulla parola di polarità —
che nessun valutatore controllava prima di questo build).

La leva è il **sogno** — un ripasso del materiale di ogni livello, senza nuovo
insegnamento. Misurato al livello 6, un ciclo ha portato la media sui sette
livelli visti da 23.7% a **84.3%**. I sogni non si contano più con una
costante: dopo ognuno il probe congelato viene rimisurato e il livello smette
di sognare quando il guadagno marginale muore
(`scripts/dream_until_plateau.py`). Misurato su questo build, il ginocchio di
L11 era a 8 sogni — il vecchio `MIN_DREAMS=6` lo avrebbe fermato a 76.9%
invece di 82.7% sul probe — e L12 ha toccato il massimo al sogno 9 di una
curva a dente di sega. La curva di ogni livello è registrata nella sua
cartella checkpoint come `dream_curve.json`.

## Requisiti

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

È l'unica dipendenza obbligatoria. **Per addestrare il curriculum italiano non
serve nessuna chiave API**: tutti i livelli 0-12 hanno il loro
`local_teacher.json`, quindi `./build.sh` insegna con il tutor offline.

### Tutor disponibili

| `--tutor-model` | Chi valuta le risposte | Richiede |
|---|---|---|
| `local` | regole deterministiche | niente |
| `hybrid` | prompt locali + un LLM locale piccolo | [llama.cpp](https://github.com/ggml-org/llama.cpp) o [ollama](https://ollama.com) attivo |
| `claude-haiku-4-5`, `claude-sonnet-4-6` | API Claude | `pip install anthropic` + chiave API |
| `auto` *(default)* | `hybrid` → `local` se il livello ha `local_teacher.json`, altrimenti Claude | — |

`build.sh` usa `local` per L0-L1 e `hybrid` da L2 in su quando un LLM locale
risponde, quindi un run italiano completo non costa nulla. Il valutatore gira su
**llama.cpp o ollama**, quello che è attivo — se ci sono entrambi vince
`llama-server` di llama.cpp (un solo modello già residente, nessuna latenza di
caricamento) — e può stare anche su un'altra macchina:

```bash
LLAMA_SERVER_BASE=http://gpu-box:8080 ./build.sh 4     # llama.cpp
OLLAMA_BASE=http://gpu-box:11434 PHYSISML_LLM_MODEL=qwen3:8b ./build.sh 4
```

Con ollama la mappa livello → modello è un requisito: un modello non installato
disabilita il valutatore LLM invece di sostituirlo con un altro. Con llama.cpp il
server ospita un modello solo, e quello è quello che si usa.

Il tutor Claude resta l'insegnante migliore ai livelli alti, ed è l'unico
disponibile per il curriculum inglese (`training_files/en/` non ha ancora un
`local_teacher.json`). È opzionale e si abilita così:

```bash
pip install anthropic python-dotenv
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

> I pesi addestrati **non** sono nel repository (`models/`, `*.pt` sono in `.gitignore`).
> Costruisci un modello da zero con `./build.sh`.

---

## Avvio rapido

```bash
# Addestra i livelli 0→1 in automatico (fase testuale + insegnamento, senza chiave API)
./build.sh 1

# Parla col modello
python3 dynamic_model/run.py                       # interattivo
python3 dynamic_model/run.py "testo" 2>/dev/null   # risposta singola
```

### Sequenza manuale per un livello

```bash
# Phase 0: training testuale su training_files/it/0/
python3 dynamic_model/train_curriculum.py --phase 0 --level 0 --epochs-0 10 --lang it

# Phase 1: insegnamento col tutor (ripetibile)
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
| `MIN_DREAMS=6 ./build.sh N` | Pavimento di sogni per livello (default 6, `0` disattiva). Sopra il pavimento il conteggio è misurato: si sogna finché il probe congelato guadagna (`MAX_DREAMS`, `DREAM_EPSILON`, `DREAM_PATIENCE`) |
| `./teach.sh [turni\|auto] [local\|hybrid\|haiku\|…] [lingua] [livello]` | Sessione di insegnamento |
| `./set_model.sh <checkpoint>` | Imposta il modello attivo (`models/active.pt`) |
| `./reset.sh [--dry-run]` | Backup + reset del modello |
| `python3 dynamic_model/train_curriculum.py` | Training testuale e/o insegnamento (vedi `--help`) |
| `python3 dynamic_model/test_model.py --level N` | Statistiche di qualità del modello corrente |
| `python3 scripts/measure_repetition.py --ckpt-base models/checkpoints/it --levels 0-12` | Exact match **e** tasso di ripetizione, in greedy |
| `python3 scripts/ retention_matrix.py --levels 0-12` | Matrice di ritenzione: ogni checkpoint su ogni livello |
| `python3 dynamic_model/run.py` | Sessione interattiva |
| `python3 scripts/download_wikipedia.py --level N` | Scarica articoli Wikipedia per il training |
| `python3 scripts/generate_qa_corpus.py --levels 0 1 2` | Genera corpus dialogico dalle coppie QA |
| `python3 scripts/generate_qa_corpus.py --check --levels 0 1 2` | Verifica che ogni `qa_corpus.txt` corrisponda al suo `qa_pairs.jsonl` (esce 1 se stantio) |
| `python3 scripts/export_gguf.py` | Esporta un checkpoint in GGUF, poi `ollama create physisml -f Modelfile` |
| `python3 scripts/export_hf.py --out hf_upload` | Prepara la cartella per Hugging Face (safetensors + model card + codice di inferenza) |
| `./scripts/build_status.sh` | Dove è arrivato un build: livello, sessione, qualità, cosa sta girando |
| `python3 scripts/curiosity_rate.py --gate off` | Ammette di non sapere sui nomi ignoti e non su quelli noti |

Flag principali di `train_curriculum.py`: `--phase 0|1`, `--level N`, `--lang it|en`,
`--epochs-0 N`, `--interactions N|auto`, `--age 0-7+` (età virtuale → stile di
insegnamento), `--tutor-model auto|local|hybrid|haiku|sonnet`.

**Insegnanti disponibili**: `local_teacher.py` (deterministico, gratuito,
offline), `hybrid_teacher.py` (prompt locali + valutazione con LLM locale via
llama.cpp o ollama, gratuito e con GPU), tutor Claude via API (opzionale — vedi [Tutor disponibili](#tutor-disponibili)).

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
| it | 11 | Classi e relazione is-a (il gatto è un animale) |
| it | 12 | Chiedere quando il nome è ignoto |
| en | 0–1 | Suoni e grammatica base (scritto a mano) |
| en | 2 | Shakespeare |
| en | 3 | Alice in Wonderland + Oliver Twist |
| en | 4 | Jane Eyre + Pride & Prejudice |
| en | 5 | Moby Dick |

**Riproducibilità del corpus.** `qa_pairs.jsonl` è la fonte (coppie
prompt→risposta estratte dalle sessioni) ed è tracciato; `qa_corpus.txt` è
derivato — le coppie ripetute 20 volte in ordine mescolato — ed è tracciato
anch'esso perché un clone deve poter addestrare senza passaggi intermedi. La
generazione usa un RNG dedicato con seed fisso, quindi lo stesso
`qa_pairs.jsonl` produce sempre lo stesso corpus su qualunque macchina.
`--check` verifica che i due siano allineati.

Ogni livello ha un `local_teacher.json` (pool chiuso di obiettivi, deterministico),
un testo curato coerente col livello e un `qa_corpus.txt` generato dalle sessioni
(coppie prompt→risposta). I testi narrativi lunghi restano nel repository, in
`training_files/it/N/_reference/`, ma **non entrano in nessuna fase**: né
training testuale, né insegnamento, né replay del sogno, né costruzione del
tokenizer. Prosa per adulti cancella le associazioni prompt→risposta appena
costruite, e l'italiano arcaico non è la lingua del curriculum. Il meccanismo
è la posizione: tutti i caricatori usano un glob `*.txt` non ricorsivo sulla
cartella del livello. Dettagli in
[training_files/it/_reference_README.md](training_files/it/_reference_README.md).

---

## Architettura

- **Modello**: TorchGPT — transformer decoder-only GPT-2 style, Pre-LayerNorm,
  testa LM con weight tying. Configurazione in uso: `d_model=512`, 6 layer,
  8 head, `d_ff=2048`, contesto 128 token, **23.6M parametri**. (Lo state dict
  serializza `lm_head.weight` e `tok_emb.weight` separatamente ma sono lo stesso
  tensore: contare le voci del file dà 28.2M per un modello da 23.6M.)
- **Tokenizer**: BPE byte-level, 2.590 slot attivi su 9.000 allocati. La fase di
  sogno può farlo crescere; nel build 0-12 non è successo — gli stessi 2.590
  token sono bastati per ogni livello, che è lo scopo del retrain senza
  punteggiatura attaccata.
- **Sistema affettivo**: stato innato (`confidence`, `pleasure`, `pain`, `fear`)
  che modula la generazione e traccia lo stato di apprendimento.
- **Anti-forgetting**: rehearsal *interleaved* sulle coppie gold durante
  l'insegnamento (4 coppie ogni 5 turni), più il replay del corpus nella fase di
  sogno. Il rehearsal è pesato sul livello corrente; il lavoro fra livelli lo fa
  il sogno, il cui N1 rigioca il `qa_corpus` di *tutti* i livelli. Sei sogni per
  livello sono ciò che ha portato il checkpoint finale dal 20% su tutti i
  livelli all'89% — vedi [Risultati](#risultati).
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

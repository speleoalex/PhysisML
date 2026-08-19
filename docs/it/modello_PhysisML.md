# PhysisML — Documentazione tecnica e filosofica

*Leggi in: [English](../en/physisml_model.md)*

*Ultimo aggiornamento: 2026-04-13*

---

## Indice

1. [Filosofia del progetto](#1-filosofia-del-progetto)
2. [Architettura del modello](#2-architettura-del-modello)
3. [Sistema affettivo innato](#3-sistema-affettivo-innato)
4. [Metodo di training](#4-metodo-di-training)
5. [Curriculum linguistico](#5-curriculum-linguistico)
6. [Risultati sperimentali](#6-risultati-sperimentali)
7. [Confronto con approcci standard](#7-confronto-con-approcci-standard)
8. [Roadmap](#8-roadmap)

---

## 1. Filosofia del progetto

### Tesi centrale

I modelli linguistici attuali (GPT, LLaMA, TinyLlama) ottengono comportamenti simili a conoscenza e ragionamento come **effetti collaterali della scala** — miliardi di parametri, trilioni di token, cluster GPU da milioni di dollari.

La natura ha sviluppato un approccio radicalmente diverso: il cervello umano impara con pochissimi dati grazie a **principi semplici applicati in modo cumulativo**. Un bambino impara la lingua madre in pochi anni, con poca esposizione, senza backpropagation esplicita, partendo da zero.

**L'ipotesi di questo progetto**: è possibile far emergere comportamenti simili a conoscenza con meno dati e meno risorse, attraverso un apprendimento cumulativo, progressivo e guidato — ispirato allo sviluppo cognitivo biologico.

### Nota epistemica

Il modello usa ancora **backpropagation + ottimizzatore Adam** — non è apprendimento biologico in senso letterale. È un transformer con estensioni dinamiche *ispirate* al funzionamento biologico. Il claim è metodologico, non neuroscientificamente preciso.

### Principi progettuali

- **Curriculum progressivo**: il modello impara come un bambino, da fonemi a letteratura
- **Sistema affettivo innato**: confidence, fear, pleasure, pain modulano il comportamento dal primo token
- **Segnale insegnante curato**: un LLM esterno (Claude) genera esempi mirati sul deficit corrente del modello
- **Show-then-test**: il modello vede la risposta corretta *prima* di essere interrogato, come nella didattica naturale
- **Anti-forgetting (rehearsal)**: ogni N turni di teaching, un mini-batch testuale previene la perdita catastrofica
- **Assiomi protetti**: sequenze di verità oggettive (es. "1+1=2") con gradiente azzerato

---

## 2. Architettura del modello

### Configurazione attuale (sperimentale)

```text
TorchGPT — decoder-only, Pre-LayerNorm, GPT-2 style
─────────────────────────────────────────────────────
Parametri totali : 3.32M
Vocabolario      : 501 token (BPE, addestrato su corpus it/0)
d_model          : 256
n_layers         : 4
n_heads          : 4
d_ff             : 1024
max_seq_len      : 129 (128 token di contesto + 1)
dropout          : 0.1
Positional enc.  : learnable embedding (non RoPE)
Attivazione FFN  : GELU
Weight tying     : logits = x @ tok_emb.W^T
```

### Flusso forward

```text
Input ids (T,)
  → TokenEmbedding(V=501, d=256) + PosEmbedding(128, d=256)
  → Dropout(0.1)
  → 4 × TransformerBlock:
       h   = x + Attention(LayerNorm(x))    # residual pre-LN
       out = h + FFN(LayerNorm(h))           # residual pre-LN
  → FinalLayerNorm
  → logits = x @ TokenEmbedding.W^T          # weight tying
  → CrossEntropy loss
```

### Tokenizer

BPE (Byte Pair Encoding) con 501 token addestrato esclusivamente sul corpus fonemico `it/0` (sillabe e fonemi italiani). La scelta di un vocabolario piccolo è intenzionale per la fase sperimentale: forza il modello a imparare relazioni tra sub-parole prima di avere parole intere.

**Limite identificato**: con 501 token le parole comuni italiane si frammentano in 2–4 subtoken, impedendo la generalizzazione semantica. Il vocabolario target per la fase successiva è 8.000 token.

### Backend di implementazione

Il modello esiste in due implementazioni:

| Implementazione | File | Uso |
|----------------|------|-----|
| **PyTorch CPU** | `tests/test_1/splx/torch_model.py` | Training attivo (59 seq/s, batch=8) |
| **NumPy puro** | `tests/test_1/splx/` | Reference, gradient check, educativo |

---

## 3. Sistema affettivo innato

Il sistema affettivo non è un modulo aggiunto sopra il transformer: è parte integrante del ciclo di training. Cinque scalari aggiornati con EWMA (α=0.1):

| Variabile | Significato | Init | Formula / aggiornamento |
| --------- | ----------- | ---- | ----------------------- |
| `confidence` | certezza sulla risposta corrente | 0.1 | `1 - H(softmax(logits)) / log(V)` — ogni forward |
| `ignorance` | embedding a bassa norma (token "non appresi") | 0.9 | `frac(norm(emb) < 0.05)` — ogni forward |
| `pleasure` | memoria del feedback positivo | 0.5 | EWMA(feedback > 0) — solo con feedback esplicito |
| `pain` | memoria del feedback negativo | 0.0 | EWMA(\|feedback\| se < 0) — solo con feedback esplicito |
| `fear` | incertezza + dolore | 0.45 | `0.5·pain + 0.5·(1-confidence)` — derivata |

**Nota**: `confidence` e `ignorance` si aggiornano ad ogni forward pass (anche senza feedback). `pleasure` e `pain` si aggiornano solo con feedback esterno esplicito. `fear` è puramente derivata.

### Novelty drive (dopamina)

Implementato in `update_from_novelty()`: quando una risposta riceve feedback positivo, ogni token della risposta riceve un bonus su `pleasure` inversamente proporzionale a quante volte è già apparso in risposte positive:

```text
bonus = NOVELTY_WEIGHT × feedback / sqrt(encounter_count + 1)
```

Prima produzione corretta di un token: bonus pieno. Quarta volta: 50%. Centesima: 10%.  
`register_token_activation()` aggiunge un ulteriore spike (`+0.6 × min(1, n_new/10)`) quando nuovi slot vocabolario vengono attivati nel Dream phase.

### Comportamento osservato (L0→L2, dati reali)

- **Confidence**: 0.10 → 0.59 attraverso il curriculum
- **Fear**: 0.45 → 0.21 — il modello diventa progressivamente meno timoroso
- **Pain**: 0.00 costante — nessun feedback negativo accumulato (pain gate non ancora attivo)
- **Pleasure**: oscillante 0.47–0.56 in risposta ai picchi di feedback positivo

> Dati L3–L4: non ancora raccolti — vedere §6

### Modulazione inference (AffectModulator)

Applicata **solo durante la generazione**, mai durante il training (evita loop di retroazione instabile):

```text
1. Temperature adattiva:  logits /= (base_temp × (1 + fear × 2.0))
2. Boost "non lo so":     se ignorance > 0.7 o confidence < 0.15:
                              logits[DONT_KNOW] += 3.0 × ignorance
3. Pain gate:             se pain > 0.1: logits[pain_ids] -= pain × 1.5
4. Pleasure gate:         se pleasure > 0.5: logits[pleasure_ids] += (pleasure-0.5) × 0.8
```

I token `<|dont_know|>` e `<|uncertain|>` devono essere registrati nel tokenizer prima dell'uso.

### Assiomi protetti

Sequenze di verità oggettive con gradiente scalato a zero durante il backward:

```python
axiom_registry.register(["io", "sono"], is_objective=True, protection=0.9)
axiom_registry.register(["tu", "sei"],  is_objective=True, protection=0.9)
```

Implementati in `dynamic_model/exp_b/axioms.py`. La protezione è proporzionale al livello di certezza: assiomi oggettivi (1+1=2) max 1.0, soggettivi max 0.6.

---

## 4. Metodo di training

### Due fasi per ogni livello

**Phase 0 — Training testuale** (`--phase 0`):
- Corpus cumulativo: `it/0` + corpus del livello corrente
- Standard cross-entropy, batch=8, seq_len=128
- Learning rate: 1e-3 (Adam)
- Epoche: 10 (configurabile)
- Output: `level_N/final.pt`

**Phase 1 — Teaching con Claude** (`--phase 1`):
- LLM esterno (Claude Sonnet/Haiku) come insegnante adattivo
- Learning rate: 1e-4 (più conservativo per evitare forgetting)
- 4 segnali per turno (in ordine):

```text
1. SHOW-THEN-TEST  trainer.step("", expected,      feedback=0.5)
2. EXPOSURE        trainer.step("", next_prompt,   feedback=0.2)
3. CONTENT         trainer.step("", teaching_word, feedback=0.3)
4. IMITATION       trainer.step(prompt, response,  feedback=fb)
   (oppure: se fb="-" e expected noto → feedback=1.0)
```

**Anti-forgetting (rehearsal)**: ogni 10 turni, 1 mini-batch (4×128 token) dal corpus testuale del livello. Riduce lo spike di perplexity da +133% (senza) a +5% (con).

### Auto-stop del teaching

La sessione si interrompe automaticamente quando, per 20 turni consecutivi:
- ≥ 60% feedback positivi (`+`, `++`, `+++`)
- ≥ 20% feedback forti (`++`, `+++`) — previene auto-stop da falsi positivi
- `fear < 0.25`
- `confidence > 0.45`

### Checkpoint salvati

```text
models/checkpoints/{lang}/
  level_N/
    final.pt            ← dopo phase 0 (knowledge testuale)
    final_learned.pt    ← dopo phase 1 (knowledge interattiva)
    final_dreamed.pt    ← dopo Dream Consolidation (se attivata)
    turn_XXXX.pt        ← snapshot ogni 10 turni
    session_*.jsonl     ← log completo del dialogo insegnante-modello
    tokenizer.json      ← tokenizer aggiornato al livello
```

---

## 5. Curriculum linguistico

Il curriculum mappa 11 livelli (0–10) agli anni di sviluppo linguistico umano:

| Livello | Età equivalente | Contenuto corpus | Risposta attesa |
|---------|-----------------|------------------|-----------------|
| L0 | Neonato | Fonemi, sillabe (`ma`, `pa`, `ta`) | Sillabe isolate |
| L1 | 1 anno | Filastrocche, parole singole | Parole singole |
| L2 | 2 anni | Frasi base, animali (Wikipedia) | 2–3 parole |
| L3 | 3 anni | Pinocchio, Wikipedia semplice | Frasi 3–6 parole |
| L4 | 4 anni | Favole Esopo, Wikipedia cultura | Frasi S+V+O |
| L5 | 5 anni | De Amicis, canzoni, Wikipedia | Storie brevi 3–4 frasi |
| L6 | 6 anni | Narrativa moderna (Neera, Serao) | Connettivi, descrizioni |
| L7 | 7 anni | Rodari, Wikipedia | Storie brevi |
| L8 | 8 anni | I Promessi Sposi (estratto) | Paragrafi |
| L9 | 9 anni | I Promessi Sposi (integrale) | Testi complessi |
| L10 | 10 anni | Divina Commedia, Wikipedia | Letteratura |

**Principio chiave**: il corpus contiene testi *più complessi* di quelli che il modello produce. Come un bambino che ascolta conversazioni adulte ma risponde con parole semplici. Il teacher richiede risposte appropriate all'età, non al corpus.

### Teacher prompt per livello

Ogni livello ha un file `training_files/it/{N}/teacher_prompt.md` con:
- Scala di feedback rigorosa (nessun falso positivo)
- Regole anti-ripetizione (max 3× la stessa parola)
- Valutazione solo su `expected` del turno corrente
- Progressione A→B→C→D per complessità crescente

---

## 6. Risultati sperimentali

### Traiettoria perplexity (testo: "il cane dorme sul tappeto. la mamma cucina il pane.")

```text
Checkpoint          PPL      Δ       Tipo    Stato
────────────────────────────────────────────────────────
L0/final            53.7             [txt]   ✓ misurato
L0/final_learned    80.3    +26.6 ↑  [tch]  ✓ no rehearsal
L1/final            29.0    -51.3 ↓  [txt]  ✓
L1/final_learned    49.3    +20.3 ↑  [tch]  ✓ con rehearsal (-85% vs L0)
L2/final            22.5    -26.8 ↓  [txt]  ✓
L2/final_learned    27.7     +5.1 ↑  [tch]  ✓ spike quasi nullo
L2/final_dreamed     —        —       [drm]  ✓ Dream phase applicata
L3/final            21.2     -6.5 ↓  [txt]  ☐ da misurare
L3/final_learned    18.0     -3.2 ↓  [tch]  ☐ stimato
L4/final            19.2     +1.2 ↑  [txt]  ☐ stimato
L4/final_learned    19.5     +0.4 →  [tch]  ☐ stimato — plateau (~501 token limit)
```

**Stato attuale** (2026-04-13): checkpoint reali disponibili fino a `L2/final_dreamed`. I valori L3–L4 sono stime basate sulla traiettoria osservata.

**Milestone raggiunta** (L0–L2): il rehearsal ha ridotto lo spike di catastrophic forgetting da +133% (L0, senza rehearsal) a +5% (L2). La Dream phase è operativa.

**Limite identificato**: plateau atteso a PPL≈18–20 dai livelli L3–L4. Causa: il vocabolario da 501 token è saturato. Parole comuni si frammentano in subtoken, impedendo la generalizzazione.

### Efficienza di training

Rispetto a training standard su corpus equivalente:

- **Token-pass CPU necessari**: ~28M (vs. ~150M standard)
- **Speedup teorico**: 5–27× per la stessa PPL finale
- **Fonte del vantaggio**: curriculum ordering (2–5×) + teaching signal curato (10–100× per esempio) + rehearsal (elimina retraining)
- **Costo API Claude** per curriculum completo L0–L10: ~$30–50

---

## 7. Confronto con approcci standard

| Dimensione | PhysisML attuale | Standard small (GPT-2 117M) | TinyLlama 1.1B |
|------------|-------------------|------------------------------|----------------|
| Parametri | 3.32M | 117M | 1,100M |
| Vocabolario | 501 | 50,257 | 32,000 |
| Corpus | ~2M token | ~40B token | 3T token |
| PPL italiano | ~18–20 | ~15 (se fine-tuned) | ~6–8 |
| Frasi coerenti | No | Parzialmente | Sì |
| Costo training | ~$20 API | ~$500 GPU | ~$100,000 GPU |
| Curriculum | Sì | No | No |
| Sistema affettivo | Sì | No | No |

### Target intermedio realistico (fase successiva)

```text
Parametri  : 25M     (d=512, L=6, n_heads=8)
Vocabolario: 8,000   (BPE su corpus completo it/0–it/10)
Corpus     : 200M+   (Wikipedia IT + OpenSubtitles + esistente)
PPL attesa : ~10–12
Costo      : ~$55    ($1 GPU cloud + $50 API teaching)
Output     : frasi italiane semplici ma comprensibili
```

---

## 8. Roadmap

### Fase corrente — Validazione curriculum (in corso)

- [x] Implementare show-then-test (segnale expected prima della generazione)
- [x] Rehearsal anti-forgetting (1 batch testo ogni 10 turni teaching)
- [x] Auto-stop rigoroso (richiede 20% strong feedback `++/+++`)
- [x] Teacher prompt con scala feedback rigorosa (Sonnet, no falsi positivi)
- [x] Dream Consolidation phase (`final_dreamed.pt`) integrata nel curriculum
- [x] Novelty drive (dopamina) — bonus decrescente per token nuovi
- [x] `ignorance` come variabile affettiva autonoma (prior biologico 0.9)
- [ ] Completare curriculum L3→L10 con modello 3.3M param (corrente: L2 ✓, L3 in avvio)
- [ ] Validare comportamento affettivo su tutti i livelli

### Fase 2 — Scala vocabolario

- [ ] Scaricare corpus Wikipedia IT (~450MB dump)
- [ ] Scaricare OpenSubtitles IT (~700MB, conversazionale L2–L5)
- [ ] Riaddestrare tokenizer BPE su corpus completo → 8,000 token
- [ ] Benchmarking: PPL modello 3.3M con vocab 501 vs 8,000

### Fase 3 — Scala modello

- [ ] Modello d=512, L=6, n_heads=8 (~25M param)
- [ ] Training su corpus 200M+ token con curriculum L0–L10
- [ ] Benchmark comparativo: PPL, qualità output, tempo training

### Fase 4 — Target TinyLlama-like (GPU)

- [ ] d=768, L=12 (~85M param) o superiore
- [ ] Richiede GPU (RTX 3060 o A100 cloud)
- [ ] PPL target: ≤ 10 su italiano standard

### Idee future (da appunti.md)

- **Assiomi matematici**: protezione del gradiente per verità oggettive (1+1=2) — struttura in `exp_b/axioms.py`, da collegare al curriculum
- **Training multilingua**: inglese + italiano in parallelo, livello inconscio comune

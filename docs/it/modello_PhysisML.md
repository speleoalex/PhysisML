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
Parametri totali : 23.59M
Vocabolario      : 9.000 slot allocati, 8.002 attivi all'avvio
                   → 8.083 dopo il curriculum L0→L10
d_model          : 512
n_layers         : 6
n_heads          : 8
d_ff             : 2048
max_seq_len      : 129 (128 token di contesto + 1)
dropout          : 0.1
Positional enc.  : learnable embedding (non RoPE)
Attivazione FFN  : GELU
Weight tying     : logits = x @ tok_emb.W^T
Optimizer        : Adam, lr 1e-3 (testo) / 2e-5 (insegnamento),
                   weight_decay = 0
```

**Nota su `weight_decay`.** Deve restare a zero. Con `torch.optim.Adam` il decay
è accoppiato al gradiente, quindi un parametro poco sollecitato viene ridotto a
ogni passo indipendentemente dal gradiente. Sui centinaia di migliaia di passi a
campione singolo del curriculum questo spegne la rete: misurato sui checkpoint di
maggio, il gain di `ln_f` passava da 0.87 (L0) a 0.0079 (L10). Se serve
regolarizzazione, usare AdamW disaccoppiato escludendo LayerNorm ed embedding.

**Slot dormienti.** `vocab_size` è la capacità allocata, `active_vocab_size`
quanti token sono visibili. I token dormienti hanno logit `-inf` e gradiente
zero; vengono attivati durante il sogno con inizializzazione dai token genitori,
scalata al 30% della norma media delle righe già addestrate (una scala assoluta
darebbe alle righe nuove una norma *maggiore* di quelle addestrate, cioè un prior
alto senza semantica nel softmax weight-tied).

### Flusso forward

```text
Input ids (T,)
  → TokenEmbedding(V=9000, d=512) + PosEmbedding(128, d=512)
  → Dropout(0.1)
  → 4 × TransformerBlock:
       h   = x + Attention(LayerNorm(x))    # residual pre-LN
       out = h + FFN(LayerNorm(h))           # residual pre-LN
  → FinalLayerNorm
  → logits = x @ TokenEmbedding.W^T          # weight tying
  → CrossEntropy loss
```

### Tokenizer

BPE (Byte Pair Encoding) da 8.000 token, più slot dormienti fino a 9.000. I token dormienti hanno logit `-inf` e gradiente zero; la fase di sogno li attiva man mano che nuovi pattern si consolidano, inizializzandoli dai token genitori al 30% della norma media delle righe già addestrate.

Nel curriculum L0→L10 il vocabolario è cresciuto da 8.002 a 8.083 token. La crescita è deliberatamente prudente: i merge si cercano e si applicano **solo dentro i confini di parola** (come fa `encode`, altrimenti il token è irraggiungibile e occupa uno slot competendo nel softmax), si rifiutano ripetizioni degeneri e frasi multi-parola, e si proteggono gli obiettivi in corso di addestramento — tokenizzare ciò che si sta insegnando orfana il percorso multi-token già appreso.

### Backend di implementazione

Il modello esiste in due implementazioni:

| Implementazione | File | Uso |
|----------------|------|-----|
| **PyTorch CPU** | `tests/test_1/physisml/torch_model.py` | Training attivo (59 seq/s, batch=8) |
| **NumPy puro** | `tests/test_1/physisml/` | Reference, gradient check, educativo |

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

La protezione agisce sulle righe dell'embedding, quindi le parole dell'assioma devono essere della lingua in addestramento: `PHONETIC_AXIOMS` e `GRAMMAR_AXIOMS` in `train_curriculum.py` tengono una lista per lingua (`mamma/papà/sì/no` e il paradigma della copula per l'italiano, `mama/papa/yes/no` e `I am / you are / he is / it is` per l'inglese). Un assioma nella lingua sbagliata non è inerte: sul vocabolario inglese `mamma` si codifica in `m|am|ma` e congela tre sottoparole arbitrarie. `add_axiom` stampa i pezzi accanto agli id, così un assioma spezzato si vede, ed elimina i token di solo spazio — lo spazio è un token a sé e vale circa un terzo di entrambi i corpus, che una protezione a 0.9 congelava come effetto collaterale di un'affermazione su `io sono`.

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

**Anti-forgetting**, due meccanismi distinti:

- **Rehearsal testuale**: ogni 10 turni, 1 mini-batch (4×128 token) dal corpus
  del livello. Riduce lo spike di perplexity da +133% (senza) a +5% (con).
- **Rehearsal interleaved sulle coppie gold**: ogni 5 turni, 4 coppie
  prompt→risposta già insegnate, escludendo l'obiettivo corrente. Il teacher
  drilla a blocchi (N successi consecutivi, poi passa oltre e non torna), e il
  solo rehearsal testuale non ripassava nulla di ciò che era stato appreso:
  misurato a L1, l'accuratezza su un obiettivo ri-chiesto entro 5 turni era 47%
  ma dopo 20 turni scendeva all'8%.

**Didattica test-then-show**: il modello risponde *prima* che il segnale con la
risposta corretta venga applicato. Nell'ordine inverso veniva interrogato sulla
domanda di cui aveva appena ricevuto la risposta, quindi il voto del teacher — e
con esso il cancello di qualità del build — misurava il richiamo dopo
suggerimento e non la conoscenza ritenuta: divario misurato a L3, 31% in
sessione contro 8% offline sugli stessi obiettivi.

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

| Livello | Età equiv. | Struttura insegnata | Esempio di obiettivo | Target |
|---------|-----------|---------------------|----------------------|--------|
| L0 | Neonato | Sillabe isolate e raddoppiate | `di ma` → `ma!` | 21 |
| L1 | 1 anno | Articolo + sostantivo, famiglia | `di: il cane` → `il cane!` | 24 |
| L2 | 2 anni | Frasi soggetto + verbo | `di: il cane dorme` → `il cane dorme!` | 18 |
| L3 | 3 anni | Domande, identità, numeri e colori | `cosa fa il cane?` → `il cane dorme.` | 28 |
| L4 | 4 anni | Soggetto + verbo + oggetto, sequenze | `cosa mangia il cane?` → `il cane mangia il pane.` | 25 |
| L5 | 5 anni | Connettivi e / ma / perché, aggettivi | `perché il cane mangia?` → `il cane mangia perché ha fame.` | 20 |
| L6 | 6 anni | Passato prossimo, cause | `cosa ha mangiato il cane?` → `il cane ha mangiato il pane.` | 21 |
| L7 | 7 anni | Futuro, contrasto fra i tempi, dialogo | `cosa mangerà il cane domani?` → `domani il cane mangerà il pane.` | 19 |
| L8 | 8 anni | Comparativi, preferenze motivate | `chi è più grande, il cane o il gatto?` → `il cane è più grande del gatto.` | 19 |
| L9 | 9 anni | Tesi + motivo + conclusione, sinonimi | `il pane è buono?` → `secondo me il pane è buono perché è caldo.` | 16 |
| L10 | 10 anni | Commento motivato, confronto | `meglio il pane o la porta?` → `meglio il pane, perché si mangia.` | 16 |

Ogni livello ha un `local_teacher.json` con un **pool chiuso** di obiettivi
suddivisi in 3–5 step (A→E) e ripetuti finché non sono consolidati. Il numero di
obiettivi distinti conta: pochi obiettivi ripetuti insegnano il *template* ma non
la discriminazione, e il modello collassa su una risposta per famiglia di prompt.

**Il corpus non è il target.** I testi contengono materiale *più complesso* di
quello che il modello produce — come un bambino che ascolta conversazioni adulte
e risponde con parole semplici. Ma la prosa per adulti va tenuta fuori
dall'addestramento dal livello 3 in su: a L3 tre epoche su 20MB di narrativa
cancellavano le associazioni prompt→risposta costruite a L2. Il filtro esclude i
file oltre i 100KB; i testi enciclopedici sotto quella soglia sono stati
archiviati a mano, perché la dimensione è solo un proxy imperfetto della
complessità.

### Teacher prompt per livello

Ogni livello ha un file `training_files/it/{N}/teacher_prompt.md` con:
- Scala di feedback rigorosa (nessun falso positivo)
- Regole anti-ripetizione (max 3× la stessa parola)
- Valutazione solo su `expected` del turno corrente
- Progressione A→B→C→D per complessità crescente

---

### Una lingua è una cartella

Il curriculum qui sopra è italiano, e niente di quel curriculum è scritto in
Python. Una seconda lingua è `training_files/<lang>/` con la stessa forma — una
cartella numerata per livello, `local_teacher.json`, `qa_pairs.jsonl`, un testo
di livello — più `training_files/<lang>/language.json` per le poche cose che
sono parole e non struttura: gli assiomi, le parole funzione, le grafie del sì e
del no, il prompt di ripiego del tutor, il repo sull'Hub.
`dynamic_model/language.py` lo legge; tutto il resto (vocabolario, probe,
scheda, cartella di export) segue una convenzione di nome e non va dichiarato.

Il curriculum inglese copre oggi i livelli 0–10:

| Livello | Struttura insegnata | Esempio di obiettivo |
|---------|---------------------|----------------------|
| L0 | Sillabe isolate e raddoppiate | `say ma` → `ma!` |
| L1 | Articolo + sostantivo, prime parole, il proprio nome | `say: the cat` → `the cat!` |
| L2 | Articolo + sostantivo + verbo, aggettivo, S+V+O | `say: the cat sleeps` → `the cat sleeps!` |
| L3 | Soggetto + verbo, `what does X do?`, numeri | `what does the cat do?` → `the cat sleeps.` |
| L4 | Domande `who` e `where`, prima/poi | `where does the cat sleep?` → `the cat sleeps in the house.` |
| L5 | Connettivi and / but / because, descrizioni | `why does the cat eat?` → `the cat eats because it is hungry.` |
| L6 | Il passato, le cause, due frasi legate | `what did the boy eat?` → `the boy ate the bread.` |
| L7 | Il futuro, e i tre tempi affiancati | `what will the boy eat tomorrow?` → `tomorrow the boy will eat the bread.` |
| L8 | Comparativi, preferenze e le loro ragioni | `who is bigger, the dog or the cat?` → `the dog is bigger than the cat.` |
| L9 | Una tesi con la sua ragione e una conclusione | `is the cat small?` → `I think the cat is small because it is fast.` |
| L10 | Un commento breve, un giudizio motivato | `comment on the cat` → `the cat is fast, and this is nice.` |

Perché un manifesto e non una tabella nel sorgente: un `dict` con chiave la
lingua dentro un file `.py` è l'elenco delle lingue che quel modulo conosce. È
completo il giorno in cui lo si scrive e sbagliato in silenzio il giorno in cui
qualcuno aggiunge una lingua — e l'errore non fa rumore. Il primo build inglese
ha girato sei ore proteggendo l'assioma italiano `mamma`, che il vocabolario
inglese codifica come `m|am|ma`: tre sottoparole arbitrarie congelate a
protezione 0.7, e nessun messaggio da nessuna parte.
`tests/test_language_manifest.py` ora attraversa i sorgenti a caccia di tabelle
del genere, e verifica che ogni assioma sia un token intero del vocabolario
della propria lingua e compaia davvero nel proprio corpus.

---

## 6. Risultati sperimentali

### Metrica

**Exact match** su tutti gli obiettivi del curriculum di ogni livello:
la risposta del modello, normalizzata, deve coincidere con quella attesa dal
teacher. Decoding **greedy** (`top_k=1`), quindi il numero è riproducibile: a
temperatura 0.8 lo stesso checkpoint aveva dato 92% e 75% in due esecuzioni
consecutive. Le domande sono poste nella forma **esatta** usata in
addestramento, cioè applicando il `prompt_template` dello step (`di: {prompt}`
per la maggior parte): interrogare col target nudo misura una domanda che il
modello non ha mai visto — `il cane` produce `mangia il pane!` mentre
`di: il cane` produce `il cane!`.

Comando: `python3 dynamic_model/test_model.py --level N --samples 0`

### Risultati per livello

Checkpoint post-sogno, exact match su tutti gli obiettivi:

| Livello | Exact match | Obiettivi | Sessioni | Note |
|---------|------------|-----------|----------|------|
| L0 | 100% | 21/21 | 3 |  |
| L1 | 96% | 23/24 | 3 |  |
| L2 | 100% | 18/18 | 3 |  |
| L3 | 82% | 23/28 | 2 | numeri e colori condividono il prefisso del prompt |
| L4 | 100% | 25/25 | 10 | livello dove lo stallo iniziava (0% a maggio) |
| L5 | 95% | 19/20 | 2 |  |
| L6 | 100% | 21/21 | 1 |  |
| L7 | 100% | 19/19 | 1 |  |
| L8 | 100% | 19/19 | 2 |  |
| L9 | 88% | 14/16 | 1 | due errori: ripetizione dell'inizio |
| L10 | 94% | 15/16 | 1 | un errore: ripetizione dell'inizio |

**Media sugli 11 livelli: 96%.**

Confronto col build di maggio, prima delle correzioni: L0 4.4%, L1 1.8%,
L2 12.8%, L3 1.0%, **L4 e oltre 0.0%**.

Il numero di sessioni necessarie è crollato con le correzioni finali: L4 ne
richiese dieci prima che il sanity check e il riordino del sogno fossero
sistemati, mentre L6, L7, L9 e L10 superano il cancello alla **prima** sessione.
Meno sessioni non significa meno apprendimento: significa che il segnale non
viene più sprecato.

### Effetto della fase di sogno

Delta exact match fra checkpoint pre-sogno (`final_learned.pt`) e post-sogno
(`final_dreamed.pt`), per livello:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|----|----|----|----|----|----|----|----|----|----|-----|
| +19 | +13 | +17 | +7 | +12 | −5 | +29 | **+53** | 0 | **+38** | **+38** |

Il sogno ora aiuta a dieci livelli su undici. Prima del riequilibrio di N3
danneggiava sistematicamente i livelli alti (L7 −26, L10 −12, L9 −6): il memory
bank copre tutti i livelli e la quota dal livello corrente scendeva al 5% a L7 e
al 7% a L10, contro il 41% a L4. N3 e REM sono le ultime fasi che il sogno
addestra, quindi quella proporzione decide cosa il modello conserva. Ora N3
mantiene tutte le memorie del livello corrente e campiona le vecchie fino a pari
numero, e gli stessi livelli guadagnano +53, +38 e +38 punti.

### Ritenzione: la diagonale non è la capacità

I numeri sopra sono la **diagonale** di una matrice: ogni livello valutato sul
proprio checkpoint. Rispondono a "il livello si è addestrato?", non a "il
modello sa ancora farlo?". Sono due domande con risposte molto diverse.

`scripts/retention_matrix.py` valuta ogni checkpoint su ogni livello. Sul build
di riferimento:

```
        target →
ckpt ↓    L0    L1    L2    L3    L4    L5    L6    L7    L8    L9   L10
L0      100%
L1       10%   96%
L2       14%   50%  100%
L3       48%   54%   78%   82%
L4      100%   83%  100%   71%  100%          ← l'unica riga che ritiene
L5      100%   42%   22%   18%   16%   95%
L10     100%   21%    0%   11%    4%    0%    0%    0%    0%    0%   94%
```

Diagonale 96%, **riga finale 20%**. Il checkpoint L10 sa fare L10 e L0, il
resto è sepolto. La diagonale è sempre 82–100%: il problema non è
l'apprendimento, è la ritenzione.

L'unica riga che ritiene è L4, e L4 è l'unico livello che ha richiesto dieci
sessioni di insegnamento. Ogni sessione finisce con un sogno, e N1 nel sogno
rigioca il `qa_corpus` di *tutti* i livelli. Gli altri livelli ne hanno avute da
una a tre. Non era una proprietà dell'architettura: era un caso.

#### Intervento 1 — cicli di consolidamento (`MIN_DREAMS`)

Sogni aggiuntivi sul checkpoint L10 finito, senza alcun insegnamento nuovo
(`./scripts/experiment_extra_dreams.sh --confirm`):

| sogni | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|-------|---|---|---|---|---|----|----|
| exact su tutti i livelli | 20% | 27% | 36% | 43% | 44% | 48% | **48%** |
| risposte con ripetizione | 37% | 25% | 19% | 17% | 18% | 15% | **18%** |

+3.6 punti per sogno da 1 a 6, +1.0 da 7 a 12 — quest'ultima pendenza è sotto
il rumore misurato fra due run identici (2.2 punti), quindi satura fra il sesto
e il decimo. `build.sh` rabbocca ora ogni livello a `MIN_DREAMS=6`
indipendentemente da quando il gate passa.

Ma il tetto è **48%, non 96%**: il consolidamento recupera metà del divario.

#### Intervento 2 — ambito del rehearsal: tentativo non riuscito

La seconda metà non si recupera sognando. Il sogno rigioca il corpus (N1) e il
memory bank (N3) di tutti i livelli, ma il **rehearsal interleaved** — il canale
che ha davvero costruito le associazioni prompt→risposta — carica solo
`qa_pairs.jsonl` del livello corrente. Estenderlo sembrava il candidato
naturale.

`--rehearsal-scope {level,all,balanced}` con `balanced` = unione dei livelli ma
metà di ogni replay riservata al corrente. Tre bracci a un solo flag di
distanza, L0→L3, budget di sessioni e sogni tenuto costante
(`./scripts/experiment_rehearsal.sh --confirm`).

**Il risultato non regge fra i seed.** Riga finale, `balanced` meno `level`:

| target | seed 1 | seed 2 |
|--------|-------:|-------:|
| L0 | +10 | **−19** |
| L1 | +21 | **−4** |
| L2 | +11 | +17 |
| L3 | +7 | +4 |
| **aggregato** | **+12** | **−1** |

Al primo seed `balanced` vinceva in ogni cella; al secondo perde sui livelli
più vecchi, che sono proprio quelli che l'intervento doveva proteggere.
**L'effetto per cui era stato costruito non si riproduce**, e il default è
tornato a `level`.

Il primo seed era stato letto come conferma perché il +12 superava il "rumore
di 2.2 punti" — ma quel numero misura la riproducibilità dello **stesso** sogno
rieseguito, non la varianza fra seed, che su questa metrica è dell'ordine di
15 punti. Due grandezze diverse, e confonderle ha reso un risultato nullo
convincente per un giorno.

**Una cosa però si riproduce**, e non è quella cercata: `balanced` alza la
**diagonale** a L2 (94% → 100%) e L3 (93/96% → 100%) in entrambi i seed. È il
livello corrente che migliora, non i precedenti che vengono ritenuti. È
un'osservazione post-hoc: merita un esperimento suo, non serve a salvare
questo.

`all` resta il peggiore dei tre al primo seed, coerente con la diluizione: a L3
l'unione è 535 coppie contro 188 del livello stesso.

#### Intervento 3 — il controllo: replay contro EWC online (exp_i)

Il canale cross-livello del sogno è experience replay, e il replay è il
trucco più vecchio del continual learning. Il claim "il sogno ha sistemato la
ritenzione" è interessante solo se un metodo di *regolarizzazione* standard —
che porta con sé statistiche riassuntive invece del corpus — non sa fare lo
stesso. `exp_i` è quel controllo: stessa rete, stesso curriculum, stesso
harness, L0→L6, due semi, tre bracci a un flag di distanza
(`--anti-forgetting`):

- **`dream`** — il meccanismo del progetto: N1 rigioca il corpus di ogni
  livello, N3 rigioca il memory bank episodico cross-livello.
- **`ewc`** — EWC online (Schwarz et al. 2018): un Fisher diagonale running
  `F ← γ·F_prev + F_new` (γ=0.95) con l'ancora θ* rinnovata a ogni confine di
  livello, penalità `½·λ·Σ F·(θ−θ*)²` nelle fasi 1 e 2. Il Fisher è quello
  empirico, stimato sulle coppie gold post-harvest del livello con la stessa
  loss prompt-masked del training (`scripts/compute_fisher.py`, sidecar
  `level_N/fisher.pt`). LayerNorm ed embedding posizionali esclusi
  dall'ancora (la lezione del weight decay su `ln_f`), embedding legato
  contato una volta, righe di vocabolario dormienti libere. In questo braccio
  il sogno gira comunque, ma N1 è ristretto al livello corrente e N3 alle
  memorie del livello corrente: tutto ciò che consolida il livello *corrente*
  resta, cambia solo il canale cross-livello.
- **`none`** — gating identico con λ=0: il pavimento.

λ=1000 viene da uno sweep preliminare su L0→L2 (λ ∈ {100, 1000, 10000}:
media riga finale 33/37/41%, diagonale 78/76/73 — la ritenzione sale
monotona con λ ma non avvicina mai il replay, mentre l'apprendimento
corrente scende). Sei sogni per livello, fissi in ogni braccio, così il
budget di consolidamento non è un confound. Per riprodurre:
`MODE=sweep ./scripts/experiment_ewc.sh --confirm`, poi `MODE=main`; le
matrici per braccio finiscono in `models/exp_i/`.

Risultato — media tra i semi (seme 1 / seme 2), rumore run-to-run 2.2 punti:

| braccio | riga finale (ritenzione) | diagonale (apprendimento) |
|---|---|---|
| dream | **64.4%** (65.0 / 63.9) | 80.7% (82.5 / 79.0) |
| ewc | 13.0% (13.6 / 12.5) | 37.1% (35.0 / 39.2) |
| none | 22.0% (20.1 / 23.9) | 77.9% (76.4 / 79.4) |

Riga finale per livello, entrambi i semi:

```
            L0   L1   L2   L3   L4   L5   L6
dream_s1   100   58   44   57   72   75  100
dream_s2   100   59   42   52   71   76  100
ewc_s1      29   32   14   19    0    0   10
ewc_s2      43   29   10   12    0    0   20
none_s1    100   26    7    9    0    0   98
none_s2    100   30   13   16    1    0  100
```

Tre verdetti, ognuno replicato su entrambi i semi. Il canale replay da solo
vale +42.5 punti di ritenzione su `none` — il cui ~20% replica i build
pre-sogno e fa anche da controllo di validità interna dell'harness. EWC
finisce 9 punti *sotto* il non fare niente e costa ~44 punti di
apprendimento corrente. Le righe per livello dicono perché è notevole:
`none` mostra l'oblio catastrofico da manuale (tiene L0 e l'appena-imparato
L6 a ~100%, perde il mezzo), mentre `ewc` perde anche L0 e anche L6 —
l'ancora danneggia la capacità del modello di imparare il livello che gli
si sta insegnando.

**Perché EWC collassa qui.** La diagnosi è passata per due ipotesi
sbagliate, tenute qui in ordine perché l'eliminazione è ciò che rende
credibile la superstite:

1. *Accumulo del Fisher tra le ancore* — uccisa: con γ=0.95 il fattore di
   accumulo è limitato a ~2.9×, mentre la massa di Fisher misurata cresce
   ~70× (0 → 421 → 5.8k → 20k → 29k lungo i livelli, sul seme 1).
2. *Una spirale a feedback dallo stimare il Fisher su un livello non
   convergente* (la penalità peggiora la convergenza → i gradienti alla
   stima restano grandi → l'ancora dopo è più rigida → …) — uccisa da un
   test di correlazione diretto: le loss di fine livello sulle stesse coppie
   usate per la stima sono ≈0 ovunque (0.0001–0.06) anche nel braccio
   collassato, e Spearman ρ = −0.14 tra loss finale e massa nuova di Fisher.
3. *La superstite:* a loss ≈ 0, `E[g²]` ≈ Var(g) — il Fisher empirico smette
   di misurare curvatura e misura il **disaccordo tra esempi**. Con la loss
   SFT prompt-masked e risposte brevi, ogni coppia spinge i token condivisi
   (separatore, spazio, ':', articoli, '!') in direzioni diverse a seconda
   della risposta: la varianza si concentra esattamente sulla macchina che
   ogni risposta riusa: 20 righe di embedding su 2.590 portano l'89–93%
   della massa di `F_new`, lo spazio da solo il 32–43%, più il 27–32%
   sull'`in_proj` del primo blocco di attention. L'ancora è
   **anti-selettiva** — congela la macchina di produzione della risposta,
   non il sapere del livello. Replicata su entrambi i semi con la stessa
   concentrazione e gli stessi token in testa a masse assolute ~3× diverse:
   il danno segue la concentrazione, non la scala. Caso limite imparentato:
   dopo un livello convergente alla *perfezione* il Fisher è esattamente
   zero (la massa di L0 è 0.0 su entrambi i semi — EWC di fatto spento a
   L1), l'altra faccia dello stesso difetto.

**Confini del claim.** In questo curriculum a memorizzazione quasi-perfetta
per livello, la nostra implementazione dell'EWC online standard con Fisher
diagonale empirico resta nettamente sotto l'experience replay — sia in
ritenzione che in apprendimento corrente — e persino sotto il baseline non
regolarizzato; il replay lo paga portandosi dietro il corpus invece delle
statistiche. Non è nel claim: "EWC è sbagliato in generale". La
normalizzazione per-token del Fisher o l'esclusione dei token strutturali è
un'altra famiglia di algoritmi (Riemannian Walk, Chaudhry et al. 2018); e
il confronto non è compute-matched (N1 rigioca fino a 7 livelli per ciclo
contro 1 — un braccio "ewc + N1 compute-matched" è lavoro futuro), quindi
l'efficienza relativa dei due metodi resta una questione aperta, anche se
il budget non può spiegare ewc che finisce sotto `none`.

### Esempi reali di domanda e risposta

Generati in greedy dai checkpoint post-sogno. Sono riportati anche gli errori:

**L0** — 21/21 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `di ma` | `ma!` | ✓ |
| `di pa` | `pa!` | ✓ |
| `di ta` | `ta!` | ✓ |

**L1** — 23/24 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `di: la casa` | `la casa!` | ✓ |
| `di: il pane` | `il pane!` | ✓ |
| `di: la mamma` | `la mamma!` | ✓ |
| `di: il cane` | `il pane!` | ✗ atteso `il cane!` |

**L2** — 18/18 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `di: il cane` | `il cane!` | ✓ |
| `di: la casa` | `la casa!` | ✓ |
| `di: il pane` | `il pane!` | ✓ |

**L3** — 23/28 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `di: il cane` | `il cane!` | ✓ |
| `di: la mamma` | `la mamma!` | ✓ |
| `di: il gatto` | `il gatto!` | ✓ |
| `di: il sole brilla` | `il sole il gatto il sole bello!` | ✗ atteso `il sole brilla!` |
| `di un numero: tre` | `due!` | ✗ atteso `tre!` |

**L4** — 25/25 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `di: cosa mangia il cane?` | `il cane mangia il pane.` | ✓ |
| `di: cosa mangia il gatto?` | `il gatto mangia il pesce.` | ✓ |
| `di: cosa beve il bambino?` | `il bambino beve il latte.` | ✓ |

**L5** — 19/20 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `com è il cane?` | `il cane è grande.` | ✓ |
| `com è la casa?` | `la casa è piccola.` | ✓ |
| `com è il sole?` | `il sole è caldo.` | ✓ |
| `com è il pane?` | `il cane è grande.` | ✗ atteso `il pane è buono.` |

**L6** — 21/21 corrette

| Domanda | Risposta del modello | |
|---|---|---|
| `cosa ha mangiato il cane?` | `il cane ha mangiato il pane.` | ✓ |
| `cosa ha bevuto il bambino?` | `il bambino ha bevuto il latte.` | ✓ |
| `cosa ha letto il papà?` | `il papà ha letto il libro.` | ✓ |


**L7** — futuro e contrasto fra i tempi (19/19)

| Domanda | Risposta del modello | |
|---|---|---|
| `cosa mangerà il cane domani?` | `domani il cane mangerà il pane.` | ✓ |
| `cosa berrà il bambino domani?` | `domani il bambino berrà il latte.` | ✓ |

**L8** — comparativi (19/19)

| Domanda | Risposta del modello | |
|---|---|---|
| `chi è più grande, il cane o il gatto?` | `il cane è più grande del gatto.` | ✓ |
| `cosa è più caldo, il sole o la luna?` | `il sole è più caldo della luna.` | ✓ |

**L9** — tesi con motivo (14/16)

| Domanda | Risposta del modello | |
|---|---|---|
| `il pane è buono?` | `secondo me il pane è buono perché è caldo.` | ✓ |
| `il cane è fedele?` | `secondo me il cane è fedele perché aspetta.` | ✓ |
| `perché ti piace il pane?` | `il pane il pane è caldo e buono, perciò mi piace.` | ✗ inizio ripetuto |

**L10** — commento motivato (15/16)

| Domanda | Risposta del modello | |
|---|---|---|
| `commenta il sole` | `il sole è caldo, e questo è utile.` | ✓ |
| `commenta il libro` | `il libro insegna, e questo è importante.` | ✓ |
| `commenta il cane` | `il cane il cane il cane è fedele, e questo è bello.` | ✗ inizio ripetuto |

Gli errori residui sono di due famiglie, entrambe circoscritte. La prima: **il
modello non discrimina l'ultima parola del prompt quando più obiettivi
condividono lo stesso prefisso** — `di un numero: tre` e `di un colore: rosso`
differiscono solo nel finale e collassano entrambi su `due!`. La seconda, ai
livelli 9–10: **ripete l'inizio della risposta** (`il cane il cane il cane è
fedele`) prima di completarla correttamente, cioè sa cosa dire ma non sempre
dove iniziare. Non è oblio né limite di capacità — un
SFT puro sugli obiettivi di un livello li porta al 100% in 30 epoche — ma
collasso su una risposta per famiglia di prompt. La leva è aumentare gli
obiettivi distinti per step e variare la testa della domanda.

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
| Parametri | 23.6M | 117M | 1,100M |
| Vocabolario | 8,083 | 50,257 | 32,000 |
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

### Fase corrente — Curriculum validato fino a L6

- [x] Dream Consolidation phase integrata nel curriculum
- [x] Novelty drive (dopamina) — bonus decrescente per token nuovi
- [x] `ignorance` come variabile affettiva autonoma (prior biologico 0.9)
- [x] Didattica **test-then-show** (il modello risponde prima di vedere la
      soluzione) — l'ordine inverso gonfiava la metrica
- [x] Rehearsal **interleaved** sulle coppie gold, oltre a quello testuale
- [x] `local_teacher.json` deterministico per **tutti** gli 11 livelli: pool
      chiuso di obiettivi ripetuti. Il teacher LLM generava prompt quasi sempre
      nuovi (0.94–0.99 distinti per turno contro 0.03–0.10 con quello locale),
      cioè un solo passo di gradiente per obiettivo
- [x] Sogno riordinato: il replay del corpus non parla più per ultimo, e N3 è
      pesato sul livello corrente
- [x] `weight_decay = 0` — con Adam il decay accoppiato spegneva la rete
- [x] Cancello di qualità reale in `build.sh`: sotto soglia il build si ferma
      invece di addestrare i livelli successivi su fondamenta assenti
- [x] Curriculum L0→L6 validato (exact match 82–100%)
- [ ] Completare la validazione L7→L10 (ricostruzione in corso)
- [ ] Aumentare gli obiettivi distinti per step: gli errori residui sono
      collassi su prompt che condividono il prefisso
- [ ] Validare comportamento affettivo su tutti i livelli

### Regime di training da preservare

Cinque elementi, tutti necessari e verificati sperimentalmente. Rimuoverne uno
riporta lo stallo:

1. Teacher deterministico con pool chiuso di obiettivi ripetuti
2. Valutazione test-then-show — mai chiedere ciò che si è appena insegnato
3. Rehearsal interleaved sulle coppie gold
4. Il sogno chiude con la supervisione, non col corpus
5. Ogni obiettivo verificato raggiungibile con `+++`
   (`scripts/validate_teacher_configs.py`)

**Regola trasversale.** Ogni fase che impara dall'*output* del modello invece
che dal target gold degenera: è capitato al pattern mining del sogno (che
rinforzava il babble), alla crescita del vocabolario (mega-token da 48
caratteri), al rinforzo per imitazione e all'addestramento sul prompt del
teacher. Corollario: la fase che chiude il training decide cosa il modello
ricorda, quindi deve essere la più supervisionata, non la più generica.

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

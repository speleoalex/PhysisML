# Piano: dynamic_model

## Context

Due esperimenti indipendenti che condividono la stessa base di codice (`tests/test_1/splx/`),
ma con obiettivi separati. Tenerli separati permette di isolare le variabili e misurare
il contributo di ciascuna innovazione.

**Esperimento A** — Vocabolario dinamico: il tokenizer cresce durante il training.
**Esperimento B** — Sistema affettivo: segnale emotivo innato nel forward pass.

Gli esperimenti vengono implementati in ordine: A prima, B dopo. B può (ma non deve)
essere combinato con A una volta che entrambi funzionano individualmente.

---

## Decisioni architetturali confermate

| Punto | Decisione |
|-------|-----------|
| Vocab di partenza | 501 token da `tests/test_1/checkpoints/it-0/tokenizer.json` |
| Ricodifica | Doppia rappresentazione accettata + DreamConsolidator |
| Soglia frequenza merge | `max(5, buffer_size × 0.005)` adattiva |
| Confronto con test_1 | Parità testo visto, perplexity su validation set fisso |
| ID dopo pruning | **Sparsi** — riga azzerata e mascherata, ID mai riassegnato. ⚠️ Ottimizzazione futura: compattazione periodica degli ID |
| Dataset training | it/0 + it/1 (18.5K chars) |
| Validation set | Ultimo 10% di it/1 (~1.4K chars), separato fisicamente, mai usato per training |
| Import da test_1 | `sys.path` via `dynamic_model/_imports.py` condiviso |
| Iperparametri modello | Stessi di test_1: d_model=128, n_heads=4, n_layers=2, d_ff=512 |

---

## Fondamenta comuni

Entrambi gli esperimenti importano direttamente da `tests/test_1/splx/`:

- `layers.py` — `Linear`, `LayerNorm`, `GELU`, `Dropout`, `FFN`
- `attention.py` — `MultiHeadSelfAttention`
- `transformer.py` — `TransformerBlock`, `GPT`
- `optimizer.py` — `AdamOptimizer`
- `utils.py` — `softmax`, `clip_grad_norm`, `sample_top_k`

Struttura directory condivisa:

```
dynamic_model/
├── piano.md
├── core/           ← componenti condivisi (tokenizer dinamico, embedding, optimizer)
├── exp_a/          ← Esperimento A
│   ├── transformer.py
│   ├── expansion_manager.py
│   ├── dream_consolidator.py   ← fase dormiente / consolidamento
│   └── trainer.py
├── exp_b/          ← Esperimento B
└── persistence/    ← checkpoint condiviso
```

---

## Esperimento A — Vocabolario Dinamico

### Ipotesi

Un vocabolario che cresce progressivamente durante il training, inizializzando i nuovi
token come combinazione dei token "genitori", produce rappresentazioni migliori rispetto
a un vocabolario fisso addestrato sugli stessi dati — a parità di passi di training.

### Cosa NON cambia rispetto a test_1

- Architettura transformer (layers, attention, FFN)
- Optimizer Adam
- Loss cross-entropy
- Training con gradient descent

### Cosa cambia

Solo due componenti vengono sostituiti o estesi:
`DynamicBPETokenizer` e `DynamicEmbedding`.

---

### `core/tokenizer.py` — DynamicBPETokenizer

Estende `BPETokenizer` di test_1:

```python
token_parents: Dict[int, Tuple[int, int]]  # new_id -> (parent_a, parent_b)
token_freq:    Dict[int, int]               # frequenza osservata nel corpus corrente

def grow(self, new_text: str, n_merges: int = 10) -> List[int]:
    """
    Analizza new_text, individua le coppie più frequenti non ancora nel vocab,
    crea fino a n_merges nuovi token. Ritorna [new_id_1, new_id_2, ...].
    
    Non ri-analizza l'intero corpus storico — usa solo il testo recente.
    Questo è il trade-off: efficienza vs ottimalità dei merge.
    """

def get_parent_embedding(self, token_id: int, W: np.ndarray) -> np.ndarray:
    """
    Calcola il vettore iniziale per un nuovo token.
    v_new = 0.7 * (W[parent_a] + W[parent_b]) / 2  +  0.3 * randn * 0.02
    
    Il 70% eredita la semantica dai genitori.
    Il 30% aggiunge rumore per permettere differenziazione durante il training.
    """
```

**Problema noto — ricodifica:** quando "mamma" diventa un token singolo, le associazioni
apprese per la sequenza `[m,a,m,m,a]` non si trasferiscono automaticamente. L'embedding
eredita la geometria dei genitori, ma le relazioni contestuali apprese nei layer di
attention rimangono legate alla sequenza vecchia.

**Strategia adottata (da verificare empiricamente):** accettare la doppia rappresentazione
e misurare se l'attention riesce a collegare le due forme nel tempo. Se non funziona,
implementare una fase di ricodifica del buffer recente dopo ogni espansione.

---

### `core/embedding.py` — DynamicEmbedding

```python
class DynamicEmbedding:
    params["W"]: np.ndarray  # (V, d_model) — tabella embedding standard

    def expand(self, new_id: int, init_vec: np.ndarray) -> None:
        """
        Aggiunge 1 riga a W. Operazione atomica.
        W: (V, d_model) → (V+1, d_model)
        
        Il LM head (weight-tied) si adatta automaticamente:
        logits = x @ W.T → diventa (T, V+1) senza modifiche.
        """

    def backward(self, dout: np.ndarray) -> None:
        """Identico a Embedding.backward() di test_1."""
```

**Nota sui 3D weights:** la versione con `W_ctx (V, 4, d_model)` è descritta nella
sezione "estensioni future" — non fa parte dell'Esperimento A per mantenere le variabili
isolate.

---

### `core/optimizer.py` — DynamicAdam

```python
class DynamicAdam(AdamOptimizer):
    def expand_moments(self, param_name: str, n_new: int = 1) -> None:
        """
        Estende i momenti Adam per un parametro cresciuto.
        
        _m[name] = vstack([_m[name], zeros((n_new, d_model))])
        _v[name] = vstack([_v[name], zeros((n_new, d_model))])
        
        Inizializzazione a zero: matematicamente corretta per un token senza storia.
        Nelle prime iterazioni Adam userà un learning rate effettivo ~ lr (bias correction
        è grande per t piccolo), poi si stabilizza.
        """
```

---

### `exp_a/transformer.py` — DynamicGPT (versione A)

Identico a `GPT` di test_1, con una sola differenza: usa `DynamicEmbedding`
invece di `Embedding`.

```python
def vocab_expand(self, new_id: int, init_vec: np.ndarray) -> None:
    """
    Espansione atomica: aggiunge 1 token.
    
    Prima:  tok_emb.W (V, 128),  logits (T, V),   Adam._m (V, 128)
    Dopo:   tok_emb.W (V+1, 128), logits (T, V+1), Adam._m (V+1, 128)
    
    Tutti gli altri ~413K parametri: invariati (non dipendono da V).
    """
    self.tok_emb.expand(new_id, init_vec)
    self.vocab_size += 1
```

---

### `exp_a/expansion_manager.py` — VocabExpansionManager

Ogni `EXPANSION_INTERVAL` step (default: 500):

```
1. checkpoint.save(reason="pre_expansion")
2. new_ids = tokenizer.grow(recent_text_buffer, n_merges=5)
3. Per ogni new_id:
     init_vec = tokenizer.get_parent_embedding(new_id, model.tok_emb.W)
     model.vocab_expand(new_id, init_vec)
     optimizer.expand_moments("tok_emb_W")
4. checkpoint.save(reason="post_expansion")
```

Il `recent_text_buffer` è una `deque` degli ultimi N caratteri di testo grezzo —
viene usato sia dall'`ExpansionManager` per rilevare nuovi pattern, sia dal
`DreamConsolidator` per il replay consolidato.

---

### `exp_a/dream_consolidator.py` — DreamConsolidator

Ispirato al consolidamento della memoria durante il sonno (hippocampal replay +
synaptic homeostasis). Si attiva ogni `DREAM_TRIGGER` nuovi token aggiunti al vocabolario
(default: 10). Non è un semplice replay — è una fase distinta con quattro obiettivi.

```
Training normale  →→→  [DREAM_TRIGGER: +10 token]  →→→  Training normale
                                   ↓
                        DreamConsolidator.run()
                                   ↓
             1. RE-ENCODING   buffer recente → nuovo vocab
             2. CONSOLIDATION gradient step sul testo re-encodificato
             3. PRUNING       rimozione token mai comparsi nel buffer
             4. DEFRAG        cerca nuovi merge nel testo consolidato
```

**Step 1 — Re-encoding (hippocampal replay):**
Il `recent_text_buffer` contiene testo grezzo acquisito nelle ultime sessioni.
Si re-encodifica con il vocabolario attuale: dove prima c'era `[m,a,m,m,a]`
ora compare `[mamma]`. Il passato viene "risognato" nella forma aggiornata.

**Step 2 — Consolidation training (trasferimento ippocampo → corteccia):**
Si eseguono `DREAM_STEPS` gradient step (default: 50) sul testo re-encodificato.
Il nuovo token riceve training reale in contesti già visti — non solo l'embedding
ereditato dai genitori. Questo risolve la doppia rappresentazione senza
richiedere ri-encodifica continua del corpus storico.

**Step 3 — Pruning (synaptic homeostasis):**
Token aggiunti ma mai apparsi nel buffer re-encodificato vengono marcati come
"inutilizzati". Dopo `PRUNE_GRACE_STEPS` step dormiente senza utilizzo, la riga
viene rimossa dall'embedding e il merge cancellato. Sono i concetti che non
hanno attecchito — il cervello "dimentica" ciò che non rinforza.

```python
def _prune_unused(self) -> List[int]:
    """
    Ritorna lista di token_id rimossi.
    Un token è candidato alla rimozione se:
    - è stato aggiunto dinamicamente (non fa parte dei 501 token base)
    - non appare nel buffer re-encodificato corrente
    - ha superato il periodo di grazia PRUNE_GRACE_STEPS
    """
```

**Step 4 — Defrag (sogni che portano a intuizioni):**
Con il vocabolario più ricco, alcune coppie potrebbero ora superare la soglia
di frequenza nel testo consolidato. Si esegue un mini-round di `tokenizer.grow()`
sul buffer re-encodificato — nuovi merge che il buffer piccolo non vedeva prima.

```python
class DreamConsolidator:
    DREAM_TRIGGER      = 10    # nuovi token prima di sognare
    DREAM_STEPS        = 50    # gradient step durante il sogno
    PRUNE_GRACE_STEPS  = 3     # fasi dormienti prima di rimuovere un token inutile
    BUFFER_CHARS       = 5000  # caratteri di testo grezzo nel buffer

    def __init__(self, model, tokenizer, optimizer, trainer):
        self.tokens_since_dream = 0
        self.unused_token_counts: Dict[int, int] = {}  # id -> fasi dormienti senza uso

    def notify_expansion(self, n_new_tokens: int) -> None:
        """Chiamato da VocabExpansionManager dopo ogni grow()."""
        self.tokens_since_dream += n_new_tokens
        if self.tokens_since_dream >= self.DREAM_TRIGGER:
            self.run()
            self.tokens_since_dream = 0

    def run(self) -> dict:
        """
        Esegue la fase dormiente completa.
        Ritorna statistiche: {pruned, new_merges, consolidation_loss}
        """
```

**Interazione con Esperimento B (futura combinazione):**
La fase dormiente può pesare il replay in base allo stato affettivo — esperienze
con alto `pleasure` o alto `pain` vengono replayed più frequentemente. Il cervello
consolida preferenzialmente i ricordi emotivamente significativi.

---

### `exp_a/trainer.py` — Trainer A

Training online: ogni sequenza è uno step, nessuna epoca fissa.
Il catastrophic forgetting è gestito dal `DreamConsolidator`, non serve
un meccanismo di replay separato nel trainer.

```python
def step(self, text_chunk: str) -> float:
    # Accumula nel buffer grezzo per il DreamConsolidator
    recent_text_buffer.append(text_chunk)

    ids = tokenizer.encode(text_chunk)
    logits = model.forward(ids, training=True)
    loss, dlogits = model.loss(logits, ids)
    model.backward(dlogits)
    grads = clip_grad_norm(model.get_grads(), max_norm=1.0)
    new_params = optimizer.step(model.get_params(), grads)
    model.apply_params(new_params)

    # Espansione vocabolario (notifica al DreamConsolidator se ci sono nuovi token)
    new_ids = expansion_manager.maybe_expand(step_count)
    if new_ids:
        dream_consolidator.notify_expansion(len(new_ids))

    return float(loss)
```

---

### Metrica di valutazione — Esperimento A

Confronto diretto con `tests/test_1`:

| Metrica | test_1 (vocab fisso) | Esperimento A (vocab dinamico) |
|---------|----------------------|-------------------------------|
| Loss finale | misurata | misurata |
| Perplexity su validation set | misurata | misurata |
| Dimensione vocab al termine | fissa (2000) | variabile |
| Token/secondo durante inference | baseline | confrontato |

Confronto a parità di **testo visto** (non di step — il vocab diverso cambia la
lunghezza delle sequenze encodificate). Si misura la perplexity su un validation
set fisso (ultimo 10% del corpus, mai usato per training).

Le fasi dormienti vengono logggate separatamente: `dream_loss`, `tokens_pruned`,
`new_merges_in_dream` — per capire quanto contribuisce il consolidamento.

---

## Esperimento B — Sistema Affettivo

### Ipotesi

Un segnale affettivo (confidence, pleasure, fear) che modula l'output del modello —
senza entrare nel backward pass — migliora la qualità delle risposte in presenza di
feedback esterno, rispetto a un modello senza tale segnale.

**Separazione critica:** il segnale affettivo modifica i logits **solo durante l'inference**,
non durante il training. Questo evita il loop di retroazione instabile identificato
nell'analisi (`alta fear → distribuzione piatta → confidence bassa → più fear → divergenza`).

---

### `exp_b/affect_state.py` — AffectState

Quattro scalari aggiornati con EWMA (alpha = 0.1):

| Variabile | Formula di aggiornamento | Init |
|-----------|--------------------------|------|
| `confidence` | `1 - entropy(softmax(logits[-1])) / log(V)` | 0.1 |
| `ignorance` | `EWMA(mean(‖W[i]‖ < threshold))` | 0.9 |
| `pleasure` | `EWMA(max(0, feedback))` | 0.5 |
| `pain` | `EWMA(max(0, -feedback))` | 0.0 |
| `fear` | `0.5 * pain + 0.5 * (1 - confidence)` | 0.45 |

`confidence` e `ignorance` si aggiornano ad ogni forward pass (senza feedback).
`pleasure` e `pain` si aggiornano solo quando arriva un segnale di feedback esterno.

Prior biologico: il modello nasce ignaro e neutro, senza dolore pregresso.

---

### `exp_b/modulator.py` — AffectModulator

Applicato **solo durante l'inference** (non nel training loop):

```python
def modulate(self, logits: np.ndarray) -> np.ndarray:
    """
    Modifica i logits in base allo stato affettivo corrente.
    Chiamato DOPO model.forward(), PRIMA del sampling.
    NON chiamato durante il training (il backward non lo vede).
    
    1. Temperature adattiva:
       effective_temp = base_temp * (1 + fear * 2.0)
       logits = logits / effective_temp
       → fear alta: output più diffuso (incerto)
       → fear bassa: output più deciso
    
    2. Boost token "non lo so":
       if ignorance > 0.7 or confidence < 0.15:
           logits[DONT_KNOW_ID] += ignorance * boost_k
    
    3. Pain gate (memoria negativa):
       logits[pain_token_ids] -= pain * inhibition_strength
    
    4. Pleasure gate (memoria positiva):
       logits[pleasure_token_ids] += pleasure * amplification_strength
    """
```

---

### `exp_b/axioms.py` — AxiomRegistry

```python
@dataclass
class Axiom:
    token_sequence:   List[int]
    protection_level: float   # 0..1, quanto è forte la protezione al gradient
    is_objective:     bool    # True: max 1.0; False: max 0.6 (opinioni modificabili)
    frequency_seen:   int     # quante volte è stato visto nel corpus

def apply_to_grads(self, grads: dict) -> dict:
    """
    Scala i gradient per i token degli assiomi prima dell'optimizer step.
    grads["tok_emb_W"][protected_ids] *= (1 - protection_level)
    """
```

**Limitazione nota:** protegge solo il token embedding, non i pesi di attention/FFN
dove la "conoscenza" è effettivamente distribuita. È una protezione parziale ma utile.
La protezione completa è un problema aperto.

Creazione assiomi:
- **Manuale** (tutti gli esperimenti usano autorità massima per ora):
  `registry.register(token_sequence, is_objective=True, protection=1.0)`
- **Automatica**: sequenza vista > 100 volte con feedback positivo consistente

---

### `exp_b/trainer.py` — Trainer B

Identico al Trainer A, con l'aggiunta del ciclo feedback:

```python
def step_with_feedback(self, prompt: str, response: str,
                        feedback: float) -> dict:
    """
    feedback: float in [-1.0, +1.0]
        +1.0 = approvazione esplicita
         0.0 = neutro (osservazione passiva: vocab cresce, pesi no)
        -1.0 = disapprovazione esplicita
    
    Se feedback == 0.0: encode il testo (il tokenizer può crescere),
                        ma NON fare il backward pass.
    Altrimenti: pipeline completa con backward.
    """
    ids = tokenizer.encode(prompt + response)
    logits = model.forward(ids, training=(feedback != 0.0))
    
    if feedback != 0.0:
        loss, dlogits = model.loss(logits, ids)
        model.backward(dlogits)
        grads = axiom_registry.apply_to_grads(model.get_grads())
        grads = clip_grad_norm(grads, max_norm=1.0)
        new_params = optimizer.step(model.get_params(), grads)
        model.apply_params(new_params)
    
    affect_state.update_from_logits(logits)
    affect_state.update_from_feedback(feedback)
    ...
```

---

### Metrica di valutazione — Esperimento B

Confronto: stesso modello con e senza modulazione affettiva, su task di Q&A con feedback:

| Metrica | Senza affetto | Con affetto |
|---------|---------------|-------------|
| % risposte corrette dopo N feedback | misurata | misurata |
| Uso appropriato del token "non lo so" | misurato | misurato |
| Stabilità (loss non diverge) | verificata | verificata |

---

## Estensioni future (non in scope per A e B)

- **3D weights** (`W_ctx (V, 4, d_model)`): correzione contestuale per-token.
  Da aggiungere dopo aver validato Esperimento A.
- **Autorevolezza della fonte**: `authority_level` nel `FeedbackSignal`.
  Gli esperimenti correnti usano sempre autorità massima.
- **Inconscio multilingua**: strato comune lingua-agnostico.
  Da progettare quando si introduce il training multilingua.
- **Combinazione A+B**: tokenizer dinamico + sistema affettivo insieme.
  Da fare dopo che entrambi funzionano separatamente.

---

## Persistenza condivisa

### `persistence/checkpoint.py` — DynamicCheckpoint

Salva un manifest JSON accanto ad ogni `.npz` con `vocab_size` e `experiment`:

```
dynamic_model/
├── exp_a/snapshots/
│   ├── step_00500.npz + manifest.json
│   ├── expansion_001.npz   (pre-espansione)
│   ├── dream_001.npz       (post-fase dormiente)
│   └── best.npz + manifest.json
└── exp_b/snapshots/
    ├── step_00500.npz
    └── best.npz
```

Il manifest include: `vocab_size`, `dream_count`, `tokens_pruned_total`,
`curriculum_level`, timestamp.

`rollback_to(n)` → ripristina il modello allo snapshot n-esimo.

---

## Ordine di implementazione

**Fase 1 — Esperimento A**

1. `core/tokenizer.py` — `DynamicBPETokenizer.grow()` + `token_parents` + soglia adattiva
2. `core/embedding.py` — `DynamicEmbedding.expand()`
3. `core/optimizer.py` — `DynamicAdam.expand_moments()`
4. `exp_a/transformer.py` — `DynamicGPT` (wrappa test_1, cambia solo embedding)
5. `exp_a/expansion_manager.py` — `VocabExpansionManager`
6. `exp_a/dream_consolidator.py` — `DreamConsolidator` (re-encoding, consolidation, pruning, defrag)
7. `exp_a/trainer.py` — `Trainer A` con `DreamConsolidator` integrato
8. `persistence/checkpoint.py` — `DynamicCheckpoint` con snapshot dream
9. Test A: train 1000 step con espansione, verifica dream phase, confronta
   perplexity con test_1 su stesso validation set

**Fase 2 — Esperimento B**

9. `exp_b/affect_state.py` — `AffectState` con EWMA
10. `exp_b/modulator.py` — `AffectModulator` (solo inference)
11. `exp_b/axioms.py` — `AxiomRegistry` con gradient scaling
12. `exp_b/trainer.py` — `Trainer B` con ciclo feedback
13. Test B: verifica che `confidence` scende con distribuzione piatta,
    verifica che assioma protetto non cambia con gradient,
    verifica che feedback=0 non aggiorna i pesi

**Fase 3 — Valutazione**

14. Script di confronto: test_1 vs Esperimento A vs Esperimento B
15. Report: loss curve, perplexity, uso token "non lo so", stabilità

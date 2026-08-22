# Come funzionano i modelli linguistici classici

*Leggi in: [English](../en/classic_language_models.md)*

*Dal token al transformer al fine-tuning*

---

## Indice

1. [Il problema: il testo non è numerico](#1-il-problema-il-testo-non-è-numerico)
2. [Tokenizzazione](#2-tokenizzazione)
3. [Embedding: i numeri che rappresentano le parole](#3-embedding-i-numeri-che-rappresentano-le-parole)
4. [Il Transformer](#4-il-transformer)
5. [Pre-training: imparare a predire il prossimo token](#5-pre-training-imparare-a-predire-il-prossimo-token)
6. [Fine-tuning e RLHF](#6-fine-tuning-e-rlhf)
7. [Inferenza: come genera il testo](#7-inferenza-come-genera-il-testo)
8. [Scala e limiti](#8-scala-e-limiti)

---

## 1. Il problema: il testo non è numerico

Una rete neurale lavora esclusivamente con numeri (vettori, matrici, tensori). Il testo è una sequenza di caratteri. Il primo problema è quindi: **come trasformare il testo in numeri su cui fare calcoli?**

La risposta non è banale. Alcune opzioni:

- **Carattere per carattere**: "c", "a", "n", "e" → [99, 97, 110, 101] (codici ASCII). Troppo granulare: le relazioni semantiche sono lontane.
- **Parola per parola**: "cane" → 4821. Vocabolario enorme (milioni di parole), nessuna generalizzazione morfologica.
- **Subword (BPE)**: compromesso ottimale — usato da tutti i modelli moderni.

---

## 2. Tokenizzazione

### Byte Pair Encoding (BPE)

L'algoritmo BPE parte dai singoli caratteri e fonde iterativamente le coppie più frequenti:

```
Corpus iniziale: "cane cani cane gatto gatti"

Iterazione 1: coppia più frequente = ("c","a") → "ca"
  "ca ne ca ni ca ne ga tto ga tti"

Iterazione 2: coppia più frequente = ("ca","ne") → "cane"
  "cane ca ni cane ga tto ga tti"

... dopo N iterazioni si ottiene un vocabolario di N+256 token
```

Il risultato è un **vocabolario** (dizionario `token_id → stringa`) e un **tokenizer** che scompone qualsiasi testo in quel vocabolario:

```
"il cane dorme"  →  [241, 4821, 8103]
"il cane mangia" →  [241, 4821, 9202]
```

I modelli moderni usano vocabolari da 32.000–100.000 token. PhysisML usa 8.000 token con slot dormienti (8.083 attivi dopo il curriculum).

### Perché subword e non parole intere?

- **Generalizzazione morfologica**: "correvo", "correvi", "correrò" condividono "corr" come token — il modello impara una sola radice
- **Copertura universale**: qualsiasi testo è rappresentabile, anche parole nuove ("ChatGPT" → ["Chat", "G", "PT"])
- **Efficienza**: vocabolario contenuto, sequenze non troppo lunghe

---

## 3. Embedding: i numeri che rappresentano le parole

Un token ID è solo un numero intero. Per fare calcoli utili, ogni token viene convertito in un **vettore denso** di dimensione `d_model` (es. 256, 768, 4096):

```
tok_emb_W : matrice (V × d_model)
  V = dimensione vocabolario
  d_model = dimensione dello spazio vettoriale

token_id=4821 ("cane") → tok_emb_W[4821] = [0.12, -0.34, 0.87, ..., 0.21]
                                              ←────────── d_model valori ──────────→
```

Questa matrice viene **appresa durante il training**: inizialmente casuale, diventa progressivamente significativa — vettori di parole simili si avvicinano nello spazio.

### Embedding posizionale

Il transformer di per sé non ha nozione di ordine. "il cane mangia il gatto" e "il gatto mangia il cane" producono gli stessi token in ordine diverso. Per distinguerli si aggiunge un **embedding posizionale**:

```
input_embedding[t] = tok_emb[token_id[t]] + pos_emb[t]
```

Dove `pos_emb` è un vettore diverso per ogni posizione (appreso o fisso con formule sinusoidali).

---

## 4. Il Transformer

Il transformer è l'architettura centrale di tutti i modelli linguistici moderni (GPT, LLaMA, BERT, T5...). È composto da **blocchi impilati**, ognuno con due sottocomponenti:

```
Input x
  ↓
┌─────────────────────────────────┐
│  h = x + Attention(LayerNorm(x))│  ← self-attention + residual
│  out = h + FFN(LayerNorm(h))    │  ← feed-forward + residual
└─────────────────────────────────┘
  ↓
Output (stessa forma dell'input)
```

### 4.1 Self-Attention

L'intuizione: ogni token "guarda" tutti gli altri token per capire di cosa si parla nel contesto.

**Meccanismo**:

```
Per ogni token t, calcola tre vettori:
  Q[t] = x[t] @ W_Q    (Query:  "cosa sto cercando?")
  K[t] = x[t] @ W_K    (Key:    "cosa offro agli altri?")
  V[t] = x[t] @ W_V    (Value:  "cosa trasmetto se vengo selezionato?")

Punteggio di attenzione da t verso ogni altro token s:
  score[t,s] = (Q[t] · K[s]) / sqrt(d_k)

Pesi normalizzati (softmax):
  attn[t,s] = softmax(score[t,:])

Output del token t:
  out[t] = Σ_s attn[t,s] · V[s]
```

In pratica: "il" vicino a "cane" avrà un peso alto verso "cane", e il suo vettore output incorporerà informazioni da "cane".

**Causal masking** (nei modelli generativi come GPT): il token alla posizione `t` può vedere solo i token `0..t`, non quelli futuri. Questo garantisce che la predizione sia causale.

**Multi-head attention**: invece di un'unica attention, si usano `n_heads` attention parallele con proiezioni diverse, poi il risultato viene concatenato. Ogni head "specializza" l'attenzione su aspetti diversi (sintassi, semantica, coreference...).

### 4.2 Feed-Forward Network (FFN)

Dopo l'attention, ogni token viene trasformato indipendentemente da un piccolo MLP:

```
FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2

Dimensioni:
  W_1 : (d_model → d_ff)    es. 256 → 1024
  W_2 : (d_ff → d_model)    es. 1024 → 256
```

L'FFN introduce non-linearità e aumenta la capacità di ogni layer. In LLaMA usa SwiGLU invece di GELU.

### 4.3 LayerNorm e residual

**Pre-LayerNorm** (usato in GPT-3, LLaMA, PhysisML):

```
h = x + Attention(LayerNorm(x))
```

La normalizzazione *prima* dell'operazione stabilizza il training. Il residual connection `x + ...` permette al gradiente di fluire direttamente agli strati profondi.

### 4.4 Struttura completa

```
Token IDs (T,)
  → Embedding (T, d_model)
  → Dropout
  → N × TransformerBlock
  → FinalLayerNorm
  → Logits = x @ Embedding.W^T     ← weight tying
  → Softmax → probabilità per V token
```

**Weight tying**: la matrice di embedding di input e quella di output (logits) sono la stessa matrice trasposta. Riduce i parametri e migliora la generalizzazione.

---

## 5. Pre-training: imparare a predire il prossimo token

### L'obiettivo

Il modello impara un unico compito: **dato il contesto, predire il prossimo token**.

```
Testo: "il cane dorme sul tappeto"
Input:  ["il", "cane", "dorme", "sul"]
Target: ["cane", "dorme", "sul", "tappeto"]
```

Per ogni posizione, il modello produce una distribuzione di probabilità su tutto il vocabolario. La loss è la **cross-entropy** tra la distribuzione predetta e il token reale:

```
loss = -log P(token_reale | contesto)
```

Minimizzando questa loss su miliardi di esempi, il modello impara:
- grammatica (quali parole si combinano)
- semantica (il significato dei token)
- fatti (associazioni nel corpus)
- ragionamento (strutture if-then nelle frasi)

### Il corpus

Il pre-training richiede quantità enormi di testo:

| Modello | Token di training |
|---------|------------------|
| GPT-2 (2019) | 40 miliardi |
| GPT-3 (2020) | 300 miliardi |
| LLaMA 2 (2023) | 2.000 miliardi |
| TinyLlama (2024) | 3.000 miliardi |

Le fonti tipiche: Common Crawl (web), Wikipedia, libri, codice, articoli scientifici.

### Ottimizzatore

**AdamW** è lo standard: combina Adam (momentum adattivo per parametro) con weight decay (regolarizzazione L2):

```
m_t = β1·m_{t-1} + (1-β1)·g_t               # momento del 1° ordine
v_t = β2·v_{t-1} + (1-β2)·g_t²              # momento del 2° ordine
θ_t = θ_{t-1} - α · m_t/(√v_t + ε) - λ·θ   # update + weight decay
```

Tipici: β1=0.9, β2=0.95, ε=1e-8, α=3e-4 (con warmup + cosine decay).

---

## 6. Fine-tuning e RLHF

Il modello pre-addestrato è un **predittore di token** — sa completare testo ma non risponde a domande, non segue istruzioni, non ha personalità. Per ottenere un assistente servono fasi successive.

### 6.1 Supervised Fine-Tuning (SFT)

Training su un dataset curato di coppie (istruzione, risposta):

```
Input:  "Spiega la fotosintesi in modo semplice"
Output: "La fotosintesi è il processo con cui..."
```

Richiede ~10.000–100.000 esempi di alta qualità (scritti o curati da umani). Questa fase insegna al modello il **formato** della conversazione e lo stile di risposta.

Tecnica comune: **LoRA** (Low-Rank Adaptation) — invece di aggiornare tutti i miliardi di parametri, si aggiungono matrici di aggiornamento piccole (rango 4–64) a ogni layer. Riduce il costo da 100× a 1000×.

### 6.2 RLHF (Reinforcement Learning from Human Feedback)

La pipeline usata da ChatGPT, Claude, Gemini:

**Step 1 — Reward Model**: si addestra un modello separato a stimare la qualità di una risposta. Umani confrontano coppie di risposte ("quale preferisci?") e questo segnale addestra il reward model.

**Step 2 — PPO**: il modello generativo viene aggiornato tramite PPO (Proximal Policy Optimization) per massimizzare il reward, con una penalità KL che impedisce di deviare troppo dal modello SFT:

```
max E[reward(x,y)] - β·KL(π_θ || π_SFT)
```

Il risultato: risposte più utili, meno dannose, più allineate alle preferenze umane.

### 6.3 DPO (alternativa moderna)

**Direct Preference Optimization** — semplifica RLHF eliminando il reward model esplicito. Si addestra direttamente su coppie (risposta buona, risposta cattiva):

```
loss = -log σ(β·(log π_θ(y_win) - log π_SFT(y_win))
              - β·(log π_θ(y_lose) - log π_SFT(y_lose)))
```

Più stabile di PPO, meno costoso, risultati comparabili.

---

## 7. Inferenza: come genera il testo

Il modello produce **un token alla volta**:

```
1. Tokenizza il prompt
2. Forward pass → distribuzione di probabilità su V token
3. Campiona un token dalla distribuzione
4. Aggiunge il token al contesto
5. Ripete da 2 finché non genera <EOS> o raggiunge max_tokens
```

### Strategie di campionamento

**Greedy**: prende sempre il token con probabilità massima. Deterministica ma ripetitiva.

**Temperature**: divide i logit per T prima del softmax. T > 1 = più casualità, T < 1 = più determinismo.

```
P(token_i) = softmax(logits / T)
```

**Top-k**: campiona solo tra i k token più probabili (es. k=50).

**Top-p (nucleus)**: campiona dai token che coprono cumulativamente p probabilità (es. p=0.9). Si adatta automaticamente alla distribuzione.

**Beam search**: mantiene k "ipotesi" parallele, sceglie la sequenza con log-probabilità totale più alta. Usata per traduzione, meno per conversazione.

---

## 8. Scala e limiti

### Scaling laws (Kaplan et al., 2020 / Hoffmann et al., 2022)

La perdita scende prevedibilmente con la scala:

```
L(N, D) ≈ A/N^α + B/D^β + const

N = numero di parametri
D = token di training
```

**Chinchilla optimal** (Hoffmann): per un budget computazionale fisso, è ottimale avere `D ≈ 20·N` token. Es. un modello da 7B param dovrebbe essere addestrato su ~140B token.

TinyLlama (1.1B param) ha usato 3T token → ~3× sovra-addestrato rispetto al Chinchilla optimal, deliberatamente per ottenere un modello piccolo ma molto capace.

### Limiti fondamentali dei modelli classici

| Limite | Descrizione |
|--------|-------------|
| **Hallucination** | Genera testo plausibile ma falso — non distingue tra "ha senso" e "è vero" |
| **Context window** | Non ha memoria oltre N token (tipicamente 2K–128K) |
| **Static knowledge** | La conoscenza si ferma alla data di cutoff del corpus |
| **No grounding** | Non percepisce il mondo fisico, non verifica le affermazioni |
| **Costo energetico** | Training: milioni di kWh; inferenza: watt per token |
| **Emergenza imprevedibile** | Capacità emergono all'improvviso a certe scale, non pianificabili |

### Perché PhysisML è diverso

Il training classico tratta tutti gli esempi uniformemente, senza curriculum, senza feedback interattivo, senza stato affettivo. La conoscenza emerge come effetto collaterale della scala. PhysisML testa l'ipotesi che sia possibile un percorso alternativo: meno dati, più struttura, apprendimento progressivo — come fa un bambino.

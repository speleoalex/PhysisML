# How classic language models work

*Read this in: [Italiano](../it/modelli_linguistici_classici.md)*

*From the token to the transformer to fine-tuning*

---

## Table of contents

1. [The problem: text is not numeric](#1-the-problem-text-is-not-numeric)
2. [Tokenization](#2-tokenization)
3. [Embeddings: the numbers that represent words](#3-embeddings-the-numbers-that-represent-words)
4. [The Transformer](#4-the-transformer)
5. [Pre-training: learning to predict the next token](#5-pre-training-learning-to-predict-the-next-token)
6. [Fine-tuning and RLHF](#6-fine-tuning-and-rlhf)
7. [Inference: how it generates text](#7-inference-how-it-generates-text)
8. [Scale and limits](#8-scale-and-limits)

---

## 1. The problem: text is not numeric

A neural network works exclusively with numbers (vectors, matrices, tensors). Text is a sequence of characters. The first problem is therefore: **how do you turn text into numbers you can compute on?**

The answer is not trivial. Some options:

- **Character by character**: "c", "a", "n", "e" → [99, 97, 110, 101] (ASCII codes). Too granular: semantic relations are far apart.
- **Word by word**: "cane" → 4821. Huge vocabulary (millions of words), no morphological generalization.
- **Subword (BPE)**: the optimal compromise — used by all modern models.

---

## 2. Tokenization

### Byte Pair Encoding (BPE)

The BPE algorithm starts from single characters and iteratively merges the most frequent pairs:

```
Initial corpus: "cane cani cane gatto gatti"

Iteration 1: most frequent pair = ("c","a") → "ca"
  "ca ne ca ni ca ne ga tto ga tti"

Iteration 2: most frequent pair = ("ca","ne") → "cane"
  "cane ca ni cane ga tto ga tti"

... after N iterations you obtain a vocabulary of N+256 tokens
```

The result is a **vocabulary** (a `token_id → string` dictionary) and a **tokenizer** that breaks any text down into that vocabulary:

```
"il cane dorme"  →  [241, 4821, 8103]
"il cane mangia" →  [241, 4821, 9202]
```

Modern models use vocabularies of 32,000–100,000 tokens. PhysisML uses 501 tokens (experimental).

### Why subwords and not whole words?

- **Morphological generalization**: "correvo", "correvi", "correrò" share "corr" as a token — the model learns a single root
- **Universal coverage**: any text is representable, even new words ("ChatGPT" → ["Chat", "G", "PT"])
- **Efficiency**: a contained vocabulary, sequences that are not too long

---

## 3. Embeddings: the numbers that represent words

A token ID is just an integer. To do useful computation, every token is converted into a **dense vector** of size `d_model` (e.g. 256, 768, 4096):

```
tok_emb_W : matrix (V × d_model)
  V = vocabulary size
  d_model = size of the vector space

token_id=4821 ("cane") → tok_emb_W[4821] = [0.12, -0.34, 0.87, ..., 0.21]
                                              ←────────── d_model values ──────────→
```

This matrix is **learned during training**: random at first, it progressively becomes meaningful — vectors of similar words move closer together in the space.

### Positional embedding

The transformer by itself has no notion of order. "il cane mangia il gatto" and "il gatto mangia il cane" produce the same tokens in a different order. To tell them apart, a **positional embedding** is added:

```
input_embedding[t] = tok_emb[token_id[t]] + pos_emb[t]
```

Where `pos_emb` is a different vector for each position (learned, or fixed with sinusoidal formulas).

---

## 4. The Transformer

The transformer is the central architecture of all modern language models (GPT, LLaMA, BERT, T5...). It is made of **stacked blocks**, each with two subcomponents:

```
Input x
  ↓
┌─────────────────────────────────┐
│  h = x + Attention(LayerNorm(x))│  ← self-attention + residual
│  out = h + FFN(LayerNorm(h))    │  ← feed-forward + residual
└─────────────────────────────────┘
  ↓
Output (same shape as the input)
```

### 4.1 Self-Attention

The intuition: every token "looks at" all the other tokens to understand what is being talked about in the context.

**Mechanism**:

```
For each token t, compute three vectors:
  Q[t] = x[t] @ W_Q    (Query:  "what am I looking for?")
  K[t] = x[t] @ W_K    (Key:    "what do I offer to the others?")
  V[t] = x[t] @ W_V    (Value:  "what do I transmit if I am selected?")

Attention score from t towards every other token s:
  score[t,s] = (Q[t] · K[s]) / sqrt(d_k)

Normalized weights (softmax):
  attn[t,s] = softmax(score[t,:])

Output of token t:
  out[t] = Σ_s attn[t,s] · V[s]
```

In practice: "il" next to "cane" will have a high weight towards "cane", and its output vector will incorporate information from "cane".

**Causal masking** (in generative models such as GPT): the token at position `t` can see only tokens `0..t`, not the future ones. This guarantees that the prediction is causal.

**Multi-head attention**: instead of a single attention, `n_heads` parallel attentions with different projections are used, and the result is then concatenated. Each head "specializes" its attention on different aspects (syntax, semantics, coreference...).

### 4.2 Feed-Forward Network (FFN)

After the attention, every token is transformed independently by a small MLP:

```
FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2

Dimensions:
  W_1 : (d_model → d_ff)    e.g. 256 → 1024
  W_2 : (d_ff → d_model)    e.g. 1024 → 256
```

The FFN introduces non-linearity and increases the capacity of each layer. In LLaMA it uses SwiGLU instead of GELU.

### 4.3 LayerNorm and residual

**Pre-LayerNorm** (used in GPT-3, LLaMA, PhysisML):

```
h = x + Attention(LayerNorm(x))
```

Normalizing *before* the operation stabilizes training. The residual connection `x + ...` lets the gradient flow directly to the deep layers.

### 4.4 Complete structure

```
Token IDs (T,)
  → Embedding (T, d_model)
  → Dropout
  → N × TransformerBlock
  → FinalLayerNorm
  → Logits = x @ Embedding.W^T     ← weight tying
  → Softmax → probabilities over V tokens
```

**Weight tying**: the input embedding matrix and the output (logits) matrix are the same matrix, transposed. It reduces the parameters and improves generalization.

---

## 5. Pre-training: learning to predict the next token

### The objective

The model learns a single task: **given the context, predict the next token**.

```
Text:   "il cane dorme sul tappeto"
Input:  ["il", "cane", "dorme", "sul"]
Target: ["cane", "dorme", "sul", "tappeto"]
```

For each position, the model produces a probability distribution over the whole vocabulary. The loss is the **cross-entropy** between the predicted distribution and the actual token:

```
loss = -log P(actual_token | context)
```

By minimizing this loss over billions of examples, the model learns:
- grammar (which words combine)
- semantics (the meaning of tokens)
- facts (associations in the corpus)
- reasoning (if-then structures in sentences)

### The corpus

Pre-training requires enormous amounts of text:

| Model | Training tokens |
|---------|------------------|
| GPT-2 (2019) | 40 billion |
| GPT-3 (2020) | 300 billion |
| LLaMA 2 (2023) | 2,000 billion |
| TinyLlama (2024) | 3,000 billion |

Typical sources: Common Crawl (web), Wikipedia, books, code, scientific papers.

### Optimizer

**AdamW** is the standard: it combines Adam (adaptive per-parameter momentum) with weight decay (L2 regularization):

```
m_t = β1·m_{t-1} + (1-β1)·g_t               # 1st order moment
v_t = β2·v_{t-1} + (1-β2)·g_t²              # 2nd order moment
θ_t = θ_{t-1} - α · m_t/(√v_t + ε) - λ·θ   # update + weight decay
```

Typical values: β1=0.9, β2=0.95, ε=1e-8, α=3e-4 (with warmup + cosine decay).

---

## 6. Fine-tuning and RLHF

The pre-trained model is a **token predictor** — it can complete text but it does not answer questions, does not follow instructions, has no personality. To obtain an assistant, further phases are needed.

### 6.1 Supervised Fine-Tuning (SFT)

Training on a curated dataset of (instruction, answer) pairs:

```
Input:  "Explain photosynthesis in simple terms"
Output: "Photosynthesis is the process by which..."
```

It requires ~10,000–100,000 high-quality examples (written or curated by humans). This phase teaches the model the **format** of the conversation and the answer style.

A common technique: **LoRA** (Low-Rank Adaptation) — instead of updating all the billions of parameters, small update matrices (rank 4–64) are added to each layer. It reduces the cost by 100× to 1000×.

### 6.2 RLHF (Reinforcement Learning from Human Feedback)

The pipeline used by ChatGPT, Claude, Gemini:

**Step 1 — Reward Model**: a separate model is trained to estimate the quality of an answer. Humans compare pairs of answers ("which do you prefer?") and this signal trains the reward model.

**Step 2 — PPO**: the generative model is updated via PPO (Proximal Policy Optimization) to maximize the reward, with a KL penalty that prevents it from deviating too much from the SFT model:

```
max E[reward(x,y)] - β·KL(π_θ || π_SFT)
```

The result: answers that are more useful, less harmful, more aligned with human preferences.

### 6.3 DPO (the modern alternative)

**Direct Preference Optimization** — it simplifies RLHF by eliminating the explicit reward model. Training happens directly on (good answer, bad answer) pairs:

```
loss = -log σ(β·(log π_θ(y_win) - log π_SFT(y_win))
              - β·(log π_θ(y_lose) - log π_SFT(y_lose)))
```

More stable than PPO, less expensive, comparable results.

---

## 7. Inference: how it generates text

The model produces **one token at a time**:

```
1. Tokenize the prompt
2. Forward pass → probability distribution over V tokens
3. Sample a token from the distribution
4. Append the token to the context
5. Repeat from 2 until it generates <EOS> or reaches max_tokens
```

### Sampling strategies

**Greedy**: always takes the token with the highest probability. Deterministic but repetitive.

**Temperature**: divides the logits by T before the softmax. T > 1 = more randomness, T < 1 = more determinism.

```
P(token_i) = softmax(logits / T)
```

**Top-k**: samples only among the k most probable tokens (e.g. k=50).

**Top-p (nucleus)**: samples from the tokens that cumulatively cover p probability (e.g. p=0.9). It adapts automatically to the distribution.

**Beam search**: keeps k parallel "hypotheses" and picks the sequence with the highest total log-probability. Used for translation, less for conversation.

---

## 8. Scale and limits

### Scaling laws (Kaplan et al., 2020 / Hoffmann et al., 2022)

The loss decreases predictably with scale:

```
L(N, D) ≈ A/N^α + B/D^β + const

N = number of parameters
D = training tokens
```

**Chinchilla optimal** (Hoffmann): for a fixed compute budget, it is optimal to have `D ≈ 20·N` tokens. E.g. a 7B-parameter model should be trained on ~140B tokens.

TinyLlama (1.1B params) used 3T tokens → ~3× over-trained with respect to the Chinchilla optimal, deliberately, in order to obtain a small but very capable model.

### Fundamental limits of classic models

| Limit | Description |
|--------|-------------|
| **Hallucination** | It generates plausible but false text — it does not distinguish between "makes sense" and "is true" |
| **Context window** | It has no memory beyond N tokens (typically 2K–128K) |
| **Static knowledge** | Knowledge stops at the corpus cutoff date |
| **No grounding** | It does not perceive the physical world, it does not verify statements |
| **Energy cost** | Training: millions of kWh; inference: watts per token |
| **Unpredictable emergence** | Capabilities appear suddenly at certain scales, and cannot be planned |

### Why PhysisML is different

Classic training treats all examples uniformly, with no curriculum, no interactive feedback, no affective state. Knowledge emerges as a side effect of scale. PhysisML tests the hypothesis that an alternative path is possible: less data, more structure, progressive learning — the way a child does it.

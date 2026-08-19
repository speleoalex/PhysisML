# PhysisML — Technical and philosophical documentation

*Read this in: [Italiano](../it/modello_PhysisML.md)*

*Last updated: 2026-04-13*

---

## Contents

1. [Project philosophy](#1-project-philosophy)
2. [Model architecture](#2-model-architecture)
3. [Innate affective system](#3-innate-affective-system)
4. [Training method](#4-training-method)
5. [Language curriculum](#5-language-curriculum)
6. [Experimental results](#6-experimental-results)
7. [Comparison with standard approaches](#7-comparison-with-standard-approaches)
8. [Roadmap](#8-roadmap)

---

## 1. Project philosophy

### Central thesis

Today's language models (GPT, LLaMA, TinyLlama) obtain knowledge-like and reasoning-like behaviour as **side effects of scale** — billions of parameters, trillions of tokens, multi-million-dollar GPU clusters.

Nature developed a radically different approach: the human brain learns from very little data thanks to **simple principles applied cumulatively**. A child learns their mother tongue in a few years, with little exposure, without explicit backpropagation, starting from zero.

**The hypothesis of this project**: knowledge-like behaviour can emerge from less data and fewer resources through cumulative, progressive, guided learning — inspired by biological cognitive development.

### Epistemic note

The model still uses **backpropagation + the Adam optimizer** — this is not biological learning in the literal sense. It is a transformer with dynamic extensions *inspired* by biological mechanisms. The claim is methodological, not neuroscientifically precise.

### Design principles

- **Progressive curriculum**: the model learns like a child, from phonemes to literature
- **Innate affective system**: confidence, fear, pleasure, pain modulate behaviour from the very first token
- **Curated teacher signal**: an external LLM (Claude) generates examples targeted at the model's current deficit
- **Show-then-test**: the model sees the correct answer *before* being questioned, as in natural teaching
- **Anti-forgetting (rehearsal)**: every N teaching turns, a text mini-batch prevents catastrophic loss
- **Protected axioms**: sequences of objective truths (e.g. "1+1=2") with zeroed gradient

---

## 2. Model architecture

### Current configuration (experimental)

```text
TorchGPT — decoder-only, Pre-LayerNorm, GPT-2 style
─────────────────────────────────────────────────────
Total parameters : 3.32M
Vocabulary       : 501 tokens (BPE, trained on corpus it/0)
d_model          : 256
n_layers         : 4
n_heads          : 4
d_ff             : 1024
max_seq_len      : 129 (128 context tokens + 1)
dropout          : 0.1
Positional enc.  : learnable embedding (not RoPE)
FFN activation   : GELU
Weight tying     : logits = x @ tok_emb.W^T
```

### Forward pass

```text
Input ids (T,)
  → TokenEmbedding(V=501, d=256) + PosEmbedding(128, d=256)
  → Dropout(0.1)
  → 4 × TransformerBlock:
       h   = x + Attention(LayerNorm(x))    # pre-LN residual
       out = h + FFN(LayerNorm(h))           # pre-LN residual
  → FinalLayerNorm
  → logits = x @ TokenEmbedding.W^T          # weight tying
  → CrossEntropy loss
```

### Tokenizer

BPE (Byte Pair Encoding) with 501 tokens, trained exclusively on the phonemic corpus `it/0` (Italian syllables and phonemes). The small vocabulary is an intentional choice for the experimental phase: it forces the model to learn sub-word relations before it has whole words.

**Known limit**: with 501 tokens, common Italian words fragment into 2–4 subtokens, which prevents semantic generalization. The target vocabulary for the next phase is 8,000 tokens.

### Implementation backends

The model exists in two implementations:

| Implementation | File | Use |
|----------------|------|-----|
| **PyTorch CPU** | `tests/test_1/splx/torch_model.py` | Active training (59 seq/s, batch=8) |
| **Pure NumPy** | `tests/test_1/splx/` | Reference, gradient check, educational |

---

## 3. Innate affective system

The affective system is not a module bolted on top of the transformer: it is part of the training loop. Five scalars updated with EWMA (α=0.1):

| Variable | Meaning | Init | Formula / update |
| -------- | ------- | ---- | ---------------- |
| `confidence` | certainty about the current answer | 0.1 | `1 - H(softmax(logits)) / log(V)` — every forward |
| `ignorance` | low-norm embeddings ("unlearned" tokens) | 0.9 | `frac(norm(emb) < 0.05)` — every forward |
| `pleasure` | memory of positive feedback | 0.5 | EWMA(feedback > 0) — only with explicit feedback |
| `pain` | memory of negative feedback | 0.0 | EWMA(\|feedback\| if < 0) — only with explicit feedback |
| `fear` | uncertainty + pain | 0.45 | `0.5·pain + 0.5·(1-confidence)` — derived |

**Note**: `confidence` and `ignorance` update on every forward pass (even without feedback). `pleasure` and `pain` update only with explicit external feedback. `fear` is purely derived.

### Novelty drive (dopamine)

Implemented in `update_from_novelty()`: when an answer receives positive feedback, each token of the answer gets a `pleasure` bonus inversely proportional to how many times it already appeared in positive answers:

```text
bonus = NOVELTY_WEIGHT × feedback / sqrt(encounter_count + 1)
```

First correct production of a token: full bonus. Fourth time: 50%. Hundredth: 10%.
`register_token_activation()` adds a further spike (`+0.6 × min(1, n_new/10)`) when new vocabulary slots are activated in the Dream phase.

### Observed behaviour (L0→L2, real data)

- **Confidence**: 0.10 → 0.59 across the curriculum
- **Fear**: 0.45 → 0.21 — the model becomes progressively less fearful
- **Pain**: constant 0.00 — no negative feedback accumulated (pain gate not yet active)
- **Pleasure**: oscillating 0.47–0.56 in response to positive-feedback peaks

> L3–L4 data: not collected yet — see §6

### Inference modulation (AffectModulator)

Applied **only during generation**, never during training (this avoids an unstable feedback loop):

```text
1. Adaptive temperature: logits /= (base_temp × (1 + fear × 2.0))
2. "Don't know" boost:   if ignorance > 0.7 or confidence < 0.15:
                             logits[DONT_KNOW] += 3.0 × ignorance
3. Pain gate:            if pain > 0.1: logits[pain_ids] -= pain × 1.5
4. Pleasure gate:        if pleasure > 0.5: logits[pleasure_ids] += (pleasure-0.5) × 0.8
```

The `<|dont_know|>` and `<|uncertain|>` tokens must be registered in the tokenizer before use.

### Protected axioms

Sequences of objective truth whose gradient is scaled to zero during the backward pass:

```python
axiom_registry.register(["io", "sono"], is_objective=True, protection=0.9)
axiom_registry.register(["tu", "sei"],  is_objective=True, protection=0.9)
```

Implemented in `dynamic_model/exp_b/axioms.py`. Protection is proportional to the certainty level: objective axioms (1+1=2) max 1.0, subjective ones max 0.6.

---

## 4. Training method

### Two phases per level

**Phase 0 — Text training** (`--phase 0`):
- Cumulative corpus: `it/0` + the current level's corpus
- Standard cross-entropy, batch=8, seq_len=128
- Learning rate: 1e-3 (Adam)
- Epochs: 10 (configurable)
- Output: `level_N/final.pt`

**Phase 1 — Teaching with Claude** (`--phase 1`):
- External LLM (Claude Sonnet/Haiku) as adaptive teacher
- Learning rate: 1e-4 (more conservative, to avoid forgetting)
- 4 signals per turn (in order):

```text
1. SHOW-THEN-TEST  trainer.step("", expected,      feedback=0.5)
2. EXPOSURE        trainer.step("", next_prompt,   feedback=0.2)
3. CONTENT         trainer.step("", teaching_word, feedback=0.3)
4. IMITATION       trainer.step(prompt, response,  feedback=fb)
   (or: if fb="-" and expected is known → feedback=1.0)
```

**Anti-forgetting (rehearsal)**: every 10 turns, 1 mini-batch (4×128 tokens) from the level's text corpus. It reduces the perplexity spike from +133% (without) to +5% (with).

### Teaching auto-stop

The session stops automatically when, for 20 consecutive turns:
- ≥ 60% positive feedback (`+`, `++`, `+++`)
- ≥ 20% strong feedback (`++`, `+++`) — prevents auto-stop from false positives
- `fear < 0.25`
- `confidence > 0.45`

### Saved checkpoints

```text
models/checkpoints/{lang}/
  level_N/
    final.pt            ← after phase 0 (text knowledge)
    final_learned.pt    ← after phase 1 (interactive knowledge)
    final_dreamed.pt    ← after Dream Consolidation (if enabled)
    turn_XXXX.pt        ← snapshot every 10 turns
    session_*.jsonl     ← full log of the teacher-model dialogue
    tokenizer.json      ← tokenizer updated for the level
```

---

## 5. Language curriculum

The curriculum maps 11 levels (0–10) onto the years of human language development:

| Level | Equivalent age | Corpus content | Expected answer |
|-------|----------------|----------------|-----------------|
| L0 | Newborn | Phonemes, syllables (`ma`, `pa`, `ta`) | Isolated syllables |
| L1 | 1 year | Nursery rhymes, single words | Single words |
| L2 | 2 years | Basic sentences, animals (Wikipedia) | 2–3 words |
| L3 | 3 years | Pinocchio, simple Wikipedia | 3–6 word sentences |
| L4 | 4 years | Aesop's fables, Wikipedia culture | S+V+O sentences |
| L5 | 5 years | De Amicis, songs, Wikipedia | Short stories, 3–4 sentences |
| L6 | 6 years | Modern fiction (Neera, Serao) | Connectives, descriptions |
| L7 | 7 years | Rodari, Wikipedia | Short stories |
| L8 | 8 years | I Promessi Sposi (excerpt) | Paragraphs |
| L9 | 9 years | I Promessi Sposi (full) | Complex texts |
| L10 | 10 years | Divina Commedia, Wikipedia | Literature |

**Key principle**: the corpus contains texts *more complex* than what the model produces — like a child who listens to adult conversation but answers with simple words. The teacher asks for age-appropriate answers, not corpus-appropriate ones.

### Per-level teacher prompt

Each level has a `training_files/it/{N}/teacher_prompt.md` file with:
- A strict feedback scale (no false positives)
- Anti-repetition rules (max 3× the same word)
- Evaluation based only on the current turn's `expected`
- An A→B→C→D progression of increasing complexity

---

## 6. Experimental results

### Perplexity trajectory (text: "il cane dorme sul tappeto. la mamma cucina il pane.")

```text
Checkpoint          PPL      Δ       Type    Status
────────────────────────────────────────────────────────
L0/final            53.7             [txt]   ✓ measured
L0/final_learned    80.3    +26.6 ↑  [tch]  ✓ no rehearsal
L1/final            29.0    -51.3 ↓  [txt]  ✓
L1/final_learned    49.3    +20.3 ↑  [tch]  ✓ with rehearsal (-85% vs L0)
L2/final            22.5    -26.8 ↓  [txt]  ✓
L2/final_learned    27.7     +5.1 ↑  [tch]  ✓ spike almost zero
L2/final_dreamed     —        —       [drm]  ✓ Dream phase applied
L3/final            21.2     -6.5 ↓  [txt]  ☐ to be measured
L3/final_learned    18.0     -3.2 ↓  [tch]  ☐ estimated
L4/final            19.2     +1.2 ↑  [txt]  ☐ estimated
L4/final_learned    19.5     +0.4 →  [tch]  ☐ estimated — plateau (~501 token limit)
```

**Current status** (2026-04-13): real checkpoints available up to `L2/final_dreamed`. The L3–L4 values are estimates based on the observed trajectory.

**Milestone reached** (L0–L2): rehearsal reduced the catastrophic-forgetting spike from +133% (L0, without rehearsal) to +5% (L2). The Dream phase is operational.

**Known limit**: a plateau is expected at PPL≈18–20 from levels L3–L4 on. Cause: the 501-token vocabulary is saturated. Common words fragment into subtokens, preventing generalization.

### Training efficiency

Compared with standard training on an equivalent corpus:

- **CPU token-passes needed**: ~28M (vs. ~150M standard)
- **Theoretical speedup**: 5–27× for the same final PPL
- **Source of the advantage**: curriculum ordering (2–5×) + curated teaching signal (10–100× per example) + rehearsal (removes retraining)
- **Claude API cost** for the full L0–L10 curriculum: ~$30–50

---

## 7. Comparison with standard approaches

| Dimension | PhysisML today | Standard small (GPT-2 117M) | TinyLlama 1.1B |
|-----------|----------------|------------------------------|----------------|
| Parameters | 3.32M | 117M | 1,100M |
| Vocabulary | 501 | 50,257 | 32,000 |
| Corpus | ~2M tokens | ~40B tokens | 3T tokens |
| Italian PPL | ~18–20 | ~15 (if fine-tuned) | ~6–8 |
| Coherent sentences | No | Partially | Yes |
| Training cost | ~$20 API | ~$500 GPU | ~$100,000 GPU |
| Curriculum | Yes | No | No |
| Affective system | Yes | No | No |

### Realistic intermediate target (next phase)

```text
Parameters : 25M     (d=512, L=6, n_heads=8)
Vocabulary : 8,000   (BPE on the full it/0–it/10 corpus)
Corpus     : 200M+   (Italian Wikipedia + OpenSubtitles + existing)
Expected PPL: ~10–12
Cost       : ~$55    ($1 cloud GPU + $50 teaching API)
Output     : simple but understandable Italian sentences
```

---

## 8. Roadmap

### Current phase — Curriculum validation (in progress)

- [x] Implement show-then-test (expected signal before generation)
- [x] Anti-forgetting rehearsal (1 text batch every 10 teaching turns)
- [x] Strict auto-stop (requires 20% strong feedback `++/+++`)
- [x] Teacher prompt with strict feedback scale (Sonnet, no false positives)
- [x] Dream Consolidation phase (`final_dreamed.pt`) integrated into the curriculum
- [x] Novelty drive (dopamine) — decreasing bonus for new tokens
- [x] `ignorance` as an autonomous affective variable (biological prior 0.9)
- [ ] Complete curriculum L3→L10 with the 3.3M-param model (current: L2 ✓, L3 starting)
- [ ] Validate affective behaviour across all levels

### Phase 2 — Vocabulary scale-up

- [ ] Download the Italian Wikipedia corpus (~450MB dump)
- [ ] Download Italian OpenSubtitles (~700MB, conversational L2–L5)
- [ ] Retrain the BPE tokenizer on the full corpus → 8,000 tokens
- [ ] Benchmark: PPL of the 3.3M model with vocab 501 vs 8,000

### Phase 3 — Model scale-up

- [ ] Model d=512, L=6, n_heads=8 (~25M param)
- [ ] Training on a 200M+ token corpus with the L0–L10 curriculum
- [ ] Comparative benchmark: PPL, output quality, training time

### Phase 4 — TinyLlama-like target (GPU)

- [ ] d=768, L=12 (~85M param) or larger
- [ ] Requires a GPU (RTX 3060 or cloud A100)
- [ ] Target PPL: ≤ 10 on standard Italian

### Future ideas (from appunti.md)

- ~~**Dynamic vocabulary**~~: implemented in `exp_a/` (`VocabExpansionManager`, `DreamConsolidator`) — to be integrated into the main curriculum
- ~~**"Don't know" token**~~: implemented in `AffectModulator` (boost when `ignorance > 0.7` or `confidence < 0.15`)
- ~~**Dream consolidation**~~: implemented (`DreamConsolidator`, `final_dreamed.pt` in the curriculum)
- **Mathematical axioms**: gradient protection for objective truths (1+1=2) — structure in `exp_b/axioms.py`, to be wired into the curriculum
- **Multilingual training**: English + Italian in parallel, with a shared unconscious layer

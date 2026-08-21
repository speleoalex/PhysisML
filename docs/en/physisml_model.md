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
Total parameters : 23.59M
Vocabulary       : 9,000 allocated slots, 8,002 active at start
                   → 8,079 after the L0→L10 curriculum
d_model          : 512
n_layers         : 6
n_heads          : 8
d_ff             : 2048
max_seq_len      : 129 (128 context tokens + 1)
dropout          : 0.1
Positional enc.  : learnable embedding (not RoPE)
FFN activation   : GELU
Weight tying     : logits = x @ tok_emb.W^T
Optimizer        : Adam, lr 1e-3 (text) / 2e-5 (teaching),
                   weight_decay = 0
```

### Forward pass

```text
Input ids (T,)
  → TokenEmbedding(V=9000, d=512) + PosEmbedding(128, d=512)
  → Dropout(0.1)
  → 4 × TransformerBlock:
       h   = x + Attention(LayerNorm(x))    # pre-LN residual
       out = h + FFN(LayerNorm(h))           # pre-LN residual
  → FinalLayerNorm
  → logits = x @ TokenEmbedding.W^T          # weight tying
  → CrossEntropy loss
```

### Tokenizer

BPE (Byte Pair Encoding) with 8,000 tokens, plus dormant slots up to 9,000. Dormant tokens carry logit `-inf` and zero gradient; the dream phase activates them as new patterns consolidate, initialising each from its parent tokens at 30% of the mean norm of the already-trained rows. An absolute scale would give a fresh row a *larger* norm than the trained ones — a high prior with no semantics in the weight-tied softmax.

Across the L0→L10 curriculum the vocabulary grew from 8,002 to 8,079 tokens. Growth is deliberately conservative: merges are found and applied within word boundaries only (matching `encode`, otherwise the token is unreachable), degenerate repetitions and multi-word phrases are rejected, and the level's active drill targets are protected — tokenising what is currently being taught orphans the multi-token path the model has already learned.

**Note on `weight_decay`**: it must stay at zero. With `torch.optim.Adam` the decay is coupled to the gradient, so a rarely-exercised parameter shrinks at every step regardless of its gradient. Over the hundreds of thousands of single-sample steps of the curriculum this kills the network: measured on the May checkpoints, the `ln_f` gain fell from 0.87 (L0) to 0.0079 (L10).

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

### Metric

**Exact match** against every curriculum target of the level: the model's
normalised answer must equal the one the teacher expects. Decoding is **greedy**
(`top_k=1`), so the figure is reproducible — at temperature 0.8 the same
checkpoint scored 92% and 75% on consecutive runs. Questions are asked in the
**exact** form used in training, i.e. with the step's `prompt_template` applied
(`di: {prompt}` for most steps): probing with the bare target asks a question the
model has never seen — `il cane` returns `mangia il pane!` while `di: il cane`
returns `il cane!`.

Command: `python3 dynamic_model/test_model.py --level N --samples 0`

### Per-level results

Post-dream checkpoints, exact match over all targets:

| Level | Exact match | Targets | Sessions | Notes |
|-------|------------|---------|----------|-------|
| L0 | 100% | 21/21 | 3 | |
| L1 | 96% | 23/24 | 3 | |
| L2 | 100% | 18/18 | 3 | |
| L3 | 82% | 23/28 | 2 | numbers and colours share a prompt prefix |
| L4 | 100% | 25/25 | 10 | the level where the stall used to begin (0% in May) |
| L5 | 95% | 19/20 | 6 | |
| L6 | 100% | 21/21 | 6 | |
| L7–L10 | — | — | — | rebuild in progress |

Against the May build, before the fixes: L0 4.4%, L1 1.8%, L2 12.8%, L3 1.0%,
**L4 and beyond 0.0%**.

New levels start low and climb monotonically — L4 from 0.07 to 0.79 quality rate
over ten sessions, L5 from 0.05 to 0.82, L6 from 0.08 to 0.94. That is the
signature of consolidation: the deterministic regime learns slowly but
accumulates, instead of oscillating.

### Effect of the dream phase

Exact-match delta between the pre-dream checkpoint (`final_learned.pt`) and the
post-dream one (`final_dreamed.pt`): **+19, +13, +17, +7, +12 points** from L0 to L4.

The dream used to hurt the higher levels as long as its deep-sleep replay (N3)
stayed dominated by older material: the memory bank spans every level, and the
current level's share fell to 5% at L7 and 7% at L10 against 41% at L4. N3 and
REM are the last phases the dream trains, so that ratio decides what the model
keeps. N3 now keeps every memory of the current level and samples the older ones
up to an equal share.

### Real question-and-answer examples

Generated greedily from the post-dream checkpoints. Failures are included:

**L0** — 21/21 correct

| Question | Model answer | |
|---|---|---|
| `di ma` | `ma!` | ✓ |
| `di pa` | `pa!` | ✓ |
| `di ta` | `ta!` | ✓ |

**L1** — 23/24 correct

| Question | Model answer | |
|---|---|---|
| `di: la casa` | `la casa!` | ✓ |
| `di: il pane` | `il pane!` | ✓ |
| `di: la mamma` | `la mamma!` | ✓ |
| `di: il cane` | `il pane!` | ✗ expected `il cane!` |

**L2** — 18/18 correct

| Question | Model answer | |
|---|---|---|
| `di: il cane` | `il cane!` | ✓ |
| `di: la casa` | `la casa!` | ✓ |
| `di: il pane` | `il pane!` | ✓ |

**L3** — 23/28 correct

| Question | Model answer | |
|---|---|---|
| `di: il cane` | `il cane!` | ✓ |
| `di: la mamma` | `la mamma!` | ✓ |
| `di: il gatto` | `il gatto!` | ✓ |
| `di: il sole brilla` | `il sole il gatto il sole bello!` | ✗ expected `il sole brilla!` |
| `di un numero: tre` | `due!` | ✗ expected `tre!` |

**L4** — 25/25 correct

| Question | Model answer | |
|---|---|---|
| `di: cosa mangia il cane?` | `il cane mangia il pane.` | ✓ |
| `di: cosa mangia il gatto?` | `il gatto mangia il pesce.` | ✓ |
| `di: cosa beve il bambino?` | `il bambino beve il latte.` | ✓ |

**L5** — 19/20 correct

| Question | Model answer | |
|---|---|---|
| `com è il cane?` | `il cane è grande.` | ✓ |
| `com è la casa?` | `la casa è piccola.` | ✓ |
| `com è il sole?` | `il sole è caldo.` | ✓ |
| `com è il pane?` | `il cane è grande.` | ✗ expected `il pane è buono.` |

**L6** — 21/21 correct

| Question | Model answer | |
|---|---|---|
| `cosa ha mangiato il cane?` | `il cane ha mangiato il pane.` | ✓ |
| `cosa ha bevuto il bambino?` | `il bambino ha bevuto il latte.` | ✓ |
| `cosa ha letto il papà?` | `il papà ha letto il libro.` | ✓ |

The remaining errors are all of one family: **the model does not discriminate the
final word of the prompt when several targets share the same prefix**.
`di un numero: tre` and `di un colore: rosso` differ only in the last token and
both collapse onto `due!`. This is neither forgetting nor a capacity limit — pure
SFT on a level's targets reaches 100% in 30 epochs — but collapse onto one answer
per prompt family. The lever is more distinct targets per step and more varied
question heads.

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
| Parameters | 23.6M | 117M | 1,100M |
| Vocabulary | 8,079 | 50,257 | 32,000 |
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

### Current phase — Curriculum validated through L6

- [x] Dream consolidation phase integrated into the curriculum
- [x] Novelty drive (dopamine) — decaying bonus for new tokens
- [x] `ignorance` as an autonomous affective variable (biological prior 0.9)
- [x] **Test-then-show** teaching: the model answers before it sees the solution.
      The reverse order tested it on the question it had just been given the
      answer to, so the grade — and the build's quality gate with it — measured
      primed recall rather than retained knowledge (31% in-session vs 8% offline
      at L3).
- [x] **Interleaved** rehearsal on gold pairs, alongside the text rehearsal
- [x] Deterministic `local_teacher.json` for **all** 11 levels: a closed pool of
      repeated targets. The LLM teacher produced a nearly-new prompt every turn
      (0.94–0.99 distinct per turn against 0.03–0.10 with the local one), i.e.
      one gradient step per target
- [x] Dream reordered: the corpus replay no longer speaks last, and N3 is
      weighted toward the current level
- [x] `weight_decay = 0` — coupled decay under Adam was killing the network
- [x] A real quality gate in `build.sh`: below threshold the build stops instead
      of training later levels on absent foundations
- [x] Curriculum L0→L6 validated (exact match 82–100%)
- [ ] Finish validating L7→L10 (rebuild in progress)
- [ ] More distinct targets per step: the remaining errors are collapses onto
      prompts that share a prefix
- [ ] Validate affective behaviour across all levels

### The training regime to preserve

Five elements, each necessary and experimentally verified. Removing any one
brings the stall back:

1. Deterministic teacher with a closed pool of repeated targets
2. Test-then-show evaluation — never ask what you have just taught
3. Interleaved rehearsal on gold pairs
4. The dream ends with supervision, not with the corpus
5. Every target verified reachable at `+++`
   (`scripts/validate_teacher_configs.py`)

**Cross-cutting rule.** Any phase that learns from the model's *output* instead
of the gold target degenerates: it happened to the dream's pattern mining (which
reinforced babble), to vocabulary growth (48-character mega-tokens), to
imitation reinforcement, and to training on the teacher's prompt. Corollary: the
phase that closes training decides what the model remembers, so it must be the
most supervised one, not the most generic.

### Phase 2 — Vocabulary scale-up

- [ ] Download the Italian Wikipedia corpus (~450MB dump)
- [ ] Download Italian OpenSubtitles (~700MB, conversational L2–L5)
- [ ] Retrain the BPE tokenizer on the full corpus → 8,000 tokens
- [x] Vocabulary scaled to 8,000 tokens with dormant slots (done: 8,079 active after L10)

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

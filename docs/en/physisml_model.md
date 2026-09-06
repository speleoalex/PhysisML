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
                   → 8,083 after the L0→L10 curriculum
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

Across the L0→L10 curriculum the vocabulary grew from 8,002 to 8,083 tokens. Growth is deliberately conservative: merges are found and applied within word boundaries only (matching `encode`, otherwise the token is unreachable), degenerate repetitions and multi-word phrases are rejected, and the level's active drill targets are protected — tokenising what is currently being taught orphans the multi-token path the model has already learned.

**Note on `weight_decay`**: it must stay at zero. With `torch.optim.Adam` the decay is coupled to the gradient, so a rarely-exercised parameter shrinks at every step regardless of its gradient. Over the hundreds of thousands of single-sample steps of the curriculum this kills the network: measured on the May checkpoints, the `ln_f` gain fell from 0.87 (L0) to 0.0079 (L10).

### Implementation backends

The model exists in two implementations:

| Implementation | File | Use |
|----------------|------|-----|
| **PyTorch CPU** | `tests/test_1/physisml/torch_model.py` | Active training (59 seq/s, batch=8) |
| **Pure NumPy** | `tests/test_1/physisml/` | Reference, gradient check, educational |

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

Protection is applied per embedding row, so the axiom words belong to the language being trained: `PHONETIC_AXIOMS` and `GRAMMAR_AXIOMS` in `train_curriculum.py` hold one list per language (`mamma/papà/sì/no` and the copula paradigm for Italian, `mama/papa/yes/no` and `I am / you are / he is / it is` for English). An axiom written in the wrong language is not inert: on the English vocabulary `mamma` encodes to `m|am|ma` and freezes three arbitrary subwords. `add_axiom` prints the pieces next to the ids so a split axiom is visible, and drops whitespace-only tokens — the space is its own token and carries about a third of either corpus, which protection 0.9 used to freeze as a side effect of a statement about `io sono`.

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

### A language is a folder

The curriculum above is Italian, and nothing in it is written in Python. A
second language is `training_files/<lang>/` with the same shape — one numbered
folder per level, `local_teacher.json`, `qa_pairs.jsonl`, a level text — plus
`training_files/<lang>/language.json` for the handful of things that are words
rather than structure: the axioms, the function words, the yes/no spellings,
the tutor fallback prompt, the Hub repo. `dynamic_model/language.py` reads it;
everything else (vocabulary, probe, card, export folder) follows a naming
convention and needs no declaration.

The English curriculum currently covers levels 0–10:

| Level | Structure taught | Example target |
|-------|------------------|----------------|
| L0 | Isolated and doubled syllables | `say ma` → `ma!` |
| L1 | Article + noun, first words, its own name | `say: the cat` → `the cat!` |
| L2 | Article + noun + verb, adjective, S+V+O | `say: the cat sleeps` → `the cat sleeps!` |
| L3 | Subject + verb, `what does X do?`, numbers | `what does the cat do?` → `the cat sleeps.` |
| L4 | `who` and `where` questions, first/then | `where does the cat sleep?` → `the cat sleeps in the house.` |
| L5 | Connectives and / but / because, descriptions | `why does the cat eat?` → `the cat eats because it is hungry.` |
| L6 | The past tense, causes, two linked sentences | `what did the boy eat?` → `the boy ate the bread.` |
| L7 | The future, and the three tenses side by side | `what will the boy eat tomorrow?` → `tomorrow the boy will eat the bread.` |
| L8 | Comparatives, preferences and their reasons | `who is bigger, the dog or the cat?` → `the dog is bigger than the cat.` |
| L9 | A thesis with a reason and a conclusion | `is the cat small?` → `I think the cat is small because it is fast.` |
| L10 | A short comment, a motivated judgement | `comment on the cat` → `the cat is fast, and this is nice.` |

Why this is a manifest and not a table in the source: a `dict` keyed by
language code inside a `.py` file is a list of the languages that module knows
about. It is complete the day it is written and silently wrong the day someone
adds a language — and the failure is not loud. The first English build ran for
six hours protecting the Italian axiom `mamma`, which the English vocabulary
encodes as `m|am|ma`: three arbitrary subwords frozen at protection 0.7, and no
error anywhere. `tests/test_language_manifest.py` now walks the sources looking
for such tables, and checks that every axiom is a whole token of its own
language's vocabulary and actually occurs in its own corpus.

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
| L0 | 100% | 21/21 | 3 |  |
| L1 | 96% | 23/24 | 3 |  |
| L2 | 100% | 18/18 | 3 |  |
| L3 | 82% | 23/28 | 2 | numbers and colours share a prompt prefix |
| L4 | 100% | 25/25 | 10 | the level where the stall used to begin (0% in May) |
| L5 | 95% | 19/20 | 2 |  |
| L6 | 100% | 21/21 | 1 |  |
| L7 | 100% | 19/19 | 1 |  |
| L8 | 100% | 19/19 | 2 |  |
| L9 | 88% | 14/16 | 1 | two errors: the opening is repeated |
| L10 | 94% | 15/16 | 1 | one error: the opening is repeated |

**Mean across the 11 levels: 96%.**

Against the May build, before the fixes: L0 4.4%, L1 1.8%, L2 12.8%, L3 1.0%,
**L4 and beyond 0.0%**.

The number of sessions needed collapsed once the last fixes were in: L4 took ten
before the sanity check and the dream reordering were repaired, while L6, L7, L9
and L10 clear the gate on the **first** session. Fewer sessions does not mean
less learning — it means the signal is no longer being wasted.

### Effect of the dream phase

Exact-match delta between the pre-dream checkpoint (`final_learned.pt`) and the
post-dream one (`final_dreamed.pt`), per level:

| L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|----|----|----|----|----|----|----|----|----|----|-----|
| +19 | +13 | +17 | +7 | +12 | −5 | +29 | **+53** | 0 | **+38** | **+38** |

The dream now helps at ten levels out of eleven. Before N3 was rebalanced it hurt
the higher levels systematically (L7 −26, L10 −12, L9 −6): the memory bank spans
every level and the current level's share fell to 5% at L7 and 7% at L10, against
41% at L4. N3 and REM are the last phases the dream trains, so that ratio decides
what the model keeps. N3 now keeps every memory of the current level and samples
the older ones up to an equal share, and those same levels gain +53, +38 and +38
points.

### Retention: the diagonal is not the capability

The numbers above are the **diagonal** of a matrix: each level scored on its own
checkpoint. They answer "did the level train?", not "does the model still know
it?". Those are two questions with very different answers.

`scripts/retention_matrix.py` scores every checkpoint against every level. On
the reference build:

```
        target →
ckpt ↓    L0    L1    L2    L3    L4    L5    L6    L7    L8    L9   L10
L0      100%
L1       10%   96%
L2       14%   50%  100%
L3       48%   54%   78%   82%
L4      100%   83%  100%   71%  100%          <- the only row that retains
L5      100%   42%   22%   18%   16%   95%
L10     100%   21%    0%   11%    4%    0%    0%    0%    0%    0%   94%
```

Diagonal 96%, **final row 20%**. The L10 checkpoint can do L10 and L0; the rest
is buried. The diagonal is always 82-100%, so the problem is not learning, it
is retention.

The one row that retains is L4, and L4 is the only level that needed ten
teaching sessions. Every session ends in a dream, and the dream's N1 replays
*every* level's `qa_corpus`. The other levels ran one to three. This was not a
property of the architecture; it was luck.

#### Intervention 1 — consolidation cycles (`MIN_DREAMS`)

Extra dreams on the finished L10 checkpoint, with no new teaching at all
(`./scripts/experiment_extra_dreams.sh --confirm`):

| dreams | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|--------|---|---|---|---|---|----|----|
| exact across all levels | 20% | 27% | 36% | 43% | 44% | 48% | **48%** |
| answers with repetition | 37% | 25% | 19% | 17% | 18% | 15% | **18%** |

+3.6 points per dream from 1 to 6, +1.0 from 7 to 12 — that second slope is
below the noise measured between two identical runs (2.2 points), so it
saturates between the sixth and the tenth. `build.sh` now tops every level up
to `MIN_DREAMS=6` regardless of when the gate passes.

But the ceiling is **48%, not 96%**: consolidation recovers half the gap.

#### Intervention 2 — rehearsal scope: an attempt that did not hold

The second half is not recoverable by dreaming. The dream replays the corpus
(N1) and the memory bank (N3) of every level, but the **interleaved rehearsal**
— the channel that actually built the prompt→answer associations — loads only
the current level's `qa_pairs.jsonl`. Extending it looked like the natural
candidate.

`--rehearsal-scope {level,all,balanced}`, where `balanced` is the union of all
levels with half of each replay reserved for the current one. Three arms one
flag apart, L0→L3, session and dream budget held constant
(`./scripts/experiment_rehearsal.sh --confirm`).

**The result does not hold across seeds.** Final row, `balanced` minus `level`:

| target | seed 1 | seed 2 |
|--------|-------:|-------:|
| L0 | +10 | **−19** |
| L1 | +21 | **−4** |
| L2 | +11 | +17 |
| L3 | +7 | +4 |
| **aggregate** | **+12** | **−1** |

At the first seed `balanced` won in every cell; at the second it loses on the
oldest levels — precisely the ones the intervention was meant to protect. **The
effect it was built for does not reproduce**, and the default is back to
`level`.

The first seed read as confirmation because +12 cleared the "2.2-point noise
floor" — but that figure measures the reproducibility of the **same** dream
re-run, not the variance between seeds, which on this metric is around 15
points. Two different quantities, and conflating them made a null result look
convincing for a day.

**One thing does reproduce**, and it is not the one being looked for:
`balanced` raises the **diagonal** at L2 (94% → 100%) and L3 (93/96% → 100%) in
both seeds. That is the current level improving, not earlier ones being
retained. It is a post-hoc observation: it deserves its own experiment rather
than being used to rescue this one.

`all` remains the worst of the three at the first seed, consistent with
dilution: at L3 the union is 535 pairs against 188 for the level itself.

#### Intervention 3 — the control: replay vs online EWC (exp_i)

The dream's cross-level channel is experience replay, and replay is the
oldest trick in the continual-learning book. The claim "the dream fixed
retention" is only interesting if a standard *regularization* method — one
that carries summary statistics instead of the corpus — cannot do the same.
`exp_i` is that control: same network, same curriculum, same harness,
L0→L6, two seeds, three arms one flag apart (`--anti-forgetting`):

- **`dream`** — the shipped mechanism: N1 replays every level's corpus, N3
  replays the episodic memory bank across levels.
- **`ewc`** — online EWC (Schwarz et al. 2018): a running diagonal Fisher
  `F ← γ·F_prev + F_new` (γ=0.95) with the anchor θ* refreshed at each level
  boundary, penalty `½·λ·Σ F·(θ−θ*)²` added in phases 1 and 2. The Fisher is
  the empirical one, estimated on the level's post-harvest gold pairs with
  the same prompt-masked loss the training uses (`scripts/compute_fisher.py`,
  sidecar `level_N/fisher.pt`). LayerNorm and positional embeddings are
  excluded from the anchor (the `ln_f` weight-decay lesson), the tied
  embedding is counted once, dormant vocabulary rows are free. In this arm
  the dream still runs, but N1 is restricted to the current level and N3 to
  the current level's memories: everything that consolidates the *current*
  level is kept, only the cross-level channel changes.
- **`none`** — identical gating with λ=0: the floor.

λ=1000 came from a preliminary sweep on L0→L2 (λ ∈ {100, 1000, 10000}:
final-row mean 33/37/41%, diagonal 78/76/73 — retention rises monotonically
with λ but never approaches replay, while current learning falls). Six
dreams per level, fixed in every arm, so consolidation budget is not a
confound. Reproduce: `MODE=sweep ./scripts/experiment_ewc.sh --confirm`,
then `MODE=main`; per-arm matrices land in `models/exp_i/`.

Result — mean across seeds (seed 1 / seed 2), run-to-run noise 2.2 points:

| arm | final row (retention) | diagonal (learning) |
|---|---|---|
| dream | **64.4%** (65.0 / 63.9) | 80.7% (82.5 / 79.0) |
| ewc | 13.0% (13.6 / 12.5) | 37.1% (35.0 / 39.2) |
| none | 22.0% (20.1 / 23.9) | 77.9% (76.4 / 79.4) |

Final row per level, both seeds:

```
            L0   L1   L2   L3   L4   L5   L6
dream_s1   100   58   44   57   72   75  100
dream_s2   100   59   42   52   71   76  100
ewc_s1      29   32   14   19    0    0   10
ewc_s2      43   29   10   12    0    0   20
none_s1    100   26    7    9    0    0   98
none_s2    100   30   13   16    1    0  100
```

Three verdicts, each replicated on both seeds. The replay channel alone is
worth +42.5 points of retention over `none` — whose ~20% replicates the
pre-dream builds and doubles as an internal validity check of the harness.
EWC lands 9 points *below* doing nothing and costs ~44 points of current
learning. The per-level rows say why that is remarkable: `none` shows
textbook catastrophic forgetting (keeps L0 and the just-learned L6 at ~100%,
loses the middle), while `ewc` loses even L0 and even L6 — the anchor damages
the model's ability to learn the level it is currently being taught.

**Why EWC collapses here.** The diagnosis went through two wrong hypotheses,
kept here in order because the elimination is what makes the survivor
credible:

1. *Fisher accumulation across anchors* — killed: with γ=0.95 the
   accumulation factor is bounded at ~2.9×, while the measured Fisher mass
   grows ~70× (0 → 421 → 5.8k → 20k → 29k across levels on seed 1).
2. *A feedback spiral from estimating the Fisher on a non-converged level*
   (penalty hurts convergence → gradients at estimation stay large → next
   anchor stiffer → …) — killed by a direct correlation test: the level-end
   losses on the very pairs used for estimation are ≈0 everywhere
   (0.0001–0.06) even in the collapsed arm, and Spearman ρ = −0.14 between
   final loss and new Fisher mass.
3. *The survivor:* at loss ≈ 0, `E[g²]` ≈ Var(g) — the empirical Fisher
   stops measuring curvature and measures the **disagreement between
   examples**. With a prompt-masked SFT loss and short answers, each pair
   pushes the shared tokens (separator, space, ':', articles, '!') in a
   different direction depending on its answer, so the variance concentrates
   exactly on the machinery every answer reuses: 20 embedding rows out of
   2,590 carry 89–93% of `F_new`'s mass, the space character alone 32–43%,
   plus 27–32% on the first attention block's `in_proj`. The anchor is
   **anti-selective** — it freezes the answer-production machinery, not the
   level's knowledge. Replicated on both seeds with the same concentration
   and the same top tokens at ~3× different absolute mass, so the damage
   tracks the concentration, not the scale. A related boundary case: after a
   *perfectly* converged level the Fisher is exactly zero (L0's mass is 0.0
   on both seeds — EWC is effectively off at L1), the other face of the same
   defect.

**Scope of the claim.** In this near-perfect per-level memorization
curriculum, our implementation of standard online EWC with an empirical
diagonal Fisher substantially underperforms experience replay — on both
retention and current learning — including against the unregularized
baseline; replay pays for it by keeping the corpus instead of statistics.
Not claimed: "EWC is wrong in general". Per-token Fisher normalization or
excluding structural tokens is a different algorithm family (Riemannian
Walk, Chaudhry et al. 2018); and the comparison is not compute-matched (N1
replays up to 7 levels per cycle against 1 — an "ewc + compute-matched N1"
arm is future work), so the relative efficiency of the two methods remains
open, though budget cannot explain ewc falling below `none`.

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


**L7** — future tense and tense contrast (19/19)

| Question | Model answer | |
|---|---|---|
| `cosa mangerà il cane domani?` | `domani il cane mangerà il pane.` | ✓ |
| `cosa berrà il bambino domani?` | `domani il bambino berrà il latte.` | ✓ |

**L8** — comparatives (19/19)

| Question | Model answer | |
|---|---|---|
| `chi è più grande, il cane o il gatto?` | `il cane è più grande del gatto.` | ✓ |
| `cosa è più caldo, il sole o la luna?` | `il sole è più caldo della luna.` | ✓ |

**L9** — thesis with a reason (14/16)

| Question | Model answer | |
|---|---|---|
| `il pane è buono?` | `secondo me il pane è buono perché è caldo.` | ✓ |
| `il cane è fedele?` | `secondo me il cane è fedele perché aspetta.` | ✓ |
| `perché ti piace il pane?` | `il pane il pane è caldo e buono, perciò mi piace.` | ✗ opening repeated |

**L10** — motivated commentary (15/16)

| Question | Model answer | |
|---|---|---|
| `commenta il sole` | `il sole è caldo, e questo è utile.` | ✓ |
| `commenta il libro` | `il libro insegna, e questo è importante.` | ✓ |
| `commenta il cane` | `il cane il cane il cane è fedele, e questo è bello.` | ✗ opening repeated |

The remaining errors fall into two contained families. First: **the model does
not discriminate the final word of the prompt when several targets share the same
prefix** — `di un numero: tre` and `di un colore: rosso` differ only in the last
token and both collapse onto `due!`. Second, at levels 9-10: it **repeats the
opening** of the answer (`il cane il cane il cane è fedele`) before completing it
correctly — it knows what to say but not always where to start. This is neither forgetting nor a capacity limit — pure
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
| Vocabulary | 8,083 | 50,257 | 32,000 |
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
- [x] Vocabulary scaled to 8,000 tokens with dormant slots (done: 8,083 active after L10)

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

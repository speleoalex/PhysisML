# FAQ — the five objections

*Read this in: [Italiano](../it/faq.md)*

These are the five things people say first when they read the results. All
five are reasonable. Four are partly or wholly right, and this page says so
rather than arguing.

---

## 1. "Exact match against the same curriculum's gold answers is memorization, not learning."

**Largely true, and it needs to be said before anyone else says it.**

The curriculum is, by construction, a near-perfect memorization regime. Each
level is a small set of question–answer pairs, trained to convergence with a
prompt-masked SFT loss. The 84% exact match on 1369 current targets means the
model reproduces the answers it was drilled on. It is not evidence of
generalization and was never offered as such.

That is not an accident to apologise for — it is the experimental condition.
The anti-forgetting benchmark (`exp_i`) is *interpretable* precisely because
each level converges: when a level's exact match falls from 97% to 13% after
training on later levels, no ambiguity is left about what happened. On a
regime with noisy per-level performance you could not read the retention
matrix at all.

Two measurements in this repository are **not** memorization, and they are the
ones to argue with:

- **The frozen probe.** 104 prompts, identical across builds, scored before
  and after: 84.6% → 90.4%. Fixed set, so it measures the build, not the
  target set — which grew from 849 to 1369 graded prompts in the same
  rebuild. It is still in-distribution, but it cannot be gamed by adding
  targets.
- **Held-out names the model was never taught.** `scripts/curiosity_rate.py
  --gate off`: 67% honest answers on unknown nouns against **0%** on known
  ones. There is no gold answer to memorize for a noun that appears nowhere
  in the curriculum. See objection 2 for how far this actually goes.

If you want to attack the results, attack those two. The exact-match table is
a convergence check, not a claim.

---

## 2. "The honesty doesn't really generalize."

**Correct, and the numbers here say it more sharply than the objection does.**

Levels 11–12 teach the model to answer *non lo so* — "I don't know" — about
nouns it was never taught, instead of inventing. The obvious question is
whether it learned the *relation* (never seen → admit) or just a longer list
of strings.

Measured twice, on independent builds, on the `mai-visto` group of
`curiosity_rate.py` (7 probe nouns × every phrasing = 35 prompts, greedy):

| build | honesty pool | honest answers |
|---|---|---|
| pre-rebuild | 6 nouns | 26% |
| 2026-09-01 | 38 nouns | **14%** |
| after the first autonomy run (2026-09-03) | 38 nouns + acquisition | **63%** |

Read the first two rows first. Growing the pool from 6 nouns to 38 made the
behaviour on *new* names worse, not better. Widening the curriculum moved
where the behaviour sits; it did not generalize it. That is a negative result
and it is recorded rather than buried.

The third row is not a rescue. Admission generalizes after the autonomy loop
— but **referent binding does not**. Shown an unknown object the model still
asks about the wrong noun (`questa è una lumaca` → `cos è una bussola?`) and
stays at 0% correct referent on unknowns. Half the mechanism works.

What makes this worth continuing rather than abandoning: the internal state
already separates known from unknown almost perfectly. The margin over the 10
classes gives AUC **0.9896** on the old checkpoint and **0.9874** on the new
one — replicated. The model knows what it does not know; the *string* is what
fails to attach to the referent. That is why the next step is an epistemic
trigger that reads the margin and names the referent, not more curriculum.

---

## 3. "The EWC comparison isn't compute-matched."

**True, and it is declared in the claim itself** — in the README, in the
technical document, and in the Zenodo abstract. The dream's N1 replays up to
7 levels per cycle against the `ewc` arm's one. The relative *efficiency* of
the two methods is open, and a compute-matched arm is listed as future work.

What the budget cannot explain is the shape of the result:

| arm | retention (final checkpoint, all levels) |
|---|---|
| `dream` | 64.4% |
| `none` (floor, λ=0) | 22.0% |
| `ewc` | **13.0%** |

`ewc` finishes **below the unregularized floor**, on both seeds. More compute
does not turn a method that is worse than doing nothing into a method that is
better than doing nothing — a penalty that actively hurts is not a penalty
that was merely underfunded. That is a result, not a caveat.

The mechanism is diagnosed rather than assumed: at loss ≈ 0 the empirical
diagonal Fisher measures inter-example variance, and under prompt-masked SFT
with short answers that variance concentrates on the tokens every example
shares. 20 embedding rows out of 2,590 carry 89–93% of the mass — the space
character alone 32–43%, then `:`, articles, `!` — plus 27–32% on the first
attention block. The anchor is **anti-selective**: it freezes the machinery
that produces *any* answer, not the knowledge of past levels.

The claim is scoped accordingly and is not "EWC is wrong in general".
Normalizing the Fisher per token, or excluding structural tokens, is a
different algorithm family (Riemannian Walk, Chaudhry et al. 2018).

---

## 4. "23.6M parameters. So what?"

The answer is not "it's good for its size". At this scale the model is not
competitive with anything, is not meant to be, and would lose to a 2019
baseline on any standard benchmark.

**The small scale is the experiment's condition, not a limitation it suffers
under.** Three things follow from it that do not survive at scale:

- **A full curriculum fits in a controlled setting.** Every token the model
  has ever seen is in this repository, versioned. There is no pretraining
  corpus of unknown composition underneath. When it answers `non lo so` about
  `falco`, it is verifiable that `falco` appears nowhere in its history —
  which is not verifiable for any model trained on a web crawl.
- **Levels converge, so forgetting is legible.** See objection 1. The
  retention matrix is readable because the diagonal is near 100%.
- **The whole benchmark reruns.** Three arms × two seeds × seven levels on a
  CPU. A negative result about EWC that costs a GPU-month to reproduce is a
  result nobody checks.

The question this repository asks is not "how well does a model perform" but
"what does a developmental curriculum plus a replay mechanism do to a
network". Scale is a confound for that question, not an ingredient.

---

## 5. "It's just nanoGPT with a curriculum."

**The architecture, yes — deliberately.** Decoder-only, pre-LayerNorm,
weight-tied LM head, 6 layers, d_model 512, 8 heads, byte-level BPE. Nothing
is novel there and nothing is meant to be. (The repository also carries a
pure-NumPy implementation with hand-written backward passes and numeric
gradient checks, but that is pedagogy, not a contribution.)

Standard is the point. If the transformer were unusual, every result would be
confounded by it: "does the dream beat EWC" would become "does the dream beat
EWC on this odd architecture". A boring network makes the curriculum and the
anti-forgetting mechanism the only moving parts.

What is measured, and what is not:

| not the contribution | the contribution |
|---|---|
| the transformer | the developmental curriculum, levels 0–12, versioned in full |
| the training loop | the dream: replay over every prior level, dreams counted by plateau rather than by constant |
| the tokenizer | the `exp_i` benchmark — dream vs online EWC vs floor, two seeds, with the Fisher-concentration diagnosis |
| — | the honesty relation and its measured failure to generalize (objection 2) |
| — | the autonomy loop: acquiring a noun retracts every curriculum shape that treated it as unknown, through a versioned ledger |

"nanoGPT with a curriculum" is an accurate description of the code. The
curriculum is the experiment.

---

*Something here wrong or incomplete? [Open an issue](https://github.com/speleoalex/PhysisML/issues/new?template=question_about_results.yml) — that is the most useful kind.*

"""
Curriculum training 0→1 with Claude as tutor agent.

Training is split into two phases:

  PHASE 0 — Autonomous (it/0: sounds, syllables, phonemes)
    The model learns the phonetic structure of Italian without feedback.
    Pure self-supervised learning on a text corpus.

  PHASE 1 — Claude-guided (it/1: basic grammar)
    A Claude agent evaluates each model response and provides feedback
    +1 (approval) / 0 (neutral) / -1 (disapproval).
    The model updates its weights based on the feedback.

Checkpoints saved in:
  dynamic_model/checkpoints/level_0/   ← after phase 0
  dynamic_model/checkpoints/level_1/   ← after each phase 1 step

Start:
  python3 dynamic_model/train_curriculum.py

  # Con chiave API esplicita:
  ANTHROPIC_API_KEY=sk-ant-... python3 dynamic_model/train_curriculum.py

  # Cambia modello tutor (default: claude-haiku-4-5 per economia):
  python3 dynamic_model/train_curriculum.py --tutor-model claude-opus-4-6

  # Solo fase 0 (senza interazione Claude):
  python3 dynamic_model/train_curriculum.py --phase 0

  # Solo fase 1 partendo da checkpoint:
  python3 dynamic_model/train_curriculum.py --phase 1 \
      --checkpoint dynamic_model/checkpoints/level_0/final.pt
"""
import sys, os, argparse, time, glob, json
_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, 'tests', 'test_1')
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Load .env from project root (if present) before any API client is created
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass  # dotenv not installed — rely on environment variables

import numpy as np
import torch
import anthropic

from splx.torch_model import TorchGPT, TorchAdamOptimizer
from splx.tokenizer   import BPETokenizer
from dynamic_model.exp_b.affect_state import AffectState
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.axioms       import AxiomRegistry
from dynamic_model.exp_b.trainer      import TrainerB

torch.set_num_threads(12)

# ---------------------------------------------------------------------------
# Percorsi
# ---------------------------------------------------------------------------

CKPT_BASE    = "models/checkpoints"   # language subfolder added at runtime: {CKPT_BASE}/{lang}/
# Tokenizer: prefer 8K if available (8001 tokens: 8000 BPE + EOS@8000).
# Fall back to base 503-token tokenizer if 8K not trained yet.
_TOK_8K   = "dynamic_model/data/tokenizer_8k.json"
_TOK_BASE = "dynamic_model/data/tokenizer_base.json"
TOKENIZER = _TOK_8K if os.path.exists(_TOK_8K) else _TOK_BASE
# Maximum vocabulary capacity: pre-allocates this many embedding slots.
# Slots above the active count are dormant (logit=-inf, grad=0).
# Grows toward MAX_VOCAB as the model learns new token patterns.
MAX_VOCAB    = 9000   # 8001 used (8K BPE + EOS@8000), 999 slots for future growth
DATA_IT0     = "training_files/it/0"
DATA_IT1     = "training_files/it/1"

# ---------------------------------------------------------------------------
# Teaching content extraction
# ---------------------------------------------------------------------------

import re as _re

def extract_teaching_content(prompt: str) -> str:
    """
    Extract the Italian content embedded in a teaching prompt.
    The teacher's words ARE training data — hearing them is passive exposure.

    Examples:
      "di' 'gatto'"           → "gatto"
      "ripeti: il cane"       → "il cane"
      "bravo! ora di' 'bello'" → "bello"
      "di': il cane dorme"    → "il cane dorme"
      "completa: il cane è"   → "il cane è"
    """
    # "di' 'X'" or "di' \"X\""
    m = _re.search(r"di['\s]+['\"]([^'\"]+)['\"]", prompt)
    if m:
        return m.group(1).strip()
    # "di': X" or "di' X" (no quotes)
    m = _re.search(r"di['\s]+:?\s*(.+)", prompt, _re.IGNORECASE)
    if m:
        content = m.group(1).strip().strip("'\"")
        if content:
            return content
    # "ripeti: X"
    m = _re.search(r"ripeti:\s*(.+)", prompt, _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # "completa: X"
    m = _re.search(r"completa:\s*(.+)", prompt, _re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""

# ---------------------------------------------------------------------------
# Teaching agent — Claude as active tutor, not just evaluator
# ---------------------------------------------------------------------------

def load_teacher_prompt(lang: str, level: int) -> str:
    """
    Load a custom teacher system prompt from training_files/{lang}/{level}/teacher_prompt.md.
    Returns the file content if found, empty string otherwise.
    """
    path = os.path.join("training_files", lang, str(level), "teacher_prompt.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def build_system_teacher(age: float) -> str:
    """
    Build the tutor system prompt adapted to the model's virtual age.

    age 0–1  : newborn — only sounds and syllables, no words yet
    age 1–2  : toddler — first words, single nouns and exclamations
    age 2–4  : early child — short phrases, article + noun, simple verbs
    age 4–7  : child — simple sentences, colours, numbers, questions
    age 7+   : school age — grammar rules, verb conjugation, full sentences
    """
    if age < 1:
        profile = (
            f"The model is like a newborn (age {age:.1f}). "
            "Use ONLY individual sounds and syllables: 'ma', 'pa', 'ta', 'sì', 'no', 'oh!'. "
            "No full words yet. Repeat the same sound 2-3 times before moving on."
        )
        steps = (
            "  Step A: isolated sounds  → \"di' 'ma'\"  \"di' 'pa'\"  \"di' 'ta'\"\n"
            "  Step B: sound pairs      → \"ripeti: ma-ma\"  \"ripeti: pa-pa\"\n"
            "  Step C: exclamations     → \"di' 'oh!'\"  \"di' 'sì'\"  \"di' 'no'\""
        )
    elif age < 2:
        profile = (
            f"The model is like a 1-year-old toddler (age {age:.1f}). "
            "Use ONLY single common nouns and family words: 'mamma', 'papà', 'cane', "
            "'gatto', 'pane', 'acqua'. One word at a time. Lots of repetition."
        )
        steps = (
            "  Step A: single nouns     → \"di' 'cane'\"  \"di' 'gatto'\"\n"
            "  Step B: family words     → \"di' 'mamma'\"  \"di' 'papà'\"\n"
            "  Step C: yes/no + greet   → \"di' 'sì'\"  \"di' 'ciao'\""
        )
    elif age < 4:
        profile = (
            f"The model is like a 2-3 year old child (age {age:.1f}). "
            "Use simple two-word combinations: article + noun, or noun + adjective. "
            "Examples: 'il cane', 'la casa', 'bello!', 'grande!'. Short and clear."
        )
        steps = (
            "  Step A: article + noun   → \"ripeti: il cane\"  \"ripeti: la casa\"\n"
            "  Step B: noun + adjective → \"il cane è ___\"  \"la casa è ___\"\n"
            "  Step C: simple verb      → \"il cane ___\"  (expected: 'dorme', 'mangia')"
        )
    elif age < 7:
        profile = (
            f"The model is like a {age:.0f}-year-old child. "
            "Use simple complete sentences: subject + verb (+ object). "
            "Introduce colours, numbers 1-5, greetings, and basic questions."
        )
        steps = (
            "  Step A: simple sentences → \"il cane dorme\"  \"il gatto mangia\"\n"
            "  Step B: colours/numbers  → \"di' il colore: rosso, blu, verde\"\n"
            "  Step C: questions        → \"completa: come ___ il cane?\"\n"
            "  Step D: short dialogue   → \"buongiorno! come stai?\""
        )
    else:
        profile = (
            f"The model is at school age (age {age:.0f}). "
            "Use proper grammar: verb conjugations, simple questions and answers, "
            "complete sentences. Introduce 'io sono', 'tu sei', 'dove vai', 'ho fame'."
        )
        steps = (
            "  Step A: verb conjugation → \"io sono ___, tu sei ___\"\n"
            "  Step B: questions        → \"dove vai?\"  \"quanti anni hai?\"\n"
            "  Step C: short dialogue   → full question-answer exchange\n"
            "  Step D: storytelling     → \"completa: c'era una volta ___\""
        )

    return f"""You are teaching Italian to a child-like AI.
{profile}

Your role is DUAL each turn:
  1. Evaluate the model's last response (was it good?)
  2. Decide and produce the next teaching prompt

TEACHING METHOD:
  - Keep prompts short and clear (max 8 words)
  - Repeat the same item 2-3 times if the model struggles
  - Celebrate success: start next_prompt with "bravo! " or "bene! "
  - If struggling, simplify: go back one step
  - Vary content within the age level — don't always use the same word

PROGRESSION for this age level:
{steps}

If the model gives a good response (+++/++), advance a step.
If it struggles (=/-), stay or go back.

Reply ONLY in this exact JSON format:
{{
  "feedback": "<one of: -, =, +, ++, +++>",
  "commento": "<brief evaluation in Italian, max 10 words>",
  "next_prompt": "<your next teaching prompt, max 10 words>",
  "expected": "<ideal model response to next_prompt, max 8 words>",
  "step": "<A, B, C, or D>"
}}

For the very FIRST turn, skip feedback/commento and just provide
next_prompt, expected, step. Always produce a next_prompt.

IMPORTANT — next_prompt formatting rules:
  - NO apostrophes, quotes or special characters: write  di mamma  not  di 'mamma'
  - NO punctuation inside words: write  cane  not  cane.  or  cane!
  - Spaces and simple exclamation at the end are OK: bravo! di mamma
  - The model learns from your exact words — keep them clean and simple"""

# Map 5-level feedback symbols to numeric values used by TrainerB
FEEDBACK_MAP = {"-": -1.0, "=": 0.0, "+": 0.3, "++": 0.6, "+++": 1.0}


# Keep the teacher's visible history SHORT: the history contains the model's
# broken outputs and the LLM teacher progressively imitates them (measured on
# the May build: degenerate expected answers went from 5% in the first 40
# turns to 100% after turn 80 at L8). A small window bounds the contamination.
MAX_HISTORY = 8   # keep last N messages to avoid context drift/degeneration

def teaching_turn(client: anthropic.Anthropic,
                  conversation: list,
                  last_prompt: str,
                  last_response: str,
                  tutor_model: str,
                  age: float = 1.0,
                  turn: int = 1,
                  first_turn: bool = False,
                  lang: str = "it",
                  level: int = 0) -> dict:
    """
    Single teaching turn: Claude evaluates the last response and decides the next prompt.
    Sliding window on conversation history prevents Claude from losing its role.
    Returns a dict with feedback, next_prompt, expected, step.
    """
    if first_turn:
        user_msg = "Start the lesson. The model is ready for its first word."
    else:
        # Periodic reminder so Claude never thinks the lesson is over and
        # never drifts toward the student's broken register.
        reminder = ""
        if turn % 10 == 0:
            reminder = ("\n\n[REMINDER: The lesson is NOT over. Always provide "
                        "next_prompt. Write next_prompt and expected in clean, "
                        "complete Italian (articles included) — NEVER imitate "
                        "the student's broken output. Re-ask targets the "
                        "student failed instead of inventing new ones.]")
        user_msg = (f"Last prompt given: {repr(last_prompt)}\n"
                    f"Model response:    {repr(last_response)}\n\n"
                    f"Evaluate and decide the next teaching prompt.{reminder}")

    conversation.append({"role": "user", "content": user_msg})

    # Sliding window: keep only the last MAX_HISTORY messages.
    # The Messages API requires the FIRST message to have role 'user'; at
    # this point the list has odd length (assistant reply not appended yet),
    # so an even-sized slice would start on an assistant message and the API
    # would reject the request with a 400. Drop leading non-user messages.
    window = conversation[-MAX_HISTORY:] if len(conversation) > MAX_HISTORY else conversation
    while window and window[0].get("role") != "user":
        window = window[1:]

    result = client.messages.create(
        model=tutor_model,
        max_tokens=300,
        system=load_teacher_prompt(lang, level) or build_system_teacher(age),
        messages=window,
    )

    raw_reply = result.content[0].text.strip()
    conversation.append({"role": "assistant", "content": raw_reply})

    try:
        raw = raw_reply
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        symbol = str(parsed.get("feedback", "=")).strip()
        parsed["feedback_symbol"] = symbol
        parsed["feedback"]        = FEEDBACK_MAP.get(symbol, 0.0)
        return parsed
    except Exception as e:
        return {
            "feedback": 0.0, "feedback_symbol": "=",
            "commento": f"Parse error: {e}",
            "next_prompt": "di' 'cane'",
            "expected": "cane",
            "step": "A",
        }

# ---------------------------------------------------------------------------
# Vocab/model synchronisation
# ---------------------------------------------------------------------------

def _sync_vocab_rows(model: "TorchGPT", tok: "BPETokenizer",
                     label: str = "") -> int:
    """
    Align model.active_vocab_size with the tokenizer, initialising newly
    exposed embedding rows from their BPE parents (same policy as the
    dream's activate_slots). A bare `active_vocab_size = len(tok)` would
    expose ZERO rows: with the weight-tied LM head a zero row scores
    logit 0 — above most trained logits — i.e. a high prior with no
    semantics. Leaving the model BEHIND the tokenizer is worse: tokens in
    [active, len(tok)) are masked to logit=-inf, and any training target
    in that window yields an infinite cross-entropy.

    Returns the number of rows activated.
    """
    needed = max(tok.vocab.keys()) + 1
    if model.active_vocab_size >= needed:
        return 0
    if needed > model.vocab_size:
        print(f"  ⚠ tokenizer needs {needed} slots but model allocates only "
              f"{model.vocab_size} — capping at {model.vocab_size}")
        needed = model.vocab_size
    parents = {nid: (a, b) for (a, b, nid) in tok.merges}
    start = model.active_vocab_size
    with torch.no_grad():
        W = model.tok_emb.weight
        d = W.shape[1]
        # Scale relative to the rows the model actually trained: absolute
        # constants (0.05x, randn*0.001) assume embedding norms ~1, but real
        # checkpoints sit around norm ~0.006 — a fixed-scale init would give
        # new rows a LARGER norm than trained ones, i.e. a high prior in the
        # weight-tied softmax. 0.3x the mean trained norm keeps them quiet.
        ref_norm = W[:start].norm(dim=1).mean().clamp_min(1e-6) if start > 0 \
                   else torch.tensor(0.02, device=W.device)
        target_norm = 0.3 * ref_norm
        for nid in range(start, needed):
            pa = parents.get(nid)
            if pa and pa[0] < nid and pa[1] < nid:
                # Parent direction + noise for differentiation
                vec = 0.7 * (W[pa[0]] + W[pa[1]]) / 2.0 \
                      + 0.3 * torch.randn(d, device=W.device) * 0.02
            else:
                # No parents (special token or byte) — random direction
                vec = torch.randn(d, device=W.device)
            W[nid] = vec * (target_norm / vec.norm().clamp_min(1e-8))
    n = needed - start
    model.active_vocab_size = needed
    if label:
        print(f"  {label}: +{n} dormant rows activated → active={needed}")
    return n


# ---------------------------------------------------------------------------
# Phase 0 — autonomous learning
# ---------------------------------------------------------------------------

def phase_0(args, ckpt_base: str) -> str:
    """
    Text-only training for level N.
    - Starts from level_(N-1)/final_learned.pt if it exists, else from scratch.
    - Saves to level_N/final.pt  (pure text knowledge, no Claude).
    """
    level = args.level
    print("\n" + "="*55)
    print(f"PHASE 0 — Text training  (level {level})")
    print("="*55)

    ckpt_dir = os.path.join(ckpt_base, f"level_{level}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load tokenizer: prefer level_(N-1)/tokenizer.json ONLY if it was created
    # in the CURRENT build (vocab must be >= base TOKENIZER vocab).
    # Stale tokenizers from previous builds (smaller vocab) are skipped.
    tok = BPETokenizer()
    tok.load(TOKENIZER)  # base/8K as safe default
    base_vocab_size = len(tok)

    if level > 0:
        prev_tok = os.path.join(ckpt_base, f"level_{level-1}", "tokenizer.json")
        if os.path.exists(prev_tok):
            _t = BPETokenizer(); _t.load(prev_tok)
            if len(_t) >= base_vocab_size:
                tok = _t
                print(f"Tokenizer aggiornato: {prev_tok}  (vocab={len(tok)})")
            else:
                print(f"  Skip stale tokenizer {prev_tok} (vocab={len(_t)} < base={base_vocab_size})")

    # Starting point priority:
    # 1. explicit --checkpoint argument
    # 2. previous level's final_dreamed.pt  (dream-consolidated — best base)
    # 3. previous level's final_learned.pt  (carries learned knowledge)
    # 4. scratch
    start = args.checkpoint
    if not start and level > 0:
        prev_dreamed = os.path.join(ckpt_base, f"level_{level-1}", "final_dreamed.pt")
        prev_learned = os.path.join(ckpt_base, f"level_{level-1}", "final_learned.pt")
        if os.path.exists(prev_dreamed):
            start = prev_dreamed
            print(f"Starting from: {start}  (previous level — dream-consolidated knowledge)")
        elif os.path.exists(prev_learned):
            start = prev_learned
            print(f"Starting from: {start}  (previous level — learned knowledge)")

    if start and os.path.exists(start):
        model = TorchGPT.load(start)
        # Ensure active_vocab_size matches current tokenizer
        # (tokenizer may have grown since the checkpoint was saved)
        _sync_vocab_rows(model, tok, label="phase_0 vocab sync")
    else:
        print("Nuovo modello da zero.")
        # Architecture: d=512, L=6, h=8 — optimal for Arc A370M GPU utilization
        # (23M params, 13× faster training than d=256 L=4 due to better GPU saturation)
        # active_vocab_size uses max(id)+1, consistent with _sync_vocab_rows
        # and the dream's N2-B contiguity check (len(tok) breaks on id gaps).
        model = TorchGPT(MAX_VOCAB, 512, 8, 6, 2048, 129, 0.1,
                         active_vocab_size=max(tok.vocab.keys()) + 1)

    opt     = TorchAdamOptimizer(model.parameters(), lr=1e-3)
    trainer = TrainerB(model, tok, opt)

    # Register core phonetic axioms
    for text in ["mamma", "papà", "sì", "no"]:
        trainer.add_axiom(text, is_objective=True, protection=0.7)

    # Load corpus for current level (level 0 = phonemes, level N = level-N text)
    # Always include level 0 as the phonemic foundation, plus the current level if different.
    corpus_parts = []
    corpus_dirs = [DATA_IT0]
    if level > 0:
        level_corpus_dir = os.path.join("training_files", args.lang, str(level))
        if os.path.isdir(level_corpus_dir):
            corpus_dirs.append(level_corpus_dir)

    parts = []
    exclude_qa_corpus = getattr(args, 'no_qa_corpus', False)
    # At L3+ the narrative corpora (opensubtitles, Manzoni, etc.) are too complex
    # and washout the prompt→content associations. Use only qa_corpus + the
    # previous-level consolidation texts (small suoni/dialoghi). This keeps
    # phase_0 focused on patterns the model can actually produce.
    narrative_exclude = level >= 3
    for corpus_dir in corpus_dirs:
        for fpath in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
            if "teacher_prompt" in fpath:
                continue
            fname = os.path.basename(fpath)
            if exclude_qa_corpus and fname == "qa_corpus.txt":
                print(f"  - {fname:35s}  (skipped: --no-qa-corpus)")
                continue
            # Skip large narrative files at L3+; keep qa_corpus and small texts
            if narrative_exclude and fname != "qa_corpus.txt":
                size = os.path.getsize(fpath)
                if size > 100 * 1024:   # > 100 KB → likely narrative
                    print(f"  - {fname:35s}  {size:>6,} chars (skipped: narrative at L{level})")
                    continue
            with open(fpath, encoding="utf-8") as f:
                parts.append(f.read())
            print(f"  + {fname:35s}  {len(parts[-1]):>6,} chars")
    text = "\n\n".join(parts)
    # Cap narrative corpus more aggressively at L3+ to prevent function-word flooding.
    # At L3+ the corpus is narrative prose where function words dominate; training 3
    # epochs on 20MB of this washes out the prompt→content associations from L2.
    # Smaller cap keeps qa_corpus influence relatively higher.
    MAX_PHASE0_CHARS = 5 * 1024 * 1024 if level >= 3 else 20 * 1024 * 1024
    if len(text) > MAX_PHASE0_CHARS:
        print(f"  Corpus: {len(text):,} chars  →  capped at {MAX_PHASE0_CHARS:,}"
              f" (L{level} narrative pressure guard)")
        text = text[:MAX_PHASE0_CHARS]
    else:
        print(f"  Corpus: {len(text):,} chars\n")

    import signal

    interrupted = [False]

    def _save_and_exit(sig=None, frame=None):
        interrupted[0] = True
        print("\n\n  Ctrl-C received — saving model...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _save_and_exit)

    # Pre-calculate total steps for progress percentage display
    n_tokens  = len(trainer.tokenizer.encode(text[:10000])) * len(text) // 10000  # fast estimate
    _batch_size_0 = 16
    steps_per_epoch = max(1, (n_tokens - 128) // 128 // _batch_size_0)
    total_steps = steps_per_epoch * args.epochs_0
    print(f"  Estimate: ~{steps_per_epoch:,} steps/epoch × {args.epochs_0} epochs = ~{total_steps:,} total steps\n")

    t0 = time.time()
    all_losses = []
    try:
        for epoch in range(1, args.epochs_0 + 1):
            losses = trainer.train_on_text(text, block_size=128,
                                            batch_size=_batch_size_0, log_every=50,
                                            total_steps_hint=total_steps)
            all_losses.extend(losses)
            avg = float(np.mean(losses[-20:])) if losses else float("nan")
            print(f"Phase 0 — Epoch {epoch}/{args.epochs_0}  "
                  f"loss={avg:.4f}  time={time.time()-t0:.0f}s")
    except KeyboardInterrupt:
        pass

    # Save as final.pt — even on Ctrl-C (model is useful at any point after convergence)
    final_path = os.path.join(ckpt_dir, "final.pt")
    model.save(final_path)
    status = "interrupted (Ctrl-C)" if interrupted[0] else "complete"
    print(f"\nPhase 0 {status}.")
    print(f"  → {final_path}  (text base level {level})")
    return final_path

# ---------------------------------------------------------------------------
# Phase 1 — teaching dialogue with Claude as active tutor
# ---------------------------------------------------------------------------

def phase_1(args, start_checkpoint: str, ckpt_base: str = "models/checkpoints/it") -> str:
    """
    Claude-guided teaching for level N.
    - Starts from level_N/final_learned.pt if it exists (continues learning),
      otherwise from start_checkpoint (first teaching session).
    - Always saves/overwrites level_N/final_learned.pt.
    - Never touches active.pt — use set_model.sh to update it manually.
    """
    level = args.level

    # ── Determine teacher: local / hybrid (local+LLM) / Claude API ──────────
    _lt_path = os.path.join("training_files", args.lang, str(level), "local_teacher.json")
    _has_local_config = os.path.exists(_lt_path)

    USE_LOCAL = False
    local_teacher = None

    if args.tutor_model == "local":
        USE_LOCAL = True
        from dynamic_model.local_teacher import LocalTeacher
        local_teacher = LocalTeacher(lang=args.lang, level=level)

    elif args.tutor_model in ("hybrid", "local-llm") or \
         (args.tutor_model == "auto" and _has_local_config):
        # Hybrid: LocalTeacher for prompts + Ollama for evaluation
        USE_LOCAL = True
        try:
            from dynamic_model.hybrid_teacher import HybridTeacher
            local_teacher = HybridTeacher(lang=args.lang, level=level)
        except Exception:
            from dynamic_model.local_teacher import LocalTeacher
            local_teacher = LocalTeacher(lang=args.lang, level=level)

    if USE_LOCAL:
        claude = None
        print("\n" + "="*55)
        print(f"PHASE 1 — LOCAL Teaching  (level {level})")
        print(f"  Teacher: {local_teacher}")
        print("="*55)
    else:
        local_teacher = None
        print("\n" + "="*55)
        print(f"PHASE 1 — Claude Teaching  (level {level})")
        print("="*55)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("\n⚠️  ANTHROPIC_API_KEY not found.")
            print("   Set the environment variable and restart:")
            print("   export ANTHROPIC_API_KEY=sk-ant-...")
            sys.exit(1)

        claude = anthropic.Anthropic()
    age_label = {
        (0, 1): "newborn (sounds)",
        (1, 2): "toddler (single words)",
        (2, 4): "early child (phrases)",
        (4, 7): "child (sentences)",
    }
    label = next((v for (lo, hi), v in age_label.items()
                  if lo <= args.age < hi), "school age (grammar)")
    print(f"Tutor model:   {args.tutor_model}")
    print(f"Virtual age:   {args.age:.1f}  ({label})")
    prompt_path = os.path.join("training_files", args.lang, str(level), "teacher_prompt.md")
    if os.path.exists(prompt_path):
        print(f"Teacher prompt: {prompt_path}  ← custom")
    else:
        print(f"Teacher prompt: generated from --age  (crea {prompt_path} per personalizzare)")
    print()

    ckpt_dir = os.path.join(ckpt_base, f"level_{level}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Determine the start checkpoint FIRST (its active_vocab_size drives the
    # tokenizer compatibility check below).
    #
    # Prefer the most recent state: phase_2 (dream) runs after every session
    # (build.sh) and writes final_dreamed.pt — same session knowledge PLUS
    # consolidation, with active_vocab_size in sync with the level tokenizer.
    # The old code always restarted from final_learned.pt, which silently
    # discarded every intra-level dream (N2.5 SFT included) AND left the model
    # desynced from the post-dream tokenizer: dream-added token ids were
    # dormant (logit=-inf), so encountering them as training targets produced
    # an infinite cross-entropy.
    learned_path = os.path.join(ckpt_dir, "final_learned.pt")
    dreamed_path = os.path.join(ckpt_dir, "final_dreamed.pt")

    def _mtime(p: str) -> float:
        return os.path.getmtime(p) if os.path.exists(p) else -1.0

    if getattr(args, "_explicit_checkpoint", False):
        # An explicit --checkpoint always wins over the automatic chain —
        # it is the operator's way to redo a level from a chosen state.
        # (args.checkpoint alone is not enough: main auto-fills it with
        # final.pt, which must NOT override the session continuation.)
        actual_start = start_checkpoint
        print(f"Starting from: {actual_start}  (explicit --checkpoint)")
    elif os.path.exists(dreamed_path) and _mtime(dreamed_path) >= _mtime(learned_path):
        actual_start = dreamed_path
        print(f"Continuing from: {actual_start}  (dream-consolidated)")
    elif os.path.exists(learned_path):
        actual_start = learned_path
        print(f"Continuing from: {actual_start}  (previous session)")
    else:
        # Neither learned nor dreamed exist (fresh level, or both deleted for
        # a manual level reset) — start from the given chain checkpoint.
        actual_start = start_checkpoint
        print(f"First session — starting from: {actual_start}")

    model = TorchGPT.load(actual_start)
    model_active = model.active_vocab_size

    # Load tokenizer priority:
    # 1. level_N/tokenizer.json (post-dream expanded, only if compatible with model)
    # 2. level_(N-1)/tokenizer.json
    # 3. TOKENIZER (tokenizer_8k.json if present)
    # IMPORTANT: stale tokenizers from previous builds (smaller vocab) must not
    # be used when the model was trained with a larger tokenizer (e.g. 8K).
    tok = BPETokenizer()
    tok.load(TOKENIZER)  # load base/8K first to get correct vocab size

    for tok_candidate in [
        os.path.join(ckpt_dir, "tokenizer.json"),
        os.path.join(ckpt_base, f"level_{level-1}", "tokenizer.json") if level > 0 else None,
    ]:
        if tok_candidate and os.path.exists(tok_candidate):
            _tok_test = BPETokenizer(); _tok_test.load(tok_candidate)
            # Only use this tokenizer if compatible (vocab >= model active tokens)
            if len(_tok_test) >= model_active:
                tok = _tok_test
                print(f"Tokenizer: {tok_candidate}  (vocab={len(tok)})")
                break
            else:
                print(f"  Skip {tok_candidate} (vocab={len(_tok_test)} < model active={model_active})")
    else:
        print(f"Tokenizer: {TOKENIZER}  (vocab={len(tok)})")

    # Fail fast: if every candidate was rejected, the fallback base tokenizer
    # is SMALLER than the model — ids the model was trained with would decode
    # as KeyError / desync. Better to stop than to corrupt the session.
    if len(tok) < model_active:
        raise RuntimeError(
            f"No tokenizer with vocab >= model active_vocab_size="
            f"{model_active} found (best available: {len(tok)}). "
            f"Restore level_N/tokenizer.json or restart from an earlier "
            f"checkpoint.")

    # Align model with the tokenizer: any token id the tokenizer can emit must
    # have an ACTIVE embedding row, otherwise it trains toward logit=-inf.
    _sync_vocab_rows(model, tok, label="phase_1 vocab sync")

    # Low LR for teaching: with large vocab (8K slots), higher LR causes
    # catastrophic forgetting. 2e-5 keeps the model stable across all levels.
    opt   = TorchAdamOptimizer(model.parameters(), lr=2e-5)

    affect  = AffectState()
    eos_id  = tok.get_special_id(tok.EOS_TOKEN) if hasattr(tok, 'get_special_id') else None
    mod     = AffectModulator(affect, eos_token_id=eos_id)
    axioms  = AxiomRegistry()
    trainer = TrainerB(model, tok, opt, affect, mod, axioms)

    # Core grammatical axioms
    for text, prot in [("io sono", 0.9), ("tu sei", 0.9),
                        ("lui è", 0.9), ("noi siamo", 0.9)]:
        trainer.add_axiom(text, is_objective=True, protection=prot)
    print()

    # Brief autonomous pre-training on the CURRENT level's corpus
    # (not hardcoded to it/1 — each level sees its own vocabulary before teaching)
    # IMPORTANT: apply the same narrative filter as phase_0 at L3+. Loading
    # opensubtitles (20MB) here would wash out L2's prompt→content associations
    # each session, which was the root cause of the L3 cliff.
    level_corpus_dir = os.path.join("training_files", args.lang, str(level))
    _narrative_exclude = level >= 3
    parts = []
    for fpath in sorted(glob.glob(os.path.join(level_corpus_dir, "*.txt"))):
        fname = os.path.basename(fpath)
        if "teacher_prompt" in fname:
            continue
        if _narrative_exclude and fname != "qa_corpus.txt":
            size = os.path.getsize(fpath)
            if size > 100 * 1024:
                continue   # skip large narrative files at L3+
        with open(fpath, encoding="utf-8") as f:
            parts.append(f.read())
    text_level = "\n\n".join(parts)
    if text_level:
        print(f"Autonomous pre-training on {args.lang}/{level}/ ({len(text_level):,} chars)...")
        pre_losses = trainer.train_on_text(text_level, block_size=128,
                                           batch_size=16, log_every=999)
        pre_avg = float(np.mean(pre_losses[-20:])) if pre_losses else float("nan")
        print(f"  Pre-training loss: {pre_avg:.4f}\n")
    else:
        print(f"  No corpus in {level_corpus_dir} — skip pre-training\n")

    # --- Rehearsal corpus: pre-tokenized blocks for anti-forgetting ─────────
    # Every REHEARSAL_EVERY teaching turns, 1 mini-batch from the text corpus
    # is injected to prevent catastrophic forgetting of the text distribution.
    REHEARSAL_EVERY = 10   # inject 1 text batch every N teaching turns
    REHEARSAL_BLOCK = 128
    REHEARSAL_BATCH = 4    # smaller than training batch to keep overhead low
    rehearsal_blocks = []
    if text_level:
        ids_corpus = np.array(trainer.tokenizer.encode(text_level), dtype=np.int32)
        starts = list(range(0, len(ids_corpus) - REHEARSAL_BLOCK, REHEARSAL_BLOCK))
        np.random.shuffle(starts)
        rehearsal_blocks = [ids_corpus[s:s + REHEARSAL_BLOCK + 1] for s in starts
                            if s + REHEARSAL_BLOCK + 1 <= len(ids_corpus)]
    rehearsal_cursor = 0   # cycles through rehearsal_blocks indefinitely

    def rehearsal_step() -> float:
        """Run one mini-batch of text rehearsal. Returns the loss."""
        nonlocal rehearsal_cursor
        if not rehearsal_blocks:
            return 0.0
        end = min(rehearsal_cursor + REHEARSAL_BATCH, len(rehearsal_blocks))
        seqs = rehearsal_blocks[rehearsal_cursor:end]
        rehearsal_cursor = end if end < len(rehearsal_blocks) else 0
        batch = torch.from_numpy(np.stack(seqs)).long().to(trainer.device)
        opt_inner = trainer.optimizer._opt
        opt_inner.zero_grad()
        logits = trainer.model.forward(batch)
        loss   = trainer.model.loss(logits, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        opt_inner.step()
        loss_val = float(loss.detach())
        # Guard against NaN/Inf (numerical overflow in degraded model state)
        if not (loss_val == loss_val) or loss_val > 1e6:
            return 0.0
        return loss_val

    # --- Teaching dialogue loop ---
    AUTO_MODE = (args.interactions == -1)
    # Auto-stop thresholds: sustained quality over a rolling window
    AUTO_WINDOW        = 20   # turns to evaluate
    AUTO_POS_THRESHOLD = 0.60  # ≥60% positive (+/++/+++) in last window
    AUTO_STRONG_MIN    = 0.20  # ≥20% must be strong (++/+++) — prevents false-positive spam
    AUTO_FEAR_MAX      = 0.25  # fear must be below this
    AUTO_CONF_MIN      = 0.45  # confidence must be above this
    AUTO_MIN_TURNS     = 50    # never stop before this many turns

    mode_label = "AUTO (Ctrl-C to stop)" if AUTO_MODE else f"{args.interactions} turns"
    print(f"Starting teaching dialogue ({mode_label})...\n")
    print(f"{'Turn':>5}  {'Stp'}  {'Claude says':25s}  {'Model responds':25s}  "
          f"{'FB*':>4}  {'Comment (* = feedback sulla risposta precedente)'}")
    print("-" * 100)

    # --- Session log (JSON Lines, one record per turn) ---
    session_ts   = time.strftime("%Y%m%d_%H%M%S")
    log_path     = os.path.join(ckpt_dir, f"session_{session_ts}.jsonl")
    log_file     = open(log_path, "w", encoding="utf-8")
    print(f"Session log: {log_path}\n")

    def log_turn(data: dict) -> None:
        log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        log_file.flush()

    conversation = []
    positive = negative = neutral = 0
    current_prompt   = ""
    current_response = ""
    recent_scores    = []   # rolling window of numeric feedback scores
    stop_reason      = "completed"
    _prev_expected   = ""   # expected from previous turn (for sanity check)

    def save_and_exit(sig=None, frame=None):
        """Graceful exit on Ctrl-C: save progress and report."""
        nonlocal stop_reason
        stop_reason = "interrupted (Ctrl-C)"
        raise KeyboardInterrupt

    import signal
    signal.signal(signal.SIGINT, save_and_exit)

    turn = 0
    try:
      while True:
        turn += 1
        if not AUTO_MODE and turn > args.interactions:
            break
        # Ask teacher: evaluate last response + give next teaching prompt
        if USE_LOCAL:
            result = local_teacher.turn(
                last_prompt   = current_prompt,
                last_response = current_response,
                turn          = turn,
            )
        else:
            result = teaching_turn(
                claude, conversation,
                last_prompt   = current_prompt,
                last_response = current_response,
                tutor_model   = args.tutor_model,
                age           = args.age,
                turn          = turn,
                first_turn    = (turn == 1),
                lang          = args.lang,
                level         = level,
            )

        symbol      = result.get("feedback_symbol", "=")
        fb          = result.get("feedback", 0.0)
        cmt         = result.get("commento", "")[:30]
        next_prompt = result.get("next_prompt", "di' 'cane'")
        expected    = result.get("expected", "")
        step_label  = result.get("step", "A")

        # ── Sanity check: grades must be proportional to real content coverage ─
        # Applied to ALL teacher types (local, hybrid, Claude). The old check
        # only downgraded ZERO-coverage positives (any() over keywords): one
        # matched word out of 25 still earned '+', which at L4+ made the grade
        # uncorrelated with correctness. Uses _prev_expected (the expected for
        # current_response); graded_* values are consumed later by SIGNAL 4.
        graded_expected = _prev_expected     # expected for current_response
        _graded_cov     = 0.0                # content coverage of graded response
        if turn > 1 and symbol in ("+++", "++", "+") and graded_expected and current_response:
            _STOP = {"il","la","lo","le","gli","i","un","una","di","del","della",
                     "dei","delle","degli","a","e","è","in","per","da","su","con",
                     "tra","che","non","si","ha","ho","hai","ai","al","alla","agli",
                     "alle","ma","anche","mi","ti","ci","vi","ne","già","più","no",
                     "sì","se","sua","suo","mia","mio","tu","io","lui","lei","noi"}
            _kws = [w.lower().strip("!.?,: ") for w in graded_expected.split()
                    if w.lower().strip("!.?,: ") not in _STOP
                    and len(w.strip("!.?,: ")) > 1]
            if _kws:
                _resp_low = current_response.lower()
                import re as _re
                def _kw_present(kw, text):
                    # Word-boundary match (primary) — prevents 'ba' inside 'labato'
                    if _re.search(r'\b' + _re.escape(kw) + r'\b', text):
                        return True
                    # Compact fallback only for words >= 4 chars
                    compact = _re.sub(r'\s+', '', text)
                    return len(kw) >= 4 and kw in compact
                # Anti-echo at L4+: a content word that also appears in the
                # PROMPT is no evidence of knowledge (97-99% of positive grades
                # at L4-L10 rewarded words parroted from the prompt) — unless
                # the task is explicitly imitative (the whole expected answer
                # is contained in the prompt, as at L0-L2 by design).
                _prompt_low = (current_prompt or "").lower()
                _exp_stripped = graded_expected.lower().strip("!.?, ")
                # Whole-phrase boundary match: a bare substring test marks
                # e.g. expected 'arte' as imitation when the prompt contains
                # 'parte', silently disabling the anti-echo.
                _is_imitation = bool(_exp_stripped) and bool(
                    _re.search(r"(?<!\w)" + _re.escape(_exp_stripped) + r"(?!\w)",
                               _prompt_low))
                if level >= 4 and not _is_imitation:
                    # Anti-echo: only content words NOT present in the prompt
                    # count as evidence. NO fallback to the full keyword list:
                    # when every expected word is already in the prompt (e.g.
                    # multiple-choice questions), a parroted answer proves
                    # nothing — grade it '=' rather than re-enabling the very
                    # echo loop this check exists to close.
                    # Boundary-only match against the prompt: the compact
                    # fallback of _kw_present is meant for degenerate MODEL
                    # output and would find 'arte' inside 'parte' in the
                    # teacher's clean prompt, wrongly zeroing the credit.
                    _scored_kws = [k for k in _kws
                                   if not _re.search(r"\b" + _re.escape(k) + r"\b",
                                                     _prompt_low)]
                else:
                    _scored_kws = _kws
                _hits = sum(1 for k in _scored_kws if _kw_present(k, _resp_low))
                _graded_cov = (_hits / len(_scored_kws)) if _scored_kws else 0.0
                # Downgrade to the highest grade the coverage supports:
                # +++ = full coverage, ++ = at least half, + = at least one hit.
                if not _scored_kws:
                    _earned = "="
                else:
                    _earned = ("+++" if _graded_cov >= 0.99 else
                               "++"  if _graded_cov >= 0.5  else
                               "+"   if _hits >= 1          else "=")
                _rank = {"=": 0, "+": 1, "++": 2, "+++": 3}
                if _rank[_earned] < _rank.get(symbol, 0):
                    symbol = _earned
                    fb     = FEEDBACK_MAP.get(symbol, 0.0)
                    cmt    = f"[sanity] cov {_hits}/{len(_scored_kws)}"

        _prev_expected = expected   # save for next turn's sanity check

        # Statistics (skip first turn — no previous feedback)
        if turn > 1:
            if symbol in ("+++", "++"): positive += 1; recent_scores.append(1.0)
            elif symbol == "+":         positive += 1; recent_scores.append(0.6)
            elif symbol == "-":         negative += 1; recent_scores.append(-1.0)
            else:                       neutral  += 1; recent_scores.append(0.0)
            # Keep rolling window
            if len(recent_scores) > AUTO_WINDOW:
                recent_scores.pop(0)

        # ── SIGNAL 1: show-then-test — correct answer first ──────────────────
        # Prime the model with the expected answer BEFORE asking it to respond.
        # Analogy: teacher says "Repeat: il cane dorme... now you try."
        # Always include EOS at the end of expected so the model learns to stop.
        #
        # Adaptive weights: higher levels need stronger content reinforcement
        # because the corpus statistical pressure toward function words is stronger.
        # At L3+, also repeat Signal 1 when the previous response was wrong (=/-),
        # like a patient teacher showing the answer again before asking.
        teaching_content = extract_teaching_content(next_prompt)
        _eos = tok.EOS_TOKEN if hasattr(tok, 'EOS_TOKEN') else ""
        _eos_id = tok.get_special_id(_eos) if _eos and hasattr(tok, 'get_special_id') else None

        # Adaptive feedback: 0.5 at L0, scales up to 0.8 at L5+
        _level_scale = min(level * 0.05, 0.3)   # 0.0 at L0, 0.15 at L3, 0.25 at L5
        _sig1_fb = round(0.5 + _level_scale, 2)   # 0.5→0.8
        _sig3_fb = round(0.3 + _level_scale, 2)   # 0.3→0.6

        if expected:
            expected_with_eos = expected + _eos if _eos_id is not None else expected
            # CONDITIONAL on prompt: train "prompt → expected" association.
            # Without the prompt context, the model learns 'il cane!' as a
            # high-probability sequence but NOT that it should be produced
            # when the prompt says 'di: il cane'. This is the root cause of
            # the "correct words, wrong association" symptom.
            trainer.step(next_prompt, expected_with_eos, feedback=_sig1_fb)
            # Also an unconditional pass at lower weight — helps the model
            # internalize the expected sequence itself as fluent Italian.
            trainer.step("", expected_with_eos, feedback=_sig1_fb * 0.5)
            # At L3+: repeat Signal 1 (conditional) when model was struggling.
            prev_fb = symbol if turn > 1 else "+"
            if level >= 3 and turn > 1 and prev_fb in ("=", "-"):
                trainer.step(next_prompt, expected_with_eos, feedback=_sig1_fb * 0.8)

        # ── SIGNAL 2: REMOVED ────────────────────────────────────────────────
        # It used to train the model to PRODUCE the teacher's raw prompt
        # (trainer.step("", next_prompt, 0.2)). With prompt masking the prompt
        # is conditioning context everywhere else, so this was the last channel
        # teaching the model to emit instruction scaffolding — and at L0 the
        # scaffolding is babble: with retry_prefix '{target} {target}.' (63% of
        # turns) it taught runs like 'mama mama', which is exactly the observed
        # step-B failure ('mama!' -> 'mamama!'). Passive exposure to the GOLD
        # answer is already provided by Signal 1b (unconditional, half weight).

        # ── SIGNAL 3: extracted content — CONDITIONAL on prompt ──────────────
        # Teaches "prompt → content_word" association.
        if teaching_content:
            trainer.step(next_prompt, teaching_content, feedback=_sig3_fb)

        # ── Model responds AFTER being primed with the correct answer ─────────
        # Generation budget scaled to the expected answer. The old fixed budget
        # (max_tokens=20, hard [:30] char cap) made the correct answer
        # physically impossible from L7 on — the majority of expected answers
        # exceeded 30 chars, the teacher graded a mutilated string, and the
        # truncated text was then fed back into training via SIGNAL 4.
        _exp_ids_n  = len(tok.encode(expected)) if expected else 0
        _gen_budget = int(max(24, min(2 * _exp_ids_n + 12, 120)))
        # Two independent gates (see TrainerB.generate):
        #  - _min_gen  : EOS floor — generous, it is what stops the model from
        #                answering with a bare '<|EOS|>' (43.8% of L0 turns).
        #  - _stop_aft : punctuation soft-stop floor — about (expected - 1), so
        #                a 2-3 token target can close on its own '!' instead of
        #                over-generating ('ma!' -> 'mamamama!').
        _min_gen    = max(4, min(_exp_ids_n, 40)) if _exp_ids_n else 4
        _stop_aft   = max(1, min(_exp_ids_n - 1, 40)) if _exp_ids_n else 2
        _char_cap   = max(60, 2 * len(expected) + 20) if expected else 60
        generated = trainer.generate(
            next_prompt, max_tokens=_gen_budget,
            base_temperature=0.7, top_k=20,
            min_tokens=_min_gen, stop_after=_stop_aft,
        )
        # generate() already strips leading noise via _clean_response(),
        # but re-strip here for safety and to apply the (generous) length cap.
        model_response = generated[len(next_prompt):].strip()[:_char_cap]

        # ── SIGNAL 4: imitation feedback ──────────────────────────────────────
        # Applied AFTER the model responds so we can compare response vs expected.
        # Skip signal on the very first turn (no previous response to evaluate).
        #
        # Targets use graded_expected — the expected for current_response.
        # The old code used `expected` here, which at this point refers to the
        # NEXT prompt: the corrective branch was teaching current_prompt → the
        # answer of a different question.
        #
        # Echo-training guard: reinforcing the model's own output is allowed
        # ONLY when the response is verified complete (_graded_cov ≈ 1).
        # At L4+ real correctness was 0%, so every positive grade used to
        # apply an imitation step on the model's own garbage — the main
        # self-poisoning loop. Partial matches now imitate the GOLD answer.
        if turn > 1:
            _graded_exp_eos = (graded_expected + _eos
                               if (graded_expected and _eos_id is not None)
                               else graded_expected)
            if symbol == "-" and graded_expected:
                # Complete failure but correct answer known → full imitation learning
                trainer.step(current_prompt, _graded_exp_eos, feedback=1.0)
            elif fb > 0:
                if _graded_cov >= 0.99:
                    # Verified complete → reinforce the model's own output.
                    # Append EOS if the response does not already end with it,
                    # so the echo also reinforces stopping.
                    _echo_target = current_response
                    if _eos_id is not None and _eos \
                            and not _echo_target.endswith(_eos):
                        _echo_target = _echo_target + _eos
                    trainer.step(current_prompt, _echo_target, feedback=fb)
                elif graded_expected:
                    # Partial match → imitate the gold answer instead
                    trainer.step(current_prompt, _graded_exp_eos, feedback=fb)

        # ── REHEARSAL: inject text batch every N turns ─────────────────────────
        # Prevents catastrophic forgetting of the text distribution while teaching.
        rehearsal_loss_str = ""
        if rehearsal_blocks and turn % REHEARSAL_EVERY == 0:
            r_loss = rehearsal_step()
            rehearsal_loss_str = f"  [R:{r_loss:.2f}]"

        # Log the exchange
        # NOTE: fb_disp is the feedback for current_response (previous turn's output),
        # shown here alongside the NEW prompt/response for compactness.
        # Read as: "this is what the teacher thought of the PREVIOUS response."
        fb_disp = symbol if turn > 1 else "  "
        content_hint = f"  ← [{teaching_content}]" if teaching_content else ""
        fb_tag = f"↑{fb_disp}" if turn > 1 else "   "
        # Rolling window: last 5 feedback symbols for at-a-glance quality trend
        _sym_map = {1.0: "+", 0.6: "+", 0.5: "+", 0.0: "=", -0.8: "-", -1.0: "-"}
        _roll = "".join(_sym_map.get(s, "?") for s in recent_scores[-5:]).rjust(5)
        print(f"{turn:5d}  [{step_label}]  "
              f"{repr(next_prompt):25s}  "
              f"{repr(model_response):25s}  "
              f"{fb_tag:>5}  {_roll}  {cmt}{content_hint}{rehearsal_loss_str}")

        current_prompt   = next_prompt
        current_response = model_response

        # Log this turn
        a = trainer.affect
        log_turn({
            "turn":      turn,
            "step":      step_label,
            "prompt":    next_prompt,
            "expected":  expected,
            "response":  model_response,
            "feedback":  fb_disp if turn > 1 else None,
            "comment":   cmt,
            "content":   teaching_content,
            "affect": {
                "confidence": round(a.confidence, 3),
                "fear":       round(a.fear,       3),
                "pleasure":   round(a.pleasure,   3),
                "pain":       round(a.pain,        3),
            },
            "stats": {"positive": positive, "negative": negative, "neutral": neutral},
        })

        # Affective state + checkpoint every 10 turns
        if turn % 10 == 0:
            a = trainer.affect
            # Auto-mode: check quality threshold
            auto_stop = False
            if AUTO_MODE and turn >= AUTO_MIN_TURNS and len(recent_scores) >= AUTO_WINDOW:
                pos_rate    = sum(1 for s in recent_scores if s > 0) / AUTO_WINDOW
                strong_rate = sum(1 for s in recent_scores if s >= 0.8) / AUTO_WINDOW  # ++ or +++
                if (pos_rate >= AUTO_POS_THRESHOLD and strong_rate >= AUTO_STRONG_MIN
                        and a.fear <= AUTO_FEAR_MAX and a.confidence >= AUTO_CONF_MIN):
                    auto_stop = True
                    stop_reason = (f"target reached ({pos_rate:.0%} positive, "
                                   f"{strong_rate:.0%} strong, fear={a.fear:.2f})")

            print(f"\n  [turn {turn}] affect: "
                  f"conf={a.confidence:.2f} ign={a.ignorance:.2f} "
                  f"pleas={a.pleasure:.2f} pain={a.pain:.2f} fear={a.fear:.2f}  |  "
                  f"✓={positive} ✗={negative} ={neutral}", end="")
            if AUTO_MODE and len(recent_scores) >= 10:
                pos_rate = sum(1 for s in recent_scores if s > 0) / len(recent_scores)
                print(f"  [last {len(recent_scores)}: {pos_rate:.0%} positive]", end="")
            print()
            print(f"  Last Claude prompt: {repr(current_prompt)}")
            print(f"  Last model output:  {repr(current_response)}\n")
            if not getattr(args, 'no_turn_ckpt', False):
                ckpt_path = os.path.join(ckpt_dir, f"turn_{turn:04d}.pt")
                model.save(ckpt_path)

            if auto_stop:
                print(f"\n  ✓ Auto-stop: {stop_reason}")
                break

    except KeyboardInterrupt:
        print(f"\n\n  Ctrl-C received — saving progress...")
    except Exception as e:
        # A teacher/API failure mid-session must not lose the whole session:
        # fall through to the final_learned.pt save below.
        import traceback
        stop_reason = f"error: {e}"
        print(f"\n\n  ⚠ Session aborted by error — saving progress anyway.")
        traceback.print_exc()
    finally:
        log_file.close()

    # Always save/overwrite final_learned.pt — even on Ctrl-C or error
    learned_path = os.path.join(ckpt_dir, "final_learned.pt")
    model.save(learned_path)

    evaluated = max(turn - 1, 1)
    print("\n" + "="*55)
    print(f"PHASE 1 — {stop_reason}.")
    print(f"  Turns executed:      {turn}")
    print(f"  Positive (+/++/+++): {positive} ({100*positive//evaluated}%)")
    print(f"  Negative (-):        {negative} ({100*negative//evaluated}%)")
    print(f"  Neutral (=):         {neutral} ({100*neutral//evaluated}%)")
    print(f"  Affect:              {trainer.affect}")
    print(f"  → {learned_path}  (learned knowledge level {level})")
    print(f"  Per testare: ./set_model.sh {learned_path}")
    print("="*55)

    return learned_path

# ---------------------------------------------------------------------------
# Phase 2 — Dream consolidation
# ---------------------------------------------------------------------------

def _load_memory_bank(ckpt_base: str, level: int) -> list:
    """
    Build a memory bank from all session logs up to `level`.
    Each entry: {"prompt", "expected", "response", "feedback", "weight", "level"}
    Weight = affective salience: +++ = 1.0, ++ = 0.8, + = 0.5, = = 0.0, - = -0.8
    Sorted by |weight| descending — most emotionally salient first.
    """
    FEEDBACK_WEIGHT = {"+++": 1.0, "++": 0.8, "+": 0.5, "=": 0.0, "-": -0.8}
    bank = []
    for lvl in range(level + 1):
        lvl_dir = os.path.join(ckpt_base, f"level_{lvl}")
        for log_path in sorted(glob.glob(os.path.join(lvl_dir, "session_*.jsonl"))):
            records = []
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            # The 'feedback' logged at row N grades the RESPONSE of row N-1,
            # so the grade for row i's (prompt, response) pair lives at row
            # i+1. Reading it from the same row (as before) attributed each
            # grade to the wrong exchange, corrupting the dream's salience.
            for i, rec in enumerate(records):
                fb = records[i + 1].get("feedback") if i + 1 < len(records) else None
                if fb not in FEEDBACK_WEIGHT:
                    continue
                prompt   = (rec.get("prompt") or "").strip()
                expected = (rec.get("expected") or rec.get("response") or "").strip()
                response = (rec.get("response") or "").strip()
                if not prompt:
                    continue
                bank.append({
                    "prompt":   prompt,
                    "expected": expected,
                    "response": response,
                    "feedback": fb,
                    "weight":   FEEDBACK_WEIGHT[fb],
                    "level":    lvl,
                })
    bank.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return bank


def _jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _update_qa_pairs_from_sessions(ckpt_base: str, level: int, lang: str,
                                    max_new: int = 40) -> int:
    """
    Extract unique (prompt, expected) pairs from the LATEST session log only
    and add any not already in qa_pairs.jsonl, up to max_new new pairs.

    Rules:
    - Only the most recent session log (just completed before this dream).
    - Positive turns (+++/++/+) are added first; neutral (=) fill remaining slots.
    - Hard cap: max_new pairs per dream call to prevent explosion.
    - Total qa_pairs cap: 300 entries (prevents N2.5 from being overwhelmed).
    - Uses `expected` from the teacher, NOT the model's response.

    Returns the number of new pairs added.
    """
    MAX_TOTAL_QA = 300   # hard cap on total qa_pairs size per level

    qa_path = os.path.join("training_files", lang, str(level), "qa_pairs.jsonl")

    def _regen_corpus(pairs: list) -> None:
        """(Re)write qa_corpus.txt from `pairs`.

        Called on EVERY exit path: the corpus used to be rewritten only when
        NEW pairs were found, so once every prompt was already known the file
        was never refreshed — and if it was missing it stayed missing. Losing
        it removes the ONLY conditional prompt->answer text from phase_0 and
        from the dream's N1 replay, leaving the model to train on
        unconditional babble alone (measured at L0: peak 0.44 -> 0.36 and
        wrong-consonant answers, 'mama!' -> 'ra!').
        """
        if not pairs:
            return
        import random as _rnd
        corpus_path = os.path.join("training_files", lang, str(level),
                                   "qa_corpus.txt")
        reps = 20
        lines = []
        for _ in range(reps):
            _rnd.shuffle(pairs)
            for pair in pairs:
                pp = _strip_demo(pair.get("prompt", ""))
                rr = (pair.get("response") or "").strip()
                if pp and rr:
                    lines.extend([pp, rr, ""])
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    # Load existing pairs
    existing = {}
    if os.path.exists(qa_path):
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pair = json.loads(line)
                        existing[pair.get("prompt", "").strip()] = pair
                    except json.JSONDecodeError:
                        pass

    # At cap: no new pairs, but still make sure the corpus file exists
    if len(existing) >= MAX_TOTAL_QA:
        _regen_corpus(list(existing.values()))
        return 0

    # Only read the LATEST session log (the one just completed)
    ckpt_dir = os.path.join(ckpt_base, f"level_{level}")
    all_logs = sorted(glob.glob(os.path.join(ckpt_dir, "session_*.jsonl")))
    if not all_logs:
        return 0
    latest_log = all_logs[-1]

    # Collect candidates: positive turns first, then neutral.
    # The 'feedback' logged at row N grades the RESPONSE of row N-1, so the
    # grade for row i's (prompt, expected) exchange lives at row i+1 — same
    # realignment as _load_memory_bank.
    records = []
    with open(latest_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    positive_pairs, neutral_pairs = {}, {}
    for i, rec in enumerate(records):
        prompt = (rec.get("prompt") or "").strip()
        expected = (rec.get("expected") or "").strip()
        if not prompt or not expected:
            continue
        if prompt in existing:
            continue
        feedback = (records[i + 1].get("feedback") if i + 1 < len(records)
                    else None) or "="
        pair = {"prompt": prompt, "response": expected}
        if feedback in ("+++", "++", "+"):
            positive_pairs[prompt] = pair
        elif prompt not in positive_pairs:
            neutral_pairs[prompt] = pair

    # Build final list: positives first, then neutrals, up to available slots
    available = min(max_new, MAX_TOTAL_QA - len(existing))
    if available <= 0:
        return 0

    new_pairs = {}
    for p, pair in list(positive_pairs.items())[:available]:
        new_pairs[p] = pair
    remaining = available - len(new_pairs)
    for p, pair in list(neutral_pairs.items())[:remaining]:
        if p not in new_pairs:
            new_pairs[p] = pair

    if not new_pairs:
        _regen_corpus(list(existing.values()))
        return 0

    n_pos = sum(1 for p in new_pairs if p in positive_pairs)
    n_neu = len(new_pairs) - n_pos

    # Append new pairs to qa_pairs.jsonl
    with open(qa_path, "a", encoding="utf-8") as f:
        for pair in new_pairs.values():
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    n_added = len(new_pairs)
    print(f"  [QA update] +{n_added} nuove coppie ({n_pos}+ {n_neu}=) → {qa_path}  "
          f"(totale: {len(existing) + n_added}/{MAX_TOTAL_QA})")

    all_pairs = list(existing.values()) + list(new_pairs.values())
    _regen_corpus(all_pairs)
    print(f"  [QA update] qa_corpus.txt rigenerato  ({len(all_pairs)} coppie × 20 reps)")

    return n_added


# Demo prefix written by the local teacher's retry ('ma ma. di ma'): the
# scaffolding must never reach a corpus that is trained as plain text.
_DEMO_RE = _re.compile(r'^(?:(\S+)\s+)(?:\1\s+)*\1[.!?]?\s+')


def _strip_demo(text: str) -> str:
    out = _DEMO_RE.sub("", text or "").strip()
    return out or (text or "").strip()


def _is_periodic_text(s: str) -> bool:
    """True for degenerate repetition loops ('bababa' = 'ba'*3, 'aaa').
    A simple double ('baba' = 'ba'*2) is a legitimate word and is allowed."""
    for ul in range(1, 5):
        if len(s) >= 3 * ul and len(s) % ul == 0 and s[:ul] * (len(s) // ul) == s:
            return True
    return False


def _extract_patterns(bank: list, tok: "BPETokenizer",
                       min_freq: int = 3, max_ngram: int = 4) -> list:
    """
    Extract recurring token n-grams from the GOLD answers of positively
    graded exchanges.

    NOT from the model's own responses: mining its raw output made its
    babble the reinforced pattern set — measured at L0 the top patterns were
    ['baba', 'bababa', 'babababa', '<|EOS|>', 'lala'] and N2-A then trained
    each up to 20 times, teaching both syllable-repetition loops ('ma!' ->
    'mamamama!') and bare-EOS answers (43.8% of turns). Gold answers keep
    the dream anchored to what the teacher actually asked for.

    Returns list of (token_sequence_as_text, frequency) sorted by freq desc.
    """
    from collections import Counter
    ngram_counts: Counter = Counter()
    for entry in bank:
        if entry["weight"] < 0.8:    # only ++ and +++
            continue
        text = entry.get("expected") or ""
        if not text:
            continue
        ids = tok.encode(text)
        for n in range(2, max_ngram + 1):
            for i in range(len(ids) - n + 1):
                ngram = tuple(ids[i:i + n])
                ngram_counts[ngram] += 1

    patterns = []
    for ngram, freq in ngram_counts.most_common(200):
        if freq < min_freq:
            break
        try:
            text = tok.decode(list(ngram)).strip()
            if len(text) < 3:            # skip noise
                continue
            if "<|" in text or "|>" in text:
                continue                  # never reinforce special-token literals
            core = text.strip("!.?,: ")
            if _is_periodic_text(core):   # 'bababa', 'lalalala', 'aaa'
                continue
            patterns.append((text, freq))
        except Exception:
            continue
    return patterns


def phase_2_dream(args, start_checkpoint: str, ckpt_base: str) -> str:
    """
    Dream consolidation phase — runs after teaching (phase 1).

    Three dream modes, selected via args.dream_mode:

    'light'    — After a retry (model stuck). Heavy N1 stabilization,
                 no vocab growth, only positive memories, no REM.
                 Goal: anchor text distribution, avoid adding confusion.
                 N1: 50MB | N2-B: off | N3: only +++ | REM: off

    'standard' — After normal level completion. Full cycle.
                 N1: 30MB | N2-B: on | N3: all | REM: 100 entries

    'deep'     — After a long/difficult level. Intensive consolidation.
                 N1: 60MB | N2-B: on | N3: all | REM: 200 entries

    Biological analogy:
      light   = sleep after a frustrating day: more NREM, stabilize
      standard = sleep after a normal day: balanced NREM + REM
      deep    = sleep after intense learning: full consolidation
    """
    import random

    dream_mode = getattr(args, 'dream_mode', 'standard')

    # Parameters per mode
    DREAM_CONFIGS = {
        'light':    {'n1_mb': 50, 'n2b': False, 'n3_min_weight': 0.8,  'rem_cap': 0, 'n3_positive_only': True},
        'standard': {'n1_mb': 30, 'n2b': True,  'n3_min_weight': 0.5,  'rem_cap': 100, 'n3_positive_only': False},
        'deep':     {'n1_mb': 60, 'n2b': True,  'n3_min_weight': 0.5,  'rem_cap': 200, 'n3_positive_only': False},
    }
    cfg = DREAM_CONFIGS.get(dream_mode, DREAM_CONFIGS['standard'])

    level = args.level
    print("\n" + "="*60)
    print(f"  PHASE 2 — Dream / Consolidation  (level {level}, mode={dream_mode})")
    print(f"  N1={cfg['n1_mb']}MB  N2-B={'on' if cfg['n2b'] else 'off'}  "
          f"N3≥{cfg['n3_min_weight']}  REM={cfg['rem_cap']}")
    print("="*60)

    ckpt_dir = os.path.join(ckpt_base, f"level_{level}")

    # Load model
    if not start_checkpoint or not os.path.exists(start_checkpoint):
        learned = os.path.join(ckpt_dir, "final_learned.pt")
        if not os.path.exists(learned):
            print(f"  Error: {learned} not found. Run --phase 1 first.")
            return None
        start_checkpoint = learned

    # Load tokenizer: prefer the most recent compatible tokenizer for this level.
    # A tokenizer is compatible only if its vocab >= model.active_vocab_size.
    # Stale tokenizers from previous builds (smaller vocab) are skipped.
    tok = BPETokenizer()
    tok.load(TOKENIZER)  # base/8K as safe default

    model = TorchGPT.load(start_checkpoint)
    model_active = model.active_vocab_size

    for tok_candidate in [
        os.path.join(ckpt_dir, "tokenizer.json"),
        os.path.join(ckpt_base, f"level_{level-1}", "tokenizer.json") if level > 0 else None,
    ]:
        if tok_candidate and os.path.exists(tok_candidate):
            _t = BPETokenizer(); _t.load(tok_candidate)
            if len(_t) >= model_active:
                tok = _t
                print(f"  Tokenizer: {tok_candidate}  (vocab={len(tok)})")
                break
            # else: skip incompatible stale tokenizer

    # Fail fast: fallback base tokenizer smaller than the model would decode
    # trained ids as KeyError / desync — stop instead of corrupting the dream.
    if len(tok) < model_active:
        raise RuntimeError(
            f"No tokenizer with vocab >= model active_vocab_size="
            f"{model_active} found (best available: {len(tok)}). "
            f"Restore level_N/tokenizer.json before dreaming.")

    # Sync active_vocab_size with tokenizer (parent-based init for new rows)
    _sync_vocab_rows(model, tok, label="phase_2 vocab sync")

    opt   = TorchAdamOptimizer(model.parameters(), lr=2e-5)  # very low LR
    affect = AffectState()
    mod    = AffectModulator(affect)
    axioms = AxiomRegistry()
    trainer = TrainerB(model, tok, opt, affect, mod, axioms)

    # ── Build memory bank (used by N3 and REM) ───────────────────────────────
    print(f"\n  Building memory bank from logs L0→{level}...")
    bank = _load_memory_bank(ckpt_base, level)
    pos_count = sum(1 for e in bank if e["weight"] > 0)
    neg_count = sum(1 for e in bank if e["weight"] < 0)
    print(f"  Memoria: {len(bank)} entries  "
          f"({pos_count} positive, {neg_count} negative, "
          f"{len(bank)-pos_count-neg_count} neutral)")

    # ── N1: Corpus replay ────────────────────────────────────────────────────
    # Cap at N1_MAX_CHARS to keep dream phase fast (~5 min).
    # N1 runs LAST (after N2/N3/REM) so it has the final word on the distribution.
    N1_MAX_CHARS = cfg['n1_mb'] * 1_000_000
    # Collect corpus for N1 (executed last, after N3/REM)
    # Same narrative filter as phase_0/phase_1: without it, N1 re-injected up
    # to 30-50MB of unfiltered adult narrative as the LAST training act of
    # every session ("final word on the distribution"), undoing the very
    # washout fix that solved the L3 cliff.
    corpus_parts = []
    total_chars  = 0
    _n1_narrative_exclude = level >= 3
    for lvl in range(level + 1):
        corpus_dir = os.path.join("training_files", args.lang, str(lvl))
        for fpath in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
            if "teacher_prompt" in fpath:
                continue
            fname = os.path.basename(fpath)
            if _n1_narrative_exclude and fname != "qa_corpus.txt" \
                    and os.path.getsize(fpath) > 100 * 1024:
                continue   # skip large narrative files at L3+
            with open(fpath, encoding="utf-8") as f:
                text = f.read()
                if total_chars + len(text) > N1_MAX_CHARS:
                    remaining = N1_MAX_CHARS - total_chars
                    if remaining > 1000:
                        corpus_parts.append(text[:remaining])
                        total_chars += remaining
                    break
                corpus_parts.append(text)
                total_chars += len(text)
        if total_chars >= N1_MAX_CHARS:
            break

    # ── N2: Pattern mining + vocab expansion ─────────────────────────────────
    # Two operations:
    #   A) Identify recurring token n-grams from +++ responses (pattern mining)
    #   B) Find NEW token merges from the same text and add them to the tokenizer
    #      → expands the vocabulary organically during the dream phase
    print(f"\n  [N2] NREM spindles — pattern mining + vocab expansion...")
    patterns = _extract_patterns(bank, tok, min_freq=3, max_ngram=4)

    # ── N2-A: reinforce existing patterns ────────────────────────────────────
    n_new_tokens = 0
    if patterns:
        print(f"  Patterns found: {len(patterns)}")
        print(f"  Top 5: {[p for p,_ in patterns[:5]]}")
        pattern_corpus = []
        for text, freq in patterns[:50]:
            reps = min(freq, 20)
            pattern_corpus.extend([text] * reps)
        random.shuffle(pattern_corpus)
        pattern_text = "\n".join(pattern_corpus)
        n2_losses = trainer.train_on_text(pattern_text, block_size=32,
                                          batch_size=8, log_every=999)
        n2_avg = float(np.mean(n2_losses)) if n2_losses else float("nan")
        print(f"  N2-A loss: {n2_avg:.4f}  ({len(pattern_corpus)} sequences)")
    else:
        print("  No frequent patterns found — skip N2-A.")

    # ── N2-B: vocab growth ────────────────────────────────────────────────────
    # Build growth text from three sources:
    #   1. GOLD expected answers of positively-graded exchanges — the lexicon
    #      the model is being taught. NOT the raw model responses: growing from
    #      the model's own babble turned repetition loops into mega-tokens
    #      ('babababa…' × 48 chars) and re-encoded the very targets being
    #      drilled, resetting mastered token paths mid-level (measured at L0:
    #      'lala!' 8/59 → 0/21 after becoming a cold single token).
    #   2. Teacher prompts with leading praise stripped ('bravo!', 'ottimo!'
    #      were becoming tokens). Biological: children consolidate words they
    #      hear before they can say them.
    #   3. Found patterns — frequent n-grams from analysis
    MAX_NEW_PER_DREAM = 200   # cap new tokens per dream session
    import re as _re_g
    _PRAISE_RE = _re_g.compile(
        r"^(bravo|brava|bene|benissimo|ottimo|perfetto|giusto|esatto|sì|si|no)"
        r"[!.,:\s]+", _re_g.IGNORECASE)
    growth_parts  = [e["expected"] for e in bank
                     if e["weight"] >= 0.8 and e.get("expected")]
    _seen_prompts = []
    for e in bank:
        p = (e.get("prompt") or "").strip()
        for _ in range(3):   # praise can stack: 'bravo! benissimo! di baba'
            p2 = _PRAISE_RE.sub("", p)
            if p2 == p:
                break
            p = p2.strip()
        if p:
            _seen_prompts.append(p)
    growth_parts += _seen_prompts
    growth_parts += [p for p, _ in patterns[:100]]
    growth_text = " ".join(growth_parts)

    if growth_text.strip() and cfg['n2b']:
        # Convert to DynamicBPETokenizer (superset of BPETokenizer)
        from dynamic_model.core.tokenizer import DynamicBPETokenizer
        dyn_tok = DynamicBPETokenizer()
        dyn_tok.vocab       = dict(tok.vocab)
        dyn_tok.token_to_id = dict(tok.token_to_id)
        dyn_tok.merges      = list(tok.merges)
        dyn_tok.token_parents = {nid: (a, b) for (a, b, nid) in tok.merges}
        # IMPORTANT: preserve special_tokens (EOS, physisml, etc.) — without this,
        # encode() splits 'physisml' into sub-tokens instead of using ID 8001.
        dyn_tok.special_tokens = dict(getattr(tok, 'special_tokens', {}))
        dyn_tok.special_ids    = dict(getattr(tok, 'special_ids', {}))
        dyn_tok._trained    = True

        # Sanity check: grow() will assign IDs starting from max(vocab)+1.
        # activate_slots() requires them to be contiguous from model.active_vocab_size.
        # If tokenizer is behind the model (e.g. base tokenizer loaded as fallback),
        # skip N2-B to prevent the ValueError rather than crashing the dream.
        tok_next_id = max(dyn_tok.vocab.keys()) + 1
        if tok_next_id != model.active_vocab_size:
            print(f"  N2-B: skip — tokenizer next_id={tok_next_id} ≠ model active={model.active_vocab_size}")
            print(f"  (tokenizer out of sync; load level tokenizer to re-enable)")
            new_ids = []
        else:
            # Word-count cap for new tokens: at L0-L2 prevent merging full
            # teacher phrases (like 'mamma la mamma. di: la ') into single tokens.
            # Early levels need compositional vocab (article + noun), not blocks.
            # L0-L2: max 2 words per token
            # L3-L5: max 3 words (short phrases allowed)
            # L6+:   unlimited (narrative chunks useful for larger context)
            if level <= 2:
                _max_words = 2
            elif level <= 5:
                _max_words = 3
            else:
                _max_words = 0   # unlimited
            # Protect the CURRENT level's drill targets from tokenisation:
            # merging what is being actively taught orphans the learned
            # multi-token path and regresses mastered targets (see grow()).
            _protect = set()
            for e in bank:
                if e.get("level") != level:
                    continue
                exp = (e.get("expected") or "").strip()
                if exp and len(exp) <= 30:
                    _protect.add(exp)
                    _protect.add(exp.strip("!.?,: "))
            new_ids = dyn_tok.grow(growth_text, n_merges=MAX_NEW_PER_DREAM,
                                   max_words=_max_words, protect=_protect)

        if new_ids:
            # Initialize embeddings for new tokens using parent vectors
            W_np = model.tok_emb.weight.data.cpu().numpy()  # (V, d)
            init_vecs = []
            for nid in new_ids:
                v = dyn_tok.get_parent_embedding(nid, W_np)
                init_vecs.append(v)

            init_tensor = torch.from_numpy(np.stack(init_vecs)).float()
            # activate_slots() writes into pre-allocated dormant rows.
            # The embedding matrix, optimizer, and axiom hooks are all unchanged.
            model.activate_slots(new_ids, init_tensor)
            trainer.tokenizer = dyn_tok
            tok = dyn_tok
            n_new_tokens = len(new_ids)

            # Novelty spike: activating new vocabulary slots = "aha moment"
            # Simulates dopaminergic burst when discovering a new word
            trainer.affect.register_token_activation(n_new_tokens)

            new_words = [dyn_tok.decode([nid]) for nid in new_ids[:10]]
            print(f"  N2-B: +{n_new_tokens} new tokens  "
                  f"vocab active {model.active_vocab_size - n_new_tokens} → {model.active_vocab_size}"
                  f"  pleasure → {trainer.affect.pleasure:.2f}")
            print(f"  New tokens (first 10): {new_words}")
        else:
            print("  N2-B: no new tokens (threshold not reached)")

    # ── N2.5: Supervised Q&A pairs (SFT-style) ───────────────────────────────
    # Gold standard prompt→response pairs for this level.
    # Reinforces exact correlations before the emotional memory replay.
    # Biologically: rehearsing the correct vocabulary before the deeper NREM phase.
    qa_path = os.path.join("training_files", args.lang, str(level), "qa_pairs.jsonl")
    if os.path.exists(qa_path):
        # Allow override via --n-qa-epochs; otherwise use per-mode defaults
        if getattr(args, 'n_qa_epochs', None) is not None:
            n_qa_epochs = args.n_qa_epochs
        else:
            n_qa_epochs = 15 if dream_mode == "standard" else (10 if dream_mode == "deep" else 5)
        print(f"\n  [N2.5] SFT Q&A pairs ({qa_path}, {n_qa_epochs} epoche)...")
        qa_losses = trainer.train_on_qa_pairs(qa_path, n_epochs=n_qa_epochs)
        if qa_losses:
            import numpy as _np
            print(f"  N2.5 loss: {float(_np.mean(qa_losses[-10:])):.4f}  ({len(qa_losses)} steps)")
    else:
        print(f"  [N2.5] No qa_pairs.jsonl for level {level} — skip.")

    # ── N3: Memory bank replay (emotional) ───────────────────────────────────
    print(f"\n  [N3] NREM deep sleep — memory bank consolidation (min_weight={cfg['n3_min_weight']})...")
    # In light mode: only positive memories (no negatives — avoid consolidating failures)
    if cfg.get('n3_positive_only'):
        n3_entries = [e for e in bank if e["weight"] >= cfg['n3_min_weight']]
    else:
        n3_entries = [e for e in bank if abs(e["weight"]) >= cfg['n3_min_weight']]
    if n3_entries:
        print(f"  {len(n3_entries)} entries with |weight| ≥ 0.5")
        n3_count = 0
        for entry in n3_entries:
            prompt   = entry["prompt"]
            expected = entry["expected"] or entry["response"]
            weight   = entry["weight"]
            if not expected:
                continue
            # Always train toward expected (positive: reinforce; negative: teach correct)
            fb_strength = abs(weight) * (1.0 if weight > 0 else 0.6)
            trainer.step(prompt, expected, feedback=fb_strength)
            n3_count += 1
        print(f"  N3: {n3_count} memories consolidated")
    else:
        print("  No memories with sufficient salience — skip N3.")

    # ── REM: Introspection / self-play ───────────────────────────────────────
    print(f"\n  [REM] Introspezione / self-play...")
    # Sample top high-salience entries with known expected
    rem_entries = [e for e in bank
                   if abs(e["weight"]) >= 0.8 and e.get("expected")]
    rem_entries = rem_entries[:cfg['rem_cap']]   # 0 = skip REM (light mode)

    if rem_entries:
        print(f"  {len(rem_entries)} entries da esaminare...")
        close_count = 0   # model already knows
        gap_count   = 0   # model still needs to learn
        rem_count   = 0

        for entry in rem_entries:
            prompt   = entry["prompt"]
            expected = entry["expected"]

            # Generate current model response (no weight update)
            try:
                generated = trainer.generate(
                    prompt, max_tokens=15, base_temperature=0.5, top_k=10
                )
                response_now = generated[len(prompt):].strip()[:30]
            except Exception:
                response_now = ""

            # Measure gap between current output and expected
            similarity = _jaccard(response_now, expected)

            if similarity >= 0.4:
                # Model already close: gentle reinforcement
                fb_strength = 0.3
                close_count += 1
            else:
                # Gap exists: stronger learning
                fb_strength = min(1.0, abs(entry["weight"]) * 0.9)
                gap_count += 1

            trainer.step(prompt, expected, feedback=fb_strength)
            rem_count += 1

        print(f"  REM: {rem_count} self-play steps  "
              f"(already knows: {close_count}, to learn: {gap_count})")
    else:
        print("  No entries with expected available — skip REM.")

    # ── N1: NREM slow wave — runs LAST to anchor text distribution ────────────
    # Running N1 after N3/REM ensures it has the final word on the distribution.
    # N3/REM add dialogue drift; N1 corrects it back toward the text corpus.
    print(f"\n  [N1] NREM slow wave — corpus replay L0→{level} (max {N1_MAX_CHARS//1_000_000}MB, last)...")
    if corpus_parts:
        random.shuffle(corpus_parts)
        nrem_text = "\n\n".join(corpus_parts)
        print(f"  {len(nrem_text):,} chars  ({len(corpus_parts)} file)")
        n1_losses = trainer.train_on_text(nrem_text, block_size=128,
                                          batch_size=32, log_every=100)
        n1_avg = float(np.mean(n1_losses[-20:])) if n1_losses else float("nan")
        print(f"  N1 loss: {n1_avg:.4f}")
    else:
        print("  No corpus — skip N1.")

    # ── QA pairs update from session logs ────────────────────────────────────
    # Extract unique (prompt, expected) from all sessions at this level and
    # add any prompts not already in qa_pairs.jsonl.  Uses the teacher's
    # `expected` (gold standard), not the model's actual response.
    n_qa_added = _update_qa_pairs_from_sessions(ckpt_base, level, args.lang)

    # ── Dream stats ──────────────────────────────────────────────────────────
    # Save the tokenizer BEFORE the checkpoint: a crash between the two writes
    # must never leave a new final_dreamed.pt paired with an old tokenizer
    # (the checkpoint loader tolerates a tokenizer that is newer, not older).
    if n_new_tokens > 0:
        tok_path = os.path.join(ckpt_dir, "tokenizer.json")
        tok.save(tok_path)
        print(f"  Updated tokenizer saved: {tok_path}  "
              f"(vocab size: {len(tok)})")

    dreamed_path = os.path.join(ckpt_dir, "final_dreamed.pt")
    model.save(dreamed_path)

    print(f"\n{'='*60}")
    print(f"  Dream complete — level {level}")
    print(f"  Affect: {trainer.affect}")
    print(f"  N1 corpus: {len(corpus_parts)} files")
    print(f"  N2 patterns: {len(patterns)}  +{n_new_tokens} new tokens")
    print(f"  N3 memories: {len(n3_entries)}")
    print(f"  REM self-play: {len(rem_entries)}")
    print(f"  QA pairs nuove: {n_qa_added}")
    print(f"  Vocab: {model.active_vocab_size} active / {model.vocab_size} allocated")
    print(f"  → {dreamed_path}")
    print("="*60)
    return dreamed_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Curriculum training 0→1 with Claude tutor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--phase", type=int, choices=[0, 1, 2], default=None,
                        help="Run only phase 0, 1 or 2 (default: all three)\n"
                             "  0 = text training → final.pt\n"
                             "  1 = Claude teaching → final_learned.pt\n"
                             "  2 = dream/consolidation → final_dreamed.pt")
    parser.add_argument("--checkpoint", default=None,
                        help="Starting checkpoint (used for --phase 1)")
    parser.add_argument("--epochs-0", type=int, default=5,
                        help="Autonomous training epochs in phase 0 (default: 5)")
    parser.add_argument("--interactions", default="60",
                        help="Interactions with Claude in phase 1 (default: 60). "
                             "Use 'auto' to continue until quality is good "
                             "(Ctrl-C to stop while keeping progress)")
    parser.add_argument("--tutor-model", default="claude-haiku-4-5",
                        help="Tutor model: claude-haiku-4-5 | claude-sonnet-4-6 | "
                             "local (rule-based, no API) | "
                             "auto (local if local_teacher.json exists, else haiku)")
    parser.add_argument("--age", type=float, default=1.0,
                        help=(
                            "Virtual age of the model — adapts teaching style:\n"
                            "  0–1  : newborn   — sounds only (ma, pa, oh!)\n"
                            "  1–2  : toddler   — single words (cane, mamma)\n"
                            "  2–4  : early child — article+noun, simple verbs\n"
                            "  4–7  : child     — full sentences, colours, numbers\n"
                            "  7+   : school age — grammar, conjugation, dialogue\n"
                            "(default: 1.0)"
                        ))
    parser.add_argument("--lang", default="it",
                        help="Language code used in checkpoint path (default: it)")
    parser.add_argument("--level", type=int, default=0,
                        help="Curriculum level to train (default: 0). "
                             "Phase 0 creates level_N/final.pt; "
                             "phase 1 creates/updates level_N/final_learned.pt.")
    parser.add_argument("--dream-mode", default="standard",
                        choices=["light", "standard", "deep"],
                        help="Dream mode: light (retry/stuck), standard (normal), "
                             "deep (long level). Default: standard")
    parser.add_argument("--n-qa-epochs", type=int, default=None,
                        help="Override N2.5 QA pairs epochs in dream (default: per dream-mode)")
    parser.add_argument("--no-qa-corpus", action="store_true",
                        help="Exclude qa_corpus.txt from text training corpus (for experiments)")
    parser.add_argument("--no-turn-ckpt", action="store_true",
                        help="Skip per-turn checkpoint saves during phase 1 (saves disk, for experiments)")
    parser.add_argument("--ckpt-base", default=None,
                        help="Override checkpoint base directory "
                             "(default: models/checkpoints/{lang}). "
                             "Useful for parallel experiments.")
    args = parser.parse_args()

    # Normalise --interactions: 'auto' → -1, else int
    if str(args.interactions).lower() == "auto":
        args.interactions = -1   # sentinel for auto mode
    else:
        args.interactions = int(args.interactions)

    if args.ckpt_base:
        ckpt_base = args.ckpt_base
    else:
        ckpt_base = os.path.join(CKPT_BASE, args.lang)
    os.makedirs(ckpt_base, exist_ok=True)
    level_dir  = os.path.join(ckpt_base, f"level_{args.level}")

    if args.phase == 0:
        phase_0(args, ckpt_base)

    elif args.phase == 1:
        # Remember whether --checkpoint came from the operator: an explicit
        # checkpoint overrides the learned/dreamed continuation chain in
        # phase_1, the auto-detected final.pt must not.
        args._explicit_checkpoint = bool(args.checkpoint)
        if not args.checkpoint:
            auto = os.path.join(level_dir, "final.pt")
            if os.path.exists(auto):
                args.checkpoint = auto
                print(f"Text base auto-detected: {auto}")
            else:
                print(f"Error: {auto} not found.")
                print(f"  Run first: --phase 0 --level {args.level}")
                sys.exit(1)
        phase_1(args, args.checkpoint, ckpt_base)

    elif args.phase == 2:
        phase_2_dream(args, args.checkpoint, ckpt_base)

    else:
        # All three phases in sequence for this level
        ckpt0    = phase_0(args, ckpt_base)
        ckpt1    = phase_1(args, ckpt0, ckpt_base)
        phase_2_dream(args, ckpt1, ckpt_base)

    # active.pt is NOT updated automatically.
    # Use ./set_model.sh to copy the desired checkpoint for testing.
    learned = os.path.join(level_dir, "final_learned.pt")
    base    = os.path.join(level_dir, "final.pt")
    best    = learned if os.path.exists(learned) else base
    if os.path.exists(best):
        print(f"\nPer testare il modello:")
        print(f"  ./set_model.sh {best}")


if __name__ == "__main__":
    main()

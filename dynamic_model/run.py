"""
Experiment B — interactive session with affective system.

Each turn:
  1. You write a prompt
  2. The model generates a response (modulated by affective state)
  3. You give feedback: +  = approval
                        =  = neutral (passive observation)
                        -  = disapproval
  4. Weights are updated and affective state changes

Special commands:
  /state     — show detailed affective state
  /axiom     — register current response as axiom
  /save      — save manual checkpoint
  /load      — load checkpoint
  /corpus    — train on a text file (autonomous learning)
  /quit      — exit

Start:
  python3 dynamic_model/run.py
  python3 dynamic_model/run.py --checkpoint dynamic_model/exp_b/checkpoints/pretrain_full.pt
"""
import sys, os, argparse

# Add the project root to path so that both `dynamic_model`
# and `physisml` are importable regardless of where the script is launched
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEST1 = os.path.join(_ROOT, 'tests', 'test_1')
for _p in [_ROOT, _TEST1]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import numpy as np

from physisml.torch_model import TorchGPT, TorchAdamOptimizer
from physisml.tokenizer   import BPETokenizer
from dynamic_model import language

from dynamic_model.exp_b.affect_state import AffectState
from dynamic_model.exp_b.modulator    import AffectModulator
from dynamic_model.exp_b.axioms       import AxiomRegistry
from dynamic_model.exp_b.trainer      import TrainerB

torch.set_num_threads(12)

# Fallback tokenizer, used only when no tokenizer is saved next to the
# checkpoint and none is given on the command line.
DEFAULT_TOKENIZER = "dynamic_model/data/tokenizer_base.json"

# ---------------------------------------------------------------------------
# Affective state display
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value:.2f}"

def print_affect(affect: AffectState, verbose: bool = False) -> None:
    s = affect
    print(f"\n  Affective state (step {s.step}):")
    print(f"  Confidence  {_bar(s.confidence)}")
    print(f"  Ignorance   {_bar(s.ignorance)}")
    print(f"  Pleasure    {_bar(s.pleasure)}")
    print(f"  Pain        {_bar(s.pain)}")
    print(f"  Fear        {_bar(s.fear)}")
    if verbose:
        history = list(s.history)[-5:]
        if history:
            print(f"\n  Last {len(history)} snapshots:")
            for h in history:
                print(f"    step {h.step:4d}  conf={h.confidence:.2f}  "
                      f"pleas={h.pleasure:.2f}  pain={h.pain:.2f}  fear={h.fear:.2f}")

def print_compact_affect(affect: AffectState) -> None:
    s = affect
    icons = []
    if s.confidence > 0.6:
        icons.append("🧠confident")
    elif s.confidence < 0.2:
        icons.append("❓uncertain")
    if s.pleasure > 0.6:
        icons.append("😊pleased")
    if s.pain > 0.3:
        icons.append("😟pained")
    if s.fear > 0.6:
        icons.append("😨fearful")
    if s.ignorance > 0.7:
        icons.append("😶ignorant")
    label = "  " + " | ".join(icons) if icons else ""
    print(f"  [conf={s.confidence:.2f} fear={s.fear:.2f} pleas={s.pleasure:.2f}]{label}")

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(args, quiet: bool = False) -> TrainerB:
    """Load model and initialise trainer. quiet=True redirects setup messages to stderr."""
    log = (lambda *a, **kw: print(*a, **kw, file=sys.stderr)) if quiet else print

    tok = BPETokenizer()

    if args.checkpoint and os.path.exists(args.checkpoint):
        log(f"Loading model from {args.checkpoint}...")
        model = TorchGPT.load(args.checkpoint)
        # Auto-detect matching tokenizer (priority order):
        # 1. models/active_tokenizer.json  (set by set_model.sh alongside active.pt)
        # 2. tokenizer.json in the checkpoint's directory
        # 3. fallback to base tokenizer
        ckpt_dir = os.path.dirname(os.path.realpath(args.checkpoint))
        candidates = [
            # active_tokenizer.json sits ALONGSIDE active.pt, not one level up.
            # The old first candidate had a spurious '..' and resolved to the
            # repo root, so it never matched: every run silently fell back to
            # the 503-token base tokenizer while the checkpoint expected the
            # grown one, producing garbage output and a KeyError on the first
            # sampled id above 503.
            os.path.join(ckpt_dir, "active_tokenizer.json"),
            os.path.join(ckpt_dir, "tokenizer.json"),
            os.path.join(ckpt_dir, "..", "tokenizer.json"),
        ]
        # An explicit --tokenizer wins over auto-detection: it used to be only
        # the last-resort fallback, so the one flag someone would reach for to
        # fix a mismatch was silently ignored whenever a candidate existed.
        tok_path = args.tokenizer or next(
            (p for p in candidates if os.path.exists(os.path.normpath(p))),
            DEFAULT_TOKENIZER)
        tok_path = os.path.normpath(tok_path)
        tok.load(tok_path)
        log(f"Tokenizer: {tok_path}  ({len(tok)} tokens)")
        # A tokenizer that does not cover the model's ACTIVE vocabulary means
        # the two do not belong together. Say so: the failure is otherwise
        # silent, and the model answers with plausible-looking noise.
        # Compare against active_vocab_size, never against vocab_size: the
        # embedding is padded with dormant rows (9000 slots for 2517 English
        # tokens), so a ratio against the padded size cried wolf on every
        # language whose vocabulary is smaller than Italian's.
        _active = getattr(model, "active_vocab_size", model.vocab_size)
        if len(tok) < _active or len(tok) > model.vocab_size:
            log(f"  WARNING: tokenizer has {len(tok)} tokens but the model has "
                f"{_active} active of {model.vocab_size} slots.")
            log(f"  This checkpoint is paired with the wrong tokenizer — output "
                f"will be meaningless.")
            log(f"  Run ./set_model.sh <checkpoint.pt> to refresh "
                f"models/active_tokenizer.json, or pass --tokenizer explicitly.")
        opt = TorchAdamOptimizer(model.parameters(), lr=1e-4)
    else:
        tok_path = args.tokenizer or DEFAULT_TOKENIZER
        tok.load(tok_path)
        log("Initialising new model...")
        model = TorchGPT(len(tok), 256, 4, 4, 1024, 129, 0.1)
        opt   = TorchAdamOptimizer(model.parameters(), lr=1e-3)

    log(f"Model: {model.num_params:,} params  vocab={model.vocab_size}\n")

    affect  = AffectState()
    mod     = AffectModulator(affect)
    axioms  = AxiomRegistry()
    trainer = TrainerB(model, tok, opt, affect, mod, axioms)

    # The axioms are the language's, not this file's. They were four Italian
    # phrases hardcoded here while --checkpoint defaults to models/active.pt --
    # one slot shared by every language -- so a REPL opened right after an
    # English build protected 'io sono' on English weights, where the words
    # encode to subwords the corpus never trains.
    #
    # Which language that is: --lang if given, otherwise whatever the loaded
    # vocabulary says it is, which is the only marker a checkpoint carries.
    lang = getattr(args, "lang", None) or language.detect(tok_path) \
        or language.DEFAULT_LANG
    lg   = language.load(lang)
    log(f"Language: {lang}  ({lg.name})"
        + ("" if getattr(args, "lang", None) else "  [from the vocabulary]"))

    base_axioms = lg.axioms("grammar")
    if not base_axioms:
        log(f"  no axioms declared for '{lang}' — add an \"axioms\" block to "
            f"{language.manifest_path(lang)}")
    # Redirect axiom print() to stderr when in quiet mode
    if quiet:
        _stdout, sys.stdout = sys.stdout, sys.stderr
    for text, prot in base_axioms:
        trainer.add_axiom(text, is_objective=True, protection=prot)
    if quiet:
        sys.stdout = _stdout

    return trainer

# ---------------------------------------------------------------------------
# Comandi speciali
# ---------------------------------------------------------------------------

def handle_command(cmd: str, trainer: TrainerB,
                   last_response: str, args) -> bool:
    """Return True if the command was handled."""
    cmd = cmd.strip().lower()

    if cmd == "/state":
        print_affect(trainer.affect, verbose=True)
        return True

    if cmd == "/axiom" and last_response:
        protection = input("  Protection level [0.0-1.0, enter=0.9]: ").strip()
        try:
            prot = float(protection) if protection else 0.9
        except ValueError:
            prot = 0.9
        obj = input("  Is it objective? [Y/n]: ").strip().lower() != "n"
        trainer.add_axiom(last_response.strip(), is_objective=obj, protection=prot)
        return True

    if cmd.startswith("/save"):
        parts = cmd.split()
        path  = parts[1] if len(parts) > 1 else \
                f"dynamic_model/exp_b/checkpoints/session_{trainer.step_count}.pt"
        trainer.model.save(path)
        print(f"  Checkpoint saved: {path}")
        return True

    if cmd.startswith("/corpus"):
        parts  = cmd.split(maxsplit=1)
        fpath  = parts[1].strip() if len(parts) > 1 else \
                 input("  File path: ").strip()
        if not os.path.exists(fpath):
            print(f"  File not found: {fpath}")
            return True
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        print(f"  Training on {len(text):,} chars...")
        losses = trainer.train_on_text(text, log_every=50)
        print(f"  Loss: {losses[0]:.3f} → {np.mean(losses[-10:]):.3f}")
        return True

    if cmd == "/quit" or cmd == "/exit":
        print("  Goodbye!")
        sys.exit(0)

    return False

# ---------------------------------------------------------------------------
# Loop principale
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt",         nargs="?", default=None,
                        help="Optional prompt — if provided, generates one response "
                             "non-interactively and exits")
    parser.add_argument("--checkpoint",
                        default="models/active.pt")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer JSON. Overrides auto-detection; when "
                             "omitted, the one saved next to the checkpoint is "
                             "used, falling back to the base tokenizer.")
    parser.add_argument("--lang", default=None,
                        help="Language of the checkpoint, for the axioms and "
                             "the word lists. Read from the vocabulary when "
                             "omitted; there is one models/active.pt for every "
                             "language, so it cannot be assumed.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40)
    parser.add_argument("--max_tokens",  type=int,   default=80)
    parser.add_argument("--chat",         action="store_true",
                        help="Chat mode: no feedback prompt, no training. "
                             "Pure conversation without weight updates.")
    args = parser.parse_args()

    trainer = setup(args, quiet=args.prompt is not None)

    # --- Non-interactive mode: single prompt → output → exit ---
    if args.prompt:
        response = trainer.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            base_temperature=args.temperature,
            top_k=args.top_k,
        )
        print(response[len(args.prompt):])
        return

    # --- Interactive mode ---
    if args.chat:
        print("\n" + "="*55)
        print("  Chat — pure conversation mode")
        print("  (no feedback, no weight updates)")
        print("  Commands: /state /quit")
        print("="*55 + "\n")
    else:
        print("\n" + "="*55)
        print("  Experiment B — Affective System")
        print("  Commands: /state /axiom /save /corpus /quit")
        print("  Feedback after response: +  =  -")
        print("="*55 + "\n")

    print_affect(trainer.affect)
    print()

    last_response = ""
    last_prompt   = ""

    while True:
        # --- Prompt ---
        try:
            prompt = input("Tu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Session ended.")
            break

        if not prompt:
            continue

        # Comandi speciali
        if prompt.startswith("/"):
            handle_command(prompt, trainer, last_response, args)
            continue

        if args.chat:
            # ── Chat mode: no training, no feedback ───────────────────────
            response = trainer.generate(
                prompt,
                max_tokens       = args.max_tokens,
                base_temperature = args.temperature,
                top_k            = args.top_k,
            )
            generated = response[len(prompt):]
            print(f"\nModel: {generated}")
            print_compact_affect(trainer.affect)
        else:
            # ── Training mode: passive exposure + generate + feedback ──────
            # Model learns from the prompt BEFORE responding (passive exposure)
            trainer.step("", prompt, feedback=0.2)

            response = trainer.generate(
                prompt,
                max_tokens       = args.max_tokens,
                base_temperature = args.temperature,
                top_k            = args.top_k,
            )
            generated = response[len(prompt):]
            print(f"\nModel: {generated}")
            print_compact_affect(trainer.affect)

            last_prompt   = prompt
            last_response = generated

            # --- Feedback ---
            try:
                fb_raw = input("\n  Feedback [+ = -]: ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if fb_raw.startswith("/"):
                handle_command(fb_raw, trainer, last_response, args)
                continue

            feedback_map = {"+": 1.0, "=": 0.0, "-": -1.0,
                            ":)": 1.0, ":|": 0.0, ":(": -1.0,
                            "y": 1.0, "n": -1.0, "": 0.0}
            feedback = feedback_map.get(fb_raw.lower(), 0.0)

            result = trainer.step(last_prompt, last_response,
                                  feedback=feedback)

            fb_label = {1.0: "approved ✓", 0.0: "neutral =",
                        -1.0: "rejected ✗"}.get(feedback, "?")
            loss_str = f"  loss={result['loss']:.4f}" if result["loss"] else ""
            print(f"  [{fb_label}]{loss_str}")
        print_compact_affect(trainer.affect)
        print()


if __name__ == "__main__":
    main()

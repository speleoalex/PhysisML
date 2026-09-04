"""The persistent drive: debt that accumulates on an unresolved unknown.

Design: docs_internal/curiosita_meccanismo.md §6. The epistemic trigger
(epistemic.py) says, turn by turn, whether the weights hold a class for the
noun in front of the model. That verdict is stateless: it forgets the noun
the moment the turn ends, and the loop goes back to asking round-robin. This
ledger is the state — a debt per word that CHARGES every time the noun is met
and remains unresolved, by an amount that comes from the margin (how far the
weights are from holding a class, not how many times it was met), and that
DISCHARGES when the oracle answers. It moves one thing in the loop: the order
of the questions. The noun asked next is the one the model is most in debt
about, which is the closest thing to a motive obtainable at this scale.

    debt[word] = {encounters, arousal, first_seen, resolved}

    charge(w, tau - margin)   arousal_w <- min(1, arousal_w + delta)
    discharge(w)              resolved <- True
    pressure()                sum of unresolved arousal / cap
    next_target(queue)        novelty first, then debt, then fairness

Contracts. It consumes an `EpistemicVerdict`'s numbers and recomputes
nothing; it never reads the weights. It is not a second trigger of the dream
(the pressure is logged beside the probe, never compared with a threshold):
two triggers in conflict with the degradation one would make the (pre, post)
series unreadable. It selects only from the queue it is handed, so what
`build_queue` filtered out — the reserve and the frozen probes — cannot come
back through it. Persisted beside affect_memory.json because it has the same
life cycle: one file per language, carried across processes.

The falsifiable claim, stated in the design: with the drive on the loop must
acquire more nouns per probe point lost than with it off. If not, the drive
goes, it is not defended.
"""
from __future__ import annotations

import json
import os
import time

ACQUIRED, DECLINED = "acquired", "declined"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class CuriosityDrive:
    """The ledger of epistemic debt, one entry per word."""

    VERSION = 1

    def __init__(self, path: str | None = None, cap: float = 8.0):
        self.path = path
        self.cap = float(cap)
        self.debt: dict[str, dict] = {}
        self.last: str | None = None

    # ── persistence ─────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str, cap: float = 8.0) -> "CuriosityDrive":
        """The ledger on disk, or an empty one if there is none yet."""
        d = cls(path, cap)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            d.debt = {w: dict(e) for w, e in data.get("debt", {}).items()}
            d.last = data.get("last")
            d.cap = float(data.get("cap", cap))
        return d

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": self.VERSION, "cap": self.cap, "last": self.last,
                       "saved": _now(), "debt": self.debt},
                      f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    # ── the three verbs ─────────────────────────────────────────────────────
    def _entry(self, word: str) -> dict:
        return self.debt.setdefault(word, {
            "encounters": 0, "arousal": 0.0, "first_seen": _now(),
            "resolved": False, "last_delta": 0.0})

    def charge(self, word: str, delta: float) -> float:
        """One encounter with `word`, `delta` = tau - margin from the verdict.

        Only a positive delta — the weights below threshold, i.e. ignorant —
        adds to the arousal: a known noun is met, counted, and leaves no debt.
        A resolved noun is met and counted too, and stays resolved.
        """
        e = self._entry(word)
        e["encounters"] += 1
        e["last_delta"] = round(float(delta), 4)
        if not e["resolved"] and delta > 0:
            e["arousal"] = round(min(1.0, e["arousal"] + float(delta)), 4)
        self.last = word
        return e["arousal"]

    def discharge(self, word: str, how: str = ACQUIRED) -> None:
        """The debt is settled: the oracle answered (`acquired`), or it could
        not classify the noun and the honesty material closed the question
        (`declined`). Either way the noun stops nagging."""
        e = self._entry(word)
        e["peak"] = max(e.get("peak", 0.0), e["arousal"])
        e["arousal"] = 0.0
        e["resolved"] = True
        e["resolved_how"] = how
        e["resolved_at"] = _now()

    # ── readings ────────────────────────────────────────────────────────────
    def arousal(self, word: str) -> float:
        e = self.debt.get(word)
        return 0.0 if e is None or e["resolved"] else e["arousal"]

    def encounters(self, word: str) -> int:
        e = self.debt.get(word)
        return 0 if e is None else e["encounters"]

    def unresolved(self) -> list[tuple[str, float]]:
        """(word, arousal) for every open debt, highest first."""
        out = [(w, e["arousal"]) for w, e in self.debt.items()
               if not e["resolved"] and e["arousal"] > 0]
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def pressure(self) -> float:
        """The aggregate scalar in [0, 1]: open debt over the cap."""
        if self.cap <= 0:
            return 0.0
        return min(1.0, sum(a for _, a in self.unresolved()) / self.cap)

    def next_target(self, queue: list[dict]) -> dict:
        """The noun to ask about next, from `queue` and nowhere else.

        Order: a noun never met comes first (until the whole queue has been
        read once, this IS the round-robin); then the highest open debt; then,
        among equals, the one met least often; then queue order. The noun
        asked last is not asked again immediately when there is a choice —
        an unresolvable debt would otherwise take every turn until it
        saturated. All of it deterministic: no randomness in the ordering.
        """
        if not queue:
            raise ValueError("empty queue")
        cands = [n for n in queue if n["w"] != self.last] or list(queue)
        index = {id(n): i for i, n in enumerate(queue)}

        def key(n):
            e = self.debt.get(n["w"])
            if e is None:
                return (0, 0.0, 0, index[id(n)])
            arousal = 0.0 if e["resolved"] else e["arousal"]
            return (1, -arousal, e["encounters"], index[id(n)])

        return min(cands, key=key)

    def summary(self, top: int = 3) -> str:
        open_ = self.unresolved()
        head = ", ".join(f"{w} {a:.2f}" for w, a in open_[:top])
        return (f"pressure {self.pressure():.2f}, {len(open_)} open"
                + (f" (top: {head})" if head else ""))


def drive_path(memory_path: str) -> str:
    """curiosity_drive.json beside the affect memory it shares a life with."""
    return os.path.join(os.path.dirname(memory_path), "curiosity_drive.json")

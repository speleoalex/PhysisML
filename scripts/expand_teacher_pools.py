#!/usr/bin/env python3
"""
Expand each level's target pool from training_files/<lang>/lexicon.json.

WHY
The curriculum's patterns are rich — article+noun, S+V+O, past, future,
comparatives, thesis+reason — but every level draws on the same ~20 nouns, so
the whole curriculum is 227 targets over 4,155 unique lines, and the trained
model emits 67 distinct words. The closed, repeated target pool is what
unblocked training in the first place (the model needs to meet a target many
times), and it is now the ceiling. This widens the pool without touching what
makes it work: still closed, still deterministic, still repeated.

WHAT IT PRESERVES
Each step keeps its description, prompt_template, thresholds and require_*
flags. Only `targets` is replaced, and every generated target carries the
fields the evaluator needs (noun/article/verb/adjective/object) so it can
reach '+++' — verified by scripts/validate_teacher_configs.py.

SEMANTICS
Nouns carry a `kind` and verbs declare which kinds they accept, so the
generator does not produce 'il pane dorme'. Adjectives agree in gender.

Usage:
    python3 scripts/expand_teacher_pools.py --per-step 24          # write
    python3 scripts/expand_teacher_pools.py --per-step 24 --dry-run
    python3 scripts/expand_teacher_pools.py --levels 1 2 3 --per-step 30
"""
import os
import sys
import json
import random
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260825          # fixed: the pools are committed, so they must not
                         # depend on the state of the global RNG


# ── lexicon helpers ─────────────────────────────────────────────────────────

def load_lexicon(lang: str) -> dict:
    with open(os.path.join(ROOT, "training_files", lang, "lexicon.json"),
              encoding="utf-8") as f:
        return json.load(f)


def phrase(n: dict) -> str:
    """'il cane', 'la casa', 'l acqua' — apostrophes are spaces in this corpus."""
    return f"{n['art']} {n['w']}"


def adj_for(n: dict, a: dict) -> str:
    return a["m"] if n["g"] == "m" else a["f"]


def di(n: dict) -> str:
    """'del cane', 'della casa', 'dell orso' — elision before a vowel."""
    if n["art"] == "l":
        return "dell " + n["w"]
    return ("del " if n["g"] == "m" else "della ") + n["w"]


def pick(rng, seq, k):
    seq = list(seq)
    if not seq:
        return []
    if k >= len(seq):
        rng.shuffle(seq)
        return seq
    return rng.sample(seq, k)


class Lex:
    def __init__(self, d):
        self.nouns = d["nouns"]
        self.verbs = d["verbs"]
        self.adjs = d["adjectives"]
        self.luoghi_verbo = d.get("luoghi_verbo", {})

    def of(self, *kinds):
        return [n for n in self.nouns if n["kind"] in kinds]

    def verbs_for(self, n, trans=None):
        # `puo` lists the verbs that make sense for this noun. Without it the
        # kind alone allows 'il lupo vola' and 'il pesce cammina'.
        allowed = set(n.get("puo", []))
        out = [v for v in self.verbs
               if n["kind"] in v["soggetti"] and v["pres"] in allowed]
        if trans is True:
            out = [v for v in out if v.get("trans")]
        if trans is False:
            out = [v for v in out if not v.get("trans")]
        return out

    def objects_for(self, v):
        # The object must accept THIS verb: 'acqua' is food but is drunk, not
        # eaten, and a table is not openable.
        kinds = v.get("oggetti", [])
        return [n for n in self.nouns
                if n["kind"] in kinds and v["pres"] in n.get("puo", [])]

    def adjs_for(self, n, dims=None, pol=None):
        out = [a for a in self.adjs if n["kind"] in a["kind"]]
        # Colours only where a colour means something.
        out = [a for a in out if "solo" not in a or n["w"] in a["solo"]]
        if dims:
            out = [a for a in out if a["dim"] in dims]
        if pol:
            out = [a for a in out if a["pol"] in pol]
        return out

    def adj_pair(self, rng, n, pol=None):
        """Two adjectives of DIFFERENT dimensions, so they cannot contradict
        ('il gelato è rosso perché è bianco')."""
        cands = self.adjs_for(n, pol=pol)
        rng.shuffle(cands)
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a["dim"] != b["dim"]:
                    return a, b
        return None

    def place_for(self, rng, verb):
        names = self.luoghi_verbo.get(verb, [])
        cands = [n for n in self.nouns if n["w"] in names]
        return rng.choice(cands) if cands else None

    def subj_verb_obj(self, rng, n):
        """A semantically valid (verb, object) for this subject, or None."""
        cands = [v for v in self.verbs_for(n, trans=True) if self.objects_for(v)]
        if not cands:
            return None
        v = rng.choice(cands)
        return v, rng.choice(self.objects_for(v))


# ── per-level generators ────────────────────────────────────────────────────
# Each returns a list of target dicts for one step. They mirror the patterns
# already in the level's config; only the lexicon they draw on is wider.

def g_art_noun(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "oggetto", "cibo",
                              "natura", "luogo"), k):
        p = phrase(n)
        out.append({"prompt": p, "expected": p + "!",
                    "article": n["art"], "noun": n["w"]})
    return out


def g_bare_noun(rng, lex, k):
    return [{"prompt": n["w"], "expected": n["w"] + "!", "noun": n["w"]}
            for n in pick(rng, lex.of("persona", "animale", "cibo"), k)]


def g_art_noun_verb(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "natura"), k):
        vs = lex.verbs_for(n, trans=False)
        if not vs:
            continue
        v = rng.choice(vs)
        p = f"{phrase(n)} {v['pres']}"
        out.append({"prompt": p, "expected": p + "!", "article": n["art"],
                    "noun": n["w"], "verb": v["pres"]})
    return out


def g_art_noun_adj(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "oggetto", "cibo",
                              "natura", "luogo"), k):
        As = lex.adjs_for(n)
        if not As:
            continue
        a = adj_for(n, rng.choice(As))
        p = f"{phrase(n)} {a}"
        out.append({"prompt": p, "expected": p + "!", "article": n["art"],
                    "noun": n["w"], "adjective": a})
    return out


def g_svo(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        p = f"{phrase(n)} {v['pres']} {phrase(o)}"
        out.append({"prompt": p, "expected": p + "!", "article": n["art"],
                    "noun": n["w"], "verb": v["pres"], "object": o["w"]})
    return out


def g_cosa_fa(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "natura"), k):
        vs = lex.verbs_for(n, trans=False)
        if not vs:
            continue
        v = rng.choice(vs)
        out.append({"prompt": f"cosa fa {phrase(n)}?",
                    "expected": f"{phrase(n)} {v['pres']}.",
                    "noun": n["w"], "verb": v["pres"]})
    return out


def g_cosa_verbo(rng, lex, k):
    """'cosa mangia il cane?' -> 'il cane mangia il pane.'"""
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        out.append({"prompt": f"cosa {v['pres']} {phrase(n)}?",
                    "expected": f"{phrase(n)} {v['pres']} {phrase(o)}.",
                    "noun": o["w"], "article": o["art"], "verb": v["pres"]})
    return out


def g_chi_verbo(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        out.append({"prompt": f"chi {v['pres']} {phrase(o)}?",
                    "expected": f"{phrase(n)} {v['pres']} {phrase(o)}.",
                    "noun": n["w"], "article": n["art"], "verb": v["pres"]})
    return out


def g_dove(rng, lex, k):
    out, places = [], None
    for n in pick(rng, lex.of("animale", "persona"), k):
        vs = lex.verbs_for(n, trans=False)
        if not vs:
            continue
        v = rng.choice(vs)
        pl = lex.place_for(rng, v["pres"])   # 'nuota in prato' is not a place
        if pl is None:
            continue
        out.append({"prompt": f"dove {v['pres']} {phrase(n)}?",
                    "expected": f"{phrase(n)} {v['pres']} in {pl['w']}.",
                    "noun": pl["w"], "verb": v["pres"]})
    return out


def g_com_e(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "oggetto", "cibo",
                              "natura", "luogo"), k):
        As = lex.adjs_for(n)
        if not As:
            continue
        a = adj_for(n, rng.choice(As))
        out.append({"prompt": f"com è {phrase(n)}?",
                    "expected": f"{phrase(n)} è {a}.",
                    "noun": a, "verb": "è", "adjective": a})
    return out


def g_due_verbi(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vs = lex.verbs_for(n, trans=False)
        if len(vs) < 2:
            continue
        v1, v2 = rng.sample(vs, 2)
        out.append({"prompt": f"cosa fa {phrase(n)}?",
                    "expected": f"{phrase(n)} {v1['pres']} e {v2['pres']}.",
                    "noun": v2["pres"], "verb": v1["pres"]})
    return out


def g_perche(rng, lex, k):
    MOTIVI = {"mangia": "ha fame", "beve": "ha sete", "dorme": "è stanco",
              "corre": "è veloce", "cerca": "ha fame", "canta": "è felice"}
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vs = [v for v in lex.verbs_for(n) if v["pres"] in MOTIVI]
        if not vs:
            continue
        v = rng.choice(vs)
        m = MOTIVI[v["pres"]]
        out.append({"prompt": f"perché {phrase(n)} {v['pres']}?",
                    "expected": f"{phrase(n)} {v['pres']} perché {m}.",
                    "noun": m.split()[-1], "verb": v["pres"]})
    return out


def g_passato(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        out.append({"prompt": f"cosa ha {v['past']} {phrase(n)}?",
                    "expected": f"{phrase(n)} ha {v['past']} {phrase(o)}.",
                    "noun": v["past"], "verb": v["past"]})
    return out


def g_futuro(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        out.append({"prompt": f"cosa {v['fut']} {phrase(n)} domani?",
                    "expected": f"domani {phrase(n)} {v['fut']} {phrase(o)}.",
                    "noun": v["fut"]})
    return out


def g_tre_tempi(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        out.append({"prompt": f"racconta i tre giorni {di(n)}",
                    "expected": f"oggi {v['pres']}, ieri ha {v['past']}, domani {v['fut']}.",
                    "noun": v["fut"]})
    return out


def g_comparativo(rng, lex, k):
    out = []
    pool = lex.of("animale", "persona", "oggetto", "natura")
    for _ in range(k * 3):
        if len(out) >= k:
            break
        a, b = rng.sample(pool, 2)
        if a["kind"] != b["kind"]:      # 'l amico è più vecchio della sedia'
            continue
        As = [x for x in lex.adjs_for(a) if x in lex.adjs_for(b)]
        if not As:
            continue
        adj = adj_for(a, rng.choice(As))
        out.append({"prompt": f"chi è più {adj}, {phrase(a)} o {phrase(b)}?",
                    "expected": f"{phrase(a)} è più {adj} {di(b)}.",
                    "noun": adj})
    return out


def g_preferenza(rng, lex, k):
    return [{"prompt": f"cosa ti piace mangiare?" if i % 2 == 0
                       else f"ti piace {phrase(n)}?",
             "expected": f"mi piace {phrase(n)}.",
             "noun": n["w"]}
            for i, n in enumerate(pick(rng, lex.of("cibo"), k))]


def g_perche_piace(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("cibo", "animale", "oggetto"), k):
        As = lex.adjs_for(n, pol=("+",))     # 'mi piace perché è vecchio' does not follow
        if not As:
            continue
        a = adj_for(n, rng.choice(As))
        out.append({"prompt": f"perché ti piace {phrase(n)}?",
                    "expected": f"mi piace {phrase(n)} perché è {a}.",
                    "noun": a})
    return out


def g_descrivi(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "oggetto", "natura"), k):
        pr = lex.adj_pair(rng, n)
        if not pr:
            continue
        a1, a2 = (adj_for(n, x) for x in pr)
        out.append({"prompt": f"descrivi {phrase(n)}",
                    "expected": f"{phrase(n)} è {a1} e {a2}.",
                    "noun": a2})
    return out


def g_tesi(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("cibo", "animale", "oggetto", "natura"), k):
        # The thesis is a JUDGEMENT and the reason a PROPERTY. The other way
        # round produces 'il gelato è giallo perché è buono', which does not
        # follow — and a curriculum that teaches non-sequiturs teaches noise.
        giud = lex.adjs_for(n, dims=("giudizio", "gusto"), pol=("+",))
        prop = lex.adjs_for(n, dims=("taglia", "colore", "temp", "moto", "stato"))
        if not giud or not prop:
            continue
        a1 = adj_for(n, rng.choice(giud))
        a2 = adj_for(n, rng.choice(prop))
        out.append({"prompt": f"{phrase(n)} è {a1}?",
                    "expected": f"secondo me {phrase(n)} è {a1} perché è {a2}.",
                    "noun": a2})
    return out


def g_perciò(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "natura"), k):
        pr = lex.adj_pair(rng, n, pol=("+",))   # '...è lenta e vecchia, perciò mi piace'
        if not pr:
            continue
        a1, a2 = (adj_for(n, x) for x in pr)
        out.append({"prompt": f"perché ti piace {phrase(n)}?",
                    "expected": f"{phrase(n)} è {a1} e {a2}, perciò mi piace.",
                    "noun": "perciò"})
    return out


def g_commenta(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "oggetto", "natura"), k):
        desc = lex.adjs_for(n, dims=("taglia", "colore", "temp", "moto", "stato"))
        giud = lex.adjs_for(n, dims=("giudizio",), pol=("+",))
        if not desc or not giud:
            continue
        a1 = adj_for(n, rng.choice(desc))
        a2 = adj_for(n, rng.choice(giud))   # '...è lungo, e questo è alto' is not a comment
        out.append({"prompt": f"commenta {phrase(n)}",
                    "expected": f"{phrase(n)} è {a1}, e questo è {a2}.",
                    "noun": a1})
    return out


def g_storia(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona"), k):
        vs = lex.verbs_for(n, trans=False)
        if not vs:
            continue
        v = rng.choice(vs)
        out.append({"prompt": f"cosa dice la storia {di(n)}",
                    "expected": f"la storia dice che {phrase(n)} {v['pres']}.",
                    "noun": v["pres"]})
    return out


def g_analizza(rng, lex, k):
    out = []
    for n in pick(rng, lex.of("animale", "persona", "natura"), k):
        vs = lex.verbs_for(n, trans=False)
        pr = lex.adj_pair(rng, n, pol=("+", "n"))
        if not vs or not pr:
            continue
        v = rng.choice(vs)
        a1, a2 = (adj_for(n, x) for x in pr)
        out.append({"prompt": f"analizza {phrase(n)}",
                    "expected": f"{phrase(n)} {v['pres']}. è {a1} e {a2}.",
                    "noun": a1})
    return out


# (level, step) -> generator. Steps not listed keep their hand-written targets:
# L0 is phonemes and L1/L3 identity steps are fixed answers, neither of which
# a lexicon can widen meaningfully.
PLAN = {
    (1, "A"): g_art_noun,   (1, "B"): g_bare_noun,
    (2, "A"): g_art_noun,   (2, "B"): g_art_noun_verb,
    (2, "C"): g_art_noun_adj, (2, "D"): g_svo,
    (3, "A"): g_art_noun,   (3, "C"): g_art_noun_verb, (3, "D"): g_cosa_fa,
    (4, "A"): g_cosa_verbo, (4, "B"): g_chi_verbo,     (4, "C"): g_dove,
    (5, "A"): g_com_e,      (5, "B"): g_due_verbi,     (5, "C"): g_perche,
    (6, "A"): g_passato,
    (7, "A"): g_futuro,     (7, "C"): g_tre_tempi,
    (8, "A"): g_comparativo, (8, "B"): g_preferenza,
    (8, "C"): g_perche_piace, (8, "D"): g_descrivi,
    (9, "A"): g_tesi,       (9, "B"): g_perciò,
    (10, "A"): g_commenta,  (10, "C"): g_storia, (10, "D"): g_analizza,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it")
    ap.add_argument("--levels", nargs="+", type=int,
                    default=list(range(11)))
    ap.add_argument("--per-step", type=int, default=24,
                    help="targets per generated step (default 24)")
    ap.add_argument("--mode", default="append", choices=["append", "replace"],
                    help="append (default) keeps the hand-written targets and "
                         "adds generated ones; replace discards them")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lex = Lex(load_lexicon(a.lang))
    grand_before = grand_after = 0

    for lvl in a.levels:
        path = os.path.join(ROOT, "training_files", a.lang, str(lvl),
                            "local_teacher.json")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        before = sum(len(s["targets"]) for s in cfg["steps"].values())
        touched = []
        for sname, step in cfg["steps"].items():
            gen = PLAN.get((lvl, sname))
            if gen is None:
                continue
            # A fresh RNG per step, seeded from the step's identity: the pools
            # are committed, so regenerating must reproduce them exactly, and
            # editing one step must not reshuffle the others.
            rng = random.Random(f"{SEED}-{a.lang}-{lvl}-{sname}")
            new = gen(rng, lex, a.per_step)
            if not new:
                continue
            if a.mode == "replace":
                step["targets"] = new
            else:
                # APPEND, not replace. Overwriting a pool drops whatever the
                # hand-written targets carried that the lexicon does not —
                # measured: replacing lost 16 words the model had already been
                # taught (aspetta, naso, importante, visto, vedrà, voglio...).
                # Losing vocabulary to widen vocabulary defeats the purpose.
                seen = {t.get("prompt") if isinstance(t, dict) else t
                        for t in step["targets"]}
                step["targets"] = step["targets"] + [
                    t for t in new if t["prompt"] not in seen]
            touched.append(f"{sname}:+{len(step['targets'])}")
        after = sum(len(s["targets"]) for s in cfg["steps"].values())
        grand_before += before
        grand_after += after
        print(f"  L{lvl}: {before} → {after} target"
              f"   [{', '.join(touched) if touched else 'invariato'}]")
        if not a.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\ntotale: {grand_before} → {grand_after} target"
          f"{'  (dry-run, nulla scritto)' if a.dry_run else ''}")


if __name__ == "__main__":
    main()

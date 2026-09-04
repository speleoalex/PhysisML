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
import re
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


def head(cls: str) -> str:
    """'un animale' -> 'animale'. The evaluator searches for the informative
    word, and the article is not it."""
    return re.sub(r"^(un|una|uno)\s+", "", cls)


def indef(n: dict) -> str:
    """'un ragno', 'una zucca', 'un aquila', 'uno sgabello'.

    Apostrophes are spaces in this corpus, so the feminine before a vowel is
    'un', not 'una'. The masculine takes 'uno' before s+consonant, z, gn, ps,
    pn, x and y — the same trigger as the definite 'lo'. Missing that rule put
    'cos è un sgabello?' and 'cos è un scoiattolo?' into L12 step E (four
    prompts, gold 'non lo so.' — the article is in the QUESTION, not the
    answer). Found when the ontology oracle rendered the same two nouns.
    """
    if n["g"] == "m":
        return "uno" if re.match(r"^(s[^aeiou]|z|gn|ps|pn|x|y)", n["w"]) else "un"
    return "un" if n["w"][0] in "aeiou" else "una"


def ask_forms(n: dict, cls_head: str) -> list:
    """Every way Italian asks 'what is X?' about this noun.

    Two axes — the interrogative ('cos è' / 'cosa è') and the article
    (definite / indefinite) — and BOTH must be crossed with the same gold, or
    the surface shape of the question becomes a feature the model reads the
    answer off instead of reading the noun.

    Measured on the finished 0-12 model, that is exactly what happened:
    'cosa è un cane?' answered 'non lo so.' The corpus had asked 'cosa è' only
    ever with a definite article (280 times, 280 definitions, 0 'non lo so'),
    while 'un' came overwhelmingly from L12's honesty step (700 of the 1040
    'cos è un X?'). Neither feature was about the noun, and the combination the
    user typed had appeared zero times, so the model fell back on the article.
    g_non_lo_so crosses the same two axes on the nouns whose honest answer IS
    'non lo so', which is what makes both axes uninformative.

    `cls_head` is the head of the gold's class, and it gates the 'cosa è'
    phrasing: 'cosa' is ALSO the head of the top class ('una cosa'), so with
    'cosa è il vento?' the gold 'il vento è una cosa.' shares every word with
    its own prompt and a bare echo scores full coverage — the self-poisoning
    loop the anti-echo exists to close (see g_cos_e). Pass '' when the gold
    cannot contain a class at all.
    """
    forms = []
    for h in ("cos è",) if cls_head == "cosa" else ("cos è", "cosa è"):
        forms.append(f"{h} {phrase(n)}?")
        # A mass noun takes no indefinite article ('cos è un acqua?') and a
        # unique referent takes the definite one ('cos è un sole?').
        if not n.get("mass") and not n.get("uniq"):
            forms.append(f"{h} {indef(n)} {n['w']}?")
    return forms


def dimostr(n: dict) -> str:
    return "questo" if n["g"] == "m" else "questa"


def intro(n: dict) -> str:
    """'questo è un ragno', 'questa è acqua', 'questo è il sole'.

    Mass nouns take no indefinite article ('questa è un acqua' is not Italian)
    and unique referents take the definite one ('questo è un sole' is not
    either).
    """
    if n.get("mass"):
        return f"{dimostr(n)} è {n['w']}"
    if n.get("uniq"):
        return f"{dimostr(n)} è {phrase(n)}"
    return f"{dimostr(n)} è {indef(n)} {n['w']}"


def pick(rng, seq, k):
    seq = list(seq)
    if not seq:
        return []
    if k >= len(seq):
        rng.shuffle(seq)
        return seq
    return rng.sample(seq, k)


def _retracted_words(lang: str) -> set:
    """Nouns whose admission of ignorance has been retracted by acquisition.

    Read through dynamic_model/retraction.py so the ledger has exactly one
    reader. Missing module or missing ledger both mean 'nothing retracted' —
    this script has to keep working in a checkout without the autonomy loop.
    """
    try:
        sys.path.insert(0, ROOT)
        from dynamic_model.retraction import retracted
        return set(retracted(lang))
    except Exception:
        return set()


class Lex:
    def __init__(self, d, lang: str = "it"):
        self.lang = lang
        self.nouns = d["nouns"]
        self.verbs = d["verbs"]
        self.adjs = d["adjectives"]
        self.luoghi_verbo = d.get("luoghi_verbo", {})
        onto = d.get("ontology", {})
        self.kind_class = onto.get("kind_class", {})
        self.classes = onto.get("classes", {})
        self.negative_pool = onto.get("negative_pool", [])
        self.unknown = d.get("unknown_nouns", [])
        # Names that appear in NO trained file at any level, and are never
        # given a class anywhere: the only honest answer about them is that the
        # model does not know. Kept out of unknown_nouns on purpose — those get
        # taught (non-probe) or measured (probe), and either would make these
        # answerable.
        # Three roles, and the split is the difference between a lesson that
        # survives acquisition and one that gets eaten by it:
        #   acquirable - taught 'non lo so'; the autonomy loop may acquire it,
        #                which RETRACTS the admission (dynamic_model/
        #                retraction.py). Every acquisition spends one example.
        #   reserve    - taught 'non lo so'; the loop must never touch it. The
        #                permanent control that acquisition has not eroded
        #                honesty, and the reason the published +67% keeps its
        #                meaning once the loop starts running.
        #   probe      - taught NOWHERE. The only way to ask whether the
        #                RELATION generalised rather than 32 names being
        #                memorised. Measured on the finished 0-12 model:
        #                7% honest on names outside the pool against 6/6 on the
        #                six inside it, which is why the pool grew.
        # A retracted noun leaves the pool here, not at the call sites: the
        # generator is re-run on every rebuild and would otherwise resurrect
        # the very admission the loop just retracted, silently.
        gone = _retracted_words(self.lang)
        bare = [n for n in d.get("bare_unknown_nouns", [])
                if n["w"] not in gone]
        self.bare_taught = [n for n in bare
                            if n.get("role", "acquirable") != "probe"]
        self.bare_reserve = [n for n in bare if n.get("role") == "reserve"]
        self.bare_probe = [n for n in bare if n.get("role") == "probe"]
        # The name the generators use: everything the curriculum teaches
        # 'non lo so' about, acquirable and reserve alike. The probe third is
        # deliberately absent — teaching it would destroy the measurement.
        self.bare_unknown = self.bare_taught

    def cls_of(self, n):
        """The class this noun belongs to, as it is SAID: 'un animale'.

        `kind` is a generator constraint, not an ontology — 'il sole e una
        natura' is not Italian. The per-noun `cls` wins over the kind default.
        """
        return n.get("cls") or self.kind_class.get(n["kind"], "")

    def wrong_cls_for(self, rng, n):
        """A class this noun does NOT belong to, for the negatives step."""
        mine = self.cls_of(n)
        cands = [c for c in self.negative_pool if c != mine]
        return rng.choice(cands) if cands else None

    def classified(self):
        """Nouns that have a class — all of them, if the lexicon is complete."""
        return [n for n in self.nouns if self.cls_of(n)]

    def unknown_of(self, probe=False):
        """The nouns held out of L0-L11.

        probe=False are the ones L12 teaches; probe=True are held out of L12 as
        well and exist only for scripts/curiosity_rate.py to measure on. Mixing
        them would measure the model on what it was just taught.
        """
        return [n for n in self.unknown if bool(n.get("probe")) == probe]

    def anchor_of_other_class(self, rng, cls):
        """A known noun whose class is NOT `cls`, to open a teaching prompt.

        Deliberately a different class: with an anchor of the SAME class
        ('il cane è un animale, questo è un ragno') the class is already in the
        prompt, so the model can produce the right answer by copying and the
        turn stops testing anything — and at L12 it could answer instead of
        asking and still be right.
        """
        cands = [n for n in self.nouns if self.cls_of(n) != cls]
        return rng.choice(cands) if cands else None

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
    """'chi mangia il pane?' -> 'il cane mangia il pane.'

    ONE subject per (verb, object). Any animal or person can eat the cake, and
    there are only 16 valid (verb, object) pairs against ~20 possible subjects,
    so drawing subjects at random produced the same question with four
    different gold answers ('chi mangia la torta?' -> cane / topo / orso /
    gallina). The level's contract is a closed pool with a fixed association
    repeated many times; two answers for one question breaks it, and the
    evaluator then marks a correct answer wrong.
    """
    out, taken = [], set()
    for n in pick(rng, lex.of("animale", "persona"), k):
        vo = lex.subj_verb_obj(rng, n)
        if not vo:
            continue
        v, o = vo
        key = (v["pres"], o["w"])
        if key in taken:
            continue
        taken.add(key)
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
        # 'racconta due cose del cane', NOT 'cosa fa il cane?': the latter is
        # L3 step D's prompt with a ONE-verb gold answer, so the same question
        # had two answers across levels and the retention matrix scored L3 as
        # regressed after L5. The imperative names the task, and it is the
        # register L7 step C already uses ('racconta i tre giorni del cane').
        out.append({"prompt": f"racconta due cose {di(n)}",
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
        # The 'due motivi:' prefix keeps this apart from L8 step C, whose
        # prompt is exactly 'perché ti piace il sole?' with a one-reason gold
        # ('mi piace il sole perché è caldo.'). The colon-prefixed task is the
        # form L3 step E already uses ('di un numero: uno').
        out.append({"prompt": f"due motivi: perché ti piace {phrase(n)}?",
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


def g_cos_e(rng, lex, k):
    """'cos e il gatto?' -> 'il gatto e un animale.'  The class is NOT in the
    prompt, so the anti-echo at level >= 4 credits it correctly.

    Both interrogatives and both articles, from ask_forms — asking each noun
    one single way made the SHAPE of the question predict the answer, and
    'cosa è un cane?' came back 'non lo so.' See the note there.

    'cosa è' is dropped only where the gold's class is 'una cosa', and not for
    elegance: 'cosa' is ALSO the head of the top class. With 'cosa è un cibo?'
    the gold 'un cibo è una cosa.' shares every word with its own prompt, so a
    bare echo of the prompt scores full coverage and SIGNAL 4 reinforces the
    model's parroting — the self-poisoning loop the anti-echo exists to close.
    Verified by test_prompt_echo_never_earns_echo_training. g_iperonimo, whose
    gold is ALWAYS a hypernym and is 'una cosa' for four of the classes, keeps
    'cos è' unconditionally.

    NOTE: this step and the two below IGNORE k and take EVERY classified noun,
    in lexicon order. Sampling k of them per step gave each step its own random
    subset: 'cane' was asked in none of the three and appeared only as the
    ANSWER to 'fai un esempio di animale', and eighteen nouns drilled in the
    yes/no steps were never asked 'cos è X?' at all. Measured on the finished
    0-12 model, that is exactly where the relation breaks — 'cos è il cane?'
    answers 'il cane è una cosa', the right shape with a superordinate instead
    of the class. A relation taught on half the nouns is not a relation.
    """
    out = []
    for n in lex.classified():
        c = lex.cls_of(n)
        gold = f"{phrase(n)} è {c}."
        # Every phrasing, same gold: see ask_forms. Asking each noun only one
        # way is what let the model answer from the article instead of the noun.
        for p in ask_forms(n, head(c)):
            out.append({"prompt": p, "expected": gold,
                        "article": n["art"], "noun": head(c)})
    return out


def g_is_a_si(rng, lex, k):
    """'il gatto è un animale?' -> 'sì, il gatto è un animale.'

    Every classified noun, like step A — see the note there.
    """
    out = []
    for n in lex.classified():
        c = lex.cls_of(n)
        out.append({"prompt": f"{phrase(n)} è {c}?",
                    "expected": f"sì, {phrase(n)} è {c}.",
                    "article": n["art"], "noun": head(c)})
    return out


def g_is_a_no(rng, lex, k):
    """'il pane è un animale?' -> 'no, il pane è un cibo.'

    Mandatory step. Without negatives the copula collapses and the model
    answers 'animale' to everything — the same dynamic that at L4 made the
    grade uncorrelated with correctness. The wrong class is drawn from
    negative_pool, which excludes the hypernyms: 'il cane è una cosa?' has no
    clean 'no', 'il cane è una persona?' does.
    """
    out = []
    for n in lex.classified():
        wrong = lex.wrong_cls_for(rng, n)
        if not wrong:
            continue
        c = lex.cls_of(n)
        out.append({"prompt": f"{phrase(n)} è {wrong}?",
                    "expected": f"no, {phrase(n)} è {c}.",
                    "article": n["art"], "noun": head(c)})
    return out


def g_membro(rng, lex, k):
    """'fai un esempio di animale' -> 'il cane è un animale.'  Class -> member,
    the inverse direction of step A: without it the relation is learned one way.

    ONE canonical member per class, not k of them. A class has many members, so
    drawing them at random gives the SAME prompt two different gold answers
    ('fai un esempio di persona' -> maestro, then -> fratello) and the evaluator
    then marks a correct answer wrong. The canonical member is the first of its
    class in lexicon order, so the pool is stable across regenerations.

    The prompt also deliberately avoids 'cosa è un animale?', which is step E's
    prompt with a different gold answer.
    """
    canonical = {}
    for n in lex.classified():
        c = lex.cls_of(n)
        if c.startswith(("un ", "una ")) and c not in canonical:
            canonical[c] = n      # first in lexicon order wins
    # `noun` is the member and `object` the class: here the class is already in
    # the prompt, so grade_by_coverage's anti-echo drops it and cannot see a
    # wrong one ('il cane è un persona.' scored +++). The evaluator checks
    # `object` too, which closes that hole.
    return [{"prompt": f"fai un esempio di {head(c)}",
             "expected": f"{phrase(n)} è {c}.",
             "article": n["art"], "noun": n["w"], "object": head(c)}
            for c, n in canonical.items()]


def g_iperonimo(rng, lex, k):
    """'cosa è un animale?' -> 'un animale è un essere vivente.'  Second level
    of the chain. Only classes that are sayable with an article and that have
    a hypernym: 'cosa è acqua?' is not Italian, and a root has no answer."""
    cands = [(c, h) for c, h in lex.classes.items()
             if h and c.startswith(("un ", "una "))]
    cands.sort()          # dict order is insertion order — sort for determinism
    return [{"prompt": f"cos è {c}?", "expected": f"{c} è {h}.",
             "noun": head(h)}
            for c, h in pick(rng, cands, k)]


def g_chiedi_ignoto(rng, lex, k):
    """'il pane è un cibo, questo è un ragno' -> 'cosa è un ragno?'

    The ask IS the gold answer: a model that has never met 'ragno' cannot know
    its class, and the anchor deliberately belongs to a DIFFERENT class, so
    guessing the class is not a way to be right. Asking is.

    Only the non-probe held-out nouns: the probe half is what
    scripts/curiosity_rate.py measures on, and teaching it would measure the
    model on what it was just taught.

    Then every bare unknown the level teaches, and that second half is the
    point. With six examples the lesson was six names: measured on the finished
    0-12 model, asked to introduce 'tegola' it answered 'cos è una zucca?' —
    the SHAPE of the question had generalised and the referent had not, because
    'cos è un ___?' had exactly six well-reinforced completions and none of
    them was read out of the prompt. Ask-rate on names the pool never contained:
    3%.

    A bare unknown has no class, so there is no anchor class to avoid: nothing
    in the prompt could be copied to produce the gold, since the gold is a
    question and not a class. Any classified anchor will do, and varying it is
    what keeps the shape from becoming attached to one class.

    Teaching both this and step E about the same name is not a contradiction,
    it is the loop: shown one, ask about it; asked about one, say you do not
    know. The two prompts are different shapes and carry one gold each.
    """
    out = []
    for n in pick(rng, lex.unknown_of(probe=False), k):
        a = lex.anchor_of_other_class(rng, n["cls"])
        if not a:
            continue
        out.append({
            "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
            "expected": f"cos è {indef(n)} {n['w']}?",
            "noun": n["w"]})
    for n in lex.bare_taught:
        a = lex.anchor_of_other_class(rng, "")
        if not a:
            continue
        out.append({
            "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
            "expected": f"cos è {indef(n)} {n['w']}?",
            "noun": n["w"]})
    return out


def g_chiedi_il_nome_giusto(rng, lex, k):
    """'la zucca è un cibo, questo è un falco' -> 'cos è un falco?'

    The negative for the failure the dry run actually produced, and it is not
    the same failure as guessing a class. Asked to introduce a name it had never
    met, the model asked a well-formed question about a DIFFERENT name — one of
    the six it had been taught to ask about. 43% of the responses, against 0%
    that asked about the right one.

    Widening step A teaches the model to read the referent out of the prompt.
    This step makes the wrong referent AVAILABLE in the prompt and wrong: the
    anchor is drawn from exactly the six nouns step A drilled, so
    'cos è una zucca?' is a question the prompt invites and the gold refuses.
    A negative with the distractor absent teaches nothing about choosing.

    The anchors cycle rather than being sampled: with six of them and a random
    draw, some would appear five times and some none, and the one that never
    appeared is the one the model would keep answering with. `tamburo` and
    `zucca` are in there by name — they are what it confabulated with.

    The anchor's statement is one the model has been taught to make (step C
    teaches 'cos è una zucca?' -> 'la zucca è un cibo.'), so the prompt is
    true. An anchor asserting something false would teach two lessons at once
    and neither cleanly.
    """
    anchors = lex.unknown_of(probe=False)
    if not anchors:
        return []
    out = []
    for i, n in enumerate(lex.bare_taught):
        a = anchors[i % len(anchors)]
        out.append({
            "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
            "expected": f"cos è {indef(n)} {n['w']}?",
            "noun": n["w"]})
    return out


def g_non_chiedere(rng, lex, k):
    """'il cane è un animale, questo è un pane' -> 'il pane è un cibo.'

    The discriminating case, and the reason the level measures curiosity rather
    than a tic: on a noun it already knows the model must answer, not ask. The
    anchor's class differs from the answer's, so the class cannot be copied
    out of the prompt.
    """
    out = []
    for n in pick(rng, lex.classified(), k):
        c = lex.cls_of(n)
        a = lex.anchor_of_other_class(rng, c)
        if not a:
            continue
        out.append({
            "prompt": f"{phrase(a)} è {lex.cls_of(a)}, {intro(n)}",
            "expected": f"{phrase(n)} è {c}.",
            "article": n["art"], "noun": head(c)})
    return out


def g_consolida(rng, lex, k):
    """'cosa è un ragno?' -> 'il ragno è un animale.'

    The turn after the teacher has answered: the fact the model asked for has
    to end up in the weights, or asking bought nothing.
    """
    return [{"prompt": f"cos è {indef(n)} {n['w']}?"
                       if not n.get("mass") else f"cos è {n['w']}?",
             "expected": f"{n['art']} {n['w']} è {n['cls']}.",
             "article": n["art"], "noun": head(n["cls"])}
            for n in pick(rng, lex.unknown_of(probe=False), k)]


def g_conferma_nuovo(rng, lex, k):
    """'il ragno è un animale?' -> 'sì, il ragno è un animale.'  Retention
    check on the nouns learned in this level."""
    return [{"prompt": f"{n['art']} {n['w']} è {n['cls']}?",
             "expected": f"sì, {n['art']} {n['w']} è {n['cls']}.",
             "article": n["art"], "noun": head(n["cls"])}
            for n in pick(rng, lex.unknown_of(probe=False), k)]


def g_non_lo_so(rng, lex, k):
    """'cos è un falco?' -> 'non lo so.'

    Level 12 teaches asking, but only inside the two-clause shape it drills
    ('il pane è un cibo, questo è un ragno' -> 'cos è un ragno?'). Measured on
    the finished 0-12 model, a BARE question about a name it has never met is
    answered with a confabulation instead: 'chi è zibaldone?' -> 'il fratello è
    una persona.' The shape was never taught, so the model falls back on the
    nearest pattern it has.

    Every phrasing of the open question, from ask_forms, plus the yes/no,
    because all of them have the same honest answer and the yes/no is the one
    L11 step B teaches to answer 'sì' for a known noun. The contrast is the
    lesson: the class of a noun you know, 'non lo so' for one you do not.

    The phrasings matter as much as the names. Asking these six only as
    'cos è un X?' made the indefinite article itself mean 'unknown' — 700 of
    the 1040 'cos è un X?' in the 0-12 corpus answered 'non lo so' — while
    'cosa è' never appeared here at all and so came to mean 'known'. The model
    then answered from the article and the interrogative, which say nothing
    about the noun. Crossing both axes on BOTH sides (here and in g_cos_e) is
    what puts the decision back on the name.

    The gold has no class in it, so ask_forms's 'una cosa' guard does not
    apply and every phrasing is emitted.

    The gold is deliberately NOT a question. Asking a question back at a bare
    'cos è un falco?' would only repeat it. That is why this does not feed
    scripts/curiosity_rate.py, which measures the two-clause shape where asking
    IS the informative answer.
    """
    # `noun` is the keyword the grader looks for in the RESPONSE, so it has to
    # be a word of the gold, not the word being asked about: with noun='falco'
    # the answer 'non lo so.' could never earn +++ and the step would train
    # against itself. validate_teacher_configs.py catches exactly this.
    # 'so' is safe as a keyword: the grader matches on word boundaries, and its
    # compact fallback only applies to words of four characters or more, so it
    # cannot match inside 'sole' or 'sorella'.
    out = []
    for n in lex.bare_unknown:
        for p in ask_forms(n, ""):
            out.append({"prompt": p, "expected": "non lo so.", "noun": "so"})
        out.append({"prompt": f"{phrase(n)} è un animale?",
                    "expected": "non lo so.", "noun": "so"})
    return out


def g_ripasso_cosa_fa(rng, lex, k):
    """L3 step D's questions, asked again at L11 and L12.

    'cosa fa il cane?' -> 'il cane dorme.' is taught at L2, L3 and L4 and then
    never again: eight levels with no 'cosa fa' at all, while L11 drills
    'cos è il cane?' -> 'il cane è un animale.' 240 times. Measured on the
    finished 0-12 model, 'cosa fa il cane?' now answers 'il cane è un
    animale.' — the prefix 'cosa ... il cane' has exactly one recently
    reinforced continuation, and the verb no longer steers it. This is the only
    interrogative of L2-L4 that no later level revisits.

    COPIED from the committed L3 pool, not regenerated. A prompt may carry only
    ONE gold answer (validate_teacher_configs.py, which was added after this
    very question got two), and g_cosa_fa draws a random verb per noun from a
    per-step RNG: generating it again here would give 'cosa fa il cane?' a
    second, contradictory answer at another level. Rehearsal means the same
    question with the same answer, so the source pool is the right source.
    """
    src = os.path.join(ROOT, "training_files", LANG, "3", "local_teacher.json")
    if not os.path.exists(src):
        return []
    with open(src, encoding="utf-8") as f:
        step = json.load(f)["steps"].get("D", {})
    # dicts only: L0-L3 pools may still hold bare strings, which carry none of
    # the fields the evaluator needs to reach '+++'.
    return [dict(t) for t in step.get("targets", []) if isinstance(t, dict)]


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
    (11, "A"): g_cos_e,     (11, "B"): g_is_a_si, (11, "C"): g_is_a_no,
    (11, "D"): g_membro,    (11, "E"): g_iperonimo,
    (11, "F"): g_ripasso_cosa_fa,
    (12, "A"): g_chiedi_ignoto, (12, "B"): g_non_chiedere,
    (12, "C"): g_consolida,     (12, "D"): g_conferma_nuovo,
    (12, "E"): g_non_lo_so,     (12, "F"): g_ripasso_cosa_fa,
    (12, "G"): g_chiedi_il_nome_giusto,
}

# Set by main(). g_ripasso_cosa_fa reads another level's committed pool, and a
# generator only receives (rng, lex, k) — the language is not among them.
LANG = "it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it")
    ap.add_argument("--levels", nargs="+", type=int,
                    default=list(range(13)))
    ap.add_argument("--per-step", type=int, default=24,
                    help="targets per generated step (default 24)")
    ap.add_argument("--mode", default="append", choices=["append", "replace"],
                    help="append (default) keeps the hand-written targets and "
                         "adds generated ones; replace discards them")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    global LANG
    LANG = a.lang
    lex = Lex(load_lexicon(a.lang), a.lang)
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

        # A prompt may carry only ONE gold answer. Dedup above only compares a
        # new target against the targets already present, never against the
        # rest of its own batch, so a generator that maps two items onto the
        # same prompt slipped duplicates straight into the committed pool
        # (measured: 'chi mangia la torta?' ended up with four gold answers).
        # Keep the first occurrence — deterministic, and it preserves whatever
        # the pool was originally built with.
        for sname, step in cfg["steps"].items():
            seen_p, kept, dropped = set(), [], []
            for t in step["targets"]:
                key = t.get("prompt") if isinstance(t, dict) else t
                if key in seen_p:
                    dropped.append(key)
                    continue
                seen_p.add(key)
                kept.append(t)
            if dropped:
                step["targets"] = kept
                print(f"    ⚠ L{lvl} step {sname}: dropped "
                      f"{len(dropped)} targets with a duplicate prompt "
                      f"({', '.join(sorted(set(dropped))[:3])}...)")

        after = sum(len(s["targets"]) for s in cfg["steps"].values())
        grand_before += before
        grand_after += after
        print(f"  L{lvl}: {before} → {after} targets"
              f"   [{', '.join(touched) if touched else 'unchanged'}]")
        if not a.dry_run:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"\ntotal: {grand_before} → {grand_after} targets"
          f"{'  (dry-run, nothing written)' if a.dry_run else ''}")


if __name__ == "__main__":
    main()

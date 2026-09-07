"""
How one language SAYS the handful of sentences the curriculum generates.

WHY THIS EXISTS
The ontology levels (L11 is-a, L12 curiosity) are the first ones whose text is
not copied from a hand-written pool but built by scripts/expand_teacher_pools.py
out of the lexicon. Every one of those sentences was an Italian f-string
compiled into the generator: `f"{art} {w} è {cls}."`, `f"sì, ..."`,
`f"fai un esempio di ..."`, `"non lo so."`. So was the question the ask gate
raises (`modulator.ASK_FORM`), the question the epistemic probe forces
(`epistemic.prefix_for`), the question LocalTeacher recognises the model asking
(`_QUESTION_RE`) and the praise the trainer strips off a prompt (`_PRAISE_RE`).

Running `--lang en` against those would not fail. It would build an English
level out of Italian sentences and train for hours on it — the exact shape of
the axiom leak that dynamic_model/language.py was written to end.

So the sentences live in the manifest, next to the axioms and the polarity:

    training_files/<lang>/language.json   ->   "surface": { ... }

WHAT STAYS IN PYTHON
Morphology, and only where it is a rule rather than a word. Italian picks
'un/uno/una' from the shape of the noun and 'questo/questa' from its gender:
that is an algorithm, and no amount of JSON makes it data. It is named by the
manifest (`"indef_rule": "italian"`) and implemented here.

Every other language uses the default rule, which reads the article off the
noun itself (`"indef": "a"` in the lexicon) — the general escape hatch, since
a lexicon can always carry the article a word takes, whereas the code cannot
always derive it.

A language whose manifest has no "surface" block gets an empty Surface, and
every accessor raises instead of quietly answering in Italian. The generators
check `available` first and say what is missing.
"""
import re

from . import language as _language


# The gender key used for a language with no grammatical gender: the lexicon
# has no 'g' field, so the lookup falls through to this.
_NO_GENDER = ""

# Italian takes 'uno' before s+consonant, z, gn, ps, pn, x and y — the same
# trigger as the definite 'lo'. Apostrophes are spaces in this corpus, so the
# feminine before a vowel is 'un', not 'una'.
_IT_UNO_RE = re.compile(r"^(s[^aeiou]|z|gn|ps|pn|x|y)")


class MissingSurface(KeyError):
    """A language asked for a sentence its manifest does not define."""


class Surface:
    """The sentences one language builds its ontology levels out of."""

    def __init__(self, lang: str, data: dict = None, polarity: dict = None):
        self.lang = lang
        self._d = data or {}
        self._pol = polarity or {}

    # ── availability ────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return bool(self._d)

    def _need(self, key: str):
        if key not in self._d:
            raise MissingSurface(
                f"training_files/{self.lang}/language.json has no "
                f"\"surface\".\"{key}\": the ontology levels cannot be "
                f"generated for '{self.lang}' without it.")
        return self._d[key]

    # ── words ───────────────────────────────────────────────────────────────
    @property
    def copula(self) -> str:
        """'è' / 'is' — the verb every is-a sentence is built on."""
        return self._need("copula")

    @property
    def yes(self) -> str:
        return (self._pol.get("yes") or [self._need("yes")])[0]

    @property
    def no(self) -> str:
        return (self._pol.get("no") or [self._need("no")])[0]

    @property
    def admission(self) -> str:
        """'non lo so.' — the whole gold of L12 step E."""
        return self._need("admission")

    @property
    def admissions(self) -> tuple:
        """Every whole answer that counts as an admission of ignorance.

        Whole answers, not a pattern: the retraction test lowercases the gold
        and strips its final punctuation, and 'non lo so perché è nuovo.' is
        an assertion, not an admission. `admission_re` is the loose reading
        used on a free response; this is the strict one used on a gold.

        Defaults to `admission` alone, which is always true; a manifest lists
        the shorter forms the same language also accepts.
        """
        forms = self._d.get("admissions") or [self._need("admission")]
        return tuple(f.strip().lower().rstrip(".!?") for f in forms if f)

    @property
    def admission_keyword(self) -> str:
        """The word of `admission` the grader looks for in the response.

        It has to be a word of the ANSWER, not of the question: with the noun
        being asked about, 'non lo so.' could never earn '+++' and the step
        would train against itself.
        """
        return self._need("admission_keyword")

    @property
    def ask_heads(self) -> list:
        """Every way this language opens 'what is ...?', canonical first.

        Italian has two ('cos è', 'cosa è') and crossing both with both
        articles is what stopped the SHAPE of the question from predicting the
        answer — see ask_forms in scripts/expand_teacher_pools.py.
        """
        return list(self._need("ask_heads"))

    @property
    def top_class_head(self) -> str:
        """The head of the root class ('cosa' / 'thing').

        An ask head whose interrogative IS this word is dropped for golds of
        that class: 'cosa è un cibo?' -> 'un cibo è una cosa.' shares every
        word with its own prompt, and a bare echo then scores full coverage.
        """
        return self._d.get("top_class_head", "")

    # ── morphology ──────────────────────────────────────────────────────────
    def phrase(self, n: dict) -> str:
        """'il cane', 'la casa', 'the cat' — apostrophes are spaces here."""
        return f"{n['art']} {n['w']}"

    def indef(self, n: dict) -> str:
        """'un ragno' / 'a spider' — the indefinite article this noun takes."""
        if self._d.get("indef_rule") == "italian":
            if n.get("g") == "m":
                return "uno" if _IT_UNO_RE.match(n["w"]) else "un"
            return "un" if n["w"][0] in "aeiou" else "una"
        art = n.get("indef")
        if not art:
            raise MissingSurface(
                f"noun {n.get('w')!r} in training_files/{self.lang}/"
                f"lexicon.json has no \"indef\" field, and '{self.lang}' "
                f"declares no indef_rule that could derive one.")
        return art

    def demonstrative(self, n: dict) -> str:
        """'questo' / 'questa' / 'this'."""
        table = self._need("demonstrative")
        return table.get(n.get("g", _NO_GENDER)) or table[_NO_GENDER]

    def head(self, cls: str) -> str:
        """'un animale' -> 'animale', 'an animal' -> 'animal'.

        The evaluator searches the response for the informative word, and the
        article is not it.
        """
        return re.sub(self._need("class_article_re"), "", cls)

    def has_class_article(self, cls: str) -> bool:
        """True for a class that can be the subject of a sentence of its own.

        'cosa è acqua?' is not Italian and 'what is water?' has no is-a answer
        in this ontology: the mass classes are excluded from the hypernym and
        the class->member steps by this test.
        """
        return bool(re.match(self._need("class_article_re"), cls))

    # ── sentences ───────────────────────────────────────────────────────────
    def is_a(self, n: dict, cls: str) -> str:
        """'il cane è un animale.'"""
        return f"{self.phrase(n)} {self.copula} {cls}."

    def is_a_q(self, n: dict, cls: str) -> str:
        """'il cane è un animale?'"""
        return f"{self.phrase(n)} {self.copula} {cls}?"

    def cls_is_a(self, cls: str, hyper: str) -> str:
        """'un animale è un essere vivente.'"""
        return f"{cls} {self.copula} {hyper}."

    def intro(self, n: dict) -> str:
        """'questo è un ragno', 'questa è acqua', 'this is the sun'.

        Mass nouns take no indefinite article and unique referents take the
        definite one.
        """
        dem = self.demonstrative(n)
        if n.get("mass"):
            return f"{dem} {self.copula} {n['w']}"
        if n.get("uniq"):
            return f"{dem} {self.copula} {self.phrase(n)}"
        return f"{dem} {self.copula} {self.indef(n)} {n['w']}"

    def ask(self, n: dict) -> str:
        """'cos è un ragno?' — the canonical open question about a noun.

        Mass and unique referents take the definite article, the rest the
        indefinite one, exactly as ask_forms does.
        """
        head = self.ask_heads[0]
        if n.get("mass") or n.get("uniq"):
            return f"{head} {self.phrase(n)}?"
        return f"{head} {self.indef(n)} {n['w']}?"

    def ask_definite(self, n: dict) -> str:
        """'cosa è il ragno?' — the LAST ask head with the definite article.

        The retry LocalTeacher issues after a question it will not answer, so
        it must be a prompt the model has actually been trained on.
        """
        return f"{self.ask_heads[-1]} {self.phrase(n)}?"

    def ask_class(self, cls: str) -> str:
        """'cos è un animale?' — the open question about a class."""
        return f"{self.ask_heads[0]} {cls}?"

    def ask_forms(self, n: dict, cls_head: str) -> list:
        """Every way this language asks 'what is X?' about this noun.

        Two axes — the interrogative and the article — and BOTH are crossed
        with the same gold, or the surface shape of the question becomes a
        feature the model reads the answer off instead of reading the noun.
        Measured on the finished Italian 0-12 model, that is exactly what
        happened: 'cosa è un cane?' answered 'non lo so.'

        `cls_head` is the head of the gold's class and gates the interrogative
        that collides with it (see top_class_head). Pass '' when the gold
        cannot contain a class at all.
        """
        forms = []
        for h in self.ask_heads:
            if cls_head and h.split()[0] == cls_head:
                continue
            forms.append(f"{h} {self.phrase(n)}?")
            # A mass noun takes no indefinite article ('cos è un acqua?') and a
            # unique referent takes the definite one ('cos è un sole?').
            if not n.get("mass") and not n.get("uniq"):
                forms.append(f"{h} {self.indef(n)} {n['w']}?")
        return forms

    def example_request(self, cls: str) -> str:
        """'fai un esempio di animale' / 'give an example of an animal'.

        Both the class and its head are offered to the template: Italian drops
        the article, English keeps it.
        """
        return self._need("example_request").format(
            cls=cls, head=self.head(cls))

    def confirm(self, n: dict, cls: str) -> str:
        """'sì, il cane è un animale.'"""
        return f"{self.yes}, {self.is_a(n, cls)}"

    def correct(self, n: dict, cls: str) -> str:
        """'no, il pane è un cibo.'"""
        return f"{self.no}, {self.is_a(n, cls)}"

    # ── patterns ────────────────────────────────────────────────────────────
    @property
    def question_re(self):
        """Recognises the model ASKING: 'cosa è un ragno?' -> 'ragno'.

        Matched loosely on purpose — a model that has not yet learned to close
        a question still asked one. Exactly one capturing group, the noun.
        """
        return re.compile(self._need("question_pattern"), re.IGNORECASE)

    @property
    def ask_openers(self) -> tuple:
        """The words a question can OPEN with, for the measurement that has
        to recognise one before the model can close it with a '?'.

        Defaults to the first word of every ask head, which is the minimum
        that is certainly true; a manifest lists more because a model asks
        'who', 'why' and 'how' long before anyone taught it to.
        """
        words = self._d.get("ask_openers") or [h.split()[0]
                                               for h in self.ask_heads]
        return tuple(w.lower() for w in words)

    @property
    def admission_re(self):
        """Recognises an admission of ignorance in a free response.

        Looser than `admission` itself: the gold is one sentence, but a model
        that produced 'non so' or 'i don t know' declared its ignorance just
        as honestly, and counting those as failures halved a measured
        separation once already (see scripts/curiosity_rate.py).
        """
        pat = self._d.get("admission_pattern")
        if not pat:
            words = self._need("admission").strip(" .!?").split()
            pat = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
        return re.compile(pat, re.IGNORECASE)

    @property
    def retry_re(self):
        """The retry prefix the teacher prepends from L4 on ('ancora. ' /
        'again. ').

        Below L4 the retry prefix repeats the prompt instead, and that shape
        is stripped by _strip_demo in train_curriculum.py, not by this.
        """
        word = self._need("retry_prefix").strip()
        return re.compile(rf"^{re.escape(word)}\s*", re.IGNORECASE)

    @property
    def praise_re(self):
        """Leading praise on a teacher prompt: 'bravo! di baba' -> 'di baba'."""
        words = self._need("praise_words")
        alt = "|".join(re.escape(w) for w in sorted(words, key=len,
                                                    reverse=True))
        return re.compile(rf"^({alt})[!.,:\s]+", re.IGNORECASE)

    def __repr__(self) -> str:
        return (f"Surface({self.lang!r}, "
                f"{'defined' if self.available else 'missing'})")


_CACHE = {}


def load(lang: str = _language.DEFAULT_LANG) -> Surface:
    """The Surface of one language. Cached: the generators call it per target."""
    if lang not in _CACHE:
        lg = _language.load(lang)
        _CACHE[lang] = Surface(lang, lg.surface, lg.polarity)
    return _CACHE[lang]

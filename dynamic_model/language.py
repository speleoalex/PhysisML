"""
The description of one language, read from disk instead of from the source.

Everything that differs between Italian and English used to be a literal in a
Python file: the axiom words in train_curriculum.py, the function words in
stop_words.py, the Hugging Face card and the `"language"` field in
export_hf.py. Adding a third language meant editing four modules and hoping
none was missed -- and the ones that WERE missed failed quietly. The English
build ran for hours protecting the Italian axiom 'mamma', which the English
vocabulary splits into m|am|ma, because nothing in the code could notice that
the words did not belong to the language being trained.

So a language is now a folder plus an optional manifest:

    training_files/<lang>/            the curriculum (levels, teachers, lexicon)
    training_files/<lang>/language.json   what the code cannot infer

Everything the manifest does not say is derived from the conventions the repo
already follows, so most of the file is optional:

    tokenizer   dynamic_model/data/tokenizer_<lang>.json
    probe_set   dynamic_model/data/probe_set_<lang>.json
    card        huggingface/README.<lang>.md
    out_dir     hf_upload_<lang>
    level text  training_files/<lang>/<N>/sentences_level<N>.txt

Italian is the exception on every one of those: it keeps the historical
filenames (tokenizer_8k.json, probe_set.json, huggingface/README.md,
hf_upload/) because every published checkpoint, baseline fingerprint and Hub
revision was made against them, and renaming them would orphan the lot.

Only the axioms and the stop words have no convention to fall back on -- they
are words, and words cannot be derived from a language code. A language whose
manifest omits them gets an empty list and the caller says so on screen, which
is the honest outcome: no protection at all is better than protection applied
to another language's subwords.

This module deliberately imports nothing but the standard library. It is read
by the training loop, by the REPL and by the export scripts, and two of those
must stay importable without torch.
"""
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_LANG  = "it"
MANIFEST_NAME = "language.json"

# Italian predates the per-language layout: these are the paths its published
# artifacts actually live at. Any other language follows the conventions in
# the module docstring.
_HISTORICAL_IT = {
    "tokenizer": "dynamic_model/data/tokenizer_8k.json",
    "probe_set": "dynamic_model/data/probe_set.json",
    "card":      "huggingface/README.md",
    "out_dir":   "hf_upload",
}

# Filenames that carry a level number. Same rule as the paths above: Italian
# keeps what is committed, every other language gets a name in its own
# language. 'frasi_livello7.txt' inside training_files/en was the same kind of
# leak as the Italian axioms -- silent, and only visible on a directory listing.
_HISTORICAL_IT_LEVEL = {
    "level_text": "frasi_livello{level}.txt",
}
_LEVEL_PATTERNS = {
    "level_text": "sentences_level{level}.txt",
}


def manifest_path(lang: str) -> str:
    return os.path.join(_ROOT, "training_files", lang, MANIFEST_NAME)


def available() -> list:
    """Every language with a curriculum on disk, in alphabetical order.

    A language exists because training_files/<lang>/ exists and holds at least
    one numbered level -- not because it is named anywhere in the code.
    """
    base = os.path.join(_ROOT, "training_files")
    out = []
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        if any(e.isdigit() and os.path.isdir(os.path.join(d, e))
               for e in os.listdir(d)):
            out.append(name)
    return out


class Language:
    """One language's paths and word lists. Missing keys fall back to the
    conventions, so a manifest only has to carry what it cannot derive."""

    def __init__(self, code: str, data: dict = None):
        self.code = code
        self._d   = data or {}

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        """Human-readable name, for screen output only."""
        return self._d.get("name", self.code)

    @property
    def has_manifest(self) -> bool:
        return bool(self._d)

    # ── paths ───────────────────────────────────────────────────────────────
    def _path(self, key: str, pattern: str) -> str:
        if key in self._d:
            return self._d[key]
        if self.code == DEFAULT_LANG:
            return _HISTORICAL_IT[key]
        return pattern.format(lang=self.code)

    @property
    def tokenizer(self) -> str:
        return self._path("tokenizer", "dynamic_model/data/tokenizer_{lang}.json")

    @property
    def probe_set(self) -> str:
        return self._path("probe_set", "dynamic_model/data/probe_set_{lang}.json")

    @property
    def card(self) -> str:
        return self._path("card", "huggingface/README.{lang}.md")

    @property
    def out_dir(self) -> str:
        return self._path("out_dir", "hf_upload_{lang}")

    def level_file(self, key: str, level: int) -> str:
        """The name of a per-level file this language writes, e.g. the curated
        text scripts/gen_level_text.py generates for `level`."""
        if key in self._d:
            pattern = self._d[key]
        elif self.code == DEFAULT_LANG:
            pattern = _HISTORICAL_IT_LEVEL[key]
        else:
            pattern = _LEVEL_PATTERNS[key]
        return pattern.format(level=level, lang=self.code)

    @property
    def curriculum_dir(self) -> str:
        return os.path.join("training_files", self.code)

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join("models", "checkpoints", self.code)

    @property
    def hf_repo(self) -> str:
        """The Hub repo this language publishes to, or '' if none is declared.

        There is no convention for this one on purpose: guessing a repo name
        and pushing to it is not a mistake that can be undone.
        """
        return self._d.get("hf_repo", "")

    # ── words ───────────────────────────────────────────────────────────────
    def axioms(self, kind: str) -> list:
        """[(text, protection), ...] for 'phonetic' or 'grammar'.

        Empty when the manifest is silent. The caller prints the emptiness --
        see the module docstring for why silence beats a wrong-language list.
        """
        raw = (self._d.get("axioms") or {}).get(kind, [])
        out = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), float(item[1])))
            else:
                out.append((str(item), 0.7))
        return out

    @property
    def teacher_fallback(self) -> dict:
        """The tutor prompt material used when a level ships no
        teacher_prompt.md: {"praise": str, "bands": {band: {profile, steps}}}.

        Empty when the manifest is silent, and the caller then builds a prompt
        with no worked examples. The examples are the one part of a tutor
        prompt that cannot be derived from a language code, and the previous
        fallback -- Italian, whatever --lang said -- taught an English level
        to answer "di' 'mamma'".
        """
        return self._d.get("teacher_fallback") or {}

    @property
    def polarity(self) -> dict:
        """The words this language answers a closed question with.

        {"yes": [spellings...], "no": [spellings...]}, the first spelling of
        each being the canonical one. Empty when the manifest is silent, which
        turns the polarity check off for that language -- the alternative is
        an Italian regex that finds no polarity in an English gold and lets
        the negative is-a steps through ungraded, which is the bug it exists
        to prevent.
        """
        raw = self._d.get("polarity") or {}
        out = {}
        for key in ("yes", "no"):
            words = [str(w) for w in raw.get(key, []) if str(w)]
            if words:
                out[key] = words
        return out

    @property
    def stop_words(self) -> set:
        """Function words, for the callers that separate content from grammar."""
        return set(self._d.get("stop_words", []))

    def __repr__(self) -> str:
        return (f"Language({self.code!r}, manifest="
                f"{'yes' if self.has_manifest else 'no'})")


def load(lang: str = DEFAULT_LANG) -> Language:
    """The manifest of one language, or a conventions-only Language."""
    path = manifest_path(lang)
    if not os.path.exists(path):
        return Language(lang)
    with open(path, encoding="utf-8") as f:
        return Language(lang, json.load(f))


def detect(tokenizer_path: str) -> str:
    """Which language a tokenizer file belongs to, or '' if none matches.

    A checkpoint carries no language marker -- config.json holds sizes and
    nothing else -- so the vocabulary is the only fingerprint available. It is
    a good one: two languages trained by scripts/train_tokenizer.py never
    agree on their merge list, and comparing the whole vocabulary is exact
    where comparing the token count is not.

    This exists so an export cannot label English weights as Italian. There is
    one models/active.pt for every language, and after an English build it
    holds English weights while every default in the export chain still says
    'it'.
    """
    try:
        with open(_abs(tokenizer_path), encoding="utf-8") as f:
            want = json.load(f).get("vocab", {})
    except (OSError, ValueError):
        return ""
    if not want:
        return ""
    best, best_score = "", 0
    for code in available():
        cand = _abs(load(code).tokenizer)
        if not os.path.exists(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                have = json.load(f).get("vocab", {})
        except (OSError, ValueError):
            continue
        score = _overlap(want, have)
        if score > best_score:
            best, best_score = code, score
    return best


def _overlap(a: dict, b: dict) -> int:
    """How much of the smaller vocabulary survives verbatim in the larger,
    or 0 when the two are not the same vocabulary at all.

    A tokenizer only ever grows: each level adds tokens for the words it
    introduces and never renumbers the ones already there, so a checkpoint's
    vocabulary contains the base vocabulary of the language it was trained on.
    Comparing sizes instead -- the same count give or take the special tokens
    -- stopped recognising a language as soon as it had learned anything:
    models/checkpoints/it/level_13/tokenizer.json is twenty tokens past
    tokenizer_8k.json and matched no language at all, which sent the caller
    to its Italian default for every language equally.

    A handful of ids may legitimately disagree (EOS is registered into a level
    snapshot at whatever slot was free), so the test is overwhelming agreement
    rather than exact containment.
    """
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    if not small:
        return 0
    agree = disagree = 0
    for k, v in small.items():
        other = large.get(k)
        if other is None:
            continue
        if other == v:
            agree += 1
        else:
            disagree += 1
    if disagree > 8 or agree < 0.9 * len(small):
        return 0
    return agree


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)

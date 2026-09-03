# Contributing to PhysisML

Thanks for looking. This is a small research repository: an Italian
developmental curriculum, a ~23.6M-parameter transformer trained on it from
scratch, and the measurements that came out. Contributions are welcome —
issues especially, since a second pair of eyes on a claim is worth more here
than a feature.

Everything below is a convention this repository already follows and that you
cannot guess from the outside. Nothing here is style for its own sake.

## Language: code English, curriculum Italian

**Python code and comments are always in English** — variable names, function
names, docstrings, inline comments, log messages, CLI help. All of it.

Italian appears in exactly two places:

- `training_files/` — the curriculum itself is Italian material. That is the
  data, not the code.
- the Italian documentation: `README.it.md`, `docs/it/`, and the project notes.

Every English document has an Italian mirror and vice versa. If you change one,
change the other, or say in the pull request that you did not so it does not
silently drift.

## `_reference/` takes part in no phase

`training_files/<lang>/<N>/_reference/` holds source material a human consulted
while writing that level — book extracts, subtitle dumps. **No training phase
reads it.** It is not a corpus, it is a bibliography. Do not add loader code
that walks it, and do not treat a file appearing there as material the model
saw.

What the model actually trains on, per level:

| file | what it is |
|---|---|
| `qa_pairs.jsonl` | the source QA pairs — **source**, edit this one |
| `qa_corpus.txt` | generated from `qa_pairs.jsonl` — never hand-edit |
| `local_teacher.json` | the rule-based teacher's targets for the level |
| `lexicon.json` | the nouns the level introduces |
| `batches/` | what the autonomy loop added, with its own ledgers |

`qa_corpus.txt` being generated is not a formality: a period where the source
was gitignored and the generated file was committed let the two drift, and a
fresh clone trained on different data than the machine that published the
results. Both are versioned now, and both have to move together.

## `retracted.jsonl` is a fact about the material

`training_files/<lang>/retracted.jsonl` is the retraction ledger: when the
autonomy loop teaches the model a noun, every curriculum shape that treated
that noun as unknown has to go, and the ledger is the record the generators
consult before rebuilding a level. It is versioned deliberately. Do not edit it
by hand — it is written by `dynamic_model/retraction.py` — and do not delete a
line to "re-enable" an admission: that resurrects supervision that contradicts
what the model was later taught.

## Secrets

`.env` is gitignored and stays that way. It holds `ANTHROPIC_API_KEY` when
someone opts into the Claude tutor; the default teacher is local and free and
needs no key at all. Never commit a key, a host, or a password — not in
`.env`, not in a notebook, not in a session log you paste into an issue.

## Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -q
```

The suite runs on CPU in a few seconds and needs no trained weights. A handful
of tests assert facts about a tokenizer that a build *produced* and skip
themselves when `models/` is absent, naming the missing file in the skip
reason — that is expected in a fresh clone and in CI. If you make a test
unrunnable in some environment, mark it `skipif` with a reason. Never exclude
it from the CI invocation: a skip is visible in the report, a `-k 'not ...'`
in the workflow is not.

`tests/test_1/` contains numeric gradient checks against the pure-NumPy
implementation. If you touch a layer's `backward()`, they are the ones to
watch.

## Pull requests

- Branch off `main`.
- One concern per pull request.
- Say what you measured, not only what you changed. This repository publishes
  numbers; a change that moves one needs the new number in the description.
- If a change makes a claim in the README or in `docs/` inaccurate, fix the
  claim in the same pull request.

## Claims

Results here are stated with their scope: implementation, regime, and the
caveat that limits them. `dream` beating online EWC on this curriculum is not
"replay dominates EWC" — the comparison is not compute-matched, and it says so
wherever it appears. Please keep new claims equally circumscribed;
[the FAQ](docs/en/faq.md) collects the ones that come up most.

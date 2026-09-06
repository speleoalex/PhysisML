"""
Tests for the polarity gate in grade_by_coverage.

'no, il giardino è un luogo.' and 'sì, il giardino è un luogo.' are the right
and the wrong answer to 'il giardino è un animale?'. Every grader scored them
identically until this gate existed: 'sì' and 'no' are in STOP_WORDS — rightly,
for the levels where they are grammar — so coverage stripped the one token that
IS the answer, and LocalTeacher._evaluate only looks for the class head. On the
rules-only path, which is what every build before 2026-09 used, both earned
'+++' and both advanced the step, so L11's negative is-a step (126 targets) was
never graded.

Run with:  python3 -m pytest tests/test_grade_polarity.py -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dynamic_model.train_curriculum import grade_by_coverage as grade  # noqa: E402

NEG_PROMPT = "il giardino è un animale?"
NEG_GOLD = "no, il giardino è un luogo."
POS_PROMPT = "la sorella è una persona?"
POS_GOLD = "sì, la sorella è una persona."


def test_the_opposite_answer_is_not_partial_credit():
    """It states the contrary of the gold, so it is wrong, not incomplete.

    0.0 matters twice: no positive weight update, and coverage is what SIGNAL 4
    reads to decide whether a turn may be used for echo training.
    """
    sym, cov, cmt = grade("+++", NEG_GOLD, "sì, il giardino è un luogo.",
                          NEG_PROMPT, 11)
    assert (sym, cov) == ("=", 0.0)
    assert "polarity" in cmt


def test_the_right_polarity_keeps_its_grade():
    sym, cov, cmt = grade("+++", NEG_GOLD, NEG_GOLD, NEG_PROMPT, 11)
    assert sym == "+++" and cmt is None


def test_a_missing_polarity_is_incomplete_not_wrong():
    """The class is right and the yes/no question is unanswered. '+' is what
    _update_state already treats as encouragement that does not advance."""
    sym, _, _ = grade("+++", NEG_GOLD, "il giardino è un luogo.", NEG_PROMPT, 11)
    assert sym == "+"


def test_the_affirmative_step_is_gated_the_same_way():
    """A model that prefixes everything with 'sì' passes B and C alike unless
    both are checked."""
    assert grade("+++", POS_GOLD, "no, la sorella è una persona.",
                 POS_PROMPT, 11)[0] == "="
    assert grade("+++", POS_GOLD, POS_GOLD, POS_PROMPT, 11)[0] == "+++"


def test_the_accent_is_not_the_lesson_the_gate_teaches():
    """'si' answers the question; the missing accent is a spelling matter.

    The gate must not read it as the opposite answer. It still loses a grade to
    the coverage check below, which counts 'sì' as a token like any other —
    that is pre-existing behaviour and defensible (5/6 of the gold restated),
    and it is emphatically not the same as being marked wrong.
    """
    sym, _, cmt = grade("+++", POS_GOLD, "si, la sorella è una persona.",
                        POS_PROMPT, 11)
    assert "polarity" not in (cmt or "")
    assert sym == "++"      # docked for the accent, not failed for the answer
    # the opposite answer, by contrast, is '='
    assert grade("+++", POS_GOLD, "no, la sorella è una persona.",
                 POS_PROMPT, 11)[0] == "="


def test_the_comma_is_required_of_the_gold_but_not_of_the_answer():
    sym, _, _ = grade("+++", NEG_GOLD, "no il giardino è un luogo.",
                      NEG_PROMPT, 11)
    assert sym == "+++"


def test_a_gold_without_polarity_is_left_alone():
    """Most of the curriculum is not a yes/no question, and the gate must be
    invisible there."""
    sym, cov, cmt = grade("+++", "il cane è un animale.",
                          "il cane è un animale.", "cos è un cane?", 11)
    assert cmt is None and sym == "+++"


def test_a_negation_inside_the_sentence_is_not_a_polarity():
    """'non' opens nothing: reading it as 'no' would mark a correct affirmative
    as the opposite answer."""
    sym, _, cmt = grade("+++", POS_GOLD, "non lo so.", POS_PROMPT, 11)
    assert cmt is None or "polarity" not in (cmt or "")


def test_the_gate_never_raises_a_grade():
    """Same contract as the rest of grade_by_coverage: it lowers only."""
    assert grade("=", NEG_GOLD, NEG_GOLD, NEG_PROMPT, 11)[0] == "="
    assert grade("+", NEG_GOLD, "il giardino è un luogo.", NEG_PROMPT, 11)[0] == "+"

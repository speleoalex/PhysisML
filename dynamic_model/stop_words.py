"""
Function words, for everything that has to separate content from grammar.

It lives in its own module because four unrelated callers need it and
train_curriculum.py — where it used to be defined — imports torch and the
anthropic client at module scope: the tests had to slice the source and exec
it to get the set, and scripts/curiosity_rate.py could not get it at all, so
its content-word filter was simply absent and every prompt read as unknown.

The list itself is no longer here either. A hardcoded Italian set is invisible
to a build in another language: the curiosity signal would read 'the', 'and'
and 'is' as content words and count every English prompt as a discovery. Each
language declares its own in training_files/<lang>/language.json, and
STOP_WORDS stays as the Italian value for the callers that never learned about
languages.
"""
from dynamic_model import language as _language


def for_language(lang: str = _language.DEFAULT_LANG) -> set:
    """The function words of one language, from its manifest.

    An empty set when the manifest declares none — which degrades the
    curiosity signal for that language but never mislabels it with another
    language's grammar.
    """
    return _language.load(lang).stop_words


#: Italian function words. Kept as a module constant because three callers and
#: several tests import it by name; new code should ask for_language(lang).
STOP_WORDS = for_language("it")

"""
Italian function words, shared by everything that has to separate content from
grammar.

It lives in its own module because three unrelated callers need it and
train_curriculum.py — where it used to be defined — imports torch and the
anthropic client at module scope: the tests had to slice the source and exec
it to get the set, and scripts/curiosity_rate.py could not get it at all, so
its content-word filter was simply absent and every prompt read as unknown.
"""

STOP_WORDS = {"il", "la", "lo", "le", "gli", "i", "un", "una", "di", "del",
              "della", "dei", "delle", "degli", "a", "e", "è", "in", "per",
              "da", "su", "con", "tra", "che", "non", "si", "ha", "ho", "hai",
              "ai", "al", "alla", "agli", "alle", "ma", "anche", "mi", "ti",
              "ci", "vi", "ne", "già", "più", "no", "sì", "se", "sua", "suo",
              "mia", "mio", "tu", "io", "lui", "lei", "noi"}

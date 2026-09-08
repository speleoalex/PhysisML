"""
The published package (scripts/export_hf.py → <out>/physisml/) ships without
the repository. Everything in it must import and run with nothing but torch and
numpy around — in particular the affect system, whose modulator reads the ask
head from the language manifest inside the repository and from the exported
config.json outside it.

The first English release (2026-09-08) shipped a modulator that imported
dynamic_model at module level: the import failed, load_affect() swallowed the
ImportError and every downloaded copy ran without the affect system while the
card described it. These tests import the bundle from an empty directory, in a
subprocess whose only path entry is that directory, so that regression cannot
come back unnoticed.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import export_hf                                                       # noqa: E402
from dynamic_model import surface                                      # noqa: E402
from dynamic_model.exp_b import modulator                              # noqa: E402


def _bundle(tmp_path, lang):
    """Lay the package out as the export does: <out>/physisml/ + config.json."""
    pkg = tmp_path / export_hf.PKG_NAME
    shutil.copytree(export_hf.PKG_SRC, pkg,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    export_hf.bundle_affect_system(str(pkg))
    if lang is not None:
        cfg = {"language": lang, "ask_heads": surface.load(lang).ask_heads}
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


def _run_outside_repo(folder, code):
    """Run `code` with the bundle folder as the only import path."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    env["PYTHONNOUSERSITE"] = os.environ.get("PYTHONNOUSERSITE", "")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(folder), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()


def test_bundled_modulator_never_imports_the_repository(tmp_path):
    pkg = tmp_path / export_hf.PKG_NAME
    pkg.mkdir()
    for dst in export_hf.bundle_affect_system(str(pkg)):
        for line in open(dst, encoding="utf-8").read().splitlines():
            # The guarded block is the one allowed repository import: it is
            # indented under `try:` and therefore never starts a line.
            if line.startswith(("from dynamic_model", "import dynamic_model")):
                pytest.fail(f"unguarded repository import in "
                            f"{os.path.basename(dst)}: {line!r}")


@pytest.mark.parametrize("lang", ["it", "en"])
def test_bundle_loads_and_reads_the_ask_head_from_config(tmp_path, lang):
    folder = _bundle(tmp_path, lang)
    out = _run_outside_repo(folder, (
        "import physisml.modulator as m\n"
        "assert m._surface is None and m._language is None, 'repo leaked in'\n"
        "print(m.default_lang()); print(m.ask_form()); print(m.ASK_FORM)\n"
    ))
    expected = surface.load(lang).ask_heads[0]
    assert out == [lang, expected, expected]


def test_bundle_load_affect_builds_the_modulator(tmp_path):
    folder = _bundle(tmp_path, "en")
    shutil.copy(os.path.join(ROOT, "dynamic_model", "data", "tokenizer_en.json"),
                folder / "tokenizer.json")
    out = _run_outside_repo(folder, (
        "from physisml.tokenizer import BPETokenizer\n"
        "from physisml.generation import load_affect\n"
        "tok = BPETokenizer(); tok.load('tokenizer.json')\n"
        "mod, aff = load_affect(tok)\n"
        "print(type(mod).__name__, type(aff).__name__)\n"
    ))
    assert out == ["AffectModulator AffectState"]


def test_bundle_without_config_imports_but_cannot_name_a_language(tmp_path):
    folder = _bundle(tmp_path, None)
    out = _run_outside_repo(folder, (
        "import physisml.modulator as m\n"
        "try:\n"
        "    m.ask_form()\n"
        "except RuntimeError as e:\n"
        "    print('RuntimeError', 'config.json' in str(e))\n"
    ))
    assert out == ["RuntimeError True"]


def test_in_repo_ask_form_follows_the_manifest():
    assert modulator.ASK_FORM == surface.load("it").ask_heads[0]
    assert modulator.ask_form("en") == surface.load("en").ask_heads[0]
    assert modulator.ask_form("en") == "what is"
    with pytest.raises(AttributeError):
        modulator.NO_SUCH_NAME

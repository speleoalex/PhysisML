#!/usr/bin/env python3
"""Every target must be able to reach '+++' with its own gold answer,
otherwise the teacher loops on it forever: '+' is a soft positive that
resets the failure counter without advancing the success counter."""
import json, re, sys, os
bad=0
# Range must cover every level that has a config, L11/L12 included: a
# hardcoded upper bound silently skips the levels most in need of checking.
for L in range(0,13):
    p=f'training_files/it/{L}/local_teacher.json'
    if not os.path.exists(p): continue
    cfg=json.load(open(p))
    stop=cfg.get('evaluation',{}).get('has_terminator_pattern','[.!?]')
    for sname,s in cfg.get('steps',{}).items():
        req={k:s.get(f'require_{k}',False) for k in ('article','verb','adjective')}
        mw=s.get('max_response_words',6)
        for t in s['targets']:
            if not isinstance(t,dict):
                continue
            r=t.get('expected','')
            noun=t.get('noun',''); art=t.get('article',''); vb=t.get('verb',''); adj=t.get('adjective',''); obj=t.get('object','')
            comp=re.sub(r'\s+','',r)
            def has(w): return True if not w else (bool(re.search(r'\b'+re.escape(w)+r'\b',r)) or (len(w)>=4 and w in comp))
            probs=[]
            if not has(noun): probs.append(f"noun {noun!r} assente")
            if req['article'] and not (art and art in r.split()[:5]): probs.append(f"article {art!r} richiesto ma assente/vuoto")
            if req['verb'] and not has(vb): probs.append(f"verb {vb!r} richiesto ma assente")
            if req['adjective'] and not has(adj): probs.append(f"adjective {adj!r} richiesto ma assente")
            if obj and not has(obj): probs.append(f"object {obj!r} assente")
            if not re.search(stop,r): probs.append("manca il terminatore")
            if len(r.split())>mw: probs.append(f"{len(r.split())} parole > max_response_words={mw}")
            if probs:
                bad+=1
                print(f"  L{L} step {sname}  {r!r}: " + "; ".join(probs))
print(("TUTTI I TARGET OK" if not bad else f"{bad} TARGET NON POSSONO OTTENERE +++"))

# ── One prompt, one gold ─────────────────────────────────────────────────────
# A prompt with two gold answers is contradictory supervision: whichever the
# model produces, the grader marks it wrong part of the time, and the level's
# exact match reads as a regression that never happened. It crept in twice --
# a generator mapping two items onto one prompt ('chi mangia la torta?' had
# four answers), and two levels asking the same question with different
# expected shapes ('cosa fa il cane?' at L3 with one verb, at L5 with two).
# Checked across pools AND the pairs harvested from sessions, because the
# harvest is the channel that reintroduced it.
import collections
gold = collections.defaultdict(lambda: collections.defaultdict(set))
for L in range(0, 13):
    d = f'training_files/it/{L}'
    p = f'{d}/local_teacher.json'
    if os.path.exists(p):
        cfg = json.load(open(p))
        for sname, s in cfg.get('steps', {}).items():
            tmpl = s.get('prompt_template', '{prompt}')
            for t in s['targets']:
                if not isinstance(t, dict):
                    continue
                pr = tmpl.format(prompt=t.get('prompt', ''),
                                 target=t.get('prompt', ''))
                gold[pr.strip()][t.get('expected', '').strip()].add(f'pool L{L}{sname}')
    q = f'{d}/qa_pairs.jsonl'
    if os.path.exists(q):
        for line in open(q, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            gold[o.get('prompt', '').strip()][o.get('response', '').strip()].add(f'qa L{L}')

clash = {p: v for p, v in gold.items() if len(v) > 1}
for p, resps in sorted(clash.items()):
    print(f'  {p!r} ha {len(resps)} risposte gold:')
    for r, srcs in sorted(resps.items()):
        print(f'      {r!r}  [{", ".join(sorted(srcs))}]')
print(('OGNI PROMPT HA UN SOLO GOLD' if not clash
       else f'{len(clash)} PROMPT CON GOLD CONTRASTANTI '
            f'(correggi con scripts/fix_gold_conflicts.py)'))

sys.exit(1 if (bad or clash) else 0)

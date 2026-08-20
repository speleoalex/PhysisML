#!/usr/bin/env python3
"""Every target must be able to reach '+++' with its own gold answer,
otherwise the teacher loops on it forever: '+' is a soft positive that
resets the failure counter without advancing the success counter."""
import json, re, sys, os
bad=0
for L in range(0,11):
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
sys.exit(1 if bad else 0)

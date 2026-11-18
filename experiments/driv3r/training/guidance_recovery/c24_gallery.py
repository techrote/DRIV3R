"""Source-matched SVG gallery from actually replayed C24 synthetic host episodes.

No text-to-image invention: every polyline is taken from a verified executed trace.
"""
from __future__ import annotations
import hashlib,html,json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from c24_history_trust import (CUES,MODES,TrustAdapter,Baseline36,rollout,
    observed_calibration,load_c12)

def load_models(prior,directory):
    models={'C18-A0':Baseline36(prior,True)}
    model=TrustAdapter(prior,True)
    model.head.load_state_dict(load_file(str(directory/'C24-TRUST.safetensors')))
    models['C24-TRUST']=model.eval()
    return models

def draw(row):
    g=row['geometry'];path=[g['start']]+[v['pos'] for v in row['trace']]
    cell=12;offset=16
    protected=set(tuple(v) for v in g['protected']);oldink=set(tuple(v) for v in g['initial_ink'])
    fill=[]
    for y in range(15):
     for x in range(15):
      p=(x,y)
      color='#465366' if p in protected else '#d6e3e4'
      if p in oldink:color='#9ec9a4'
      fill.append(f'<rect x="{offset+x*cell}" y="{offset+y*cell}" width="11" height="11" fill="{color}"/>')
    poly=' '.join(f'{offset+x*cell+5},{offset+y*cell+5}' for x,y in path)
    gx,gy=g['goal'];sx,sy=g['start']
    title=f"{row['model']} · {row['mode_generator_only']} · {row['outcome']} · {row['directive']} · {row['strength']}"
    svg=f'<svg viewBox="0 0 215 215" role="img" aria-label="Verified synthetic cursor path">'+''.join(fill)+f'<polyline points="{poly}" fill="none" stroke="#f29150" stroke-width="2.4"/>' +f'<circle cx="{offset+gx*cell+5}" cy="{offset+gy*cell+5}" r="5" fill="#1684d0"/>'+f'<circle cx="{offset+sx*cell+5}" cy="{offset+sy*cell+5}" r="4" fill="#d43748"/>'+'</svg>'
    return '<article><h3>'+html.escape(title)+'</h3>'+svg+f'<p>{"GOAL" if row["reached"] else "MISS"} · {row["steps"]} decisions · {row["backtracks"]} backtracks · persona {row["personality"]:+d}</p></article>'

def render(prereq,results,out):
    rows=json.loads((results/'heldout_trajectories.json').read_text())
    paths=[prereq/'prerequisites'/('guidance_recovery_'+n+'.safetensors') for n in
        ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
         'persistent_feedback','signed_feedback','selective_continue')]
    prior=load_c12(paths);models=load_models(prior,results)
    pairs={}
    for r in rows:
        if r['model'] not in models or r['directive']=='none':continue
        key=(r['source'],r['seed'],r['personality'],r['arm'],r['directive'],r['strength'],r['selection'])
        pairs.setdefault(key,{})[r['model']]=r
    divergent=[(key,p) for key,p in pairs.items() if len(p)==2 and p['C18-A0']['actions_sha256']!=p['C24-TRUST']['actions_sha256']]
    # Prefer outcome differences, then separate host regimes, then various worlds.
    divergent.sort(key=lambda v:(v[1]['C18-A0']['reached']==v[1]['C24-TRUST']['reached'],v[0][0],v[0][1]))
    selected=[];seen=set()
    for key,p in divergent:
        family=(p['C18-A0']['mode_generator_only'],p['C18-A0']['outcome'],p['C18-A0']['selection'])
        if family in seen:continue
        seen.add(family);selected.append((key,p))
        if len(selected)==10:break
    replayed=[]
    history_cache={}
    for key,p in selected:
        reference=p['C18-A0'];source=reference['source'];mode=reference['mode_generator_only']
        if source not in history_cache:
            history_cache[source]=observed_calibration(prior,source,mode,train=False)[0]
        mem=history_cache[source]
        for label,model in models.items():
            new=rollout(model,prior,source,mode,reference['outcome'],reference['personality'],
                        reference['arm'],reference['directive'],reference['strength'],reference['seed'],
                        train=False,horizon=32,trace=True,mem=mem)
            assert new['actions_sha256']==p[label]['actions_sha256']
            assert new['ink_sha256']==p[label]['ink_sha256']
            new.update({k:reference[k] for k in ('source','mode_generator_only','outcome','personality','arm','directive','strength')});new['model']=label
            replayed.append(new)
    content='<!doctype html><html lang="en"><meta charset="utf-8"><title>C24 executed source-history trajectories</title><style>body{font:14px system-ui;color:#e6edf4;background:#162032;margin:18px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:10px}article{padding:12px;background:#29384c;border-radius:10px}h3{font-size:12px;min-height:35px}p{font-size:12px}svg{width:100%}</style><h1>C24 history-conditioned guidance: matched executed paths</h1><p>Protected material: slate; available geometry: pale; initial ink: green; host-executed cursor trajectory: orange; goal: blue; start: red. Each trace was replayed and independently matched against recorded action and ink SHA-256 receipts. Synthetic CyberSand-like host only, not native gameplay.</p><main>'+''.join(draw(r) for r in replayed)+'</main></html>'
    out.write_text(content,encoding='utf-8')
    return {'verified_panels':len(replayed),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gallery_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'matched_pairs':len(selected)}

if __name__=='__main__':
    print(json.dumps(render(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3])),sort_keys=True))

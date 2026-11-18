"""C42 selected native-free trajectory replay and SVG gallery.

Renders only actual host-executed policy traces after checking hashes against
held-out cohort records. All 15x15 tiles show the synthetic obstacle mask.
"""
from __future__ import annotations
import copy,hashlib,html,json
from pathlib import Path
import torch
from safetensors.torch import load_file
from experiment import sha
from c42_wilson_transfer import FAMILIES,COHORTS,world,ActionHead,play,PREREQS,OptionHead,a0,load_c12

MODELS=('C34-RESET','C40-PAIR','C42-PAIR','C42-SHUFFLED')

def render(root,output):
    root=Path(root);output=Path(output)
    torch.set_num_threads(1)
    heads={}
    for name in MODELS[1:]:
        src=root/('out_c40/C40-PAIR.safetensors' if name=='C40-PAIR' else 'output/'+name+'.safetensors')
        head=ActionHead();head.load_state_dict(load_file(str(src)));head.eval();heads[name]=head
    base=a0(load_c12([root/'root/prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]))
    option=OptionHead();option.load_state_dict(load_file(str(root/'root/results/C28-TEMPORAL.safetensors')));option.eval()
    selected=[];cards=[];checked=0
    for cohort,seed in COHORTS:
        previous=json.loads((root/'output'/f'C42-cohort-{cohort}.json').read_text())
        bykey={(r['family'],r['model'],r['case'],r['switch'],r['personality'],r['directive'],r['strength']):r for r in previous}
        for family in FAMILIES:
            case='detour';person=1;switch=True;directive='backtrack';strength=1.
            s=seed+person*371+switch*2039+FAMILIES.index(family)*47011
            initial,info=world(s,family,case,person)
            for model in MODELS:
                if model=='C34-RESET':mode=model;h=[]
                else:mode='C39-ALIGNED';h=[heads[model]]*3
                result=play(base,option,h,copy.deepcopy(initial),info,s,person,mode,directive,strength,switch,78,True)
                prev=bykey[(family,model,case,switch,person,directive,strength)]
                assert (result['actions_sha'],result['ink_sha'],result['goals'])==(prev['actions_sha'],prev['ink_sha'],prev['goals'])
                assert prev['protected_sha']==sha(sorted(initial.protected))
                checked+=1
                record={'cohort':cohort,'family':family,'model':model,'seed':s,
                    'graph_sha':info['graph_sha'],'protected_sha':prev['protected_sha'],
                    'actions_sha':result['actions_sha'],'ink_sha':result['ink_sha'],
                    'goals':result['goals'],'initial':list(initial.pos),'goal':list(initial.goal),
                    'protected':sorted(map(list,initial.protected)),'trajectory':result['trajectory']}
                selected.append(record)
                tile=13;w=15*tile
                cells=''.join(f'<rect x="{x*tile}" y="{y*tile}" width="{tile}" height="{tile}" fill="#303c4b"/>' for x,y in sorted(initial.protected))
                points=[initial.pos]+[step['position'] for step in result['trajectory']]
                path=' '.join(f'{x*tile+tile/2},{y*tile+tile/2}' for x,y in points)
                goals=[step['goal'] for step in result['trajectory']]
                goalsvg=''.join(f'<rect x="{g[0]*tile+2}" y="{g[1]*tile+2}" width="{tile-4}" height="{tile-4}" fill="#eab84c" />' for g in set(map(tuple,goals)))
                svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{w}" viewBox="0 0 {w} {w}" role="img" aria-label="replayed legal worm path">'+f'<rect width="{w}" height="{w}" fill="#e8eff2"/>'+cells+goalsvg+f'<polyline points="{path}" fill="none" stroke="#167a9d" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'+f'<circle cx="{points[-1][0]*tile+tile/2}" cy="{points[-1][1]*tile+tile/2}" r="3.6" fill="#d83a45"/></svg>'
                cards.append(f'<section><h3>{cohort} · {family} · {model}</h3>{svg}<p>Goals {result["goals"]} · {result["steps"]} legal actions · action hash {result["actions_sha"][:12]}</p></section>')
    assert checked==16
    (output/'C42-selected-traces.json').write_text(json.dumps(selected,sort_keys=True)+'\n')
    css='body{background:#101923;color:#e7eff5;font:14px system-ui;padding:20px}h1{font-size:1.4rem}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:14px}section{background:#1e2c39;padding:12px;border:1px solid #415568;border-radius:6px}svg{max-width:100%;height:auto;image-rendering:pixelated}h3{margin:.1rem 0 .8rem}p{color:#b8ccdb;font-size:12px}small{color:#b8ccdb}'
    htmltext=f'<!doctype html><html lang="en"><meta charset="utf-8"><title>C42 executed-host synthetic recovery gallery</title><style>{css}</style><h1>C42 — Wilson/Kruskal recovery transfer</h1><small>16 independently replayed selected synthetic-host trajectories. Blue line = actual path; dark cells = protected; gold = goal locations; red = final position. Not a CyberSand native capture.</small><main>'+''.join(cards)+'</main></html>'
    (output/'C42-gallery.html').write_text(htmltext)
    return checked

if __name__=='__main__':
    import sys
    print('matched_replays',render(sys.argv[1],sys.argv[2]))

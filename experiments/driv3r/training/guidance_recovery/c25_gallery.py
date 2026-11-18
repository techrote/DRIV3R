"""C25 matched SVG gallery from freshly executed synthetic host trajectories.
No image generation or predicted paths; renders exactly host-applied decisions.
"""
from pathlib import Path
import json,sys,html,torch
from safetensors.torch import load_file
from personality_retention_core import load_c12
from c25_online_recovery import rollout,Baseline,OnlineResidual

PREREQ=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')

def load_models(archive,ckpt):
    paths=[Path(archive)/'prerequisites'/('guidance_recovery_'+x+'.safetensors') for x in PREREQ]
    prior=load_c12(paths)
    models={'C12':Baseline(prior,False),'C18-A0':Baseline(prior,True)}
    for name in ('C25-ONLINE','C25-NOMEM'):
        mdl=OnlineResidual(prior,history=(name=='C25-ONLINE')).eval()
        mdl.head.load_state_dict(load_file(str(Path(ckpt)/(name+'.safetensors'))))
        models[name]=mdl
    return models

def render(rec):
    geom=rec['geometry'];scale=17;N=15
    protected=''.join(f'<rect x="{x*scale}" y="{y*scale}" width="{scale}" height="{scale}" fill="#253344" />' for x,y in geom['protected'])
    start=geom['start'];goal=geom['goal'];points=[start]+[a['pos'] for a in rec['trajectory']]
    line=' '.join(f'{(x+.5)*scale},{(y+.5)*scale}' for x,y in points)
    return f'''<svg viewBox="0 0 {N*scale} {N*scale}" width="255" height="255" role="img" aria-label="Actual host-executed path">
    <rect width="{N*scale}" height="{N*scale}" fill="#e8ecee"/>{protected}
    <polyline points="{line}" fill="none" stroke="#df8f21" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round"/>
    <circle cx="{(start[0]+.5)*scale}" cy="{(start[1]+.5)*scale}" r="4" fill="#1688aa"/>
    <circle cx="{(goal[0]+.5)*scale}" cy="{(goal[1]+.5)*scale}" r="5" fill="none" stroke="#00a277" stroke-width="3"/>
    <circle cx="{(points[-1][0]+.5)*scale}" cy="{(points[-1][1]+.5)*scale}" r="4" fill="#d84e45"/></svg>'''

def make_gallery(archive,checkpoints,out):
    torch.set_num_threads(1)
    models=load_models(archive,checkpoints)
    # Paired states, independent cohorts, four distinct failure mechanisms.
    cases=[('reverse','backtrack',.35,0,-1,.35,250999),
           ('reverse','continue',1.,2,1,1.,251999),
           ('shortcut','backtrack',1.,1,-1,.35,250999),
           ('bridge','backtrack',1.,3,1,1.,251999),
           ('ahead','backtrack',1.,1,-1,.35,251999)]
    cards=[];receipts=[]
    for outcome,cue,strength,i,person,arm,evalseed in cases:
        s=evalseed+i*10117+('reverse','ahead','shortcut','bridge').index(outcome)*101111+person*131
        for name,model in models.items():
            rec=rollout(model,i,outcome,person,arm,cue,strength,s,48,True)
            txt=f'{name} · {outcome} · {cue} {strength:g} · {"GOAL" if rec["success"] else "no goal"} · {rec["steps"]} steps'
            cards.append(f'<section><h3>{html.escape(txt)}</h3>{render(rec)}<p>{html.escape(rec["actions_sha256"][:20])}…</p></section>')
            receipts.append({'model':name,'outcome':outcome,'cue':cue,'strength':strength,'index':i,
               'personality':person,'arm':arm,'seed':s,'actions_sha256':rec['actions_sha256'],
               'ink_sha256':rec['ink_sha256'],'source_sha256':rec['source_sha256'],
               'host_rejected':rec['host_rejected'],'goal_reached':rec['success']})
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><title>C25 executed trajectory gallery</title>
    <style>body{font:14px system-ui;background:#f5f6f8;color:#111827;margin:25px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(255px,1fr));gap:12px}
    section{background:white;padding:11px;border:1px solid #d8dce1;border-radius:6px}
    section h3{font-size:12px;height:34px;margin:0 0 8px}section p{font:10px monospace;color:#666}
    svg{border:1px solid #b6bdc4}p{max-width:850px}</style>
    <h1>C25 — matched host-executed trajectories</h1><p>Dark cells: host-protected material. Amber: actual cursor steps.
    Blue: starting position. Green ring: goal. Red: final position. No rendered path was inferred from model logits alone.
    Synthetic research geometry is not native CyberSand gameplay acceptance.</p><main>'''+''.join(cards)+'</main></html>'
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'gallery.html').write_text(doc)
    (out/'gallery_receipts.json').write_text(json.dumps(receipts,indent=2,sort_keys=True)+'\n')
    assert all(x['host_rejected']==0 for x in receipts)
    return len(receipts)

if __name__=='__main__':
    print(make_gallery(sys.argv[1],sys.argv[2],sys.argv[3]))

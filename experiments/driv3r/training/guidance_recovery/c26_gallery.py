"""C26 paired replay gallery: every rendered segment was host-executed.

Only visualizes verified traces from independent synthetic test episodes.
"""
from pathlib import Path
import json,html,sys,torch
from safetensors.torch import load_file
from c26_stuck_interventions import InterventionHead,Control,rollout,OUTCOMES
from c19_return_anchored import a0
from c24_history_trust import seed_world
from personality_retention_core import load_c12

PRE=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
     'persistent_feedback','signed_feedback','selective_continue')


def models(archive,checkpoints):
    root=Path(archive)/'prerequisites'
    prior=load_c12([root/f'guidance_recovery_{k}.safetensors' for k in PRE])
    m={'C12':Control(prior,False),'C18-A0':Control(prior,True)}
    for tag in ('C26-TRIGGER','C26-FULL'):
        c=InterventionHead(prior,tag=='C26-TRIGGER').eval()
        c.head.load_state_dict(load_file(str(Path(checkpoints)/f'{tag}.safetensors')))
        m[tag]=c
    return prior,m


def render(rec):
    size=17;n=15;g=rec['geometry']
    blocked=''.join(f'<rect x="{x*size}" y="{y*size}" width="{size}" height="{size}" fill="#374151" />' for x,y in g['protected'])
    start=tuple(g['start'])
    prefix=[start]+[tuple(e[1]) for e in rec['prefix']]
    after=[prefix[-1]]+[tuple(e['pos']) for e in rec['path']]
    def poly(points,color,stroke):
        pts=' '.join(f'{(p[0]+.5)*size},{(p[1]+.5)*size}' for p in points)
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linejoin="round" stroke-linecap="round"/>'
    goal=g['goal']
    return f'<svg viewBox="0 0 {n*size} {n*size}" width="240" height="240">' + (
      f'<rect width="{n*size}" height="{n*size}" fill="#f4f7fa"/>{blocked}'
      +poly(prefix,'#6b7280',2.5)+poly(after,'#ed9121',3.3)
      +f'<circle cx="{(prefix[-1][0]+.5)*size}" cy="{(prefix[-1][1]+.5)*size}" r="5" fill="#1d4ed8"/>'
      +f'<circle cx="{(goal[0]+.5)*size}" cy="{(goal[1]+.5)*size}" r="5.5" fill="none" stroke="#059669" stroke-width="2.8"/>'
      +f'<circle cx="{(after[-1][0]+.5)*size}" cy="{(after[-1][1]+.5)*size}" r="3.5" fill="#dc2626"/></svg>')


def create(archive,checkpoints,evalfile,out,max_cases=12):
    torch.set_num_threads(1)
    prior,policies=models(archive,checkpoints)
    data=json.loads(Path(evalfile).read_text())
    grouped={}
    for row in data:
        if row['directive']=='none':continue
        key=(row['index'],row['outcome'],row['personality'],row['arm'],row['mode'],row['directive'],row['strength'],row['seed'])
        grouped.setdefault(key,{})[row['model']]=row
    # Prefer informative, verifiable paired successes/regressions. Include
    # diverse outcome/cue/mode families rather than a handpicked single win.
    candidates=[]
    for key,record in grouped.items():
        if len(record)<4:continue
        base=record['C18-A0'];variant=record['C26-TRIGGER']
        changed=int(base['success']!=variant['success'])
        different=int(base['actions_sha256']!=variant['actions_sha256'])
        if key[4]=='badturn' and key[5]=='backtrack':order=0
        elif key[4]=='badturn':order=1
        else:order=2
        candidates.append(((-changed,-different,order,key[0]),key,record))
    candidates.sort(key=lambda q:q[0]);used=set();selected=[]
    for _,key,record in candidates:
        group=(key[1],key[4],key[5])
        if group in used:continue
        used.add(group);selected.append((key,record))
        if len(selected)>=max_cases:break
    # If exactly the same group cannot fill the requested gallery, repeat
    # different seeds but keep all hashes verified.
    for _,key,record in candidates:
        if len(selected)>=max_cases:break
        if any(x[0]==key for x in selected):continue
        selected.append((key,record))
    cards=[];receipts=[]
    for key,reference in selected:
        idx,outcome,person,arm,mode,cue,strength,seed=key
        for label,policy in policies.items():
            rec=rollout(policy,a0(prior),idx,outcome,person,arm,cue,strength,mode,seed,horizon=36,trace=True)
            rec['geometry']['start']=seed_world(idx,outcome,person,seed,False).pos
            source=reference[label]
            assert (rec['actions_sha256'],rec['ink_sha256'],rec['prefix_sha256'])==(source['actions_sha256'],source['ink_sha256'],source['prefix_sha256'])
            assert rec['success']==source['success'] and not rec['host_rejected']
            title=f'{label} · {outcome}/{mode} · {cue} {strength} · {"GOAL" if rec["success"] else "no goal"}'
            cards.append('<section><h3>'+html.escape(title)+'</h3>'+render(rec)+
                         f'<p>{len(rec["prefix"])} induced/policy prefix moves, {rec["steps_after_cue"]} after cue</p></section>')
            receipts.append({'model':label,'key':key,'success':rec['success'],
               'host_rejected':rec['host_rejected'],'actions_sha256':rec['actions_sha256'],
               'ink_sha256':rec['ink_sha256'],'prefix_sha256':rec['prefix_sha256'],
               'replayed_matches_evaluation':True})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    document='''<!DOCTYPE html><html lang="en"><meta charset="utf-8"><title>C26 executed trajectory gallery</title>
    <style>body{font:14px system-ui;background:#f0f3f5;color:#111827;padding:22px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
    section{background:white;border:1px solid #d2d8df;border-radius:8px;padding:10px}
    h3{font-size:12px;min-height:35px}section p{font:11px monospace;color:#6b7280}</style><h1>C26: executed intervention trajectories</h1>
    <p>Matched source and induced prefix within every group of four. Grey = host-executed pre-cue mistakes; amber = actual post-cue policy movements.
    Blue = host intervention position, green = goal, red = final position, dark = protected material. This is synthetic research, not native CyberSand.</p><main>'''+''.join(cards)+'</main></html>'
    (out/'gallery.html').write_text(document)
    (out/'gallery_receipts.json').write_text(json.dumps(receipts,sort_keys=True,indent=2)+'\n')
    return len(cards)

if __name__=='__main__':
    print(create(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))

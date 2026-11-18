"""Reconstruct and hash-match chosen C54 trajectories, then render true paths."""
from __future__ import annotations
import argparse, hashlib, html, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
import c54_delayed_branch_option as c54

def gallery(root, out):
    root=Path(root);out=Path(out)
    rows=json.loads((out/'C54-online.json').read_text())
    norm=json.loads((out/'C54-norm.json').read_text())
    mu=np.asarray(norm['mean'],np.float32);sc=np.asarray(norm['scale'],np.float32)
    heads={}
    for m in ('C54-ALIGNED','C54-SHUFFLED'):
        h=c54.Head();h.load_state_dict(load_file(str(out/(m+'.safetensors'))));h.eval();heads[m]=h
    base,option,_=c54.c52.c50.c47.c44.prerequisites(root/'c47/previous/predecessor_C42')
    selection=[('tree',1265722,9),('braid',1428251,9),('wilson',949235,9)]
    modelnames=('C34-RESET','C54-ROUTE','C54-ALIGNED','C54-SHUFFLED')
    traces=[];receipts=[]
    for fam,seed,tick in selection:
        rec=next(r for r in rows if r['family']==fam and r['seed']==seed and r['tick']==tick and r['arm']=='backtrack' and r['strength']==1.)
        maker=c54.c52.c50.c47.c42.world if fam in ('wilson','kruskal') else c54.c52.c50.c47.c41.world
        w,info=maker(seed,fam,rec['case'],rec['personality'])
        assert info['graph_sha']==rec['graph']
        snap,rng,log,meta=c54.prefix(base,w,seed,tick)
        assert meta['prefix_sha']==rec['prefix_sha']
        for name in modelnames:
            z=c54.suffix(base,option,snap,rng,log,tick,rec['target'],('backtrack',1.),name,heads.get(name),mu,sc,True)
            original=rec['outcomes'][name]
            assert (z['action_sha'],z['ink_sha'],z['goal'])==(original['action_sha'],original['ink_sha'],original['goal'])
            traces.append({'family':fam,'seed':seed,'tick':tick,'case':rec['case'],
                           'model':name,'start':meta['positions'],'protected':sorted(snap.protected),
                           'target':rec['target'],'goal':z['goal'],'outcome':z})
            receipts.append({'family':fam,'seed':seed,'model':name,'action_sha':z['action_sha'],
                             'ink_sha':z['ink_sha'],'original_action_sha':original['action_sha'],
                             'replay_equal':True})
    assert len(traces)==12
    sz=12;board=15*sz;pw=255;ph=238
    W=pw*4+24;H=ph*3+88
    b=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<rect width="100%" height="100%" fill="#0b1119"/>',
       '<text x="14" y="26" fill="#f3f7fa" font-size="19" font-family="sans-serif">C54 · Host-observed delayed branch option</text>',
       '<text x="14" y="46" fill="#a4b3c5" font-size="12" font-family="sans-serif">12 source-matched, separately reconstructed synthetic-host executions. Cyan: executed prefix · Amber: guided suffix · Red: goal</text>']
    for j,z in enumerate(traces):
        ri,ci=divmod(j,4);x=12+pw*ci;y=62+ph*ri;goal=z['goal'];o=z['outcome']
        b.append(f'<g transform="translate({x},{y})">')
        b.append('<rect width="245" height="229" rx="6" fill="#1c2834" stroke="#455769"/>')
        b.append(f'<text x="9" y="19" font-size="11.5" font-family="sans-serif" fill="#f1f6fa">{html.escape(z["model"])} · {"GOAL" if goal else "NO GOAL"}</text>')
        b.append(f'<text x="9" y="33" font-size="10" font-family="sans-serif" fill="#a5b6c8">{z["family"]} · {z["case"]} · branch steps {o["option_turns"]}</text>')
        ox=9;oy=40;blocked={tuple(p) for p in z['protected']}
        for yy in range(15):
            for xx in range(15):
                fill='#050a0f' if (xx,yy) in blocked else '#657583'
                b.append(f'<rect x="{ox+xx*sz}" y="{oy+yy*sz}" width="{sz-1}" height="{sz-1}" fill="{fill}"/>')
        def trail(points,col):
            if len(points)>1:
                path=' '.join(f'{ox+x*sz+5.5},{oy+y*sz+5.5}' for x,y in points)
                b.append(f'<polyline points="{path}" fill="none" stroke="{col}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>')
        a=[tuple(p) for p in z['start']]
        q=[tuple(v['pos']) for v in o['path']]
        trail(a,'#3bfcf0');trail(a[-1:]+q,'#ffcb4d')
        tx,ty=z['target'];b.append(f'<circle cx="{ox+tx*sz+5.5}" cy="{oy+ty*sz+5.5}" r="4" fill="#f0525a" stroke="white" stroke-width="0.9"/>')
        if goal and q:
            ex,ey=q[-1];b.append(f'<circle cx="{ox+ex*sz+5.5}" cy="{oy+ey*sz+5.5}" r="3" fill="#61ffb6"/>')
        b.append(f'<text x="193" y="69" fill="#bbccdb" font-size="9" font-family="monospace">{o["steps"]} ticks</text>')
        b.append(f'<text x="193" y="84" fill="#bbccdb" font-size="9" font-family="monospace">legal</text>')
        b.append('</g>')
    b.append('</svg>');gallery=out/'C54-gallery.svg';gallery.write_text('\n'.join(b)+'\n')
    (out/'C54-gallery-replay-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    return {'gallery_sha256':hashlib.sha256(gallery.read_bytes()).hexdigest(),
           'replays':len(receipts),'all_replays_matched':all(v['replay_equal'] for v in receipts)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args()
    print(gallery(a.root,a.out))

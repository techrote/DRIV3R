"""Replay and render C60 selected synthetic-host trajectories from their source seeds.

Only actual executed prefix/suffix points are shown. Graph topology and unseen
walls are deliberately not rendered as guessed gameplay art.
"""
from __future__ import annotations
import html,json,sys,collections
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
import c60_remote_frontier_rl as c60


def select(rows):
    """Predefined illustrative categories, not a scientific selection gate."""
    by_group=collections.defaultdict(list)
    for r in rows:
        if r['arm']!=['backtrack',1.] or r['drop']!=0.:continue
        z=r['outcomes'];f=r['family']
        if f in ('wilson','kruskal') and z['C60-FRONTIER']['goal']>z['C34-RESET']['goal']:
            by_group['ood_frontier_win'].append(r)
        if f not in ('wilson','kruskal') and z['C60-ALIGNED']['goal']>z['C34-RESET']['goal']:
            by_group['familiar_learned_win'].append(r)
        if f in ('wilson','kruskal') and all(z[k]['goal']==0 for k in ('C34-RESET','C60-FRONTIER','C60-ALIGNED')):
            by_group['ood_shared_fail'].append(r)
        if f not in ('wilson','kruskal') and all(z[k]['goal']==1 for k in ('C34-RESET','C60-FRONTIER','C60-ALIGNED')):
            by_group['familiar_shared_success'].append(r)
    selected=[];seen=set()
    for category in ('ood_frontier_win','familiar_learned_win','ood_shared_fail','familiar_shared_success'):
        for r in by_group[category]:
            if r['graph'] not in seen:
                selected.append((category,r));seen.add(r['graph']);break
    return selected


def main(prior,out):
    prior=Path(prior);out=Path(out)
    rows=json.loads((out/'C60-online.json').read_text())
    sel=select(rows)
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    (_, _),(onlinew, _),_=c60.worlds()
    states,_=c60.c59.c58.c56.states_for_goal(base,onlinew,'remote')
    bygraph={x[6]['graph_sha']:x for x in states}
    s=json.loads((out/'C60-summary.json').read_text())
    weights={k:np.asarray(v,np.float32) for k,v in s['weights'].items()}
    panels=[];receipts=[]
    for category,r in sel:
        st=bygraph[r['graph']]
        for pol in ('C34-RESET','C60-FRONTIER','C60-ALIGNED','C60-SHUFFLED'):
            z,_=c60.execute(base,opt,st,('backtrack',1.),pol,weights.get(pol,np.zeros(c60.NFEAT,np.float32)),0.,trace=True)
            original=r['outcomes'][pol]
            assert z['action_sha']==original['action_sha'] and z['ink_sha']==original['ink_sha']
            assert z['goal']==original['goal'] and z['host_rejections']==0
            panels.append((category,r,pol,z,st[3]['positions']))
            receipts.append({'category':category,'graph':r['graph'],'policy':pol,'action_sha':z['action_sha'],'ink_sha':z['ink_sha'],'verified':True})
    cols=4;rows_count=(len(panels)+cols-1)//cols;width=1256;height=76+195*rows_count
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
      '<rect width="100%" height="100%" fill="#111b29"/>',
      '<text x="22" y="30" font-family="monospace" font-size="19" fill="#ecf2ff">C60 | Remote-goal recovery | executed trajectory replay</text>',
      '<text x="22" y="48" font-family="monospace" font-size="11" fill="#a4b4c9">Grey = executed host prefix. Mint = executed recovery suffix. Gold = published goal. No invented terrain or unseen geometry.</text>']
    for i,(cat,r,pol,z,prefix) in enumerate(panels):
        x=14+(i%cols)*312;y=62+(i//cols)*195
        p=[tuple(a) for a in prefix];q=[tuple(a['p']) for a in z['path']];goal=tuple(r['target'])
        allp=p+q+[goal];xs=[a[0] for a in allp];ys=[a[1] for a in allp]
        lx,ux=min(xs),max(xs);ly,uy=min(ys),max(ys)
        scale=min(13.,275./max(2,ux-lx),136./max(2,uy-ly))
        def XY(t):return x+155+(t[0]-(lx+ux)/2)*scale,y+102+(t[1]-(ly+uy)/2)*scale
        def PL(arr):return ' '.join(f'{XY(v)[0]:.1f},{XY(v)[1]:.1f}' for v in arr)
        svg.append(f'<rect x="{x}" y="{y}" width="304" height="184" rx="7" fill="#19283a" stroke="#455b76"/>')
        ttl=f'{cat} / {r["family"]} / {pol}'
        svg.append(f'<text x="{x+8}" y="{y+16}" font-size="9" font-family="monospace" fill="#eef4fa">{html.escape(ttl)}</text>')
        if len(p)>1:svg.append(f'<polyline points="{PL(p)}" fill="none" stroke="#8198b6" stroke-width="1.4"/>')
        if len(q)>1:svg.append(f'<polyline points="{PL(q)}" fill="none" stroke="#53e0b9" stroke-width="2.4"/>')
        gx,gy=XY(goal);svg.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="4.2" fill="#ffce68"/>')
        svg.append(f'<text x="{x+8}" y="{y+177}" font-size="10" font-family="monospace" fill="#c8d5e5">goal={z["goal"]}  steps={z["steps"]}  legal=PASS</text>')
    svg.append('</svg>')
    (out/'C60-gallery.svg').write_text('\n'.join(svg)+'\n')
    (out/'C60-gallery-replay.json').write_text(json.dumps({'panels':receipts,'categories':[q[0] for q in sel]},sort_keys=True,indent=2)+'\n')
    print('Gallery panels',len(panels),'categories',[q[0] for q in sel]);return len(panels)

if __name__=='__main__': main(sys.argv[1],sys.argv[2])

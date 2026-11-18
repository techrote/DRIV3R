"""C39 independent graph-cluster statistics, host trace replays and SVG gallery."""
from __future__ import annotations
import argparse, collections, hashlib, html, json
from pathlib import Path
import numpy as np
from experiment import sha
from c34_independent_maze import make_world

PALETTE={'C18-A0':'#87919b','C34-RESET':'#d7b56d','C39-ALIGNED':'#5dc49f','C39-SHUFFLED':'#bd93da','C39-RANDOM':'#d98e86'}

def bootstrap_online(cohorts,reps=5000,seed=39109):
    allrows=[r for rows in cohorts.values() for r in rows if r['directive']!='none']
    sums=collections.defaultdict(lambda:collections.Counter())
    for r in allrows:sums[(r['seed'],r['graph_sha'])][r['model']]+=r['goals']
    keys=sorted(sums);rng=np.random.default_rng(seed);results={}
    for control in ('C34-RESET','C39-SHUFFLED','C39-RANDOM'):
        d=np.asarray([sums[key]['C39-ALIGNED']-sums[key][control] for key in keys],float)
        draws=np.empty(reps)
        for k in range(reps):draws[k]=d[rng.integers(len(d),size=len(d))].sum()
        results[control]={'paired_goal_delta':int(d.sum()),'graph_clusters':len(d),
                          'descriptive_95pct_interval':[float(x) for x in np.quantile(draws,[.025,.975])],
                          'positive_clusters':int((d>0).sum()),'negative_clusters':int((d<0).sum())}
    return results


def bootstrap_offline(held,reps=5000,seed=39110):
    clusters=collections.defaultdict(lambda:[0.,0.,0.,0])
    for r in held:
        key=r['graph'];vals=r['returns']
        cluster=clusters[key]
        for k in range(3):cluster[k]+=max(vals)-vals[int(np.argmax(r['values'][k]))]
        cluster[3]+=1
    rng=np.random.default_rng(seed);arr=np.asarray([clusters[x] for x in sorted(clusters)])
    results={}
    for ctrl,k in (('C39-SHUFFLED',1),('C39-RANDOM',2)):
        d=(arr[:,k]-arr[:,0]);draws=np.empty(reps)
        for i in range(reps):
            picks=rng.integers(len(arr),size=len(arr));draws[i]=d[picks].sum()/arr[picks,3].sum()
        results[ctrl]={'mean_regret_reduction':round(float(d.sum()/arr[:,3].sum()),6),
                       'graph_clusters':len(arr),'descriptive_95pct_interval':[
                         round(float(x),6) for x in np.quantile(draws,[.025,.975])]}
    return results


def replay(row):
    if row['trajectory'] is None:return None
    world,info=make_world(row['seed'],row['family'],row['case'],row['personality'])
    assert info['graph_sha']==row['graph_sha'] and info['prefix_sha']==row['prefix_sha']
    actions=[];changed=False
    for step in row['trajectory']:
        if step['t']>=50 and row['switch'] and not changed:
            world.goal=info['second_goal'];world.goal_change=True;world.goals_seen+=1;world.stuck=2;changed=True
        menu=world.candidates(1.)
        selected=[c for c in menu if c.action[0]==step['action'] and c.target==tuple(step['position'])]
        assert len(selected)==1 and selected[0].legal
        c=selected[0];assert world.apply(c)
        assert world.pos==tuple(step['position']) and world.goal==tuple(step['goal'])
        actions.append((c.action,c.target))
    assert sha(actions)==row['actions_sha']
    assert sha(sorted(world.inked))==row['ink_sha']
    assert not world.invalid_attempts
    return len(actions)


def audit(root,out):
    root=Path(root);out=Path(out)
    cohorts={p.stem[-1]:json.loads(p.read_text()) for p in sorted(root.glob('C39-cohort-*.json'))}
    weak={p.stem[-1]:json.loads(p.read_text()) for p in sorted(root.glob('C39-weak-?.json'))}
    held=json.loads((root/'C39-heldout.json').read_text())
    training=json.loads((root/'C39-summary.json').read_text())
    assert set(cohorts)==set('ABCDEF') and set(weak)==set('AB')
    allrows=[r for rr in cohorts.values() for r in rr]
    graphset={r['graph_sha'] for r in allrows}
    heldgraphs={r['graph'] for r in held}
    traingraphs={r['graph'] for r in json.loads((root/'C39-train-receipts.json').read_text())}
    assert not traingraphs.intersection(heldgraphs)
    assert not traingraphs.intersection(graphset)
    assert not heldgraphs.intersection(graphset)
    weakrows=[r for rr in weak.values() for r in rr]
    assert {r['graph_sha'] for r in weakrows}.issubset(graphset)
    replayed=[replay(r) for r in allrows+weakrows if r['trajectory'] is not None]
    assert replayed and all(x is not None for x in replayed)
    paired=bootstrap_online(cohorts)
    reg=bootstrap_offline(held)
    report={'schema':'driv3r-c39-audit/1','strong_cohorts':len(cohorts),'weak_cohorts':len(weak),
        'episodes':len(allrows)+len(weakrows),'strong_guided_episodes':sum(r['directive']!='none' for r in allrows),
        'weak_guided_episodes':len(weakrows),
        'evaluation_graphs':len(graphset),'training_graphs':len(traingraphs),
        'label_holdout_graphs':len(heldgraphs),
        'independent_execution_replays':len(replayed),
        'replayed_steps':sum(replayed),'paired_goal_bootstrap':paired,
        'paired_label_bootstrap':reg,
        'weak_guidance_bootstrap':bootstrap_online(weak),
        'protected_modifications':0,
        'host_rejections':sum(r['host_rejections'] for r in allrows+weakrows),
        'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in allrows+weakrows)}
    (out/'C39-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    make_gallery(cohorts,out/'C39-gallery.html')
    return report


def make_gallery(cohorts,path):
    # Every displayed polyline comes from an actual host-executed trace.
    items=[]
    for cohort in sorted(cohorts):
        subset=[r for r in cohorts[cohort] if r['trajectory'] is not None and
             r['model'] in ('C34-RESET','C39-ALIGNED','C39-SHUFFLED') and r['switch']]
        for family,case in (('tree','detour'),('braid','forward')):
            rr=[r for r in subset if r['family']==family and r['case']==case]
            if rr:items.append((cohort,family,case,rr))
    panels=[]
    for cohort,family,case,rr in items:
        world,info=make_world(rr[0]['seed'],family,case,rr[0]['personality'])
        grid=[]
        for y in range(15):
            for x in range(15):
                if (x,y) in world.protected:grid.append(f'<rect x="{x*14}" y="{y*14}" width="14" height="14" fill="#314155"/>')
        for row in rr:
            visited=[world.pos]+[tuple(r['position']) for r in row['trajectory']]
            points=' '.join(f'{x*14+7},{y*14+7}' for x,y in visited)
            grid.append(f'<polyline points="{points}" stroke="{PALETTE[row["model"]]}" stroke-width="2.6" fill="none" stroke-linejoin="round" opacity=".82"/>')
        goal=rr[0]['trajectory'][-1]['goal'] if rr[0]['trajectory'] else world.goal
        grid.append(f'<circle cx="{goal[0]*14+7}" cy="{goal[1]*14+7}" r="5" fill="#fff"/>')
        legend=' · '.join(f'<span style="color:{PALETTE[r["model"]]}">{r["model"]} {r["goals"]} goals</span>' for r in rr)
        panels.append(f'<article><h3>{cohort} / {family} / {case}</h3><svg viewBox="0 0 210 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:#16212b">{''.join(grid)}</svg><p>{legend}</p><small>Host-executed paths, no oracle paths.</small></article>')
    text='<!doctype html><meta charset="utf-8"><title>C39 executed trajectories</title><style>body{font:14px system-ui;background:#101823;color:#e1e9f0;max-width:1300px;margin:auto;padding:18px}.g{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}article{border:1px solid #39485a;padding:12px;border-radius:8px}svg{image-rendering:pixelated}p{font-size:11px}h3{margin:4px 0 8px}@media(max-width:850px){.g{grid-template-columns:repeat(2,1fr)}}</style><h1>C39 / Action-conditional return</h1><p>12 independently replayed matched synthetic-host maps. Each path includes only executed legal actions. Blue cells are protected. White dot is the final shown goal. Not native CyberSand footage.</p><div class="g">'+''.join(panels)+'</div>'
    path.write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args()
    print(json.dumps(audit(a.root,a.out),indent=2))

"""UNSEEN-MASK-STRESS-001: frozen policy and interventions on novel DEVELOPMENT masks.

No sealed HELDOUT access, no training and no native CyberSand execution.
Novel masks were declared before reading episode results. World/action logic is
reused without edits. This lab deliberately avoids cherry-picking feasible cases.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, resource, statistics, time
from collections import Counter
from pathlib import Path

from . import topology_adaptive_handoff_lab as prior
from .goal_conditioning_lab import GoalRanker
from .detour_recovery_lab import prefix_plan
from .train import sha_file
from .world import World, SOLID, MATERIAL_Z, Spec, specs

VERSION='unseen-mask-stress/1'
MASKS=('broken_diamond','offset_comb','slanted_fences','nested_bays')
ARMS=('neural','absolute24','planner_each_stage','second_switch','second_switch_stall3')
SEEDS=(711,712)
GOALS=prior.GOALS
CAP=64


def novel_world(sp:Spec,mask:str)->World:
    if sp.split!='development' or sp.challenge!='normal' or mask not in MASKS:
        raise ValueError('Only prespecified synthetic DEVELOPMENT masks')
    w=World.create(sp)
    for p in tuple(w.materials):
        if 41<=p[0]<=71 and 39<=p[1]<=69: del w.materials[p]
    w.protected.difference_update({p for p in w.protected if 41<=p[0]<=71 and 39<=p[1]<=69})
    reserved={w.cursor,*GOALS}
    # Explicitly synthetic, deterministic and independent of checkpoint outputs.
    points=[]
    def wall(items,kind=SOLID):
        for p in items:points.append((p,kind))
    if mask=='broken_diamond':
        for ring in (8,13):
            for dx in range(-ring,ring+1):
                x,y=55+dx,51+(ring-abs(dx))
                if 42<=x<=70 and abs(dx) not in (2,3):wall(((x,y),),MATERIAL_Z if ring==8 else SOLID)
                x,y=55+dx,51-(ring-abs(dx))
                if 42<=x<=70 and abs(dx) not in (5,6):wall(((x,y),),SOLID)
    elif mask=='offset_comb':
        for i,y in enumerate((42,47,53,59,65)):
            if i%2==0:wall(((x,y) for x in range(42,64)),MATERIAL_Z)
            else:wall(((x,y) for x in range(52,71)),SOLID)
        wall(((x,55) for x in range(65,70)),SOLID)
    elif mask=='slanted_fences':
        for d in range(23):
            if d not in (7,8,19):wall(((43+d,41+d//2),),SOLID)
            if d not in (2,13,14):wall(((69-d,65-d//2),),MATERIAL_Z)
        wall(((x,61) for x in range(45,51)),SOLID)
    elif mask=='nested_bays':
        for x0,y0,x1,y1,door in ((53,41,70,57,45),(43,55,62,68,61)):
            wall(((x,y0) for x in range(x0,x1+1) if x not in (door,door+1)),MATERIAL_Z)
            wall(((x,y1) for x in range(x0,x1+1) if x not in (x1-3,x1-2)),SOLID)
            wall(((x0,y) for y in range(y0,y1+1)),SOLID)
            wall(((x1,y) for y in range(y0,y1+1) if y not in (y0+6,y0+7)),MATERIAL_Z)
    for p,kind in points:
        if p in reserved:continue
        if not 41<=p[0]<=71 or not 39<=p[1]<=69:continue
        w.materials[p]=kind
    assert all(not w.occupied(p) for p in reserved)
    assert len(w.ink)==0 and w.logical_moves==0
    return w


@contextlib.contextmanager
def use_novel_masks():
    old=prior.overlay_world
    prior.overlay_world=novel_world
    try:yield
    finally:prior.overlay_world=old


def study(out:Path,checkpoints:Path,n:int=1):
    if not 1<=n<=2:raise ValueError('bounded n: 1..2')
    out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    t0=time.perf_counter()
    models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    modelhash={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];count=0;vecs=0;preflight=[]
    with use_novel_masks():
        for sp in specs('development',n):
            for mask in MASKS:
                w=novel_world(sp,mask)
                preflight.append({'spec':sp.ident,'family':sp.family,'mask':mask,
                                  'initial_state':w.state_sha(),'topology_sha':w.base_sha(),
                                  'initial_reachable':{str(g):prefix_plan(w,g,CAP) is not None for g in GOALS}})
                for order in (0,1):
                    for seed in SEEDS:
                        for arm in ARMS:
                            name=f'{seed}-{sp.family}-{sp.seed}-{mask}-{order}-{arm}.json'
                            path=out/'traces'/name
                            if path.exists():tr=json.loads(path.read_text())
                            else:
                                row,tr=prior.episode(sp,mask,order,seed,arm,models[seed])
                                path.write_text(json.dumps(tr,sort_keys=True,separators=(',',':'))+'\n')
                            c,v=prior.verify_trace(tr,models[seed]);count+=c;vecs+=v
                            rows.append(tr['row'])
    summary={}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        ac=Counter()
        for r in subset:ac.update(r['action_counts'])
        summary[arm]={'episodes':len(subset),'arrived':sum(r['arrived_final'] for r in subset),
                      'planner_calls':sum(r['planner_calls'] for r in subset),
                      'neural_actions':sum(r['neural_actions'] for r in subset),
                      'action_counts':dict(sorted(ac.items())),
                      'by_mask':{m:sum(r['arrived_final'] for r in subset if r['layout']==m) for m in MASKS},
                      'by_family':{f:sum(r['arrived_final'] for r in subset if r['world_family']==f)
                                   for f in sorted({r['world_family'] for r in subset})},
                      'median_episode_latency_ms':round(statistics.median(r['median_choose_ms'] for r in subset),6)}
    # Paired disagreements (identical scenario, checkpoint, order).
    pairs=[]
    for seed in SEEDS:
        for sp in specs('development',n):
            for mask in MASKS:
                for order in (0,1):
                    group={r['arm']:r for r in rows if r['model_seed']==seed and r['world_seed']==sp.seed
                           and r['world_family']==sp.family and r['layout']==mask and r['order']==order}
                    pairs.append({'seed':seed,'family':sp.family,'world_seed':sp.seed,'mask':mask,'order':order,
                                  'arrived':{a:group[a]['arrived_final'] for a in ARMS}})
    files=('unseen_mask_stress_lab.py','topology_adaptive_handoff_lab.py',
           'topology_goal_churn_lab.py','world.py','detour_recovery_lab.py',
           'goal_conditioning_lab.py','goal_observability_lab.py','train.py','policy.py')
    e={'version':VERSION,'split':'development','sealed_heldout_opened':False,
       'native_receipts':0,'training_updates':0,'external_64_logical_move_gate':True,
       'synthetic_new_masks':list(MASKS),'arms':list(ARMS),'switch_schedule':list(prior.SCHEDULE),
       'source_sha256':{p:sha_file(Path(__file__).parent/p) for p in files},
       'checkpoint_sha256':modelhash,'initial_feasibility':preflight,
       'evaluated_without_feasibility_exclusion':len(preflight),'episodes':len(rows),
       'verified_receipts':count,'verified_probability_vectors':vecs,
       'summary':summary,'pairs':pairs,'rows':rows,
       'wall_s':round(time.perf_counter()-t0,3),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'limitations':['Novel masks built inside the same DEVELOPMENT arena and share seed-family backgrounds; not structural HELDOUT.',
                      'No reactive material changes; every mask static throughout an episode.',
                      'Frozen tiny neural checkpoints with engineered goal-distance features.',
                      'Prefix planner restricted to backtrack-prefix then unpainted traversal.',
                      '64 logical moves enforced synthetically, not native CyberSand acceptance.',
                      'All scenarios retained regardless of initial reachability; per-scenario feasibility logged.',
                      'Goal switches are externally scheduled, not autonomous decisions.',
                      'External checkpoints retained locally; not immutable GitHub Actions artifacts.']}
    (out/'evidence.json').write_text(json.dumps(e,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},indent=2,sort_keys=True)+'\n')
    return e


def verify(out:Path,checkpoints:Path):
    e=json.loads((out/'evidence.json').read_text())
    hashes=json.loads((out/'SHA256SUMS.json').read_text())
    assert hashes=={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert e['checkpoint_sha256']=={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    assert e['source_sha256']=={p:sha_file(Path(__file__).parent/p) for p in e['source_sha256']}
    models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    counts=vecs=0;rows=[]
    with use_novel_masks():
        for p in sorted((out/'traces').glob('*.json')):
            tr=json.loads(p.read_text());c,v=prior.verify_trace(tr,models[tr['row']['model_seed']]);counts+=c;vecs+=v;rows.append(tr['row'])
    assert len(rows)==e['episodes'] and counts==e['verified_receipts'] and vecs==e['verified_probability_vectors']
    assert all(r['illegal_material_edits']==0 and r['logical_moves']<=CAP for r in rows)
    for a in ARMS:
        s=e['summary'][a];group=[r for r in rows if r['arm']==a]
        assert (s['episodes'],s['arrived'],s['planner_calls'])==(len(group),sum(r['arrived_final'] for r in group),sum(r['planner_calls'] for r in group))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('run','verify'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoints',type=Path,required=True)
    p.add_argument('--n',type=int,default=1);args=p.parse_args()
    e=study(args.out,args.checkpoints,args.n) if args.command=='run' else None
    assert verify(args.out,args.checkpoints)
    print(json.dumps({'verified':True,'episodes':e['episodes'] if e else 'from_archive','summary':e['summary'] if e else None}))
if __name__=='__main__':main()

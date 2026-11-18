"""MATERIAL-EPOCH-001: synthetic host-state changes versus sparse plan caches.

Only prespecified DEVELOPMENT fixtures and frozen tiny checkpoints; no game writes,
real reaction physics, native receipts, optimizer updates, or HELDOUT access.
Every externally scripted material mutation is recorded with pre/post state hashes.
"""
from __future__ import annotations
import argparse, hashlib, json, resource, statistics, time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
from . import topology_adaptive_handoff_lab as old
from .world import World, Spec, SOLID, MATERIAL_Z, digest, specs
from .goal_conditioning_lab import GoalRanker
from .detour_recovery_lab import prefix_plan
from .goal_observability_lab import strict_step
from .train import sha_file

VERSION='material-epoch/1'
CAP=64
ARMS=('neural','fixed24','fixed24_epoch','event_only','each_stage_epoch')
SEEDS=(711,712)
LAYOUTS=old.LAYOUTS
GOALS=old.GOALS
SCHEDULE=old.SCHEDULE
EVENT_TICKS=(20,36)
# Candidate cells fixed before evaluation, not chosen from rollout outcomes.
# For each topology: (remove at 20, add at 20, remove at 36, add at 36).
PATCHES={
 'opposing_gates':((53,50),(53,43),(60,50),(60,58)),
 'enclosed_courtyard':((57,50),(59,42),(70,51),(60,42)),
 'horizontal_switchback':((53,52),(66,52),(52,59),(44,59)),
 'protected_slalom':((53,49),(53,42),(63,52),(63,44)),
}
SOURCE=('material_epoch_lab.py','topology_adaptive_handoff_lab.py','world.py','detour_recovery_lab.py',
        'goal_conditioning_lab.py','goal_observability_lab.py','train.py','policy.py')


def apply_event(w:World,layout:str,tick:int)->dict:
    """External synthetic material-only transition; do not touch ink/protected.

    Closing a gate is suppressed if the cell became ink/cursor; the suppression
    is retained as evidence, not silently counted as successful closure.
    """
    assert tick in EVENT_TICKS
    pre=w.state_sha()
    pairs=PATCHES[layout]
    remove,close=(pairs[0],pairs[1]) if tick==20 else (pairs[2],pairs[3])
    log=[]
    for operation,p in (('open',remove),('close',close)):
        oldmat=w.materials.get(p)
        if operation=='open':
            if oldmat is not None:w.materials.pop(p)
            applied=oldmat is not None
            reason='opened' if applied else 'already_empty'
        else:
            applied=p not in w.protected and p not in w.ink and p!=w.cursor and oldmat is None
            if applied:w.materials[p]=MATERIAL_Z
            reason='closed' if applied else ('blocked_by_ink_or_cursor' if p in w.ink or p==w.cursor else 'already_occupied_or_protected')
        log.append({'op':operation,'cell':list(p),'before':oldmat,'after':w.materials.get(p),
                    'applied':applied,'reason':reason})
    assert not w.ink.intersection(w.materials.keys())
    return {'tick':tick,'pre_sha256':pre,'post_sha256':w.state_sha(),'operations':log,
            'material_epoch':1 if tick==20 else 2}


def choose(w,goal,model,arm,tick,phase,epoch,cache,attempted,planned_epoch,changed):
    """Cache is one plan at a time, never an unbounded sequence of searches."""
    if cache is not None and arm.endswith('_epoch') and planned_epoch!=epoch:
        cache=None;attempted=False
    if cache is not None:
        if cache:
            next_act=cache[0]
            if next_act in w.menu(12):
                return next_act,None,cache[1:],attempted,planned_epoch,0,'cached'
            if arm.endswith('_epoch'):
                cache=None;attempted=False
            else:
                return 'STOP',None,(),attempted,planned_epoch,0,'stale_stop'
        else:
            # Prior controller's one-shot plan is exhausted.
            return 'STOP',None,(),attempted,planned_epoch,0,'exhausted_stop'
    ask=(arm in ('fixed24','fixed24_epoch') and tick>=24
         or arm=='event_only' and changed
         or arm=='each_stage_epoch')
    if ask and not attempted:
        plan=prefix_plan(w,goal,CAP)
        if plan is None:
            return 'STOP',None,(),True,epoch,1,'planner_no_route'
        if len(plan)==0:
            return 'STOP',None,(),True,epoch,1,'planner_empty'
        return plan[0],None,plan[1:],True,epoch,1,'new_plan'
    act,probs=model.choose(w,goal,CAP,False)
    return act,probs,None,attempted,planned_epoch,0,'neural'


def episode(sp,layout,order,seed,arm,model):
    assert sp.split=='development' and arm in ARMS and layout in LAYOUTS
    w=old.overlay_world(sp,layout);initial=w.state_sha();base=w.base_sha()
    goals=(GOALS[order],GOALS[1-order],GOALS[order]);phase=0;epoch=0
    cache=None;attempted=False;planned_epoch=-1
    entries=[];events=[];switches=[];lat=[];hits=[False]*3;calls=0;reason='decision_gate'
    for tick in range(CAP):
        changed=False
        if tick in EVENT_TICKS:
            ev=apply_event(w,layout,tick);events.append(ev);epoch=ev['material_epoch'];changed=True
        if phase<2 and tick==SCHEDULE[phase]:
            switches.append({'tick':tick,'old_phase':phase,'pre_sha256':w.state_sha(),
                             'next_goal':list(goals[phase+1]),'discarded_cache':cache is not None})
            phase+=1;cache=None;attempted=False;planned_epoch=-1
        goal=goals[phase]
        if w.cursor==goal:
            hits[phase]=True
            if phase==2:reason='final_goal';break
            continue
        if w.stopped:reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        pre=w.state_sha();menu=w.menu(12);t0=time.perf_counter_ns()
        action,probs,cache,attempted,planned_epoch,ncalls,source=choose(
            w,goal,model,arm,tick,phase,epoch,cache,attempted,planned_epoch,changed)
        lat.append((time.perf_counter_ns()-t0)/1e6)
        assert action in menu,(arm,action,menu)
        rec=strict_step(w,action)
        if rec is None:reason='logical_gate';break
        calls+=ncalls
        entries.append({'tick':tick,'phase':phase,'epoch':epoch,'goal':list(goal),
                        'pre_sha256':pre,'menu':menu,'action':action,'probabilities':probs,
                        'source':source,'planner_calls':ncalls,'receipt':rec})
    row={'world_family':sp.family,'world_seed':sp.seed,'layout':layout,'order':order,
         'seed':seed,'arm':arm,'final_goal':list(goals[2]),'arrived_final':phase==2 and w.cursor==goals[2],
         'phase_hits':hits,'reason':reason,'switches_seen':len(switches),'material_events':len(events),
         'initial_sha256':initial,'base_sha256':base,'final_sha256':w.state_sha(),
         'trace_sha256':digest(w.decisions),'logical_moves':w.logical_moves,'decisions':len(entries),
         'planner_calls':calls,'neural_actions':sum(e['source']=='neural' for e in entries),
         'action_counts':dict(sorted(Counter(e['action'] for e in entries).items())),
         'source_counts':dict(sorted(Counter(e['source'] for e in entries).items())),
         'material_applied':sum(p['applied'] for ev in events for p in ev['operations']),
         'material_suppressed':sum(not p['applied'] for ev in events for p in ev['operations']),
         'illegal_edits':len(w.ink.intersection(w.materials.keys()|w.protected)),
         'median_choose_ms':round(statistics.median(lat),6) if lat else 0,
         'p95_choose_ms':round(sorted(lat)[int(.95*(len(lat)-1))],6) if lat else 0}
    trace={'spec':vars(sp),'row':row,'entries':entries,'switches':switches,'material_events':events}
    return row,trace


def verify_trace(tr,model):
    row=tr['row'];sp=Spec(**tr['spec']);w=old.overlay_world(sp,row['layout'])
    assert row['initial_sha256']==w.state_sha() and row['base_sha256']==w.base_sha()
    goals=(GOALS[row['order']],GOALS[1-row['order']],GOALS[row['order']]);phase=0;epoch=0
    cache=None;attempted=False;planned_epoch=-1
    its=iter(tr['entries']);events=iter(tr['material_events']);switches=iter(tr['switches'])
    hits=[False]*3;calls=0;num=vecs=0
    for tick in range(CAP):
        changed=False
        if tick in EVENT_TICKS:
            ev=apply_event(w,row['layout'],tick);assert next(events)==ev
            epoch=ev['material_epoch'];changed=True
        if phase<2 and tick==SCHEDULE[phase]:
            expected={'tick':tick,'old_phase':phase,'pre_sha256':w.state_sha(),
                      'next_goal':list(goals[phase+1]),'discarded_cache':cache is not None}
            assert next(switches)==expected
            phase+=1;cache=None;attempted=False;planned_epoch=-1
        goal=goals[phase]
        if w.cursor==goal:
            hits[phase]=True
            if phase==2:reason='final_goal';break
            continue
        if w.stopped:reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        pre=w.state_sha();menu=w.menu(12)
        action,probs,cache,attempted,planned_epoch,ncalls,source=choose(
            w,goal,model,row['arm'],tick,phase,epoch,cache,attempted,planned_epoch,changed)
        assert action in menu
        rec=strict_step(w,action)
        if rec is None:reason='logical_gate';break
        item=next(its)
        assert (item['tick'],item['phase'],item['epoch'],item['goal'],item['pre_sha256'],item['menu'])==(
            tick,phase,epoch,list(goal),pre,menu)
        assert (item['action'],item['probabilities'],item['source'],item['planner_calls'],item['receipt'])==(
            action,probs,source,ncalls,rec)
        calls+=ncalls;num+=1;vecs+=probs is not None
    else:reason='decision_gate'
    assert next(its,None) is None and next(events,None) is None and next(switches,None) is None
    assert row['phase_hits']==hits and row['reason']==reason
    assert row['arrived_final']==(phase==2 and w.cursor==goals[2])
    assert row['final_sha256']==w.state_sha() and row['trace_sha256']==digest(w.decisions)
    assert row['planner_calls']==calls and row['decisions']==num and row['logical_moves']==w.logical_moves<=CAP
    assert not w.ink.intersection(w.materials.keys()|w.protected) and row['illegal_edits']==0
    assert row['material_events']==len(tr['material_events'])
    return num,vecs


def summarize(rows):
    out={}
    for arm in ARMS:
        r=[x for x in rows if x['arm']==arm];actions=Counter();sources=Counter()
        for x in r:actions.update(x['action_counts']);sources.update(x['source_counts'])
        out[arm]={'episodes':len(r),'final_goals':sum(x['arrived_final'] for x in r),
                  'planner_calls':sum(x['planner_calls'] for x in r),
                  'neural_actions':sum(x['neural_actions'] for x in r),
                  'material_events':sum(x['material_events'] for x in r),
                  'applied_patches':sum(x['material_applied'] for x in r),
                  'suppressed_patches':sum(x['material_suppressed'] for x in r),
                  'mean_decisions':round(statistics.mean(x['decisions'] for x in r),3),
                  'median_episode_choose_ms':round(statistics.median(x['median_choose_ms'] for x in r),6),
                  'by_layout':{k:sum(x['arrived_final'] for x in r if x['layout']==k) for k in LAYOUTS},
                  'reasons':dict(sorted(Counter(x['reason'] for x in r).items())),
                  'actions':dict(sorted(actions.items())),'sources':dict(sorted(sources.items()))}
    return out


def study(out:Path, checkpoints:Path, n:int=1):
    if not 1<=n<=2:raise ValueError('prescribed bounded DEVELOPMENT n=1..2')
    out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    start=time.perf_counter();models={seed:GoalRanker.load(checkpoints/f'{seed}-conditioned.npz') for seed in SEEDS}
    rows=[];receipts=vecs=0
    for sp in specs('development',n):
        for layout in LAYOUTS:
            for order in (0,1):
                for seed in SEEDS:
                    for arm in ARMS:
                        name=f'{seed}-{sp.family}-{sp.seed}-{layout}-{order}-{arm}.json'
                        target=out/'traces'/name
                        if target.exists():
                            tr=json.loads(target.read_text());r=tr['row']
                        else:
                            r,tr=episode(sp,layout,order,seed,arm,models[seed])
                            target.write_text(json.dumps(tr,sort_keys=True,separators=(',',':'))+'\n')
                        assert (r['world_seed'],r['layout'],r['order'],r['seed'],r['arm'])==(sp.seed,layout,order,seed,arm)
                        receipts+=len(tr['entries']);vecs+=sum(e['probabilities'] is not None for e in tr['entries'])
                        rows.append(r)
    pairs=[]
    for seed in SEEDS:
        for sp in specs('development',n):
            for layout in LAYOUTS:
                for order in (0,1):
                    group={x['arm']:x for x in rows if x['seed']==seed and x['world_seed']==sp.seed and x['layout']==layout and x['order']==order}
                    pairs.append({'seed':seed,'family':sp.family,'world_seed':sp.seed,'layout':layout,'order':order,
                                  'success':{arm:group[arm]['arrived_final'] for arm in ARMS}})
    ev={'version':VERSION,'evaluation_split':'development','heldout_opened':False,
        'native_receipts':0,'optimizer_updates':0,'paid_gpu_api_calls':0,
        'synthetic_64_move_gate':True,'material_event_ticks':EVENT_TICKS,
        'goal_switch_ticks':SCHEDULE,'arms':ARMS,'layouts':LAYOUTS,'material_patches':PATCHES,
        'source_sha256':{name:sha_file(Path(__file__).parent/name) for name in SOURCE},
        'checkpoint_sha256':{str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS},
        'episodes':len(rows),'verified_action_receipts':receipts,'verified_neural_distributions':vecs,
        'summary':summarize(rows),'pairs':pairs,'rows':rows,
        'execution_wall_s':round(time.perf_counter()-start,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limitations':['Prespecified DEVELOPMENT overlays with externally scripted material changes, NOT reactive native physics.',
                       'Immutable historical d_common is a fixture; no proof of native ABI compatibility.',
                       'Material closure suppressed if ink/cursor occupies cell; source hashes record actual result.',
                       'Prefix planner restricted to BACKTRACK-prefix then empty-cell paths; no global completeness.',
                       'Goal-switch ticks and exact material events are externally scheduled, not learned.',
                       'Frozen goal-conditioned tiny neural checkpoints with engineered Manhattan features.',
                       'No new checkpoint, structural HELDOUT, or native game execution; no immutable Actions artifacts.']}
    (out/'evidence.json').write_text(json.dumps(ev,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return ev


def _verify_shard(paths,checkpoints):
    models={seed:GoalRanker.load(Path(checkpoints)/f'{seed}-conditioned.npz') for seed in SEEDS}
    receipts=vecs=0;rows=[]
    for name in paths:
        tr=json.loads(Path(name).read_text());a,b=verify_trace(tr,models[tr['row']['seed']])
        receipts+=a;vecs+=b;rows.append(tr['row'])
    return receipts,vecs,rows


def verify(out:Path,checkpoints:Path):
    ev=json.loads((out/'evidence.json').read_text());h=json.loads((out/'SHA256SUMS.json').read_text())
    assert h=={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert ev['source_sha256']=={name:sha_file(Path(__file__).parent/name) for name in SOURCE}
    assert ev['checkpoint_sha256']=={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    paths=[str(p) for p in sorted((out/'traces').glob('*.json'))]
    assert len(paths)==ev['episodes']
    shards=[paths[i::4] for i in range(4)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(_verify_shard,shards,[str(checkpoints)]*4))
    receipts=sum(x[0] for x in results);vecs=sum(x[1] for x in results)
    rows=[r for x in results for r in x[2]]
    assert (len(rows),receipts,vecs)==(ev['episodes'],ev['verified_action_receipts'],ev['verified_neural_distributions'])
    assert ev['summary']==summarize(rows)
    assert sorted(json.dumps(x,sort_keys=True) for x in rows)==sorted(json.dumps(x,sort_keys=True) for x in ev['rows'])
    return ev


def main():
    ap=argparse.ArgumentParser();ap.add_argument('cmd',choices=('run','verify'))
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--n',type=int,default=1);a=ap.parse_args()
    ev=study(a.out,a.checkpoints,a.n) if a.cmd=='run' else verify(a.out,a.checkpoints)
    assert verify(a.out,a.checkpoints)==json.loads(json.dumps(ev))
    print(json.dumps({'episodes':ev['episodes'],'receipts':ev['verified_action_receipts'],
                      'neural_vectors':ev['verified_neural_distributions'],
                      'summary':ev['summary'],'execution_wall_s':ev['execution_wall_s'],
                      'peak_rss_kib':ev['peak_rss_kib']},sort_keys=True))
if __name__=='__main__':main()

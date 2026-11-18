"""TOPOLOGY-ADAPTIVE-HANDOFF-001: prespecified adaptive sparse interventions on held-fixed DEV overlays.

No CyberSand mutations; immutable World/ABI. All overlays are research-only,
synthetic, constructed before the first action. No HELDOUT access or training.
"""
from __future__ import annotations
import argparse, json, resource, statistics, time
from collections import Counter
from pathlib import Path

from .world import World, Spec, specs, SOLID, MATERIAL_Z, digest
from .detour_recovery_lab import prefix_plan
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .train import sha_file

VERSION='topology-adaptive-handoff/1'
SEEDS=(711,712)
CAP=64
LAYOUTS=('opposing_gates','enclosed_courtyard','horizontal_switchback','protected_slalom')
GOALS=((67,48),(46,64))
SCHEDULE=(12,30)
ARMS=('neural','absolute24','planner_each_stage','second_switch','each_switch','budget_slack10','second_switch_stall3')
SOURCES=('topology_adaptive_handoff_lab.py','topology_goal_churn_lab.py','world.py','goal_conditioning_lab.py','goal_observability_lab.py',
         'detour_recovery_lab.py','policy.py','train.py')

def overlay_world(spec,layout):
    """Distinct, predeclared material/protected topology, fixed from episode start.

    The central 30x30 arena is cleared of seed-specific materials/protection
    before introducing a deterministic experiment-only mask. Surrounding
    DEVELOPMENT background remains untouched and is included in base_sha.
    """
    if spec.split!='development' or spec.challenge!='normal' or layout not in LAYOUTS:
        raise ValueError('DEVELOPMENT normal challenge and enumerated layouts only')
    w=World.create(spec)
    for p in tuple(w.materials):
        if 41<=p[0]<=71 and 39<=p[1]<=69: del w.materials[p]
    w.protected.difference_update({p for p in w.protected if 41<=p[0]<=71 and 39<=p[1]<=69})
    def line(points, kind=SOLID):
        for p in points:
            assert 41<=p[0]<=71 and 39<=p[1]<=69 and p not in (w.cursor,*GOALS)
            assert p not in w.ink
            if kind=='protected': w.protected.add(p)
            else: w.materials[p]=kind
    if layout=='opposing_gates':
        for x,gap in ((53,43),(60,58)):
            line(((x,y) for y in range(40,68) if abs(y-gap)>1), MATERIAL_Z if x==53 else SOLID)
    elif layout=='enclosed_courtyard':
        # Courtyard around eastern objective, one north-facing entrance.
        line(((x,42) for x in range(57,71) if x not in (59,60)),MATERIAL_Z)
        line(((x,57) for x in range(57,71)),SOLID)
        line(((57,y) for y in range(42,58)),SOLID)
        line(((70,y) for y in range(42,58)),SOLID)
        # two detached pillars near the southern objective
        line(((x,59) for x in range(50,55)), MATERIAL_Z)
    elif layout=='horizontal_switchback':
        for y,gap in ((52,66),(59,44)):
            line(((x,y) for x in range(42,71) if abs(x-gap)>1), SOLID if y==52 else MATERIAL_Z)
        line(((62,y) for y in range(43,49)),SOLID)
    elif layout=='protected_slalom':
        # Protected cells alternate with material barriers, distinct from the other three.
        for x,gap in ((53,42),(58,57),(63,44)):
            line(((x,y) for y in range(41,65) if abs(y-gap)>2), 'protected' if x==58 else MATERIAL_Z)
        line(((x,62) for x in range(47,52)),SOLID)
    assert not w.occupied(w.cursor) and all(not w.occupied(g) for g in GOALS)
    assert not w.ink and w.logical_moves==0
    return w


def l1(w,goal):
    return abs(w.cursor[0]-goal[0])+abs(w.cursor[1]-goal[1])


def plan_or_neural(w,goal,model,arm,tick,stage,stage_at,stall,cache,attempted):
    """One bounded search per goal stage; plan is invalidated on every switch."""
    if cache is not None:
        if cache:
            if cache[0] not in w.menu(12):return 'STOP',None,(),True,0,'stale_plan'
            return cache[0],None,cache[1:],True,0,'cached_plan'
        return 'STOP',None,(),True,0,'exhausted_plan'
    want=(arm=='planner_each_stage' or (arm=='absolute24' and tick>=24)
          or (arm=='second_switch' and stage==2)
          or (arm=='each_switch' and stage>0)
          or (arm=='budget_slack10' and stage>0 and
              (CAP-w.logical_moves<=l1(w,goal)+10 or stall>=4))
          or (arm=='second_switch_stall3' and stage==2 and (stall>=3 or tick>=36)))
    if want and not attempted:
        plan=prefix_plan(w,goal,CAP)
        return (plan[0] if plan else 'STOP'),None,(plan[1:] if plan else ()),True,1,('new_plan' if plan else 'infeasible_plan')
    act,prob=model.choose(w,goal,CAP,False)
    return act,prob,None,attempted,0,'neural'


def episode(sp,layout,order,seed,arm,model):
    w=overlay_world(sp,layout);prelude=list(w.decisions);base=w.base_sha();initial=w.state_sha()
    goals=(GOALS[order],GOALS[1-order],GOALS[order])
    phase=0;phase_at=0;stall=0;best=l1(w,goals[0]);cache=None;attempted=False
    switches=[];entries=[];latencies=[];calls=0;hits=[False]*3;reason='decision_gate'
    for tick in range(CAP):
        if phase<2 and tick==SCHEDULE[phase]:
            switches.append({'tick':tick,'old_phase':phase,'pre_sha256':w.state_sha(),
                             'next_goal':list(goals[phase+1]),'discarded_cache':cache is not None})
            phase+=1;phase_at=tick;stall=0;best=l1(w,goals[phase]);cache=None;attempted=False
        goal=goals[phase]
        if w.cursor==goal:
            hits[phase]=True
            if phase==2:reason='final_goal';break
            continue  # external idle; no host-legal command executed on this tick
        if w.stopped: reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        pre=w.state_sha();menu=w.menu(12)
        st=time.perf_counter_ns()
        action,prob,cache,attempted,used,src=plan_or_neural(w,goal,model,arm,tick,phase,phase_at,stall,cache,attempted)
        latencies.append((time.perf_counter_ns()-st)/1e6)
        assert action in menu,(arm,action,menu)
        receipt=strict_step(w,action)
        if receipt is None:reason='logical_gate';break
        calls+=used
        current=l1(w,goal)
        if current<best:best=current;stall=0
        else:stall+=1
        entries.append({'tick':tick,'phase':phase,'goal':list(goal),'pre_sha256':pre,
                        'menu':menu,'action':action,'probabilities':prob,'source':src,
                        'planner_calls':used,'receipt':receipt})
    row={'world_family':sp.family,'world_seed':sp.seed,'layout':layout,'order':order,
         'model_seed':seed,'arm':arm,'final_goal':list(goals[2]),'arrived_final':phase==2 and w.cursor==goals[2],
         'switches_seen':len(switches),'phase_hits':hits,'reason':reason,
         'initial_sha256':initial,'base_sha256':base,'final_sha256':w.state_sha(),
         'trace_sha256':digest(w.decisions),'logical_moves':w.logical_moves,
         'decisions':len(entries),'planner_calls':calls,
         'neural_actions':sum(e['source']=='neural' for e in entries),
         'planner_actions':sum(e['source'] in ('new_plan','cached_plan') for e in entries),
         'action_counts':dict(sorted(Counter(e['action'] for e in entries).items())),
         'illegal_material_edits':len(w.ink & (set(w.materials)|w.protected)),
         'median_choose_ms':round(statistics.median(latencies),6) if latencies else 0,
         'max_choose_ms':round(max(latencies),6) if latencies else 0}
    return row,{'spec':vars(sp),'row':row,'switches':switches,'entries':entries,'prelude':prelude}


def verify_trace(tr,model):
    row=tr['row'];sp=Spec(**tr['spec']);w=overlay_world(sp,row['layout'])
    assert row['initial_sha256']==w.state_sha() and row['base_sha256']==w.base_sha() and tr['prelude']==w.decisions
    goals=(GOALS[row['order']],GOALS[1-row['order']],GOALS[row['order']]);phase=0;phase_at=0
    best=l1(w,goals[0]);stall=0;cache=None;attempted=False;hits=[False]*3
    si=iter(tr['switches']);ei=iter(tr['entries']);calls=0;count=0;vec=0
    for tick in range(CAP):
        if phase<2 and tick==SCHEDULE[phase]:
            sw=next(si);assert sw=={'tick':tick,'old_phase':phase,'pre_sha256':w.state_sha(),
                                   'next_goal':list(goals[phase+1]),'discarded_cache':cache is not None}
            phase+=1;phase_at=tick;stall=0;best=l1(w,goals[phase]);cache=None;attempted=False
        goal=goals[phase]
        if w.cursor==goal:
            hits[phase]=True
            if phase==2:reason='final_goal';break
            continue
        if w.stopped:reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        before=w.state_sha();menu=w.menu(12)
        action,prob,cache,attempted,used,src=plan_or_neural(w,goal,model,row['arm'],tick,phase,phase_at,stall,cache,attempted)
        assert action in menu
        cost=len(w.path)-1 if action=='RELOCATE_START' else 0 if action=='STOP' else 1
        if w.logical_moves+cost>CAP:reason='logical_gate';break
        ent=next(ei)
        assert (ent['tick'],ent['phase'],ent['goal'],ent['pre_sha256'],ent['menu'])==(tick,phase,list(goal),before,menu)
        assert (ent['action'],ent['probabilities'],ent['source'],ent['planner_calls'])==(action,prob,src,used)
        assert ent['receipt']==strict_step(w,action)
        count+=1;vec+=int(prob is not None);calls+=used
        current=l1(w,goal)
        if current<best:best=current;stall=0
        else:stall+=1
    else:reason='decision_gate'
    assert next(si,None) is None and next(ei,None) is None
    assert row['reason']==reason and row['phase_hits']==hits and row['switches_seen']==len(tr['switches'])
    assert row['arrived_final']==(phase==2 and w.cursor==goals[2])
    assert row['planner_calls']==calls<=3 and row['logical_moves']==w.logical_moves<=CAP
    assert row['decisions']==count and row['final_sha256']==w.state_sha() and row['trace_sha256']==digest(w.decisions)
    assert len(w.ink & (set(w.materials)|w.protected))==row['illegal_material_edits']==0
    return count,vec


def summarize(rows):
    out={}
    for arm in ARMS:
        group=[r for r in rows if r['arm']==arm]
        actions=Counter()
        for r in group:actions.update(r['action_counts'])
        out[arm]={'episodes':len(group),'final_arrivals':sum(r['arrived_final'] for r in group),
                  'both_switches_seen':sum(r['switches_seen']==2 for r in group),
                  'planner_calls':sum(r['planner_calls'] for r in group),
                  'neural_actions':sum(r['neural_actions'] for r in group),
                  'planner_actions':sum(r['planner_actions'] for r in group),
                  'mean_actions':round(statistics.mean(r['decisions'] for r in group),3),
                  'reasons':dict(sorted(Counter(r['reason'] for r in group).items())),
                  'layouts':{layout:sum(r['arrived_final'] for r in group if r['layout']==layout) for layout in LAYOUTS},
                  'goal_orders':{str(order):sum(r['arrived_final'] for r in group if r['order']==order) for order in (0,1)},
                  'median_episode_choose_ms':round(statistics.median(r['median_choose_ms'] for r in group),6),
                  'action_counts':dict(sorted(actions.items()))}
    return out


def run(out,checkpoints,n=1):
    out.mkdir(parents=True,exist_ok=True);(out/'traces').mkdir(exist_ok=True)
    started=time.perf_counter();models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    cps={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];receipts=0;vectors=0;excluded=[];initial_feasible=0
    for sp in specs('development',n):
        for layout in LAYOUTS:
            w=overlay_world(sp,layout)
            if any(prefix_plan(w,g,CAP) is None for g in GOALS):
                excluded.append({'spec':sp.ident,'layout':layout,'reason':'one_or_more_initial_goals_unreachable'})
                continue
            initial_feasible+=1
            for order in (0,1):
                for seed in SEEDS:
                    for arm in ARMS:
                        nm=f'{seed}-{sp.family}-{sp.seed}-{layout}-{order}-{arm}.json'
                        trace_path=out/'traces'/nm
                        if trace_path.exists():
                            tr=json.loads(trace_path.read_text());row=tr['row']
                            assert (row['world_family'],row['world_seed'],row['layout'],row['order'],row['model_seed'],row['arm'])==(sp.family,sp.seed,layout,order,seed,arm)
                        else:
                            row,tr=episode(sp,layout,order,seed,arm,models[seed])
                            trace_path.write_text(json.dumps(tr,sort_keys=True,separators=(',',':'))+'\n')
                        count,vec=verify_trace(tr,models[seed]);receipts+=count;vectors+=vec;rows.append(row)
    evidence={'version':VERSION,'split':'development','heldout_opened':False,
              'native_receipts':0,'new_optimizer_updates':0,'synthetic_64_gate':True,
              'source_sha256':{p:sha_file(Path(__file__).parent/p) for p in SOURCES},
              'checkpoint_sha256':cps,'predeclared_arms':list(ARMS),'goals':[list(g) for g in GOALS],
              'scheduled_switch_ticks':list(SCHEDULE),'layouts':list(LAYOUTS),
              'candidate_scenarios':len(specs('development',n))*len(LAYOUTS),
              'initially_feasible_scenarios':initial_feasible,'excluded':excluded,
              'episodes':len(rows),'verified_action_receipts':receipts,'verified_neural_distributions':vectors,
              'summary':summarize(rows),'rows':rows,'wall_s':round(time.perf_counter()-started,3),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations':['Research-only overlay clears 31x31 central area; no native environment editing or acceptance.',
                             'All tests remain DEVELOPMENT families, not sealed structural HELDOUT.',
                             'Goals switch by external synthetic schedule, with early-arrival idles not host actions.',
                             'Initial feasibility filter may bias scenarios; infeasible continuation states count as failures.',
                             'Prefix search complete only within backtrack-prefix then empty-cell movement.',
                             'The frozen goal ranker receives engineered Manhattan-distance features.',
                             'The budget_slack10 trigger uses a Manhattan lower bound, not an actual route-feasibility estimate.',
                             'Second-switch-only arms can fail before second switch and do not rescue premature STOP.',
                             'No material changes during episodes; masks are fixed at initialization.',
                             'Candidate triggers were fixed before viewing outcomes; reused DEVELOPMENT geometry is not independent validation.',
                             'Checkpoint binaries locally retained, not yet immutable Actions artifacts.']}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},indent=2,sort_keys=True)+'\n')
    return evidence


def verify(out,checkpoints):
    e=json.loads((out/'evidence.json').read_text());hashes=json.loads((out/'SHA256SUMS.json').read_text())
    assert hashes=={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert e['source_sha256']=={p:sha_file(Path(__file__).parent/p) for p in SOURCES}
    assert e['checkpoint_sha256']=={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];receipts=0;vectors=0
    for p in sorted((out/'traces').glob('*.json')):
        tr=json.loads(p.read_text());rows.append(tr['row'])
        c,v=verify_trace(tr,models[tr['row']['model_seed']]);receipts+=c;vectors+=v
    assert sorted(json.dumps(r,sort_keys=True) for r in rows)==sorted(json.dumps(r,sort_keys=True) for r in e['rows'])
    assert e['summary']==summarize(rows) and e['episodes']==len(rows)
    assert e['verified_action_receipts']==receipts and e['verified_neural_distributions']==vectors
    return e


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--worlds-per-family',type=int,default=1);ap.add_argument('--verify',action='store_true')
    a=ap.parse_args()
    e=verify(a.out,a.checkpoints) if a.verify else run(a.out,a.checkpoints,a.worlds_per_family)
    print(json.dumps({'episodes':e['episodes'],'receipts':e['verified_action_receipts'],
                      'vectors':e['verified_neural_distributions'],'summary':e['summary'],
                      'excluded':e['excluded'],'wall_s':e['wall_s'],'peak_rss_kib':e['peak_rss_kib']},indent=2))
if __name__=='__main__':main()

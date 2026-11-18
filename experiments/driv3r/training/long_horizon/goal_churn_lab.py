"""GOAL-CHURN-001: two externally scheduled target reversals on fixed legal World.

Synthetic DEVELOPMENT-only counterfactual. No World/ABI mutations, no heldout.
Each controller gets the same timed target schedule and base geometry. Planner
state must be invalidated on *every* target replacement. An early goal may
pause host actions until the scheduled switch (external idle, not STOP).
"""
from __future__ import annotations
import argparse, hashlib, json, resource, statistics, time
from collections import Counter
from pathlib import Path
from .retarget_handoff_lab import LAYOUTS as ALL_LAYOUTS
LAYOUTS = ('original', 'east_sleeve', 'staggered_piers')
from .sparse_handoff_lab import fixture, distance, CAP, SEEDS
from .detour_recovery_lab import GOALS, prefix_plan
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .world import Spec, specs, MOVE, digest
from .train import sha_file

VERSION = 'goal-churn/1'
SCHEDULES = ((8,24),)
ARMS = ('neural','absolute24','relative8','stall4','switch_plan','plan_every_stage')
SOURCES = ('goal_churn_lab.py','retarget_handoff_lab.py','sparse_handoff_lab.py',
           'detour_recovery_lab.py','goal_conditioning_lab.py','goal_observability_lab.py',
           'world.py','policy.py','train.py')


def propose(w,goal,model,arm,tick,phase,phase_start,stall,planned,cache):
    """Return (action, probabilities, plan_remaining, planned, calls, selection).

    planned records at most one attempt per stage, including infeasible searches.
    """
    if cache is not None:
        if cache:
            if cache[0] not in w.menu(12):
                return 'STOP',None,(),True,0,'stale_plan'
            return cache[0],None,cache[1:],True,0,'cached_plan'
        return 'STOP',None,(),True,0,'exhausted_plan'
    request = (arm=='plan_every_stage' or
               (arm=='switch_plan' and phase>0) or
               (arm=='absolute24' and tick>=24) or
               (arm=='relative8' and tick-phase_start>=8) or
               (arm=='stall4' and stall>=4))
    if request and not planned:
        plan=prefix_plan(w,goal,CAP)
        if plan is None: return 'STOP',None,(),True,1,'infeasible_plan'
        assert plan and plan[0] in w.menu(12)
        return plan[0],None,plan[1:],True,1,'new_plan'
    action,prob=model.choose(w,goal,CAP,False)
    return action,prob,None,planned,0,'neural'


def episode(sp,layout,order,seed,arm,schedule,model):
    w=fixture(sp,5,layout)
    goals=(GOALS[order],GOALS[1-order],GOALS[order])
    initial=w.state_sha();base=w.base_sha();prelude=list(w.decisions)
    phase=0;phase_start=0;best=distance(w,goals[0]);stall=0;planned=False;cache=None
    entries=[];switches=[];ms=[];reason='gate';calls=0;phase_hits=[False]*3
    for tick in range(CAP):
        if phase<2 and tick==schedule[phase]:
            switches.append({'tick':tick,'stage_before':phase,'pre_sha256':w.state_sha(),
                             'next_goal':list(goals[phase+1]),'cache_discarded':cache is not None})
            phase+=1;phase_start=tick;best=distance(w,goals[phase]);stall=0;planned=False;cache=None
        goal=goals[phase]
        if w.cursor==goal:
            phase_hits[phase]=True
            if phase==2:reason='final_goal';break
            # Research controller is externally idle until the next scheduled goal.
            continue
        if w.stopped:reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        pre=w.state_sha();menu=w.menu(12)
        start=time.perf_counter_ns()
        action,prob,cache,planned,used,source=propose(w,goal,model,arm,tick,phase,phase_start,stall,planned,cache)
        ms.append((time.perf_counter_ns()-start)/1e6)
        assert action in menu
        receipt=strict_step(w,action)
        if receipt is None: reason='logical_gate';break
        calls+=used
        current=distance(w,goal)
        if current<best:best=current;stall=0
        else:stall+=1
        entries.append({'tick':tick,'phase':phase,'goal':list(goal),'pre_sha256':pre,
                        'menu':menu,'action':action,'probabilities':prob,'source':source,
                        'planner_calls':used,'receipt':receipt})
    else:reason='decision_gate'
    row={'family':sp.family,'world_seed':sp.seed,'layout':layout,'order':order,'seed':seed,
         'arm':arm,'schedule':list(schedule),'arrived_final':w.cursor==goals[2] and phase==2,
         'switches_seen':len(switches),'phase_hits':phase_hits,'reason':reason,
         'initial_sha256':initial,'base_sha256':base,'final_sha256':w.state_sha(),
         'trace_sha256':digest(w.decisions),'logical_moves':w.logical_moves,
         'decisions':len(entries),'planner_calls':calls,
         'neural_actions':sum(e['source']=='neural' for e in entries),
         'planner_actions':sum(e['source'] in ('new_plan','cached_plan') for e in entries),
         'action_counts':dict(sorted(Counter(e['action'] for e in entries).items())),
         'illegal_material_edits':len(w.ink & (set(w.materials)|w.protected)),
         'median_choose_ms':round(statistics.median(ms),6) if ms else 0,
         'max_choose_ms':round(max(ms),6) if ms else 0}
    return row,{'spec':vars(sp),'row':row,'prelude':prelude,'switches':switches,'entries':entries}


def verify_trace(tr,model):
    """Independent deterministic action/receipt/probability replay and sha check."""
    row=tr['row'];sp=Spec(**tr['spec']);schedule=tuple(row['schedule'])
    w=fixture(sp,5,row['layout']);goals=(GOALS[row['order']],GOALS[1-row['order']],GOALS[row['order']])
    assert tr['prelude']==w.decisions and w.state_sha()==row['initial_sha256'] and w.base_sha()==row['base_sha256']
    phase=0;phase_start=0;stall=0;best=distance(w,goals[0]);planned=False;cache=None
    entries=iter(tr['entries']);switches=iter(tr['switches']);calls=0;seen=0;hits=[False]*3
    for tick in range(CAP):
        if phase<2 and tick==schedule[phase]:
            sw=next(switches);assert sw=={'tick':tick,'stage_before':phase,'pre_sha256':w.state_sha(),
                   'next_goal':list(goals[phase+1]),'cache_discarded':cache is not None}
            phase+=1;phase_start=tick;stall=0;best=distance(w,goals[phase]);planned=False;cache=None;seen+=1
        goal=goals[phase]
        if w.cursor==goal:
            hits[phase]=True
            if phase==2:reason='final_goal';break
            continue
        if w.stopped:reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        pre=w.state_sha();menu=w.menu(12)
        act,prob,cache,planned,used,src=propose(w,goal,model,row['arm'],tick,phase,phase_start,stall,planned,cache)
        assert act in menu
        cost=(len(w.path)-1 if act=='RELOCATE_START' else 0 if act=='STOP' else 1)
        if w.logical_moves+cost>CAP:reason='logical_gate';break
        e=next(entries)
        assert (e['tick'],e['phase'],e['goal'],e['pre_sha256'],e['menu'])==(tick,phase,list(goal),pre,menu)
        assert (e['action'],e['probabilities'],e['source'],e['planner_calls'])==(act,prob,src,used)
        assert e['receipt']==strict_step(w,act)
        calls+=used
        now=distance(w,goal)
        if now<best:best=now;stall=0
        else:stall+=1
    else:reason='decision_gate'
    assert next(entries,None) is None and next(switches,None) is None
    assert row['reason']==reason and row['switches_seen']==seen and row['phase_hits']==hits
    assert row['arrived_final']==(phase==2 and w.cursor==goals[2])
    assert row['planner_calls']==calls<=3 and row['logical_moves']==w.logical_moves<=CAP
    assert row['trace_sha256']==digest(w.decisions) and row['final_sha256']==w.state_sha()
    assert len(w.ink & (set(w.materials)|w.protected))==row['illegal_material_edits']==0
    return len(tr['entries']),sum(e['probabilities'] is not None for e in tr['entries'])


def summarize(rows):
    out={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        actions=Counter()
        for r in rr:actions.update(r['action_counts'])
        out[arm]={'episodes':len(rr),'at_least_one_switch':sum(r['switches_seen']>=1 for r in rr),
                  'both_switches_seen':sum(r['switches_seen']==2 for r in rr),
                  'final_arrivals':sum(r['arrived_final'] for r in rr),
                  'planner_calls':sum(r['planner_calls'] for r in rr),
                  'neural_actions':sum(r['neural_actions'] for r in rr),
                  'planner_actions':sum(r['planner_actions'] for r in rr),
                  'mean_decisions':round(statistics.mean(r['decisions'] for r in rr),3),
                  'median_episode_choose_ms':round(statistics.median(r['median_choose_ms'] for r in rr),6),
                  'action_counts':dict(sorted(actions.items()))}
    return out


def run(out,checkpoints,n=1):
    assert not out.exists();out.mkdir(parents=True);(out/'traces').mkdir()
    start=time.perf_counter();models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    cps={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];receipts=vectors=0;excluded=[]
    for base in specs('development',n):
        sp=Spec(base.split,base.family,base.seed,base.mode,'trap_corridor')
        for layout in LAYOUTS:
            for order in (0,1):
                w=fixture(sp,5,layout)
                if any(prefix_plan(w,g,CAP) is None for g in GOALS):
                    excluded.append({'spec':sp.ident,'layout':layout,'order':order,'reason':'initial_goal_unreachable'})
                    continue
                for schedule in SCHEDULES:
                    for seed in SEEDS:
                        for arm in ARMS:
                            r,tr=episode(sp,layout,order,seed,arm,schedule,models[seed])
                            count,nvec=verify_trace(tr,models[seed]);receipts+=count;vectors+=nvec
                            rows.append(r)
                            name=f'{seed}-{base.family}-{base.seed}-{layout}-{order}-{schedule[0]}-{schedule[1]}-{arm}.json'
                            (out/'traces'/name).write_text(json.dumps(tr,sort_keys=True,separators=(',',':'))+'\n')
    evidence={'version':VERSION,'split':'development','heldout_opened':False,
              'native_receipts':0,'new_optimizer_updates':0,'synthetic_64_gate':True,
              'schedule_ticks':SCHEDULES,'checkpoint_sha256':cps,
              'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in SOURCES},
              'candidate_scenarios':len(specs('development',n))*len(LAYOUTS)*2*len(SCHEDULES),
              'excluded':excluded,'episodes':len(rows),'verified_action_receipts':receipts,
              'verified_neural_distributions':vectors,'summary':summarize(rows),'rows':rows,
              'cpu_wall_s':round(time.perf_counter()-start,3),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'caveats':['Synthetic scheduled goal changes are external instructions; native CyberSand does not establish them.',
                         'DEVELOPMENT shares structural families with training; HELDOUT remains sealed.',
                         'All layouts use a trap-corridor base; results are not broad topology transfer.',
                         'When a goal is reached early, controller idles externally without issuing an action.',
                         'A neural STOP prevents future target switches, which reduces exposure denominators.',
                         'Prefix planner is restricted to backtrack-prefix then empty-cell BFS; not a native optimal controller.',
                         'No material-state changes during episodes. No host runtime/ABI acceptance.',
                         'No checkpoint updates; frozen tiny-model weights only.']}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},indent=2,sort_keys=True)+'\n')
    return evidence


def verify(out,checkpoints):
    e=json.loads((out/'evidence.json').read_text()); hashes=json.loads((out/'SHA256SUMS.json').read_text())
    assert hashes=={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert e['source_sha256']=={n:sha_file(Path(__file__).parent/n) for n in SOURCES}
    models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    assert e['checkpoint_sha256']=={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];receipts=vectors=0
    for p in sorted((out/'traces').glob('*.json')):
        tr=json.loads(p.read_text());rows.append(tr['row']);n,v=verify_trace(tr,models[tr['row']['seed']]);receipts+=n;vectors+=v
    assert sorted(json.dumps(r,sort_keys=True) for r in e['rows'])==sorted(json.dumps(r,sort_keys=True) for r in rows)
    assert e['summary']==summarize(rows) and e['episodes']==len(rows)
    assert e['verified_action_receipts']==receipts and e['verified_neural_distributions']==vectors
    return e


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--worlds-per-family',type=int,default=1);ap.add_argument('--verify',action='store_true')
    a=ap.parse_args()
    e=verify(a.out,a.checkpoints) if a.verify else run(a.out,a.checkpoints,a.worlds_per_family)
    print(json.dumps({k:e[k] for k in ('episodes','verified_action_receipts','verified_neural_distributions','summary','cpu_wall_s','peak_rss_kib') if k in e},indent=2))


if __name__=='__main__':main()

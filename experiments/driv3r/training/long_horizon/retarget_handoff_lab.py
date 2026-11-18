"""RETARGET-HANDOFF-001: a scheduled goal replacement during legal synthetic play.

DEVELOPMENT only. Frozen real NumPy checkpoints; no native commands, training,
changes to the immutable World, or claims of host support for goal replacement.
The one-time goal switch is an external research instruction. Cached planner
paths must be discarded on switch; per-goal searches are explicitly bounded.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import statistics
import time
from .sparse_handoff_lab import fixture, overlay, distance, SEEDS, CAP
from .detour_recovery_lab import prefix_plan, GOALS
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .world import Spec, specs, digest, MOVE
from .train import sha_file

VERSION = 'retarget-handoff/1'
LAYOUTS = ('original', 'east_sleeve', 'north_ledge', 'south_ledge', 'staggered_piers')
ARMS = ('neural', 'late24', 'stall4', 'switch_plan', 'switch_stall4', 'plan_per_goal')
SOURCE = ('retarget_handoff_lab.py','sparse_handoff_lab.py','detour_recovery_lab.py',
          'goal_conditioning_lab.py','goal_observability_lab.py','world.py','policy.py','train.py')
SWITCH_TICK = 8  # external command before decision #9, not a World mutation


def select(w, goal, model, arm, tick, stall, cache, plan_goal):
    """Return selection and new cache; no more than one plan per active goal.

    cache None means no plan requested; empty tuple means a consumed plan. A
    recomputation cannot occur within the same goal stage even on failure.
    """
    if cache is not None and plan_goal != goal:
        raise AssertionError('Stale cached plan permitted across target switch')
    trigger = ((arm == 'plan_per_goal' and cache is None) or
               (arm == 'switch_plan' and tick >= SWITCH_TICK and cache is None) or
               (arm == 'switch_stall4' and tick >= SWITCH_TICK and cache is None) or
               (arm in ('stall4','switch_stall4') and stall >= 4 and cache is None) or
               (arm == 'late24' and tick >= 24 and cache is None))
    if trigger:
        cache = prefix_plan(w,goal,CAP)
        if cache is None:
            return 'STOP', None, (), goal, 1, 'planner_infeasible'
        plan_goal = goal
        source = 'planner'
        return (cache[0] if cache else 'STOP'), None, (cache[1:] if cache else ()), plan_goal, 1, source
    if cache is not None:
        if cache:
            action = cache[0]
            if action not in w.menu(12):
                return 'STOP', None, (), goal, 0, 'stale_legal_plan'
            return action, None, cache[1:], plan_goal, 0, 'planner_cached'
        # Terminal plan already consumed; no new search is authorised.
        return 'STOP', None, (), plan_goal, 0, 'plan_exhausted'
    action, probs = model.choose(w,goal,CAP,False)
    return action,probs,None,None,0,'neural'


def episode(spec, layout, goals, model_seed, arm, model):
    w = fixture(spec,5,layout)
    initial = w.state_sha(); base = w.base_sha(); prelude=list(w.decisions)
    cache=None; plan_goal=None; best=distance(w,goals[0]); stall=0; stage=0
    entries=[]; calls=0; timings=[]; switched=False; reason='decision_gate'
    for tick in range(CAP):
        if tick == SWITCH_TICK:
            stage=1; switched=True; cache=None; plan_goal=None
            best=distance(w,goals[stage]); stall=0
        goal=goals[stage]
        if w.cursor == goal: reason='goal'; break
        if w.stopped: reason='policy_stop'; break
        if w.logical_moves >= CAP: reason='logical_gate'; break
        before=w.state_sha(); menu=w.menu(12)
        t=time.perf_counter_ns()
        action, probs, cache, plan_goal, used, source=select(w,goal,model,arm,tick,stall,cache,plan_goal)
        timings.append((time.perf_counter_ns()-t)/1e6)
        assert action in menu
        cost = len(w.path)-1 if action=='RELOCATE_START' else 0 if action=='STOP' else 1
        if w.logical_moves+cost>CAP: reason='logical_gate'; break
        receipt=strict_step(w,action)
        assert receipt is not None
        calls+=used
        now=distance(w,goal)
        if now<best:best=now;stall=0
        else:stall+=1
        entries.append({'tick':tick,'stage':stage,'goal':list(goal),'pre_sha256':before,
                        'menu':menu,'action':action,'probabilities':probs,'source':source,
                        'planner_calls':used,'receipt':receipt})
    else: reason='decision_gate'
    continuation=[e['action'] for e in entries]
    row={'family':spec.family,'world_seed':spec.seed,'model_seed':model_seed,'layout':layout,
         'goals':[list(g) for g in goals],'arm':arm,'switched':switched,'arrived':w.cursor==goals[stage],
         'reason':reason,'initial_sha256':initial,'base_sha256':base,'final_sha256':w.state_sha(),
         'trace_sha256':digest(w.decisions),'decisions':len(entries),'logical_moves':w.logical_moves,
         'planner_calls':calls,'neural_actions':sum(e['source']=='neural' for e in entries),
         'planner_actions':sum(e['source'].startswith('planner') for e in entries),
         'action_counts':dict(sorted(Counter(continuation).items())),
         'material_protected_edits':len(w.ink & (set(w.materials)|w.protected)),
         'median_choose_ms':round(statistics.median(timings),6) if timings else 0,
         'total_choose_ms':round(sum(timings),6),'maximum_choose_ms':round(max(timings),6) if timings else 0}
    trace={'spec':vars(spec),'row':row,'prelude':prelude,'entries':entries}
    return row,trace


def verify_trace(trace,model):
    row=trace['row']; spec=Spec(**trace['spec']); goals=[tuple(g) for g in row['goals']]
    w=fixture(spec,5,row['layout'])
    assert w.state_sha()==row['initial_sha256'] and w.base_sha()==row['base_sha256']
    assert w.decisions==trace['prelude']
    cache=None; plan_goal=None; best=distance(w,goals[0]);stall=0;stage=0;calls=0; nvec=0
    it=iter(trace['entries']); switched=False
    for tick in range(CAP):
        if tick==SWITCH_TICK:
            switched=True; stage=1; cache=None;plan_goal=None
            best=distance(w,goals[stage]);stall=0
        goal=goals[stage]
        if w.cursor==goal:reason='goal';break
        if w.stopped:reason='policy_stop';break
        if w.logical_moves>=CAP:reason='logical_gate';break
        before=w.state_sha();menu=w.menu(12)
        action,probs,cache,plan_goal,used,source=select(w,goal,model,row['arm'],tick,stall,cache,plan_goal)
        assert action in menu
        cost=len(w.path)-1 if action=='RELOCATE_START' else 0 if action=='STOP' else 1
        if w.logical_moves+cost>CAP:reason='logical_gate';break
        e=next(it)
        assert (e['tick'],e['stage'],e['goal'],e['pre_sha256'],e['menu'])==(tick,stage,list(goal),before,menu)
        assert (e['action'],e['probabilities'],e['source'],e['planner_calls'])==(action,probs,source,used)
        assert e['receipt']==strict_step(w,action)
        calls+=used;nvec+=probs is not None
        now=distance(w,goal)
        if now<best:best=now;stall=0
        else:stall+=1
    else:reason='decision_gate'
    assert next(it,None) is None
    assert row['reason']==reason and row['arrived']==(w.cursor==goals[stage])
    assert row['switched']==switched and row['planner_calls']==calls<=2
    assert row['logical_moves']==w.logical_moves<=CAP
    assert row['trace_sha256']==digest(w.decisions) and row['final_sha256']==w.state_sha()
    assert row['material_protected_edits']==len(w.ink & (set(w.materials)|w.protected))==0
    return len(w.decisions),nvec


def summarize(rows):
    out={}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        actions=Counter()
        for r in subset: actions.update(r['action_counts'])
        out[arm]={'episodes':len(subset),'retarget_seen':sum(r['switched'] for r in subset),
                  'retarget_arrivals':sum(r['switched'] and r['arrived'] for r in subset),
                  'planner_searches':sum(r['planner_calls'] for r in subset),
                  'neural_actions':sum(r['neural_actions'] for r in subset),
                  'mean_decisions':round(statistics.mean(r['decisions'] for r in subset),3),
                  'median_episode_choice_ms':round(statistics.median(r['median_choose_ms'] for r in subset),6),
                  'mean_episode_choose_total_ms':round(statistics.mean(r['total_choose_ms'] for r in subset),6),
                  'max_single_choice_ms':round(max(r['maximum_choose_ms'] for r in subset),6),
                  'actions':dict(sorted(actions.items()))}
    return out


def run(out, checkpoints, n=1):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'traces').mkdir()
    start=time.perf_counter()
    models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    cp={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];excluded=[];receipts=0;vectors=0
    for base in specs('development',n):
        spec=Spec(base.split,base.family,base.seed,base.mode,'trap_corridor')
        for layout in LAYOUTS:
            w=fixture(spec,5,layout)
            for ordered in (GOALS,tuple(reversed(GOALS))):
                if any(prefix_plan(w,goal,CAP) is None for goal in ordered):
                    excluded.append({'spec':spec.ident,'layout':layout,'goals':[list(g) for g in ordered],
                                     'base_sha256':w.base_sha(),'reason':'initially_unreachable'})
                    continue
                for seed in SEEDS:
                    for arm in ARMS:
                        row,trace=episode(spec,layout,ordered,seed,arm,models[seed]); rows.append(row)
                        count,p=verify_trace(trace,models[seed]);receipts+=count;vectors+=p
                        f=f'{seed}-{base.family}-{base.seed}-{layout}-{ordered==GOALS}-{arm}.json'
                        (out/'traces'/f).write_text(json.dumps(trace,sort_keys=True,separators=(',',':'))+'\n')
    evidence={'version':VERSION,'split':'development','structural_heldout_opened':False,
              'native_receipts':0,'optimizer_updates':0,'synthetic_64_gate':True,
              'scheduled_switch_tick':SWITCH_TICK,'checkpoints_sha256':cp,
              'source_sha256':{name:sha_file(Path(__file__).parent/name) for name in SOURCE},
              'candidate_scenarios':len(specs('development',n))*len(LAYOUTS)*2,
              'excluded':excluded,'retained_scenarios':len(rows)//len(SEEDS)//len(ARMS),
              'episodes':len(rows),'verified_receipts':receipts,'verified_probability_vectors':vectors,
              'summary':summarize(rows),'rows':rows,
              'cpu_wall_s':round(time.perf_counter()-start,3),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'caveats':['Synthetic command changes the external goal, not native gameplay state.',
                         'All five overlay layouts share one deliberately trapped base corridor.',
                         'Initial reachability does not imply reachability after a policy paints.',
                         'Cached planner is a restricted history-prefix/BFS strategy, not complete optimal control.',
                         'A policy may STOP before the scheduled switch; exposure denominators reported.',
                         'On retarget, cached planner search state is invalidated for all arms.',
                         'Local CPU Python benchmark, not native timing or HELDOUT results.']}
    (out/'evidence.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    sums={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(sums,sort_keys=True,indent=2)+'\n')
    return evidence


def verify(out,checkpoints):
    e=json.loads((out/'evidence.json').read_text()); sums=json.loads((out/'SHA256SUMS.json').read_text())
    assert sums=={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert e['source_sha256']=={name:sha_file(Path(__file__).parent/name) for name in SOURCE}
    models={s:GoalRanker.load(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    assert e['checkpoints_sha256']=={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];receipts=vectors=0
    for f in sorted((out/'traces').glob('*.json')):
        tr=json.loads(f.read_text());row=tr['row']; rows.append(row)
        n,v=verify_trace(tr,models[row['model_seed']]);receipts+=n;vectors+=v
    assert sorted(json.dumps(r,sort_keys=True) for r in e['rows']) == sorted(json.dumps(r,sort_keys=True) for r in rows)
    assert summarize(rows)==e['summary']
    assert e['verified_receipts']==receipts and e['verified_probability_vectors']==vectors
    return e


def main():
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True)
    a.add_argument('--checkpoints',type=Path,required=True)
    a.add_argument('--worlds-per-family',type=int,default=1);a.add_argument('--verify',action='store_true')
    args=a.parse_args()
    e=verify(args.out,args.checkpoints) if args.verify else run(args.out,args.checkpoints,args.worlds_per_family)
    print(json.dumps({k:e[k] for k in ('episodes','verified_receipts','verified_probability_vectors','summary','cpu_wall_s','peak_rss_kib') if k in e},indent=2))


if __name__=='__main__': main()

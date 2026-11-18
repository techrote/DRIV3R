"""FEASIBILITY-SHIELD-001: synthetic bounded recovery with a frozen neural policy.

This explicitly measures the cost of checking an exact restricted planner at
inference time. It is *not* a native CyberSand or full-state-space guarantee.
The held-out structural families are never read.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import resource
import statistics
import time

from .detour_recovery_lab import fixture, prefix_plan, DEPTHS, GOALS
from .goal_conditioning_lab import GoalRanker
from .goal_observability_lab import strict_step
from .world import Spec, specs, digest, MOVE
from .train import sha_file

VERSION = 'feasibility-shield/1'
ARMS = ('neural', 'planner', 'late24', 'shield0', 'shield8')
SEEDS = (711, 712)
CAP = 64
SOURCE = ('feasibility_shield_lab.py', 'world.py', 'goal_conditioning_lab.py',
          'goal_observability_lab.py', 'detour_recovery_lab.py', 'policy.py', 'train.py')


def decide(w, goal, model, arm, tick, committed):
    """Return legal selection and auditable decision context, no mutation."""
    menu = w.menu(12)
    if arm == 'planner' or committed or (arm == 'late24' and tick >= 24):
        plan = prefix_plan(w, goal, CAP)
        return (plan[0] if plan else 'STOP'), {'source': 'planner', 'planner_calls': 1,
                                               'intervened': arm != 'planner', 'probabilities': None}
    action, prob = model.choose(w, goal, CAP, False)
    assert action in menu
    info = {'source': 'neural', 'planner_calls': 0, 'intervened': False,
            'proposed': action, 'probabilities': prob}
    if arm in ('shield0', 'shield8'):
        reserve = 0 if arm == 'shield0' else 8
        # A restricted-geometry feasibility certificate, recalculated for each
        # proposed action. This is an expensive and explicitly privileged oracle.
        info['planner_calls'] += 1
        current_plan = prefix_plan(w, goal, CAP)
        assert current_plan is not None, 'The canonical entry state must be solvable'
        viable = action in MOVE or action == 'BACKTRACK'
        if viable:
            after = deepcopy(w)
            after.apply(action, 12)
            remaining = () if after.cursor == goal else prefix_plan(after, goal, CAP)
            info['planner_calls'] += int(after.cursor != goal)
            viable = (remaining is not None and
                      after.logical_moves + len(remaining) + reserve <= CAP)
        if not viable:
            info.update(source='planner', intervened=True)
            action = current_plan[0] if current_plan else 'STOP'
    return action, info


def episode(sp, depth, goal, seed, arm, model):
    w = fixture(sp, depth)
    init = w.state_sha(); prelude = list(w.decisions)
    entries = []; committed = False; calls = 0; interventions = 0
    inference_ns = []; control_ns = []; stop = '64_decision_gate'
    for tick in range(CAP):
        if w.cursor == goal: stop = 'goal'; break
        if w.stopped: stop = 'policy_stop'; break
        if w.logical_moves >= CAP: stop = '64_logical_gate'; break
        ts = time.perf_counter_ns()
        action, info = decide(w, goal, model, arm, tick, committed)
        elapsed = time.perf_counter_ns() - ts
        control_ns.append(elapsed)
        if info.get('probabilities') is not None: inference_ns.append(elapsed)
        calls += info['planner_calls']
        interventions += int(info['intervened'] and not committed)
        if info['source'] == 'planner' and arm in ('late24','shield0','shield8'):
            committed = True
        assert action in w.menu(12)
        before = w.state_sha()
        if strict_step(w, action) is None: stop = '64_logical_gate'; break
        entries.append({'tick':tick, 'pre_sha256':before, 'action':action,
                        'source':info['source'], 'proposed':info.get('proposed'),
                        'probabilities':info['probabilities'],
                        'planner_calls':info['planner_calls'],
                        'intervened':info['intervened'], 'receipt':w.decisions[-1]})
    else: stop = '64_decision_gate'
    continuation = [e['receipt']['chosen'] for e in entries]
    row = {'family':sp.family, 'world_seed':sp.seed, 'depth':depth, 'goal':list(goal),
           'seed':seed, 'arm':arm, 'initial_sha256':init, 'base_sha256':w.base_sha(),
           'arrived':w.cursor==goal, 'stop':stop,
           'decisions':len(entries), 'logical_moves':w.logical_moves,
           'planner_calls':calls, 'interventions':interventions,
           'neural_selected_actions':sum(e['source']=='neural' for e in entries),
           'planner_selected_actions':sum(e['source']=='planner' for e in entries),
           'action_counts':dict(sorted(Counter(continuation).items())),
           'final_sha256':w.state_sha(), 'trace_sha256':digest(w.decisions),
           'median_controller_ms':round(statistics.median(control_ns)/1e6,6) if control_ns else 0,
           'neural_decision_count':len(inference_ns),
           'protected_material_edits':len(w.ink & (set(w.materials)|w.protected))}
    trace = {'spec':dict(split=sp.split,family=sp.family,seed=sp.seed,mode=sp.mode,challenge=sp.challenge),
             'row':row,'prelude':prelude,'entries':entries}
    return row,trace


def verify_trace(trace,model):
    row=trace['row']; sp=Spec(**trace['spec']);goal=tuple(row['goal'])
    w=fixture(sp,row['depth']); assert w.decisions==trace['prelude']
    assert w.state_sha()==row['initial_sha256'] and w.base_sha()==row['base_sha256']
    committed=False;calls=0;interventions=0
    for tick,entry in enumerate(trace['entries']):
        action,info=decide(w,goal,model,row['arm'],tick,committed)
        assert entry['tick']==tick and entry['pre_sha256']==w.state_sha()
        assert entry['action']==action and entry['source']==info['source']
        assert entry['proposed']==info.get('proposed')
        assert entry['probabilities']==info['probabilities']
        assert entry['planner_calls']==info['planner_calls']
        assert entry['intervened']==info['intervened']
        calls+=info['planner_calls'];interventions+=int(info['intervened'] and not committed)
        if info['source']=='planner' and row['arm'] in ('late24','shield0','shield8'):
            committed=True
        assert entry['receipt']==strict_step(w,action)
    assert row['final_sha256']==w.state_sha() and row['trace_sha256']==digest(w.decisions)
    assert row['arrived']==(w.cursor==goal) and row['planner_calls']==calls
    assert row['interventions']==interventions and row['decisions']==len(trace['entries'])
    assert row['protected_material_edits']==len(w.ink & (set(w.materials)|w.protected))==0
    assert w.logical_moves <= CAP
    if row['stop']=='goal':assert w.cursor==goal
    elif row['stop']=='policy_stop':assert w.stopped
    elif row['stop']=='64_logical_gate':assert w.logical_moves>=CAP
    elif row['stop']=='64_decision_gate':assert len(trace['entries'])==CAP
    else: raise AssertionError('Invalid stop reason')
    return len(w.decisions), sum(e['probabilities'] is not None for e in trace['entries'])


def summary(rows):
    out={}
    for arm in ARMS:
        data=[r for r in rows if r['arm']==arm]
        total=len(data)
        out[arm]={'episodes':total,'arrivals':sum(r['arrived'] for r in data),
                  'mean_actions':round(statistics.mean(r['decisions'] for r in data),3),
                  'planner_calls':sum(r['planner_calls'] for r in data),
                  'planner_actions':sum(r['planner_selected_actions'] for r in data),
                  'neural_actions':sum(r['neural_selected_actions'] for r in data),
                  'interventions':sum(r['interventions'] for r in data),
                  'mean_episode_median_control_ms':round(statistics.mean(r['median_controller_ms'] for r in data),5),
                  'actions':dict(sorted(Counter({k:sum(r['action_counts'].get(k,0) for r in data)
                    for k in sorted({x for r in data for x in r['action_counts']})}).items()))}
    return out


def run(out, checkpoints, num_per_family=1):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'traces').mkdir()
    t=time.perf_counter();rows=[];rec=vec=0
    cp={seed:checkpoints/f'{seed}-conditioned.npz' for seed in SEEDS}
    models={seed:GoalRanker.load(cp[seed]) for seed in SEEDS}
    hashes={str(seed):sha_file(cp[seed]) for seed in SEEDS}
    for seed in SEEDS:
        assert models[seed].updates==120
        for original in specs('development',num_per_family):
            sp=Spec(original.split,original.family,original.seed,original.mode,'trap_corridor')
            for depth in DEPTHS:
                for gi,goal in enumerate(GOALS):
                    assert prefix_plan(fixture(sp,depth),goal,CAP)
                    for arm in ARMS:
                        row,trace=episode(sp,depth,goal,seed,arm,models[seed])
                        n,p=verify_trace(trace,models[seed]);rec+=n;vec+=p
                        rows.append(row)
                        fname=f'{seed}-{sp.family}-{sp.seed}-{depth}-{gi}-{arm}.json'
                        (out/'traces'/fname).write_text(json.dumps(trace,sort_keys=True,separators=(',',':'))+'\n')
    evidence={'version':VERSION,'split':'development','heldout_opened':False,
              'fixture':'trap_corridor','synthetic_64_gate':True,'native_receipts':0,
              'checkpoint_sha256':hashes,'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in SOURCE},
              'new_optimizer_steps':0,'episodes':len(rows),'verified_receipts':rec,
              'verified_neural_distributions':vec,'summary':summary(rows),
              'per_episode':rows,'cpu_wall_s':round(time.perf_counter()-t,3),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations':['All challenge worlds share a purpose-built synthetic corridor.',
                  'Oracle has full privileged synthetic map and history; not a native policy ABI.',
                  'Prefix search is complete only for monotone history retreat then empty-space painting.',
                  'No native gameplay; no structural HELDOUT opened.',
                  'Shield uses costly repeated oracle calls: intervention count is NOT compute cost.']}
    (out/'evidence.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
           for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return evidence


def verify(out,checkpoints):
    ev=json.loads((out/'evidence.json').read_text())
    hashes=json.loads((out/'SHA256SUMS.json').read_text())
    assert hashes=={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*'))
                    if p.is_file() and p.name!='SHA256SUMS.json'}
    assert ev['source_sha256']=={n:sha_file(Path(__file__).parent/n) for n in SOURCE}
    models={seed:GoalRanker.load(checkpoints/f'{seed}-conditioned.npz') for seed in SEEDS}
    assert ev['checkpoint_sha256']=={str(s):sha_file(checkpoints/f'{s}-conditioned.npz') for s in SEEDS}
    rows=[];receipts=vectors=0
    for path in sorted((out/'traces').glob('*.json')):
        tr=json.loads(path.read_text());r=tr['row']
        n,p=verify_trace(tr,models[r['seed']]);receipts+=n;vectors+=p;rows.append(r)
    assert len(rows)==ev['episodes'] and summary(rows)==ev['summary']
    assert receipts==ev['verified_receipts'] and vectors==ev['verified_neural_distributions']
    return {'episodes':len(rows),'receipts':receipts,'vectors':vectors,'artifact_files':len(hashes)}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path)
    ap.add_argument('--verify',type=Path)
    ap.add_argument('--checkpoints',type=Path,required=True)
    ap.add_argument('--per-family',type=int,default=1)
    a=ap.parse_args()
    print(json.dumps(verify(a.verify,a.checkpoints) if a.verify else run(a.out,a.checkpoints,a.per_family),sort_keys=True))

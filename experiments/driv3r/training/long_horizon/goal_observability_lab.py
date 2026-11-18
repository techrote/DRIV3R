"""GOAL-OBSERVABILITY-001: goal changes and host-bounded gameplay audit.

Read-only evaluation of SHA-pinned existing tiny Ranker checkpoints. World menus and
materials never change; objective is intentionally *not* in Ranker features.
A strict external 64-logical-move gate ends synthetic execution before overshoot.
This is NOT a native CyberSand host, acceptance, or a new training checkpoint.
"""
from __future__ import annotations
import argparse, hashlib, json, random, resource, statistics, time
from collections import Counter, deque
from pathlib import Path
from .world import World, MOVE, specs, replay, digest
from .policy import Ranker, teacher, random_legal_persistent, FEATURES
from .train import sha_file

VERSION='goal-observability/1'
NATIVE_ANALOG_CAP=64  # external analytical gate; original World is unchanged
MAX_RECOVERIES=12
POLICY_SEEDS=(1051,1052)
SOURCE_FILES=('goal_observability_lab.py','world.py','policy.py','train.py')


def strict_step(w,action):
    """Never execute a synthetic action that would exceed the external move cap."""
    if action not in w.menu(MAX_RECOVERIES):
        raise ValueError('not on original World legal menu')
    cost=(len(w.path)-1 if action=='RELOCATE_START' else
          1 if action in MOVE or action=='BACKTRACK' else 0)
    if w.logical_moves+cost>NATIVE_ANALOG_CAP:
        return None
    receipt=w.apply(action,MAX_RECOVERIES)
    assert w.logical_moves<=NATIVE_ANALOG_CAP
    return receipt


def source_route(sp,n_steps=44):
    """Fixed, goal-independent reference waypoints painted by legal moves only."""
    w=World.create(sp)
    rng=random.Random(sp.seed*451+17)
    points=[]
    last='MOVE_E'
    for t in range(n_steps):
        menu=w.menu(MAX_RECOVERIES)
        moves=[a for a in menu if a in MOVE]
        if not moves: break
        # Curving exploratory walks, not a straight eastward target or free teleport.
        def score(a):
            dx,dy=MOVE[a]
            nextp=(w.cursor[0]+dx,w.cursor[1]+dy)
            ahead=sum(not w.occupied((nextp[0]+dx*k,nextp[1]+dy*k)) for k in (1,2,3))
            return .32*a.__eq__(last) + .24*a.__ne__(last) + .24*a.endswith('NESW'[(t//7)%4]) + .10*a.endswith('NESW'[(t//11+2)%4])+ .16*ahead + rng.random()*.18
        chosen=max(moves,key=score)
        strict_step(w,chosen)
        last=chosen
        if (t+1)%4==0:points.append(w.cursor)
    assert len(points)>=4, ('route too short',sp.ident,len(points))
    return points,w


def conflict_states(sp):
    """Exactly identical World and legal menu, two different one-step goals."""
    w=World.create(sp)
    rng=random.Random(sp.seed*331+9)
    result=[]
    for t in range(34):
        moves=[a for a in w.menu(MAX_RECOVERIES) if a in MOVE]
        if len(moves)>=2 and (t%3==0):
            a,b=sorted(moves)[:2]
            goals=[]
            for move in (a,b):
                dx,dy=MOVE[move]
                goals.append((w.cursor[0]+dx,w.cursor[1]+dy))
            result.append((w,tuple(goals),tuple((a,b))))
            # Retain the *same* current state by creating an actual deep copy.
            import copy
            w=copy.deepcopy(w)
        action,*_=teacher(w,64,rng)
        if action=='STOP':break
        if not strict_step(w,action): break
    return result


def shortest_path_action(w,goal):
    """A bounded deterministic observable-goal oracle; no illegal editing.

    BFS considers only currently empty cells plus the cursor. BACKTRACK is
    possible when the target lies on the owned stack or no empty route exists.
    The oracle does not use synthetic RELOCATE_START.
    """
    legal=w.menu(MAX_RECOVERIES)
    if w.cursor==goal:return 'STOP'
    if goal in w.path:
        return 'BACKTRACK' if 'BACKTRACK' in legal else 'STOP'
    if w.occupied(goal):return 'BACKTRACK' if 'BACKTRACK' in legal else 'STOP'
    start=w.cursor
    queue=deque([start]); seen={start}; predecessor={}
    found=False
    while queue:
        point=queue.popleft()
        if point==goal:found=True;break
        for a,(dx,dy) in MOVE.items():
            nxt=(point[0]+dx,point[1]+dy)
            if nxt not in seen and (nxt==goal or not w.occupied(nxt)):
                seen.add(nxt);predecessor[nxt]=(point,a);queue.append(nxt)
    if found:
        pos=goal
        while predecessor[pos][0]!=start:
            pos=predecessor[pos][0]
        action=predecessor[pos][1]
        if action in legal:return action
    return 'BACKTRACK' if 'BACKTRACK' in legal else 'STOP'


def run_episode(sp,waypoints,arm,checkpoint=None,perturb=False):
    w=World.create(sp)
    rng=random.Random(sp.seed*481+(1 if perturb else 0)*103)
    model=Ranker.restore(checkpoint) if checkpoint is not None else None
    active_goal=0; attained=[]; action_counts=Counter(); probabilities=[]; latencies=[]; overrides=[]
    stopped_by='64_logical_gate'
    for tick in range(128):
        if active_goal>=len(waypoints):stopped_by='all_goals';break
        if w.stopped:stopped_by='policy_stop';break
        if w.logical_moves>=NATIVE_ANALOG_CAP:stopped_by='64_logical_gate';break
        goal=waypoints[active_goal]
        if w.cursor==goal:
            attained.append({'goal_index':active_goal,'tick':tick,'logical_moves':w.logical_moves})
            active_goal+=1
            continue
        start=time.perf_counter_ns()
        if arm=='goal_oracle':action=shortest_path_action(w,goal);p=None
        elif arm=='random_persistent':action=random_legal_persistent(w,rng);p=None
        else: action,p=model.choose(w,128)
        latencies.append((time.perf_counter_ns()-start)/1e6)
        original=action
        if perturb and tick in (7,22):
            alternative=sorted(a for a in w.menu() if a in MOVE and a!=action)
            if alternative:action=alternative[-1]
            overrides.append({'step':tick,'original':original,'actual':action,'changed':action!=original})
        if p is not None:
            probabilities.append({'step':tick,'goal_index':active_goal,'goal':list(goal),
                                  'pre_sha256':w.state_sha(),'menu':w.menu(),'p':p,'chosen':original})
        receipt=strict_step(w,action)
        if receipt is None:stopped_by='64_logical_gate';break
        action_counts[action]+=1
        if action=='STOP':stopped_by='policy_stop';break
        if w.cursor==goal:
            attained.append({'goal_index':active_goal,'tick':tick+1,'logical_moves':w.logical_moves})
            active_goal+=1
    else:stopped_by='128_decision_safety_gate'
    # replay uses original World rules, no hidden modification to menu or moves.
    assert replay(sp,w.decisions,MAX_RECOVERIES).state_sha()==w.state_sha()
    summary={'spec':sp.ident,'family':sp.family,'challenge':sp.challenge,'arm':arm,
             'perturb':perturb,'goals_total':len(waypoints),'goals_reached':len(attained),
             'decisions':len(w.decisions),'logical_moves':w.logical_moves,
             'recovery_count':w.recoveries,'painted':len(w.ink),'stop_reason':stopped_by,
             'synthetic_relocations':action_counts['RELOCATE_START'],
             'protected_or_material_edits':len(w.ink&(set(w.materials)|w.protected)),
             'action_counts':dict(sorted(action_counts.items())),
             'p50_choice_ms':round(statistics.median(latencies),6) if latencies else None,
             'trace_sha256':digest(w.decisions),'final_state_sha256':w.state_sha()}
    trace={'summary':summary,'waypoints':[list(g) for g in waypoints],
           'attained':attained,'events':w.decisions,'probabilities':probabilities,
           'overrides':overrides,'world':{'split':sp.split,'family':sp.family,'seed':sp.seed,
                                      'mode':sp.mode,'challenge':sp.challenge}}
    return summary,trace


def run(out_dir,checkpoint_dir):
    out=Path(out_dir)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.perf_counter()
    cps={}
    for seed in POLICY_SEEDS:
        f=Path(checkpoint_dir)/f'{seed}-128.npz'
        if not f.exists():raise FileNotFoundError(f)
        model=Ranker.restore(f)
        assert model.updates>0
        cps[str(seed)]={'sha256':sha_file(f),'parameter_count':model.parameter_count,
                         'optimizer_updates':model.updates}
    assert not any('goal' in feat or 'target' in feat for feat in FEATURES)
    spec_list=specs('development',3)
    # Paired objective intervention, identical state and legal actions.
    interventions=[]
    for sp in spec_list:
        for w,goals,actions in conflict_states(sp):
            state_sha=w.state_sha();legal=w.menu()
            for seed in POLICY_SEEDS:
                model=Ranker.restore(Path(checkpoint_dir)/f'{seed}-128.npz')
                chosen,p=model.choose(w,128)
                correct=sum(chosen==a for a in actions)
                assert correct<=1
                interventions.append({'spec':sp.ident,'seed':seed,'world_sha256':state_sha,
                    'goals':[list(x) for x in goals],'optimal_actions':actions,
                    'legal':legal,'goal_blind_choice':chosen,'correct_across_pair':correct,
                    'action_probabilities':p})
    assert interventions
    rows=[];traces=[];source_route_summaries=[]
    for sp in spec_list:
        waypoints,reference_world=source_route(sp)
        source_route_summaries.append({'spec':sp.ident,'goals':len(waypoints),
                  'route_sha256':digest([list(x) for x in waypoints]),
                  'base_sha256':reference_world.base_sha()})
        for perturbed in (False,True):
            for arm in ('goal_oracle','random_persistent','neural_1051','neural_1052'):
                checkpoint=Path(checkpoint_dir)/f'{arm[-4:]}-128.npz' if arm.startswith('neural_') else None
                row,trace=run_episode(sp,waypoints,arm,checkpoint,perturbed)
                rows.append(row);traces.append(trace)
    assert len(rows)==12*2*4
    (out/'traces.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    (out/'paired_objectives.json').write_text(json.dumps(interventions,sort_keys=True,separators=(',',':'))+'\n')
    groups={}
    for arm in ('goal_oracle','random_persistent','neural_1051','neural_1052'):
        for p in (False,True):
            rs=[r for r in rows if r['arm']==arm and r['perturb']==p]
            groups[arm+'/'+('perturbed' if p else 'ordinary')]={
                'episodes':len(rs),'goals_reached':sum(r['goals_reached'] for r in rs),
                'goals_offered':sum(r['goals_total'] for r in rs),
                'all_goals':sum(r['goals_reached']==r['goals_total'] for r in rs),
                'host_64_gate':sum(r['stop_reason']=='64_logical_gate' for r in rs),
                'policy_stop':sum(r['stop_reason']=='policy_stop' for r in rs),
                'mean_decisions':round(statistics.mean(r['decisions'] for r in rs),2),
                'mean_logical_moves':round(statistics.mean(r['logical_moves'] for r in rs),2),
                'synthetic_relocations':sum(r['synthetic_relocations'] for r in rs)}
    all_correct=sum(x['correct_across_pair'] for x in interventions)
    evidence={'experiment':'GOAL-OBSERVABILITY-001','version':VERSION,
              'source_ref_before_edit':'2972d20ec4d7fa51adb749e81350de48a8c398ec',
              'source_sha256':{f:sha_file(Path(__file__).parent/f) for f in SOURCE_FILES},
              'checkpoint_identities':cps,'source_route_summaries':source_route_summaries,
              'split':'development','structural_heldout_opened':False,'training_updates':0,
              'native_executed':False,'synthetic_host_64_gate_only':True,
              'paired_objective_states':len(interventions),'paired_objective_conditions':2*len(interventions),
              'paired_goal_blind_correct':all_correct,
              'paired_goal_blind_accuracy':round(all_correct/(2*len(interventions)),5),
              'paired_objectives_sha256':sha_file(out/'paired_objectives.json'),
              'episodes':len(rows),'executed_actions':sum(r['decisions'] for r in rows),
              'original_trace_sha256':sha_file(out/'traces.json'),
              'per_episode':rows,'groups':groups,'wall_s':round(time.perf_counter()-started,3),
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'caveats':['World is synthetic; external gate is not native host execution.',
                'Source teacher route goals are generated from legal moves and can be infeasible after off-route errors.',
                'This is a zero-update diagnostic, not evidence of learning new goal-conditioned behavior.',
                'Ranker features omit objective, so same-state goal conflict is fundamentally unobservable.',
                'No final structural HELDOUT or native game acceptance; no claims for SLMs.']}
    (out/'evidence.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in ('paired_objective_states','paired_goal_blind_accuracy','episodes','executed_actions','wall_s','peak_rss_kib','groups')},indent=2))
    return evidence

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--checkpoints',required=True)
    args=ap.parse_args();run(args.out,args.checkpoints)

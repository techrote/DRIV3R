"""INK-ALIAS-MEMORY-001: trained memory probe on 36/72/128-action ink aliases.

These deliberately engineered, source-World synthetic states are NOT native
CyberSand and exceed its historical 64-logical-move allowance at long depths.
Never access structural HELDOUT. Route-following after probe decision is a
scripted planner control, NOT an autonomous trained policy.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import resource
import statistics
import time
from pathlib import Path
import numpy as np
from .world import Spec, World, MOVE, digest
from .detour_recovery_lab import shortest_empty_path
from .goal_conditioning_lab import GoalRanker
from .train import sha_file

VERSION = 'ink-alias-memory/1'
GOAL = (48, 44)
DEPTHS = (36, 72, 128)
FAMILIES = ('pickets', 'ridges', 'islands', 'zigzag')
SIDES = ('E', 'W')
COMMON = ('MOVE_N',)*22 + ('MOVE_E',)*30 + ('MOVE_S',)*60
ACTIONS = ('MOVE_W', 'MOVE_E')
CHECKPOINT_SEEDS = (711, 712)


def script(side: str, depth: int):
    if side not in SIDES or depth not in DEPTHS:
        raise ValueError('unapproved synthetic side or horizon')
    other = 'W' if side == 'E' else 'E'
    return ((f'MOVE_{side}',)*2 + ('MOVE_N',)*7 + (f'MOVE_{other}',)*2 +
            ('MOVE_N',)*5 + COMMON[:depth - 23] + ('RELOCATE_START',) + ('MOVE_S',)*6)


def world(spec: Spec, depth: int):
    if spec.split not in ('train','development') or spec.challenge != 'normal':
        raise ValueError('structural HELDOUT and corridor challenges excluded')
    w = World.create(spec)
    # A fixed common test arena is opened BEFORE the pair's histories execute;
    # no policy outcome or label can select which cells are cleared.
    cleared = {(x,y) for x in range(44, 53) for y in range(34, 56)}
    for side in SIDES:
        at = (48,48)
        for action in script(side,depth):
            if action == 'RELOCATE_START':
                at = (48,48)
            else:
                dx,dy = MOVE[action]
                at = (at[0]+dx,at[1]+dy)
                cleared.add(at)
    for p in cleared:
        w.materials.pop(p,None)
        w.protected.discard(p)
    return w


def state(spec: Spec, depth: int, side: str):
    w = world(spec, depth)
    for action in script(side,depth):
        if action not in w.menu(12):
            raise AssertionError(('non-legal prelude action', action, w.cursor))
        w.apply(action,12)
    assert len(w.decisions) == depth and w.cursor == (48,54)
    return w


def limited_observation(w):
    """Explicitly limited view; deliberately excludes distant ink/custody hashes."""
    x,y = w.cursor
    window = tuple((dx,dy, int(w.occupied((x+dx,y+dy))),
                    int((x+dx,y+dy) in w.materials),
                    int((x+dx,y+dy) in w.protected))
                   for dy in range(-2,3) for dx in range(-2,3))
    return (w.cursor, w.heading, tuple(w.menu(12)), len(w.ink),
            tuple(d['chosen'] for d in w.decisions[-8:]), GOAL, window)


def history_features(w):
    # Observable prior ink on two *distant*, originally empty branch cells;
    # not hard-coded ground-truth side labels.
    return np.asarray([int((50,48) in w.ink), int((46,48) in w.ink), 1.],dtype=np.float64)


def train_probe(rows, updates=240):
    X=np.asarray([history_features(w) for w,label in rows])
    y=np.asarray([label for w,label in rows],dtype=np.int64)
    weights=np.zeros((3,2),dtype=np.float64);m=np.zeros_like(weights);v=np.zeros_like(weights)
    losses=[]
    for t in range(1,updates+1):
        z=X@weights;z-=z.max(axis=1,keepdims=True)
        exp=np.exp(z);p=exp/exp.sum(axis=1,keepdims=True)
        losses.append(float(-np.log(np.maximum(1e-30,p[np.arange(len(y)),y])).mean()))
        delta=p.copy();delta[np.arange(len(y)),y]-=1
        grad=X.T@delta/len(y)
        m=.9*m+.1*grad;v=.999*v+.001*grad*grad
        weights-=.04*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
    return weights, {'updates':updates,'train_rows':len(y),'loss_start':round(losses[0],7),
                     'loss_end':round(losses[-1],7)}


def probe(w, weights):
    x=history_features(w);z=x@weights;z-=z.max()
    p=np.exp(z);p/=p.sum()
    return ACTIONS[int(p.argmax())], {a:round(float(v),8) for a,v in zip(ACTIONS,p)}


def execute_suffix(w, proposed):
    """One model decision followed by explicitly controlled empty-cell BFS."""
    if proposed not in w.menu(12):
        raise AssertionError('proposed action not offered by host')
    w.apply(proposed,12)
    if w.stopped: return False,0
    path=shortest_empty_path(w,w.cursor,GOAL)
    if path is None: return False,0
    for a in path:
        w.apply(a,12)
    return w.cursor==GOAL,len(path)


def make_cases(split, n):
    rows=[]
    base=1000 if split=='train' else 3000
    for i,fam in enumerate(FAMILIES):
        for j in range(n):
            sp=Spec(split,fam,base+i*100+j,('wall','route','motif')[(i+j)%3])
            for depth in DEPTHS:
                pair=[]
                for side in SIDES:
                    w=state(sp,depth,side)
                    plan=shortest_empty_path(w,w.cursor,GOAL)
                    assert plan is not None and plan[0] in ACTIONS
                    label=ACTIONS.index(plan[0]);pair.append((w,label,side))
                assert pair[0][1] != pair[1][1]
                assert limited_observation(pair[0][0])==limited_observation(pair[1][0])
                rows.extend((sp,depth,side,w,label) for w,label,side in pair)
    return rows


def replay_receipt(rec):
    sp=Spec(**rec['spec']);w=world(sp,rec['depth'])
    for ev in rec['events']:
        assert ev['pre_sha256'] == w.state_sha()
        assert ev['legal']==w.menu(12) and ev['chosen'] in w.menu(12)
        assert w.apply(ev['chosen'],12)==ev
    assert w.state_sha()==rec['final_state_sha256']
    assert digest(w.decisions)==rec['trace_sha256']
    assert len(w.ink & (set(w.materials)|w.protected))==0
    return len(rec['events'])


def run(output: Path, check_dir: Path):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);start=time.perf_counter()
    train=make_cases('train',3)
    learned,fit=train_probe([(w,label) for _,_,_,w,label in train])
    cases=make_cases('development',2)
    models={s:GoalRanker.load(check_dir/f'{s}-conditioned.npz') for s in CHECKPOINT_SEEDS}
    for s,m in models.items():assert m.updates == 120
    counts=Counter();answers=[];records=[];latency={a:[] for a in ['memory','limited','frozen711','frozen712','oracle']}
    for sp,depth,side,w,label in cases:
        expected=ACTIONS[label]
        options={}
        ts=time.perf_counter_ns();options['memory']=probe(w,learned);latency['memory'].append((time.perf_counter_ns()-ts)/1e6)
        ts=time.perf_counter_ns();options['limited']=(ACTIONS[0],{a:0.5 for a in ACTIONS});latency['limited'].append((time.perf_counter_ns()-ts)/1e6)
        ts=time.perf_counter_ns();options['oracle']=(expected,{expected:1.0});latency['oracle'].append((time.perf_counter_ns()-ts)/1e6)
        for s,m in models.items():
            ts=time.perf_counter_ns();options[f'frozen{s}']=m.choose(w,GOAL,256,False)
            latency[f'frozen{s}'].append((time.perf_counter_ns()-ts)/1e6)
        for arm,(action,probs) in options.items():
            trial=copy.deepcopy(w)
            success,suffix=execute_suffix(trial,action)
            prefix=w.decisions
            rec={'spec':{'split':sp.split,'family':sp.family,'seed':sp.seed,'mode':sp.mode,'challenge':sp.challenge},
                 'depth':depth,'side':side,'arm':arm,'expected_first_action':expected,'chosen':action,
                 'probs':probs,'first_optimal':action==expected,'arrived_after_scripted_suffix':success,
                 'planner_suffix_steps':suffix,'prelude_sha256':digest(prefix),
                 'state_sha256':w.state_sha(),'final_state_sha256':trial.state_sha(),
                 'trace_sha256':digest(trial.decisions),'events':trial.decisions}
            assert replay_receipt(rec)==len(trial.decisions)
            counts[(arm,'optimal')]+=int(action==expected)
            counts[(arm,'arrivals')]+=int(success)
            counts[(arm,'actions')]+=len(trial.decisions)
            counts[(arm,'suffix')]+=suffix
            counts[(arm,'n')]+=1
            answers.append({k:v for k,v in rec.items() if k!='events'})
            records.append(rec)
    cp=output/'history_probe.npz';np.savez_compressed(cp,weights=learned,updates=fit['updates'])
    recfile=output/'receipts.jsonl.gz'
    with recfile.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as z:
            for rec in records:z.write((json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n').encode())
    summary={arm:{'episodes':counts[(arm,'n')],'optimal_first_actions':counts[(arm,'optimal')],
                  'scripted_suffix_arrivals':counts[(arm,'arrivals')],
                  'executed_actions_including_prelude':counts[(arm,'actions')],
                  'planned_suffix_steps':counts[(arm,'suffix')],
                  'median_choice_ms':round(statistics.median(latency[arm]),6),
                  'p95_choice_ms':round(float(np.percentile(latency[arm],95)),6),
                  'first_action_counts':dict(sorted(Counter(x['chosen'] for x in answers if x['arm']==arm).items()))}
             for arm in ['memory','limited','frozen711','frozen712','oracle']}
    data={'version':VERSION,'split':'TRAIN learning; DEVELOPMENT evaluation; structural HELDOUT unopened',
          'fit':fit,'train_scenarios':len(train),'development_scenarios':len(cases),
          'paired_observation_aliases':len(cases)//2,'depths':list(DEPTHS),
          'synthetic_logical_moves_at_prelude':{'36':64,'72':136,'128':248},
          'checkpoint_sha256':{str(s):sha_file(check_dir/f'{s}-conditioned.npz') for s in CHECKPOINT_SEEDS},
          'trained_probe_checkpoint_sha256':sha_file(cp),'receipts_gzip_sha256':sha_file(recfile),
          'total_replayed_actions':sum(counts[(a,'actions')] for a in summary),
          'controllers':summary,'results':answers,
          'runtime_seconds':round(time.perf_counter()-start,4),
          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'native_receipts':0,'paid_gpu_or_api':False,'optimizer_updates':fit['updates'],
          'caveats':['common engineered cleared arena with unchanged source World legality',
                     'restricted 2-bit full-history probe, not a general model or SLM',
                     'scripted BFS continuation after first model decision',
                     'synthetic 72/128 lengths exceed historical native logical-move cap',
                     'no structural heldout, no native CyberSand, no immutable Actions artifact']}
    (output/'summary.json').write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return data,records


def verify_archive(path):
    ct=0;actions=0
    with gzip.open(path,'rt') as z:
        for line in z:
            rec=json.loads(line);actions+=replay_receipt(rec);ct+=1
    return ct,actions


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,required=True)
    a=p.parse_args()
    result,_=run(a.out,a.checkpoints)
    count,actions=verify_archive(a.out/'receipts.jsonl.gz')
    assert count==len(result['results']) and actions==result['total_replayed_actions']
    print(json.dumps({'result':result['controllers'],'replay_episodes':count,
                      'replay_actions':actions,'seconds':result['runtime_seconds'],
                      'rss_kib':result['peak_rss_kib']},indent=2))
if __name__=='__main__':main()

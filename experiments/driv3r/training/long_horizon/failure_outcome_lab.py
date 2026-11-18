"""FAILURE-OUTCOME-001: empirical student-failure suffix supervision.

TRAIN-only counterfactual student continuations; autonomous DEVELOPMENT evaluation.
Synthetic DRIV3R host fixture only: not the CyberSand native ABI.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import goal_recovery_lab as host, goal_retarget_lab as fixture, policy
from .world import World, Spec, TRAIN_FAMILIES, MODES, MOVE, specs
from .train import sha_file

VERSION='failure-outcome/1'
ARMS=('frozen','imitation','survival_guard','outcome_aggressive')
STAGES=(36,72,128)
# Explicit DEVELOPMENT worlds, same known topologies, separate from TRAIN and previous DEV seeds.
DEV=tuple(Spec('development',family,3058+100*i,MODES[i%len(MODES)])
          for i,family in enumerate(TRAIN_FAMILIES))
EVAL_SEED=709
SOURCE=('failure_outcome_lab.py','goal_recovery_lab.py','goal_retarget_lab.py',
        'goal_conditioning_lab.py','world.py','policy.py','train.py')

def cases():
    return list(DEV)+[replace(s,challenge='trap_corridor') for s in DEV]

def student_suffix(w, first, targets, t, horizon, frozen, depth=24):
    """Actual host-legal counterfactual execution by frozen STUDENT, not a heuristic teacher.

    Frozen continuation sees only the current phase goal as time progresses.
    Stores the terminal state and causal path; never labels from DEVELOPMENT.
    """
    if first not in w.menu(12) or not 1<=depth<=32: raise ValueError('illegal/oversized suffix')
    sim=copy.deepcopy(w); before=len(sim.ink); rec=sim.recoveries; reached=set(); moves=[]
    for j in range(depth):
        if sim.stopped: break
        step=t+j; phase=fixture.active_phase(step,'retarget')
        target=targets[phase]
        action=first if j==0 else frozen.choose(sim,target,horizon,False)[0]
        if action not in sim.menu(12): raise ValueError('frozen requested nonlegal action')
        event=sim.apply(action,12); moves.append(action)
        if list(target) in event['painted']: reached.add(phase)
    return {'completed':len(moves),'early_stop':bool(sim.stopped and len(moves)<depth),
            'goals':len(reached),'paint':len(sim.ink)-before,
            'recovery':sim.recoveries-rec,'actions':moves,
            'last_sha256':sim.state_sha()}

def outcome_labels(menu, imitation, outcome):
    """Predeclared strict survival improvement; aggressive arm chooses outcome rank.

    No scalar reward tuning. The guard only changes imitation when an observed candidate Pareto-dominates
    on completed decisions, goal contacts and unique paint.
    """
    if not menu or imitation not in menu or set(outcome)!=set(menu): raise ValueError('menu mismatch')
    b=outcome[imitation]
    eligible=[a for a in menu if outcome[a]['completed']>=b['completed']
              and outcome[a]['goals']>=b['goals'] and outcome[a]['paint']>=b['paint']
              and (outcome[a]['completed']>b['completed'] or outcome[a]['goals']>b['goals']
                   or outcome[a]['paint']>b['paint'])]
    guard=max(eligible,key=lambda a:(outcome[a]['completed'],outcome[a]['goals'],
           outcome[a]['paint'],-outcome[a]['recovery'],-menu.index(a))) if eligible else imitation
    # Explicitly diagnostic: aggressive long-horizon lexicographic outcome objective.
    best=max(menu,key=lambda a:(outcome[a]['goals'],outcome[a]['completed'],
            outcome[a]['paint'],-outcome[a]['recovery'],int(a==imitation),-menu.index(a)))
    return guard,best

def corpus(frozen,seed,stage,n=48,depth=24):
    if stage not in STAGES or not 24<=n<=128 or depth>32:raise ValueError('bounded corpus')
    reservoir=random.Random(seed*991+stage*37);seen=0;samples=[];forced=0
    for sp in specs('train',2)+[replace(s,challenge='trap_corridor') for s in specs('train',2) if s.seed%2==0]:
        for variant in (0,1):
            w=World.create(sp);targets=fixture.targets_for(sp,seed+variant*41,'retarget')
            perturb=random.Random(seed*713+sp.seed*13+variant*31+stage)
            for t in range(stage):
                if w.stopped:break
                snap=(copy.deepcopy(w),targets,t,sp.ident,variant)
                seen+=1
                if len(samples)<n:samples.append(snap)
                else:
                    j=reservoir.randrange(seen)
                    if j<n:samples[j]=snap
                asked=frozen.choose(w,targets[fixture.active_phase(t,'retarget')],stage,False)[0]
                actual=asked
                if t in (19,57,91,119):
                    candidates=[a for a in w.menu(12) if a in MOVE and a!=asked]
                    if candidates:actual=perturb.choice(candidates);forced+=1
                w.apply(actual,12)
    X=[];M=[];Y={a:[] for a in ARMS[1:]};audit=[];changes=Counter();opportunities=0
    for i,(w,targets,t,ident,variant) in enumerate(samples):
        target=targets[fixture.active_phase(t,'retarget')]
        menu,x=host.matrix(w,target,stage,False)
        imitation=host.label(w,target,stage,random.Random(seed*30011+t*173+w.spec.seed+i))
        if imitation not in menu: raise ValueError('nonlegal teacher')
        outcome={a:student_suffix(w,a,targets,t,stage,frozen,depth) for a in menu}
        guard,aggressive=outcome_labels(menu,imitation,outcome)
        opportunities+=int(any(v['completed']>outcome[imitation]['completed'] for v in outcome.values()))
        changes['guard']+=int(guard!=imitation);changes['aggressive']+=int(aggressive!=imitation)
        X.append(x);m=np.zeros(policy.ACTION_MAX,bool);m[:len(menu)]=True;M.append(m)
        for arm,a in [('imitation',imitation),('survival_guard',guard),('outcome_aggressive',aggressive)]:
            Y[arm].append(menu.index(a))
        audit.append({'world':ident,'variant':variant,'t':t,'state_sha256':w.state_sha(),
                      'goal':list(target),'menu':menu,'teacher':imitation,
                      'guard':guard,'aggressive':aggressive,'outcomes':outcome})
    return np.stack(X),np.stack(M),{a:np.array(y,np.int64) for a,y in Y.items()},audit,{
        'split':'train','stage':stage,'student_visited':seen,'samples':len(samples),
        'forced_errors':forced,'guard_relabeled':changes['guard'],
        'aggressive_relabeled':changes['aggressive'],
        'survival_opportunities':opportunities}

def train(seed,base_updates=60,adapt_updates=20,n=48,depth=24):
    if not 1<=base_updates<=100 or not 1<=adapt_updates<=60:raise ValueError('budget')
    base,hist=host.train(seed,base_updates);frozen=base['goal_recovery']
    models={a:copy.deepcopy(frozen) for a in ARMS}
    stages=[];ledgers=[]
    for stage in STAGES:
        X,M,Y,audit,meta=corpus(frozen,seed,stage,n,depth)
        updates={a:models[a].fit(X,M,Y[a],adapt_updates,seed*313+stage*7)
                 for a in ARMS[1:]}
        stages.append({'stage':stage,'meta':meta,'optimizer':updates})
        ledgers.append({'stage':stage,'state_outcomes':audit})
    return models,{'base':hist,'adaptation':stages},ledgers

def evaluate(models,quick=False,png_dir=None):
    eps=[]
    selected=cases()[:1] if quick else cases()
    for sp in selected:
        for horizon in ((128,) if quick else (128,160)):
            for arm in ARMS:
                png=png_dir/f'{sp.family}-{arm}.png' if png_dir is not None and horizon==160 and sp in DEV else None
                ep=fixture.run_episode(sp,{'goal_recovery':models[arm]},'goal_recovery',EVAL_SEED,horizon,'retarget',png)
                fixture.verify_episode(ep,sp,{'goal_recovery':models[arm]})
                ep['trained_arm']=arm;eps.append(ep)
    return eps

def summary(eps):
    result={}
    for arm in ARMS:
        rows=[e for e in eps if e['trained_arm']==arm]
        cnt=Counter();[cnt.update(e['action_counts']) for e in rows]
        wins=[w for e in rows for w in e['switch_windows'].values()]
        result[arm]={'episodes':len(rows),'targets':sum(len(e['reached_epochs']) for e in rows),
                     'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in rows),4),
                     'early_stop':sum(e['metrics']['stopped_early'] for e in rows),
                     'post_switch_ink':sum(w['painted'] for w in wins),
                     'recoveries':sum(e['metrics']['recoveries'] for e in rows),
                     'latency_ms_mean':round(statistics.mean(e['mean_latency_ms'] for e in rows),6),
                     'actions':dict(sorted(cnt.items()))}
    return result

def provenance():
    return {name:sha_file(Path(__file__).parent/name) for name in SOURCE}

def run(out,seed=651,base_updates=60,adapt_updates=20,n=48,depth=24,quick=False):
    if out.exists(): raise FileExistsError('immutable evidence exists')
    out.mkdir(parents=True);start=time.perf_counter()
    models,hist,ledgers=train(seed,base_updates,adapt_updates,n,depth)
    cps={}
    for arm,m in models.items():
        f=out/f'{arm}.npz';m.save(f);restored=host.Ranker.load(f)
        if any(not np.array_equal(restored.p[k],m.p[k]) for k in m.p):raise ValueError('restore')
        cps[arm]={'sha256':sha_file(f),'parameters':sum(p.size for p in m.p.values()),'updates':m.updates}
    (out/'traces').mkdir();(out/'png').mkdir()
    eps=evaluate(models,quick,out/'png')
    for i,e in enumerate(eps):
        (out/'traces'/f'{i:03d}-{e["trained_arm"]}.json').write_text(json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
    (out/'outcomes.json').write_text(json.dumps(ledgers,sort_keys=True,separators=(',',':'))+'\n')
    rep={'version':VERSION,'seed':seed,'eval_seed':EVAL_SEED,'curriculum':STAGES,
         'base_updates':base_updates,'adapt_updates':adapt_updates,'samples_per_stage':n,
         'suffix_depth':depth,'training':hist,'checkpoints':cps,
         'development':summary(eps),'episodes':len(eps),
         'decisions':sum(len(e['rows']) for e in eps),'source_sha256':provenance(),
         'development_specs':[s.__dict__ for s in cases()],
         'sealed_heldout_opened':False,'native_receipts':0,'paid_compute':False,
         'elapsed_s':round(time.perf_counter()-start,3),
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(rep,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f)
         for f in out.rglob('*') if f.is_file()},sort_keys=True,indent=2)+'\n')
    return rep

def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=provenance():raise ValueError('source mismatch')
    orig=json.loads((out/'SHA256SUMS.json').read_text())
    actual={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*')
            if f.is_file() and f.name!='SHA256SUMS.json'}
    if orig!=actual:raise ValueError('evidence hashes')
    models={a:host.Ranker.load(out/f'{a}.npz') for a in ARMS}
    for a,m in models.items():
        if sha_file(out/f'{a}.npz')!=rep['checkpoints'][a]['sha256'] or m.updates!=rep['checkpoints'][a]['updates']:
            raise ValueError('checkpoint mismatch')
    specs_by_id={s.ident:s for s in cases()};eps=[]
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text());arm=ep['trained_arm']
        fixture.verify_episode(ep,specs_by_id[ep['spec']],{'goal_recovery':models[arm]});eps.append(ep)
    if len(eps)!=rep['episodes'] or sum(len(e['rows']) for e in eps)!=rep['decisions']:
        raise ValueError('trajectory count')
    recomputed=summary(eps)
    for arm in ARMS:
        a=dict(recomputed[arm]);b=dict(rep['development'][arm]);
        a.pop('latency_ms_mean');b.pop('latency_ms_mean')
        if a!=b:raise ValueError('aggregate mismatch')
    return {'checkpoints':len(models),'episodes':len(eps),'decisions':rep['decisions'],
            'files':len(actual)}

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--out',type=Path);cli.add_argument('--verify',type=Path)
    cli.add_argument('--seed',type=int,default=651);cli.add_argument('--base-updates',type=int,default=60)
    cli.add_argument('--adapt-updates',type=int,default=20);cli.add_argument('--samples',type=int,default=48)
    cli.add_argument('--depth',type=int,default=24);cli.add_argument('--quick',action='store_true')
    args=cli.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.base_updates,
                    args.adapt_updates,args.samples,args.depth,args.quick),sort_keys=True))

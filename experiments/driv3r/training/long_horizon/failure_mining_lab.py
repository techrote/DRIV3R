"""FAILURE-MINING-001: legally replayable TRAIN-only risk sampling and suffix imitation.

Uses unchanged synthetic host fixture; this is not the CyberSand native ABI.
All labels are computed from actual frozen-student legal continuations, not a learned oracle.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gc
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import failure_outcome_lab as old, goal_recovery_lab as host, goal_retarget_lab as fixture, policy
from .world import World, MOVE, Spec, specs, digest
from .train import sha_file

VERSION = 'failure-mining/1'
STAGES = (36,72,128)
ARMS = ('frozen','risk_imitation','risk_suffix')
SOURCES = ('failure_mining_lab.py','failure_outcome_lab.py','goal_recovery_lab.py',
           'goal_retarget_lab.py','goal_conditioning_lab.py','world.py','policy.py','train.py')
DEV = tuple(Spec('development',sp.family,3201+100*i,sp.mode)
            for i,sp in enumerate(specs('development',1)))
EVAL_SEED = 773


def risk(w):
    menu=w.menu(12);moves=sum(a in MOVE for a in menu)
    if not moves and 'RELOCATE_START' in menu and 'BACKTRACK' in menu:return 4
    if not moves and len(menu)>1:return 3
    if moves==1 or w.recoveries>=8:return 2
    if moves==2 or w.recoveries:return 1
    return 0


def collect(seed,stage,n):
    if stage not in STAGES or not 16<=n<=96:raise ValueError('bounded curriculum')
    pool=[];source_counts=Counter(); forced=0
    base=specs('train',2)
    for sp0 in base:
        for variant in range(2):
            sp=replace(sp0,challenge='trap_corridor') if variant else sp0
            w=World.create(sp);targets=fixture.targets_for(sp,seed+variant*47,'retarget')
            rng=random.Random(seed*137+sp.seed*17+stage*13+variant)
            for t in range(stage):
                if w.stopped:break
                menu=w.menu(12)
                available=[a for a in menu if a!='STOP']
                if len(available)>=2 and t>=5:
                    tier=risk(w);key=digest([seed,stage,sp.ident,t,variant]);
                    pool.append((tier,key,copy.deepcopy(w),targets,t,variant))
                    source_counts[str(tier)]+=1
                target=targets[fixture.active_phase(t,'retarget')]
                action=old_model_action(w,target,stage)
                if variant and t<6 and 'MOVE_E' in menu:
                    forced+=int(action!='MOVE_E');action='MOVE_E'
                elif t in (19,57,91,119):
                    others=[a for a in menu if a in MOVE and a!=action]
                    if others:action=rng.choice(others);forced+=1
                w.apply(action,12)
    # Predeclared, outcome-blind risk stratification. Prevent all samples being
    # identical attractive trap tips; the low-risk stratum is a negative control.
    pool.sort(key=lambda x:x[1])
    buckets={tier:[row for row in pool if row[0]==tier] for tier in (4,3,2,1,0)}
    chosen=[]
    for tier in (4,2,1,0):
        limit=n//4
        chosen.extend(buckets[tier][:limit]);buckets[tier]=buckets[tier][limit:]
    remaining=sorted((row for tier in buckets for row in buckets[tier]),key=lambda row:(-row[0],row[1]))
    chosen.extend(remaining[:n-len(chosen)])
    return chosen,{'pool':len(pool),'risk_tiers':dict(source_counts),
                   'selected_risk_tiers':dict(Counter(str(p[0]) for p in chosen)),
                   'forced_bad_moves':forced,'cases':len(base)*2,'split':'train'}

# Bound frozen policy to one corpus run; never consult DEVELOPMENT results for labels.
_FROZEN=None
def old_model_action(w,goal,horizon):
    return _FROZEN.choose(w,goal,horizon,False)[0]


def mine(seed,stage,n=48,depth=16):
    if not 8<=depth<=32:raise ValueError('suffix bound')
    states,meta=collect(seed,stage,n)
    X=[];M=[];y0=[];y1=[];entries=[];changed=0;survival_forks=0;better_than_imitation=0
    for i,(tier,_,w,targets,t,variant) in enumerate(states):
        menu,x=host.matrix(w,targets[fixture.active_phase(t,'retarget')],stage,False)
        non_stop=[a for a in menu if a!='STOP']
        if len(non_stop)<2:raise ValueError('invalid risk state')
        imitation=host.label(w,targets[fixture.active_phase(t,'retarget')],stage,
                             random.Random(seed*31013+stage*101+i))
        outcomes={a:old.student_suffix(w,a,targets,t,stage,_FROZEN,depth) for a in non_stop}
        chosen=max(non_stop,key=lambda a:(outcomes[a]['completed'],outcomes[a]['goals'],
            outcomes[a]['paint'],-outcomes[a]['recovery'],-menu.index(a)))
        if imitation=='STOP':chosen=chosen
        else:
            b=outcomes[imitation]
            # Guard: no worse actual survival or goal contact, plus strict improvement.
            eligible=[a for a in non_stop if outcomes[a]['completed']>=b['completed']
                 and outcomes[a]['goals']>=b['goals']
                 and (outcomes[a]['completed']>b['completed'] or outcomes[a]['goals']>b['goals']
                      or outcomes[a]['paint']>b['paint'])]
            chosen=max(eligible,key=lambda a:(outcomes[a]['completed'],outcomes[a]['goals'],
                   outcomes[a]['paint'],-outcomes[a]['recovery'],-menu.index(a))) if eligible else imitation
        changed+=int(chosen!=imitation)
        survival_forks+=int(len({v['completed'] for v in outcomes.values()})>1)
        better_than_imitation+=int(imitation in outcomes and outcomes[chosen]['completed']>outcomes[imitation]['completed'])
        m=np.zeros(policy.ACTION_MAX,bool);m[:len(menu)]=True
        X.append(x);M.append(m);y0.append(menu.index(imitation));y1.append(menu.index(chosen))
        entries.append({'spec':w.spec.__dict__,'variant':variant,'stage':stage,'t':t,'risk':tier,
              'prefix_actions':[d['chosen'] for d in w.decisions], 'state_sha256':w.state_sha(),
              'targets':[list(a) for a in targets], 'menu':menu,'imitation':imitation,
              'suffix_label':chosen,'suffix_outcomes':outcomes})
    meta.update({'selected':len(entries),'relabeled':changed,'nonstop_survival_forks':survival_forks,
                 'strict_survival_gain_over_imitation':better_than_imitation,'suffix_depth':depth})
    return np.stack(X),np.stack(M),np.asarray(y0),np.asarray(y1),entries,meta


def train(seed,base_updates=30,adapt_updates=20,n=48,depth=16):
    if not 1<=base_updates<=80 or not 1<=adapt_updates<=50:raise ValueError('optimizer bound')
    base,prior=host.train(seed,base_updates);frozen=base['goal_recovery']
    global _FROZEN
    _FROZEN=frozen
    models={name:copy.deepcopy(frozen) for name in ARMS};steps=[];ledger=[]
    for stage in STAGES:
        X,M,Y0,Y1,entries,meta=mine(seed,stage,n,depth)
        fits={}
        for name,labels in [('risk_imitation',Y0),('risk_suffix',Y1)]:
            fits[name]=models[name].fit(X,M,labels,adapt_updates,seed*331+stage)
        steps.append({'stage':stage,'mining':meta,'optimizer':fits})
        ledger.extend(entries)
    _FROZEN=None
    return models,{'base':prior,'stages':steps},ledger


def cases():
    return list(DEV)+[replace(s,challenge='trap_corridor') for s in DEV]


def evaluate(models,quick=False):
    episodes=[]
    for sp in cases()[:1 if quick else len(cases())]:
        for horizon in ((128,) if quick else (128,160)):
            for arm in ARMS:
                ep=fixture.run_episode(sp,{'goal_recovery':models[arm]},'goal_recovery',
                                       EVAL_SEED,horizon,'retarget')
                fixture.verify_episode(ep,sp,{'goal_recovery':models[arm]})
                ep['trained_arm']=arm;episodes.append(ep)
    return episodes


def summary(eps):
    out={}
    for name in ARMS:
        rows=[e for e in eps if e['trained_arm']==name]
        counts=Counter()
        for ep in rows:counts.update(ep['action_counts'])
        out[name]={'episodes':len(rows),'reached_epochs':sum(len(e['reached_epochs']) for e in rows),
           'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in rows),4),
           'early_stop':sum(e['metrics']['stopped_early'] for e in rows),
           'post_switch_paint':sum(v['painted'] for e in rows for v in e['switch_windows'].values()),
           'recoveries':sum(e['metrics']['recoveries'] for e in rows),
           'latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in rows),6),
           'actions':dict(sorted(counts.items()))}
    return out


def provenance():return {n:sha_file(Path(__file__).parent/n) for n in SOURCES}


def run(out,seed=661,base_updates=30,adapt_updates=20,n=48,depth=16,quick=False):
    if out.exists():raise FileExistsError('immutable evidence already exists')
    out.mkdir(parents=True);started=time.perf_counter()
    # Copy-heavy, replayable TRAIN state reservoirs trigger pathological cyclic-GC
    # scans during CPU evaluation. This is a bounded temporary runtime setting.
    gc.collect();enabled=gc.isenabled()
    try:
        if enabled:gc.disable()
        models,history,ledger=train(seed,base_updates,adapt_updates,n,depth)
        cps={}
        for name,m in models.items():
            p=out/(name+'.npz');m.save(p)
            if not all(np.array_equal(a,m.p[k]) for k,a in host.Ranker.load(p).p.items()):
                raise ValueError('checkpoint restore')
            cps[name]={'sha256':sha_file(p),'parameters':sum(x.size for x in m.p.values()),'updates':m.updates}
        (out/'traces').mkdir()
        # Evaluate checkpoints restored from the exact saved files, never an
        # uncheckpointed in-memory optimizer model.
        models={name:host.Ranker.load(out/(name+'.npz')) for name in ARMS}
        eps=evaluate(models,quick)
    finally:
        if enabled:gc.enable()
    for i,ep in enumerate(eps):
        (out/'traces'/f'{i:03}-{ep["trained_arm"]}.json').write_text(json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n')
    (out/'mined_pairs.json').write_text(json.dumps(ledger,sort_keys=True,separators=(',',':'))+'\n')
    rep={'version':VERSION,'seed':seed,'base_updates':base_updates,'adapt_updates':adapt_updates,
       'curriculum':STAGES,'sample_count':n,'suffix_depth':depth,'training':history,
       'checkpoints':cps,'development_specs':[s.__dict__ for s in cases()],
       'development':summary(eps),'episodes':len(eps),'decisions':sum(len(e['rows']) for e in eps),
       'source_sha256':provenance(),'sealed_heldout_opened':False,'native_receipts':0,
       'paid_compute':False,'elapsed_s':round(time.perf_counter()-started,3),
       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(rep,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
          for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return rep


def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=provenance():raise ValueError('source identity')
    check={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if check!=json.loads((out/'SHA256SUMS.json').read_text()):raise ValueError('evidence sha')
    models={name:host.Ranker.load(out/(name+'.npz')) for name in ARMS}
    for name,m in models.items():
        if sha_file(out/(name+'.npz'))!=rep['checkpoints'][name]['sha256'] or m.updates!=rep['checkpoints'][name]['updates']:
            raise ValueError('checkpoint')
    pairs=json.loads((out/'mined_pairs.json').read_text())
    for row in pairs:
        sp=Spec(**row['spec'])
        if sp.split!='train' or row['stage'] not in STAGES:raise ValueError('training partition')
        w=World.create(sp)
        for action in row['prefix_actions']:
            w.apply(action,12)
        if (w.steps!=row['t'] or w.state_sha()!=row['state_sha256']
            or risk(w)!=row['risk'] or row['menu']!=w.menu(12)):
            raise ValueError('mined-state state/legality mismatch')
        ts=fixture.targets_for(sp,rep['seed']+row['variant']*47,'retarget')
        if [list(a) for a in ts]!=row['targets'] or row['suffix_label'] not in row['menu']:
            raise ValueError('training instruction mismatch')
        for action,original in row['suffix_outcomes'].items():
            if action=='STOP' or action not in w.menu(12):raise ValueError('nonlegal counterfactual')
            now=old.student_suffix(w,action,ts,w.steps,row['stage'],models['frozen'],rep['suffix_depth'])
            if now!=original:raise ValueError('counterfactual suffix mismatch')
    available={s.ident:s for s in cases()};eps=[]
    for p in sorted((out/'traces').glob('*.json')):
        ep=json.loads(p.read_text());arm=ep['trained_arm']
        fixture.verify_episode(ep,available[ep['spec']],{'goal_recovery':models[arm]});eps.append(ep)
    if len(eps)!=rep['episodes'] or sum(len(e['rows']) for e in eps)!=rep['decisions']:raise ValueError('counts')
    fresh=summary(eps)
    for arm in ARMS:
        a=dict(fresh[arm]);b=dict(rep['development'][arm]);a.pop('latency_ms');b.pop('latency_ms')
        if a!=b:raise ValueError('summary')
    return {'checkpoints':len(models),'episodes':len(eps),'decisions':rep['decisions'],
            'mined_pairs':len(pairs),'hashed_files':len(check)}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--verify',type=Path)
    ap.add_argument('--seed',type=int,default=661);ap.add_argument('--quick',action='store_true')
    ap.add_argument('--base-updates',type=int,default=30);ap.add_argument('--adapt-updates',type=int,default=20)
    ap.add_argument('--samples',type=int,default=48);ap.add_argument('--depth',type=int,default=16)
    args=ap.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.base_updates,
          args.adapt_updates,args.samples,args.depth,args.quick),sort_keys=True))

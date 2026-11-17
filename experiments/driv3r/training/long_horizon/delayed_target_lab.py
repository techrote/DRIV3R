"""DELAYED-TARGET-001: compare delayed target-outcome vs short lookahead imitation.

Research-only: TRAIN synthetic host worlds; DEVELOPMENT autonomous 128/160
rollouts. The teacher sees only the CURRENT target, never future switches.
Never uses the sealed structural family or native CyberSand runtime.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import cost_retarget_lab as short, goal_retarget_lab as fixture
from . import goal_recovery_lab as host, retarget_onpolicy_lab as predecessor
from . import policy
from .world import World, MOVE
from .train import sha_file

VERSION='delayed-target/1'
ARMS=('frozen','imitation','short_cost','delayed_target')
STAGES=(36,72,128)
SOURCE_FILES=('delayed_target_lab.py','cost_retarget_lab.py','retarget_onpolicy_lab.py',
              'goal_retarget_lab.py','goal_recovery_lab.py', 'goal_conditioning_lab.py',
              'world.py','policy.py','train.py')


def delayed_value(w,first,target,stage,seed,steps=12):
    """Counterfactual teacher-only outcome; never a model observation.

    The initial action is varied, followed by a deterministic CURRENT-goal
    teacher rollout. Terminal target contact is rewarded more than transient
    distance gain. Branch/corridor legality is delegated to the synthetic host.
    """
    if steps<9 or steps>18 or first not in w.menu(12):
        raise ValueError('illegal first move or unbounded horizon')
    sim=copy.deepcopy(w)
    before=len(sim.ink); d0=short.distance(sim.cursor,target); rec0=sim.recoveries
    rng=random.Random(seed)
    reached = target in sim.ink
    contact_step = None
    for k in range(steps):
        if sim.stopped:break
        action=first if k==0 else host.label(sim,target,stage,rng)
        if action not in sim.menu(12):raise ValueError('nonlegal synthetic teacher')
        event=sim.apply(action,12)
        if target in [tuple(p) for p in event['painted']]:
            reached=True
            if contact_step is None:contact_step=k+1
    gain=d0-short.distance(sim.cursor,target)
    painted=len(sim.ink)-before
    spent=sim.recoveries-rec0
    # No terminal reward for pre-painted goals: training credits first contact.
    reward=(22*int(contact_step is not None)
            + 0.25*gain + 0.25*painted - 1.5*spent
            - 12*int(sim.stopped))
    return {'value':round(reward,6),'reached_new_target':contact_step is not None,
            'contact_step':contact_step,'painted':painted,'gain':gain,
            'recovery_spent':spent,'stopped':bool(sim.stopped)}


def corpus(frozen,seed,stage,n=64):
    """Matched student-visited snapshots and training labels across arms."""
    if stage not in STAGES or n<32 or n>256:raise ValueError('corpus budget')
    reservoir=random.Random(seed*991+stage*37)
    samples=[];seen=0;forced=0
    for sp in short.specs('train',2):
        for variant in (0,1):
            w=World.create(sp)
            targets=fixture.targets_for(sp,seed+variant*41,'retarget')
            perturb=random.Random(seed*713+sp.seed*13+variant*31+stage)
            for t in range(stage):
                if w.stopped:break
                target=targets[fixture.active_phase(t,'retarget')]
                seen+=1
                snap=(copy.deepcopy(w),target,t,sp.ident,variant)
                if len(samples)<n:samples.append(snap)
                else:
                    j=reservoir.randrange(seen)
                    if j<n:samples[j]=snap
                asked,_=frozen.choose(w,target,stage,False)
                actual=asked
                if t in (19,57,91,119):
                    alternatives=[a for a in w.menu(12) if a in MOVE and a!=asked]
                    if alternatives:actual=perturb.choice(alternatives);forced+=1
                w.apply(actual,12)
    X=[];M=[];Y={'imitation':[],'short_cost':[],'delayed_target':[]}
    audit=[];disagreements=Counter();contacts=0
    for i,(w,target,t,ident,variant) in enumerate(samples):
        legal,imitation,short_label,short_scores=short.labels_at_state(
            w,target,stage,t,seed+i*3,5)
        menu,x=host.matrix(w,target,stage,False)
        if legal!=menu:raise ValueError('host menu mismatch')
        delayed={a:delayed_value(w,a,target,stage,
                    seed*113+t*791+w.spec.seed+i*29+17) for a in menu}
        values={a:delayed[a]['value'] for a in menu}
        best=max(values.values())
        tied=[a for a in menu if values[a]==best]
        delayed_label=imitation if imitation in tied else sorted(tied)[0]
        for arm,a in [('imitation',imitation),('short_cost',short_label),
                      ('delayed_target',delayed_label)]:Y[arm].append(menu.index(a))
        X.append(x)
        mask=np.zeros(policy.ACTION_MAX,dtype=bool);mask[:len(menu)]=True
        M.append(mask)
        disagreements['short_vs_imitation']+=int(imitation!=short_label)
        disagreements['delayed_vs_imitation']+=int(imitation!=delayed_label)
        disagreements['delayed_vs_short']+=int(delayed_label!=short_label)
        contacts+=int(any(v['reached_new_target'] for v in delayed.values()))
        audit.append({'spec':ident,'variant':variant,'decision':t,
            'state_sha256':w.state_sha(),'target':list(target),'menu':menu,
            'imitation':imitation,'short_cost':short_label,'delayed_target':delayed_label,
            'short_scores':short_scores,'delayed_scores':delayed})
    return np.stack(X),np.stack(M),{a:np.asarray(y,np.int64) for a,y in Y.items()},audit,{
        'split':'train','stage':stage,'samples':len(samples),'visited':seen,
        'forced_moves':forced,'candidate_sets_with_target_contact':contacts,
        'label_disagreements':dict(disagreements)}


def train(seed,base_steps=60,adapt_steps=20,n=64):
    if not(1<=base_steps<=100 and 1<=adapt_steps<=60):
        raise ValueError('optimizer bounds')
    base,history=host.train(seed,base_steps)
    frozen=base['goal_recovery']
    models={a:copy.deepcopy(frozen) for a in ARMS}
    stage_reports=[];ledgers=[]
    for stage in STAGES:
        X,M,Y,audit,meta=corpus(frozen,seed,stage,n)
        fit={}
        for arm in ARMS[1:]:
            fit[arm]=models[arm].fit(X,M,Y[arm],adapt_steps,seed*313+stage*7)
        stage_reports.append({'stage':stage,'corpus':meta,'optimizer':fit})
        ledgers.append({'stage':stage,'counterfactuals':audit})
    return models,{'base':history,'adaptation':stage_reports},ledgers


def evaluate(models,seed,quick=False,png_dir=None):
    cases=fixture.cases()[:1] if quick else fixture.cases()[:2]+fixture.cases()[4:6]
    episodes=[]
    for sp in cases:
        for horizon in ((128,) if quick else (128,160)):
            for mode in ('retarget',):
                for arm in ARMS:
                    png=(png_dir/f'{sp.ident}-{seed}-{arm}.png'
                        if png_dir is not None and sp==cases[0] and horizon==160
                        and mode=='retarget' else None)
                    ep=fixture.run_episode(sp,{'goal_recovery':models[arm]},
                        'goal_recovery',seed,horizon,mode,png)
                    fixture.verify_episode(ep,sp,{'goal_recovery':models[arm]})
                    ep['trained_arm']=arm
                    episodes.append(ep)
    return episodes


def summarize(episodes):
    result={}
    for mode in ('retarget','static'):
        result[mode]={}
        for arm in ARMS:
            rows=[e for e in episodes if e['mode']==mode and e['trained_arm']==arm]
            if not rows:continue
            actions=Counter();[actions.update(e['action_counts']) for e in rows]
            switches=[w for e in rows for w in e['switch_windows'].values()]
            result[mode][arm]={'episodes':len(rows),
                'targets':sum(len(e['reached_epochs']) for e in rows),
                'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in rows),4),
                'early_stops':sum(bool(e['metrics']['stopped_early']) for e in rows),
                'recoveries':sum(e['metrics']['recoveries'] for e in rows),
                'post_switch_8step_ink':sum(s['painted'] for s in switches) if mode=='retarget' else None,
                'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in rows),6),
                'action_counts':dict(sorted(actions.items()))}
    return result


def provenance():
    return {f:sha_file(Path(__file__).parent/f) for f in SOURCE_FILES}


def run(out,seed=631,base_steps=60,adapt_steps=20,n=64,quick=False):
    if out.exists():raise FileExistsError('immutable evidence dir exists')
    out.mkdir(parents=True)
    started=time.perf_counter()
    models,hist,ledgers=train(seed,base_steps,adapt_steps,n)
    checkpoints={}
    for arm,m in models.items():
        path=out/f'{arm}.npz';m.save(path)
        restore=host.Ranker.load(path)
        if any(not np.array_equal(m.p[k],restore.p[k]) for k in m.p):
            raise ValueError('restoration mismatch')
        checkpoints[arm]={'sha256':sha_file(path),'parameters':sum(x.size for x in m.p.values()),
                          'optimizer_updates':m.updates}
    (out/'png').mkdir();(out/'traces').mkdir()
    episodes=evaluate(models,seed,quick,out/'png')
    for i,ep in enumerate(episodes):
        (out/'traces'/f'{i:03d}-{ep["trained_arm"]}.json').write_text(
            json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n')
    (out/'counterfactuals.json').write_text(json.dumps(ledgers,sort_keys=True)+'\n')
    report={'version':VERSION,'seed':seed,'base_steps':base_steps,
        'adapt_steps':adapt_steps,'samples_per_stage':n,'curriculum':list(STAGES),
        'history':hist,'checkpoints':checkpoints,'development':summarize(episodes),
        'episodes':len(episodes),'decisions':sum(len(e['rows']) for e in episodes),
        'source_sha256':provenance(),'split':'development','sealed_heldout_opened':False,
        'native_receipts':0,'paid_compute':False,
        'elapsed_s':round(time.perf_counter()-started,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
        for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=provenance():
        raise ValueError('source provenance mismatch')
    expected=json.loads((out/'SHA256SUMS.json').read_text())
    observed={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*')
        if p.is_file() and p.name!='SHA256SUMS.json'}
    if expected!=observed:raise ValueError('artifact hash mismatch')
    models={a:host.Ranker.load(out/f'{a}.npz') for a in ARMS}
    for arm,m in models.items():
        if sha_file(out/f'{arm}.npz')!=r['checkpoints'][arm]['sha256'] or \
           m.updates!=r['checkpoints'][arm]['optimizer_updates']:
            raise ValueError('checkpoint identity mismatch')
    sp_by_id={s.ident:s for s in fixture.cases()};episodes=[]
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text());arm=ep['trained_arm']
        if arm not in models:raise ValueError('unknown controller')
        fixture.verify_episode(ep,sp_by_id[ep['spec']],{'goal_recovery':models[arm]})
        episodes.append(ep)
    if len(episodes)!=r['episodes'] or sum(len(e['rows']) for e in episodes)!=r['decisions']:
        raise ValueError('count mismatch')
    check=summarize(episodes)
    for mode in check:
        for arm in check[mode]:
            a=dict(check[mode][arm]);b=dict(r['development'][mode][arm])
            a.pop('mean_latency_ms');b.pop('mean_latency_ms')
            if a!=b:raise ValueError('aggregate mismatch')
    return {'verified_checkpoints':len(models),'verified_episodes':len(episodes),
            'verified_decisions':r['decisions'],'verified_file_hashes':len(observed)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=631)
    p.add_argument('--base-steps',type=int,default=60)
    p.add_argument('--adapt-steps',type=int,default=20)
    p.add_argument('--samples-per-stage',type=int,default=64)
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.seed,a.base_steps,
        a.adapt_steps,a.samples_per_stage,a.quick),sort_keys=True))

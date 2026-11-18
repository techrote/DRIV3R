"""GOAL-SWITCH-CURRICULUM-001: equal-budget, on-policy target-switch training.

Synthetic DRIV3R host fixture only. Never opens sealed heldout families, modifies
CyberSand or asserts that RELOCATE_START is legal in native CyberSand.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, random, resource, statistics, time
from collections import Counter
from dataclasses import replace
from pathlib import Path
import numpy as np
from . import goal_recovery_lab as base, goal_retarget_lab as old, policy
from .world import World, MOVE, specs, digest
from .train import sha_file

VERSION='goal-switch-curriculum/1'
ARMS=('sham','switch','frozen')
STAGES=(36,72,128)
SWITCHES=(32,72,112)
DEPENDENCIES=('goal_switch_curriculum_lab.py','goal_recovery_lab.py','goal_retarget_lab.py','goal_conditioning_lab.py','policy.py','world.py','train.py')

def sha_sources():
    return {n:sha_file(Path(__file__).parent/n) for n in DEPENDENCIES}

def train_specs():
    s=specs('train',2)
    return s+[replace(x,challenge='trap_corridor') for x in s[::2]]

def dev_specs():
    s=specs('development',3)[2::3]  # New seeds, familiar families, NOT structural heldout
    return s+[replace(x,challenge='trap_corridor') for x in s]

def goals_for(w,seed,variant=0):
    # Fixed before any policy rollout: never adapt target sequence to action outcome.
    r=base.goal.goals(w,seed+variant*29)
    if len(r)<4 or len(set(r))<4:raise ValueError('non-distinct target schedule')
    return tuple(r[:4])

def phase(t,stage,arm):
    if arm=='sham':return 0
    switches=tuple(round(stage*p) for p in (.25,.52,.78))
    return sum(t>=s for s in switches)

def collect(model,seed,stage,arm):
    X=[];M=[];Y=[];counts=Counter();actions=Counter();episodes=[];phase_counts=Counter()
    for sp in train_specs():
        for variant in range(2):
            w=World.create(sp);goals=goals_for(w,seed,variant)
            rng=random.Random(seed*1009+sp.seed*19+stage*3+variant)
            for t in range(stage):
                if w.stopped:break
                p=phase(t,stage,arm);target=goals[p]
                menu,x=base.matrix(w,target,stage,False)
                teacher=base.label(w,target,stage,rng)
                asked,_=model.choose(w,target,stage,False)
                if asked not in menu or teacher not in menu:raise ValueError('host menu violated')
                X.append(x);mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                M.append(mask);Y.append(menu.index(teacher));counts[teacher]+=1;actions[asked]+=1;phase_counts[p]+=1
                # Student acts; the teacher supplies labels but never drives state.
                # No forced rescue from STOP, and no use of DEVELOPMENT states.
                w.apply(asked,12)
            episodes.append({'spec':sp.ident,'variant':variant,'steps':w.steps,'stop':w.stopped})
    if not Y:raise ValueError('no on-policy examples')
    return np.stack(X),np.stack(M),np.asarray(Y,dtype=np.int64),{'samples':len(Y),'teacher_actions':dict(counts),'student_actions':dict(actions),'phase_samples':dict(phase_counts),'train_paths':episodes}

def train(seed,updates=36,pretrain=20):
    if not 1<=updates<=80 or not 1<=pretrain<=80:raise ValueError('bounded optimizer budget')
    seed_models,hist=base.train(seed,pretrain)
    frozen=seed_models['goal_recovery']
    models={'sham':copy.deepcopy(frozen),'switch':copy.deepcopy(frozen),'frozen':frozen}
    stages=[]
    for h in STAGES:
        info={'horizon':h,'arms':{}}
        for arm in ('sham','switch'):
            x,m,y,ledger=collect(models[arm],seed,h,arm)
            loss=models[arm].fit(x,m,y,updates,seed*101+h)
            info['arms'][arm]={'corpus':ledger,'optimization':loss}
        stages.append(info)
    return models,{'pretrain':hist,'stage_updates':updates,'pretrain_updates':pretrain,'stages':stages}

def episode(sp,seed,model,arm,horizon=160,variant=0,retain=True):
    if sp.split!='development' or horizon not in (128,160):raise ValueError('development-only')
    w=World.create(sp);schedule=goals_for(w,seed,variant);initial=w.base_sha();rows=[];lat=[]
    painted_epochs=set();window_gain={str(k):0 for k in SWITCHES};window_steps={str(k):0 for k in SWITCHES}
    for t in range(horizon):
        if w.stopped:break
        p=sum(t>=s for s in SWITCHES);target=schedule[p]
        before=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        menu,x=base.matrix(w,target,horizon,False)
        begin=time.perf_counter_ns();probs=model.probs(x,len(menu));lat.append((time.perf_counter_ns()-begin)/1e6)
        chosen=menu[int(probs.argmax())]
        event=w.apply(chosen,12)
        after=abs(w.cursor[0]-target[0])+abs(w.cursor[1]-target[1])
        if list(target) in event['painted']:painted_epochs.add(p)
        for s in SWITCHES:
            if s<=t<s+8:
                window_gain[str(s)]+=before-after;window_steps[str(s)]+=1
        if retain:
            rows.append({'phase':p,'target':list(target),'menu':menu,'p':[round(float(z),8) for z in probs],
                         'event':event,'gain':before-after})
    return {'version':VERSION,'seed':seed,'arm':arm,'split':'development','spec':sp.__dict__,
        'horizon':horizon,'variant':variant,'base_sha256':initial,'targets':[list(t) for t in schedule],
        'rows':rows,'decisions':w.steps,'final_sha256':w.state_sha(),'trace_sha256':digest(w.decisions),
        'painted_epochs':sorted(painted_epochs),'switch_gain':window_gain,'switch_steps':window_steps,
        'new_painted':w.metrics(horizon)['unique_painted_cells'],'stopped_early':w.metrics(horizon)['stopped_early'],
        'action_counts':dict(sorted(Counter(r['chosen'] for r in w.decisions).items())),
        'latency_ms':round(statistics.mean(lat),6) if lat else 0.,'native_receipts':0}

def replay(ep,models):
    from .world import Spec
    sp=Spec(**ep['spec']);model=models[ep['arm']]
    q=episode(sp,ep['seed'],model,ep['arm'],ep['horizon'],ep['variant'])
    for k in ep:
        if k=='latency_ms':continue
        if ep[k]!=q[k]:raise ValueError('source replay mismatch: '+k)
    return ep['decisions']

def summary(rows):
    res={}
    for arm in ARMS:
        es=[e for e in rows if e['arm']==arm]; c=Counter()
        for e in es:c.update(e['action_counts'])
        res[arm]={'episodes':len(es),'epochs_painted':sum(len(e['painted_epochs']) for e in es),
          'mean_painted':round(statistics.mean(e['new_painted'] for e in es),3),
          'early_stops':sum(e['stopped_early'] for e in es),
          'total_decisions':sum(e['decisions'] for e in es),
          'switch_gain_observed_mean':round(statistics.mean(v for e in es for k,v in e['switch_gain'].items() if e['switch_steps'][k]==8),3) if any(e['switch_steps'][k]==8 for e in es for k in e['switch_gain']) else None,
          'switch_windows_full':sum(e['switch_steps'][k]==8 for e in es for k in e['switch_steps']),
          'mean_latency_ms':round(statistics.mean(e['latency_ms'] for e in es),5),
          'actions':dict(sorted(c.items()))}
    return res

def manifest(out):return {str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}

def run(out,seeds=(619,620),updates=36):
    if out.exists():raise FileExistsError('immutable output path exists')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'episodes').mkdir();t0=time.perf_counter()
    rep={'version':VERSION,'seeds':list(seeds),'source_sha256':sha_sources(),'checkpoint_sha256':{},
         'training':{},'train_partition':'train seeded 1000-1301','development_partition':'development seed 3002/3102/3202/3302',
         'sealed_heldout_opened':False,'native_receipts':0,'paid_compute':False,'episodes':0}
    es=[]
    for seed in seeds:
        models,history=train(seed,updates);rep['training'][str(seed)]=history;rep['checkpoint_sha256'][str(seed)]={}
        for arm,m in models.items():
            cp=out/'checkpoints'/f'{seed}-{arm}.npz';m.save(cp);restored=base.Ranker.load(cp)
            if any(not np.array_equal(m.p[k],restored.p[k]) for k in m.p):raise ValueError('checkpoint restore')
            rep['checkpoint_sha256'][str(seed)][arm]=sha_file(cp)
        for sp in dev_specs():
            for h in (128,160):
                for arm in ARMS:
                    e=episode(sp,seed,models[arm],arm,h)
                    filename=f'{len(es):04d}.json';(out/'episodes'/filename).write_text(json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
                    es.append(e)
    rep['episodes']=len(es);rep['decisions']=sum(e['decisions'] for e in es)
    rep['summary']=summary(es);rep['elapsed_s']=round(time.perf_counter()-t0,3)
    rep['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/'report.json').write_text(json.dumps(rep,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest(out),sort_keys=True,indent=2)+'\n')
    return rep

def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=sha_sources() or r['sealed_heldout_opened'] is not False:raise ValueError('identity')
    if json.loads((out/'SHA256SUMS.json').read_text())!=manifest(out):raise ValueError('manifest mismatch')
    d=n=0;rows=[]
    for seed in r['seeds']:
        models={}
        for arm in ARMS:
            cp=out/'checkpoints'/f'{seed}-{arm}.npz'
            if sha_file(cp)!=r['checkpoint_sha256'][str(seed)][arm]:raise ValueError('checkpoint drift')
            models[arm]=base.Ranker.load(cp)
        for f in sorted((out/'episodes').glob('*.json')):
            e=json.loads(f.read_text())
            if e['seed']!=seed:continue
            d+=replay(e,models);n+=1;rows.append(e)
    if n!=r['episodes'] or d!=r['decisions']:raise ValueError('replay counts')
    got=summary(rows)
    for a in ARMS:
        x=dict(got[a]);y=dict(r['summary'][a]);x.pop('mean_latency_ms');y.pop('mean_latency_ms')
        if x!=y:raise ValueError('summary mismatch')
    return {'verified_episodes':n,'verified_decisions':d,'checkpoints':len(r['seeds'])*len(ARMS),'sha256_entries':len(manifest(out))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--quick',action='store_true');a=p.parse_args()
    if a.verify: print(json.dumps(verify(a.verify),sort_keys=True))
    else: print(json.dumps(run(a.out,(619,) if a.quick else (619,620),8 if a.quick else 36),sort_keys=True))
if __name__=='__main__':main()

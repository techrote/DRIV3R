"""GOAL-COND-001: host-observable goal commands versus matched goal-masked policy.

Synthetic CyberSand-style world/host menu only, never native CyberSand. No sealed
heldout family access. Three stages of REAL NumPy Adam optimization. Results are
original source-replayed trajectories, not model capability claims beyond DEV.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import policy
from .world import World, MOVE, digest, replay, specs
from .train import _png, sha_file

VERSION = 'goal-conditioned-autonomous/1'
STAGES = (36, 72, 128)
HORIZONS = (128, 160)
GOAL_OFFSETS = ((9,0),(0,9),(-9,0),(0,-9),(7,7),(-7,7),(7,-7),(-7,-7))
FEATURE_DIM = len(policy.FEATURES) + 6
MAX_ACTIONS = policy.ACTION_MAX


def goals(world, run_seed):
    """Public, host-observable target coordinates; deterministic and unoccupied at start."""
    offsets = list(GOAL_OFFSETS)
    random.Random(run_seed * 31 + world.spec.seed).shuffle(offsets)
    candidates = []
    for dx,dy in offsets:
        for delta in ((0,0),(1,0),(0,1),(-1,0),(0,-1)):
            p = (48+dx+delta[0], 48+dy+delta[1])
            if not world.occupied(p) and p not in candidates:
                candidates.append(p)
                break
    return candidates[:4]


def matrix(w, goal, horizon, mask_goal):
    menu = w.menu(12)
    x = np.zeros((MAX_ACTIONS, FEATURE_DIM), dtype=np.float64)
    for i,a in enumerate(menu):
        x[i,:len(policy.FEATURES)] = policy.features(w,a,horizon)
        if not mask_goal:
            cx,cy = w.cursor
            dx,dy = MOVE.get(a,(0,0))
            p=(cx+dx,cy+dy) if a in MOVE else (cx,cy)
            old = abs(goal[0]-cx)+abs(goal[1]-cy)
            new = abs(goal[0]-p[0])+abs(goal[1]-p[1])
            # Only explicit goal and current host-observable position are used.
            x[i,-6:] = ((goal[0]-cx)/48.,(goal[1]-cy)/48.,
                        dx*(goal[0]-cx)/48.+dy*(goal[1]-cy)/48.,
                        float(a in MOVE)*(old-new),
                        (old-new)/max(1.,old), 1./(1.+new/9.))
    return menu,x


class GoalRanker:
    def __init__(self, seed, hidden=24):
        self.seed, self.hidden, self.updates = seed,hidden,0
        r=np.random.default_rng(seed)
        self.p={'w1':r.normal(0,.09,(FEATURE_DIM,hidden)),'b1':np.zeros(hidden),
                'w2':r.normal(0,.09,hidden),'b2':np.zeros(1)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}

    def probs(self,x,n):
        h=np.tanh(x[:n]@self.p['w1']+self.p['b1'])
        z=h@self.p['w2']+self.p['b2'][0];z=z-z.max()
        p=np.exp(z)
        return p/p.sum()

    def choose(self,w,goal,horizon,mask_goal):
        menu,x=matrix(w,goal,horizon,mask_goal)
        p=self.probs(x,len(menu))
        return menu[int(p.argmax())],{a:round(float(p[i]),8) for i,a in enumerate(menu)}

    def fit(self,X, masks, labels, steps, seed):
        r=np.random.default_rng(seed); losses=[]
        for _ in range(steps):
            ii=r.integers(0,len(labels),size=32)
            x,valid,y=X[ii],masks[ii],labels[ii]
            h=np.tanh(x@self.p['w1']+self.p['b1'])
            z=h@self.p['w2']+self.p['b2'][0]
            z=np.where(valid,z,-1e8);z=z-z.max(axis=1,keepdims=True)
            q=np.exp(z)*valid;q/=q.sum(axis=1,keepdims=True)
            losses.append(float(-np.log(np.maximum(1e-30,q[np.arange(32),y])).mean()))
            d=q.copy();d[np.arange(32),y]-=1;d/=32.
            dh=d[:,:,None]*self.p['w2'][None,None,:]*(1-h*h)
            g={'w1':np.einsum('baf,bah->fh',x,dh),'b1':dh.sum(axis=(0,1)),
               'w2':np.einsum('bah,ba->h',h,d),'b2':np.array([d.sum()])}
            norm=math.sqrt(sum(float((v*v).sum()) for v in g.values()))
            self.updates+=1
            for k,grad in g.items():
                grad*=min(1.,5./max(norm,1e-12))
                self.m[k]=.9*self.m[k]+.1*grad;self.v[k]=.999*self.v[k]+.001*grad*grad
                mh=self.m[k]/(1-.9**self.updates);vh=self.v[k]/(1-.999**self.updates)
                self.p[k]-=.014*mh/(np.sqrt(vh)+1e-8)
        return {'examples':len(labels),'updates':steps,'initial_loss':round(losses[0],5),
                'final_loss':round(statistics.mean(losses[-min(20,len(losses)):]),5)}

    def save(self,path):
        np.savez_compressed(path,**self.p,seed=np.array(self.seed),
                            hidden=np.array(self.hidden),updates=np.array(self.updates))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=cls(int(data['seed']),int(data['hidden']))
            for k in m.p:
                if data[k].shape!=m.p[k].shape or not np.isfinite(data[k]).all():
                    raise ValueError('checkpoint shape/finite')
                m.p[k]=data[k].copy()
            m.updates=int(data['updates'])
            return m


def next_goal(w,targets,i):
    while i < len(targets) and targets[i] in w.ink:i+=1
    return i


def teacher_choice(w,goal,horizon,rng):
    menu,x=matrix(w,goal,horizon,False)
    score=policy.teacher_scores(w,menu,x[:,:len(policy.FEATURES)]).copy()
    for j,a in enumerate(menu):
        if a in MOVE:
            dx,dy=MOVE[a]
            old=abs(goal[0]-w.cursor[0])+abs(goal[1]-w.cursor[1])
            new=abs(goal[0]-w.cursor[0]-dx)+abs(goal[1]-w.cursor[1]-dy)
            score[j]+=3.5*(old-new)
        score[j]+=rng.uniform(-.11,.11)
    return menu[int(score.argmax())]


def training_corpus(seed,stage):
    X=[];M=[];Y=[];counts=Counter();interventions=0
    for sp in specs('train',2):
        for variant in range(2):
            w=World.create(sp)
            targets=goals(w,seed+variant*11); i=0
            r=random.Random(seed*10000+sp.seed*17+stage*7+variant)
            for t in range(stage):
                i=next_goal(w,targets,i)
                if w.stopped:break
                goal=targets[min(i,len(targets)-1)]
                menu,x=matrix(w,goal,stage,False)
                a=teacher_choice(w,goal,stage,r)
                X.append(x); mask=np.zeros(MAX_ACTIONS,dtype=bool);mask[:len(menu)]=True
                M.append(mask);Y.append(menu.index(a));counts[a]+=1
                alternatives=[k for k in menu if k in MOVE and k!=a]
                if alternatives and r.random()<.16:
                    a=r.choice(alternatives);interventions+=1
                w.apply(a,12)
    return np.stack(X),np.stack(M),np.array(Y,dtype=np.int64),{
        'examples':len(Y),'teacher_actions':dict(sorted(counts.items())),
        'off_policy_interventions':interventions,'world_partition':'train'}


def train(seed,updates):
    models={'blind':GoalRanker(seed),'conditioned':GoalRanker(seed)}
    stages=[]
    for h in STAGES:
        X,M,Y,meta=training_corpus(seed,h)
        arms={}
        for arm,m in models.items():
            input_X=X.copy()
            if arm=='blind':input_X[:,:,-6:]=0.
            arms[arm]=m.fit(input_X,M,Y,updates,seed*133+h)
        stages.append({'horizon':h,'training':meta,'optimizer':arms})
    return models,stages


def episode(sp,models,arm,seed,horizon,goal_seed=None,save_png=None):
    if sp.split!='development' or horizon not in HORIZONS or arm not in models:
        raise ValueError('DEVELOPMENT-only episode')
    w=World.create(sp); initial=w.base_sha()
    targets=goals(w,seed if goal_seed is None else goal_seed)
    i=0;rows=[];lat=[];rng=random.Random(seed+sp.seed)
    for t in range(horizon):
        i=next_goal(w,targets,i)
        if w.stopped:break
        goal=targets[min(i,len(targets)-1)]
        start=time.perf_counter_ns()
        action,p=models[arm].choose(w,goal,horizon,arm=='blind')
        lat.append((time.perf_counter_ns()-start)/1e6)
        requested=action
        if t in (27,68,107):
            alternatives=[a for a in w.menu(12) if a in MOVE and a!=action]
            if alternatives:action=alternatives[rng.randrange(len(alternatives))]
        event=w.apply(action,12)
        rows.append({'event':event,'goal':list(goal),'requested':requested,
                     'probabilities':p})
    if save_png is not None:_png(save_png,w)
    return {'version':VERSION,'arm':arm,'split':'development','spec':sp.ident,
        'seed':seed,'goal_seed':seed if goal_seed is None else goal_seed,'horizon':horizon,
        'targets':[list(p) for p in targets],'world_base_sha256':initial,
        'rows':rows,'trajectory_sha256':digest([r['event'] for r in rows]),
        'state_sha256':w.state_sha(),'ink':sorted([list(p) for p in w.ink]),
        'reached':sum(p in w.ink for p in targets),'metrics':w.metrics(horizon),
        'actions':dict(sorted(Counter(r['event']['chosen'] for r in rows).items())),
        'mean_latency_ms':round(statistics.mean(lat),6) if lat else 0.,
        'native_receipts':0}


def verify_episode(ep,sp,models):
    if ep['version']!=VERSION or ep['split']!='development' or ep['spec']!=sp.ident:
        raise ValueError('episode identity')
    w=World.create(sp)
    if w.base_sha()!=ep['world_base_sha256'] or ep['targets']!=[list(p) for p in goals(w,ep['goal_seed'])]:
        raise ValueError('source world or goals mismatch')
    rng=random.Random(ep['seed']+sp.seed)
    for t,row in enumerate(ep['rows']):
        goal=next((tuple(g) for g in ep['targets'] if tuple(g) not in w.ink),tuple(ep['targets'][-1]))
        if row['goal']!=list(goal):raise ValueError('goal state mismatch')
        action,p=models[ep['arm']].choose(w,goal,ep['horizon'],ep['arm']=='blind')
        if action!=row['requested'] or p!=row['probabilities']:
            raise ValueError('model action distribution mismatch')
        chosen=action
        if t in (27,68,107):
            alternatives=[a for a in w.menu(12) if a in MOVE and a!=action]
            if alternatives:chosen=alternatives[rng.randrange(len(alternatives))]
        event=row['event']
        if chosen!=event['chosen'] or event!=w.apply(chosen,12):
            raise ValueError('forced action / host state mismatch')
    if (ep['state_sha256']!=w.state_sha() or ep['trajectory_sha256']!=digest(w.decisions)
        or ep['ink']!=sorted([list(p) for p in w.ink])
        or ep['metrics']!=w.metrics(ep['horizon'])
        or ep['reached']!=sum(tuple(p) in w.ink for p in ep['targets'])
        or ep['actions']!=dict(sorted(Counter(r['chosen'] for r in w.decisions).items()))
        or (not w.stopped and len(ep['rows'])!=ep['horizon'])):
        raise ValueError('replay tamper')
    return len(w.decisions)


def summarize(eps):
    result={}
    for arm in ('blind','conditioned'):
        e=[x for x in eps if x['arm']==arm]
        actions=Counter();[actions.update(x['actions']) for x in e]
        result[arm]={'episodes':len(e),'reached_goals':sum(x['reached'] for x in e),
          'mean_ink':round(statistics.mean(x['metrics']['unique_painted_cells'] for x in e),3),
          'early_stops':sum(x['metrics']['stopped_early'] for x in e),
          'mean_turns':round(statistics.mean(x['metrics']['turns'] for x in e),3),
          'actions':dict(sorted(actions.items())),
          'mean_latency_ms':round(statistics.mean(x['mean_latency_ms'] for x in e),6)}
    return result


def dependency_shas():
    d=Path(__file__).parent
    return {n:sha_file(d/n) for n in ('goal_conditioning_lab.py','world.py','policy.py','train.py')}


def run(out,seed=613,updates=60,quick=False):
    if out.exists():raise FileExistsError('output immutable')
    if not (1<=updates<=200):raise ValueError('bounded optimizer')
    out.mkdir(parents=True);t=time.perf_counter()
    models,history=train(seed,updates)
    checks={}
    for arm,m in models.items():
        path=out/f'{arm}.npz';m.save(path)
        rt=GoalRanker.load(path)
        if any(not np.array_equal(rt.p[k],m.p[k]) for k in m.p):raise ValueError('restore')
        checks[arm]={'sha256':sha_file(path),'parameters':sum(v.size for v in m.p.values()),
                     'optimizer_updates':m.updates}
    (out/'traces').mkdir();eps=[];n=0
    for sp in specs('development',1)[:1 if quick else 4]:
        for horizon in HORIZONS[:1 if quick else 2]:
            for variant in range(1 if quick else 3):
                gseed=seed+variant*29
                for arm in models:
                    png=(out/'traces'/f'{sp.ident}-{horizon}-g{variant}-{arm}.png') if variant==0 and horizon==160 else None
                    ep=episode(sp,models,arm,seed,horizon,gseed,png)
                    n+=verify_episode(ep,sp,models)
                    (out/'traces'/f'{sp.ident}-{horizon}-g{variant}-{arm}.json').write_text(json.dumps(ep,sort_keys=True)+'\n')
                    eps.append(ep)
    report={'version':VERSION,'seed':seed,'updates_per_stage':updates,'training':history,
       'checks':checks,'summary':summarize(eps),'episodes':len(eps),'decisions':n,
       'source_sha256':dependency_shas(),'split':'development',
       'development_worlds':[s.ident for s in specs('development',1)[:1 if quick else 4]],
       'heldout_uninspected':True,'native_receipts':0,'paid_compute':False,
       'elapsed_s':round(time.perf_counter()-t,3),
       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
      for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=dependency_shas():raise ValueError('source')
    expected=json.loads((out/'SHA256SUMS.json').read_text())
    got={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if expected!=got:raise ValueError('manifest')
    models={a:GoalRanker.load(out/f'{a}.npz') for a in ('blind','conditioned')}
    for arm,m in models.items():
        if sha_file(out/f'{arm}.npz')!=r['checks'][arm]['sha256'] or m.updates!=r['checks'][arm]['optimizer_updates']:
            raise ValueError('checkpoint')
    dev={s.ident:s for s in specs('development',1)}; eps=[];n=0
    for path in sorted((out/'traces').glob('*.json')):
        e=json.loads(path.read_text());n+=verify_episode(e,dev[e['spec']],models);eps.append(e)
    summary=summarize(eps)
    for arm in summary:
        summary[arm].pop('mean_latency_ms')
        rr=dict(r['summary'][arm]);rr.pop('mean_latency_ms')
        if rr!=summary[arm]:raise ValueError('summary')
    if len(eps)!=r['episodes'] or n!=r['decisions']:raise ValueError('counts')
    return {'checkpoints':len(models),'episodes':len(eps),'decisions':n,'hashes':len(expected)}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=613);p.add_argument('--updates',type=int,default=60)
    p.add_argument('--quick',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.seed,a.updates,a.quick),sort_keys=True))

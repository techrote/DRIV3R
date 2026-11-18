"""GOAL-RECOVERY-001: goal-following vs matched recovery-aware tiny rankers.

TRAIN/DEVELOPMENT only; synthetic host-authoritative cells, not native CyberSand.
No structural heldout access, no paid compute, checkpoints outside Git.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import goal_conditioning_lab as goal, policy
from .world import World, MOVE, digest, specs
from .train import _png, sha_file

VERSION = 'goal-recovery-ablation/1'
WIDTH = goal.FEATURE_DIM + 8
HORIZONS = (128, 160)
STAGES = (36, 72, 128)
ARMS = ('goal_only', 'goal_recovery')


def matrix(w, target, horizon, mask_recovery):
    menu, old = goal.matrix(w, target, horizon, False)
    x = np.zeros((policy.ACTION_MAX, WIDTH), np.float64)
    x[:, :goal.FEATURE_DIM] = old
    if mask_recovery:
        return menu, x
    remaining = max(0, 12-w.recoveries)
    root = w.path[0]
    root_free = sum(not w.occupied((root[0]+dx,root[1]+dy)) for dx,dy in MOVE.values())
    for i,a in enumerate(menu):
        if a in MOVE:
            dx,dy = MOVE[a]; nxt=(w.cursor[0]+dx,w.cursor[1]+dy)
        elif a == 'BACKTRACK':
            nxt=w.path[-2]
        elif a == 'RELOCATE_START':
            nxt=root
        else:
            nxt=w.cursor
        free=sum(not w.occupied((nxt[0]+dx,nxt[1]+dy)) for dx,dy in MOVE.values())
        x[i,-8:] = (remaining/12., min(1.,(len(w.path)-1)/32.),
                    free/4., root_free/4.,
                    max(-1.,min(1.,(remaining-(len(w.path)-1))/12.)),
                    float(a=='RELOCATE_START')*root_free/4.,
                    float(a=='BACKTRACK')*free/4.,
                    float(a in MOVE)*max(0.,(3-free)/3.))
    return menu,x


class Ranker(goal.GoalRanker):
    def __init__(self, seed, hidden=24):
        super().__init__(seed,hidden)
        self.p['w1']=np.random.default_rng(seed).normal(0,.09,(WIDTH,hidden))
        self.m['w1']=np.zeros_like(self.p['w1'])
        self.v['w1']=np.zeros_like(self.p['w1'])

    def choose(self,w,target,horizon,mask_recovery):
        menu,x=matrix(w,target,horizon,mask_recovery)
        p=self.probs(x,len(menu))
        return menu[int(p.argmax())],{a:round(float(p[i]),8) for i,a in enumerate(menu)}


def label(w,target,horizon,rng):
    menu=w.menu(12)
    moves=[a for a in menu if a in MOVE]
    if not moves:
        if 'RELOCATE_START' in menu:
            root=w.path[0]
            exits=sum(not w.occupied((root[0]+dx,root[1]+dy)) for dx,dy in MOVE.values())
            if exits and (len(w.path)>5 or len(w.path)-1>12-w.recoveries):
                return 'RELOCATE_START'
        if 'BACKTRACK' in menu:return 'BACKTRACK'
        if 'RELOCATE_START' in menu:return 'RELOCATE_START'
        return 'STOP'
    return goal.teacher_choice(w,target,horizon,rng)


def corpus(seed,horizon):
    examples=[]; masks=[]; labels=[]; actions=Counter();forced=0
    base=specs('train',2)
    cases=base+[replace(s,challenge='trap_corridor') for s in base if s.seed%2==0]
    for sp in cases:
        for variant in range(2):
            w=World.create(sp);targets=goal.goals(w,seed+variant*13)
            r=random.Random(seed*1003+sp.seed*17+horizon*3+variant)
            for t in range(horizon):
                if w.stopped:break
                current=next((p for p in targets if p not in w.ink),targets[-1])
                menu,x=matrix(w,current,horizon,False)
                chosen=label(w,current,horizon,r)
                examples.append(x)
                m=np.zeros(policy.ACTION_MAX,bool);m[:len(menu)]=True
                masks.append(m);labels.append(menu.index(chosen));actions[chosen]+=1
                applied=chosen
                # Deliberately visit an attractive but unproductive dead end.
                if sp.challenge=='trap_corridor' and t<6 and 'MOVE_E' in menu:
                    applied='MOVE_E';forced+=int(applied!=chosen)
                else:
                    wrong=[a for a in menu if a in MOVE and a!=chosen]
                    if wrong and r.random()<.18:
                        applied=r.choice(wrong);forced+=1
                w.apply(applied,12)
    return (np.stack(examples),np.stack(masks),np.asarray(labels,dtype=np.int64),
            {'examples':len(labels),'teacher_actions':dict(actions),'forced_moves':forced,
             'worlds':len(cases)*2,'split':'train'})


def train(seed,updates):
    models={a:Ranker(seed) for a in ARMS};history=[]
    for h in STAGES:
        X,M,Y,meta=corpus(seed,h)
        arms={}
        for a,m in models.items():
            xx=X.copy()
            if a=='goal_only':xx[:,:,-8:]=0.
            arms[a]=m.fit(xx,M,Y,updates,seed*101+h)
        history.append({'stage':h,'corpus':meta,'optimizer':arms})
    return models,history


def episode(sp,models,arm,seed,horizon,variant=0,png=None):
    if sp.split!='development' or horizon not in HORIZONS or arm not in ARMS:
        raise ValueError('DEV-only rollout')
    w=World.create(sp);initial=w.base_sha();targets=goal.goals(w,seed+variant*29)
    rows=[];lat=[];rng=random.Random(seed*11+sp.seed+variant)
    for t in range(horizon):
        if w.stopped:break
        target=next((p for p in targets if p not in w.ink),targets[-1])
        start=time.perf_counter_ns()
        asked,probs=models[arm].choose(w,target,horizon,arm=='goal_only')
        lat.append((time.perf_counter_ns()-start)/1e6)
        actual=asked
        if t in (27,68,107):
            wrong=[a for a in w.menu(12) if a in MOVE and a!=asked]
            if wrong:actual=rng.choice(wrong)
        event=w.apply(actual,12)
        rows.append({'goal':list(target),'requested':asked,'probabilities':probs,'event':event})
    if png is not None:_png(png,w)
    return {'version':VERSION,'arm':arm,'split':'development','spec':sp.ident,
            'challenge':sp.challenge,'seed':seed,'variant':variant,'horizon':horizon,
            'targets':[list(g) for g in targets],'world_base_sha256':initial,
            'rows':rows,'trajectory_sha256':digest(w.decisions),'state_sha256':w.state_sha(),
            'metrics':w.metrics(horizon),'reached':sum(p in w.ink for p in targets),
            'ink':sorted([list(p) for p in w.ink]),
            'actions':dict(sorted(Counter(r['chosen'] for r in w.decisions).items())),
            'mean_latency_ms':round(statistics.mean(lat),6) if lat else 0.,'native_receipts':0}


def verify_episode(ep,sp,models):
    if ep['version']!=VERSION or ep['spec']!=sp.ident or ep['split']!='development':
        raise ValueError('identity')
    w=World.create(sp);targets=goal.goals(w,ep['seed']+ep['variant']*29)
    if w.base_sha()!=ep['world_base_sha256'] or ep['targets']!=[list(g) for g in targets]:
        raise ValueError('world/goal source mismatch')
    rng=random.Random(ep['seed']*11+sp.seed+ep['variant'])
    for t,row in enumerate(ep['rows']):
        if w.stopped:raise ValueError('post-stop action')
        target=next((p for p in targets if p not in w.ink),targets[-1])
        asked,probs=models[ep['arm']].choose(w,target,ep['horizon'],ep['arm']=='goal_only')
        if row['goal']!=list(target) or row['requested']!=asked or row['probabilities']!=probs:
            raise ValueError('neural action / probability corruption')
        actual=asked
        if t in (27,68,107):
            wrong=[a for a in w.menu(12) if a in MOVE and a!=asked]
            if wrong:actual=rng.choice(wrong)
        if row['event']['chosen']!=actual or row['event']!=w.apply(actual,12):
            raise ValueError('host-legal state transition mismatch')
    if (ep['trajectory_sha256']!=digest(w.decisions) or ep['state_sha256']!=w.state_sha()
        or ep['metrics']!=w.metrics(ep['horizon'])
        or ep['reached']!=sum(p in w.ink for p in targets)
        or ep['ink']!=sorted([list(p) for p in w.ink])
        or ep['actions']!=dict(sorted(Counter(r['chosen'] for r in w.decisions).items()))
        or (not w.stopped and len(ep['rows'])!=ep['horizon'])):
        raise ValueError('trajectory summary corruption')
    return len(ep['rows'])


def summary(eps):
    ret={}
    for arm in ARMS:
        es=[e for e in eps if e['arm']==arm]; actions=Counter()
        for e in es:actions.update(e['actions'])
        ret[arm]={'episodes':len(es),'goals_reached':sum(e['reached'] for e in es),
            'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in es),3),
            'early_stops':sum(e['metrics']['stopped_early'] for e in es),
            'recovery_actions':sum(e['metrics']['recoveries'] for e in es),
            'mean_turns':round(statistics.mean(e['metrics']['turns'] for e in es),3),
            'actions':dict(sorted(actions.items())),
            'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in es),6)}
    return ret


def source_sha():
    directory=Path(__file__).parent
    return {n:sha_file(directory/n) for n in ('goal_recovery_lab.py','goal_conditioning_lab.py','world.py','policy.py','train.py')}


def run(out,seed=615,updates=60,quick=False):
    if out.exists():raise FileExistsError('immutable output')
    if not 1<=updates<=200:raise ValueError('bounded optimizer')
    out.mkdir(parents=True);start=time.perf_counter()
    models,history=train(seed,updates);checks={}
    for arm,m in models.items():
        dest=out/f'{arm}.npz';m.save(dest)
        restored=Ranker.load(dest)
        if any(not np.array_equal(m.p[k],restored.p[k]) for k in m.p):raise ValueError('checkpoint')
        checks[arm]={'sha256':sha_file(dest),'parameters':sum(p.size for p in m.p.values()),'updates':m.updates}
    (out/'traces').mkdir();episodes=[];decisions=0
    families=specs('development',1)[:1 if quick else 4]
    cases=families+[replace(s,challenge='trap_corridor') for s in families]
    for sp in cases:
        for h in HORIZONS[:1 if quick else 2]:
            for variant in range(1 if quick else 2):
                for arm in ARMS:
                    png=out/'traces'/f'{sp.ident}-{h}-v{variant}-{arm}.png' if h==160 and variant==0 else None
                    ep=episode(sp,models,arm,seed,h,variant,png)
                    decisions+=verify_episode(ep,sp,models)
                    (out/'traces'/f'{sp.ident}-{h}-v{variant}-{arm}.json').write_text(json.dumps(ep,sort_keys=True)+'\n')
                    episodes.append(ep)
    report={'version':VERSION,'seed':seed,'history':history,'checks':checks,
        'summary':summary(episodes),'episodes':len(episodes),'decisions':decisions,
        'source_sha256':source_sha(),'heldout_uninspected':True,'native_receipts':0,
        'paid_compute':False,'elapsed_s':round(time.perf_counter()-start,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
       for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=source_sha():raise ValueError('source mismatch')
    expected=json.loads((out/'SHA256SUMS.json').read_text())
    observed={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*')
              if p.is_file() and p.name!='SHA256SUMS.json'}
    if observed!=expected:raise ValueError('manifest mismatch')
    models={a:Ranker.load(out/f'{a}.npz') for a in ARMS}
    for arm,m in models.items():
        if sha_file(out/f'{arm}.npz')!=r['checks'][arm]['sha256'] or m.updates!=r['checks'][arm]['updates']:
            raise ValueError('checkpoint mismatch')
    dev={s.ident:s for s in specs('development',1)}
    dev.update({s.ident:s for s in [replace(s,challenge='trap_corridor') for s in specs('development',1)]})
    episodes=[];n=0
    for p in sorted((out/'traces').glob('*.json')):
        ep=json.loads(p.read_text());n+=verify_episode(ep,dev[ep['spec']],models);episodes.append(ep)
    if n!=r['decisions'] or len(episodes)!=r['episodes']:raise ValueError('counts')
    got=summary(episodes)
    for a in ARMS:
        got[a].pop('mean_latency_ms');ref=dict(r['summary'][a]);ref.pop('mean_latency_ms')
        if got[a]!=ref:raise ValueError('summary mismatch')
    return {'checks':len(models),'episodes':len(episodes),'decisions':n,'file_hashes':len(expected)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=615);p.add_argument('--updates',type=int,default=60)
    p.add_argument('--quick',action='store_true');args=p.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.updates,args.quick),sort_keys=True))

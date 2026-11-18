"""HISTORY-ALIAS-001: controlled identical-ink forks with different path stacks.

Only the read-only historical synthetic World host fixture is used; never a
native CyberSand acknowledgement. TRAIN and DEVELOPMENT use disjoint path
lengths, seeds and terrain specs. Structural HELDOUT is not opened.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np
from . import history_destination_lab as prior, policy
from .world import MOVE, Spec, World, digest
from .train import sha_file

VERSION='history-alias/1'
STAGES=(36,72,128)
N_EXTRA=4
WIDTH=prior.WIDTH+N_EXTRA
# Real, host-legal eight-move outward-and-return prehistory; recoveries=8.
DETOUR=((-1,0),(-2,0),(-3,0),(-4,0),(-4,-1),(-3,-1),(-2,-1),(-1,-1))
SOURCES=('history_alias_lab.py','history_destination_lab.py','goal_recovery_lab.py',
         'goal_conditioning_lab.py','world.py','policy.py','train.py',
         'test_history_alias_lab.py','history_alias_gallery.py')

class Ranker(prior.Model):
    def __init__(self,seed,hidden=24):
        super().__init__(seed,hidden)
        self.p['w1']=np.random.default_rng(seed).normal(0,.09,(WIDTH,hidden))
        self.m['w1']=np.zeros_like(self.p['w1']);self.v['w1']=np.zeros_like(self.p['w1'])
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            m=cls(int(d['seed']),int(d['hidden']))
            for k in m.p:
                if d[k].shape!=m.p[k].shape or not np.isfinite(d[k]).all():
                    raise ValueError('Invalid checkpoint tensor')
                m.p[k]=d[k].copy()
            m.updates=int(d['updates'])
            return m

def proposals():
    # All self-avoiding paths are generated, rather than consulting hidden holdout.
    groups=defaultdict(list)
    def dfs(path):
        if 10<=len(path)<=14:
            groups[(frozenset(path),tuple(path[-2:]))].append(tuple(path))
        if len(path)==14:return
        x,y=path[-1]
        for v in ((x+1,y),(x,y+1),(x-1,y),(x,y-1)):
            if 0<=v[0]<4 and 0<=v[1]<4 and v not in path:dfs(path+[v])
    dfs([(0,0)])
    result=[];seen=set()
    for k,paths in groups.items():
        for i,p in enumerate(paths):
            for q in paths[i+1:]:
                for goal in sorted(k[0]):
                    if goal in p[-2:]:continue
                    dp=len(p)-2-p.index(goal);dq=len(q)-2-q.index(goal)
                    if not ((1<=dp<=3 and dq>3) or (1<=dq<=3 and dp>3)):
                        continue
                    pos,neg=(p,q) if dp<dq else (q,p)
                    sig=digest([pos,neg,goal])
                    if sig not in seen:
                        seen.add(sig);result.append((len(p),sig,pos,neg,goal))
                    break
    return sorted(result,key=lambda x:(x[0],x[1]))

def specs(split):
    for fi,f in enumerate(('pickets','ridges','islands','zigzag')):
        yield Spec(split,f,(1000 if split=='train' else 3000)+fi*100+7,
                   ('wall','route','motif')[(fi+1)%3])

def legal_move(a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    return next((action for action,d in MOVE.items() if d==(dx,dy)),None)

def state(sp,path):
    w=World.create(sp);origin=w.cursor
    old=(0,0)
    for q in DETOUR:
        action=legal_move(old,q)
        if action not in w.menu(12):raise ValueError('Detour not host legal')
        w.apply(action,12);old=q
    for _ in DETOUR:
        if 'BACKTRACK' not in w.menu(12):raise ValueError('Illegal recovery prehistory')
        w.apply('BACKTRACK',12)
    for a,b in zip(path,path[1:]):
        action=legal_move(a,b)
        if action not in w.menu(12):raise ValueError('Hamiltonian path obstructed')
        w.apply(action,12)
    w.apply('BACKTRACK',12)
    assert w.recoveries==9 and w.cursor==(origin[0]+path[-2][0],origin[1]+path[-2][1])
    return w

def pairs(split):
    selected=[r for r in proposals() if (r[0]<=12 if split=='train' else r[0]>=13)]
    selected=selected[:36 if split=='train' else 40]
    world_specs=list(specs(split))
    for i,(_,sig,pos,neg,g) in enumerate(selected):
        sp=world_specs[i%len(world_specs)]
        a,b=state(sp,pos),state(sp,neg)
        goal=(48+g[0],48+g[1])
        if (a.ink!=b.ink or a.materials!=b.materials or a.protected!=b.protected
           or a.cursor!=b.cursor or a.heading!=b.heading or a.steps!=b.steps
           or a.recoveries!=b.recoveries or a.menu()!=b.menu()
           or a.last_action!=b.last_action):
            raise ValueError('Alias pair not controlled')
        da=len(a.path)-1-a.path.index(goal)
        db=len(b.path)-1-b.path.index(goal)
        if not (1<=da<=3 and db>12-b.recoveries):raise ValueError('Target-label mistake')
        # Full old feature tensor must be exactly indistinguishable at the fork.
        assert np.array_equal(prior.prior.goal.matrix(a,goal,128,False)[1],
                              prior.prior.goal.matrix(b,goal,128,False)[1])
        yield {'case':sig,'spec':sp,'positive':a,'negative':b,'goal':goal,
               'recoverable_depth':da,'unrecoverable_depth':db}

def matrix(w,goal,horizon,mask=False):
    menu,base=prior.matrix(w,goal,horizon,False)
    x=np.zeros((policy.ACTION_MAX,WIDTH),np.float64);x[:,:prior.WIDTH]=base
    if not mask:
        remaining=max(0,12-w.recoveries)
        depth=(len(w.path)-1-w.path.index(goal)) if goal in w.path else 16
        slack=(remaining-depth)/12.
        for i,a in enumerate(menu):
            # Exact stack depth and recovery budget: both host-observable.
            # No answer/teacher label is injected; slack is a derived feature.
            x[i,-4:]=(min(depth,16)/16., remaining/12.,
                        slack*float(a=='BACKTRACK'),-slack*float(a=='STOP'))
    return menu,x

def examples(horizon):
    X=[];M=[];Y=[];count=Counter()
    for pair in pairs('train'):
        goal=pair['goal']
        for kind in ('positive','negative'):
            w=copy.deepcopy(pair[kind])
            depth=pair['recoverable_depth']
            # Match positive and negative training weights; the frozen features
            # really cannot disambiguate the paired first decisions.
            for i in range(depth):
                menu,x=matrix(w,goal,horizon)
                label='BACKTRACK' if kind=='positive' else 'STOP'
                X.append(x);mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                M.append(mask);Y.append(menu.index(label));count[label]+=1
                if kind=='positive':w.apply('BACKTRACK',12)
    return np.stack(X),np.stack(M),np.asarray(Y,np.int64),dict(count)

def train(seed,updates):
    models={a:Ranker(seed) for a in ('blind','legacy','raw','depth')};history=[]
    for horizon in STAGES:
        X,M,Y,counts=examples(horizon)
        row={'horizon':horizon,'examples':len(Y),'labels':counts,'arms':{}}
        for arm,m in models.items():
            xx=X.copy()
            if arm in ('blind','legacy'):xx[:,:,-N_EXTRA:]=0.
            if arm=='raw':xx[:,:,-2:]=0.
            if arm=='blind':xx[:,:,prior.prior.goal.FEATURE_DIM:prior.WIDTH]=0.
            row['arms'][arm]=m.fit(xx,M,Y,updates,seed*1009+horizon)
        history.append(row)
    return models,history

def rollout(w,goal,model,arm,horizon,case,kind,seed,save=False):
    w=copy.deepcopy(w);rows=[];first=None;hits=0 if w.cursor==goal else None
    action_count=Counter()
    for t in range(min(12,horizon)):
        if hits is not None or w.stopped:break
        menu,x=matrix(w,goal,horizon,arm in ('blind','legacy'))
        if arm=='raw':x[:,-2:]=0.
        if arm=='blind':x[:,prior.prior.goal.FEATURE_DIM:prior.WIDTH]=0.
        t0=time.perf_counter_ns();p=model.probs(x,len(menu))
        infer_ms=(time.perf_counter_ns()-t0)/1e6
        if not np.isfinite(p).all() or abs(p.sum()-1)>1e-9:raise ValueError('bad probabilities')
        action=menu[int(p.argmax())]
        if first is None:first=action
        event=w.apply(action,12)
        action_count[action]+=1
        if w.cursor==goal:hits=t+1
        if save:rows.append({'menu':menu,'probs':[round(float(z),10) for z in p],
                             'action':action,'receipt':event,'inference_ms':round(infer_ms,6)})
    return {'case':case,'split':'development','kind':kind,'seed':seed,'arm':arm,
            'horizon':horizon,'first':first,'first_correct':first==('BACKTRACK' if kind=='positive' else 'STOP'),
            'hit_step':hits,'decisions':sum(action_count.values()),
            'stopped':w.stopped,'actions':dict(sorted(action_count.items())),
            'final_sha':w.state_sha(),'trace_sha':digest(w.decisions),
            'rows':rows}

def evaluate(models,store_rows=False):
    all_rows=[]
    for seed,arms in models.items():
        for horizon in STAGES:
            for i,pair in enumerate(pairs('development')):
                for kind in ('positive','negative'):
                    for arm in ('blind','legacy','raw','depth'):
                        sample=store_rows and i<3 and horizon==128
                        all_rows.append(rollout(pair[kind],pair['goal'],arms[arm],arm,
                            horizon,pair['case'],kind,seed,sample))
    return all_rows

def summary(rows):
    table={}
    for arm in ('blind','legacy','raw','depth'):
        select=[r for r in rows if r['arm']==arm]
        counts=Counter();[counts.update(r['actions']) for r in select]
        table[arm]={'episodes':len(select),'initial_correct':sum(r['first_correct'] for r in select),
          'positive_hits':sum(r['hit_step'] is not None for r in select if r['kind']=='positive'),
          'positive_episodes':sum(r['kind']=='positive' for r in select),
          'negative_immediate_stop':sum(r['first']=='STOP' for r in select if r['kind']=='negative'),
          'negative_episodes':sum(r['kind']=='negative' for r in select),
          'decisions':sum(r['decisions'] for r in select),
          'actions':dict(sorted(counts.items()))}
    return table

def run(out,updates=96):
    if out.exists():raise FileExistsError(out)
    if not 1<=updates<=240:raise ValueError('Unbounded optimizer budget')
    t0=time.perf_counter();out.mkdir(parents=True);(out/'checkpoints').mkdir()
    sources={p:sha_file(Path(__file__).parent/p) for p in SOURCES}
    trained={};training={};checkpoints={}
    for seed in (619,620):
        trained[seed],training[str(seed)]=train(seed,updates);checkpoints[str(seed)]={}
        for arm,m in trained[seed].items():
            p=out/'checkpoints'/f'{seed}-{arm}.npz';m.save(p)
            checkpoints[str(seed)][arm]={'sha256':sha_file(p),'parameters':sum(v.size for v in m.p.values()),
                                           'updates':m.updates}
    rows=evaluate(trained,True)
    samples=[r for r in rows if r['rows']]
    (out/'samples.json').write_text(json.dumps(samples,sort_keys=True,separators=(',',':'))+'\n')
    for r in rows:r.pop('rows')
    (out/'development.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    summary_data=summary(rows)
    latency={}
    pair=next(pairs('development'))
    for seed,mm in trained.items():
        for arm,m in mm.items():
            menu,x=matrix(pair['positive'],pair['goal'],128,arm in ('blind','legacy'))
            if arm=='raw':x[:,-2:]=0.
            if arm=='blind':x[:,prior.prior.goal.FEATURE_DIM:prior.WIDTH]=0.
            for _ in range(10):m.probs(x,len(menu))
            vals=[]
            for _ in range(80):
                a=time.perf_counter_ns();m.probs(x,len(menu));vals.append((time.perf_counter_ns()-a)/1e6)
            latency[f'{seed}-{arm}']={'median_ms':round(statistics.median(vals),6),
                                     'p95_ms':round(float(np.quantile(vals,.95)),6)}
    result={'version':VERSION,'source_sha256':sources,
       'world_partitions':{'TRAIN':'path lengths 10-12, seeds 1007-1307',
         'DEVELOPMENT':'path lengths 13-14, seeds 3007-3307',
         'structural_HELDOUT':'sealed'},
       'train_pairs':len(list(pairs('train'))),'development_pairs':len(list(pairs('development'))),
       'true_alias_property':'same host menu, terrain, ink, cursor, heading, steps, recoveries and goal-only features; legacy path features may differ',
       'training':training,'checkpoints':checkpoints,'summary':summary_data,
       'episodes':len(rows),'decisions':sum(r['decisions'] for r in rows),
       'elapsed_s':round(time.perf_counter()-t0,3),
       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'inference_latency_ms':latency,'native_executions':0,'paid_compute':False}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
        for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return result

def verify(out):
    report=json.loads((out/'report.json').read_text())
    if report['source_sha256']!={p:sha_file(Path(__file__).parent/p) for p in SOURCES}:
        raise ValueError('Source SHA-256 drift')
    sums=json.loads((out/'SHA256SUMS.json').read_text())
    if sums!={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}:
        raise ValueError('Evidence SHA-256 drift')
    models={}
    for seed in (619,620):
        models[seed]={}
        for arm in ('blind','legacy','raw','depth'):
            p=out/'checkpoints'/f'{seed}-{arm}.npz'
            assert report['checkpoints'][str(seed)][arm]['sha256']==sha_file(p)
            models[seed][arm]=Ranker.load(p)
    recorded=json.loads((out/'development.json').read_text())
    new=evaluate(models,False)
    for r in new:r.pop('rows')
    assert new==recorded,'DEVELOPMENT replay divergence'
    assert summary(new)==report['summary']
    samples=json.loads((out/'samples.json').read_text())
    # Replay all saved full action/probability sequences apart from nondeterministic latency.
    for r in samples:
        p=next(p for p in pairs('development') if p['case']==r['case'])
        check=rollout(p[r['kind']],p['goal'],models[r['seed']][r['arm']],r['arm'],
                      r['horizon'],r['case'],r['kind'],r['seed'],True)
        for row in check['rows']:row.pop('inference_ms')
        for row in r['rows']:row.pop('inference_ms')
        assert check==r,'probability/host receipt replay divergence'
    return {'complete_episode_replay':len(new),'full_probability_and_receipt_samples':len(samples),
            'checkpoint_count':8,'hashes_pass':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--verify',action='store_true');ap.add_argument('--updates',type=int,default=96)
    args=ap.parse_args()
    print(json.dumps(verify(args.output) if args.verify else run(args.output,args.updates),indent=2))

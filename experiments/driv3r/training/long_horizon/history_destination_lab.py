"""HISTORY-DESTINATION-001: CPU-only source-faithful synthetic backtrack features.

No native CyberSand ABI claim. TRAIN and DEVELOPMENT seeds are disjoint.
A goal may be an already-inked history cell: success is CURSOR ARRIVAL, not
new painting. No structural HELDOUT, game edits, paid inference or GPU.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, resource, time
from collections import Counter
from pathlib import Path
import numpy as np
from . import goal_recovery_lab as prior, policy
from .world import World, Spec, MOVE, digest
from .train import sha_file

VERSION='history-destination/1'
STAGES=(36,72,128)
EXTRA=4
WIDTH=prior.WIDTH+EXTRA
SRC=('history_destination_lab.py','world.py','policy.py','goal_conditioning_lab.py','goal_recovery_lab.py','train.py')

class Model(prior.Ranker):
    def __init__(self,seed,hidden=24):
        super().__init__(seed,hidden)
        self.p['w1']=np.random.default_rng(seed).normal(0,.09,(WIDTH,hidden))
        self.m['w1']=np.zeros_like(self.p['w1'])
        self.v['w1']=np.zeros_like(self.p['w1'])
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=cls(int(data['seed']),int(data['hidden']))
            for k in m.p:
                if m.p[k].shape!=data[k].shape or not np.isfinite(data[k]).all():
                    raise ValueError('incorrect checkpoint shape or nonfinite values')
                m.p[k]=data[k].copy()
            m.updates=int(data['updates'])
            return m

def matrix(w,goal,horizon,mask=False):
    menu,base=prior.matrix(w,goal,horizon,False)
    x=np.zeros((policy.ACTION_MAX,WIDTH),dtype=np.float64)
    x[:,:prior.WIDTH]=base
    if not mask:
        old=abs(goal[0]-w.cursor[0])+abs(goal[1]-w.cursor[1])
        on_history=float(goal in w.path[:-1])
        for i,a in enumerate(menu):
            if a in MOVE:
                dd=MOVE[a];dst=(w.cursor[0]+dd[0],w.cursor[1]+dd[1])
            elif a=='BACKTRACK':dst=w.path[-2]
            elif a=='RELOCATE_START':dst=w.path[0]
            else:dst=w.cursor
            new=abs(goal[0]-dst[0])+abs(goal[1]-dst[1])
            # All values derived from host-observable goal, cursor and legal history.
            x[i,-4:]=((old-new)/9., (old*old-new*new)/256.,
                       float(dst in w.path[:-1]),on_history)
    return menu,x

def candidates(split):
    for fi,family in enumerate(('pickets','ridges','islands','zigzag')):
        for j in range(5 if split=='train' else 1):
            yield Spec(split,family,(1000 if split=='train' else 3000)+fi*100+j,
                       ('wall','route','motif')[(fi+j)%3],
                       'trap_corridor' if j%2 else 'normal')

def prefix(sp,variant,stage):
    w=World.create(sp)
    rng=random.Random(sp.seed*1009+variant*113+stage*37)
    for t in range(6+variant%8+stage//36):
        moves=[a for a in w.menu(12) if a in MOVE]
        if not moves:break
        # Deterministic off-policy synthetic history; every state transition is host legal.
        score={a:1.9*(a[-1]==w.heading)+rng.random()*2 for a in moves}
        chosen=max(moves,key=lambda a:score[a]);w.apply(chosen,12)
    moves=[a for a in w.menu(12) if a in MOVE]
    if len(w.path)<3 or 'BACKTRACK' not in w.menu(12) or not moves:
        return None
    return w

def scenarios(split,stage):
    for sp in candidates(split):
        for v in range(10 if split=='train' else 4):
            w=prefix(sp,v,stage)
            if w is None:continue
            moves=[a for a in w.menu(12) if a in MOVE]
            rng=random.Random(sp.seed*19+v*31+stage)
            # True recovery target is an already-painted ancestor: it can only
            # be revisited by consuming legal BACKTRACK operations.
            depth=1+(v%min(3,len(w.path)-1))
            target=w.path[-1-depth]
            yield sp,v,'history',w,target,'BACKTRACK'
            # Counterbalanced untouched neighboring goal is reachable in one MOVE.
            forward=moves[rng.randrange(len(moves))]
            dx,dy=MOVE[forward]
            target=(w.cursor[0]+dx,w.cursor[1]+dy)
            yield sp,v,'forward',w,target,forward

def corpus(stage):
    X=[];M=[];Y=[];c=Counter();ids=[]
    for sp,v,kind,w,target,label in scenarios('train',stage):
        menu,x=matrix(w,target,stage)
        if label not in menu:raise ValueError('teacher action not legal')
        mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
        X.append(x);M.append(mask);Y.append(menu.index(label));c[label]+=1
        ids.append((sp.ident,v,kind,w.state_sha()))
    if not len(Y):raise ValueError('empty train set')
    return np.stack(X),np.stack(M),np.array(Y,dtype=np.int64),{
       'examples':len(Y),'teacher_actions':dict(sorted(c.items())),
       'source_state_digest':digest(ids),'train_unique_specs':len(set(x[0] for x in ids)),
       'retreat_fraction':sum(k=='history' for _,_,k,_ in ids)/len(ids)}

def train(seed,updates):
    models={'masked':Model(seed),'visible':Model(seed)};history=[]
    for stage in STAGES:
        X,M,Y,info=corpus(stage)
        result={}
        for arm,m in models.items():
            xx=X.copy()
            if arm=='masked':xx[:,:,-EXTRA:]=0.
            result[arm]=m.fit(xx,M,Y,updates,seed*1009+stage)
        history.append({'horizon':stage,'corpus':info,'optimizer':result})
    return models,history

def predict(model,w,target,horizon,arm):
    if arm=='frozen':menu,x=prior.matrix(w,target,horizon,False)
    else:menu,x=matrix(w,target,horizon,arm in ('masked','ablated'))
    p=model.probs(x,len(menu))
    if menu!=w.menu(12) or abs(float(sum(p))-1)>1.e-8:
        raise ValueError('host menu or probabilities invalid')
    return menu,p,menu[int(p.argmax())]

def episode(sp,v,kind,start,target,model,arm,horizon,save_rows=False):
    w=copy.deepcopy(start);records=[];first=None;hit=0 if w.cursor==target else None
    moves=0
    for t in range(horizon):
        if w.stopped:break
        menu,p,choice=predict(model,w,target,horizon,arm)
        if first is None:first=choice
        receipt=w.apply(choice,12)
        if choice in MOVE or choice=='BACKTRACK':moves+=1
        if hit is None and w.cursor==target:hit=t+1
        if save_rows:records.append({'menu':menu,'p':[round(float(z),9) for z in p],
                                     'chosen':choice,'receipt':receipt})
    return {'spec':sp.__dict__,'variant':v,'kind':kind,'arm':arm,'horizon':horizon,
        'target':list(target),'prefix_sha':start.state_sha(),'first':first,
        'correct_first':first==('BACKTRACK' if kind=='history' else next(
           a for a,d in MOVE.items() if (start.cursor[0]+d[0],start.cursor[1]+d[1])==target)),
        'hit_step':hit,'new_ink':len(w.ink)-len(start.ink),'actions':dict(sorted(
             Counter(z['chosen'] for z in w.decisions[len(start.decisions):]).items())),
        'end_sha':w.state_sha(),'trace_sha':digest(w.decisions),
        'stopped':w.stopped,'decisions':w.steps-start.steps,'logical_moves':w.logical_moves-start.logical_moves,
        'rows':records}

def aggregate(rows):
    answer={}
    for arm in ('frozen','masked','visible','ablated'):
        answer[arm]={}
        for kind in ('history','forward'):
            subset=[r for r in rows if r['arm']==arm and r['kind']==kind]
            actions=Counter();[actions.update(r['actions']) for r in subset]
            answer[arm][kind]={'n':len(subset),'correct_first':sum(r['correct_first'] for r in subset),
                'target_hits':sum(r['hit_step'] is not None for r in subset),
                'mean_capped_arrival':round(sum(r['hit_step'] if r['hit_step'] is not None else r['horizon']+1 for r in subset)/max(1,len(subset)),3),
                'mean_new_ink':round(sum(r['new_ink'] for r in subset)/max(1,len(subset)),3),
                'premature_stop':sum(r['stopped'] and (r['hit_step'] is None) for r in subset),
                'executed_actions':dict(sorted(actions.items()))}
    return answer

def latency(models):
    sp=next(candidates('development'));w=prefix(sp,0,128)
    target=w.path[-2]
    report={}
    for seed,mm in models.items():
        for arm in ('frozen','masked','visible'):
            menu,x=(prior.matrix(w,target,128,False) if arm=='frozen' else
                    matrix(w,target,128,arm=='masked'))
            for _ in range(20):mm[arm].probs(x,len(menu))
            times=[]
            for _ in range(128):
                t=time.perf_counter_ns();mm[arm].probs(x,len(menu));times.append((time.perf_counter_ns()-t)/1e6)
            report[f'{seed}-{arm}']={'median_ms':round(float(np.median(times)),6),
                                     'p95_ms':round(float(np.quantile(times,.95)),6),
                                     'scope':'model probabilities only'}
    return report


def run(out,ckpt,updates=72):
    if out.exists():raise FileExistsError(out)
    t0=time.perf_counter();out.mkdir(parents=True);(out/'checkpoints').mkdir()
    old={}
    for seed in (619,620):old[seed]=prior.Ranker.load(ckpt/f'{seed}-frozen.npz')
    checkpoints={};training={};models={}
    for seed in (619,620):
        trained,history=train(seed,updates)
        models[seed]={**trained,'frozen':old[seed],'ablated':trained['visible']}
        training[str(seed)]=history;checkpoints[str(seed)]={}
        for arm in ('masked','visible'):
            p=out/'checkpoints'/f'{seed}-{arm}.npz';trained[arm].save(p)
            checkpoints[str(seed)][arm]={'sha256':sha_file(p),'updates':trained[arm].updates,
                                         'parameters':sum(x.size for x in trained[arm].p.values())}
        checkpoints[str(seed)]['frozen']={'sha256':sha_file(ckpt/f'{seed}-frozen.npz'),
                                         'updates':old[seed].updates}
    records=[];sample=[]
    for seed in (619,620):
        for horizon in STAGES:
            for sp,v,kind,w,target,_ in scenarios('development',horizon):
                for arm,m in models[seed].items():
                    store=(sp.seed%100==0 and v in (0,2) and horizon==128)
                    rec=episode(sp,v,kind,w,target,m,arm,horizon,store)
                    if store:sample.append(copy.deepcopy(rec))
                    rec.pop('rows');records.append(rec)
    summary={str(h):aggregate([r for r in records if r['horizon']==h]) for h in STAGES}
    manifest={'version':VERSION,'stages':STAGES,'sources':{p:sha_file(Path(__file__).parent/p) for p in SRC},
       'splits':{'train_seeds':'1000–1404','development_seeds':'3000–3301','heldout_opened':False},
       'training':training,'checkpoints':checkpoints,'summary':summary,
       'episodes':len(records),'decisions':sum(r['decisions'] for r in records),
       'elapsed_s':round(time.perf_counter()-t0,3),
       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'model_only_latency_ms':latency(models),
       'native_executions':0,'paid_compute':False,'no_extra_repo_changes':True}
    (out/'report.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    (out/'development_metrics.json').write_text(json.dumps(records,sort_keys=True,separators=(',',':'))+'\n')
    (out/'sample_trajectories.json').write_text(json.dumps(sample,sort_keys=True,separators=(',',':'))+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return manifest

def verify(out,ckpt):
    report=json.loads((out/'report.json').read_text())
    if report['version']!=VERSION or report['sources']!={p:sha_file(Path(__file__).parent/p) for p in SRC}:
        raise ValueError('source drift')
    if json.loads((out/'SHA256SUMS.json').read_text())!={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}:
        raise ValueError('manifest drift')
    models={}
    for seed in (619,620):
        mm={a:Model.load(out/'checkpoints'/f'{seed}-{a}.npz') for a in ('masked','visible')}
        mm.update(frozen=prior.Ranker.load(ckpt/f'{seed}-frozen.npz'))
        mm['ablated']=mm['visible'];models[seed]=mm
        for arm in ('masked','visible'):
            if sha_file(out/'checkpoints'/f'{seed}-{arm}.npz')!=report['checkpoints'][str(seed)][arm]['sha256']:
                raise ValueError('checkpoint mismatch')
    recorded=json.loads((out/'development_metrics.json').read_text())
    if len(recorded)!=report['episodes'] or sum(r['decisions'] for r in recorded)!=report['decisions']:
        raise ValueError('episode counts')
    assert report['summary']=={str(h):aggregate([r for r in recorded if r['horizon']==h]) for h in STAGES}
    sample=json.loads((out/'sample_trajectories.json').read_text())
    count=0
    for i,r in enumerate(sample):
        sp=Spec(**r['spec']);w=prefix(sp,r['variant'],r['horizon'])
        seed=619 if i<len(sample)//2 else 620
        checked=episode(sp,r['variant'],r['kind'],w,tuple(r['target']),models[seed][r['arm']],r['arm'],r['horizon'],True)
        if checked!=r:raise ValueError('sample trajectory probability/receipt replay mismatch')
        count+=1
    # Independent recomputation of every recorded DEVELOPMENT metric on restored checkpoints.
    for i,r in enumerate(recorded):
        sp=Spec(**r['spec']);w=prefix(sp,r['variant'],r['horizon'])
        seed=619 if i<len(recorded)//2 else 620
        new=episode(sp,r['variant'],r['kind'],w,tuple(r['target']),models[seed][r['arm']],r['arm'],r['horizon'],False)
        new.pop('rows')
        if new!=r:raise ValueError('full development replay mismatch at '+str(i))
    return {'full_metrics_independently_replayed':len(recorded),'probability_and_receipt_replayed_samples':count,
            'decisions_recorded':report['decisions']}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
    a.add_argument('--checkpoint-dir',type=Path,required=True);a.add_argument('--updates',type=int,default=72)
    a.add_argument('--verify',action='store_true');args=a.parse_args()
    print(json.dumps(verify(args.output,args.checkpoint_dir) if args.verify else run(args.output,args.checkpoint_dir,args.updates),indent=2))

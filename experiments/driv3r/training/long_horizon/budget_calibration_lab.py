"""BUDGET-CALIBRATION-001: train bounded recovery discrimination with unseen budgets.

Read-only synthetic World contract; no native CyberSand run, no sealed HELDOUT.
No fixture/instrumentation modifications.  The evaluated STOP decision is only
for the narrowly defined previously-painted target task, not general play.
"""
from __future__ import annotations
import argparse
import copy
from functools import lru_cache
from collections import Counter
import json
import resource
from pathlib import Path
import statistics
import time
import numpy as np
from . import history_alias_lab as alias
from . import policy
from .world import Spec, World, digest, replay
from .train import sha_file

VERSION = 'budget-calibration/1'
STAGES=(36,72,128)
TRAIN_BUDGETS=(2,3,4)
EVAL_BUDGETS=(1,3,5)  # 1 and 5 never supervised
ROTATIONS=(0,1,2,3)
SEEDS=(731,732)
UPDATES=64
HORIZON=6
ARMS=('calibrated','fixed3','budget_blind')
# Original history-alias fixture trained on the FIRST 36 paths of lengths 10-12;
# these are the last 14 paths of the train-length regime (all length 12).
TRAIN_MOTIFS=alias.proposals()[36:50]
# Original history-alias / transfer used the FIRST 40 / 20 paths respectively
# from length 13+. These 20 length-14 motifs were untouched by both.
DEVELOPMENT_MOTIFS=[r for r in alias.proposals() if r[0]>=13][70:90]
DETOUR=alias.DETOUR+((-1,-2),(-2,-2))
SOURCES=('budget_calibration_lab.py','test_budget_calibration_lab.py',
         'budget_calibration_gallery.py',
         'history_alias_lab.py','history_alias_transfer_lab.py',
         'history_destination_lab.py','goal_conditioning_lab.py',
         'goal_recovery_lab.py','world.py','policy.py','train.py')

def rotated(p,quarter):
    x,y=p
    for _ in range(quarter):x,y=-y,x
    return (x,y)

def make_state(sp,path,rot,budget):
    if budget not in (1,2,3,4,5):raise ValueError('invalid budget')
    w=World.create(sp)
    n=11-budget  # consume 10,9,8,7,6 recovery credits legally
    p=(0,0)
    for q in DETOUR[:n]:
        act=alias.legal_move(rotated(p,rot),rotated(q,rot))
        if act not in w.menu(12):raise ValueError(('illegal outward detour',sp,rot,budget,act))
        w.apply(act,12);p=q
    for _ in range(n):
        if 'BACKTRACK' not in w.menu(12):raise ValueError('illegal detour recovery')
        w.apply('BACKTRACK',12)
    for p,q in zip(path,path[1:]):
        act=alias.legal_move(rotated(p,rot),rotated(q,rot))
        if act not in w.menu(12):raise ValueError(('illegal path',sp,rot,budget,act))
        w.apply(act,12)
    if 'BACKTRACK' not in w.menu(12):raise ValueError('illegal terminal backtrack')
    w.apply('BACKTRACK',12)
    if 12-w.recoveries!=budget or w.logical_moves>64:
        raise ValueError('history budget or synthetic 64-move ceiling breached')
    return w

def _make_cases(split):
    motifs=TRAIN_MOTIFS if split=='train' else DEVELOPMENT_MOTIFS
    budgets=TRAIN_BUDGETS if split=='train' else EVAL_BUDGETS
    for i,(_,sig,pos,neg,g) in enumerate(motifs):
        fi=i%4
        sp=Spec(split,('pickets','ridges','islands','zigzag')[fi],
                (1611 if split=='train' else 3601)+fi*75+i//4,
                ('wall','route','motif')[i%3])
        for rot in ROTATIONS:
            for budget in budgets:
                a=make_state(sp,pos,rot,budget)
                b=make_state(sp,neg,rot,budget)
                goal=(48+rotated(g,rot)[0],48+rotated(g,rot)[1])
                if not (a.ink==b.ink and a.materials==b.materials and
                    a.protected==b.protected and a.cursor==b.cursor and
                    a.heading==b.heading and a.recoveries==b.recoveries and
                    a.steps==b.steps and a.last_action==b.last_action and
                    a.menu()==b.menu()):
                    raise AssertionError('failed exact-ink/history-only pairing')
                for kind,w in (('former_positive',a),('former_negative',b)):
                    depth=len(w.path)-1-w.path.index(goal) if goal in w.path else 999
                    label='BACKTRACK' if depth<=budget else 'STOP'
                    assert label in w.menu(12)
                    yield {'case':sig,'family':sp.family,'spec':sp,'rot':rot,
                           'budget':budget,'kind':kind,'state':w,'goal':goal,
                           'depth':depth,'label':label}

@lru_cache(maxsize=2)
def cases(split):
    return tuple(_make_cases(split))

def matrix(w,goal,arm):
    menu,x=alias.matrix(w,goal,128)
    x[:,-2:]=0.  # no engineered action-specific slack for any new arm
    if arm=='budget_blind':
        # Remove all explicit budget and recovery-fraction channels, including
        # earlier synthetic fixture features. Other state features such as
        # elapsed time can still correlate with budget, so not information-theoretic blinding.
        x[:,[18,30,34,alias.WIDTH-3]]=0.
    if menu!=w.menu(12):raise AssertionError('host offered action mismatch')
    return menu,x

@lru_cache(maxsize=9)
def corpus(stage,arm):
    X=[];M=[];Y=[];counts=Counter();source=[]
    for c in cases('train'):
        if arm=='fixed3' and c['budget']!=3:continue
        w=copy.deepcopy(c['state'])
        depth=c['depth'] if c['label']=='BACKTRACK' else 0
        for t in range(max(1,depth)):
            menu,x=matrix(w,c['goal'],arm)
            label=c['label']
            mask=np.zeros(policy.ACTION_MAX,dtype=bool);mask[:len(menu)]=True
            X.append(x);M.append(mask);Y.append(menu.index(label));counts[label]+=1
            source.append([c['case'],c['rot'],c['budget'],c['kind'],t,w.state_sha()])
            if label=='BACKTRACK':w.apply('BACKTRACK',12)
    return np.stack(X),np.stack(M),np.asarray(Y,np.int64),{
       'examples':len(Y),'actions':dict(counts),'state_digest':digest(source)}

def train_models():
    models={};logs=[]
    for seed in SEEDS:
        models[seed]={arm:alias.Ranker(seed) for arm in ARMS}
        for stage in STAGES:
            for arm,m in models[seed].items():
                X,M,Y,info=corpus(stage,arm)
                fit=m.fit(X,M,Y,UPDATES,seed*1009+stage)
                logs.append({'seed':seed,'stage':stage,'arm':arm,'corpus':info,'fit':fit})
    return models,logs

def rollout(c,m,arm,seed,full=False,limit=1):
    w=copy.deepcopy(c['state']);g=c['goal'];first=None;records=[]
    hit=0 if w.cursor==g else None;actions=Counter();lat=[]
    start_n=len(w.decisions)
    for t in range(limit):
        if hit is not None or w.stopped:break
        menu,x=matrix(w,g,arm)
        tick=time.perf_counter_ns();prob=m.probs(x,len(menu));lat.append((time.perf_counter_ns()-tick)/1e6)
        if not np.isfinite(prob).all() or abs(float(prob.sum())-1)>1e-8:
            raise AssertionError('invalid probability vector')
        chosen=menu[int(prob.argmax())];first=first or chosen
        receipt=w.apply(chosen,12);actions[chosen]+=1
        if w.cursor==g:hit=t+1
        if full:records.append({'menu':menu,'probs':[round(float(v),10) for v in prob],
                                'action':chosen,'receipt':receipt})
    return {'case':c['case'],'family':c['family'],'rotation':c['rot']*90,
        'budget':c['budget'],'kind':c['kind'],'seed':seed,'arm':arm,
        'depth':c['depth'],'label':c['label'],'first':first,'correct':first==c['label'],
        'hit_step':hit,'decisions':w.steps-c['state'].steps,
        'actions':dict(sorted(actions.items())),'state_before':c['state'].state_sha(),
        'state_after':w.state_sha(),'trace_sha':digest(w.decisions[start_n:]),
        'rows':records,'_lat':lat}

def evaluate(models,full=False):
    rows=[];samples=[];lat=[]
    for idx,c in enumerate(cases('development')):
        for seed,mm in models.items():
            for arm,m in mm.items():
                sample=full and (idx%10==0 or (c['label']=='BACKTRACK' and c['depth']>=2 and idx%7==0))
                r=rollout(c,m,arm,seed,False,1)
                lat.extend(r.pop('_lat'))
                r.pop('rows');rows.append(r)
                if sample:
                    full_r=rollout(c,m,arm,seed,True,HORIZON)
                    full_r.pop('_lat');samples.append(full_r)
    return rows,samples,lat

def summarize(rows):
    ans={}
    for arm in ARMS:
        ans[arm]={}
        for k in ('all','budget_1','budget_3','budget_5'):
            sub=[r for r in rows if r['arm']==arm and (k=='all' or r['budget']==int(k[-1]))]
            action=Counter();[action.update(r['actions']) for r in sub]
            ans[arm][k]={'n':len(sub),'correct':sum(r['correct'] for r in sub),
              'reachable':sum(r['label']=='BACKTRACK' for r in sub),
              'reachable_hits':sum(r['hit_step'] is not None for r in sub if r['label']=='BACKTRACK'),
              'false_retreat':sum(r['first']=='BACKTRACK' and r['label']=='STOP' for r in sub),
              'futile_any_action':sum(r['first']!='STOP' and r['label']=='STOP' for r in sub),
              'correct_stop':sum(r['first']=='STOP' and r['label']=='STOP' for r in sub),
              'actions':dict(sorted(action.items())),
              'max_decisions':max(r['decisions'] for r in sub)}
    return ans

def paired_motif_bootstrap(rows):
    # Motif is the independent scenario unit; rotations/budgets/seeds within
    # a motif are correlated, so naive binomial episode intervals are invalid.
    groups={}
    for r in rows:
        key=(r['case'],r['rotation'],r['budget'],r['kind'],r['seed'])
        groups.setdefault(key,{})[r['arm']]=int(r['correct'])
    result={}
    for arm in ('calibrated','budget_blind'):
        bymotif={}
        for key,arms in groups.items():
            bymotif.setdefault(key[0],[]).append(arms[arm]-arms['fixed3'])
        cluster=np.array([np.mean(v) for _,v in sorted(bymotif.items())],dtype=float)
        rng=np.random.default_rng(4344)
        sampling=rng.choice(cluster,size=(5000,len(cluster)),replace=True)
        bounds=np.quantile(sampling.mean(axis=1),[.025,.975])
        result[f'{arm}_minus_fixed3']={
          'mean_accuracy_delta':round(float(cluster.mean()),6),
          'descriptive_cluster_bootstrap_95':list(map(lambda q:round(float(q),6),bounds)),
          'n_independent_motifs':len(cluster),'motifs_positive':int(sum(cluster>0)),
          'motifs_negative':int(sum(cluster<0))}
    return result

def run(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    (out/'checkpoints').mkdir()
    t=time.perf_counter()
    models,logs=train_models()
    for seed,mm in models.items():
        for arm,m in mm.items():m.save(out/'checkpoints'/f'{seed}-{arm}.npz')
    rows,samples,lat=evaluate(models,True)
    (out/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'samples.json').write_text(json.dumps(samples,sort_keys=True,separators=(',',':'))+'\n')
    model_hash={p.name:sha_file(p) for p in sorted((out/'checkpoints').glob('*.npz'))}
    info={'version':VERSION,'partitions':{
         'train':'last 14 motifs length 12; budgets 2/3/4; four rotations; synthetic TRAIN world seeds 1611-1840',
         'development':'20 untouched length-14 motifs indexes 70:90 in long-path set; budgets 1/3/5; four rotations; fresh synthetic DEVELOPMENT seeds 3601-3830',
         'structural_heldout':'sealed'},
        'model_sha256':model_hash,'source_sha256':{p:sha_file(Path(__file__).parent/p) for p in SOURCES},
        'training':logs,'summary':summarize(rows),
        'paired_motif_bootstrap':paired_motif_bootstrap(rows),'episodes':len(rows),
        'first_action_decisions':sum(r['decisions'] for r in rows),
        'full_rollout_decisions':sum(r['decisions'] for r in samples),'full_rollouts':len(samples),
        'wall_s':round(time.perf_counter()-t,3),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'model_probability_ms':{'median':round(statistics.median(lat),6),'p95':round(float(np.quantile(lat,.95)),6)},
        'analytic_rule_accuracy':1.0,'evaluation_mode':'complete first-action census; 6-step sampled continuation','native_execution':False,'paid_spend':False,
        'note':'Synthetic classification uses STOP for futile retreat; not a general game policy.'}
    (out/'report.json').write_text(json.dumps(info,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return info

def verify(out):
    report=json.loads((out/'report.json').read_text())
    assert report['source_sha256']=={p:sha_file(Path(__file__).parent/p) for p in SOURCES}
    sums=json.loads((out/'SHA256SUMS.json').read_text())
    assert sums=={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    models={seed:{arm:alias.Ranker.load(out/'checkpoints'/f'{seed}-{arm}.npz') for arm in ARMS} for seed in SEEDS}
    assert report['model_sha256']=={p.name:sha_file(p) for p in sorted((out/'checkpoints').glob('*.npz'))}
    rows,_,_=evaluate(models)
    assert rows==json.loads((out/'episodes.json').read_text())
    assert summarize(rows)==report['summary']
    assert paired_motif_bootstrap(rows)==report['paired_motif_bootstrap']
    samples=json.loads((out/'samples.json').read_text())
    look={(c['case'],c['rot']*90,c['budget'],c['kind']):c for c in cases('development')}
    for old in samples:
        c=look[(old['case'],old['rotation'],old['budget'],old['kind'])]
        w=replay(c['spec'],c['state'].decisions)
        assert w.state_sha()==c['state'].state_sha()
        for r in old['rows']:
            assert r['menu']==w.menu(12)
            assert r['receipt']==w.apply(r['action'],12)
        again=rollout(c,models[old['seed']][old['arm']],old['arm'],old['seed'],True,HORIZON)
        again.pop('_lat')
        assert again==old
    return {'episode_metrics_replayed':len(rows),'independent_full_action_probability_receipt_replays':len(samples),'checkpoint_sha256_verified':True}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--verify',action='store_true')
    a=parser.parse_args()
    print(json.dumps(verify(a.output) if a.verify else run(a.output),indent=2))

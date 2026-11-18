"""CAP-CONDITIONING-001: paired-cap supervised synthetic recovery discriminator.

Only DRIV3R's frozen synthetic World; no native CyberSand claim or HELDOUT.
A synthetic max_recoveries intervention changes four *host-derived observation*
channels while leaving world state, legal action list and entire prehistory
constant. Task-specific STOP means 'do not spend recovery on this old goal';
not generic worm stop advice. A deterministic exact-history rule is the oracle.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, resource, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
from functools import lru_cache
import numpy as np
from . import budget_calibration_lab as prior
from . import history_alias_lab as alias, policy
from .world import digest, replay
from .train import sha_file

VERSION='cap-conditioning/1'
SEEDS=(833,834)
ARMS=('paired','static','cap_blind')
STAGES=(36,72,128)
TRAIN_CAPS=(2,4,6)  # never used for DEVELOPMENT scoring
DEV_CAPS=(1,3,5)
UPDATES=96
HORIZON=7
SOURCES=('cap_conditioning_lab.py','test_cap_conditioning_lab.py','cap_conditioning_gallery.py','budget_calibration_lab.py','history_alias_lab.py',
         'history_destination_lab.py','goal_recovery_lab.py','goal_conditioning_lab.py',
         'world.py','policy.py','train.py')
# Same four host feature coordinates as previous cap-intervention evidence.
INTERVENED=(18,30,34,alias.WIDTH-3)


def counterfactual(c,remaining,arm='paired'):
    if not 1<=remaining<=6:raise ValueError('synthetic credit limit exceeded')
    w=c['state'];cap=w.recoveries+remaining
    root=w.state_sha();initial_menu=w.menu(12)
    if initial_menu!=w.menu(cap) or not {'BACKTRACK','STOP'}<=set(initial_menu):
        raise ValueError('legal action menu not held constant')
    menu,x=prior.matrix(w,c['goal'],'calibrated')
    assert menu==initial_menu
    x=x.copy();baseline=x.copy()
    x[:len(menu),18]=w.recoveries/cap
    x[:len(menu),30]=remaining/cap
    x[:len(menu),34]=(remaining-(len(w.path)-1))/cap
    x[:len(menu),alias.WIDTH-3]=remaining/cap
    if arm=='cap_blind':x[:len(menu),list(INTERVENED)]=0.
    if w.state_sha()!=root or w.menu(12)!=initial_menu:
        raise AssertionError('counterfactual changed baseline world')
    if arm!='cap_blind':
        idx=[i for i in range(x.shape[1]) if i not in INTERVENED]
        assert np.array_equal(x[:,idx],baseline[:,idx])
    oracle='BACKTRACK' if c['depth']<=remaining else 'STOP'
    return menu,x,cap,oracle

@lru_cache(maxsize=8)
def corpus(stage,arm):
    X=[];M=[];Y=[];labels=Counter();samples=[]
    for c in prior.cases('train'):
        for cap in TRAIN_CAPS:
            # The static sham has an identical number of optimizer draws,
            # replacing all cap probes with identical cap=3 observations.
            applied_cap=3 if arm=='static' else cap
            menu,x,_,label=counterfactual(c,applied_cap,arm)
            mask=np.zeros(policy.ACTION_MAX,dtype=bool);mask[:len(menu)]=True
            X.append(x);M.append(mask);Y.append(menu.index(label));labels[label]+=1
            samples.append((c['case'],c['rot'],c['budget'],c['kind'],applied_cap,label))
    return np.stack(X),np.stack(M),np.asarray(Y,dtype=np.int64),{
        'examples':len(Y),'teacher_actions':dict(sorted(labels.items())),
        'train_state_cap_digest':digest(samples),
        'stage':stage,'training_cap_values':(3,) if arm=='static' else TRAIN_CAPS}


def fit():
    models={};logs=[];tic=time.perf_counter()
    for seed in SEEDS:
        models[seed]={arm:alias.Ranker(seed) for arm in ARMS}
        for stage in STAGES:
            for arm,m in models[seed].items():
                X,M,Y,meta=corpus(stage,arm)
                result=m.fit(X,M,Y,UPDATES,seed*1013+stage)
                logs.append({'seed':seed,'stage':stage,'arm':arm,'training':meta,'optimizer':result})
    return models,logs,round(time.perf_counter()-tic,3)


def predict(c,remaining,model,arm):
    menu,x,cap,oracle=counterfactual(c,remaining,arm)
    t=time.perf_counter_ns();p=model.probs(x,len(menu));lat=(time.perf_counter_ns()-t)/1e6
    if len(p)!=len(menu) or not np.isfinite(p).all() or abs(float(sum(p))-1)>1e-8:
        raise ValueError('probabilities invalid')
    return {'menu':menu,'probs':[round(float(q),10) for q in p],
            'action':menu[int(p.argmax())],'oracle':oracle,'cap':cap,'latency_ms':lat}


def table_rows(models,split='development'):
    if split!='development':raise ValueError('Only DEVELOPMENT evaluated')
    rows=[];lat=[]
    for c in prior.cases('development'):
        pre=c['state'].state_sha()
        for seed,arms in models.items():
            for arm,m in arms.items():
                for remaining in DEV_CAPS:
                    out=predict(c,remaining,m,arm);lat.append(out.pop('latency_ms'))
                    row={'case':c['case'],'family':c['family'],'rotation':c['rot']*90,
                         'kind':c['kind'],'original_budget':c['budget'],
                         'goal':list(c['goal']),'depth':c['depth'],
                         'state_sha':pre,'seed':seed,'arm':arm,'budget':remaining,
                         'max_recoveries':out['cap'],'menu':out['menu'],
                         'p':[round(z,9) for z in out['probs']],
                         'first':out['action'],'oracle':out['oracle'],
                         'correct':out['action']==out['oracle'],
                         'p_backtrack':round(out['probs'][out['menu'].index('BACKTRACK')],9)}
                    rows.append(row)
        assert c['state'].state_sha()==pre
    return rows,lat


def summary(rows):
    d={};grouped=defaultdict(dict)
    for r in rows:
        key=(r['case'],r['rotation'],r['kind'],r['original_budget'],r['seed'],r['arm'])
        grouped[key][r['budget']]=r
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        triads=[v for k,v in grouped.items() if k[-1]==arm]
        assert len(triads)==len(subset)//3 and all(set(v)==set(DEV_CAPS) for v in triads)
        d[arm]={'n':len(subset),'correct':sum(r['correct'] for r in subset),
         'wrong_futile_backtrack':sum(r['first']=='BACKTRACK' and r['oracle']=='STOP' for r in subset),
         'wrong_stops_reachable':sum(r['first']=='STOP' and r['oracle']=='BACKTRACK' for r in subset),
         'off_task_first':sum(r['first'] not in ('BACKTRACK','STOP') for r in subset),
         'actions':dict(sorted(Counter(r['first'] for r in subset).items())),
         'by_budget':{str(b):{'n':sum(r['budget']==b for r in subset),
            'correct':sum(r['correct'] for r in subset if r['budget']==b),
            'oracle_recoverable':sum(r['oracle']=='BACKTRACK' for r in subset if r['budget']==b),
            'futile_backtrack':sum(r['first']=='BACKTRACK' and r['oracle']=='STOP' for r in subset if r['budget']==b)}
            for b in DEV_CAPS},
         'choice_order_inversions':sum(any(v[a]['first']=='BACKTRACK' and v[b]['first']=='STOP' for a,b in ((1,3),(3,5))) for v in triads),
         'probability_nonmonotone':sum(any(v[b]['p_backtrack']+1e-8<v[a]['p_backtrack'] for a,b in ((1,3),(3,5))) for v in triads)}
    return d


def bootstrap(rows):
    # Group by motif (not correlated rotations/caps/seed as IID episodes).
    merged=defaultdict(dict)
    for r in rows:
        merged[(r['case'],r['rotation'],r['kind'],r['original_budget'],r['seed'],r['budget'])][r['arm']]=int(r['correct'])
    ans={};rng=np.random.default_rng(90210)
    for arm in ('paired','cap_blind'):
        groups=defaultdict(list)
        for k,v in merged.items():groups[k[0]].append(v[arm]-v['static'])
        z=np.array([statistics.mean(a) for _,a in sorted(groups.items())]);draw=rng.choice(z,size=(3000,len(z)),replace=True).mean(axis=1)
        ans[arm+'_minus_static']={'mean':round(float(z.mean()),5),
          'motif_bootstrap_95':list(map(lambda x:round(float(x),5),np.quantile(draw,[.025,.975]))),
          'independent_motifs':len(z)}
    return ans


def continuation(c,m,arm,remaining,seed):
    """Autonomous, legal 7-step synthetic continuation at a fixed new cap."""
    w=copy.deepcopy(c['state']);cap=w.recoveries+remaining;rows=[];hit=w.cursor==c['goal']
    first=None;start=len(w.decisions)
    for t in range(HORIZON):
        if hit or w.stopped:break
        # Changing the synthetic cap is a research fixture counterfactual.
        # Policy features must be recomputed for the current state, not frozen root.
        menu,x=prior.matrix(w,c['goal'],'calibrated')
        if menu!=w.menu(cap):break  # No unsupported menu remapping
        x=x.copy()
        x[:len(menu),18]=w.recoveries/cap
        x[:len(menu),30]=(cap-w.recoveries)/cap
        x[:len(menu),34]=((cap-w.recoveries)-(len(w.path)-1))/cap
        x[:len(menu),alias.WIDTH-3]=(cap-w.recoveries)/cap
        if arm=='cap_blind':x[:len(menu),list(INTERVENED)]=0.
        p=m.probs(x,len(menu));action=menu[int(p.argmax())];first=first or action
        receipt=w.apply(action,cap)
        rows.append({'menu':menu,'p':[round(float(q),9) for q in p],
                     'action':action,'receipt':receipt})
        hit=hit or w.cursor==c['goal']
    return {'case':c['case'],'rotation':c['rot']*90,'kind':c['kind'],
       'original_budget':c['budget'],'budget':remaining,'seed':seed,'arm':arm,
       'state_before':c['state'].state_sha(),'state_after':w.state_sha(),
       'first':first,'hit':hit,'steps':len(rows),
       'actions':dict(sorted(Counter(z['action'] for z in rows).items())),
       'trace_sha':digest(w.decisions[start:]),'rows':rows}


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('output must be new')
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    start=time.perf_counter();models,logs,train_s=fit()
    for seed,mm in models.items():
        for arm,m in mm.items():m.save(out/'checkpoints'/f'{seed}-{arm}.npz')
    rows,lat=table_rows(models)
    (out/'decisions.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    samples=[]
    for i,c in enumerate(prior.cases('development')):
        if i%40==0 or (i%29==0 and c['depth']<=5):
            for arm in ARMS:
                for budget in DEV_CAPS:
                    samples.append(continuation(c,models[SEEDS[0]][arm],arm,budget,SEEDS[0]))
    (out/'trajectories.json').write_text(json.dumps(samples,sort_keys=True,separators=(',',':'))+'\n')
    report={'experiment':'CAP-CONDITIONING-001','version':VERSION,
       'source_commit':'9ed4d813565e24cab41fcc41d5f90c1c91c76b43',
       'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in SOURCES},
       'checkpoints_sha256':{p.name:sha_file(p) for p in sorted((out/'checkpoints').glob('*.npz'))},
       'training':logs,'training_wall_s':train_s,'summary':summary(rows),
       'bootstrap':bootstrap(rows),'decision_vectors':len(rows),
       'unique_development_states':len({r['state_sha'] for r in rows}),
       'sampled_legal_continuations':len(samples),
       'continuation_actions':sum(e['steps'] for e in samples),
       'continuation_stats':{arm:{'n':sum(s['arm']==arm for s in samples),
           'reach':sum(s['hit'] for s in samples if s['arm']==arm),
           'actions':dict(sorted(Counter(a for s in samples if s['arm']==arm for a in (z['action'] for z in s['rows'])).items()))}
           for arm in ARMS},
       'wall_s':round(time.perf_counter()-start,3),
       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'model_only_median_ms':round(statistics.median(lat),6),
       'model_only_p95_ms':round(float(np.quantile(lat,.95)),6),
       'structural_heldout_opened':False,'native_cybersand':False,'paid_spend':False,
       'warnings':['Variable max_recoveries is only a synthetic fixture intervention, not a native CyberSand API.',
         'Deterministic stack depth oracle is an analytic comparator; task-specific STOP is not general game policy.',
         'Synthetic generated prehistories and a 7-step continuation do not prove sustained 72-128-decision autonomy.',
         'Checkpoint archive is local, not immutable GitHub Actions artifact.']}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'},sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    out=Path(out);report=json.loads((out/'report.json').read_text())
    sums=json.loads((out/'SHA256SUMS.json').read_text())
    assert sums=={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    assert report['source_sha256']=={n:sha_file(Path(__file__).parent/n) for n in SOURCES}
    models={seed:{arm:alias.Ranker.load(out/'checkpoints'/f'{seed}-{arm}.npz') for arm in ARMS} for seed in SEEDS}
    assert report['checkpoints_sha256']=={p.name:sha_file(p) for p in sorted((out/'checkpoints').glob('*.npz'))}
    original=json.loads((out/'decisions.json').read_text());replayed,_=table_rows(models)
    assert original==replayed and summary(replayed)==report['summary'] and bootstrap(replayed)==report['bootstrap']
    samples=json.loads((out/'trajectories.json').read_text())
    key={(c['case'],c['rot']*90,c['kind'],c['budget']):c for c in prior.cases('development')}
    for sample in samples:
        c=key[(sample['case'],sample['rotation'],sample['kind'],sample['original_budget'])]
        w=replay(c['spec'],c['state'].decisions)
        assert w.state_sha()==sample['state_before']
        cap=w.recoveries+sample['budget']
        for r in sample['rows']:
            assert r['menu']==w.menu(cap)
            assert w.apply(r['action'],cap)==r['receipt']
        assert w.state_sha()==sample['state_after']
        renewed=continuation(c,models[sample['seed']][sample['arm']],sample['arm'],sample['budget'],sample['seed'])
        assert renewed==sample
    return {'decision_probability_vectors_recomputed':len(replayed),
            'complete_legal_receipt_continuations_replayed':len(samples),
            'legal_continuation_actions_replayed':sum(s['steps'] for s in samples),
            'checkpoint_hashes_verified':True,'unmodified_source_hashes_verified':True}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args();print(json.dumps(verify(a.out) if a.verify else run(a.out),indent=2))

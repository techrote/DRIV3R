"""OUTCOME-FORK-001: compare actual suffix outcomes for MOVE versus BACKTRACK.

Synthetic research only. Never opens structural HELDOUT, changes native CyberSand,
or changes the frozen historical d_common fixture. TRAIN labels use exact state
counterfactuals, DEVELOPMENT reports remain independent of teacher selection.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import resource
import statistics
import tempfile
import time
import zipfile
import numpy as np
from . import retrace_choice_lab as prior, sequential_probe_lab as seq
from .world import MOVE, Spec, specs
from .train import sha_file

VERSION='outcome-fork/1'
SEEDS=(811,814)
STAGES=(36,72,128)
ARMS=('frozen','always_move','contrastive','outcome')
SOURCE=('outcome_fork_lab.py','retrace_choice_lab.py','sequential_probe_lab.py',
        'sensing_probe_lab.py','menu_alias_lab.py','contrastive_fork_lab.py',
        'recovery_fork_lab.py','goal_recovery_lab.py','goal_conditioning_lab.py',
        'goal_retarget_lab.py','world.py','policy.py','train.py')

def read_frozen(prior_zip,seed):
    with zipfile.ZipFile(prior_zip) as z:
        cp=z.read(f'evidence/checkpoints/{seed}-frozen.npz')
        manifest=json.loads(z.read('evidence/report.json'))
        if hashlib.sha256(cp).hexdigest()!=manifest['checkpoint_info'][str(seed)]['frozen']['sha256']:
            raise ValueError('frozen checkpoint hash mismatch')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'frozen.npz';p.write_bytes(cp)
        return seq.ProbeRanker.load(p),hashlib.sha256(cp).hexdigest()

def suffix(spec,turns,kind,first,model,horizon,trace=False):
    if horizon not in STAGES or spec.split not in ('train','development'):
        raise ValueError('TRAIN/DEVELOPMENT only')
    q=prior.fork(spec,turns,kind);w=q['world'];start=len(w.ink)
    if first not in (q['side'],'BACKTRACK') or first not in w.menu(12):
        raise ValueError('first choice not host-legal')
    initial=w.state_sha();first_event=w.apply(first,12);counts=Counter([first]);events=[first_event]
    lat=[]
    for step in range(1,horizon):
        if w.stopped:break
        t=time.perf_counter_ns();menu,x=prior.inputs(q,horizon)
        pp=model.probs(x,len(menu));a=menu[int(pp.argmax())]
        lat.append((time.perf_counter_ns()-t)/1e6)
        event=w.apply(a,12);counts[a]+=1
        if trace:events.append({'event':event,'probabilities':{v:round(float(pp[i]),8) for i,v in enumerate(menu)}})
    return {'split':spec.split,'spec':spec.__dict__,'turns':turns,'kind':kind,'first':first,
            'initial_sha256':initial,'final_sha256':w.state_sha(),
            'painted':len(w.ink)-start,'steps':w.steps-len(q['prefix']),
            'recovery':w.recoveries,'stopped':w.stopped,
            'actions':dict(sorted(counts.items())),
            'latency_ms':round(statistics.mean(lat),6) if lat else 0.,
            'events':events if trace else None}

def compare(spec,turns,kind,model,horizon=128,trace=False):
    q=prior.fork(spec,turns,kind)
    a=suffix(spec,turns,kind,q['side'],model,horizon,trace)
    b=suffix(spec,turns,kind,'BACKTRACK',model,horizon,trace)
    if a['initial_sha256']!=b['initial_sha256']:
        raise ValueError('counterfactual start drift')
    # Primary outcome is real painted cells, not geometry proxy. Tie prefers
    # MOVE (avoids unnecessary recovery); STOP is reported separately.
    delta=a['painted']-b['painted']
    return {'split':spec.split,'spec':spec.__dict__,'turns':turns,'kind':kind,
            'side_action':q['side'],'side':a,'backtrack':b,
            'paint_delta_move_minus_backtrack':delta,
            'winner':q['side'] if delta>=0 else 'BACKTRACK',
            'tie':delta==0}

def train_targets(seed,frozen):
    cases=[]
    for sp in specs('train',1):
        for turn in range(4):
            for kind in prior.KINDS:
                cases.append(compare(sp,turn,kind,frozen,128))
    return cases

def train(seed,source_zip,updates=40):
    if seed not in SEEDS or not 1<=updates<=120:raise ValueError('bounded training')
    frozen,cp_sha=read_frozen(source_zip,seed)
    models={'frozen':frozen,'always_move':copy.deepcopy(frozen),
            'contrastive':copy.deepcopy(frozen),'outcome':copy.deepcopy(frozen)}
    comparisons=train_targets(seed,frozen)
    targets={(c['spec']['family'],c['turns'],c['kind']):c['winner'] for c in comparisons}
    history=[]
    for stage in STAGES:
        X,M,Y,ledger,meta=prior.examples(seed,stage,natural=24,repeats=8)
        x=X['contrastive'].copy();m=M['contrastive'].copy();y=Y['contrastive'].copy()
        idx=24
        for sp in specs('train',1):
            for turns in range(4):
                for q in prior.paired(sp,turns):
                    menu,_=prior.inputs(q,stage)
                    y[idx:idx+8]=menu.index(targets[(sp.family,turns,q['kind'])]);idx+=8
        if idx!=len(y):raise ValueError('supervision indexing drift')
        for arm in ('always_move','contrastive','outcome'):
            xx,mm,yy=(x,m,y) if arm=='outcome' else (X[arm],M[arm],Y[arm])
            history.append({'seed':seed,'stage':stage,'arm':arm,
                            'optimizer':models[arm].fit(xx,mm,yy,updates,seed*71+stage),
                            'rows':len(yy),'source_split':meta['natural_split']})
    return models,comparisons,history,cp_sha

def episode(spec,turn,kind,arm,model,horizon=128):
    if spec.split!='development' or arm not in ARMS or horizon not in STAGES:raise ValueError('DEV only')
    return prior.episode(spec,turn,kind,'contrastive' if arm=='outcome' else arm,model,horizon)

def stats(episodes):
    out={}
    for kind in prior.KINDS:
        for arm in ARMS:
            es=[e for e in episodes if e['kind']==kind and e['_arm']==arm]
            if not es:continue
            counts=Counter(r['action'] for e in es for r in e['rows'])
            out[kind+'/'+arm]={'n':len(es),'painted_mean':round(statistics.mean(e['painted'] for e in es),3),
                'early_stop':sum(e['early_stop'] for e in es),
                'recovery_mean':round(statistics.mean(e['recovery'] for e in es),3),
                'first_move':sum(e['first_action']==prior.fork(Spec(**e['spec']),e['turns'],kind)['side'] for e in es),
                'latency_ms_mean':round(statistics.mean(e['mean_latency_ms'] for e in es),6),
                'actions':dict(sorted(counts.items()))}
    return out

def run(out,source_zip,quick=False,updates=40):
    if out.exists():raise FileExistsError('never overwrite evidence')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'trajectories').mkdir()
    start=time.perf_counter();all_eps=[];all_cf=[];models_by_seed={};hist=[];cps={}
    for seed in ((811,) if quick else SEEDS):
        models,comparisons,history,parent= train(seed,source_zip,updates)
        hist+=history;all_cf+=comparisons;models_by_seed[seed]=models;cps[str(seed)]={'parent_sha256':parent}
        for arm,model in models.items():
            f=out/'checkpoints'/f'{seed}-{arm}.npz';model.save(f)
            loaded=seq.ProbeRanker.load(f)
            if any(not np.array_equal(model.p[k],loaded.p[k]) for k in model.p):raise ValueError('checkpoint restoration')
            cps[str(seed)][arm]={'sha256':sha_file(f),'parameters':int(sum(x.size for x in model.p.values())),
                                 'optimizer_steps':model.updates}
        development=specs('development',1)[:1 if quick else 4]
        for sp in development:
            for turn in ((0,) if quick else (0,1)):
                for kind in prior.KINDS:
                    for arm,model in models.items():
                        ep=episode(sp,turn,kind,arm,model,36 if quick else 128)
                        ep['_arm']=arm
                        if prior.verify_episode({k:v for k,v in ep.items() if k!='_arm'},model)!=len(ep['rows']):raise ValueError('replay')
                        name=f'{seed}-{sp.family}-{turn}-{kind}-{arm}.json'
                        (out/'trajectories'/name).write_text(json.dumps(ep,sort_keys=True)+'\n')
                        all_eps.append(ep)
    (out/'train_counterfactuals.json').write_text(json.dumps(all_cf,sort_keys=True)+'\n')
    cf_counts=Counter(c['winner']=='BACKTRACK' for c in all_cf)
    paired={}
    for arm in ARMS:
        for kind in prior.KINDS:
            es=[e for e in all_eps if e['_arm']==arm and e['kind']==kind]
            paired[arm+'/'+kind]=sum(e['painted'] for e in es)
    result={'version':VERSION,'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in SOURCE},
        'prior_zip_sha256':sha_file(source_zip),'checkpoints':cps,'optimizer':hist,
        'train_counterfactuals':len(all_cf),'train_backtrack_wins':cf_counts[True],
        'train_move_wins_or_ties':cf_counts[False],
        'train_delta_distribution':dict(sorted(Counter(str(c['paint_delta_move_minus_backtrack']) for c in all_cf).items())),
        'development':stats(all_eps),'development_episodes':len(all_eps),
        'development_decisions':sum(len(e['rows']) for e in all_eps),
        'elapsed_s':round(time.perf_counter()-start,3),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'sealed_heldout_opened':False,'native_receipts':0,'paid_compute':False}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file()},sort_keys=True,indent=2)+'\n')
    return result

def verify(out):
    result=json.loads((out/'report.json').read_text());sums=json.loads((out/'SHA256SUMS.json').read_text())
    files={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'}
    if files!=sums or result['source_sha256']!={n:sha_file(Path(__file__).parent/n) for n in SOURCE}:raise ValueError('identity/manifest')
    models={};eps=[]
    for sd,cs in result['checkpoints'].items():
        models[int(sd)]={}
        for arm in ARMS:
            p=out/'checkpoints'/f'{sd}-{arm}.npz'
            if cs[arm]['sha256']!=sha_file(p):raise ValueError('checkpoint hash')
            models[int(sd)][arm]=seq.ProbeRanker.load(p)
    for p in sorted((out/'trajectories').glob('*.json')):
        ep=json.loads(p.read_text());seed=int(p.name.split('-')[0]);arm=ep['_arm']
        prior.verify_episode({k:v for k,v in ep.items() if k!='_arm'},models[seed][arm]);eps.append(ep)
    if stats(eps)!=result['development'] or len(eps)!=result['development_episodes']:
        raise ValueError('reported metrics differ from replay')
    cfs=json.loads((out/'train_counterfactuals.json').read_text())
    if any(c['split']!='train' for c in cfs):raise ValueError('heldout contamination')
    if len(cfs)!=result['train_counterfactuals']:raise ValueError('counterfactual count')
    return {'checkpoints':sum(len(v) for v in models.values()),'episodes':len(eps),
            'decisions':sum(len(e['rows']) for e in eps),'files':len(files),'train_counterfactuals':len(cfs)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path)
    parser.add_argument('--verify',type=Path);parser.add_argument('--prior-zip',type=Path)
    parser.add_argument('--quick',action='store_true');parser.add_argument('--updates',type=int,default=40)
    p=parser.parse_args();print(json.dumps(verify(p.verify) if p.verify else run(p.out,p.prior_zip,p.quick,p.updates),sort_keys=True))

"""RECOVERY-COST-SWEEP-001: class-cost tradeoff vs actual recovery arrival.

Synthetic DRIV3R World only. TRAIN motifs and developmental motifs are disjoint;
structural HELDOUT never touched. Full host menus remain, including synthetic
RELOCATE_START; task-specific STOP is not a general game recommendation.
Recovery caps may be changed only in this synthetic fixture, not native CyberSand.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, resource, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from . import cap_conditioning_lab as cap, budget_calibration_lab as prior
from . import history_alias_lab as alias, policy
from .world import digest, replay
from .train import sha_file

VERSION='recovery-cost-sweep/1'
SEEDS=(941,942)
ARMS={'standard':1,'recover2':2,'recover4':4}
STAGES=(36,72,128)
UPDATES=96
CAPS=(1,3,5)
CONTINUATION_LIMIT=7
SOURCE_FILES=('recovery_cost_sweep_lab.py','test_recovery_cost_sweep_lab.py','recovery_cost_sweep_gallery.py',
              'cap_conditioning_lab.py','budget_calibration_lab.py','history_alias_lab.py',
              'history_destination_lab.py','goal_recovery_lab.py','goal_conditioning_lab.py',
              'world.py','policy.py','train.py')


def resampled(stage, multiplier):
    """Same examples/masks/labels, only BACKTRACK sampling frequency differs."""
    X,M,Y,metadata=cap.corpus(stage,'paired')
    menu=prior.cases('train')[0]['state'].menu(12)
    back=menu.index('BACKTRACK')
    idx=np.concatenate([np.arange(len(Y))]+[np.flatnonzero(Y==back) for _ in range(multiplier-1)])
    assert len(idx)==len(Y)+(multiplier-1)*int(np.sum(Y==back))
    return X[idx], M[idx],Y[idx],{'original_n':len(Y),'resampled_n':len(idx),
            'backtrack_original':int(np.sum(Y==back)),'backtrack_weight':multiplier,
            'train_digest':metadata['train_state_cap_digest'],'seed_split':'train'}


def fit():
    models={};logs=[];start=time.perf_counter()
    for seed in SEEDS:
        models[seed]={arm:alias.Ranker(seed) for arm in ARMS}
        for stage in STAGES:
            for arm,weight in ARMS.items():
                X,M,Y,meta=resampled(stage,weight)
                training=models[seed][arm].fit(X,M,Y,UPDATES,seed*1013+stage)
                logs.append({'seed':seed,'stage':stage,'arm':arm,'corpus':meta,'optimizer':training})
    return models,logs,round(time.perf_counter()-start,3)


def probabilities(case, remaining, model):
    menu,x,cap_,oracle=cap.counterfactual(case,remaining,'paired')
    before=case['state'].state_sha()
    t=time.perf_counter_ns();p=model.probs(x,len(menu));elapsed=(time.perf_counter_ns()-t)/1e6
    if case['state'].state_sha()!=before:raise AssertionError('mutation on prediction')
    if abs(float(p.sum())-1)>1e-8 or not np.isfinite(p).all():raise ValueError('bad probabilities')
    return menu,p,oracle,elapsed


def first_actions(models):
    rows=[];lat=[]
    for case in prior.cases('development'):
        h=case['state'].state_sha()
        # Build the same fixed state/cap observations exactly once for all
        # model seeds and arms; no state or menu modifications between arms.
        for allowance in CAPS:
            menu,x,cap_,oracle=cap.counterfactual(case,allowance,'paired')
            for seed,arms in models.items():
                for arm,m in arms.items():
                    t=time.perf_counter_ns();p=m.probs(x,len(menu));ms=(time.perf_counter_ns()-t)/1e6
                    if abs(float(p.sum())-1)>1e-8 or not np.isfinite(p).all():raise ValueError('invalid neural probabilities')
                    lat.append(ms);action=menu[int(p.argmax())]
                    rows.append({'case':case['case'],'family':case['family'],'rot':case['rot'],
                        'kind':case['kind'],'original_budget':case['budget'],'depth':case['depth'],
                        'state_sha':h,'seed':seed,'arm':arm,'remaining':allowance,
                        'goal':list(case['goal']),'max_recoveries':cap_,
                        'legal':menu,'p':[round(float(v),9) for v in p],
                        'chosen':action,'oracle':oracle,'correct':action==oracle})
        if h!=case['state'].state_sha():raise AssertionError('development state mutation')
    return rows,lat


class CapMenu:
    """Read-only feature adapter for synthetic World.menu(cap) presentation."""
    def __init__(self,world,maximum):self.world=world;self.maximum=maximum
    def menu(self,_default=12):return self.world.menu(self.maximum)
    def __getattr__(self,name):return getattr(self.world,name)

def continued(case, allowance, model, seed, arm, capture=False):
    """Actual synthetic world transitions, never a fabricated counterfactual terminal score."""
    w=copy.deepcopy(case['state']);initial=w.state_sha();cap_=w.recoveries+allowance
    steps=[];hit=w.cursor==case['goal'];actions=Counter()
    before_steps=w.steps;before_moves=w.logical_moves
    before_ink=set(w.ink)
    for i in range(CONTINUATION_LIMIT):
        if hit or w.stopped:break
        # Host-legal menu adapter, strictly read-only: the original feature
        # constructors always call menu(12). A narrow proxy supplies the
        # cap-specific offered menu without altering physics, state or scores.
        proxy=CapMenu(w,cap_)
        menu,x=prior.matrix(proxy,case['goal'],'calibrated')
        if menu!=w.menu(cap_):raise AssertionError('offered menu mismatch')
        x=x.copy()
        remain=cap_-w.recoveries
        x[:len(menu),18]=w.recoveries/cap_
        x[:len(menu),30]=remain/cap_
        x[:len(menu),34]=(remain-(len(w.path)-1))/cap_
        x[:len(menu),alias.WIDTH-3]=remain/cap_
        p=model.probs(x,len(menu))
        if not np.isfinite(p).all() or abs(float(p.sum())-1)>1e-8:raise ValueError('bad probability vector')
        action=menu[int(p.argmax())]
        receipt=w.apply(action,cap_);actions[action]+=1
        if capture:steps.append({'legal':menu,'p':[round(float(t),9) for t in p],
                  'chosen':action,'receipt':receipt,'cursor':list(w.cursor)})
        hit=hit or w.cursor==case['goal']
    expected='BACKTRACK' if case['depth']<=allowance else 'STOP'
    return {'case':case['case'],'family':case['family'],'kind':case['kind'],
        'rot':case['rot'],'original_budget':case['budget'],'depth':case['depth'],
        'goal':list(case['goal']),'seed':seed,'arm':arm,'remaining':allowance,
        'oracle_first':expected,'initial_sha':initial,'end_sha':w.state_sha(),
        'steps':w.steps-before_steps,'logical_moves':w.logical_moves-before_moves,
        'hit':hit,'stopped':w.stopped,'recoveries_spent':w.recoveries-case['state'].recoveries,
        'new_ink':len(w.ink-before_ink),'actions':dict(sorted(actions.items())),
        'trajectory_sha':digest(w.decisions[len(case['state'].decisions):]),'rows':steps,
        'start_path':[list(p) for p in case['state'].path] if capture else None,
        'end_path':[list(p) for p in w.path] if capture else None}


def run_continuations(models):
    records=[];examples=[]
    # Predeclared systematic paired-position sample, both alias histories,
    # no post-outcome cherry-picking. All decisions separately censused.
    for ci,case in enumerate(prior.cases('development')):
        if (ci//2)%10!=0:continue
        for seed,arms in models.items():
            for arm,m in arms.items():
                for allowance in CAPS:
                    capture=((ci//2)%40==0 and seed==SEEDS[0])
                    row=continued(case,allowance,m,seed,arm,capture)
                    if capture:examples.append(row)
                    summary={k:v for k,v in row.items() if k not in ('rows','start_path','end_path')}
                    records.append(summary)
    return records,examples


def summarise(first,continuations):
    tables={}
    for arm in ARMS:
        a=[x for x in first if x['arm']==arm]
        b=[x for x in continuations if x['arm']==arm]
        reachable=[x for x in b if x['oracle_first']=='BACKTRACK']
        unreachable=[x for x in b if x['oracle_first']=='STOP']
        tables[arm]={'first_n':len(a),'correct_first':sum(x['correct'] for x in a),
            'false_retreat':sum(x['chosen']=='BACKTRACK' and x['oracle']=='STOP' for x in a),
            'missed_recoverable':sum(x['chosen']=='STOP' and x['oracle']=='BACKTRACK' for x in a),
            'other_initial':sum(x['chosen'] not in ('STOP','BACKTRACK') for x in a),
            'reachable_n':len(reachable),'reachable_hits':sum(x['hit'] for x in reachable),
            'unreachable_n':len(unreachable),'unreachable_terminated':sum(x['stopped'] for x in unreachable),
            'unreachable_recoveries_spent':sum(x['recoveries_spent'] for x in unreachable),
            'all_horizon_actions':sum(x['steps'] for x in b),
            'recovery_actions':sum(x['recoveries_spent'] for x in b),
            'relocations':sum(x['actions'].get('RELOCATE_START',0) for x in b),
            'by_cap':{str(v):{'n':sum(x['remaining']==v for x in a),
                'correct':sum(x['correct'] for x in a if x['remaining']==v),
                'reachable_n':sum(x['remaining']==v and x['oracle_first']=='BACKTRACK' for x in b),
                'reachable_hits':sum(x['remaining']==v and x['oracle_first']=='BACKTRACK' and x['hit'] for x in b)}
                for v in CAPS}}
    return tables


def motif_bootstrap(continuations):
    # Unit of independence is motif, not its rotations, budgets or seeds.
    groups=defaultdict(dict)
    for r in continuations:
        k=(r['case'],r['kind'],r['rot'],r['original_budget'],r['seed'],r['remaining'])
        groups[k][r['arm']]=int(r['hit'] and r['oracle_first']=='BACKTRACK')
    rng=np.random.default_rng(20261118);out={}
    for arm in ('recover2','recover4'):
        bymotif=defaultdict(list)
        for k,v in groups.items():
            assert set(v)==set(ARMS)
            bymotif[k[0]].append(v[arm]-v['standard'])
        array=np.array([np.mean(xs) for _,xs in sorted(bymotif.items())]);s=rng.choice(array,size=(4000,len(array)),replace=True).mean(axis=1)
        out[arm+'_minus_standard_reachable_arrivals']={
            'delta_unconditional':round(float(array.mean()),6),
            'motif_cluster_95':list(map(lambda z:round(float(z),6),np.quantile(s,[.025,.975]))),
            'motifs':len(array)}
    return out


def run(out):
    out=Path(out)
    if out.exists():raise FileExistsError('output must be new')
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    started=time.perf_counter();models,logs,train_s=fit()
    for seed,arms in models.items():
        for arm,m in arms.items():m.save(out/'checkpoints'/f'{seed}-{arm}.npz')
    first,lat=first_actions(models)
    continue_rows,samples=run_continuations(models)
    (out/'first_actions.json').write_text(json.dumps(first,sort_keys=True,separators=(',',':'))+'\n')
    (out/'continuations.json').write_text(json.dumps(continue_rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'trajectory_samples.json').write_text(json.dumps(samples,sort_keys=True,separators=(',',':'))+'\n')
    report={'experiment':'RECOVERY-COST-SWEEP-001','version':VERSION,'trained_seeds':list(SEEDS),
        'training_stages':list(STAGES),'updates_per_stage':UPDATES,'training_caps':list(cap.TRAIN_CAPS),
        'development_caps':list(CAPS),'cost_resampling':ARMS,'source_head':'df0ae48772e62ae302b515b57937c629f6c0292f',
        'train_cases':len(prior.cases('train')),'development_cases':len(prior.cases('development')),
        'source_sha256':{f:sha_file(Path(__file__).parent/f) for f in SOURCE_FILES},
        'checkpoints_sha256':{f.name:sha_file(f) for f in sorted((out/'checkpoints').glob('*.npz'))},
        'optimizer_logs':logs,'summary':summarise(first,continue_rows),
        'motif_bootstrap':motif_bootstrap(continue_rows),
        'first_action_probability_vectors':len(first), 'sampled_dev_continuations':len(continue_rows),
        'sampled_full_receipts':len(samples),'sampled_receipt_actions':sum(len(s['rows']) for s in samples),
        'train_wall_s':train_s,'wall_s':round(time.perf_counter()-started,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'median_model_only_ms':round(statistics.median(lat),6),
        'p95_model_only_ms':round(float(np.quantile(lat,.95)),6),
        'heldout_opened':False,'native_execution':False,'paid_spend':False,
        'limitations':['Class resampling changes supervised first-action emphasis, not an end-to-end optimizer.',
         'Identical synthetic histories and variable cap are not established native CyberSand capabilities.',
         'Synthetic STOP is task-specific and must not be extrapolated to general navigation.',
         'Seven-action continuations do not prove 72/128 sequential play.',
         'RELOCATE_START remains on the legal synthetic menu and its effects are counted.',
         'Structural HELDOUT remains unopened; checkpoint binaries retained outside Git history.']}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'},indent=2,sort_keys=True)+'\n')
    return report


def verify(out):
    out=Path(out);report=json.loads((out/'report.json').read_text())
    sums=json.loads((out/'SHA256SUMS.json').read_text())
    assert sums=={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    assert report['source_sha256']=={f:sha_file(Path(__file__).parent/f) for f in SOURCE_FILES}
    assert report['checkpoints_sha256']=={f.name:sha_file(f) for f in sorted((out/'checkpoints').glob('*.npz'))}
    models={seed:{arm:alias.Ranker.load(out/'checkpoints'/f'{seed}-{arm}.npz') for arm in ARMS} for seed in SEEDS}
    first,_=first_actions(models)
    assert json.loads((out/'first_actions.json').read_text())==first
    cont,_=run_continuations(models)
    assert json.loads((out/'continuations.json').read_text())==cont
    assert report['summary']==summarise(first,cont)
    assert report['motif_bootstrap']==motif_bootstrap(cont)
    smpl=json.loads((out/'trajectory_samples.json').read_text())
    lookup={(c['case'],c['rot'],c['kind'],c['budget']):c for c in prior.cases('development')}
    for s in smpl:
        c=lookup[(s['case'],s['rot'],s['kind'],s['original_budget'])]
        w=replay(c['spec'],c['state'].decisions)
        assert w.state_sha()==s['initial_sha']
        limit=w.recoveries+s['remaining']
        for step in s['rows']:
            assert w.menu(limit)==step['legal']
            assert w.apply(step['chosen'],limit)==step['receipt']
        assert w.state_sha()==s['end_sha']
    return {'verified_first_vectors':len(first),'verified_continuations':len(cont),
            'replayed_sampled_action_receipts':sum(len(s['rows']) for s in smpl)}


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','verify'));p.add_argument('--out',required=True)
    args=p.parse_args();print(json.dumps(run(args.out) if args.mode=='run' else verify(args.out),sort_keys=True,default=str))

if __name__=='__main__':main()

"""CONTACT-DENSITY-001: contact/recovery factorial, fixed-budget density control.

TRAIN-only counterfactual teaching; previously-unused DEVELOPMENT seed geometries.
Source-faithful only to DRIV3R synthetic-host fixture, NOT native CyberSand.
No sealed family inspection, API dispatch, production write, or paid GPU.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import delayed_target_lab as delayed
from . import goal_recovery_lab as host
from . import goal_retarget_lab as fixture
from .world import Spec, TRAIN_FAMILIES, World, MODES
from .train import sha_file

VERSION = 'contact-density/1'
STAGES = (36, 72, 128)
ARMS = ('frozen', 'imitation', 'proxy_only', 'recovery_only',
        'contact_only', 'full', 'balanced_full')
FACTORS = {'proxy_only': (0., 0.), 'recovery_only': (0., 1.5),
           'contact_only': (22., 0.), 'full': (22., 1.5),
           'balanced_full': (22., 1.5)}
SOURCE_FILES = ('contact_density_lab.py', 'delayed_target_lab.py',
                'cost_retarget_lab.py', 'retarget_onpolicy_lab.py',
                'goal_retarget_lab.py', 'goal_recovery_lab.py',
                'goal_conditioning_lab.py', 'world.py', 'policy.py', 'train.py')
# Predeclared new DEVELOPMENT seeds; families overlap TRAIN by design.
# Neither these seed geometries nor their goal instruction schedules are a
# structural heldout. Keep original fixture.cases() out of this experiment.
DEV_SPECS = tuple(Spec('development', family, 3059 + 100*i,
                       MODES[i % len(MODES)])
                  for i, family in enumerate(TRAIN_FAMILIES))
EVAL_SEED = 681


def score(v: dict, contact_bonus: float, recovery_penalty: float) -> float:
    """Same 12-step teacher rollout, only terminal reward coefficients differ."""
    return (contact_bonus * int(v['reached_new_target'])
            + .25 * v['gain'] + .25 * v['painted']
            - recovery_penalty * v['recovery_spent']
            - 12 * int(v['stopped']))


def labels_from_audit(audit: list[dict], mask: np.ndarray):
    labels = {a: [] for a in ARMS if a != 'frozen'}
    positive=[];dist={a:Counter() for a in labels}
    disagreements=Counter()
    for i, row in enumerate(audit):
        menu=row['menu']
        assert menu and len(menu) == int(mask[i].sum())
        assert set(menu) == set(row['delayed_scores'])
        baseline=row['imitation']
        if baseline not in menu: raise ValueError('imitation not legal')
        is_positive=any(v['reached_new_target'] for v in row['delayed_scores'].values())
        positive.append(is_positive)
        labels['imitation'].append(menu.index(baseline))
        dist['imitation'][baseline]+=1
        for arm, (contact, recovery) in FACTORS.items():
            values={a:score(row['delayed_scores'][a],contact,recovery) for a in menu}
            peak=max(values.values())
            tied=[a for a in menu if abs(values[a]-peak)<1e-9]
            best=baseline if baseline in tied else sorted(tied)[0]
            if not mask[i,menu.index(best)]:raise ValueError('illegal counterfactual label')
            labels[arm].append(menu.index(best))
            dist[arm][best]+=1
            disagreements[arm]+=int(best!=baseline)
    return ({arm:np.array(ids,dtype=np.int64) for arm, ids in labels.items()},
            np.array(positive,dtype=bool),
            {'contact_positive':int(sum(positive)),
             'contact_negative':len(positive)-int(sum(positive)),
             'label_disagreement_with_imitation':dict(disagreements),
             'label_actions':{k:dict(sorted(v.items())) for k,v in dist.items()}})


def balance_indices(positive: np.ndarray, seed: int) -> np.ndarray:
    """Fixed total examples, 50:50 positive/negative resampling with replacement.

    Same optimizer updates but *not* same unique state count, deliberately.
    """
    pos=np.flatnonzero(positive);neg=np.flatnonzero(~positive)
    if not len(pos) or not len(neg):
        raise ValueError('cannot evaluate density ablation without both classes')
    rng=np.random.default_rng(seed)
    n=len(positive)
    ids=np.concatenate((rng.choice(pos,size=n//2,replace=True),
                        rng.choice(neg,size=n-n//2,replace=True)))
    rng.shuffle(ids)
    return ids


def cases() -> list[Spec]:
    return list(DEV_SPECS)+[replace(s,challenge='trap_corridor') for s in DEV_SPECS]


def train(seed: int, base_updates=60, adapt_updates=20, samples=64):
    if not(1<=base_updates<=100 and 1<=adapt_updates<=60 and 32<=samples<=128):
        raise ValueError('unbounded optimizer or corpus')
    base, base_history=host.train(seed,base_updates)
    frozen=base['goal_recovery']
    models={arm:copy.deepcopy(frozen) for arm in ARMS}
    stages=[]; ledgers=[]
    for h in STAGES:
        X,M,_,audit,meta=delayed.corpus(frozen,seed,h,samples)
        labels,positive,label_meta=labels_from_audit(audit,M)
        idx=balance_indices(positive,seed*173+h)
        fit={}
        for arm in ARMS[1:]:
            selected=idx if arm=='balanced_full' else np.arange(len(X))
            fit[arm]=models[arm].fit(X[selected],M[selected],labels[arm][selected],
                                      adapt_updates,seed*313+h*7)
        stages.append({'horizon':h,'corpus':meta,'labels':label_meta,
                       'balanced':{'examples':len(idx),'unique_states':len(set(map(int,idx))),
                                   'positive_fraction':float(np.mean(positive[idx]))},
                       'optimizer':fit})
        ledgers.append({'horizon':h,'candidate_scores':audit,
                        'balanced_sample_indices':idx.tolist(),
                        'factorial_labels':{k:v.tolist() for k,v in labels.items()},
                        'contact_eligible':positive.astype(int).tolist()})
    return models,{'initial_training':base_history,'adaptation':stages},ledgers


def evaluate(models: dict, quick=False, png_dir=None):
    specs=cases()[:1] if quick else cases()
    result=[]
    for sp in specs:
        for horizon in ((128,) if quick else (128,160)):
            for arm in ARMS:
                png=None
                if png_dir is not None and horizon==160 and sp in DEV_SPECS:
                    png=png_dir/f'{sp.family}-{arm}.png'
                ep=fixture.run_episode(sp,{'goal_recovery':models[arm]},
                                       'goal_recovery',EVAL_SEED,horizon,'retarget',png)
                fixture.verify_episode(ep,sp,{'goal_recovery':models[arm]})
                ep['trained_arm']=arm
                result.append(ep)
    return result


def summarize(episodes):
    output={}
    for arm in ARMS:
        rows=[ep for ep in episodes if ep['trained_arm']==arm]
        counts=Counter();[counts.update(ep['action_counts']) for ep in rows]
        switches=[v for ep in rows for v in ep['switch_windows'].values()]
        output[arm]={'episodes':len(rows),
                     'targets':sum(len(ep['reached_epochs']) for ep in rows),
                     'mean_ink':round(statistics.mean(ep['metrics']['unique_painted_cells'] for ep in rows),4),
                     'early_stops':sum(ep['metrics']['stopped_early'] for ep in rows),
                     'recoveries':sum(ep['metrics']['recoveries'] for ep in rows),
                     'post_switch_ink':sum(v['painted'] for v in switches),
                     'mean_switch_gain':round(statistics.mean(v['net_distance_gain'] for v in switches),4),
                     'mean_latency_ms':round(statistics.mean(ep['mean_latency_ms'] for ep in rows),6),
                     'actions':dict(sorted(counts.items()))}
    return output


def provenance():
    return {file:sha_file(Path(__file__).parent/file) for file in SOURCE_FILES}


def run(out: Path, seed=641, base_updates=60, adapt_updates=20,
        samples=64,quick=False):
    if out.exists():raise FileExistsError('evidence output must be immutable')
    out.mkdir(parents=True)
    t0=time.perf_counter()
    models,history,ledgers=train(seed,base_updates,adapt_updates,samples)
    checks={}
    for arm,m in models.items():
        path=out/f'{arm}.npz';m.save(path)
        restored=host.Ranker.load(path)
        if any(not np.array_equal(m.p[k],restored.p[k]) for k in m.p):
            raise ValueError('checkpoint round-trip')
        checks[arm]={'sha256':sha_file(path),'parameters':sum(p.size for p in m.p.values()),
                     'updates':m.updates}
    (out/'traces').mkdir();(out/'png').mkdir()
    episodes=evaluate(models,quick,out/'png')
    for i,ep in enumerate(episodes):
        (out/'traces'/f'{i:03d}-{ep["trained_arm"]}.json').write_text(
             json.dumps(ep,sort_keys=True,separators=(',',':'))+'\n')
    (out/'counterfactuals.json').write_text(json.dumps(ledgers,sort_keys=True)+'\n')
    result={'version':VERSION,'train_seed':seed,'eval_seed':EVAL_SEED,
            'stages':list(STAGES),'base_updates':base_updates,
            'adapt_updates_per_stage':adapt_updates,'samples_per_stage':samples,
            'history':history,'checkpoints':checks,
            'development':summarize(episodes),'episodes':len(episodes),
            'decisions':sum(len(ep['rows']) for ep in episodes),
            'source_sha256':provenance(),'development_specs':[sp.__dict__ for sp in cases()],
            'split':'development','sealed_heldout_opened':False,'native_receipts':0,
            'paid_compute':False,'elapsed_s':round(time.perf_counter()-t0,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f)
        for f in out.rglob('*') if f.is_file()},sort_keys=True,indent=2)+'\n')
    return result


def verify(out: Path):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=provenance():
        raise ValueError('version or source mismatch')
    expected=json.loads((out/'SHA256SUMS.json').read_text())
    observed={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*')
              if f.is_file() and f.name!='SHA256SUMS.json'}
    if observed!=expected:raise ValueError('evidence integrity mismatch')
    models={arm:host.Ranker.load(out/f'{arm}.npz') for arm in ARMS}
    for arm,m in models.items():
        if (sha_file(out/f'{arm}.npz')!=rep['checkpoints'][arm]['sha256']
            or m.updates!=rep['checkpoints'][arm]['updates']):
            raise ValueError('checkpoint hash or update mismatch')
    worlds={s.ident:s for s in cases()};episodes=[]
    for trace in sorted((out/'traces').glob('*.json')):
        ep=json.loads(trace.read_text())
        if ep['trained_arm'] not in ARMS:raise ValueError('unknown arm')
        fixture.verify_episode(ep,worlds[ep['spec']],
                               {'goal_recovery':models[ep['trained_arm']]})
        episodes.append(ep)
    if len(episodes)!=rep['episodes'] or sum(len(e['rows']) for e in episodes)!=rep['decisions']:
        raise ValueError('trajectory/decision count mismatch')
    result=summarize(episodes)
    for arm in ARMS:
        a=dict(result[arm]);b=dict(rep['development'][arm]);
        a.pop('mean_latency_ms');b.pop('mean_latency_ms')
        if a!=b:raise ValueError('aggregated outcome mismatch')
    for stage in json.loads((out/'counterfactuals.json').read_text()):
        eligible=np.asarray(stage['contact_eligible'],bool)
        ii=np.asarray(stage['balanced_sample_indices'],np.int64)
        if len(ii)!=len(eligible) or not np.all(ii>=0) or not np.all(ii<len(eligible)):
            raise ValueError('invalid resampling ledger')
        if not (0.45<=np.mean(eligible[ii])<=0.55):
            raise ValueError('density balance corrupted')
    return {'checkpoints':len(models),'episodes':len(episodes),
            'decisions':rep['decisions'],'file_hashes':len(observed)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=641)
    p.add_argument('--base-updates',type=int,default=60)
    p.add_argument('--adapt-updates',type=int,default=20)
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.seed,
        a.base_updates,a.adapt_updates,a.samples,a.quick),sort_keys=True))

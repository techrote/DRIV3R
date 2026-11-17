"""CONTRASTIVE-FORK-001: counterbalanced trapped/open failure discrimination.

Synthetic CyberSand-style host-legal one-cell worlds only. Not a native ABI
or proof of physical movement. No sealed heldout worlds or paid compute.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import copy
import gc
import json
from pathlib import Path
import resource
import statistics
import time

import numpy as np

from . import recovery_fork_lab as prior, goal_recovery_lab as host, goal_retarget_lab as retarget, policy
from .world import World, MOVE, Spec, specs
from .train import sha_file, _png

VERSION = 'contrastive-fork/1'
STAGES = (36,72,128)
ARMS = ('frozen','goal_imitation','universal_retreat','contrastive')
ADAPT = ARMS[1:]
DEV_SEED = 782
SOURCE = ('contrastive_fork_lab.py', 'recovery_fork_lab.py', 'goal_recovery_lab.py',
          'goal_conditioning_lab.py','goal_retarget_lab.py', 'world.py', 'policy.py','train.py')
ROTATION = (0,1,2,3)
VECTOR_TO_ACTION = {v:k for k,v in MOVE.items()}


def rotate_xy(p,turns):
    if turns not in ROTATION: raise ValueError('rotation')
    x,y=p[0]-48,p[1]-48
    for _ in range(turns):x,y=-y,x
    return (x+48,y+48)


def rotate_action(action,turns):
    if action not in MOVE:return action
    dx,dy=MOVE[action]
    p=rotate_xy((48+dx,48+dy),turns)
    return VECTOR_TO_ACTION[(p[0]-48,p[1]-48)]


def oriented_world(original,turns):
    """Reconstruct via transformed immutable terrain and entirely legal actions.

    Only synthetic terrain coordinate rotation occurs before the first move;
    runtime cursor, path, recoveries and ink are NEVER assigned directly.
    """
    w=World.create(original.spec)
    w.materials={rotate_xy(p,turns):v for p,v in w.materials.items()}
    w.protected={rotate_xy(p,turns) for p in w.protected}
    for e in original.decisions:
        w.apply(rotate_action(e['chosen'],turns),12)
    if (w.steps!=original.steps or w.recoveries!=original.recoveries
        or len(w.ink)!=len(original.ink) or w.cursor!=rotate_xy(original.cursor,turns)):
        raise ValueError('rotated legal replay')
    return w


@lru_cache(maxsize=128)
def base_pair(sp,seed,variant):
    trap,tc=prior.fork(sp,seed,variant)
    opening,oc=prior.open_fork(sp,seed,variant)
    return trap,tc,opening,oc


def pair(sp,seed,variant,turns):
    if sp.split not in ('train','development') or sp.challenge!='normal':
        raise ValueError('split/fixture')
    if turns not in ROTATION:raise ValueError('turns')
    trap,tc,opening,oc=base_pair(sp,seed,variant)
    # Geometry is rotated *before* replay, preserving every legal step.
    t=oriented_world(trap,turns)
    o=oriented_world(opening,turns)
    forward=rotate_action('MOVE_E',turns)
    retreat=rotate_action('MOVE_W',turns)
    target=rotate_xy(prior.east_goal(trap),turns)
    target_o=rotate_xy(prior.east_goal(opening),turns)
    if not ({forward,retreat}.issubset(t.menu(12)) and
            {forward,retreat}.issubset(o.menu(12))):raise ValueError('menu mismatch')
    risk=prior.viability(t,forward)
    escape=prior.viability(t,retreat)
    open_f=prior.viability(o,forward)
    open_b=prior.viability(o,retreat)
    # Do not silently substitute an unqualified geometry or labels.
    if not (risk['forced_stop'] and risk['moves']<=7 and
            not escape['forced_stop'] and escape['moves']>=20 and
            open_f['moves']>=20 and open_b['moves']>=20 and
            not open_f['forced_stop'] and not open_b['forced_stop']):
        raise ValueError('orientation invalidated legal viability witness')
    return [
        {'kind':'trap','world':t,'goal':target,'forward':forward,'retreat':retreat,
         'witness':{'forward':risk,'retreat':escape},'source':tc},
        {'kind':'open','world':o,'goal':target_o,'forward':forward,'retreat':retreat,
         'witness':{'forward':open_f,'retreat':open_b},'source':oc},
    ]


def train_corpus(seed,stage,natural_limit=64,variants=1):
    if stage not in STAGES or not 16<=natural_limit<=128 or variants not in (1,2):
        raise ValueError('bounded curriculum')
    natural,masks,labels,prior_meta=host.corpus(seed,stage)
    rng=np.random.default_rng(seed*13+stage)
    idx=np.sort(rng.choice(len(labels),size=min(natural_limit,len(labels)),replace=False))
    X=list(natural[idx]);M=list(masks[idx]);ys={a:list(labels[idx]) for a in ADAPT}
    cases=[]
    for sp in specs('train',1):
        for variant in range(variants):
            for turns in ROTATION:
                for case in pair(sp,seed,variant,turns):
                    w=case['world'];menu,x=host.matrix(w,case['goal'],stage,False)
                    if len(menu)>policy.ACTION_MAX:raise ValueError('menu width')
                    mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                    X.append(x);M.append(mask)
                    choices={'goal_imitation':case['forward'],
                             'universal_retreat':case['retreat'],
                             'contrastive':case['retreat'] if case['kind']=='trap' else case['forward']}
                    for a in ADAPT:
                        if choices[a] not in menu:raise ValueError('illegal label')
                        ys[a].append(menu.index(choices[a]))
                    cases.append({'split':'train','spec':w.spec.__dict__,'seed':seed,
                                  'variant':variant,'turns':turns,'stage':stage,
                                  'kind':case['kind'],'state_sha256':w.state_sha(),
                                  'goal':list(case['goal']),'menu':menu,
                                  'base_sha256':w.base_sha(),
                                  'prefix_actions':[rotate_action(a,turns) for a in case['source']['prefix_actions']],
                                  'witness':case['witness'],'labels':choices})
    return (np.stack(X),np.stack(M),{a:np.array(ys[a],dtype=np.int64) for a in ADAPT},cases,
            {'natural_examples':len(idx),'paired_states':len(cases),'pairs':len(cases)//2,
             'trap_states':sum(c['kind']=='trap' for c in cases),
             'open_states':sum(c['kind']=='open' for c in cases),
             'source_corpus':prior_meta,'split':'train'})


def train(seed,base_updates=18,adapt_updates=32,natural_limit=64,variants=1):
    if not 1<=base_updates<=80 or not 1<=adapt_updates<=100:
        raise ValueError('optimizer budget')
    baseline,history=host.train(seed,base_updates)
    frozen=baseline['goal_recovery']
    models={a:copy.deepcopy(frozen) for a in ARMS}
    stats=[];ledger=[]
    for stage in STAGES:
        X,M,Y,pairs,meta=train_corpus(seed,stage,natural_limit,variants)
        trained={a:models[a].fit(X,M,Y[a],adapt_updates,seed*349+stage) for a in ADAPT}
        stats.append({'stage':stage,'corpus':meta,'optimizers':trained});ledger.extend(pairs)
    return models,{'baseline':history,'adaptation':stats},ledger


def reconstruct(row):
    if row['split'] not in ('train','development') or row['spec']['split']!=row['split']:
        raise ValueError('wrong split')
    sp=Spec(**row['spec'])
    if ((row['kind']=='trap') != (sp.challenge=='trap_corridor')):
        raise ValueError('challenge mismatch')
    w=World.create(sp)
    turns=row['turns']
    w.materials={rotate_xy(p,turns):v for p,v in w.materials.items()}
    w.protected={rotate_xy(p,turns) for p in w.protected}
    for action in row['prefix_actions']:
        w.apply(action,12)
    if (w.base_sha()!=row['base_sha256'] or w.state_sha()!=row['state_sha256'] or
        row['menu']!=w.menu(12)):
        raise ValueError('source replay mismatch')
    return w


def reconstructed_goal(row):
    """Derive the original goal from a legally replayed unrotated prefix."""
    sp=Spec(**row['spec'])
    raw=World.create(sp)
    for a in row['prefix_actions']:
        raw.apply(rotate_action(a,(4-row['turns'])%4),12)
    return rotate_xy(prior.east_goal(raw),row['turns'])


def verify_train_case(row):
    w=reconstruct(row)
    forward=rotate_action('MOVE_E',row['turns'])
    retreat=rotate_action('MOVE_W',row['turns'])
    if row['goal']!=list(reconstructed_goal(row)):
        raise ValueError('goal mismatch')
    if (row['labels']!={'goal_imitation':forward,'universal_retreat':retreat,
             'contrastive':retreat if row['kind']=='trap' else forward}):
        raise ValueError('labels')
    if row['witness']!={'forward':prior.viability(w,forward),'retreat':prior.viability(w,retreat)}:
        raise ValueError('counterfactual witness')
    return True


def episode(sp,model,arm,seed,variant,turns,kind,horizon=128,png=None):
    if sp.split!='development' or kind not in ('trap','open') or arm not in ARMS or horizon not in (72,128):
        raise ValueError('evaluation split')
    c=pair(sp,seed,variant,turns)[0 if kind=='trap' else 1]
    w=c['world'];goal=c['goal'];first_sha=w.state_sha();first_base=w.base_sha();first_menu=w.menu(12);prefix=[e['chosen'] for e in w.decisions]; ink_before=len(w.ink)
    rows=[];lat=[]
    for _ in range(horizon):
        if w.stopped:break
        t=time.perf_counter_ns()
        action,prob=model.choose(w,goal,horizon,False)
        lat.append((time.perf_counter_ns()-t)/1e6)
        rows.append({'requested':action,'probabilities':prob,'event':w.apply(action,12)})
    if png is not None:_png(png,w)
    return {'version':VERSION,'split':'development','spec':w.spec.__dict__,'seed':seed,
            'variant':variant,'turns':turns,'kind':kind,'horizon':horizon,'arm':arm,
            'goal':list(goal),'menu':first_menu,'first_state_sha256':first_sha,'base_sha256':first_base,
            'prefix_actions':prefix,'certificate':c['witness'],
            'forward':c['forward'],'retreat':c['retreat'],
            'first_action':rows[0]['requested'],'rows':rows,
            'paint_after_fork':len(w.ink)-ink_before,'final_state_sha256':w.state_sha(),
            'metrics':w.metrics(horizon+len(prefix)),
            'latency_ms':round(statistics.mean(lat),6) if lat else 0.}


def verify_episode(ep,model):
    if ep['version']!=VERSION or ep['split']!='development':
        raise ValueError('version/split')
    witness_row={'split':'development','spec':ep['spec'],'kind':ep['kind'],
                 'turns':ep['turns'],'prefix_actions':ep['prefix_actions'],
                 'base_sha256':ep['base_sha256'],'state_sha256':ep['first_state_sha256'],
                 'menu':ep['menu']}
    w=reconstruct(witness_row)
    forward=rotate_action('MOVE_E',ep['turns'])
    retreat=rotate_action('MOVE_W',ep['turns'])
    if (ep['certificate']!={'forward':prior.viability(w,forward),'retreat':prior.viability(w,retreat)} or
        ep['goal']!=list(reconstructed_goal(witness_row)) or
        ep['forward']!=forward or ep['retreat']!=retreat):
        raise ValueError('source fixture')
    ink_before=len(w.ink);prefix_size=len(w.decisions)
    for row in ep['rows']:
        asked,prob=model.choose(w,tuple(ep['goal']),ep['horizon'],False)
        if (row['requested']!=asked or row['probabilities']!=prob or
            row['event']!=w.apply(asked,12)):
            raise ValueError('neural/host replay')
    if (not ep['rows'] or ep['first_action']!=ep['rows'][0]['requested'] or
        ep['final_state_sha256']!=w.state_sha() or ep['paint_after_fork']!=len(w.ink)-ink_before or
        ep['metrics']!=w.metrics(ep['horizon']+prefix_size) or
        (not w.stopped and len(ep['rows'])!=ep['horizon'])):
        raise ValueError('trajectory summary')
    return len(ep['rows'])


def summarize(episodes):
    out={}
    for kind in ('trap','open'):
        for arm in ARMS:
            cases=[e for e in episodes if e['kind']==kind and e['arm']==arm]
            acts=Counter(r['event']['chosen'] for e in cases for r in e['rows'])
            out[f'{kind}/{arm}']={
                'episodes':len(cases),
                'certified_first':sum(e['first_action']==(e['retreat'] if kind=='trap' else e['forward']) for e in cases),
                'unnecessary_retreat':sum(e['first_action']==e['retreat'] for e in cases) if kind=='open' else None,
                'wrong_forward':sum(e['first_action']==e['forward'] for e in cases) if kind=='trap' else None,
                'premature_stop':sum(len(e['rows'])<e['horizon'] for e in cases),
                'mean_paint':round(statistics.mean(e['paint_after_fork'] for e in cases),3),
                'mean_decisions':round(statistics.mean(len(e['rows']) for e in cases),3),
                'mean_latency_ms':round(statistics.mean(e['latency_ms'] for e in cases),6),
                'actions':dict(sorted(acts.items()))}
    return out


def provenance():return {name:sha_file(Path(__file__).parent/name) for name in SOURCE}


def run(out,seed=691,base_updates=18,adapt_updates=32,natural_limit=64,
        variants=1,quick=False):
    if out.exists():raise FileExistsError('immutable experiment output')
    out.mkdir(parents=True);started=time.perf_counter();gc.collect();old_gc=gc.isenabled()
    try:
        if old_gc:gc.disable()
        models,history,ledger=train(seed,base_updates,adapt_updates,natural_limit,variants)
        checkpoints={}
        for arm,model in models.items():
            path=out/(arm+'.npz');model.save(path);restored=host.Ranker.load(path)
            if any(not np.array_equal(model.p[k],restored.p[k]) for k in model.p):
                raise ValueError('checkpoint roundtrip')
            checkpoints[arm]={'sha256':sha_file(path),'parameters':sum(v.size for v in model.p.values()),
                              'updates':model.updates}
        models={a:host.Ranker.load(out/(a+'.npz')) for a in ARMS}
        (out/'traces').mkdir();(out/'png').mkdir()
        eps=[]
        for sp in specs('development',1)[:1 if quick else 4]:
            for turns in ((0,) if quick else ROTATION):
                for kind in ('trap','open'):
                    for arm in ARMS:
                        name=f'{sp.family}-{sp.seed}-{turns}-{kind}-{arm}'
                        ep=episode(sp,models[arm],arm,DEV_SEED,0,turns,kind,72 if quick else 128,
                                   out/'png'/(name+'.png') if turns==0 and not quick else None)
                        verify_episode(ep,models[arm]);eps.append(ep)
                        (out/'traces'/(name+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
    finally:
        if old_gc:gc.enable()
    for row in ledger:verify_train_case(row)
    (out/'training_cases.json').write_text(json.dumps(ledger,sort_keys=True)+'\n')
    rep={'version':VERSION,'seed':seed,'curriculum':STAGES,'base_updates':base_updates,
         'adapt_updates':adapt_updates,'natural_limit':natural_limit,'variants':variants,
         'checkpoint':checkpoints,'history':history,'metrics':summarize(eps),
         'episodes':len(eps),'decisions':sum(len(e['rows']) for e in eps),
         'training_states':len(ledger),'source_sha256':provenance(),
         'synthetic_only':True,'sealed_heldout_opened':False,'native_receipts':0,
         'paid_compute':False,'elapsed_s':round(time.perf_counter()-started,3),
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
        for p in out.rglob('*') if p.is_file()},indent=2,sort_keys=True)+'\n')
    return rep


def verify(out):
    rep=json.loads((out/'report.json').read_text());hashes=json.loads((out/'SHA256SUMS.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=provenance():raise ValueError('source mismatch')
    if hashes!={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*')
               if p.is_file() and p.name!='SHA256SUMS.json'}:raise ValueError('hash mismatch')
    models={a:host.Ranker.load(out/(a+'.npz')) for a in ARMS}
    for a,m in models.items():
        if sha_file(out/(a+'.npz'))!=rep['checkpoint'][a]['sha256'] or m.updates!=rep['checkpoint'][a]['updates']:
            raise ValueError('checkpoint mismatch')
    ledger=json.loads((out/'training_cases.json').read_text())
    for row in ledger:verify_train_case(row)
    eps=[]
    for p in sorted((out/'traces').glob('*.json')):
        ep=json.loads(p.read_text());verify_episode(ep,models[ep['arm']]);eps.append(ep)
    if (len(ledger)!=rep['training_states'] or len(eps)!=rep['episodes'] or
        sum(len(e['rows']) for e in eps)!=rep['decisions']):raise ValueError('episode count')
    summed=summarize(eps)
    for a in summed:
        # Wall-clock latency is recorded but intentionally not replay-deterministic.
        summed[a].pop('mean_latency_ms');rep['metrics'][a].pop('mean_latency_ms')
    if summed!=rep['metrics']:raise ValueError('summary mismatch')
    return {'checkpoints':len(models),'training_states':len(ledger),
            'episodes':len(eps),'decisions':sum(len(e['rows']) for e in eps),
            'hashes':len(hashes)}


if __name__ == '__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--out',type=Path)
    cli.add_argument('--verify',type=Path)
    cli.add_argument('--seed',type=int,default=691)
    cli.add_argument('--base-updates',type=int,default=18)
    cli.add_argument('--adapt-updates',type=int,default=32)
    cli.add_argument('--natural-limit',type=int,default=64)
    cli.add_argument('--variants',type=int,default=1)
    cli.add_argument('--quick',action='store_true')
    args=cli.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,
        args.base_updates,args.adapt_updates,args.natural_limit,args.variants,args.quick),sort_keys=True))

"""MENU-ALIAS-001: equal-menu recoverable traps vs same-observation open routes.

Synthetic CyberSand-style research fixture ONLY; not native host ABI validation.
Never inspect HELDOUT, mutate historical d_common, or modify the game.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gc
import json
from functools import lru_cache
from pathlib import Path
import resource
import statistics
import time
import numpy as np

from . import contrastive_fork_lab as rotated, recovery_fork_lab as forks
from . import goal_recovery_lab as host, policy
from .world import World, Spec, SOLID, specs
from .train import sha_file, _png

VERSION = 'menu-alias/1'
STAGES = (36, 72, 128)
ARMS = ('frozen', 'goal_forward', 'universal_retreat', 'visible_contrastive')
ADAPT = ARMS[1:]
KINDS = ('near_trap', 'far_trap', 'open')
DEV_SEED = 783
EVAL_ROTATIONS = (0, 1)  # predeclared, orthogonal; budget-bounded development evaluation
SOURCE = ('menu_alias_lab.py', 'contrastive_fork_lab.py', 'recovery_fork_lab.py',
          'goal_recovery_lab.py', 'goal_conditioning_lab.py',
          'goal_retarget_lab.py', 'world.py', 'policy.py', 'train.py')


@lru_cache(maxsize=128)
def source_fork(sp, seed, variant):
    return forks.fork(sp, seed, variant)


def construct(sp, seed, variant, turns, kind):
    """Alter ONLY static terrain before an exact legal replay of the same prefix.

    near_trap has a near obstruction visible to current lookahead. far_trap
    and open have the SAME host menu and SAME policy features at the fork,
    but different distant material. They intentionally create perceptual aliasing.
    """
    if sp.split not in ('train', 'development') or sp.challenge != 'normal':
        raise ValueError('no sealed HELDOUT or modified challenge specification')
    if kind not in KINDS or turns not in rotated.ROTATION or variant not in (0, 1):
        raise ValueError('invalid bounded fixture')
    base, cert = source_fork(sp, seed, variant)
    w = World.create(base.spec)
    if kind == 'near_trap':
        w.materials[(53, 48)] = SOLID
    if kind == 'open':
        w.materials.pop((55, 48), None)
        w.protected.discard((55, 48))
    w.materials = {rotated.rotate_xy(p, turns): v for p, v in w.materials.items()}
    w.protected = {rotated.rotate_xy(p, turns) for p in w.protected}
    prefix = [rotated.rotate_action(a, turns) for a in cert['prefix_actions']]
    for action in prefix:
        w.apply(action, 12)
    if (w.steps != base.steps or w.recoveries != 12 or
        w.cursor != rotated.rotate_xy(base.cursor, turns)):
        raise ValueError('prefix reconstruction mismatch')
    forward = rotated.rotate_action('MOVE_E', turns)
    retreat = rotated.rotate_action('MOVE_W', turns)
    goal = rotated.rotate_xy(forks.east_goal(base), turns)
    # Same exact first menu for all three materials, never a fabricated menu.
    if w.menu(12) != sorted((forward, retreat, 'STOP')):
        raise ValueError('unequal or illegal first action menu')
    f = forks.viability(w, forward)
    b = forks.viability(w, retreat)
    if not (b['moves'] >= 20 and not b['forced_stop'] and
            ((f['moves'] <= 7 and f['forced_stop']) if kind != 'open' else
             (f['moves'] >= 20))):
        raise ValueError('counterfactual geometry fails viability witness')
    return dict(world=w, prefix=prefix, forward=forward, retreat=retreat,
                goal=goal, witness={'forward': f, 'retreat': b},
                spec=w.spec.__dict__, turns=turns, seed=seed, variant=variant,
                kind=kind)


def matched(sp, seed, variant, turns, horizon=128):
    cases = {kind: construct(sp, seed, variant, turns, kind) for kind in KINDS}
    menus = {}
    features = {}
    for kind, row in cases.items():
        menus[kind], features[kind] = host.matrix(row['world'], row['goal'], horizon, False)
    if not all(m == menus['open'] for m in menus.values()):
        raise ValueError('menu cardinality mismatch')
    if not np.array_equal(features['far_trap'], features['open']):
        raise ValueError('far/open not a strict observation alias')
    if np.array_equal(features['near_trap'], features['open']):
        raise ValueError('near obstruction not observably different')
    return cases, features


def corpus(seed, stage, natural_limit=64, variants=1):
    natural, masks, labels, prior = host.corpus(seed, stage)
    rng = np.random.default_rng(seed * 19 + stage)
    indices = np.sort(rng.choice(len(labels), size=min(natural_limit, len(labels)), replace=False))
    X = list(natural[indices]); M = list(masks[indices]); Y = {a:list(labels[indices]) for a in ADAPT}
    ledger = []
    for sp in specs('train', 1):
        for variant in range(variants):
            for turns in rotated.ROTATION:
                cases, features = matched(sp, seed, variant, turns, stage)
                for kind in ('near_trap', 'open'):
                    row = cases[kind]; menu = row['world'].menu(12)
                    mask = np.zeros(policy.ACTION_MAX, bool); mask[:len(menu)] = True
                    X.append(features[kind]); M.append(mask)
                    choices = {'goal_forward':row['forward'], 'universal_retreat':row['retreat'],
                               'visible_contrastive':row['retreat'] if kind == 'near_trap' else row['forward']}
                    for arm in ADAPT: Y[arm].append(menu.index(choices[arm]))
                    ledger.append(dict(split='train', spec=row['spec'], seed=seed,
                                       variant=variant, turns=turns, kind=kind, stage=stage,
                                       prefix=row['prefix'], goal=list(row['goal']), menu=menu,
                                       base_sha256=row['world'].base_sha(), state_sha256=row['world'].state_sha(),
                                       witness=row['witness'], labels=choices))
    return np.stack(X), np.stack(M), {a:np.asarray(Y[a], np.int64) for a in ADAPT}, ledger, {
        'natural':len(indices), 'supplemental':len(ledger), 'paired':len(ledger)//2,
        'train_only':True, 'natural_source':prior}


def train(seed, base_updates=18, adapt_updates=32, natural_limit=64, variants=1):
    if not 1<=base_updates<=80 or not 1<=adapt_updates<=100 or not 16<=natural_limit<=128:
        raise ValueError('unbounded optimizer')
    baseline, history = host.train(seed, base_updates)
    models = {a:copy.deepcopy(baseline['goal_recovery']) for a in ARMS}
    stats = []; ledger = []
    for stage in STAGES:
        X,M,Y,rows,meta = corpus(seed,stage,natural_limit,variants)
        fits = {a:models[a].fit(X,M,Y[a],adapt_updates,seed*709+stage) for a in ADAPT}
        stats.append(dict(stage=stage, fit=fits, corpus=meta))
        ledger.extend(rows)
    return models,dict(baseline=history,adaptation=stats),ledger


def verify_train(row):
    if row['split'] != 'train' or row['kind'] not in ('near_trap','open'):
        raise ValueError('train ledger split/kind')
    sp0 = Spec(**row['spec']); sp = Spec(sp0.split,sp0.family,sp0.seed,sp0.mode,'normal')
    q = construct(sp,row['seed'],row['variant'],row['turns'],row['kind'])
    w = q['world']
    expected = {'goal_forward':q['forward'],'universal_retreat':q['retreat'],
                'visible_contrastive':q['retreat'] if row['kind']=='near_trap' else q['forward']}
    if (row['labels']!=expected or row['state_sha256']!=w.state_sha() or
        row['base_sha256']!=w.base_sha() or row['menu']!=w.menu(12) or
        row['witness']!=q['witness'] or row['prefix']!=q['prefix'] or
        row['goal']!=list(q['goal'])):
        raise ValueError('training ledger not source-replayable')
    return True


def episode(sp, seed, variant, turns, kind, arm, model, horizon=128, png=None):
    if sp.split != 'development' or kind not in KINDS or arm not in ARMS:
        raise ValueError('evaluation not development')
    q = construct(sp,seed,variant,turns,kind)
    w = q['world']; before=len(w.ink); starting=w.state_sha(); base=w.base_sha()
    trace=[];lat=[]
    for _ in range(horizon):
        if w.stopped:break
        tic=time.perf_counter_ns()
        chosen,prob=model.choose(w,q['goal'],horizon,False)
        lat.append((time.perf_counter_ns()-tic)/1e6)
        trace.append(dict(requested=chosen, probabilities=prob, event=w.apply(chosen,12)))
    if png is not None:_png(png,w)
    return dict(version=VERSION, split='development', spec=q['spec'],
                seed=seed, variant=variant, turns=turns, kind=kind,
                arm=arm, horizon=horizon, goal=list(q['goal']),
                first_menu=sorted([q['forward'],q['retreat'],'STOP']),
                forward=q['forward'], retreat=q['retreat'],
                prefix=q['prefix'], first_sha256=starting, base_sha256=base,
                witness=q['witness'], first_action=trace[0]['requested'],
                rows=trace, new_paint=len(w.ink)-before, final_sha256=w.state_sha(),
                metrics=w.metrics(horizon+len(q['prefix'])),
                latency_ms=round(statistics.mean(lat),6))


def verify_episode(ep, model):
    if ep['version']!=VERSION or ep['split']!='development':
        raise ValueError('wrong version/split')
    sp0=Spec(**ep['spec']); sp=Spec(sp0.split,sp0.family,sp0.seed,sp0.mode,'normal')
    q=construct(sp,ep['seed'],ep['variant'],ep['turns'],ep['kind']); w=q['world']
    if (ep['first_sha256']!=w.state_sha() or ep['base_sha256']!=w.base_sha() or
        ep['prefix']!=q['prefix'] or ep['goal']!=list(q['goal']) or
        ep['witness']!=q['witness'] or ep['first_menu']!=w.menu(12) or
        ep['forward']!=q['forward'] or ep['retreat']!=q['retreat']):
        raise ValueError('first-state evidence mismatch')
    before=len(w.ink)
    for entry in ep['rows']:
        action,p=model.choose(w,q['goal'],ep['horizon'],False)
        if (entry['requested']!=action or entry['probabilities']!=p or
            entry['event']!=w.apply(action,12)):
            raise ValueError('policy/menu/material replay mismatch')
    if (len(ep['rows'])>ep['horizon'] or ep['first_action']!=ep['rows'][0]['requested'] or
        ep['new_paint']!=len(w.ink)-before or ep['final_sha256']!=w.state_sha() or
        ep['metrics']!=w.metrics(ep['horizon']+len(q['prefix'])) or
        (len(ep['rows'])<ep['horizon'] and not w.stopped)):
        raise ValueError('terminal summary mismatch')
    return len(ep['rows'])


def summarize(eps):
    out={}
    for kind in KINDS:
        for arm in ARMS:
            rows=[e for e in eps if e['kind']==kind and e['arm']==arm]
            acts=Counter(r['event']['chosen'] for e in rows for r in e['rows'])
            expected=lambda e: e['forward'] if kind=='open' else e['retreat']
            out[kind+'/'+arm]=dict(n=len(rows), correct_first=sum(e['first_action']==expected(e) for e in rows),
                early_stop=sum(len(e['rows'])<e['horizon'] for e in rows),
                mean_new_paint=round(statistics.mean(e['new_paint'] for e in rows),3),
                mean_steps=round(statistics.mean(len(e['rows']) for e in rows),3),
                mean_latency_ms=round(statistics.mean(e['latency_ms'] for e in rows),6),
                action_distribution=dict(sorted(acts.items())))
    return out


def source_sha():return {f:sha_file(Path(__file__).parent/f) for f in SOURCE}


def run(out, seed=701, base_updates=18, adapt_updates=32, natural_limit=64, variants=1, quick=False):
    if out.exists():raise FileExistsError('immutable evidence exists')
    out.mkdir(parents=True); t=time.perf_counter(); gc.collect(); oldgc=gc.isenabled()
    try:
        if oldgc:gc.disable()
        models,hist,ledger=train(seed,base_updates,adapt_updates,natural_limit,variants)
        checkpoints={}
        for arm,m in models.items():
            f=out/(arm+'.npz');m.save(f);r=host.Ranker.load(f)
            if any(not np.array_equal(m.p[k],r.p[k]) for k in m.p):raise ValueError('restore')
            checkpoints[arm]=dict(sha256=sha_file(f), parameters=sum(v.size for v in m.p.values()),updates=m.updates)
        models={a:host.Ranker.load(out/(a+'.npz')) for a in ARMS}
        (out/'traces').mkdir();(out/'png').mkdir()
        eps=[];aliases=[]
        for sp in specs('development',1)[:1 if quick else 4]:
            for turns in ((0,) if quick else EVAL_ROTATIONS):
                cases, features=matched(sp, DEV_SEED, 0, turns)
                assert np.array_equal(features['far_trap'],features['open'])
                alias=dict(family=sp.family,turns=turns,first_menu=cases['open']['world'].menu(12),
                           far_sha256=cases['far_trap']['world'].state_sha(),
                           open_sha256=cases['open']['world'].state_sha(),
                           same_policy_input=True,policy_choices={})
                for arm in ARMS:
                    ff=models[arm].choose(cases['far_trap']['world'],cases['far_trap']['goal'],128,False)
                    oo=models[arm].choose(cases['open']['world'],cases['open']['goal'],128,False)
                    if ff!=oo:raise ValueError('aliased policy inputs gave different output')
                    alias['policy_choices'][arm]=ff[0]
                aliases.append(alias)
                for kind in KINDS:
                    for arm in ARMS:
                        name=f'{sp.family}-{sp.seed}-{turns}-{kind}-{arm}'
                        png=out/'png'/(name+'.png') if not quick and turns==0 and arm in ('goal_forward','visible_contrastive') else None
                        ep=episode(sp,DEV_SEED,0,turns,kind,arm,models[arm],72 if quick else 128,png)
                        verify_episode(ep,models[arm]);eps.append(ep)
                        (out/'traces'/(name+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
    finally:
        if oldgc:gc.enable()
    for row in ledger:verify_train(row)
    (out/'training_cases.json').write_text(json.dumps(ledger,sort_keys=True)+'\n')
    (out/'aliases.json').write_text(json.dumps(aliases,sort_keys=True)+'\n')
    rep=dict(version=VERSION,seed=seed,curriculum=STAGES,base_updates=base_updates,
             adapt_updates=adapt_updates,natural_limit=natural_limit,variants=variants,
             history=hist,checkpoints=checkpoints,metrics=summarize(eps),
             episodes=len(eps),decisions=sum(len(e['rows']) for e in eps),
             training_states=len(ledger),aliases=len(aliases),source_sha256=source_sha(),
             synthetic_only=True,sealed_heldout_opened=False,native_receipts=0,paid_compute=False,
             elapsed_s=round(time.perf_counter()-t,3),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file()},indent=2,sort_keys=True)+'\n')
    return rep


def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=source_sha():raise ValueError('source identity')
    actual={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'}
    if actual!=json.loads((out/'SHA256SUMS.json').read_text()):raise ValueError('file hash mismatch')
    models={a:host.Ranker.load(out/(a+'.npz')) for a in ARMS}
    for a,m in models.items():
        if sha_file(out/(a+'.npz'))!=rep['checkpoints'][a]['sha256'] or m.updates!=rep['checkpoints'][a]['updates']:
            raise ValueError('model identity')
    ledger=json.loads((out/'training_cases.json').read_text())
    for row in ledger:verify_train(row)
    eps=[]
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text());verify_episode(ep,models[ep['arm']]);eps.append(ep)
    aliases=json.loads((out/'aliases.json').read_text())
    for row in aliases:
        sp=next(s for s in specs('development',1) if s.family==row['family'])
        cases,feats=matched(sp,DEV_SEED,0,row['turns'])
        if (not np.array_equal(feats['far_trap'],feats['open']) or
            row['far_sha256']!=cases['far_trap']['world'].state_sha() or
            row['open_sha256']!=cases['open']['world'].state_sha() or
            row['first_menu']!=cases['open']['world'].menu(12) or
            any(models[a].choose(cases['far_trap']['world'],cases['far_trap']['goal'],128,False)[0]!=choice
                for a,choice in row['policy_choices'].items())):
            raise ValueError('alias certificate tampered')
    summed=summarize(eps)
    for a in summed:
        summed[a].pop('mean_latency_ms'); rep['metrics'][a].pop('mean_latency_ms')
    if (summed!=rep['metrics'] or len(ledger)!=rep['training_states'] or
        len(eps)!=rep['episodes'] or len(aliases)!=rep['aliases'] or
        sum(len(e['rows']) for e in eps)!=rep['decisions']):
        raise ValueError('evidence replay aggregation')
    return dict(checkpoints=len(models),training_cases=len(ledger),episodes=len(eps),
                decisions=sum(len(e['rows']) for e in eps),aliases=len(aliases),hashes=len(actual))


if __name__ == '__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--out',type=Path);cli.add_argument('--verify',type=Path)
    cli.add_argument('--seed',type=int,default=701)
    cli.add_argument('--base-updates',type=int,default=18)
    cli.add_argument('--adapt-updates',type=int,default=32)
    cli.add_argument('--natural-limit',type=int,default=64)
    cli.add_argument('--variants',type=int,default=1)
    cli.add_argument('--quick',action='store_true')
    args=cli.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,args.base_updates,args.adapt_updates,args.natural_limit,args.variants,args.quick),sort_keys=True))

"""RECOVERY-FORK-001: legal near-exhausted recovery forks, matched neural adaptation.

This is the DRIV3R synthetic host fixture, NOT the CyberSand native ABI.
No structural HELDOUT specimens, native receipts, or paid models are accessed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import goal_recovery_lab as host, goal_retarget_lab as fixture, policy
from .world import World, Spec, MOVE, specs, digest, replay
from .train import _png, sha_file

VERSION = 'recovery-fork/1'
STAGES = (36,72,128)
ARMS = ('frozen','paired_imitation','paired_rescue')
SOURCE = ('recovery_fork_lab.py','goal_recovery_lab.py','goal_conditioning_lab.py',
          'goal_retarget_lab.py','world.py','policy.py','train.py')
FORK_DEPTH = 24
DEV_SEED = 781


def viability(world, first, depth=FORK_DEPTH):
    """Replay a deterministic, non-neural legal MOVE-only continuation.

    This is a *geometric viability certificate*, not the trained policy or an
    estimate of autonomous model success. Never generates or applies illegal moves.
    """
    if first not in MOVE or first not in world.menu(12) or not 1<=depth<=32:
        raise ValueError('invalid counterfactual first move')
    w = copy.deepcopy(world); before = len(w.ink); rows=[]
    for t in range(depth):
        options = [a for a in w.menu(12) if a in MOVE]
        if t == 0: action = first
        elif options:
            # Greedy deterministic open-space continuation; no STOP until forced.
            quality = []
            for a in options:
                delta = MOVE[a]; p = (w.cursor[0]+delta[0],w.cursor[1]+delta[1])
                freedom = sum(not w.occupied((p[0]+dx,p[1]+dy))
                              for dx,dy in MOVE.values())
                quality.append((freedom,a))
            action = max(quality)[1]
        else: action = 'STOP'
        rows.append(w.apply(action,12))
        if action == 'STOP': break
    return {'completed':len(rows),'moves':sum(r['chosen'] in MOVE for r in rows),
            'forced_stop':bool(w.stopped), 'painted':len(w.ink)-before,
            'actions':[r['chosen'] for r in rows], 'state_sha256':w.state_sha()}


def _main_path(spec, seed, attempt, challenge="trap_corridor"):
    """Construct eleven real BACKTRACKs, then one real RELOCATE_START.

    Preserve root and eastern trap access. All movement uses World.apply,
    never arbitrary cursor/recovery mutation or manufactured legal menus.
    """
    w = World.create(replace(spec,challenge=challenge))
    rng = random.Random(seed*91+spec.seed*341+attempt*445)
    root=w.cursor
    for i in range(11):
        probes=[a for a in w.menu(12) if a in MOVE and (i or a!='MOVE_E')]
        rng.shuffle(probes)
        choice=None
        for probe in probes:
            v=MOVE[probe]; dst=(w.cursor[0]+v[0],w.cursor[1]+v[1])
            if dst==root: continue
            sim=copy.deepcopy(w);sim.apply(probe,12);sim.apply('BACKTRACK',12)
            forwards=[a for a in sim.menu(12) if a in MOVE and (i or a!='MOVE_E')]
            eligible=[]
            for forward in forwards:
                d=MOVE[forward];p=(sim.cursor[0]+d[0],sim.cursor[1]+d[1])
                if p==root:continue
                trial=copy.deepcopy(sim);trial.apply(forward,12)
                degree=sum(a in MOVE for a in trial.menu(12))
                if degree>=2 or i>=9:eligible.append((degree,forward))
            if eligible:
                best=max(d for d,_ in eligible)
                choice=sim,rng.choice([f for d,f in eligible if d==best]);break
        if choice is None:return None
        w,forward=choice;w.apply(forward,12)
    if w.recoveries!=11 or 'RELOCATE_START' not in w.menu(12):return None
    w.apply('RELOCATE_START',12)
    if w.recoveries!=12 or 'MOVE_E' not in w.menu(12):return None
    w.apply('MOVE_E',12) # One step *inside* cul-de-sac; next E seals the retreat.
    if not {'MOVE_E','MOVE_W'}.issubset(w.menu(12)) or w.cursor!=(49,48):return None
    return w


def fork(spec, seed, variant=0):
    """Success/failure is chosen solely from actual host-legal counterfactuals."""
    if spec.split not in ('train','development'):
        raise ValueError('sealed heldout forbidden')
    if spec.challenge!='normal' or not 0<=variant<=7:
        raise ValueError('bad variant')
    for attempt in range(variant*11,variant*11+60):
        w=_main_path(spec,seed+variant*17,attempt)
        if w is None:continue
        bad=viability(w,'MOVE_E');good=viability(w,'MOVE_W')
        if bad['moves']<=7 and bad['forced_stop'] and good['moves']>=20 and not good['forced_stop']:
            return w,{'attempt':attempt,'bad':bad,'good':good,
                      'prefix_actions':[r['chosen'] for r in w.decisions]}
    raise RuntimeError('no qualifying fork; no synthetic forced success permitted')


def open_fork(spec,seed,variant=0):
    """Matched exhausted-recovery CONTROL with a viable eastward route."""
    if spec.split not in ('train','development') or spec.challenge!='normal':
        raise ValueError('no heldout controls')
    for attempt in range(variant*11,variant*11+60):
        w=_main_path(spec,seed+variant*17,attempt,'normal')
        if w is None or 'MOVE_W' not in w.menu(12):continue
        east=viability(w,'MOVE_E');west=viability(w,'MOVE_W')
        if east['moves']>=20 and west['moves']>=20:
            return w,{'attempt':attempt,'east':east,'west':west,
                      'prefix_actions':[r['chosen'] for r in w.decisions]}
    raise RuntimeError('no qualifying open control')


def open_control(sp,models,seed,variant):
    w,cert=open_fork(sp,seed,variant);goal=east_goal(w);choices={}
    for arm in ARMS:
        started=time.perf_counter_ns();a,p=models[arm].choose(w,goal,128,False)
        choices[arm]={'chosen':a,'probabilities':p,
                      'latency_ms':round((time.perf_counter_ns()-started)/1e6,6)}
    return {'spec':w.spec.__dict__,'seed':seed,'variant':variant,
            'goal':list(goal),'state_sha256':w.state_sha(),'menu':w.menu(12),
            'certificate':cert,'choices':choices}


def verify_open_control(row,models):
    sp=Spec(**row['spec'])
    if sp.split!='development' or sp.challenge!='normal':raise ValueError('open split')
    w,cert=open_fork(sp,row['seed'],row['variant'])
    if (row['state_sha256']!=w.state_sha() or row['certificate']!=cert
        or row['goal']!=list(east_goal(w)) or row['menu']!=w.menu(12)):
        raise ValueError('open state')
    for arm in ARMS:
        a,p=models[arm].choose(w,tuple(row['goal']),128,False)
        if row['choices'][arm]['chosen']!=a or row['choices'][arm]['probabilities']!=p:
            raise ValueError('open policy response')
    return len(row['choices'])


def summary_open(rows):
    return {arm:{'episodes':len(rows),
                 'eastward_viable_choice':sum(e['choices'][arm]['chosen']=='MOVE_E' for e in rows),
                 'westward_choice':sum(e['choices'][arm]['chosen']=='MOVE_W' for e in rows),
                 'other_choice':sum(e['choices'][arm]['chosen'] not in ('MOVE_E','MOVE_W') for e in rows),
                 'mean_east_probability':round(statistics.mean(
                     e['choices'][arm]['probabilities'].get('MOVE_E',0) for e in rows),6)}
            for arm in ARMS}


def east_goal(w):
    # Explicit public instruction; never manufacture a protected/occupied goal.
    for p in ((57,48),(58,48),(58,50),(58,46),(60,50),(60,45)):
        if not w.occupied(p):return p
    raise RuntimeError('no eastward goal target')


def paired_corpus(seed,stage,natural_limit=64,fork_variants=2):
    if stage not in STAGES or not 16<=natural_limit<=256 or not 1<=fork_variants<=4:
        raise ValueError('bounded corpus')
    X,M,Y,corpus=host.corpus(seed,stage)
    # Identical, predeclared non-risk samples for both adapted controllers.
    rng=np.random.default_rng(seed*11+stage)
    indices=np.sort(rng.choice(len(Y),size=min(natural_limit,len(Y)),replace=False))
    xs=list(X[indices]);masks=list(M[indices]);teacher=list(Y[indices]);rescue=list(Y[indices]);pairs=[]
    for sp in specs('train',2):
        for variant in range(fork_variants):
            w,certificate=fork(sp,seed,variant)
            goal=east_goal(w);menu,features=host.matrix(w,goal,stage,False)
            if menu!=['MOVE_E','MOVE_W','STOP']:raise ValueError('fork menu changed')
            imitation=host.label(w,goal,stage,random.Random(seed*571+sp.seed+variant))
            if imitation not in menu:raise ValueError('illegal teacher label')
            mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
            xs.append(features);masks.append(mask)
            teacher.append(menu.index(imitation));rescue.append(menu.index('MOVE_W'))
            pairs.append({'spec':w.spec.__dict__,'seed':seed,'variant':variant,
                          'stage':stage,'goal':list(goal),'menu':menu,'teacher':imitation,
                          'rescue':'MOVE_W','state_sha256':w.state_sha(),
                          'prefix_actions':certificate['prefix_actions'],
                          'certificate':{k:certificate[k] for k in ('attempt','bad','good')}})
    return (np.stack(xs),np.stack(masks),np.array(teacher,dtype=np.int64),
            np.array(rescue,dtype=np.int64),pairs,
            {'split':'train','baseline_examples':len(indices),'fork_states':len(pairs),
             'teacher_risky_forks':sum(p['teacher']=='MOVE_E' for p in pairs),
             'duplicates_avoided':True,'source_corpus':corpus})


def train(seed,base_updates=18,adapt_updates=32,natural_limit=64,fork_variants=2):
    if not 1<=base_updates<=80 or not 1<=adapt_updates<=80:raise ValueError('optimizer bound')
    baseline,prior=host.train(seed,base_updates)
    frozen=baseline['goal_recovery']
    models={a:copy.deepcopy(frozen) for a in ARMS};hist=[];ledger=[]
    for stage in STAGES:
        X,M,Y0,Y1,pairs,meta=paired_corpus(seed,stage,natural_limit,fork_variants)
        fits={}
        for arm,y in (('paired_imitation',Y0),('paired_rescue',Y1)):
            fits[arm]=models[arm].fit(X,M,y,adapt_updates,seed*337+stage)
        hist.append({'stage':stage,'corpus':meta,'optimizers':fits});ledger.extend(pairs)
    return models,{'baseline':prior,'adaptation':hist},ledger


def development_cases():
    return specs('development',2)


def replay_prefix(sp,prefix):
    w=World.create(sp)
    for action in prefix:w.apply(action,12)
    return w


def targeted_episode(sp,model,arm,seed,variant=0,horizon=128,png=None):
    if sp.split!='development' or horizon not in (72,128):raise ValueError('DEV only')
    w,cert=fork(sp,seed,variant);goal=east_goal(w)
    prefix=cert['prefix_actions'];fork_sha=w.state_sha();prefix_ink=len(w.ink);rows=[];lat=[]
    for _ in range(horizon):
        if w.stopped:break
        started=time.perf_counter_ns()
        action,probs=model.choose(w,goal,horizon,False)
        lat.append((time.perf_counter_ns()-started)/1e6)
        rows.append({'requested':action,'probabilities':probs,'event':w.apply(action,12)})
    if png is not None:_png(png,w)
    return {'version':VERSION,'split':'development','arm':arm,'spec':w.spec.__dict__,
            'seed':seed,'variant':variant,'horizon':horizon,'goal':list(goal),
            'prefix':prefix,'fork_sha256':fork_sha,'certificate':cert,
            'first_action':rows[0]['requested'],'rows':rows,'state_sha256':w.state_sha(),
            'masked_rescue_choice':model.choose(replay_prefix(w.spec,prefix),goal,horizon,True)[0] if arm=='paired_rescue' else None,
            'metrics':w.metrics(len(prefix)+horizon),'paint_after_fork':len(w.ink)-prefix_ink,
            'mean_latency_ms':round(statistics.mean(lat),6) if lat else 0.}


def verify_targeted(ep,model):
    sp=Spec(**ep['spec']);w=World.create(sp)
    if sp.split!='development' or [r['chosen'] for r in w.decisions]:raise ValueError('spec')
    for action in ep['prefix']:w.apply(action,12)
    if (w.state_sha()!=ep['fork_sha256'] or w.menu(12)!=['MOVE_E','MOVE_W','STOP']
        or ep['goal']!=list(east_goal(w))):raise ValueError('fork replay')
    prefix_ink=len(w.ink)
    if ep['masked_rescue_choice']!=(model.choose(w,tuple(ep['goal']),ep['horizon'],True)[0]
        if ep['arm']=='paired_rescue' else None):raise ValueError('masked counterfactual')
    for i,row in enumerate(ep['rows']):
        asked,probs=model.choose(w,tuple(ep['goal']),ep['horizon'],False)
        if asked!=row['requested'] or probs!=row['probabilities'] or row['event']!=w.apply(asked,12):
            raise ValueError('model probability or host event mismatch')
    if (not ep['rows'] or ep['first_action']!=ep['rows'][0]['requested']
        or w.state_sha()!=ep['state_sha256'] or ep['metrics']!=w.metrics(len(ep['prefix'])+ep['horizon'])
        or ep['paint_after_fork']!=len(w.ink)-prefix_ink
        or (not w.stopped and len(ep['rows'])!=ep['horizon'])):raise ValueError('trajectory')
    return len(ep['rows'])


def summarize_targeted(episodes):
    result={}
    for arm in ARMS:
        xs=[e for e in episodes if e['arm']==arm]
        result[arm]={'episodes':len(xs),'safe_first':sum(e['first_action']=='MOVE_W' for e in xs),
             'risky_first':sum(e['first_action']=='MOVE_E' for e in xs),
             'stop_first':sum(e['first_action']=='STOP' for e in xs),
             'early_terminations':sum(len(e['rows'])<e['horizon'] for e in xs),
             'mean_postfork_decisions':round(statistics.mean(len(e['rows']) for e in xs),3),
             'mean_postfork_paint':round(statistics.mean(
                 len({tuple(p) for row in e['rows'] for p in row['event']['painted']}) for e in xs),3),
             'decision_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in xs),6),
             'actions':dict(sorted(Counter(r['event']['chosen'] for e in xs for r in e['rows']).items()))}
    return result


def standard_dev(models,quick=False):
    eps=[]
    for sp in development_cases()[:1 if quick else 4]:
        for horizon in ((128,) if quick else (128,160)):
            for arm in ARMS:
                ep=fixture.run_episode(sp,{'goal_recovery':models[arm]},'goal_recovery',
                                       DEV_SEED,horizon,'retarget')
                fixture.verify_episode(ep,sp,{'goal_recovery':models[arm]})
                ep['trained_arm']=arm;eps.append(ep)
    return eps


def summary_standard(eps):
    out={}
    for arm in ARMS:
        xs=[e for e in eps if e['trained_arm']==arm]
        out[arm]={'episodes':len(xs),'goals':sum(len(e['reached_epochs']) for e in xs),
                  'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in xs),3),
                  'early_stop':sum(e['metrics']['stopped_early'] for e in xs),
                  'actions':dict(sorted(Counter(a for e in xs for a,c in e['action_counts'].items()
                                                 for _ in range(c)).items()))}
    return out


def provenance():return {f:sha_file(Path(__file__).parent/f) for f in SOURCE}


def run(out,seed=681,base_updates=18,adapt_updates=32,natural_limit=64,fork_variants=2,quick=False):
    if out.exists():raise FileExistsError('immutable evidence')
    out.mkdir(parents=True);start=time.perf_counter();gc.collect();enabled=gc.isenabled()
    try:
        if enabled:gc.disable()
        models,hist,pairs=train(seed,base_updates,adapt_updates,natural_limit,fork_variants)
        checkpoints={}
        for arm,m in models.items():
            p=out/(arm+'.npz');m.save(p);loaded=host.Ranker.load(p)
            if any(not np.array_equal(m.p[k],loaded.p[k]) for k in m.p):raise ValueError('restore')
            checkpoints[arm]={'sha256':sha_file(p),'params':sum(x.size for x in m.p.values()),
                              'updates':m.updates}
        models={a:host.Ranker.load(out/(a+'.npz')) for a in ARMS}
        (out/'traces').mkdir();(out/'png').mkdir()
        targets=[]
        for sp in development_cases()[:1 if quick else len(development_cases())]:
            for variant in range(1 if quick else 2):
                for arm in ARMS:
                    name=f'{sp.family}-{sp.seed}-{variant}-{arm}'
                    ep=targeted_episode(sp,models[arm],arm,DEV_SEED,variant,
                                        72 if quick else 128,
                                        out/'png'/(name+'.png') if variant==0 and not quick else None)
                    verify_targeted(ep,models[arm]);targets.append(ep)
                    (out/'traces'/(name+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
        std=standard_dev(models,quick)
        controls=[open_control(sp,models,DEV_SEED,variant)
             for sp in development_cases()[:1 if quick else len(development_cases())]
             for variant in range(1 if quick else 2)]
        for row in controls:verify_open_control(row,models)
    finally:
        if enabled:gc.enable()
    for i,e in enumerate(std):
        (out/'traces'/f'standard-{i:03}-{e["trained_arm"]}.json').write_text(json.dumps(e,sort_keys=True)+'\n')
    (out/'paired_forks.json').write_text(json.dumps(pairs,sort_keys=True)+'\n')
    (out/'open_controls.json').write_text(json.dumps(controls,sort_keys=True)+'\n')
    rep={'version':VERSION,'seed':seed,'curriculum':STAGES,'base_updates':base_updates,
         'adapt_updates':adapt_updates,'training':hist,'checkpoints':checkpoints,
         'targeted':summarize_targeted(targets),'standard':summary_standard(std),
         'open_controls':summary_open(controls),'open_control_cases':len(controls),
         'masked_rescue_choices':dict(Counter(e['masked_rescue_choice'] for e in targets
             if e['arm']=='paired_rescue')),
         'targeted_episodes':len(targets),'standard_episodes':len(std),
         'decisions':sum(len(e['rows']) for e in targets+std),'source_sha256':provenance(),
         'sealed_heldout_opened':False,'native_receipts':0,'paid_compute':False,
         'elapsed_s':round(time.perf_counter()-start,3),
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(rep,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
            for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return rep


def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=provenance():raise ValueError('source')
    hashes=json.loads((out/'SHA256SUMS.json').read_text())
    if hashes!={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*')
                if p.is_file() and p.name!='SHA256SUMS.json'}:raise ValueError('hash')
    models={a:host.Ranker.load(out/(a+'.npz')) for a in ARMS}
    for a,m in models.items():
        if sha_file(out/(a+'.npz'))!=rep['checkpoints'][a]['sha256'] or m.updates!=rep['checkpoints'][a]['updates']:
            raise ValueError('checkpoint')
    pair=json.loads((out/'paired_forks.json').read_text())
    for p in pair:
        sp=Spec(**p['spec']);w=World.create(sp)
        if sp.split!='train':raise ValueError('training split')
        for a in p['prefix_actions']:w.apply(a,12)
        if (w.state_sha()!=p['state_sha256'] or w.menu(12)!=p['menu']
            or p['goal']!=list(east_goal(w)) or p['rescue']!='MOVE_W'):
            raise ValueError('pair')
        for a,key in (('MOVE_E','bad'),('MOVE_W','good')):
            if viability(w,a)!=p['certificate'][key]:raise ValueError('fork counterfactual')
    controls=json.loads((out/'open_controls.json').read_text())
    for row in controls:verify_open_control(row,models)
    if (len(controls)!=rep['open_control_cases'] or
        summary_open(controls)!=rep['open_controls']):raise ValueError('open summary')
    targeted=[];std=[];dev={sp.ident:sp for sp in development_cases()}
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text());arm=ep['arm'] if ep.get('version')==VERSION else ep['trained_arm']
        if ep.get('version')==VERSION:verify_targeted(ep,models[arm]);targeted.append(ep)
        else:fixture.verify_episode(ep,dev[ep['spec']],{'goal_recovery':models[arm]});std.append(ep)
    if (len(targeted)!=rep['targeted_episodes'] or len(std)!=rep['standard_episodes']
        or sum(len(e['rows']) for e in targeted+std)!=rep['decisions']):raise ValueError('episodes')
    for key,summ,expected in (('targeted',summarize_targeted(targeted),rep['targeted']),
                              ('standard',summary_standard(std),rep['standard'])):
        if key=='targeted':
            for a in ARMS:summ[a].pop('decision_latency_ms');expected={k:dict(v) for k,v in expected.items()};expected[a].pop('decision_latency_ms')
        if summ!=expected:raise ValueError(key+' summary')
    if dict(Counter(e['masked_rescue_choice'] for e in targeted
        if e['arm']=='paired_rescue'))!=rep['masked_rescue_choices']:raise ValueError('masked summary')
    return {'checkpoints':len(models),'fork_pairs':len(pair),
            'open_controls':len(controls),
            'targeted_episodes':len(targeted),'standard_episodes':len(std),
            'decisions':rep['decisions'],'hashes':len(hashes)}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--verify',type=Path)
    ap.add_argument('--seed',type=int,default=681);ap.add_argument('--quick',action='store_true')
    ap.add_argument('--base-updates',type=int,default=18)
    ap.add_argument('--adapt-updates',type=int,default=32)
    ap.add_argument('--natural-limit',type=int,default=64)
    ap.add_argument('--fork-variants',type=int,default=2)
    args=ap.parse_args()
    print(json.dumps(verify(args.verify) if args.verify else run(args.out,args.seed,
       args.base_updates,args.adapt_updates,args.natural_limit,args.fork_variants,args.quick),sort_keys=True))

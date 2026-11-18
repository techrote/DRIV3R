"""AUTONOMOUS-CONTINUATION-001: test full neural continuation after recovery.

Model-side synthetic DEV experiment, NOT CyberSand native. A frozen host-like World
validates each action. The teacher controls TRAIN labels and explicit diagnostic
arms only; it never controls the learned arms after the recovery fork. No sealed
HELDOUT families are instantiated. Checkpoints and full traces stay in artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import policy
from .side_exit_cost_lab import (CostRanker, corpus, counterfactual, make_forced)
from .train import make_corpus, sha_file, _png
from .world import MOVE, digest, specs

VERSION='autonomous-continuation/1'
STAGES=(36,72,128)
ARMS=('imitation','on_policy','teacher_control','oracle_fork_neural','oracle_fork_teacher')
CAPS=(3,7,10)
CASES=(('W',6,2),('W',8,4),('S',8,2),('S',10,4))
HORIZON=160


def legal_matrix(w, cap, horizon):
    legal=w.menu(cap)
    x=np.zeros((policy.ACTION_MAX,len(policy.FEATURES)),dtype=np.float64)
    for i,act in enumerate(legal):x[i]=policy.features(w,act,horizon)
    return legal,x


def student_choice(w,cap,horizon,m):
    legal,x=legal_matrix(w,cap,horizon)
    p=m.probs(x,len(legal))
    return legal[int(p.argmax())],{a:round(float(p[i]),8) for i,a in enumerate(legal)}


def teacher_choice(w,cap,horizon,rng,goal):
    from .side_exit_cost_lab import step_controller
    return step_controller(w,cap,goal,horizon,rng)


def student_state_corpus(model,seed,stage,n_worlds=6):
    """DAgger-style labels on actual student-visited TRAIN states only."""
    worlds=specs('train',2)[:n_worlds]
    X=[];M=[];Y=[];moves=Counter();interventions=0
    for spec in worlds:
        from .world import World
        w=World.create(spec);rng=random.Random(seed*37+spec.seed+stage*31)
        for step in range(stage):
            if w.stopped:break
            menu,x=legal_matrix(w,12,stage)
            s=policy.teacher_scores(w,menu,x[:len(menu)])
            label=int(s.argmax())
            mask=np.zeros(policy.ACTION_MAX,dtype=bool);mask[:len(menu)]=True
            X.append(x);M.append(mask);Y.append(label)
            chosen,_=student_choice(w,12,stage,model)
            # Inject bounded legal deviation while gathering states, not labels.
            progress=[a for a in menu if a in MOVE and a!=chosen]
            if progress and rng.random()<.14:
                chosen=rng.choice(progress);interventions+=1
            moves[chosen]+=1;w.apply(chosen,12)
    if not X:raise ValueError('on-policy corpus empty')
    return np.stack(X),np.stack(M),np.asarray(Y,dtype=np.int64),{
        'train_examples':len(Y),'train_worlds':len(worlds),
        'interventions':interventions,'action_counts':dict(sorted(moves.items()))}


def train_models(seed,updates=24,adapt_updates=36):
    if updates<1 or updates>100 or adapt_updates<1 or adapt_updates>120:raise ValueError('optimizer bound')
    base=policy.Ranker(seed,hidden=24)
    chronology=[]
    for stage in STAGES:
        X,M,Y,meta=make_corpus(stage,seed,1,adversarial_fraction=.16)
        fit=base.optimize(X,M,Y,updates,batch=32,lr=.013,seed=seed*100+stage)
        chronology.append({'stage':stage,'examples':len(Y),'labels':meta['labels'],'fit':fit})
    replay=deepcopy(base);adapt=deepcopy(base)
    X,M,Y,meta=make_corpus(128,seed+11,1,adversarial_fraction=.16)
    rfit=replay.optimize(X,M,Y,adapt_updates,batch=32,lr=.012,seed=seed*71)
    xx,mm,yy,state_meta=student_state_corpus(adapt,seed,128)
    afit=adapt.optimize(xx,mm,yy,adapt_updates,batch=32,lr=.012,seed=seed*71)
    fork=CostRanker(seed,'nonlinear_geometry')
    ftrain=[]
    for stage in STAGES:
        xx,yy,ww,meta=corpus(seed,stage,quick=True)
        ftrain.append({'stage':stage,'examples':len(yy),
                       'labels':{'backtrack':int(yy.sum()),'relocate':int(len(yy)-yy.sum())},
                       'fit':fork.train(xx['nonlinear_geometry'],yy,ww,updates,seed*11+stage)})
    return {'imitation':replay,'on_policy':adapt,'fork':fork},{
        'stage_training':chronology,'matched_extra_updates':adapt_updates,
        'replay_fit':rfit,'on_policy_fit':afit,'on_policy_data':state_meta,
        'fork_training':ftrain}


def episode(spec, models, arm, depth, direction, back, cap,seed,horizon=HORIZON):
    if spec.split!='development' or arm not in ARMS or (direction,depth,back) not in CASES or cap not in CAPS or horizon!=HORIZON:
        raise ValueError('development partition/parameters')
    w,goal=make_forced(spec,depth,direction,back,1,cap)
    w0,_=make_forced(spec,depth,direction,back,1,cap)
    # Initialize the original geometry before forced moves, including overlays.
    from .side_exit_cost_lab import make_world
    original,_=make_world(spec,depth,direction,back,1)
    base_sha=original.base_sha()
    rows=[{'event':ev,'probabilities':None,'origin':'forced_initial_error'} for ev in w.decisions]
    target,cf=counterfactual(w0,goal,cap,horizon,seed+spec.seed)
    if 'oracle_fork' in arm:
        fork_action='BACKTRACK' if target else 'RELOCATE_START'
        fork_probs=None
    else:
        fork_action,fork_probs=models['fork'].decide(w,cap)
    if fork_action not in w.menu(cap):raise ValueError('illegal fork')
    rows.append({'event':w.apply(fork_action,cap),'probabilities':fork_probs,'origin':'fork_model' if fork_probs else 'counterfactual_control'})
    rng=random.Random(seed*17+spec.seed+depth+cap)
    inference_ms=[];injected=0
    for t in range(depth+1,horizon):
        if w.stopped:break
        start=time.perf_counter_ns()
        if arm in ('teacher_control','oracle_fork_teacher'):
            candidate=teacher_choice(w,cap,horizon,rng,goal);probs=None;source='diagnostic_teacher'
        else:
            model=models['on_policy'] if arm in ('on_policy','oracle_fork_neural') else models['imitation']
            candidate,probs=student_choice(w,cap,horizon,model);source='learned_autonomous'
        inference_ms.append((time.perf_counter_ns()-start)/1e6)
        legal=w.menu(cap)
        chosen=candidate
        if t in (37,79,109):
            alternate=[a for a in legal if a in MOVE and a!=candidate]
            if alternate:
                chosen=alternate[(seed+t)%len(alternate)];injected+=1;source='forced_deviation'
        rows.append({'event':w.apply(chosen,cap),'probabilities':probs,'origin':source,
                     'requested':candidate})
    if len(rows)!=w.steps:raise ValueError('event count mismatch')
    return {'version':VERSION,'world':spec.ident,'split':'development',
            'seed':seed,'arm':arm,'depth':depth,'direction':direction,'back':back,
            'cap':cap,'horizon':horizon,'goal':list(goal),'base_sha':base_sha,
            'fork':fork_action,'fork_correct':int(fork_action=='BACKTRACK')==target,
            'oracle_rewards':{k:cf[k]['reward'] for k in ('BACKTRACK','RELOCATE_START')},
            'injected_deviations':injected,'rows':rows,'trace_sha':digest([r['event'] for r in rows]),
            'final_sha':w.state_sha(),'metrics':w.metrics(horizon),'goal_reached':goal in w.ink,
            'post_fork_ink':len(w.ink)-depth,'ink':sorted([list(p) for p in w.ink]),
            'action_counts':dict(sorted(Counter(r['event']['chosen'] for r in rows).items())),
            'neural_latency_mean_ms':round(statistics.mean(inference_ms),6) if inference_ms else 0.,
            'native_receipts':0,'sealed_uninspected':True}


def replay_episode(ep,models,spec):
    if ep['version']!=VERSION or ep['split']!='development' or ep['world']!=spec.ident or ep['native_receipts']!=0:
        raise ValueError('episode identity')
    from .side_exit_cost_lab import make_world
    w,goal=make_world(spec,ep['depth'],ep['direction'],ep['back'],1)
    if w.base_sha()!=ep['base_sha'] or list(goal)!=ep['goal']:raise ValueError('world mismatch')
    from .return_generalization_lab import force_error
    evs=force_error(w,ep['depth'],ep['direction'],ep['cap'])
    rows=ep['rows']
    if [r['event'] for r in rows[:ep['depth']]]!=evs:raise ValueError('forced event mismatch')
    fork_truth,_=counterfactual(w,goal,ep['cap'],ep['horizon'],ep['seed']+spec.seed)
    if ep['arm'].startswith('oracle_fork'):
        choice='BACKTRACK' if fork_truth else 'RELOCATE_START';p=None
    else:choice,p=models['fork'].decide(w,ep['cap'])
    row=rows[ep['depth']]
    if choice!=ep['fork'] or row['probabilities']!=p or row['origin']!=('fork_model' if p else 'counterfactual_control'):
        raise ValueError('fork checkpoint mismatch')
    if w.apply(choice,ep['cap'])!=row['event']:raise ValueError('fork event mismatch')
    rng=random.Random(ep['seed']*17+spec.seed+ep['depth']+ep['cap']);injected=0
    for t,row in enumerate(rows[ep['depth']+1:],start=ep['depth']+1):
        if w.stopped:raise ValueError('continued after stop')
        if ep['arm'] in ('teacher_control','oracle_fork_teacher'):
            candidate=teacher_choice(w,ep['cap'],ep['horizon'],rng,goal);p=None;origin='diagnostic_teacher'
        else:
            mod=models['on_policy'] if ep['arm'] in ('on_policy','oracle_fork_neural') else models['imitation']
            candidate,p=student_choice(w,ep['cap'],ep['horizon'],mod);origin='learned_autonomous'
        chosen=candidate
        if t in (37,79,109):
            alternate=[a for a in w.menu(ep['cap']) if a in MOVE and a!=candidate]
            if alternate:
                chosen=alternate[(ep['seed']+t)%len(alternate)];injected+=1;origin='forced_deviation'
        if (row['origin']!=origin or row['requested']!=candidate or row['probabilities']!=p
                or w.apply(chosen,ep['cap'])!=row['event']):raise ValueError('checkpoint/transition mismatch')
    m=w.metrics(ep['horizon']);events=[r['event'] for r in rows]
    if (w.state_sha()!=ep['final_sha'] or digest(events)!=ep['trace_sha'] or
        m!=ep['metrics'] or ep['goal_reached']!=(goal in w.ink) or
        ep['post_fork_ink']!=len(w.ink)-ep['depth'] or ep['injected_deviations']!=injected or
        ep['ink']!=sorted([list(p) for p in w.ink]) or
        ep['fork_correct']!=(int(choice=='BACKTRACK')==fork_truth) or
        ep['action_counts']!=dict(sorted(Counter(e['chosen'] for e in events).items())) or
        (not w.stopped and len(rows)!=ep['horizon'])):raise ValueError('metrics or trace tamper')
    return len(rows)


def aggregate(eps):
    c=Counter()
    for x in eps:c.update(x['action_counts'])
    return {'episodes':len(eps),'decisions':sum(x['metrics']['decisions'] for x in eps),
        'mean_ink':round(statistics.mean(x['metrics']['unique_painted_cells'] for x in eps),3),
        'mean_post_fork_ink':round(statistics.mean(x['post_fork_ink'] for x in eps),3),
        'early_stop':sum(x['metrics']['stopped_early'] for x in eps),
        'fork_correct':sum(x['fork_correct'] for x in eps),
        'goal_reached':sum(x['goal_reached'] for x in eps),
        'mean_branches':round(statistics.mean(x['metrics']['branch_cells'] for x in eps),3),
        'neural_latency_ms':round(statistics.mean(x['neural_latency_mean_ms'] for x in eps),6),
        'actions':dict(sorted(c.items()))}


def dependencies():
    parent=Path(__file__).parent
    return {name:sha_file(parent/name) for name in ('autonomous_continuation_lab.py','side_exit_cost_lab.py',
                'return_generalization_lab.py','policy.py','world.py','train.py')}


def run(out,seed=610,updates=24,adapt_updates=36,quick=False):
    if out.exists():raise FileExistsError('immutable artifact path already exists')
    if not 0<=seed<100000:raise ValueError('seed bound')
    out.mkdir(parents=True);start=time.perf_counter()
    models,history=train_models(seed,updates,adapt_updates)
    ck={}
    for arm in ('imitation','on_policy'):
        path=out/(arm+'.npz');models[arm].checkpoint(path)
        restored=policy.Ranker.restore(path)
        if any(not np.array_equal(restored.params[k],models[arm].params[k]) for k in restored.params):raise ValueError('checkpoint restore')
        ck[arm]={'sha256':sha_file(path),'updates':restored.updates,'parameters':restored.parameter_count}
    models['fork'].save(out/'fork.npz')
    restored=CostRanker.load(out/'fork.npz')
    if any(not np.array_equal(restored.p[k],models['fork'].p[k]) for k in restored.p):raise ValueError('fork restore')
    ck['fork']={'sha256':sha_file(out/'fork.npz'),'updates':restored.updates,'parameters':sum(x.size for x in restored.p.values())}
    dev=specs('development',1)[:1]
    (out/'traces').mkdir();eps=[];count=0
    for spec in dev:
        for direction,depth,back in CASES[:1 if quick else None]:
            for cap in CAPS[:1 if quick else None]:
                for arm in ARMS:
                    ep=episode(spec,models,arm,depth,direction,back,cap,seed)
                    count+=replay_episode(ep,models,spec)
                    name=f'{arm}-{direction}-d{depth}-b{back}-c{cap}'
                    (out/'traces'/(name+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
                    if (direction,depth,back)==CASES[0] and cap==CAPS[0]:
                        from .side_exit_cost_lab import make_world
                        w,_=make_world(spec,depth,direction,back,1)
                        for r in ep['rows']:w.apply(r['event']['chosen'],cap)
                        _png(out/'traces'/(name+'.png'),w)
                    eps.append(ep)
    summary={arm:aggregate([x for x in eps if x['arm']==arm]) for arm in ARMS}
    result={'version':VERSION,'seed':seed,'updates_per_stage':updates,'adapt_updates':adapt_updates,
        'train':history,'checkpoints':ck,'episodes':len(eps),'decisions':count,'summary':summary,
        'source_sha256':dependencies(),'split':'development',
        'development_cases':list(CASES[:1 if quick else None]),'development_caps':list(CAPS[:1 if quick else None]),
        'horizon':HORIZON,'heldout_uninspected':True,'native_receipts':0,'paid_compute':False,
        'elapsed_s':round(time.perf_counter()-start,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'quick':quick}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return result


def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=dependencies() or not r['heldout_uninspected']:raise ValueError('source mismatch')
    manifest=json.loads((out/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if manifest!=actual:raise ValueError('file hash mismatch')
    models={'imitation':policy.Ranker.restore(out/'imitation.npz'),
            'on_policy':policy.Ranker.restore(out/'on_policy.npz'),
            'fork':CostRanker.load(out/'fork.npz')}
    for k,v in models.items():
        if sha_file(out/(k+'.npz'))!=r['checkpoints'][k]['sha256'] or v.updates!=r['checkpoints'][k]['updates']:
            raise ValueError('checkpoint mismatch')
    dev={s.ident:s for s in specs('development',1)[:1]}
    eps=[];count=0
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text())
        if ep['world'] not in dev:raise ValueError('development identity')
        count+=replay_episode(ep,models,dev[ep['world']]);eps.append(ep)
    if (len(eps),count)!=(r['episodes'],r['decisions']):raise ValueError('episode count')
    for arm in ARMS:
        x=aggregate([e for e in eps if e['arm']==arm]);y=dict(r['summary'][arm]);
        x.pop('neural_latency_ms');y.pop('neural_latency_ms')
        if x!=y:raise ValueError('aggregate mismatch')
    return {'checkpoints':len(models),'episodes':len(eps),'replayed_decisions':count,'verified_files':len(manifest)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=610);p.add_argument('--updates',type=int,default=24)
    p.add_argument('--adapt-updates',type=int,default=36);p.add_argument('--quick',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.seed,a.updates,a.adapt_updates,a.quick),sort_keys=True))

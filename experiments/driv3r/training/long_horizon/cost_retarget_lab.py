"""COST-RETARGET-001: matched cost-to-go labels vs plain teacher imitation.

Research-only CyberSand-like synthetic host menus. 36/72/128 TRAIN curriculum;
128/160 DEVELOPMENT evaluation, no sealed structural heldout or native execution.
All evaluation actions are selected by restored autonomous neural rankers.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import goal_retarget_lab as fixture, goal_recovery_lab as prior
from . import retarget_onpolicy_lab as predecessor
from .world import World, MOVE, specs
from . import policy
from .train import sha_file

VERSION = 'cost-retarget/1'
STAGES = (36, 72, 128)
ARMS = ('frozen', 'imitation', 'cost_mixture')
EVAL_MODES = ('retarget', 'static')
SOURCE_FILES = ('cost_retarget_lab.py','retarget_onpolicy_lab.py',
                'goal_retarget_lab.py','goal_recovery_lab.py',
                'goal_conditioning_lab.py','world.py','policy.py','train.py')


def distance(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def future_score(w: World, first: str, target: tuple[int,int], step: int,
                 horizon: int, seed: int, lookahead: int = 5) -> dict:
    """Teacher-only, bounded simulation of legal actions, never a model input.

    No future retarget instruction is exposed, including at instruction switches.
    Synthetic World copies are offline TRAIN teacher rollouts, not native calls.
    """
    if not 1 <= lookahead <= 8 or first not in w.menu(12):
        raise ValueError('bounded legal counterfactual required')
    sim = copy.deepcopy(w)
    origin_ink=len(sim.ink); origin_rec=sim.recoveries
    first_distance=distance(sim.cursor,target)
    reached = target in sim.ink
    actions = []
    r = random.Random(seed)
    for k in range(lookahead):
        if sim.stopped: break
        if k==0:
            action=first
        else:
            # A cheap, imperfect existing teacher continues the chosen path.
            action=prior.label(sim,target,horizon,r)
            if action not in sim.menu(12):
                raise ValueError('teacher violated synthetic host action menu')
        sim.apply(action,12)
        actions.append(action)
        reached = reached or target in sim.ink
        if sim.stopped: break
    painted=len(sim.ink)-origin_ink
    spent=sim.recoveries-origin_rec
    gain=first_distance-distance(sim.cursor,target)
    # Pre-registered scalar cost. Balances new paint, goal distance, actual
    # target contact, recovery debt and early loss of available progression.
    score=(1.5*painted + 0.65*gain + 5.0*int(reached)
           - 1.75*spent - 9.0*int(sim.stopped))
    return {'score':round(score,6),'paint':painted,'gain':gain,
            'recoveries':spent,'stopped':sim.stopped,'reached':bool(reached),
            'continuation':actions}


def labels_at_state(w, target, horizon, step, seed, lookahead=5):
    menu,_=prior.matrix(w,target,horizon,False)
    rng=random.Random(seed*30011+step*173+w.spec.seed)
    imitation=prior.label(w,target,horizon,rng)
    scores={a:future_score(w,a,target,step,horizon,
              seed*113+step*791+w.spec.seed+17,lookahead) for a in menu}
    # Explicit stable tie-breaking: preserve imitation if utility is tied.
    best=max(scores[a]['score'] for a in menu)
    tied=[a for a in menu if scores[a]['score']==best]
    cost=imitation if imitation in tied else sorted(tied)[0]
    if imitation not in menu or cost not in menu:
        raise ValueError('nonlegal training label')
    return menu,imitation,cost,scores


def training_corpus(model,seed,stage,sample_n=256,lookahead=5):
    """Same frozen student-visited states, masks, minibatches for both arms.

    Stores an exact, bounded counterfactual label ledger; no DEVELOPMENT leak.
    Sampling reservoir keeps memory modest while TRAIN trajectory remains real.
    """
    if stage not in STAGES or not 32<=sample_n<=512:
        raise ValueError('curriculum budget')
    visited=[];seen=0;forced=0;stop_count=0
    reservoir=random.Random(seed*991+stage*37)
    for sp in specs('train',2):
        for variant in (0,1):
            w=World.create(sp)
            targets=fixture.targets_for(sp,seed+variant*41,'retarget')
            perturb=random.Random(seed*713+sp.seed*13+variant*31+stage)
            for t in range(stage):
                if w.stopped:break
                target=targets[fixture.active_phase(t,'retarget')]
                seen+=1
                snapshot={'world':copy.deepcopy(w),'target':target,'step':t,
                          'spec':sp.ident,'variant':variant}
                if len(visited)<sample_n:
                    visited.append(snapshot)
                else:
                    index=reservoir.randrange(seen)
                    if index<sample_n:visited[index]=snapshot
                asked,_=model.choose(w,target,stage,False)
                actual=asked
                if t in (19,57,91,119):
                    wrong=[a for a in w.menu(12) if a in MOVE and a!=asked]
                    if wrong:
                        actual=perturb.choice(wrong);forced+=1
                w.apply(actual,12)
            stop_count+=int(w.stopped and w.steps<stage)
    x_all=[];m_all=[];y_im=[];y_cost=[];audit=[]
    counts=Counter();disagreements=0
    for i,s in enumerate(visited):
        w=s['world'];target=s['target'];t=s['step']
        menu,x=prior.matrix(w,target,stage,False)
        legal,imitation,cost,scores=labels_at_state(w,target,stage,t,seed+i*3,lookahead)
        if legal!=menu:raise ValueError('matrix/menu divergence')
        mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
        x_all.append(x);m_all.append(mask)
        y_im.append(menu.index(imitation));y_cost.append(menu.index(cost))
        counts.update((cost,));disagreements+=int(cost!=imitation)
        audit.append({'spec':s['spec'],'variant':s['variant'],'t':t,
                      'state_sha256':w.state_sha(),'target':list(target),'menu':menu,
                      'imitation':imitation,'cost':cost,'scores':scores})
    return (np.stack(x_all),np.stack(m_all),np.asarray(y_im,np.int64),
            np.asarray(y_cost,np.int64),audit,
            {'split':'train','stage':stage,'student_visited':seen,'sample_n':len(visited),
             'deviation_actions':forced,'student_early_stops':stop_count,
             'cost_disagreements':disagreements,'cost_labels':dict(sorted(counts.items()))})


def train(seed,base_updates=60,adapt_updates=20,sample_n=256,lookahead=5):
    if not (1<=base_updates<=100 and 1<=adapt_updates<=60):
        raise ValueError('bounded Adam budgets')
    initial,hist=prior.train(seed,base_updates)
    frozen=initial['goal_recovery']
    models={'frozen':frozen,'imitation':copy.deepcopy(frozen),
            'cost_mixture':copy.deepcopy(frozen)}
    training=[];ledgers=[]
    for stage in STAGES:
        X,M,yi,yc,audit,meta=training_corpus(frozen,seed,stage,sample_n,lookahead)
        # Sample indices are identical for every trained arm. Cost arm receives
        # 70% scored labels and 30% teacher labels, deterministic selection.
        rng=np.random.default_rng(seed*677+stage)
        cost_gate=rng.random(len(yi))<.7
        mix=np.where(cost_gate,yc,yi)
        fit={}
        for arm,y in [('imitation',yi),('cost_mixture',mix)]:
            fit[arm]=models[arm].fit(X,M,y,adapt_updates,seed*313+stage*7)
        meta['mixture_cost_examples']=int(cost_gate.sum())
        meta['mixture_changed_examples']=int(np.sum(yi!=mix))
        training.append({'stage':stage,'corpus':meta,'optimizer':fit})
        ledgers.append({'stage':stage,'sampled_counterfactuals':audit,
                        'cost_gate':cost_gate.astype(int).tolist()})
    return models,{'initial_training':hist,'adaptation':training},ledgers


def evaluate(models,seed,quick=False,save_png=None):
    cases=fixture.cases()[:1] if quick else fixture.cases()
    episodes=[]
    for sp in cases:
        for horizon in ((128,) if quick else (128,160)):
            for mode in (('retarget',) if quick else EVAL_MODES):
                for arm in ARMS:
                    png=(save_png/f'{sp.ident}-{seed}-{arm}.png'
                        if save_png is not None and sp==cases[0]
                        and horizon==160 and mode=='retarget' else None)
                    ep=fixture.run_episode(sp,{'goal_recovery':models[arm]},
                                            'goal_recovery',seed,horizon,mode,png)
                    fixture.verify_episode(ep,sp,{'goal_recovery':models[arm]})
                    ep['trained_arm']=arm
                    episodes.append(ep)
    return episodes


def summarize(episodes):
    output={}
    for mode in EVAL_MODES:
        output[mode]={}
        for arm in ARMS:
            rows=[e for e in episodes if e['mode']==mode and e['trained_arm']==arm]
            if not rows:continue
            action=Counter();[action.update(e['action_counts']) for e in rows]
            windows=[v for e in rows for v in e['switch_windows'].values()]
            post_painted=sum(w['painted'] for w in windows) if mode=='retarget' else None
            output[mode][arm]={
                'episodes':len(rows),'targets':sum(len(e['reached_epochs']) for e in rows),
                'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in rows),4),
                'early_stops':sum(bool(e['metrics']['stopped_early']) for e in rows),
                'total_recoveries':sum(e['metrics']['recoveries'] for e in rows),
                'post_switch_painted_8step':post_painted,
                'mean_switch_gain':round(statistics.mean(w['net_distance_gain'] for w in windows),4) if mode=='retarget' and windows else None,
                'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in rows),6),
                'actions':dict(sorted(action.items()))}
    return output


def provenance():
    p=Path(__file__).parent
    return {name:sha_file(p/name) for name in SOURCE_FILES}


def run(out,seed=621,base_updates=60,adapt_updates=20,sample_n=256,
        lookahead=5,quick=False):
    if out.exists():raise FileExistsError('immutable output directory already exists')
    begin=time.perf_counter();out.mkdir(parents=True)
    models,hist,ledgers=train(seed,base_updates,adapt_updates,sample_n,lookahead)
    checkpoint={}
    for arm,model in models.items():
        cp=out/(arm+'.npz');model.save(cp)
        restored=prior.Ranker.load(cp)
        if any(not np.array_equal(restored.p[k],model.p[k]) for k in model.p):
            raise ValueError('checkpoint restore mismatch')
        checkpoint[arm]={'sha256':sha_file(cp),'params':sum(p.size for p in model.p.values()),
                         'optimizer_steps':model.updates}
    (out/'png').mkdir();(out/'traces').mkdir()
    episodes=evaluate(models,seed,quick,out/'png')
    for i,e in enumerate(episodes):
        (out/'traces'/f'{i:03d}-{e["trained_arm"]}.json').write_text(
            json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
    (out/'training_counterfactuals.json').write_text(json.dumps(ledgers,sort_keys=True)+'\n')
    report={'version':VERSION,'seed':seed,'base_updates_per_stage':base_updates,
            'adapt_updates_per_stage':adapt_updates,'lookahead':lookahead,
            'samples_per_stage':sample_n,'training':hist,'checkpoints':checkpoint,
            'evaluation':summarize(episodes),'episodes':len(episodes),
            'decisions':sum(len(e['rows']) for e in episodes),
            'source_sha256':provenance(),'evaluation_split':'development',
            'sealed_structural_heldout_opened':False,'native_receipts':0,'paid_compute':False,
            'elapsed_s':round(time.perf_counter()-begin,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
            for p in out.rglob('*') if p.is_file()},sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    report=json.loads((out/'report.json').read_text())
    if report['version']!=VERSION or report['source_sha256']!=provenance():
        raise ValueError('source mismatch')
    observed={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*')
              if f.is_file() and f.name!='SHA256SUMS.json'}
    if observed!=json.loads((out/'SHA256SUMS.json').read_text()):
        raise ValueError('artifact hash mismatch')
    models={arm:prior.Ranker.load(out/(arm+'.npz')) for arm in ARMS}
    for arm in ARMS:
        cp=out/(arm+'.npz')
        if sha_file(cp)!=report['checkpoints'][arm]['sha256'] or \
           models[arm].updates!=report['checkpoints'][arm]['optimizer_steps']:
            raise ValueError('checkpoint identity mismatch')
    sp_map={s.ident:s for s in fixture.cases()};episodes=[]
    for file in sorted((out/'traces').glob('*.json')):
        e=json.loads(file.read_text());arm=e['trained_arm']
        if arm not in models:raise ValueError('unknown model arm')
        fixture.verify_episode(e,sp_map[e['spec']],{'goal_recovery':models[arm]})
        episodes.append(e)
    if (len(episodes)!=report['episodes'] or
        sum(len(e['rows']) for e in episodes)!=report['decisions']):
        raise ValueError('episode counts mismatch')
    aggregate=summarize(episodes)
    for mode in aggregate:
        for arm in aggregate[mode]:
            actual=dict(aggregate[mode][arm]);stored=dict(report['evaluation'][mode][arm])
            actual.pop('mean_latency_ms');stored.pop('mean_latency_ms')
            if actual!=stored:raise ValueError('aggregate mismatch')
    return {'checkpoints':len(models),'episodes':len(episodes),
            'decisions':report['decisions'],'verified_hashes':len(observed)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=621)
    p.add_argument('--base-updates',type=int,default=60)
    p.add_argument('--adapt-updates',type=int,default=20)
    p.add_argument('--sample-n',type=int,default=256)
    p.add_argument('--lookahead',type=int,default=5)
    p.add_argument('--quick',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(
        a.out,a.seed,a.base_updates,a.adapt_updates,a.sample_n,a.lookahead,a.quick),
        sort_keys=True))

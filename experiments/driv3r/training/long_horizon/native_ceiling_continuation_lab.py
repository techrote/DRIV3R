"""NATIVE-CEILING-CONTINUATION-001: genuine 36/72/128 synthetic decisions.

The native 64-logical-move limit is an *analysis-only censor*: this module
never changes World.menu, World.apply, native code, or the immutable d_common.
TRAIN and DEVELOPMENT are disjoint; structural HELDOUT is never instantiated.
"""
from __future__ import annotations
import argparse, hashlib, json, random, resource, statistics, time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
from .world import MOVE, World, digest, replay, specs
from .policy import Ranker, teacher, random_legal_persistent
from .train import make_corpus, sha_file

VERSION='native-ceiling-continuation/1'
STAGES=(36,72,128)
SEEDS=(1051,1052)
MAX_RECOVERIES=12
NATIVE_LOGICAL_LIMIT=64
STEPS_PER_STAGE=64
SOURCE_FILES=('native_ceiling_continuation_lab.py','world.py','policy.py','train.py')


def dev_specs():
    """Eight ordinary and four trap states from DEVELOPMENT only."""
    ordinary=specs('development',2)
    traps=[replace(s,challenge='trap_corridor') for s in specs('development',1)]
    rows=ordinary+traps
    assert len(rows)==12 and all(s.split=='development' for s in rows)
    return rows


def fit(out):
    models={seed:Ranker(seed) for seed in SEEDS}
    logs=[]; checkpoint={}
    for horizon in STAGES:
        for seed,model in models.items():
            X,M,Y,meta=make_corpus(horizon,seed,2,adversarial_fraction=.18)
            result=model.optimize(X,M,Y,STEPS_PER_STAGE,batch=64,lr=.008,seed=seed*919+horizon)
            path=out/'checkpoints'/f'{seed}-{horizon}.npz'
            model.checkpoint(path)
            restored=Ranker.restore(path)
            assert all(np.array_equal(model.params[k],restored.params[k]) for k in model.params)
            checkpoint[f'{seed}-{horizon}']=sha_file(path)
            logs.append({'seed':seed,'stage':horizon,'train_examples':len(Y),
                        'train_family_digest':meta['original_worlds_sha256'],
                        'train_labels':meta['labels'],'optimizer':result})
    return logs, checkpoint


def intervention(w,t,chosen,enabled):
    """Inject game-host-legally executed errors, never relabel training corpus."""
    if not enabled:return chosen,None
    legal=w.menu(MAX_RECOVERIES)
    if w.spec.challenge=='trap_corridor' and t<6:
        desired='MOVE_E'
    elif t in (18,45,80):
        moves=sorted(a for a in legal if a in MOVE and a!=chosen)
        desired=moves[(w.spec.seed+t)%len(moves)] if moves else chosen
    else:return chosen,None
    actual=desired if desired in legal else chosen
    return actual,{'requested':desired,'executed':actual,'changed':actual!=chosen}


def one_episode(spec,horizon,arm,model,forced,seed):
    w=World.create(spec);rng=random.Random(seed*3509+spec.seed*17+horizon)
    probs=[];overrides=[];lat=[];action_counts=Counter()
    for t in range(horizon):
        if w.stopped:break
        legal=w.menu(MAX_RECOVERIES);tick=time.perf_counter_ns()
        if arm=='teacher':
            chosen,*_=teacher(w,horizon,rng)
        elif arm=='random_persistent':
            chosen=random_legal_persistent(w,rng)
        else:
            assert model is not None
            chosen,p=model.choose(w,horizon)
            # Store exact *offered* legal action distributions for audit.
            probs.append({'step':t,'legal':legal,'p':p,'chosen':chosen,
                          'pre_sha256':w.state_sha()})
        lat.append((time.perf_counter_ns()-tick)/1e6)
        assert chosen in legal
        action,override=intervention(w,t,chosen,forced)
        if override is not None:overrides.append({'step':t,**override})
        receipt=w.apply(action,MAX_RECOVERIES)
        assert receipt['legal']==legal and receipt['chosen']==action
        action_counts[action]+=1
    assert replay(spec,w.decisions,MAX_RECOVERIES).state_sha()==w.state_sha()
    # Native logical-move ceiling is NOT a synthetic host action budget.
    # Censor the recorded trace; do not rewrite menus or pretend native execution.
    physical=[e for e in w.decisions if e['logical_moves']<=NATIVE_LOGICAL_LIMIT]
    assert physical==w.decisions[:len(physical)]
    floor=physical[-1]['logical_moves'] if physical else 0
    first_excess=next((e['index'] for e in w.decisions if e['logical_moves']>NATIVE_LOGICAL_LIMIT),None)
    if first_excess is not None: assert first_excess==len(physical)
    a={'world':spec.ident,'family':spec.family,'challenge':spec.challenge,'seed':seed,
       'horizon':horizon,'arm':arm,'forced':forced,'steps':w.steps,
       'logical_moves':w.logical_moves,'painted':len(w.ink),
       'recoveries':w.recoveries,'stopped':w.stopped,
       'reached_horizon':w.steps==horizon,'prefix_64_decisions':len(physical),
       'prefix_64_logical_moves':floor,'excess_first_index':first_excess,
       'actions':dict(sorted(action_counts.items())),
       'rejected_or_protected_edits':len(w.ink & (set(w.materials)|w.protected)),
       'relocations':action_counts['RELOCATE_START'],
       'forced_changed':sum(x['changed'] for x in overrides),
       'mean_model_choice_ms':round(statistics.mean(lat),6) if lat else None,
       'trace_sha256':digest(w.decisions),'end_sha256':w.state_sha()}
    raw={'summary':a,'spec':{'split':spec.split,'family':spec.family,'seed':spec.seed,
                            'mode':spec.mode,'challenge':spec.challenge},
         'events':w.decisions,'probabilities':probs,'overrides':overrides}
    return a,raw


def run(output):
    out=Path(output)
    if out.exists():raise FileExistsError('refuse to overwrite evidence')
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    started=time.perf_counter();logs,checkpoints=fit(out)
    rows=[];all_traces=[]
    for horizon in STAGES:
        for sp in dev_specs():
            for forced in (False,True):
                for arm,seed in [('teacher',0),('random_persistent',0),('neural',*SEEDS[:1]),('neural',SEEDS[1])]:
                    model=None
                    if arm=='neural':
                        model=Ranker.restore(out/'checkpoints'/f'{seed}-{horizon}.npz')
                    summary,raw=one_episode(sp,horizon,arm,model,forced,seed)
                    rows.append(summary);all_traces.append(raw)
    assert len(rows)==len(STAGES)*len(dev_specs())*2*4
    (out/'traces.json').write_text(json.dumps(all_traces,separators=(',',':'),sort_keys=True)+'\n')
    tables={}
    for horizon in STAGES:
      for forced in (False,True):
       for arm in ('teacher','random_persistent','neural'):
        group=[r for r in rows if r['horizon']==horizon and r['forced']==forced and r['arm']==arm]
        tables[f'{horizon}/{int(forced)}/{arm}']={
           'n':len(group),'reached_nominal':sum(r['reached_horizon'] for r in group),
           'mean_executed_decisions':round(statistics.mean(r['steps'] for r in group),3),
           'mean_prefix_64_decisions':round(statistics.mean(r['prefix_64_decisions'] for r in group),3),
           'native_limit_exceeded':sum(r['excess_first_index'] is not None for r in group),
           'mean_ink':round(statistics.mean(r['painted'] for r in group),3),
           'mean_recoveries':round(statistics.mean(r['recoveries'] for r in group),3),
           'relocations':sum(r['relocations'] for r in group),
           'forced_changed':sum(r['forced_changed'] for r in group)}
    evidence={'experiment':'NATIVE-CEILING-CONTINUATION-001','version':VERSION,
       'remote_source_head_at_start':'13893b492ca023ba13694c7416b704243c0f58e2',
       'source_sha256':{f:sha_file(Path(__file__).parent/f) for f in SOURCE_FILES},
       'checkpoints_sha256':checkpoints,'train_logs':logs,'development_worlds':len(dev_specs()),
       'train_split':'train','eval_split':'development','structural_heldout_opened':False,
       'native_execution':False,'native_64_limit_analytical_only':True,
       'evaluated_episodes':len(rows),'executed_synthetic_actions':sum(r['steps'] for r in rows),
       'full_original_trace_sha256':sha_file(out/'traces.json'),
       'action_totals':dict(sorted(sum((Counter(r['actions']) for r in rows),Counter()).items())),
       'per_episode':rows,'summary':tables,'wall_s':round(time.perf_counter()-started,3),
       'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'median_measured_choice_ms':statistics.median(r['mean_model_choice_ms'] for r in rows if r['mean_model_choice_ms'] is not None),
       'limitations':['Synthetic World only, no physical native receipts.',
         '64-logical-move comparison censors evidence; it does not simulate native enforcement.',
         'Synthetic RELOCATE_START is not validated as native host functionality.',
         'TRAIN/DEVELOPMENT geometries are known families; structural HELDOUT not opened.',
         'Teacher and persistent-random controls have different training privileges than neural model.']}
    (out/'evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'episodes':len(rows),'decisions':evidence['executed_synthetic_actions'],
                     'wall_s':evidence['wall_s'],'peak_rss_kib':evidence['peak_rss_kib'],
                     'table':tables},indent=2))
    return evidence

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--output',required=True)
    run(pa.parse_args().output)

"""GOAL-TIME-TRANSFER-001: TEST measured target arrival time as supervision.

Development is autonomous on source-faithful synthetic one-cell CyberSand-style
terrain; heldout untouched. All edits to terrain happen before any legal action.
Checkpoint-derived suffix labels are TRAIN-only. No native acceptance claim.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import replace
import argparse, copy, hashlib, json, math, random, resource, statistics, time
from pathlib import Path
import numpy as np
from . import recovery_debt_audit as source, retrace_choice_lab as inputs, contrastive_fork_lab as rotation, sequential_probe_lab as neural, policy
from .world import World, Spec, specs, SOLID, MOVE
from .train import sha_file

VERSION = 'goal-time-transfer/1'
STAGES = (36,72,128)
ARMS = ('frozen','arrival','paint')
SEEDS = (811,814)
SUFFIX = ('frozen','contrastive')
TRAIN_LENGTHS = ((2,2),(4,3))
DEV_LENGTHS = ((3,4),(5,2))
MAX_RECOVERIES = 12
GOAL = ('north','south')
SRC = ('goal_time_transfer_lab.py','recovery_debt_audit.py','retrace_choice_lab.py',
       'sequential_probe_lab.py','contrastive_fork_lab.py','goal_recovery_lab.py',
       'goal_conditioning_lab.py','world.py','policy.py','train.py')


def fixture(sp, turns, north_len, side_len):
    if sp.split not in ('train','development') or sp.challenge!='normal' or turns not in range(4):
        raise ValueError('split/rotation violation: HELDOUT is sealed')
    if (north_len,side_len) not in TRAIN_LENGTHS+DEV_LENGTHS:raise ValueError('unregistered geometry')
    w=World.create(replace(sp,challenge='trap_corridor'))
    # Build independent static authored terrain BEFORE applying any host actions.
    for x in range(49,71): w.materials.pop((x,48),None);w.protected.discard((x,48))
    for x in range(49,56):
        for y in (47,49):w.materials[(x,y)]=SOLID
    w.materials[(55,48)]=SOLID;w.protected.add((55,48))
    for x in range(48,53):
        for y in (47,49):w.materials.pop((x,y),None);w.protected.discard((x,y))
    for pt in ((53,47),(54,49),(47,48)):
        w.materials.pop(pt,None);w.protected.discard(pt)
    w.materials[(53,49)]=SOLID
    # Two geometrically distinct exits, each requiring incompatible first move.
    for y in range(48-north_len,48):
        w.materials.pop((53,y),None);w.protected.discard((53,y))
        for x in ((54,) if y==47 else (52,54)):w.materials[(x,y)]=SOLID
    for y in range(49,49+side_len):
        w.materials.pop((54,y),None);w.protected.discard((54,y))
        for x in (53,55):w.materials[(x,y)]=SOLID
    # Retain entry to the south spur and the north exit at x=53.
    w.materials.pop((53,47),None);w.protected.discard((53,47))
    w.materials.pop((54,49),None);w.protected.discard((54,49))
    w.materials[(54,48-north_len-1)]=SOLID
    w.materials[(54,49+side_len)]=SOLID
    w.materials={rotation.rotate_xy(p,turns):v for p,v in w.materials.items()}
    w.protected={rotation.rotate_xy(p,turns) for p in w.protected}
    act=lambda a:rotation.rotate_action(a,turns)
    prefix=[]
    prefix.append(w.apply(act('MOVE_W'),MAX_RECOVERIES));prefix.append(w.apply('BACKTRACK',MAX_RECOVERIES))
    for x in range(48,55):
        if x<53:
            for a in ('MOVE_N','MOVE_S'):
                prefix.append(w.apply(act(a),MAX_RECOVERIES));prefix.append(w.apply('BACKTRACK',MAX_RECOVERIES))
        if x<54:prefix.append(w.apply(act('MOVE_E'),MAX_RECOVERIES))
    if len(prefix)!=28 or w.cursor!=rotation.rotate_xy((54,48),turns) or w.recoveries!=11:
        raise ValueError('prefix invariant')
    side=act('MOVE_S')
    if side not in w.menu(12) or 'BACKTRACK' not in w.menu(12):raise ValueError('fork not host legal')
    targets={'north':rotation.rotate_xy((53,48-north_len),turns),
             'south':rotation.rotate_xy((54,48+side_len),turns)}
    if any(t in w.materials or t in w.protected or t in w.ink for t in targets.values()):
        raise ValueError('goal must be initially reachable and unoccupied')
    return {'world':w,'prefix':prefix,'turns':turns,'side':side,'goal':targets['north'],
            'targets':targets,'geometry':[north_len,side_len], 'spec':sp.__dict__}


def observation(q,horizon):
    return inputs.inputs(q,horizon)


def follow(q,model,first,horizon=96,keep=False):
    w=q['world']; t=q['goal'];start_painted=len(w.ink);rows=[];actions=Counter();goal_step=None;lat=[]
    initial=w.state_sha()
    for i in range(horizon):
        if w.stopped:break
        menu,x=observation(q,horizon)
        begin=time.perf_counter_ns();p=model.probs(x,len(menu));lat.append((time.perf_counter_ns()-begin)/1e6)
        action=(first if i==0 and first is not None else menu[int(p.argmax())])
        if action not in menu:raise ValueError('illegal forced action')
        receipt=w.apply(action,12);actions[action]+=1
        if w.cursor==t and goal_step is None: goal_step=i+1
        if keep: rows.append({'menu':menu,'probabilities':{a:round(float(p[j]),8) for j,a in enumerate(menu)},'action':action,'event':receipt})
    return {'initial_sha256':initial,'final_sha256':w.state_sha(),
            'goal_step':goal_step,'capped_goal_step':goal_step or horizon+1,
            'target_reached':goal_step is not None,'painted':len(w.ink)-start_painted,
            'early_stop':w.stopped and w.steps<len(q['prefix'])+horizon,
            'decisions':sum(actions.values()),'actions':dict(sorted(actions.items())),
            'latency_mean_ms':round(statistics.mean(lat),6) if lat else 0.,
            'rows':rows if keep else None,'final_cursor':list(w.cursor),
            'target':list(t)}


def paired_suffix_labels(sp,turns,lengths,models):
    cases={}
    for g in GOAL:
        times=[];paints=[];record=[]
        for suffix in SUFFIX:
            outcomes={}
            for first in ('BACKTRACK','SIDE'):
                q=fixture(sp,turns,*lengths);q['goal']=q['targets'][g]
                action=q['side'] if first=='SIDE' else 'BACKTRACK'
                outcomes[first]=follow(q,models[suffix],action,horizon=96)
            b,s=outcomes['BACKTRACK'],outcomes['SIDE']
            if b['initial_sha256']!=s['initial_sha256']:raise ValueError('counterfactual state drift')
            times.append(s['capped_goal_step']-b['capped_goal_step'])
            paints.append(b['painted']-s['painted'])
            record.append({'suffix':suffix,'time_advantage_backtrack':times[-1],
                           'paint_advantage_backtrack':paints[-1],
                           'backtrack_arrival':b['goal_step'],'side_arrival':s['goal_step']})
        # Accept supervision only where BOTH different existing suffix policies
        # agree on a nontrivial time-to-goal improvement (no proxy label).
        if min(times)>=3: winner='BACKTRACK'
        elif max(times)<=-3: winner='SIDE'
        else: winner=None
        if min(paints)>=3:paint_winner='BACKTRACK'
        elif max(paints)<=-3:paint_winner='SIDE'
        else:paint_winner=None
        cases[g]={'arrival_label':winner,'paint_label':paint_winner,'suffixes':record}
    return cases


def corpus(seed,baseline,seed_list=None):
    result=[];X=[];M=[];Y={'arrival':[],'paint':[]};index={'arrival':[],'paint':[]}
    specs_train=specs('train',2) if seed_list is None else seed_list
    for sp in specs_train:
        for turns in range(4):
            for lengths in TRAIN_LENGTHS:
                labels=paired_suffix_labels(sp,turns,lengths,baseline)
                for g in GOAL:
                    q=fixture(sp,turns,*lengths);q['goal']=q['targets'][g]
                    menu,x=observation(q,128)
                    mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                    row={'split':'train','spec':sp.__dict__,'turns':turns,'lengths':list(lengths),
                         'goal':g,'menu':menu,'initial_sha256':q['world'].state_sha(),
                         'world_base_sha256':q['world'].base_sha(),
                         'features_sha256':hashlib.sha256(x.tobytes()).hexdigest(),**labels[g]}
                    result.append(row);X.append(x);M.append(mask)
                    for arm in ('arrival','paint'):
                        lab=row[arm+'_label'];
                        if lab is not None:
                            index[arm].append(len(X)-1);Y[arm].append(menu.index(q['side'] if lab=='SIDE' else 'BACKTRACK'))
    return np.stack(X),np.stack(M),{a:np.asarray(v,dtype=np.int64) for a,v in Y.items()},index,result


def train(seed,archive,out,steps=45):
    baseline,hashes=source.restored(archive,seed)
    X,M,Y,indices,ledger=corpus(seed,baseline)
    models={'frozen':baseline['contrastive']}
    stats={}
    for a in ('arrival','paint'):
        models[a]=copy.deepcopy(models['frozen'])
        if len(indices[a])>=16:
            stats[a]=models[a].fit(X[indices[a]],M[indices[a]],Y[a],steps,seed*129+5)
            stats[a]['accepted_examples']=len(indices[a]);stats[a]['trained']=True
        else:
            stats[a]={'accepted_examples':len(indices[a]),'updates':0,'trained':False,'reason':'insufficient robust labels'}
    for a,m in models.items():
        p=out/(str(seed)+'-'+a+'.npz');m.save(p)
        loaded=neural.ProbeRanker.load(p)
        if any(not np.array_equal(m.p[k],loaded.p[k]) for k in m.p):raise ValueError('restoration mismatch')
    return models, hashes, stats, ledger


def episode(sp,seed,arm,model,turns,lengths,goal_name,horizon=128):
    if sp.split!='development' or lengths not in DEV_LENGTHS or horizon!=128:
        raise ValueError('DEVELOPMENT only')
    q=fixture(sp,turns,*lengths);q['goal']=q['targets'][goal_name]
    initial=q['world'].state_sha();base=q['world'].base_sha()
    outcome=follow(q,model,None,horizon,keep=True)
    return {'version':VERSION,'seed':seed,'arm':arm,'split':'development',
            'spec':sp.__dict__,'turns':turns,'lengths':list(lengths),
            'goal':goal_name,'prefix':q['prefix'],'base_sha256':base,
            'initial_sha256':initial,'outcome':outcome}


def verify_episode(e,models):
    sp=Spec(**e['spec']);q=fixture(sp,e['turns'],*e['lengths'])
    q['goal']=q['targets'][e['goal']]
    if sp.split!='development' or tuple(e['lengths']) not in DEV_LENGTHS or e['version']!=VERSION:
        raise ValueError('invalid evaluation identity')
    if q['prefix']!=e['prefix'] or q['world'].state_sha()!=e['initial_sha256'] or q['world'].base_sha()!=e['base_sha256']:
        raise ValueError('prefix or geometry tampering')
    actual=follow(q,models[e['arm']],None,128,keep=True)
    # Timing varies between calls, compare all immutable source/model/material claims.
    for key in actual:
        if key=='latency_mean_ms':continue
        if actual[key]!=e['outcome'][key]:raise ValueError('episode model/receipt mismatch: '+key)
    return actual['decisions']


def summarize(episodes):
    result={}
    for arm in ARMS:
        es=[e['outcome'] for e in episodes if e['arm']==arm]
        moves=Counter();[moves.update(e['actions']) for e in es]
        result[arm]={'episodes':len(es),'targets_reached':sum(e['target_reached'] for e in es),
          'mean_capped_arrival':round(statistics.mean(e['capped_goal_step'] for e in es),3),
          'mean_new_paint':round(statistics.mean(e['painted'] for e in es),3),
          'forced_stop':sum(e['early_stop'] for e in es),
          'mean_decisions':round(statistics.mean(e['decisions'] for e in es),3),
          'mean_neural_latency_ms':round(statistics.mean(e['latency_mean_ms'] for e in es),6),
          'actions':dict(sorted(moves.items()))}
    return result


def manifest(out):
    return {str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}


def run(archive,out,seeds=SEEDS,steps=45):
    if out.exists():raise FileExistsError('immutable evidence directory exists')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'episodes').mkdir()
    begin=time.perf_counter();report={'version':VERSION,'archive_sha256':sha_file(archive),
      'source_sha256':{name:sha_file(Path(__file__).parent/name) for name in SRC},
      'world_partition':{'train':[s.__dict__ for s in specs('train',2)],
                         'development':[s.__dict__ for s in specs('development',1)],
                         'heldout_opened':False},
      'split_goals':{'train_lengths':TRAIN_LENGTHS,'development_lengths':DEV_LENGTHS},
      'seeds':list(seeds),'source_checkpoint_sha256':{},'checkpoint_sha256':{},
      'optimizer':{},'training_labels':{},'early_goal_stop_policy':'normal host-legal STOP',
      'native_receipts':0,'paid_compute':False,'comparison':{}}
    episodes=[]
    for seed in seeds:
        models,sha,stats,ledger=train(seed,archive,out/'checkpoints',steps)
        report['source_checkpoint_sha256'][str(seed)]=sha
        report['checkpoint_sha256'][str(seed)]={a:sha_file(out/'checkpoints'/f'{seed}-{a}.npz') for a in ARMS}
        report['optimizer'][str(seed)]=stats
        report['training_labels'][str(seed)]=ledger
        for sp in specs('development',1):
            for turns in (0,1,2,3):
                for lengths in DEV_LENGTHS:
                    for goal_name in GOAL:
                        for arm in ARMS:
                            e=episode(sp,seed,arm,models[arm],turns,lengths,goal_name)
                            (out/'episodes'/f'{len(episodes):04d}.json').write_text(json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
                            episodes.append(e)
    report['comparison']=summarize(episodes)
    report['episodes']=len(episodes);report['decisions']=sum(e['outcome']['decisions'] for e in episodes)
    report['elapsed_s']=round(time.perf_counter()-begin,3)
    report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest(out),sort_keys=True,indent=2)+'\n')
    return report


def verify(archive,out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['archive_sha256']!=sha_file(archive):raise ValueError('archive mismatch')
    if r['source_sha256']!={n:sha_file(Path(__file__).parent/n) for n in SRC}:raise ValueError('source drift')
    if json.loads((out/'SHA256SUMS.json').read_text())!=manifest(out):raise ValueError('payload tampering')
    n=0;d=0
    for seed in r['seeds']:
        models={}
        for a in ARMS:
            p=out/'checkpoints'/f'{seed}-{a}.npz'
            if r['checkpoint_sha256'][str(seed)][a]!=sha_file(p):raise ValueError('checkpoint drift')
            models[a]=neural.ProbeRanker.load(p)
        for f in sorted((out/'episodes').glob('*.json')):
            e=json.loads(f.read_text())
            if e['seed']!=seed:continue
            d+=verify_episode(e,models);n+=1
    return {'verified_episodes':n,'verified_decisions':d,
            'verified_files':len(manifest(out)),
            'verified_checkpoints':len(r['seeds'])*len(ARMS)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--verify',action='store_true')
    ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));ap.add_argument('--steps',type=int,default=45)
    a=ap.parse_args();print(json.dumps(verify(a.archive,a.out) if a.verify else run(a.archive,a.out,a.seeds,a.steps),sort_keys=True))

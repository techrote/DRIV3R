"""Finalize/independently verify bounded single-seed GOAL-TIME-TRANSFER evidence.

Keeps the original executed trainer file byte-identical, retains all negative
results, and does not quietly invent an uncompleted second training seed.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json, resource, statistics, time
from pathlib import Path
import numpy as np
from . import goal_time_transfer_lab as lab
from . import recovery_debt_audit as prior, sequential_probe_lab as neural
from .train import sha_file


def prepare(archive,out,seed=811):
    if (out/'report.json').exists() or (out/'SHA256SUMS.json').exists():raise FileExistsError('refuse overwrite report or manifest')
    source_models,source_hashes=prior.restored(archive,seed)
    X,M,Y,indices,ledger=lab.corpus(seed,source_models)
    ckpts={};models={}
    for arm in lab.ARMS:
        path=out/'checkpoints'/f'{seed}-{arm}.npz'
        if not path.is_file():raise FileNotFoundError(path)
        models[arm]=neural.ProbeRanker.load(path)
        ckpts[arm]=sha_file(path)
    # Strong assurance that the frozen baseline has unchanged neural weights.
    for k in models['frozen'].p:
        if not np.array_equal(source_models['contrastive'].p[k],models['frozen'].p[k]):
            raise ValueError('frozen control weights modified')
    for arm in ('arrival','paint'):
        if models[arm].updates-models['frozen'].updates!=45:raise ValueError('wrong optimizer budget')
        if len(indices[arm])<16:raise ValueError('not enough robust labels for completed run')
    files=sorted((out/'episodes').glob('*.json'))
    if len(files)!=192 or [p.name for p in files]!=[f'{i:04d}.json' for i in range(192)]:
        raise ValueError('single-seed evaluation incomplete')
    episodes=[json.loads(f.read_text()) for f in files]
    if any(e['seed']!=seed or e['arm'] not in lab.ARMS for e in episodes):raise ValueError('foreign episode')
    configurations=Counter((e['spec']['family'],e['spec']['seed'],e['turns'],tuple(e['lengths']),e['goal'],e['arm']) for e in episodes)
    if len(configurations)!=192 or set(configurations.values())!={1}:
        raise ValueError('duplicated/missing evaluation fixture')
    compare=lab.summarize(episodes)
    bygoal={}
    for arm in lab.ARMS:
        bygoal[arm]={}
        for g in lab.GOAL:
            es=[e['outcome'] for e in episodes if e['goal']==g and e['arm']==arm]
            bygoal[arm][g]={'episodes':len(es),'reached':sum(x['target_reached'] for x in es),
              'mean_capped_first_visit':round(statistics.mean(x['capped_goal_step'] for x in es),3),
              'mean_new_paint':round(statistics.mean(x['painted'] for x in es),3),
              'stops':sum('STOP' in x['actions'] for x in es),
              'first_actions':dict(sorted(Counter(x['rows'][0]['action'] for x in es).items()))}
    report={'version':lab.VERSION,'status':'bounded single-seed result; second seed not completed',
      'archive_sha256':sha_file(archive),'source_sha256':{n:sha_file(Path(lab.__file__).parent/n) for n in lab.SRC},
      'source_checkpoint_sha256':{str(seed):source_hashes},
      'checkpoint_sha256':{str(seed):ckpts},'source_checkpoint_update_count':models['frozen'].updates,
      'adaptation_updates_per_arm':45,'parameters_per_policy':sum(x.size for x in models['frozen'].p.values()),
      'training_labels':{'total_cases':len(ledger),'by_arm':{
        a:{'robust_labels':len(indices[a]),'BACKTRACK':sum(ledger[j][a+'_label']=='BACKTRACK' for j in indices[a]),
           'SIDE':sum(ledger[j][a+'_label']=='SIDE' for j in indices[a])} for a in ('arrival','paint')},
        'original_train_only_ledger':ledger},
      'world_partition':{'train_specs':[s.__dict__ for s in lab.specs('train',2)],
                         'development_specs':[s.__dict__ for s in lab.specs('development',1)],
                         'train_geometries':lab.TRAIN_LENGTHS,'development_geometries':lab.DEV_LENGTHS,
                         'rotation_degrees':(0,90,180,270),'heldout_opened':False},
      'evaluation_seeds':[seed],'evaluation':compare,'by_goal':bygoal,
      'episodes':len(episodes),'decisions':sum(e['outcome']['decisions'] for e in episodes),
      'native_receipts':0,'paid_compute':False,
      'provenance_limit':'trainer timed out before second seed; first seed checkpoints and 192 DEVELOPMENT episodes complete',
      'measured_campaign_wall_s':None,'measured_campaign_peak_rss_kib':None,
      'source_report_build_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps(lab.manifest(out),sort_keys=True,indent=2)+'\n')
    return {k:report[k] for k in ('episodes','decisions','training_labels','evaluation','by_goal','checkpoint_sha256') if k!='training_labels'} | {'training_labels':report['training_labels']['by_arm']}


def verify(archive,out,seed=811,first=0,last=192):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=lab.VERSION or r['archive_sha256']!=sha_file(archive) or r['evaluation_seeds']!=[seed]:
        raise ValueError('report identity mismatch')
    if {n:sha_file(Path(lab.__file__).parent/n) for n in lab.SRC}!=r['source_sha256']:
        raise ValueError('dependency or executed trainer source drift')
    if json.loads((out/'SHA256SUMS.json').read_text())!=lab.manifest(out):
        raise ValueError('evidence manifest mismatch')
    models={}
    for arm in lab.ARMS:
        p=out/'checkpoints'/f'{seed}-{arm}.npz'
        if sha_file(p)!=r['checkpoint_sha256'][str(seed)][arm]:raise ValueError('model drift')
        models[arm]=neural.ProbeRanker.load(p)
    if not 0<=first<last<=192:raise ValueError('bad verification range')
    total=0
    for i in range(first,last):
        e=json.loads((out/'episodes'/f'{i:04d}.json').read_text());total+=lab.verify_episode(e,models)
    return {'verified_episode_range':[first,last],'episodes':last-first,'decisions':total,'manifest_files':len(lab.manifest(out))}

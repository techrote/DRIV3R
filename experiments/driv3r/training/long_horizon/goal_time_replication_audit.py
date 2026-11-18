"""GOAL-TIME-REPLICATION-001: seed-814 replication and matched goal-cue ablation.

Read-only checkpoint evaluation in synthetic DEVELOPMENT worlds. No additional
training, native CyberSand ABI claim, or structural HELDOUT inspection.
Requires the original seed-811 and seed-814 evidence directories and a pinned
OUTCOME-FORK checkpoint ZIP; does not modify pre-existing evidence.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, statistics
from pathlib import Path
from collections import Counter
import numpy as np
from . import goal_time_transfer_lab as trial, goal_time_finish_lab as finish
from . import goal_conditioning_lab as goal, recovery_debt_audit as recovery
from . import sequential_probe_lab as neural, policy
from .train import sha_file
from .world import specs

VERSION='goal-time-replication/1'
ARMS=('frozen','arrival','paint')
SEEDS=(811,814)

def restore(evidence, seed):
    report=json.loads((evidence/'report.json').read_text())
    model={}
    for arm in ARMS:
        cp=evidence/'checkpoints'/f'{seed}-{arm}.npz'
        if sha_file(cp)!=report['checkpoint_sha256'][str(seed)][arm]:
            raise ValueError('checkpoint identity mismatch')
        model[arm]=neural.ProbeRanker.load(cp)
    return report,model

def causal_evidence(seed, models, evidence):
    """Paired goal swap on identical host state/menu; separately mask goal cues."""
    result=[]
    for sp in specs('development',1):
        for turns in range(4):
            for lengths in trial.DEV_LENGTHS:
                obs={}
                for goal_name in trial.GOAL:
                    q=trial.fixture(sp,turns,*lengths)
                    q['goal']=q['targets'][goal_name]
                    menu,x=trial.observation(q,128)
                    mask=x.copy()
                    mask[:,len(policy.FEATURES):goal.FEATURE_DIM]=0.
                    obs[goal_name]={'menu':menu,'x':x,'mask':mask,
                                    'state_sha256':q['world'].state_sha(),
                                    'base_sha256':q['world'].base_sha(),
                                    'expected': 'BACKTRACK' if goal_name=='north' else q['side']}
                n,s=obs['north'],obs['south']
                if n['state_sha256']!=s['state_sha256'] or n['base_sha256']!=s['base_sha256'] or n['menu']!=s['menu']:
                    raise ValueError('goal swap changed host legal state/menu')
                if not np.array_equal(n['mask'],s['mask']):
                    raise ValueError('goal mask failed to remove all goal features')
                if np.array_equal(n['x'],s['x']):
                    raise ValueError('unmasked target cue absent')
                pair={'seed':seed,'split':'development','spec':sp.__dict__,'turns':turns,
                      'lengths':list(lengths),'initial_sha256':n['state_sha256'],
                      'base_sha256':n['base_sha256'],'menu':n['menu'],'arms':{}}
                for arm,m in models.items():
                    readings={}
                    for name,t in obs.items():
                        probs=m.probs(t['x'],len(t['menu']))
                        masked=m.probs(t['mask'],len(t['menu']))
                        action=t['menu'][int(probs.argmax())]
                        masked_action=t['menu'][int(masked.argmax())]
                        readings[name]={'goal':name,'expected':t['expected'],
                             'action':action,'masked_action':masked_action,
                             'p':{a:round(float(v),8) for a,v in zip(t['menu'],probs)},
                             'masked_p':{a:round(float(v),8) for a,v in zip(t['menu'],masked)},
                             'correct_first':action==t['expected']}
                        if action not in t['menu'] or masked_action not in t['menu']:
                            raise ValueError('action escaped host legality')
                    if readings['north']['masked_p']!=readings['south']['masked_p']:
                        raise ValueError('goal masking failed paired probability equality')
                    pair['arms'][arm]=readings
                result.append(pair)
    # Validate that the audited first actions actually match the executed,
    # independently source-replayed trajectories, not just a standalone score.
    cases={}
    for p in sorted((evidence/'episodes').glob('*.json')):
        e=json.loads(p.read_text())
        if e['seed']!=seed:continue
        k=(e['spec']['family'],e['spec']['seed'],e['turns'],tuple(e['lengths']),e['goal'],e['arm'])
        if k in cases:raise ValueError('duplicated evaluated case')
        cases[k]=e
    if len(cases)!=192:raise ValueError('incomplete original episode set')
    for pair in result:
        for arm in ARMS:
            for name,reading in pair['arms'][arm].items():
                k=(pair['spec']['family'],pair['spec']['seed'],pair['turns'],tuple(pair['lengths']),name,arm)
                original=cases[k]
                if original['initial_sha256']!=pair['initial_sha256'] or original['outcome']['rows'][0]['action']!=reading['action']:
                    raise ValueError('score/actual executed episode discrepancy')
    return result

def summarize(records):
    result={}
    for seed in SEEDS:
        for arm in ARMS:
            rr=[r['arms'][arm] for r in records if r['seed']==seed]
            nn=[r['north'] for r in rr];ss=[r['south'] for r in rr]
            result[f'{seed}/{arm}']={
              'contexts':len(rr),'correct_north':sum(x['correct_first'] for x in nn),
              'correct_south':sum(x['correct_first'] for x in ss),
              'unmasked_goal_flip':sum(a['action']!=b['action'] for a,b in zip(nn,ss)),
              'masked_goal_flip':sum(a['masked_action']!=b['masked_action'] for a,b in zip(nn,ss)),
              'masked_choices':dict(sorted(Counter(x['masked_action'] for x in nn).items())),
              'mean_p_expected_north':round(statistics.mean(x['p'][x['expected']] for x in nn),6),
              'mean_p_expected_south':round(statistics.mean(x['p'][x['expected']] for x in ss),6)}
    return result

def run(archive, original811, replication814):
    checks={}
    checks['seed811']=finish.verify(archive,original811,811)
    checks['seed814']=trial.verify(archive,replication814)
    if checks['seed811']['episodes']!=192 or checks['seed814']['verified_episodes']!=192:
        raise ValueError('incomplete source replay')
    r811,m811=restore(original811,811)
    r814,m814=restore(replication814,814)
    if r811['archive_sha256']!=sha_file(archive) or r814['archive_sha256']!=sha_file(archive):
        raise ValueError('source archive mismatch')
    pairs=causal_evidence(811,m811,original811)+causal_evidence(814,m814,replication814)
    return {'version':VERSION,'archive_sha256':sha_file(archive),
      'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in
       ('goal_time_replication_audit.py','goal_time_transfer_lab.py','goal_time_finish_lab.py','goal_conditioning_lab.py','recovery_debt_audit.py','sequential_probe_lab.py','world.py','policy.py')},
      'source_checkpoint_sha256':r814['source_checkpoint_sha256'],
      'checkpoint_sha256':{'811':r811['checkpoint_sha256']['811'],'814':r814['checkpoint_sha256']['814']},
      'evidence_manifest_sha256':{'811':sha_file(original811/'SHA256SUMS.json'),
         '814':sha_file(replication814/'SHA256SUMS.json')},
      'replays':checks,'verified_episodes':384,'verified_decisions':checks['seed811']['decisions']+checks['seed814']['verified_decisions'],
      'training_updates_per_adapted_arm':45,
      'evaluation':{'811':r811['evaluation'],'814':r814['comparison']},
      'causal_goal_mask_summary':summarize(pairs),'causal_goal_pairs':pairs,
      'heldout_opened':False,'native_receipts':0,'paid_compute':False}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--seed811',type=Path,required=True)
    ap.add_argument('--seed814',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError('never overwrite original result')
    report=run(args.archive,args.seed811,args.seed814)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('verified_episodes','verified_decisions','causal_goal_mask_summary')},sort_keys=True))

if __name__=='__main__':main()

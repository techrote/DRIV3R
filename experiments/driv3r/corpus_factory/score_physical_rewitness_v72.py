#!/usr/bin/env python3
"""EVALUATOR-ONLY v72 controls; reads certificates, never training stream.

Conservative teacher completion is established by independent physical host audit.
No figures here are neural-model scores. A naive-control goal-loss indicator
is an evaluator logical consequence of omitted physical repairs, not replay.
"""
import argparse
import collections
import json
import pathlib
import statistics

def score(root):
    result = {'schema':'driv3r/corpus_factory/physical-rewitness-v72-controls/1',
              'synthetic_host_not_native_CyberSand':True,
              'neural_model_evaluation':False,'splits':{}}
    for split in ('TRAIN','DEV','HELDOUT'):
        file = root / f'{split}_physical_rewitness_certificates_EVALUATOR_ONLY.jsonl'
        rows = [json.loads(line) for line in file.open(encoding='utf8')]
        groups=collections.defaultdict(list)
        by_witness=collections.defaultdict(list)
        unsafe=0
        for row in rows:
            assert row['terminal_success'], 'teacher host failure certificate'
            key=(row['world_index'],row['threshold'],tuple(row['sequence']))
            groups[key].append(row)
            by_witness[row['witness']].append(row)
            # The source-time latest PRIMARY reading is physically accurate,
            # but treating it as *pre-fault calibrated* after a common-bus
            # inversion misdecodes erosion. This intentionally unsafe control
            # uses no hidden fault flip in its decision rule.
            targets = {m for m, raw in row['late_primary_raw'].items()
                       if (('INTACT' if raw=='ERODED' else 'ERODED')
                           if row['calibration'][m] else raw)=='ERODED'}
            unsafe += bool(set(row['actual_broken'])-targets)
        assert all(len(v)==64 for v in groups.values()), 'incomplete 64-arm group'
        upper=sum(max(collections.Counter(c['teacher_first_action'] for c in group).values())
                  for group in groups.values())/len(rows)
        wstats={}
        for name in ('NONE','A','B','BOTH'):
            r=by_witness[name]
            cost=[c['witness_actions'] for c in r]
            assert all(k=={'NONE':0,'A':2,'B':2,'BOTH':4}[name] for k in cost)
            gross=[c['always_actions']-c['teacher_actions'] for c in r]
            net=[x-y for x,y in zip(gross,cost)]
            wstats[name]={
                'episodes':len(r),
                'mean_gross_action_savings':sum(gross)/len(r),
                'mean_net_action_savings':sum(net)/len(r),
                'negative_net_value_episodes':sum(x<0 for x in net),
                'net_savings_vs_witness_NONE':None}
        baseline=wstats['NONE']['mean_net_action_savings']
        for w in wstats.values():
            w['net_savings_vs_witness_NONE']=w['mean_net_action_savings']-baseline
        result['splits'][split]={
            'episodes':len(rows),
            'counterfactual_groups':len(groups),
            'instantaneous_only_critical_accuracy_upper_bound':upper,
            'aggressive_latest_primary_stale_calibration_goal_losses':unsafe,
            'aggressive_latest_primary_stale_calibration_goal_loss_fraction':unsafe/len(rows),
            'teacher_host_replayed_completions':len(rows),
            'by_witness':wstats}
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('directory')
    args=ap.parse_args()
    print(json.dumps(score(pathlib.Path(args.directory)),indent=2,sort_keys=True))

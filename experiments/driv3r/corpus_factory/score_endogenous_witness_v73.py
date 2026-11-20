#!/usr/bin/env python3
"""Evaluator-only ex-ante probe cost controls. Never use HELDOUT certificates in TRAIN."""
import collections
import json
import pathlib
import argparse

MODES=('NONE','A','B','BOTH')

def evaluate(root):
    result={'schema':'driv3r/corpus_factory/endogenous-witness-v73-controls/1',
            'synthetic_host_not_native_CyberSand':True,
            'neural_model_scores':False,'splits':{}}
    for split in ('TRAIN','DEV','HELDOUT'):
        rows=[json.loads(s) for s in (root/f'{split}_endogenous_witness_certificates_EVALUATOR_ONLY.jsonl').open()]
        groups=collections.defaultdict(list)
        for r in rows:
            assert r['terminal_success']
            key=(r['world_index'],r['threshold'],r['calibration']['A'],r['calibration']['B'],r['fault']['domain'])
            groups[key].append(r)
        choices=collections.Counter(r['selected'] for r in rows)
        oracle=sum(r['policy_means'][r['selected']] for r in rows)/len(rows)
        def gap(selector):
            return sum(r['policy_means'][selector(r)]-r['policy_means'][r['selected']] for r in rows)/len(rows)
        no_probe=gap(lambda r:'NONE')
        always_both=gap(lambda r:'BOTH')
        domain_control=gap(lambda r:'BOTH' if r['fault']['domain']=='COMMON_BUS' else 'NONE')
        # Current-only is allowed to use geography, threshold and past local
        # calibration, but NOT the old witness tariff (identical current obs).
        current_only_regret=0
        dominated=0
        for group in groups.values():
            assert len(group)==16
            best=min(MODES,key=lambda m:sum(r['policy_means'][m] for r in group))
            current_only_regret+=sum(r['policy_means'][best]-r['policy_means'][r['selected']] for r in group)
            dominated+=len(set(r['selected'] for r in group))>1
        result['splits'][split]={
            'episodes':len(rows),'groups':len(groups),'cost_profiles_with_different_optimum':dominated,
            'choices':dict(choices),'oracle_mean_expected_total_actions':oracle,
            'no_probe_mean_regret':no_probe,'always_both_mean_regret':always_both,
            'domain_only_mean_regret':domain_control,'best_current_only_mean_regret':current_only_regret/len(rows),
            'teacher_terminal_successes':sum(r['terminal_success'] for r in rows),
            'mean_actions_before_probe':sum(r['decision_step'] for r in rows)/len(rows),
            'mean_actions_before_recovery':sum(r['critical_step'] for r in rows)/len(rows),
        }
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path)
    print(json.dumps(evaluate(p.parse_args().directory),indent=2,sort_keys=True))

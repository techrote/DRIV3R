"""Research-only active-guidance personality gate for saved synthetic trajectories.

Consumes recorded host-executed episodes; no model, native ABI or game imports.
Scientific uncertainty is explicit: low turn counts cannot prove preservation.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict


def key(row):
    return (row['seed'],row['outcome'],row['directive'],row['personality'],row['arm'])


def metric(rows):
    outcome={}
    for personality in (-1,1):
        group=[r for r in rows if r['personality']==personality]
        left=sum(r['left'] for r in group)
        right=sum(r['right'] for r in group)
        n=left+right
        outcome[personality]={'turns':n,'signed_bias':(left-right)/n if n else None}
    if any(outcome[p]['signed_bias'] is None for p in (-1,1)):
        gap=None
    else:
        gap=outcome[1]['signed_bias']-outcome[-1]['signed_bias']
    return {'n':len(rows),'goals':sum(int(r['reached']) for r in rows),
            'gap':gap,'personality':outcome}


def gate(rows,reference='C18-A0',candidate='C20-STYLE',min_turns=10,max_gap_shift=.10):
    by_model=defaultdict(list)
    for r in rows:by_model[r['model']].append(r)
    if reference not in by_model or candidate not in by_model or 'C12' not in by_model:
        raise ValueError('requires reference, candidate and frozen C12 rows')
    for model in (reference,candidate):
        seen=set()
        for r in by_model[model]:
            k=key(r)
            if k in seen:raise ValueError(f'duplicate episode {model}: {k}')
            seen.add(k)
            if r['host_rejected'] or r['bad']:
                raise ValueError(f'host safety invariant violated: {model}: {k}')
    # No-cue equivalence is exactly matched, not a distributional average.
    baseline={key(r):r for r in by_model['C12'] if r['directive']=='none'}
    uncued=[r for r in by_model[candidate] if r['directive']=='none']
    cue_off_exact=sum(key(r) in baseline and baseline[key(r)]['trace']==r['trace']
                      and baseline[key(r)]['ink_sha256']==r['ink_sha256'] for r in uncued)
    conditions={};failures=[]
    for outcome in ('reverse','ahead','shortcut','bridge'):
        for directive in ('backtrack','continue'):
            selector=lambda model:[r for r in by_model[model] if r['outcome']==outcome and r['directive']==directive]
            a=metric(selector(reference));b=metric(selector(candidate))
            if a['n']!=b['n']:raise ValueError('unmatched cohort length')
            eligible=all(a['personality'][p]['turns']>=min_turns and b['personality'][p]['turns']>=min_turns for p in (-1,1))
            shift=(b['gap']-a['gap']) if eligible and a['gap'] is not None and b['gap'] is not None else None
            condition=f'{outcome}/{directive}'
            reasons=[]
            if b['goals']<a['goals']:reasons.append('goal_regression')
            if not eligible:reasons.append('insufficient_turn_events')
            if shift is not None and abs(shift)>max_gap_shift:reasons.append('style_gap_shift')
            conditions[condition]={'reference':a,'candidate':b,'gap_shift':shift,'reasons':reasons}
            if reasons:failures.append(condition)
    if cue_off_exact!=len(uncued):failures.append('cue_off_drift')
    return {'schema':'guidance-recovery-personality-gate/1','reference':reference,
      'candidate':candidate,'uncued_pairs':len(uncued),'uncued_exact':cue_off_exact,
      'conditions':conditions,'failures':failures,'passed':not failures,
      'gate_parameters':{'min_turns_per_personality':min_turns,'max_abs_gap_shift':max_gap_shift}}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('trajectories_json')
    parser.add_argument('--candidate',default='C20-STYLE')
    args=parser.parse_args()
    with open(args.trajectories_json,encoding='utf-8') as f:rows=json.load(f)
    print(json.dumps(gate(rows,candidate=args.candidate),indent=2,sort_keys=True))

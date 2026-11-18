"""Read-only switch-state feasibility audit for the frozen unseen-mask campaign.

Checks recorded action receipts and pre-switch state identities independently of
model inference. Prefix-plan 'unreachable' is NOT global host-state infeasibility.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from .unseen_mask_stress_lab import novel_world, ARMS, MASKS, CAP
from .detour_recovery_lab import prefix_plan
from .world import Spec


def examine(run:Path)->dict:
    totals={a:{m:{str(stage):Counter() for stage in (1,2)} for m in MASKS} for a in ARMS}
    examined=0
    for src in sorted((run/'traces').glob('*.json')):
        t=json.loads(src.read_text());r=t['row'];w=novel_world(Spec(**t['spec']),r['layout'])
        pending=iter(t['entries']);entry=next(pending,None)
        for switch in t['switches']:
            while entry is not None and entry['tick']<switch['tick']:
                assert w.menu(12)==entry['menu']
                assert w.state_sha()==entry['pre_sha256']
                assert w.apply(entry['action'])==entry['receipt']
                entry=next(pending,None)
            assert w.state_sha()==switch['pre_sha256']
            # Same deterministic restricted planner used by prior intervention controls.
            feasible=prefix_plan(w,tuple(switch['next_goal']),CAP) is not None
            stage=switch['old_phase']+1
            totals[r['arm']][r['layout']][str(stage)]['feasible' if feasible else 'infeasible']+=1
            examined+=1
        # Validate all trailing receipts without changing recorded switch states.
        while entry is not None:
            assert w.menu(12)==entry['menu']
            assert w.state_sha()==entry['pre_sha256']
            assert w.apply(entry['action'])==entry['receipt']
            entry=next(pending,None)
        assert w.state_sha()==r['final_sha256']
    return {'audited_switches':examined,'by_arm_mask_stage':{
        a:{m:{stage:{k:v.get(k,0) for k in ('feasible','infeasible')}
              for stage,v in stages.items()} for m,stages in masks.items()}
        for a,masks in totals.items()}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    result=examine(a.run)
    ref=json.loads(a.reference.read_text())
    assert result['by_arm_mask_stage']==ref['switch_stage_prefix_feasibility']
    print(json.dumps({'switch_states_verified':result['audited_switches'],
                      'reference_match':True,'nested_bays':result['by_arm_mask_stage']['planner_each_stage']['nested_bays']}))

if __name__=='__main__':main()

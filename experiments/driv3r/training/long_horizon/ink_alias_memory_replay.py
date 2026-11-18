"""Independent probability + host-receipt replay of INK-ALIAS-MEMORY-001 archive."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from .ink_alias_memory_lab import (ACTIONS, GOAL, state, probe, replay_receipt,
                                   CHECKPOINT_SEEDS)
from .goal_conditioning_lab import GoalRanker
from .detour_recovery_lab import shortest_empty_path
from .world import Spec, digest

def verify_scientific_record(r, weights, frozen):
    sp=Spec(**r['spec']);w=state(sp,r['depth'],r['side'])
    assert digest(w.decisions)==r['prelude_sha256']
    assert w.state_sha()==r['state_sha256']
    expected=shortest_empty_path(w,w.cursor,GOAL)[0]
    assert r['expected_first_action']==expected
    if r['arm']=='memory':chosen,probs=probe(w,weights)
    elif r['arm']=='limited':chosen,probs=ACTIONS[0],{a:0.5 for a in ACTIONS}
    elif r['arm']=='oracle':chosen,probs=expected,{expected:1.0}
    else:
        assert r['arm'] in ('frozen711','frozen712')
        chosen,probs=frozen[int(r['arm'][-3:])].choose(w,GOAL,256,False)
    assert r['chosen']==chosen and r['probs']==probs
    assert r['first_optimal']==(chosen==expected)
    return replay_receipt(r)

def verify(path, probe_checkpoint, frozen_dir):
    with np.load(probe_checkpoint,allow_pickle=False) as z:
        weights=z['weights'].copy();assert int(z['updates'])==240
    frozen={i:GoalRanker.load(frozen_dir/f'{i}-conditioned.npz') for i in CHECKPOINT_SEEDS}
    count=decisions=vectors=0
    with gzip.open(path,'rt') as z:
        for line in z:
            r=json.loads(line)
            decisions+=verify_scientific_record(r,weights,frozen)
            vectors+=int(r['arm'] in ('memory','frozen711','frozen712'))
            count+=1
    return {'verified_episodes':count,'verified_host_decisions':decisions,
            'recomputed_model_probability_vectors':vectors}

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--receipts',required=True,type=Path)
    p.add_argument('--probe',required=True,type=Path)
    p.add_argument('--checkpoints',required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.receipts,a.probe,a.checkpoints),indent=2))

"""Independent replay of original legal events and neural distribution receipts.

Opt-in: --run points to an existing local evidence directory; never trains.
"""
from __future__ import annotations
import argparse, json, hashlib, statistics, time
from collections import Counter
from pathlib import Path
import numpy as np
from . import native_ceiling_continuation_lab as lab
from .world import Spec,World,replay,digest
from .policy import Ranker
from .train import sha_file


def verify(run):
    run=Path(run);report=json.loads((run/'evidence.json').read_text())
    raw=json.loads((run/'traces.json').read_text())
    assert sha_file(run/'traces.json')==report['full_original_trace_sha256']
    assert len(raw)==report['evaluated_episodes']==288
    for name,sha in report['checkpoints_sha256'].items():
        assert sha_file(run/'checkpoints'/(name+'.npz'))==sha
    summary=report['per_episode'];total=0;num_probs=0
    models={(seed,h):Ranker.restore(run/'checkpoints'/f'{seed}-{h}.npz') for seed in lab.SEEDS for h in lab.STAGES}
    for entry,item in zip(summary,raw):
        assert item['summary']==entry
        s=Spec(**item['spec']);w=World.create(s);recorded=item['events']
        assert digest(recorded)==entry['trace_sha256']
        pred_by_step={p['step']:p for p in item['probabilities']}
        for e in recorded:
            ix=e['index'];assert ix==w.steps
            assert e['legal']==w.menu(12)
            if ix in pred_by_step:
                p=pred_by_step[ix]
                model=models[entry['seed'],entry['horizon']]
                choice,distribution=model.choose(w,entry['horizon'])
                assert p['pre_sha256']==w.state_sha()
                assert choice==p['chosen'] and distribution==p['p']
                assert p['legal']==e['legal']
                assert abs(sum(distribution.values())-1)<2e-7
                num_probs+=1
            assert w.apply(e['chosen'],12)==e
            total+=1
        assert w.state_sha()==entry['end_sha256']
        assert replay(s,recorded,12).state_sha()==w.state_sha()
        prefix=[e for e in recorded if e['logical_moves']<=64]
        assert prefix==recorded[:len(prefix)] and len(prefix)==entry['prefix_64_decisions']
        assert not (w.ink & (set(w.materials)|w.protected))
    assert total==report['executed_synthetic_actions']
    assert num_probs==sum(r['steps'] for r in summary if r['arm']=='neural')
    return {'verified_episodes':len(raw),'verified_legal_actions':total,
            'verified_neural_probability_vectors':num_probs,
            'verified_checkpoints':len(report['checkpoints_sha256']),
            'all_worlds_development':all(x['spec']['split']=='development' for x in raw)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    print(json.dumps(verify(parser.parse_args().run),indent=2))

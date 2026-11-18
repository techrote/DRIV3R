"""Independent-source replay of original synthetic receipts and neural probabilities.

The verifier does not use cached summary statistics to recreate actions.
"""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from .world import World, Spec, MOVE, replay, digest
from .policy import Ranker
from .train import sha_file

def verify(root,cps):
    root=Path(root);cps=Path(cps)
    data=json.loads((root/'evidence.json').read_text())
    traces=json.loads((root/'traces.json').read_text())
    pairs=json.loads((root/'paired_objectives.json').read_text())
    assert len(traces)==data['episodes']==96
    for seed,meta in data['checkpoint_identities'].items():
        file=cps/f'{seed}-128.npz';assert sha_file(file)==meta['sha256']
    models={int(seed):Ranker.restore(cps/f'{seed}-128.npz') for seed in data['checkpoint_identities']}
    count=0;vectors=0
    for tr in traces:
        x=tr['world'];sp=Spec(**x)
        w=World.create(sp)
        r=tr['summary']; assert r['spec']==sp.ident
        pp={p['step']:p for p in tr['probabilities']}
        for receipt in tr['events']:
            step=receipt['index'];assert w.steps==step
            if step in pp:
                stored=pp[step]
                assert w.state_sha()==stored['pre_sha256']
                assert w.menu()==stored['menu']
                chosen,p=models[int(r['arm'][-4:])].choose(w,128)
                assert (chosen,p)==(stored['chosen'],stored['p'])
                vectors+=1
            assert w.apply(receipt['chosen'])==receipt
            assert w.logical_moves<=64
            assert not w.ink&(w.protected|set(w.materials))
            count+=1
        assert w.state_sha()==r['final_state_sha256']
        assert digest(tr['events'])==r['trace_sha256']
        assert len(tr['attained'])==r['goals_reached']
        for item in tr['attained']:
            assert tr['waypoints'][item['goal_index']] in [e['cursor_after'] for e in tr['events']]
    for x in pairs:
        assert x['correct_across_pair']==sum(x['goal_blind_choice']==a for a in x['optimal_actions'])
        assert x['correct_across_pair']<=1
        assert len(x['goals'])==2 and x['goals'][0]!=x['goals'][1]
    assert count==data['executed_actions']
    assert len(pairs)==data['paired_objective_states']
    assert sha_file(root/'paired_objectives.json')==data['paired_objectives_sha256']
    assert sha_file(root/'traces.json')==data['original_trace_sha256']
    return {'replayed_episodes':len(traces),'replayed_actions':count,'neural_probability_vectors':vectors,'paired_states':len(pairs)}

if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--run',required=True);pa.add_argument('--checkpoints',required=True)
    a=pa.parse_args(); print(json.dumps(verify(a.run,a.checkpoints),indent=2))

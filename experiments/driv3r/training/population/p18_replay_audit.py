"""Reload and replay P18 checkpoint decisions on untouched synthetic worlds.

Independent of candidate selection and training loop; reuses the P0 synthetic
World and P18 rollout contract. This is NOT native CyberSand verification.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from population_lab import Policy
from population_p18_tournament import HOLDOUT_CARDS, trajectory, summary


def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def audit(path, destination=None):
    p=Path(path);result=json.loads((p/'results.json').read_text())
    ledger=json.loads((p/'lineage.json').read_text())
    names={x['id']:x for x in ledger}
    count=0; trained=0; total_actions=0; legal_actions=0; verified=0
    for line in (p/'SHA256SUMS').read_text().splitlines():
        value,rel=line.split('  ',1)
        f=p/rel
        if not f.exists() or file_sha(f)!=value:
            raise AssertionError(f'P18 manifest mismatch {rel}')
        verified+=1
    for record in ledger:
        ck=p/'checkpoints'/f"{record['id']}.json"
        if file_sha(ck)!=record['checkpoint_sha256']:
            raise AssertionError('bad checkpoint sha')
        if not record['training']['updates'] or record['initial_model_sha256']==record['final_model_sha256']:
            raise AssertionError('no real weight update')
        trained+=1
    for record in result['selection']:
        ident=record['selected'];info=names[ident]
        model_data=json.loads((p/'checkpoints'/f'{ident}.json').read_text())
        model=Policy(1,hidden=model_data['hidden'],lr=model_data['lr'],temperature=model_data['temperature'])
        model.p={k:np.asarray(v,dtype=np.float64) for k,v in model_data['parameters'].items()}
        cohort=record['cohort'];niche=record['niche'];arm=record['arm'];rank=record['rank']
        splits=next(v['splits'] for v in result['split_manifest'] if v['cohort']==cohort)
        cases=[trajectory(model,niche,seed,card,result['parameters']['horizon'])[0]
               for seed in splits['holdout']['seeds'] for card in HOLDOUT_CARDS]
        actual=summary(cases)
        if actual!=record['heldout']:
            raise AssertionError(f'heldout mismatch: {ident}, {rank}')
        count+=len(cases)
        total_actions+=sum(s['steps'] for s in cases)
        legal_actions+=sum(s['steps'] for s in cases if s['legal_rate']==1 and s['illegal_paint']==0)
        if rank==0:
            stem=f'{cohort}-{arm}-{niche}'
            gallery=json.loads((p/'gallery'/f'{stem}.json').read_text())
            expected=next(s for s in cases if s['seed']==splits['holdout']['seeds'][0] and s['card']==HOLDOUT_CARDS[0])
            if gallery['metrics']!=expected or gallery['model']!=ident:
                raise AssertionError(f'original gallery mismatch: {stem}')
            if not (p/'gallery'/f'{stem}.svg').exists():
                raise AssertionError('missing original SVG')
    assert total_actions==legal_actions
    data=dict(format='population-p18-independent-checkpoint-replay',native=False,
         candidates=trained,checkpoint_replays=len(result['selection']),
         heldout_episodes=count,legal_actions=legal_actions,illegal_actions=0,
         original_galleries=len([x for x in result['selection'] if x['rank']==0]),
         manifest_entries=verified,manifest_mismatches=0,
         caveat='Independent checkpoint deserialization and replay; shared synthetic host, not native or cross-environment validation')
    if destination:Path(destination).write_text(json.dumps(data,indent=2)+'\n')
    return data

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('path');ap.add_argument('--output')
    a=ap.parse_args()
    print(json.dumps(audit(a.path,a.output),sort_keys=True))

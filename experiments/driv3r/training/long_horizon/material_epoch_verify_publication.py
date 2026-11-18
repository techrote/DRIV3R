"""Verify the committed compressed MATERIAL-EPOCH receipts using external checkpoints.

This verifier does not modify or access CyberSand, Neurounner or HELDOUT.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .material_epoch_lab import SEEDS, SOURCE, summarize, verify_trace
from .goal_conditioning_lab import GoalRanker
from .train import sha_file


def _verify_group(lines:list[str],checkpoint_dir:str):
    models={seed:GoalRanker.load(Path(checkpoint_dir)/f'{seed}-conditioned.npz') for seed in SEEDS}
    rows=[];receipts=vectors=0
    for line in lines:
        tr=json.loads(line);c,v=verify_trace(tr,models[tr['row']['seed']])
        rows.append(tr['row']);receipts+=c;vectors+=v
    return rows,receipts,vectors


def verify(ledger:Path,receipts:Path,checkpoints:Path,workers:int=4):
    if not 1<=workers<=4:raise ValueError('Bounded to 1..4 CPU workers')
    t=time.perf_counter();e=json.loads(ledger.read_text());raw=receipts.read_bytes()
    assert e['compressed_sha256']==hashlib.sha256(raw).hexdigest()
    assert e['compressed_bytes']==len(raw)
    plain=gzip.decompress(raw)
    assert e['uncompressed_sha256']==hashlib.sha256(plain).hexdigest()
    assert e['uncompressed_bytes']==len(plain)
    lines=plain.decode('utf8').splitlines()
    assert len(lines)==e['episodes']==320
    src=Path(__file__).parent
    for name in SOURCE:
        assert sha_file(src/name)==e['source_sha256'][name],f'Source changed: {name}'
    for seed in SEEDS:
        assert sha_file(checkpoints/f'{seed}-conditioned.npz')==e['checkpoint_sha256'][str(seed)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(_verify_group,[lines[i::workers] for i in range(workers)],[str(checkpoints)]*workers))
    rows=[r for rs,_,_ in results for r in rs]
    actions=sum(n for _,n,_ in results);vectors=sum(n for _,_,n in results)
    assert (len(rows),actions,vectors)==(e['episodes'],e['verified_action_receipts'],e['verified_neural_distributions'])
    assert summarize(rows)==e['summary']
    for arm in e['summary']:
        assert len([x for x in rows if x['arm']==arm])==e['summary'][arm]['episodes']
    assert all(x['logical_moves']<=64 and x['illegal_edits']==0 for x in rows)
    return {'verified':True,'episodes':len(rows),'action_receipts':actions,
            'probability_vectors':vectors,'compressed_sha256':e['compressed_sha256'],
            'elapsed_replay_wall_s':round(time.perf_counter()-t,3),
            'summary':{a:e['summary'][a]['final_goals'] for a in e['summary']}}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--ledger',type=Path,required=True)
    a.add_argument('--receipts',type=Path,required=True);a.add_argument('--checkpoints',type=Path,required=True)
    a.add_argument('--workers',type=int,default=4);x=a.parse_args()
    print(json.dumps(verify(x.ledger,x.receipts,x.checkpoints,x.workers),sort_keys=True))

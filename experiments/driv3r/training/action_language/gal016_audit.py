"""GAL-016 exact-output replay audit, using frozen checkpoint bytes."""
from pathlib import Path
import argparse
import hashlib
import json
from gal016_volatile_memory import episode, audit_trace
from route_scaffold import PaintScorer, load_frozen

def audit(root,frozen,paint):
    root=Path(root)
    saved=json.loads((root/'GAL016_episodes.json').read_text())
    base=load_frozen(frozen)
    cp=json.loads(Path(paint).read_text())
    assert cp['schema']=='gal-015-route-scaffold/1' and len(cp['weights'])==6
    scorer=PaintScorer(cp['weights'])
    counts={'matched_rows':0,'full_receipt_replays':0,'mismatches':[]}
    for r in saved:
        w,_,actual=episode(r['seed'],r['kind'],r['target'],r['capacity'],r['dynamic'],base,scorer,trace=True)
        if actual!=r:
            counts['mismatches'].append((r['seed'],r['kind'],r['target'],r['capacity'],r['dynamic']))
            continue
        counts['matched_rows']+=1
        audit_trace(w,actual)
        counts['full_receipt_replays']+=1
    counts['source_checkpoint_sha256']={
        'GAL011':hashlib.sha256(Path(frozen).read_bytes()).hexdigest(),
        'GAL015':hashlib.sha256(Path(paint).read_bytes()).hexdigest()}
    counts['pass']=counts['matched_rows']==counts['full_receipt_replays']==len(saved) and not counts['mismatches']
    (root/'GAL016_audit.json').write_text(json.dumps(counts,indent=2,sort_keys=True)+'\n')
    if not counts['pass']:raise AssertionError(counts)
    return counts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--frozen',required=True);p.add_argument('--paint',required=True)
    a=p.parse_args();print(json.dumps(audit(a.root,a.frozen,a.paint),indent=2))

"""Independent P17 checkpoint replay and evidence checks. CPU-only, external host."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from population_p17 import ARMS,NICHES,SmallPolicy,rollout,summarize,stable_seed,sha,ALL_FEATURES


def audit(folders,output=None):
    results=[];verified_checkpoints=0;heldout_missions=0;legal=0;gallery_checks=0
    for dirname in folders:
        folder=Path(dirname);overall=json.loads((folder/'results.json').read_text())
        lineage=json.loads((folder/'lineage.json').read_text()); seed=overall['seeds'][0]
        if len(overall['seeds'])!=1:raise AssertionError('expect one cohort per evidence folder')
        manifest=(folder/'SHA256SUMS').read_text().splitlines()
        for line in manifest:
            expected,relative=line.split('  ',1)
            if hashlib.sha256((folder/relative).read_bytes()).hexdigest()!=expected:
                raise AssertionError('SHA manifest mismatch: '+relative)
        for rec in lineage:
            ck=folder/'checkpoints'/f"{rec['id']}.json"
            if hashlib.sha256(ck.read_bytes()).hexdigest()!=rec['checkpoint_sha256']:
                raise AssertionError('checkpoint digest mismatch')
            data=json.loads(ck.read_text())
            if tuple(data['features'])!=ALL_FEATURES or data['updates']!=overall['updates_per_candidate']:
                raise AssertionError('checkpoint schema or optimizer updates mismatch')
            if not all(np.isfinite(np.asarray(x)).all() for x in data['parameters'].values()):
                raise AssertionError('nonfinite parameters')
            verified_checkpoints+=1
        for winner in overall['champions']:
            pth=folder/'checkpoints'/f"{winner['winner']}.json"; data=json.loads(pth.read_text())
            p=SmallPolicy(0,data['hidden'],data['lr'])
            for key in p.p:p.p[key]=np.asarray(data['parameters'][key],dtype=float)
            hold=[stable_seed('p17',seed,'heldout',i) for i in range(overall['heldout_worlds'])]
            fresh=summarize(p,winner['niche'],hold,winner['arm'])
            for key,expected in winner['heldout'].items():
                if not abs(fresh[key]-expected)<1e-5:raise AssertionError(f'heldout mismatch {winner["winner"]} {key}')
            for worldseed in hold:
                z=rollout(p,winner['niche'],worldseed,winner['arm'],full_trace=True)
                if z['legal_actions']!=z['steps']:raise AssertionError('illegal action')
                if not all(t['action'] in t['legal_actions'] for t in z['history']):raise AssertionError('illegal action trace')
                if {tuple(x) for x in z['painted']}&({tuple(x) for x in z['protected']}|{tuple(x) for x in z['obstacles']}):
                    raise AssertionError('protected paint')
                heldout_missions+=1;legal+=z['legal_actions']
                gallery=folder/'gallery'/f"{winner['winner']}-{worldseed}.json"
                if gallery.exists():
                    recorded=json.loads(gallery.read_text())
                    if recorded!=z:raise AssertionError('gallery trace mismatch')
                    gallery_checks+=1
        results.append(dict(seed=seed,manifest_entries=len(manifest),candidates=overall['candidates'],
                            successes={k:overall['aggregate'][k]['success'] for k in ARMS},
                            manifests_sha256=hashlib.sha256((folder/'SHA256SUMS').read_bytes()).hexdigest()))
    out=dict(format='population-p17-independent-replay-v1',source='local P0-derived synthetic host',
             seeds=[r['seed'] for r in results],verified_checkpoints=verified_checkpoints,
             replayed_heldout_missions=heldout_missions,replayed_host_legal_actions=legal,
             matched_original_gallery_trajectories=gallery_checks,
             illegal_actions=0,protected_obstacle_violations=0,cohorts=results)
    if output is not None:Path(output).write_text(json.dumps(out,indent=2)+'\n')
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folders',nargs='+');ap.add_argument('--output');a=ap.parse_args()
    print(json.dumps(audit(a.folders,a.output),sort_keys=True))

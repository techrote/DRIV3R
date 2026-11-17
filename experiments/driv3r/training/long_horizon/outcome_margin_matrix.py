"""OUTCOME-MARGIN-001: source-matched suffix-controller sensitivity audit.

No training or native execution. Only TRAIN and DEVELOPMENT synthetic fixtures;
structural HELDOUT stays sealed. Requires the prior outcome-fork evidence ZIP.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import tempfile
import time
import zipfile
from . import outcome_fork_lab as parent, retrace_choice_lab as forks, sequential_probe_lab as seq
from .world import specs
from .train import sha_file

ARMS=('frozen','always_move','contrastive','outcome')
SEEDS=(811,814)

def restored(archive,seed):
    models={};shas={}
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as td:
        original=json.loads(z.read('evidence/report.json'))['checkpoints'][str(seed)]
        for arm in ARMS:
            raw=z.read(f'evidence/checkpoints/{seed}-{arm}.npz')
            sha=hashlib.sha256(raw).hexdigest()
            if sha!=original[arm]['sha256']:raise ValueError('checkpoint hash mismatch')
            p=Path(td)/f'{arm}.npz';p.write_bytes(raw)
            models[arm]=seq.ProbeRanker.load(p);shas[arm]=sha
    return models,shas

def run(archive):
    start=time.perf_counter();matrix={};stats={};identities={};actions={}
    for split in ('train','development'):
        matrix[split]={};stats[split]={a:[] for a in ARMS};actions[split]={a:Counter() for a in ARMS}
    for seed in SEEDS:
        models,identities[str(seed)]=restored(archive,seed)
        for split in ('train','development'):
            for sp in specs(split,1):
                for kind in forks.KINDS:
                    key=f'{sp.family}/{kind}';case=matrix[split].setdefault(key,{})
                    q=forks.fork(sp,0,kind);menu,features=forks.inputs(q,128)
                    for arm,m in models.items():
                        d=parent.compare(sp,0,kind,m,128)
                        p=m.probs(features,len(menu));v=d['paint_delta_move_minus_backtrack']
                        case[f'{seed}/{arm}']=v
                        stats[split][arm].append((v,d['side']['stopped'],d['backtrack']['stopped'],
                            d['side']['latency_ms'],d['backtrack']['latency_ms'],menu[int(p.argmax())]))
                        for first,tag in (('side','MOVE'),('backtrack','BACKTRACK')):
                            actions[split][arm].update({f'{tag}/{a}':n for a,n in d[first]['actions'].items()})
    summary={}
    for split,cases in matrix.items():
        summary[split]={'cases':len(cases),'comparisons':sum(len(c) for c in cases.values()),
          'robust_backtrack_margin8':sum(all(v<=-8 for v in c.values()) for c in cases.values()),
          'sign_conflicts':sum(min(c.values())<0<max(c.values()) for c in cases.values()),
          'any_backtrack_win':sum(any(v<0 for v in c.values()) for c in cases.values()),
          'by_arm':{a:{'n':len(r:=stats[split][a]),'backtrack_wins':sum(v[0]<0 for v in r),
             'backtrack_margin8':sum(v[0]<=-8 for v in r),
             'mean_delta':round(statistics.mean(v[0] for v in r),3),
             'stops_move':sum(v[1] for v in r),'stops_backtrack':sum(v[2] for v in r),
             'first_retraces':sum(v[5]=='BACKTRACK' for v in r),
             'latency_ms':round(statistics.mean((v[3]+v[4])/2 for v in r),6),
             'action_counts':dict(sorted(actions[split][a].items()))} for a in ARMS}}
    return {'version':'outcome-margin-matrix/1','checkpoint_zip_sha256':sha_file(archive),
      'checkpoints_sha256':identities,'sampling':{'seeds':list(SEEDS),'turns':[0],
      'families_per_split':4,'kinds':list(forks.KINDS),'horizon':128},
      'matrix':matrix,'summary':summary,'wall_seconds':round(time.perf_counter()-start,3),
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'optimizer_updates':0,'opened_heldout':False,'native_receipts':0,'paid_compute':False}

def verify(archive, evidence):
    old=json.loads(evidence.read_text());new=run(archive)
    for split in ('train','development'):
        for arm in ARMS:
            old['summary'][split]['by_arm'][arm].pop('latency_ms',None)
            new['summary'][split]['by_arm'][arm].pop('latency_ms',None)
    for k in ('version','checkpoint_zip_sha256','checkpoints_sha256','sampling','matrix','summary'):
        if old[k]!=new[k]:raise ValueError('tampered or non-replayable '+k)
    return {'replayed_comparisons':128,'checkpoint_hashes':8}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-zip',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:print(json.dumps(verify(a.checkpoint_zip,a.out)))
    else:
        if a.out.exists():raise FileExistsError('refuse to overwrite evidence')
        a.out.write_text(json.dumps(run(a.checkpoint_zip),sort_keys=True,indent=2)+'\n')
        print(a.out)

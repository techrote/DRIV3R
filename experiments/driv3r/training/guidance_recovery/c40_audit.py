"""C40 independent matched-cluster audit; replays synthetic legal actions."""
from __future__ import annotations
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
from c39_audit import replay, bootstrap_online, bootstrap_offline, make_gallery
from c40_pairwise_rank import collect

NAMES={'C39-ALIGNED':'C40-PAIR','C39-SHUFFLED':'C40-SHUFFLED','C39-RANDOM':'C40-RANDOM'}

def audit(out):
    out=Path(out)
    cohorts={p.stem[-1]:json.loads(p.read_text()) for p in out.glob('C40-cohort-?.json')}
    assert set(cohorts)==set('ABCD')
    offline=json.loads((out/'C40-offline.json').read_text())
    states,train_receipts=collect()
    trainset={r[0] for r in train_receipts}
    labelset={r['graph'] for r in offline}
    online={r['graph_sha'] for rows in cohorts.values() for r in rows}
    assert len(trainset)==56 and len(labelset)==112 and len(online)==64
    assert not (trainset&labelset or trainset&online or labelset&online)
    allrows=[r for rows in cohorts.values() for r in rows]
    replays=[replay(r) for r in allrows if r['trajectory'] is not None]
    assert len(replays)>=12
    base={(r['seed'],r['family'],r['case'],r['switch'],r['personality'],r['directive']):r for r in allrows if r['model']=='C18-A0'}
    for r in allrows:
        assert r['host_rejections']==0
        if r['directive']=='none':
            b=base[(r['seed'],r['family'],r['case'],r['switch'],r['personality'],r['directive'])]
            assert (r['actions_sha'],r['ink_sha'])==(b['actions_sha'],b['ink_sha'])
    sim={(r['seed'],r['graph_sha'],r['switch'],r['directive'],r['personality']):r for r in allrows if r['model']=='C34-RESET'}
    changed=collections.Counter()
    for r in allrows:
        if r['model'].startswith('C39-') and r['directive']!='none':
            b=sim[(r['seed'],r['graph_sha'],r['switch'],r['directive'],r['personality'])]
            changed[NAMES[r['model']]]+= (r['actions_sha']!=b['actions_sha'])
    boot=bootstrap_online(cohorts,reps=4000,seed=400149)
    offline_boot=bootstrap_offline(offline,reps=4000,seed=400151)
    report={'schema':'driv3r-c40-audit/1','online_episodes':len(allrows),
       'guided_episodes':sum(r['directive']!='none' for r in allrows),
       'train_graphs':len(trainset),'offline_graphs':len(labelset),'online_graphs':len(online),
       'cross_split_graph_overlap':0,'executed_replays':len(replays),'replayed_actions':sum(replays),
       'host_rejections':sum(r['host_rejections'] for r in allrows),
       'raw_illegal_top_choices':sum(r['raw_illegal_top1'] for r in allrows),
       'matched_guided_trajectories_changed_vs_RESET':dict(changed),
       'bootstrap_online':{NAMES.get(k,k):v for k,v in boot.items()},
       'bootstrap_offline':{NAMES.get(k,k):v for k,v in offline_boot.items()}}
    (out/'C40-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    make_gallery(cohorts,out/'C40-gallery.html')
    page=(out/'C40-gallery.html').read_text()
    for a,b in [('C39-ALIGNED','C40-PAIR'),('C39-SHUFFLED','C40-SHUFFLED'),('C39-RANDOM','C40-RANDOM'),('C39 / Action-conditional return','C40 / Pairwise action ranking'),('C39 executed trajectories','C40 executed trajectories')]:page=page.replace(a,b)
    (out/'C40-gallery.html').write_text(page)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
    print(json.dumps(audit(a.output),indent=2))

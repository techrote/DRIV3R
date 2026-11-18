"""C66 pre-specified fresh-graph transfer audit for frozen C65, synthetic host.

No new model training, no native CyberSand inference. Requires local C65 checkpoint
custody (see owner archive), and uses the immutable C60-C64 fixture copies.
"""
from __future__ import annotations
import argparse, base64, collections, hashlib, json, math, zlib
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
import c65_paired_delayed_credit as c65

SEED=660118
DROPS=(0.,.5)
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
POLICIES=c65.POLICIES

def sha(b):return hashlib.sha256(b).hexdigest()

def run(out,fixture):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    c65.c60.SEED=SEED;c65.c60.HORIZON=95
    c65.c60.c59.c58.c55.TICKS=(9,)
    # C65's frozen cohort function reconstructs all C60-C64 exclusions.
    _,_,older,_=c65.cohorts()
    past={r['graph'] for group in older for r in group}
    c65_original=json.loads((fixture/'C65-summary.json').read_text())
    past.update(r['graph'] for group in c65_original['graph_receipts'] for r in group)
    worldgen=c65.c60.c59.c58.c55.c54.c52.worlds
    familiar=c65.c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    familiarfn=c65.c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    unseenfn=c65.c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    worlds=[];receipts=[]
    for fam,fn,seed,reps in [(('wilson','kruskal'),unseenfn,SEED+503,8),
                             (familiar,familiarfn,SEED+907,8)]:
        w,r=worldgen(fam,fn,seed,reps,exclude=past)
        assert not {x['graph'] for x in r}&past
        past.update(x['graph'] for x in r);worlds+=w;receipts+=r
    base,opt,_=c65.c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(c65.ROOT/'frozen/predecessor_C42')
    states,skips=c65.c60.c59.c58.c56.states_for_goal(base,worlds,'remote')
    groups=collections.defaultdict(list)
    for st in states:groups[st[8]].append(st)
    # Predeclared first 4 eligible states per family; no outcome selection.
    chosen=[s for family in sorted(groups) for s in groups[family][:4]]
    assert len(chosen)>=20 and len(set(s[6]['graph_sha'] for s in chosen))==len(chosen)
    cs=json.loads((c65.ROOT/'publication/C60-summary.json').read_text())
    def checkpoint(p):
        t=load_file(str(fixture/p))
        return next(iter(t.values())).detach().numpy().reshape(-1).copy()
    weights={'C60-FROZEN':np.asarray(cs['weights']['C60-ALIGNED'],np.float32),
             'C65-ALIGNED':checkpoint('C65-ALIGNED.safetensors'),
             'C65-SHIFT':checkpoint('C65-SHIFT.safetensors')}
    assert all(len(x)==c65.c60.NFEAT for x in weights.values())
    records=[];samples=[]
    for i,st in enumerate(chosen):
        if i%8==0:print('C66',i,'/',len(chosen),flush=True)
        for drop in DROPS:
            for arm in ARMS:
                if drop and arm[0]=='none':continue
                outs={p:c65.policy(base,opt,st,arm,drop,p,weights,trace=False) for p in POLICIES}
                assert all(z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha'] for z in outs.values())
                if arm[0]=='none':assert len({(z['action_sha'],z['ink_sha']) for z in outs.values()})==1
                row={'graph':st[6]['graph_sha'],'family':st[8],'arm':list(arm),'drop':drop,
                     'outcomes':{p:{k:v for k,v in z.items() if k!='path'} for p,z in outs.items()}}
                records.append(row)
                if drop==0 and arm==ARMS[-1] and len(samples)<36:
                    for p in POLICIES:
                        z=c65.policy(base,opt,st,arm,drop,p,weights,trace=True)
                        assert (z['action_sha'],z['ink_sha'],z['goal'])==(outs[p]['action_sha'],outs[p]['ink_sha'],outs[p]['goal'])
                        samples.append({'graph':row['graph'],'family':st[8],'policy':p,'goal':z['goal'],
                                        'start':z['start_pos'],'target':z['target'],'path':z['path'],
                                        'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
    aggregates={}
    for drop in DROPS:
        rows=[r for r in records if r['drop']==drop and r['arm'][0]!='none']
        aggregates[str(drop)]={'n':len(rows),'policies':{p:{
            'goals':sum(r['outcomes'][p]['goal'] for r in rows),
            'new_sites':sum(r['outcomes'][p]['new_sites'] for r in rows),
            'wins_vs_reset':sum(r['outcomes'][p]['goal']>r['outcomes']['C34-RESET']['goal'] for r in rows),
            'losses_vs_reset':sum(r['outcomes'][p]['goal']<r['outcomes']['C34-RESET']['goal'] for r in rows),
            'family_goals':{f:sum(r['outcomes'][p]['goal'] for r in rows if r['family']==f) for f in sorted(groups)}
        } for p in POLICIES}}
    summary={'schema':'driv3r-c66-fresh-c65-transfer/1','seed':SEED,'graph_count':len(chosen),
        'graph_ids':[s[6]['graph_sha'] for s in chosen],'graph_receipts':receipts,
        'skipped':skips,'excluded_historical_graph_count':len(past)-len(receipts),
        'synthetic_host_suffixes':len(records)*len(POLICIES),'matched_rows':len(records),
        'gallery_traces':len(samples),'checkpoints':{p:sha((fixture/(p+'.safetensors')).read_bytes()) for p in ('C65-ALIGNED','C65-SHIFT')},
        'aggregate':aggregates,'limitations':['frozen previously trained C65 checkpoints; no optimizer this study',
        'synthetic host only; no native CyberSand acceptance','independent fresh graph identities but not preregistered externally',
        'personality metrics are partial proxies only']}
    (out/'C66-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (out/'C66-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(json.dumps({'summary':summary,'records':records,'samples':samples},sort_keys=True,separators=(',',':')).encode(),9)).decode()+'\n')
    c65.gallery(samples,out/'C66-gallery.svg')
    print(json.dumps({'n':len(chosen),'suffixes':len(records)*len(POLICIES),'aggregate':aggregates},indent=2),flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('output');q.add_argument('--fixture',default=str(Path(__file__).resolve().parent));a=q.parse_args();run(a.output,Path(a.fixture))

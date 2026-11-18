"""C61: predeclared cross-family remote-goal recovery transfer experiment.

Synthetic CyberSand-like host ONLY. Frozen C60, same host-legal stochastic
action API and synthetic world. Fit on Wilson/familiar graphs; independently
measure transfer to completely unseen Kruskal graph family (as well as fresh
Wilson/familiar diagnostic cohorts). No hidden graph available to policy.
This file is immutable protocol+experiment source for the C61 evidence.
"""
from __future__ import annotations
import collections, hashlib, json, math, sys, zlib, base64, copy
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
# The C60 and historical research sources are frozen from the verified C60 owner
# archive, kept out of Git; exact source identities are receipted below.
sys.path.insert(0,str(ROOT/'frozen/research-source'))
import c60_remote_frontier_rl as c60

SEED=610118
HORIZON=110
TRAIN_FAMILIAR_SEED=SEED+1001
TRAIN_WILSON_SEED=SEED+2002
TEST_WILSON_KRUSKAL_SEED=SEED+3003
TEST_FAMILIAR_SEED=SEED+4004
TRAIN_REPS=3
TEST_REPS=6
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0., .5)
POLICIES=('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN','C61-WILSON','C61-SHUFFLED')
CHECKPOINTS=('C61-WILSON','C61-SHUFFLED')


def provenance():
    names=['c60_remote_frontier_rl.py','c52_shortest_step_cue.py',
           'c34_independent_maze.py','c56_memory_recovery.py','experiment.py']
    return {n:hashlib.sha256((ROOT/'frozen/research-source'/n).read_bytes()).hexdigest() for n in names}


def cohorts():
    history=json.loads((ROOT/'publication/C60-historical-graph-exclusions.json').read_text())
    c60summary=json.loads((ROOT/'publication/C60-summary.json').read_text())
    past=set(history['excluded_graphs'])
    past|={r['graph'] for phase in ('train','online') for r in c60summary['cohorts'][phase]}
    assert len(past)>=80, 'unexpected historical exclusion loss'
    w=c60.c59.c58.c55.c54.c52.worlds
    f=c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    basefn=c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    oodfn=c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    train_f,rf=w(f,basefn,TRAIN_FAMILIAR_SEED,1,exclude=past)
    used=past|{v['graph'] for v in rf}
    train_w,rw=w(('wilson',),oodfn,TRAIN_WILSON_SEED,TRAIN_REPS,exclude=used)
    used|={v['graph'] for v in rw}
    test_u,ru=w(('wilson','kruskal'),oodfn,TEST_WILSON_KRUSKAL_SEED,TEST_REPS,exclude=used)
    used|={v['graph'] for v in ru}
    test_f,rt=w(f,basefn,TEST_FAMILIAR_SEED,1,exclude=used)
    ids=[r['graph'] for r in rf+rw+ru+rt]
    assert len(ids)==len(set(ids)) and not set(ids)&past
    return (train_f+train_w,rf+rw),(test_u+test_f,ru+rt),past


def execute(base,opt,state,arm,policy,weights,drop,trace=False):
    mapped={'C60-FROZEN':'C60-ALIGNED','C61-WILSON':'C60-ALIGNED',
            'C61-SHUFFLED':'C60-SHUFFLED'}.get(policy,policy)
    return c60.execute(base,opt,state,arm,mapped,weights,drop,trace=trace)


def cluster_interval(rows,policy,cohort,drop):
    grouped=collections.defaultdict(int)
    for r in rows:
        if r['family'] not in cohort or r['drop']!=drop or r['arm'][0]=='none':continue
        grouped[r['graph']]+=r['outcomes'][policy]['goal']-r['outcomes']['C34-RESET']['goal']
    if not grouped:return None
    vals=np.array(list(grouped.values()),dtype=int)
    rng=np.random.default_rng(SEED+len(policy)+int(drop*10)+sum(map(len,cohort)))
    boot=vals[rng.integers(0,len(vals),(3000,len(vals)))].sum(1)
    return {'net_delta':int(vals.sum()),'contributing_graphs':len(vals),
        'positive_graphs':int(sum(vals>0)),'negative_graphs':int(sum(vals<0)),
        'exploratory_cluster95':[int(x) for x in np.quantile(boot,[.025,.975])]}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c60.SEED=SEED;c60.HORIZON=HORIZON
    c60.c59.c58.c55.TICKS=(9,)
    prior=ROOT/'frozen/predecessor_C42'
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(prior)
    (train,train_receipts),(test,test_receipts),old=cohorts()
    states=c60.c59.c58.c56.states_for_goal
    trst,tskip=states(base,train,'remote')
    evst,eskip=states(base,test,'remote')
    assert len(trst)>=10 and len(evst)>=20
    assert not {s[6]['graph_sha'] for s in trst}&{s[6]['graph_sha'] for s in evst}
    # Two matched reward-credit arms, no heldout feedback or tuning.
    episodes=[(s,arm) for s in trst for arm in ARMS[1:]]
    aligned,fit=c60.policy_fit(episodes,base,opt,'aligned')
    shuffled,control=c60.policy_fit(episodes,base,opt,'shuffled')
    with (ROOT/'publication/C60-summary.json').open() as f: prior_summary=json.load(f)
    frozen=np.array(prior_summary['weights']['C60-ALIGNED'],dtype=np.float32)
    weights={'C60-FROZEN':frozen,'C61-WILSON':aligned.weight.detach().numpy().reshape(-1).copy(),
             'C61-SHUFFLED':shuffled.weight.detach().numpy().reshape(-1).copy()}
    checkpoint_sha={}
    for label,model in (('C61-WILSON',aligned),('C61-SHUFFLED',shuffled)):
        p=output/(label+'.safetensors');save_file(model.state_dict(),str(p))
        checkpoint_sha[label]=hashlib.sha256(p.read_bytes()).hexdigest()
    results=[];prefix_receipts=[]
    for state in evst:
        snap,_,_,receipt,tick,goal,info,seed,fam,case,_=state
        for drop in DROPS:
          for arm in ARMS:
            outcomes={}
            for pol in POLICIES:
                z,_=execute(base,opt,state,arm,pol,weights.get(pol,np.zeros(c60.NFEAT,np.float32)),drop)
                assert z['host_rejections']==0 and z['protected_sha']==receipt['protected_sha']
                outcomes[pol]=z
            if arm[0]=='none':
                assert len({(z['action_sha'],z['ink_sha']) for z in outcomes.values()})==1
                prefix_receipts.append((info['graph_sha'],drop,receipt['prefix_sha']))
            results.append({'graph':info['graph_sha'],'family':fam,'case':case,'personality':snap.personality,
              'seed':seed,'tick':tick,'target':list(goal),'prefix_sha':receipt['prefix_sha'],
              'drop':drop,'arm':list(arm),'outcomes':outcomes})
    # Predeclared gallery selection: first *eligible* graph in each of Wilson,
    # Kruskal, familiar braid. Strong guidance, both memory states, four policies.
    chosen={}
    for fam in ('wilson','kruskal','braid'):
        s=next((s for s in evst if s[8]==fam),None)
        if s:chosen[fam]=s
    traces=[]
    for fam,st in chosen.items():
        for drop in DROPS:
            for pol in ('C34-RESET','C60-FRONTIER','C60-FROZEN','C61-WILSON'):
                z,_=execute(base,opt,st,ARMS[-1],pol,weights.get(pol,np.zeros(c60.NFEAT,np.float32)),drop,trace=True)
                expected=[r['outcomes'][pol] for r in results if r['graph']==st[6]['graph_sha'] and
                  r['drop']==drop and r['arm']==list(ARMS[-1])]
                assert len(expected)==1 and (z['action_sha'],z['ink_sha'])==(expected[0]['action_sha'],expected[0]['ink_sha'])
                traces.append({'family':fam,'graph':st[6]['graph_sha'],'policy':pol,'drop':drop,
                  'goal':z['goal'],'start_pos':z['start_pos'],'target':z['target'],
                  'steps':z['steps'],'action_sha':z['action_sha'],'ink_sha':z['ink_sha'],
                  'path':z['path']})
    aggregate={}
    for fam in ('wilson','kruskal','braid','tree','prim','prim_braid'):
      for drop in DROPS:
        selected=[r for r in results if r['family']==fam and r['drop']==drop and r['arm'][0]!='none']
        if not selected:continue
        aggregate[f'{fam}|{drop}']={'episodes':len(selected),'policies':{pol:{'goals':sum(r['outcomes'][pol]['goal'] for r in selected),
          'new_sites':sum(r['outcomes'][pol]['new_sites'] for r in selected),
          'paired_wins_vs_RESET':sum(r['outcomes'][pol]['goal']>r['outcomes']['C34-RESET']['goal'] for r in selected),
          'paired_losses_vs_RESET':sum(r['outcomes'][pol]['goal']<r['outcomes']['C34-RESET']['goal'] for r in selected)}
          for pol in POLICIES}}
    paired={}
    for cohort,families in [('kruskal',['kruskal']),('wilson',['wilson']),('familiar',['tree','braid','prim','prim_braid'])]:
      for drop in DROPS:
        paired[f'{cohort}|{drop}']={p:cluster_interval(results,p,families,drop) for p in POLICIES[2:]}
    personality={}
    for fam in ('wilson','kruskal','braid'):
      for pol in POLICIES[2:]:
        selected=[r for r in results if r['family']==fam and r['arm'][0]!='none' and r['drop']==0 and
           r['outcomes'][pol]['personality20'] and r['outcomes']['C34-RESET']['personality20']]
        means={k:float(np.mean([r['outcomes'][pol]['personality20'][k]-r['outcomes']['C34-RESET']['personality20'][k] for r in selected]))
          for k in ('turn_rate','reverse_rate','unique_ratio','direction_entropy','paint_fraction')} if selected else {}
        personality[f'{fam}|{pol}']={'n':len(selected),'deltas':means}
    summary={'schema':'driv3r-c61-wilson-to-kruskal/1','protocol_seed':SEED,'horizon':HORIZON,
      'policy_source_hashes':provenance(),'historical_excluded_graph_count':len(old),
      'frozen_C60_checkpoint_sha256':prior_summary['checkpoint_sha256']['ALIGNED'],
      'checkpoint_sha256':checkpoint_sha,'train_graphs':train_receipts,'test_graphs':test_receipts,
      'eligible_train_prefixes':len(trst),'eligible_test_prefixes':len(evst),'train_skips':tskip,'test_skips':eskip,
      'training':fit,'shuffled_training':control,'weights':{k:v.tolist() for k,v in weights.items()},
      'matched_conditions':len(results),'suffixes':len(results)*len(POLICIES),
      'unguided_parity_conditions':len(prefix_receipts),'replayed_traces':len(traces),
      'aggregate':aggregate,'clustered':paired,'personality20':personality,
      'limitations':['Synthetic host only, no native CyberSand acceptance',
         'Wilson is a seen TRAIN family; Kruskal is completely absent from training',
         'Local action head cannot plan through genuinely unknown graph connections',
         'Descriptive exploratory intervals, not preregistered statistical confirmation',
         '20-step movement proxies do not qualify holistic artistic personality',
         'Checkpoints locally archived, not immutable GitHub Actions artifact custody']}
    # Full ledger compacted losslessly. Never discard negative or no-guidance episodes.
    ledger={'schema':'driv3r-c61-full-executed-ledger/1','summary':summary,'results':results,'traces':traces}
    raw=json.dumps(ledger,sort_keys=True,separators=(',',':')).encode()
    (output/'C61-executed-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(raw,9)).decode()+'\n')
    (output/'C61-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (output/'C61-traces.json').write_text(json.dumps(traces,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'train_graphs':len(train_receipts),'test_graphs':len(test_receipts),'train_prefixes':len(trst),
      'test_prefixes':len(evst),'suffixes':len(results)*len(POLICIES),'updates':[fit['updates'],control['updates']],
      'strong_kruskal':aggregate.get('kruskal|0.0'),'replay_traces':len(traces)},indent=2))
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output');args=p.parse_args()
    run(args.output)

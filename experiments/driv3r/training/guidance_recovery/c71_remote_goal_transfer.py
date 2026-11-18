"""C71: unseen-remote-goal transfer and route-feature ablation on synthetic host.

Exact reference: frozen C70 actor, frozen C60 controller and C69 graph exclusions.
Train ONLY on new generated graph identities; evaluate separate topology-stratified
identities without selecting by any suffix-policy outcomes. The target is never
visited or on an observed host-menu route at prefix selection. No graph oracle
is consulted in inference. This is external research, not CyberSand native ABI.
"""
from __future__ import annotations
import argparse, base64, collections, copy, hashlib, json, os, sys, zlib
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file, load_file

HERE=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get('C71_ARCHIVE_ROOT',str(HERE))).resolve()
sys.path.insert(0,str(DATA_ROOT/'research'))
import c70_route_memory_actor as c70

SEED=710119
FAMILIES=('tree','braid','prim','wilson','kruskal')
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0.,1.)
TRAIN_PER_FAMILY=2
TEST_PER_FAMILY=2
WORLD_REPS=5
HORIZON=70
POLICIES=('RESET','FRONTIER','C60-FROZEN',
          'C70-FROZEN','C71-ALIGNED','C71-SHIFTED','C71-NO-ROUTE')

def sha256(x:bytes)->str:return hashlib.sha256(x).hexdigest()

def cohort(base):
    c60=c70.c60;c55=c60.c59.c58.c55;c52=c55.c54.c52
    # Historical graph leakage excluded using frozen source/publication receipts,
    # including all prior C69 and C70 train/test evaluation graphs.
    seen=c70.c69.known_ids()
    s69=json.loads((DATA_ROOT/'output/C69-summary.json').read_text())
    seen.update(r['graph'] for r in s69['selected_prefixes'])
    s70=json.loads((DATA_ROOT/'evidence/C70-summary.json').read_text())
    seen.update(r['graph'] for r in s70['train_graphs']+s70['test_graphs'])
    before=len(seen);train=[];test=[];receipts=[];elig=[]
    c55.TICKS=(9,)
    for i,fam in enumerate(FAMILIES):
        gen=c52.c50.c47.c42.world if fam in ('wilson','kruskal') else c52.c50.c47.c41.world
        worlds,wr=c52.worlds((fam,),gen,SEED+i*211111,WORLD_REPS,exclude=seen)
        seen.update(x['graph'] for x in wr)
        states,sk=c60.c59.c58.c56.states_for_goal(base,worlds,'remote')
        # Strictly unseen graph target, no target in visited memory, no recorded
        # route to target at the decision prefix. Rejected cases retained.
        eligible=[s for s in states if tuple(s[5]) not in s[2].visits
                  and s[2].route(s[0].pos,s[5]) is None]
        required=TRAIN_PER_FAMILY+TEST_PER_FAMILY
        assert len(eligible)>=required,(fam,len(eligible),sk)
        train+=eligible[:TRAIN_PER_FAMILY]
        test+=eligible[TRAIN_PER_FAMILY:required]
        receipts.extend(wr)
        elig.append({'family':fam,'generated':len(wr),'remote_eligible':len(states),
                     'strict_unseen_eligible':len(eligible),'selected_train':TRAIN_PER_FAMILY,
                     'selected_test':TEST_PER_FAMILY,'skips':sk})
    trids={s[6]['graph_sha'] for s in train};teids={s[6]['graph_sha'] for s in test}
    assert len(train)==10 and len(test)==10
    assert len(trids)==10 and len(teids)==10 and not trids&teids
    return train,test,elig,receipts,before


def run_one(base,opt,st,arm,drop,policy,weights,trace):
    # C70 source is imported unchanged. One named ablation zeroes just the
    # observed-route-match input in memory-derived legal-menu features.
    if policy=='RESET':internal='RESET';weight=np.zeros(c70.NFEAT,np.float32)
    elif policy in ('FRONTIER','STATIC-ROUTE'):internal=policy;weight=np.zeros(c70.NFEAT,np.float32)
    elif policy=='C60-FROZEN':internal='C60-FROZEN';weight=weights['C60-FROZEN']
    elif policy=='C71-SHIFTED':internal='C70-SHIFTED';weight=weights['C71-SHIFTED']
    else:internal='C70-ALIGNED';weight=weights[policy]
    orig=c70.features
    if policy=='C71-NO-ROUTE':
        def no_route(w,led,menu):
            x,available=orig(w,led,menu);x[:,0]=0.
            return x,available
        c70.features=no_route
    try:z,_=c70.execute(base,opt,st,arm,internal,weight,drop,trace=trace)
    finally:c70.features=orig
    return z


def run(output:Path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c70.SEED=SEED;c70.HORIZON=HORIZON;c70.c60.HORIZON=HORIZON
    c60=c70.c60
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(DATA_ROOT/'frozen/predecessor_C42')
    tr,te,elig,receipts,excluded=cohort(base)
    fit={};weights={};checks={}
    for mode in ('aligned','shifted'):
        name='C71-'+mode.upper();path=output/(name+'.safetensors')
        statpath=output/(name+'-train.json')
        if path.exists() and statpath.exists():
            weight_state=load_file(str(path));stats=json.loads(statpath.read_text())
            weight=weight_state['weight'].numpy().reshape(-1).copy()
        else:
            model,stats=c70.train(base,opt,[(s,{'graph':s[6]['graph_sha']}) for s in tr],mode)
            save_file(model.state_dict(),str(path));statpath.write_text(json.dumps(stats,sort_keys=True,indent=2)+'\n')
            weight=model.weight.detach().numpy().reshape(-1).copy()
        checks[name]=sha256(path.read_bytes());fit[name]=stats;weights[name]=weight
    c70_old=json.loads((DATA_ROOT/'evidence/C70-summary.json').read_text())
    weights['C70-FROZEN']=np.asarray(c70_old['weights']['C70-ALIGNED'],np.float32)
    weights['C60-FROZEN']=np.asarray(json.loads((DATA_ROOT/'publication/C60-summary.json').read_text())['weights']['C60-ALIGNED'],np.float32)
    weights['C71-NO-ROUTE']=weights['C71-ALIGNED']
    partial=output/'C71-partial.json'
    if partial.exists():
        done=json.loads(partial.read_text())
        rows=done['rows'];gallery=done['gallery'];actions=done['actions']
        ag=collections.defaultdict(collections.Counter,{k:collections.Counter(v) for k,v in done['aggregates'].items()})
    else:
        rows=[];gallery=[];ag=collections.defaultdict(collections.Counter);actions=0
    start=len(rows)//(len(ARMS)*len(DROPS))
    assert len(rows)==start*len(ARMS)*len(DROPS)
    for i,st in enumerate(te):
        if i<start:continue
        graph=st[6]['graph_sha'];fam=st[8]
        print(f'C71 unseen target {i+1:02d}/{len(te)} family={fam}',flush=True)
        for arm in ARMS:
            for drop in DROPS:
                outs={}
                for policy in POLICIES:
                    trace=(i<4 and arm[1]==1. and drop in (0.,1.) and policy in ('FRONTIER','C70-FROZEN','C71-ALIGNED'))
                    z=run_one(base,opt,st,arm,drop,policy,weights,trace)
                    assert z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha']
                    actions+=z['steps'];path=z.pop('path')
                    outs[policy]=z
                    if trace:
                        gallery.append({'family':fam,'graph':graph,'policy':policy,'drop':drop,
                            'start':z['start_pos'],'goal_pos':z['target'],'goal':z['goal'],
                            'path':path,'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
                    if arm[1]:
                        a=ag[f'{drop}/{arm[1]}/{policy}'];a['goal']+=z['goal']
                        a['new_sites']+=z['new_sites'];a['route_ticks']+=z['route_available_ticks']
                        a['route_matching_actions']+=z['route_matching_actions']
                if not arm[1]:
                    assert len({(z['action_sha'],z['ink_sha']) for z in outs.values()})==1
                rows.append({'graph':graph,'family':fam,'arm':list(arm),'drop':drop,
                             'prefix_sha':st[3]['prefix_sha'],'tick':st[4],
                             'target_unobserved':True,'initial_route_unavailable':True,
                             'outcomes':outs})
        partial.write_text(json.dumps({'rows':rows,'gallery':gallery,
            'actions':actions,'aggregates':{k:dict(v) for k,v in ag.items()}},
            sort_keys=True,separators=(',',':')))
    tr_ids=[{'graph':s[6]['graph_sha'],'family':s[8],'prefix_sha':s[3]['prefix_sha'],'target':list(s[5])} for s in tr]
    te_ids=[{'graph':s[6]['graph_sha'],'family':s[8],'prefix_sha':s[3]['prefix_sha'],'target':list(s[5])} for s in te]
    summary={'schema':'driv3r-c71-remote-goal-transfer/1','seed':SEED,
        'source_sha256':sha256(Path(__file__).read_bytes()),
        'frozen_c70_source_sha256':sha256((DATA_ROOT/'research/c70_route_memory_actor.py').read_bytes()),
        'frozen_c70_summary_sha256':sha256((DATA_ROOT/'evidence/C70-summary.json').read_bytes()),
        'excluded_prior_graphs':excluded,'train_graph_ids':tr_ids,'test_graph_ids':te_ids,
        'generated_graph_receipts':receipts,'cohort_eligibility':elig,
        'selection_rule':'Prefix tick 9; goal not in observed visits, target not a known neighbor and no observed ledger route to target; fixed generation and stable eligible order; no suffix outcomes used',
        'training':fit,'checkpoint_sha256':checks,
        'conditions':len(rows),'executed_suffixes':len(rows)*len(POLICIES),'host_actions_executed':actions,
        'aggregate':{k:dict(v) for k,v in ag.items()},'gallery_traces':len(gallery),
        'limitations':['Synthetic CyberSand-like host only; no native acceptance',
          '10 training graphs, 10 test graphs: exploratory topology-stratified cohort',
          'On-policy aligned and shifted training suffix action exposure differs despite equal updates',
          'Wrong credit rolls within episode and can retain autocorrelation',
          'Ablating observed-route action feature is not the same as erasing all movement history',
          'Personality metrics are limited movement/ink proxies, not native identity tests']}
    ledger={'schema':'driv3r-c71-executed-ledger/1','summary':summary,'rows':rows,'gallery':gallery}
    (output/'C71-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    (output/'C71-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(json.dumps(ledger,sort_keys=True,separators=(',',':')).encode(),9)).decode()+'\n')
    c70.c69.gallery(gallery,output/'C71-gallery.svg')
    partial.unlink(missing_ok=True)
    print(json.dumps({'fit':fit,'strong_intact':{p:summary['aggregate'].get(f'0.0/1.0/{p}') for p in POLICIES},
                      'strong_erased':{p:summary['aggregate'].get(f'1.0/1.0/{p}') for p in POLICIES},
                      'actions':actions,'suffixes':summary['executed_suffixes']},indent=2),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output');args=ap.parse_args();run(Path(args.output))

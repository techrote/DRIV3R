"""C42: Wilson/Kruskal out-of-generator recovery evaluation (synthetic only).

This study changes graph construction, not the game's action contract. Candidate
return labels are computed by executed host suffixes only during training and
blind assessment; no oracle/shortest-path information reaches inference.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
import c34_independent_maze as m34
import c41_topology_transfer as c41
from c39_action_value import ActionHead, action_features, play
from c40_pairwise_rank import train as train_pairwise
from c34_independent_maze import executed_suffix
from c28_temporal_option import OptionHead
from c29_probe_recovery import PREREQS
from c36_personality_budget import load_c12, a0
from experiment import DIRECTIONS,sha

FAMILIES=('wilson','kruskal')
MODELS=('C18-A0','C34-RESET','C40-PAIR','C41-MIX','C41-SHUFFLED','C42-PAIR','C42-SHUFFLED','C42-RANDOM')
COHORTS=(('A',420701),('B',629081))
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))
_CELLS={(1+2*x,1+2*y) for x in range(7) for y in range(7)}
_EDGES=sorted((p,q) for p in _CELLS for dx,dy in ((1,0),(0,1)) if (q:=(p[0]+2*dx,p[1]+2*dy)) in _CELLS)


def generate(seed:int,family:str):
    """Independent graph algorithms with deterministic per-seed RNG."""
    if family not in FAMILIES:raise ValueError(family)
    rng=np.random.default_rng(seed)
    tree_edges=[]
    if family=='wilson':
        # Wilson's algorithm: loop-erased random walks produce uniform spanning trees.
        visited={(1,1)}
        for start in sorted(_CELLS):
            if start in visited:continue
            path=[start]
            while path[-1] not in visited:
                p=path[-1]
                choices=[(p[0]+2*dx,p[1]+2*dy) for dx,dy in DIRECTIONS
                         if (p[0]+2*dx,p[1]+2*dy) in _CELLS]
                q=choices[int(rng.integers(len(choices)))]
                if q in path:
                    path=path[:path.index(q)+1]
                else:path.append(q)
            tree_edges.extend(zip(path[:-1],path[1:]))
            visited.update(path)
    else:
        # Randomized Kruskal spanning tree, different construction from DFS/Prim.
        parent={p:p for p in _CELLS}
        def find(p):
            while parent[p]!=p:
                parent[p]=parent[parent[p]];p=parent[p]
            return p
        for k in rng.permutation(len(_EDGES)):
            p,q=_EDGES[int(k)];a,b=find(p),find(q)
            if a!=b:
                parent[a]=b;tree_edges.append((p,q))
    assert len(tree_edges)==len(_CELLS)-1
    allowed=set(_CELLS)
    for p,q in tree_edges:
        allowed.add(((p[0]+q[0])//2,(p[1]+q[1])//2))
    assert len(allowed)==97
    return allowed


def maze(seed,family):
    # Predeclare morphology criterion for host-executed 24-action prefix. The
    # retry is challenge construction only, not any information used by policy.
    for retry in range(100):
        used=seed+retry*104729
        allowed=generate(used,family)
        longest=max(len(m34.shortest(allowed,(1,1),p)) for p in allowed)
        if longest>=25:return allowed
    raise AssertionError(('no_long_prefix',family,seed))


def world(seed,family,case,personality):
    m34.maze=maze
    return m34.make_world(seed,family,case,personality)


def collect(seed=420317,per_group=9):
    states=[];receipts=[]
    for family in FAMILIES:
     for case in ('detour','forward'):
      for personality in (-1,1):
       for i in range(per_group):
        w,info=world(seed+17111*i+personality*107+(case=='forward')*991+FAMILIES.index(family)*31031,
                     family,case,personality)
        for t in range(3):
            menu=w.candidates(.35);legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:break
            returns=np.asarray([executed_suffix(w,j,steps=10) for j in legal],np.float32)
            states.append((action_features(w,menu)[legal],returns))
            receipts.append((info['graph_sha'],sha([w.pos,w.history,w.goal]),sha(returns.tolist())))
            fresh=[j for j in legal if j<8 and menu[j].target not in w.history]
            j=max(fresh,key=lambda k:w.distance()-w.distance(menu[k].target)) if fresh else 8
            assert w.apply(menu[j]) and w.invalid_attempts==0
    assert len(set(x[0] for x in receipts))==8*per_group
    return states,receipts


def offline(heads,seed=421801,per_group=8):
    rows=[]
    for family in FAMILIES:
     for case in ('detour','forward'):
      for person in (-1,1):
       for i in range(per_group):
        w,info=world(seed+19079*i+person*187+(case=='forward')*509+FAMILIES.index(family)*37091,
                     family,case,person)
        for t in range(3):
            menu=w.candidates(.35);legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:break
            y=np.asarray([executed_suffix(w,j,steps=10) for j in legal],np.float32)
            x=torch.from_numpy(action_features(w,menu)[legal]);pred={}
            with torch.no_grad():
                for name,head in heads.items():
                    score=head(x).numpy();ix=int(np.argmax(score))
                    pred[name]={'action':legal[ix],'regret':float(y.max()-y[ix])}
            rows.append({'family':family,'graph_sha':info['graph_sha'],
                         'state_sha':sha([w.pos,w.history,w.goal]),'case':case,'personality':person,
                         'legal':legal,'returns':y.tolist(),'predictions':pred})
            fresh=[j for j in legal if j<8 and menu[j].target not in w.history]
            j=max(fresh,key=lambda k:w.distance()-w.distance(menu[k].target)) if fresh else 8
            assert w.apply(menu[j]) and w.invalid_attempts==0
    return rows



def evaluate(base,option,heads,cohort,seed,horizon=78):
    rows=[];graphs=[]
    for family in FAMILIES:
     for case in ('detour','forward'):
      for person in (-1,1):
       for switch in (False,True):
        s=seed+person*371+(case=='forward')*673+switch*2039+FAMILIES.index(family)*47011
        initial,info=world(s,family,case,person)
        graphs.append(info['graph_sha'])
        for directive,strength in ARMS:
         for model in MODELS:
            if model in ('C18-A0','C34-RESET'):
                mode=model;heads3=[]
            else:
                mode='C39-ALIGNED';heads3=[heads[model]]*3
            trace=(family=='wilson' and case=='detour' and switch and person==1
                   and directive=='backtrack' and strength==1.)
            outcome=play(base,option,heads3,copy.deepcopy(initial),info,s,person,mode,directive,strength,switch,horizon,trace)
            rows.append({'cohort':cohort,'model':model,'family':family,'case':case,
                'personality':person,'switch':switch,'directive':directive,'strength':strength,
                'graph':info['graph_sha'],'initial_pos':initial.pos,
                'protected_sha':sha(sorted(initial.protected)),**outcome})
    assert len(set(graphs))==16
    return rows

def report(online,blind,train_receipts,checkpoints,training):
    assert len({r['graph'] for r in online})==len(COHORTS)*len(FAMILIES)*8
    assert len({r['graph_sha'] for r in blind})==len(FAMILIES)*4*8
    graph_a={r['graph'] for r in online};graph_b={r['graph_sha'] for r in blind}
    graph_t={r[0] for r in train_receipts}
    assert not (graph_a&graph_b or graph_a&graph_t or graph_b&graph_t), 'graph leakage'
    summary={'schema':'driv3r-c42-wilson-kruskal/1','train_graphs':len(graph_t),
             'blind_graphs':len(graph_b),'online_graphs':len(graph_a),
             'blind_states':len(blind),'online_episodes':len(online),
             'train':training,'checkpoints_sha256':checkpoints,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'models':{},'paired':{},'condition_regressions':{},'host_rejections':sum(r['host_rejections'] for r in online)}
    assert summary['host_rejections']==0
    key=lambda r:(r['cohort'],r['graph'],r['directive'],r['strength'])
    base={key(r):r for r in online if r['model']=='C18-A0'}
    for model in MODELS:
        records=[r for r in online if r['model']==model]
        guided=[r for r in records if r['directive']!='none']
        idle=[r for r in records if r['directive']=='none']
        assert all((r['actions_sha'],r['ink_sha'])==(base[key(r)]['actions_sha'],base[key(r)]['ink_sha']) for r in idle)
        summary['models'][model]={'guided_goals':sum(r['goals'] for r in guided),
           'by_family':{f:sum(r['goals'] for r in guided if r['family']==f) for f in FAMILIES},
           'unguided_exact':len(idle),'guided_episodes':len(guided),
           'guided_changed_decision_states':sum(r['action_changed'] for r in guided),
           'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in records),
           'preferred_ink':sum(r['preferred'] for r in guided),'opposed_ink':sum(r['opposed'] for r in guided),
           'guided_steps':sum(r['steps'] for r in guided),
           'mean_state_style_shift':(sum(r['personality_shift_post'] for r in guided)/max(1,sum(r['action_states'] for r in guided)))}
    condition=lambda r:(r['cohort'],r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    grouped=collections.defaultdict(lambda:collections.Counter())
    for r in online:
        if r['directive']!='none':grouped[condition(r)][r['model']]+=r['goals']
    for model in MODELS:
        summary['condition_regressions'][model]=[str(k) for k,x in grouped.items() if x[model]<x['C34-RESET']]
    graphs=collections.defaultdict(lambda:collections.Counter())
    for r in online:
        if r['directive']!='none':graphs[r['graph']][r['model']]+=r['goals']
    rng=np.random.default_rng(422009)
    for model in ('C40-PAIR','C41-MIX','C41-SHUFFLED','C42-PAIR','C42-SHUFFLED','C42-RANDOM'):
        for against in ('C34-RESET','C41-SHUFFLED','C42-SHUFFLED'):
            if model==against:continue
            delta=np.asarray([v[model]-v[against] for v in graphs.values()])
            ids=rng.integers(len(delta),size=(4000,len(delta)))
            low,high=np.quantile(delta[ids].sum(axis=1),[.025,.975])
            summary['paired'][model+'__vs__'+against]={'events':int(delta.sum()),
              'graph_cluster_bootstrap_95_descriptive':[float(low),float(high)],
              'positive_graphs':int((delta>0).sum()),'negative_graphs':int((delta<0).sum())}
    for name in ('C40-PAIR','C41-MIX','C41-SHUFFLED','C42-PAIR','C42-SHUFFLED','C42-RANDOM'):
        pred=[r['predictions'][name]['regret'] for r in blind]
        summary['models'][name]['blind_mean_regret']=round(float(np.mean(pred)),6)
        summary['models'][name]['blind_oracle_top1']=sum(v<=1e-8 for v in pred)
    return summary


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    states,receipts=collect()
    models,stats=train_pairwise(states,seed=420119,epochs=8)
    replicas,repstats=train_pairwise(states,seed=420119,epochs=8)
    assert stats==repstats
    for m,r in zip(models,replicas):
        assert all(torch.equal(m.state_dict()[k],r.state_dict()[k]) for k in m.state_dict())
    heads={}
    checks={}
    for name,head in zip(('C42-PAIR','C42-SHUFFLED','C42-RANDOM'),models):
        dest=out/(name+'.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(dest))
        checks[name]=hashlib.sha256(dest.read_bytes()).hexdigest();heads[name]=head
    for name,relative in [('C40-PAIR','out_c40/C40-PAIR.safetensors'),
                          ('C41-MIX','c41_evidence/C41-MIX.safetensors'),
                          ('C41-SHUFFLED','c41_evidence/C41-SHUFFLED.safetensors')]:
        h=ActionHead();h.load_state_dict(load_file(str(root/relative)));h.eval();heads[name]=h
    blind=offline(heads)
    base=a0(load_c12([root/'root/prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]))
    option=OptionHead();option.load_state_dict(load_file(str(root/'root/results/C28-TEMPORAL.safetensors')));option.eval()
    online=[]
    for cohort,seed in COHORTS:
        print('C42 cohort',cohort,flush=True,file=sys.stderr)
        path=out/f'C42-cohort-{cohort}.json'
        if path.exists():
            rows=json.loads(path.read_text())
        else:
            rows=evaluate(base,option,heads,cohort,seed,horizon=78)
            path.write_text(json.dumps(rows,sort_keys=True)+'\n')
        online.extend(rows)
    summary=report(online,blind,receipts,checks,{'states':len(states),'examples':sum(len(y) for _,y in states),
                          'updates_per_trained_head':stats['C40-PAIR']['optimizer_updates'],
                          'reproduced_byte_identical':True,'losses':stats})
    (out/'C42-blind.json').write_text(json.dumps(blind,sort_keys=True)+'\n')
    (out/'C42-train-receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    (out/'C42-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'models':summary['models'],'pairs':summary['paired'],
           'regressions':{k:len(v) for k,v in summary['condition_regressions'].items()}},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.root,args.output)

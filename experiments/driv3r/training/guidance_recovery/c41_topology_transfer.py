"""C41: paired topology-transfer evaluation of real CPU-trained action rankers.

Synthetic DRIV3R fixture only. No CyberSand native or ABI claim. No actions outside
host-legal menus. Blind labels are executed futures, never available to inference.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
import c34_independent_maze as m34
from experiment import DIRECTIONS, sha
from c34_independent_maze import executed_suffix
from c39_action_value import ActionHead, action_features, play
from c40_pairwise_rank import train as train_pairwise
from c28_temporal_option import OptionHead
from c29_probe_recovery import PREREQS
from c36_personality_budget import load_c12, a0

FAMILIES=('tree','braid','prim','prim_braid')
MODELS=('C18-A0','C34-RESET','C40-PAIR','C41-MIX','C41-SHUFFLED','C41-RANDOM')
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))
COHORTS=(('A',410701),('B',571909),('C',773111))
_ORIGINAL_MAZE=m34.maze

def independent_prim(seed, loops=False):
    """Prim frontier selection, not DFS: independent connectivity generator."""
    rng=np.random.default_rng(seed)
    cells={(1+2*x,1+2*y) for x in range(7) for y in range(7)}
    visited={(1,1)};allowed={(1,1)};frontier=[]
    def advance(p):
        for dx,dy in DIRECTIONS:
            q=(p[0]+2*dx,p[1]+2*dy)
            if q in cells and q not in visited:frontier.append((p,q))
    advance((1,1))
    while frontier:
        idx=int(rng.integers(len(frontier)));p,q=frontier.pop(idx)
        if q in visited:continue
        visited.add(q);allowed.add(q)
        allowed.add(((p[0]+q[0])//2,(p[1]+q[1])//2));advance(q)
    assert len(visited)==49 and len(allowed)==97
    if loops:
        wall=[]
        for x,y in sorted(cells):
            for dx,dy in ((1,0),(0,1)):
                q=(x+2*dx,y+2*dy);mid=(x+dx,y+dy)
                if q in cells and mid not in allowed:wall.append(mid)
        rng.shuffle(wall);allowed.update(wall[:10])
    return allowed

def maze(seed,family):
    if family in ('prim','prim_braid'):return independent_prim(seed,family=='prim_braid')
    return _ORIGINAL_MAZE(seed,family)

def world(seed,family,case,personality):
    m34.maze=maze
    return m34.make_world(seed,family,case,personality)

def collect(seed=411107,per_group=10):
    states=[];receipts=[]
    for family in ('tree','braid'):
      for case in ('detour','forward'):
       for person in (-1,1):
        for i in range(per_group):
            w,info=world(seed+17011*i+person*109+(case=='forward')*991+(family=='braid')*7901,family,case,person)
            for t in range(3):
                menu=w.candidates(.35);legal=[j for j in range(9) if menu[j].legal]
                if len(legal)<2:break
                returns=np.asarray([executed_suffix(w,j,steps=10) for j in legal],np.float32)
                states.append((action_features(w,menu)[legal],returns))
                receipts.append((info['graph_sha'],sha([w.pos,w.history,w.goal]),sha(returns.tolist())))
                forward=[j for j in legal if j<8 and menu[j].target not in w.history]
                j=max(forward,key=lambda x:w.distance()-w.distance(menu[x].target)) if forward else 8
                assert menu[j].legal and w.apply(menu[j]) and w.invalid_attempts==0
    assert len(set(x[0] for x in receipts))==8*per_group
    return states,receipts

def offline(heads,seed=415011,per_group=9):
    rows=[]
    for family in FAMILIES:
     for case in ('detour','forward'):
      for person in (-1,1):
       for i in range(per_group):
        w,info=world(seed+i*19079+person*187+(case=='forward')*509+FAMILIES.index(family)*37091,family,case,person)
        for t in range(3):
            menu=w.candidates(.35);legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:break
            values=np.asarray([executed_suffix(w,j,steps=10) for j in legal])
            features=torch.from_numpy(action_features(w,menu)[legal])
            oracle=float(values.max())
            record={'graph':info['graph_sha'],'family':family,'state_sha':sha([w.pos,w.history,w.goal]),'case':case,'personality':person,'legal':legal,'returns':values.tolist(),'oracle':oracle,'predictions':{}}
            with torch.no_grad():
                for label,head in heads.items():
                    scores=head(features).numpy()
                    j=int(np.argmax(scores))
                    record['predictions'][label]={'index':legal[j],'return':float(values[j]),'regret':float(oracle-values[j])}
            rows.append(record)
            forward=[j for j in legal if j<8 and menu[j].target not in w.history]
            j=max(forward,key=lambda x:w.distance()-w.distance(menu[x].target)) if forward else 8
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
                mode=model;three=[]
            else:
                mode='C39-ALIGNED';three=[heads[model]]*3
            trace=(family in ('prim','prim_braid') and case=='detour' and switch and person==1 and directive=='backtrack' and strength==1.)
            r=play(base,option,three,copy.deepcopy(initial),info,s,person,mode,directive,strength,switch,horizon,trace)
            rows.append({'cohort':cohort,'model':model,'family':family,'case':case,'personality':person,
                  'switch':switch,'directive':directive,'strength':strength,'graph':info['graph_sha'],
                  'initial_pos':initial.pos,'protected_sha':sha(sorted(initial.protected)),**r})
    assert len(set(graphs))==32
    return rows

def describe(online,blind):
    summary={'schema':'driv3r-c41-independent-prim-transfer/1','online_episodes':len(online),
             'graphs_online':len(set(r['graph'] for r in online)),
             'graphs_offline':len(set(r['graph'] for r in blind)),
             'host_rejections':sum(r['host_rejections'] for r in online),
             'models':{},'offline':{},'condition_regressions_vs_RESET':{}}
    assert not (set(r['graph'] for r in online)&set(r['graph'] for r in blind))
    for label in MODELS:
        records=[r for r in online if r['model']==label]
        guided=[r for r in records if r['directive']!='none']
        byfam={family:sum(r['goals'] for r in guided if r['family']==family) for family in FAMILIES}
        summary['models'][label]={'guided_events':sum(r['goals'] for r in guided),
              'guided_episodes':len(guided),'by_family':byfam,
              'preferred_ink':sum(r['preferred'] for r in guided),
              'opposed_ink':sum(r['opposed'] for r in guided),
              'guided_actions':sum(r['steps'] for r in guided),
              'guidance_action_states':sum(r['action_states'] for r in guided),
              'guidance_changed_states':sum(r['action_changed'] for r in guided),
              'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in records),
              'host_rejections':sum(r['host_rejections'] for r in records)}
    key=lambda r:(r['cohort'],r['graph'],r['directive'],r['strength'])
    baseline={key(r):r for r in online if r['model']=='C18-A0'}
    for label in MODELS:
        no=[r for r in online if r['model']==label and r['directive']=='none']
        summary['models'][label]['unguided_exact']=sum((r['actions_sha'],r['ink_sha'])==(baseline[key(r)]['actions_sha'],baseline[key(r)]['ink_sha']) for r in no)
        summary['models'][label]['unguided_n']=len(no)
        assert summary['models'][label]['unguided_exact']==len(no)
    ckey=lambda r:(r['cohort'],r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    group=collections.defaultdict(lambda:collections.Counter())
    for r in online:
        if r['directive']!='none':group[ckey(r)][r['model']]+=r['goals']
    for label in MODELS:
        summary['condition_regressions_vs_RESET'][label]=[str(k) for k,v in group.items() if v[label]<v['C34-RESET']]
    for label in blind[0]['predictions']:
        sub=[r for r in blind]
        summary['offline'][label]={'regret':round(float(np.mean([r['predictions'][label]['regret'] for r in sub])),6),
            'oracle_top1':sum(r['predictions'][label]['regret']<1e-8 for r in sub),
            'by_family':{f:round(float(np.mean([r['predictions'][label]['regret'] for r in sub if r['family']==f])),6) for f in FAMILIES}}
    # Paired graph bootstrap, with episodes aggregated within each independent graph.
    graph=collections.defaultdict(lambda:collections.Counter())
    for r in online:
        if r['directive']!='none':graph[r['graph']][r['model']]+=r['goals']
    rng=np.random.default_rng(411888)
    clusters=list(graph)
    for candidate in ('C41-MIX','C40-PAIR'):
      summary['models'][candidate]['paired_delta']={}
      for against in ('C34-RESET','C40-PAIR','C41-SHUFFLED','C41-RANDOM'):
        if candidate==against:continue
        differences=np.asarray([graph[g][candidate]-graph[g][against] for g in clusters],np.float64)
        draws=rng.integers(len(differences),size=(4000,len(differences)))
        low,high=np.quantile(differences[draws].sum(axis=1),[.025,.975])
        summary['models'][candidate]['paired_delta'][against]={'events':int(differences.sum()),'graph_bootstrap_95_descriptive':[float(low),float(high)],
            'positive_graphs':int((differences>0).sum()),'negative_graphs':int((differences<0).sum())}
    assert summary['host_rejections']==0
    return summary

def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    states,receipts=collect()
    headlist,train_stats=train_pairwise(states,seed=411117)
    replicate,repeat_stats=train_pairwise(states,seed=411117)
    assert train_stats==repeat_stats
    assert all(all(torch.equal(h.state_dict()[k],r.state_dict()[k]) for k in h.state_dict()) for h,r in zip(headlist,replicate))
    heads={}
    for model,head in zip(('C41-MIX','C41-SHUFFLED','C41-RANDOM'),headlist):
        save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(out/(model+'.safetensors')))
        heads[model]=head
    for old in ('C40-PAIR','C40-SHUFFLED','C40-RANDOM'):
        h=ActionHead();h.load_state_dict(load_file(str(root/'out_c40'/(old+'.safetensors'))));h.eval();heads[old]=h
    blind=offline({m:heads[m] for m in ('C40-PAIR','C41-MIX','C41-SHUFFLED','C41-RANDOM')})
    base=a0(load_c12([root/'root/prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]))
    option=OptionHead();option.load_state_dict(load_file(str(root/'root/results/C28-TEMPORAL.safetensors')));option.eval()
    online=[]
    for label,seed in COHORTS:
        batch=evaluate(base,option,heads,label,seed)
        (out/f'C41-cohort-{label}.json').write_text(json.dumps(batch,sort_keys=True)+'\n')
        online.extend(batch)
    summary=describe(online,blind)
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['train']={'states':len(states),'graphs':len(set(x[0] for x in receipts)),
         'candidate_examples':sum(len(y) for _,y in states),'optimizer':train_stats,
         'replicated_bit_exact':True,'receipts_sha256':sha(receipts)}
    summary['checkpoints']={name:hashlib.sha256((out/(name+'.safetensors')).read_bytes()).hexdigest() for name in ('C41-MIX','C41-SHUFFLED','C41-RANDOM')}
    summary['c40_checkpoint_sha256']=hashlib.sha256((root/'out_c40/C40-PAIR.safetensors').read_bytes()).hexdigest()
    (out/'C41-offline.json').write_text(json.dumps(blind,sort_keys=True)+'\n')
    (out/'C41-training-receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    (out/'C41-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'training':summary['train'],'models':summary['models'],'offline':summary['offline'],
        'condition_regressions_vs_RESET':{k:len(v) for k,v in summary['condition_regressions_vs_RESET'].items()}},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');args=p.parse_args()
    run(args.root,args.output)

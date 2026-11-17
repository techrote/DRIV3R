"""C46: train a cue-conditional selector on matched *complete* goal outcomes.

Experimental synthetic CyberSand-like host, not native CyberSand validation.
No graph, true goal route or future counterfactual is an inference feature.
Each experimental arm executes its own full host-legal policy trajectory.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import sha
from c34_independent_maze import obs
from c39_action_value import play
import c41_topology_transfer as c41
import c42_wilson_transfer as c42
import c44_onpolicy_pairwise as c44

SEED=461107
BASES=('C18-A0','C34-RESET','C40-PAIR')
POLICIES=BASES+('C46-ALIGNED','C46-SHUFFLED','C46-RANDOM')
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))
COHORTS=(('A',466011),('B',567013),('C',668017))

class Arbiter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,18),nn.Tanh(),nn.Linear(18,3))
    def forward(self,x):return self.net(x)


def inputs(w, directive, strength, switch):
    """Only current host observation, legal action menu and issued cue metadata."""
    menu=w.candidates(strength if directive!='none' else 0.)
    o=obs(w,menu)
    context=np.asarray([float(w.personality),float(switch),float(strength),
          float(directive=='backtrack'),float(directive=='continue')],np.float32)
    return np.concatenate((o,context)).astype(np.float32)


def execute(base,option,prior,w,info,seed,directive,strength,switch,policy,trace=False):
    mode='C39-ALIGNED' if policy=='C40-PAIR' else policy
    heads=[prior]*3 if policy=='C40-PAIR' else []
    return play(base,option,heads,copy.deepcopy(w),info,seed,w.personality,mode,
                directive,strength,switch,78,trace)


def utility(out):
    # Success dominates all step improvements. Failures tie rather than rewarding early stop.
    return out['goals']*100. + ((78-out['steps'])/78. if out['goals'] else 0.)


def worlds(families,which,seed,rep):
    out=[]
    for fi,fam in enumerate(families):
      for case in ('detour','forward'):
       for personality in (-1,1):
        for switch in (False,True):
         for i in range(rep):
          s=seed+fi*110003+(case=='forward')*19013+(personality+1)*3737+switch*2971+i*18041
          w,info=which(s,fam,case,personality)
          out.append((w,info,s,fam,case,switch))
    assert len({r[1]['graph_sha'] for r in out})==len(out)
    return out


def counterfactuals(base,option,prior,world_rows,arms):
    records=[];graphs=[]
    for w,info,seed,family,case,switch in world_rows:
      graphs.append(info['graph_sha'])
      for directive,strength in arms:
        results={name:execute(base,option,prior,w,info,seed,directive,strength,switch,name)
                 for name in BASES}
        assert all(v['host_rejections']==0 for v in results.values())
        scores=[utility(results[name]) for name in BASES]
        # Deterministic tie preference for unchanged baseline; explicit scientific caveat.
        label=int(np.argmax(scores))
        records.append({'graph':info['graph_sha'],'family':family,'case':case,
          'personality':w.personality,'switch':switch,'directive':directive,
          'strength':strength,'observation':inputs(w,directive,strength,switch).tolist(),
          'label':label,'score':scores,'goals':[results[k]['goals'] for k in BASES],
          'actions_sha':[results[k]['actions_sha'] for k in BASES]})
    return records,set(graphs)


def train(records,seed=SEED,epochs=18):
    x=torch.tensor(np.asarray([r['observation'] for r in records]),dtype=torch.float32)
    y=np.asarray([r['label'] for r in records],np.int64)
    rng=np.random.default_rng(seed+171);groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[r['graph']].append(i)
    permuted=y.copy()
    graph_keys=sorted(groups)
    reassign=rng.permutation(len(graph_keys))
    # Shuffle entire per-graph label vectors, preserving cue pairing/class frequency.
    for dst,src in enumerate(reassign):
        a,b=groups[graph_keys[dst]],groups[graph_keys[int(src)]]
        assert len(a)==len(b)
        permuted[a]=y[b]
    assert len(set(y.tolist()))>=2 and not np.array_equal(y,permuted)
    torch.set_num_threads(1);torch.manual_seed(seed)
    init=Arbiter();heads=[copy.deepcopy(init),copy.deepcopy(init),init]
    orders=[rng.permutation(len(records)) for _ in range(epochs)]
    stats={}
    for name,net,labels in zip(('aligned','shuffled'),heads,(y,permuted)):
        optimizer=torch.optim.AdamW(net.parameters(),lr=.006,weight_decay=.01)
        truth=torch.tensor(labels,dtype=torch.long);curves=[];updates=0
        for order in orders:
            loss_sum=0.
            for offset in range(0,len(order),16):
                j=order[offset:offset+16]
                loss=nn.functional.cross_entropy(net(x[j]),truth[j]);optimizer.zero_grad(set_to_none=True)
                loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                loss_sum+=float(loss.detach());updates+=1
            curves.append(round(loss_sum,6))
        net.eval();stats[name]={'optimizer_updates':updates,'epoch_losses':curves,
               'training_accuracy':float((net(x).argmax(-1)==torch.tensor(y)).float().mean())}
    stats.update({'records':len(records),'graph_clusters':len(graph_keys),
       'class_counts':np.bincount(y,minlength=3).tolist(),
       'targets_sha256':sha([(r['graph'],r['directive'],r['strength'],r['label'],r['goals']) for r in records]),
       'shuffled_sha256':hashlib.sha256(permuted.tobytes()).hexdigest()})
    return heads,stats


def evaluate(base,option,prior,heads,world_rows):
    rows=[];traces=[];decisions=collections.Counter()
    for w,info,seed,family,case,switch in world_rows:
      for directive,strength in ARMS:
        x=torch.from_numpy(inputs(w,directive,strength,switch))
        cached={base_name:execute(base,option,prior,w,info,seed,directive,strength,switch,base_name) for base_name in BASES}
        for name in POLICIES:
          if name in BASES:chosen=name
          elif directive=='none':chosen='C18-A0'
          else:
            with torch.no_grad():chosen=BASES[int(heads[POLICIES.index(name)-3](x).argmax())]
          decisions[(name,chosen)]+=1
          record=cached[chosen]
          rows.append({'graph':info['graph_sha'],'cohort':next((a for a,b in COHORTS if b==seed),''),
            'family':family,'case':case,'personality':w.personality,'switch':switch,
            'directive':directive,'strength':strength,'model':name,'selected':chosen,
            'protected_sha':sha(sorted(w.protected)),**record})
          if directive=='backtrack' and strength==1. and family=='wilson' and case=='detour' and w.personality==1 and switch:
            if len([t for t in traces if t['model']==name])<2:
                replay=execute(base,option,prior,w,info,seed,directive,strength,switch,chosen,True)
                assert (replay['actions_sha'],replay['ink_sha'])==(record['actions_sha'],record['ink_sha'])
                traces.append({'model':name,'selected':chosen,'graph':info['graph_sha'],
                   'initial':w.pos,'protected':sorted(w.protected),'trajectory':replay['trajectory'],
                   'goals':replay['goals'],'actions_sha':replay['actions_sha'],'ink_sha':replay['ink_sha']})
    return rows,traces,{a:{b:int(decisions[(a,b)]) for b in BASES} for a in POLICIES}


def summarize(train_rows,blind_rows,online_rows,graphs,stats,checks,selections,heads):
    train_graphs,blind_graphs,online_graphs=graphs
    assert not(train_graphs&blind_graphs or train_graphs&online_graphs or blind_graphs&online_graphs)
    assert len(online_rows)%len(POLICIES)==0
    b={ (r['graph'],r['directive'],r['strength']):r for r in online_rows if r['model']=='C18-A0'}
    assert all(r['host_rejections']==0 for r in online_rows)
    for r in online_rows:
        if r['directive']=='none':
            z=b[r['graph'],r['directive'],r['strength']]
            assert (r['actions_sha'],r['ink_sha'])==(z['actions_sha'],z['ink_sha'])
    grouped=collections.defaultdict(lambda:collections.Counter())
    by_graph=collections.defaultdict(collections.Counter)
    for r in online_rows:
      if r['directive']=='none':continue
      grouped[(r['family'],r['case'],r['switch'],r['directive'],r['strength'])][r['model']]+=r['goals']
      by_graph[r['graph']][r['model']]+=r['goals']
    out={'schema':'driv3r-c46-goal-paired-arbitration/1','graph_counts':{'train':len(train_graphs),
       'blind':len(blind_graphs),'online':len(online_graphs)},'training':stats,'checkpoints_sha256':checks,
       'online_episode_count':len(online_rows),'guided_episodes_per_policy':sum(r['directive']!='none' for r in online_rows)//len(POLICIES),
       'selected_controllers':selections,'models':{},'paired':{},'blind':{}}
    for name in POLICIES:
      rows=[r for r in online_rows if r['model']==name and r['directive']!='none']
      out['models'][name]={'goals':sum(r['goals'] for r in rows),
        'condition_regressions_vs_RESET':sum(v[name]<v['C34-RESET'] for v in grouped.values()),
        'by_family':{f:sum(r['goals'] for r in rows if r['family']==f) for f in c42.FAMILIES},
        'unguided_hash_equal':sum(r['directive']=='none' for r in online_rows if r['model']==name),
        'host_rejections':sum(r['host_rejections'] for r in rows),
        'preferred':sum(r['preferred'] for r in rows),'opposed':sum(r['opposed'] for r in rows)}
    for name in ('C46-ALIGNED','C46-SHUFFLED','C46-RANDOM'):
      for control in ('C18-A0','C34-RESET','C40-PAIR','C46-SHUFFLED'):
        if name==control:continue
        d=np.asarray([v[name]-v[control] for v in by_graph.values()]);rng=np.random.default_rng(46011+POLICIES.index(name)*29+POLICIES.index(control))
        draws=rng.integers(len(d),size=(2000,len(d)))
        out['paired'][name+'__'+control]={'delta':int(d.sum()),'positive_graphs':int((d>0).sum()),
          'negative_graphs':int((d<0).sum()),'95pct_graph_cluster_bootstrap':np.quantile(d[draws].sum(1),[.025,.975]).tolist()}
    for name,net in zip(POLICIES[3:],(heads[0],heads[1],heads[2])):
      x=torch.tensor(np.asarray([r['observation'] for r in blind_rows]),dtype=torch.float32)
      y=np.asarray([r['label'] for r in blind_rows])
      with torch.no_grad():pred=net(x).argmax(-1).numpy()
      out['blind'][name]={'states':len(y),'oracle_selection_accuracy':float((pred==y).mean()),
          'mean_oracle_goal_regret':float(np.mean([max(r['goals'])-r['goals'][j] for r,j in zip(blind_rows,pred)]))}
    return out


def run(root,out):
    root=Path(root);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1)
    base,option,prior=c44.prerequisites(root)
    train_w=worlds(c41.FAMILIES,c41.world,SEED,2)
    training,train_graphs=counterfactuals(base,option,prior,train_w,ARMS[1:3])
    heads,stats=train(training);repeat,stats2=train(training)
    assert stats==stats2
    assert all(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()) for a,b in zip(heads,repeat))
    checks={}
    for name,net in zip(POLICIES[3:5],heads[:2]):
       path=out/(name+'.safetensors');save_file({k:v.contiguous() for k,v in net.state_dict().items()},str(path));checks[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    blind_w=worlds(c42.FAMILIES,c42.world,461901,1)
    blind_rows,blind_graphs=counterfactuals(base,option,prior,blind_w,ARMS[1:3])
    online_rows=[];traces=[];selections=collections.defaultdict(collections.Counter);online_graphs=set()
    for cname,cseed in COHORTS:
       rows=worlds(c42.FAMILIES,c42.world,cseed,1)
       g=set(r[1]['graph_sha'] for r in rows);assert g.isdisjoint(online_graphs);online_graphs|=g
       rr,tt,ss=evaluate(base,option,prior,heads,rows)
       for r in rr:r['cohort']=cname
       online_rows+=rr;traces+=tt
       for a,v in ss.items():selections[a].update(v)
       print('C46 online cohort',cname,'rows',len(rr),flush=True)
    assert len(online_graphs)==len(COHORTS)*len(rows)
    summary=summarize(training,blind_rows,online_rows,(train_graphs,blind_graphs,online_graphs),stats,checks,dict(selections),heads)
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['limitations']=['synthetic host fixture; native CyberSand ABI unvalidated',
       'single episode-start controller selection, no mid-episode adaptation',
       'counterfactual full-goal labels are hindsight supervision; training on both outcomes is not a deployable oracle',
       'tie labels choose A0; effect can dominate a small dataset','correlated cue arms; cluster by graph',
       'whole-trajectory personality accepted only by descriptive counts, not behavioral equivalence',
       'no immutable GitHub Actions checkpoint artifact custody']
    for n,obj in [('C46-summary.json',summary),('C46-training.json',training),('C46-blind.json',blind_rows),
       ('C46-online.json',online_rows),('C46-traces.json',traces)]:
       (out/n).write_text(json.dumps(obj,indent=2 if n=='C46-summary.json' else None,sort_keys=True)+'\n')
    print(json.dumps({'models':summary['models'],'graph_counts':summary['graph_counts'],
       'blind':summary['blind'],'selection':summary['selected_controllers']},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');a=p.parse_args();run(a.root,a.out)
"""C45 selective recovery intervention: train a *host-visible* gate over C40.

Research-only synthetic host; no native CyberSand validation. Counterfactual
training labels execute legal candidate futures and NEVER enter policy features.
All topology families, seeds and controls are fixed before inspecting results.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from signed_feedback import HostAdvice
from c28_temporal_option import probs
from c34_independent_maze import ReArmedWatch, executed_suffix
from c36_personality_budget import style_project
from c39_action_value import ActionHead, action_features, play
from experiment import sha
import c41_topology_transfer as c41
import c42_wilson_transfer as c42
import c44_onpolicy_pairwise as c44

SEED=451701
COHORTS=(('A',454001),('B',564013),('C',674027))
MODELS=('C18-A0','C34-RESET','C40-PAIR','C45-GATE','C45-SHUFFLE','C45-MARGIN')
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))


def shared_features(x,prior):
    with torch.no_grad(): p=prior(x).detach()
    top=torch.sort(p)[0]
    margin=top[-1]-(top[-2] if len(top)>1 else top[-1]);spread=p.max()-p.min()
    return torch.cat([x.mean(0),torch.tensor([float(p.std(unbiased=False)),float(margin),float(spread),len(x)/9.])])

class GateNet(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(38,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).view(())

class SelectiveHead(nn.Module):
    def __init__(self,prior,gate=None,rule=False):
        super().__init__();self.prior=copy.deepcopy(prior).eval();self.gate=gate;self.rule=rule
        self.eligible=0;self.enabled=0
    def forward(self,x):
        scores=self.prior(x)
        if self.gate is None:
            conf=(float(shared_features(x,self.prior)[35])>.18) if self.rule else True
        else:
            with torch.no_grad():conf=bool(self.gate(shared_features(x,self.prior)).sigmoid().item()>=.5)
        self.eligible+=1;self.enabled+=int(conf)
        return scores if conf else torch.zeros_like(scores)


def local_baseline(snap,phase,base,option):
    w=copy.deepcopy(snap);advice=HostAdvice(w);advice.issue_backtrack(w,strength=.35,ttl=9)
    policy=ReArmedWatch(option,None)
    # Approximate RESET's phase at a saved snapshot, not its full recurrent state.
    policy.phase=phase
    menu,logits,_,_=policy.modify(base,w,advice,1.)
    return menu,np.asarray(logits,np.float64)


def collect(base,option,prior,seed=SEED,per_group=4,cap=3):
    records=[];receipts=[];graphs=[]
    for fam in c41.FAMILIES:
      for case in ('detour','forward'):
       for person in (-1,1):
        for i in range(per_group):
          s=seed+i*19717+person*197+(case=='forward')*853+c41.FAMILIES.index(fam)*49109
          w,info=c41.world(s,fam,case,person);graphs.append(info['graph_sha'])
          for snap,phase,t in c44.guided_states(w,base,option,s,cap=cap):
            menu,logits=local_baseline(snap,phase,base,option)
            legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:continue
            features=torch.from_numpy(action_features(snap,menu)[legal])
            returns=np.asarray([executed_suffix(snap,j,steps=20) for j in legal],np.float64)
            with torch.no_grad(): prediction=prior(features).numpy()
            proposed=logits.copy();centered=prediction-np.mean(prediction)
            for j,v in zip(legal,centered): proposed[j]+=float(.8*np.tanh(v))
            proposed,_,_,_=style_project(menu,logits,proposed,snap.personality,max_shift=.045)
            advantage=float(np.dot(probs(proposed)[legal]-probs(logits)[legal],returns))
            label=int(advantage>1e-5)
            records.append((shared_features(features,prior).numpy(),label,advantage,info['graph_sha'],fam))
            receipts.append((info['graph_sha'],sha([snap.pos,snap.goal,snap.history]),phase,t,sha(returns.tolist()),round(advantage,9)))
    assert len(graphs)==len(set(graphs)), 'duplicate train graph'
    assert len(records)>30
    return records,receipts,set(graphs)


def train(records,seed=SEED,epochs=12):
    torch.set_num_threads(1);torch.manual_seed(seed+31)
    baseline=GateNet().eval();models=(copy.deepcopy(baseline),copy.deepcopy(baseline),baseline)
    X=torch.from_numpy(np.asarray([r[0] for r in records],dtype=np.float32))
    true=np.asarray([r[1] for r in records],dtype=np.float32)
    rng=np.random.default_rng(seed+13);permuted=true[rng.permutation(len(true))]
    orders=[rng.permutation(len(X)).tolist() for _ in range(epochs)]
    npos=int(true.sum());nneg=len(true)-npos
    assert npos>5 and nneg>5, (npos,nneg)
    weights=torch.tensor(np.clip(nneg/max(1,npos),.3,5.),dtype=torch.float32)
    stats={}
    for name,net,labels in zip(('ALIGNED','SHUFFLED'),models,(true,permuted)):
        opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.01)
        target=torch.from_numpy(labels)
        curves=[];updates=0
        for order in orders:
            acc=0.
            for k in order:
                pred=net(X[k]);loss=nn.functional.binary_cross_entropy_with_logits(pred,target[k],pos_weight=weights)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                acc+=float(loss.detach());updates+=1
            curves.append(round(acc/len(order),6))
        net.eval();stats[name]={'optimizer_updates':updates,'epoch_mean_loss':curves}
    stats['training_states']=len(true);stats['positive']=npos;stats['negative']=nneg
    return models,stats


def blind(base,option,prior,heads,seed=452201,per_group=3):
    records=[];graphs=[]
    for fam in c42.FAMILIES:
      for case in ('detour','forward'):
       for person in (-1,1):
        for i in range(per_group):
          s=seed+i*18889+person*173+(case=='forward')*983+c42.FAMILIES.index(fam)*53003
          w,info=c42.world(s,fam,case,person);graphs.append(info['graph_sha'])
          for snap,phase,t in c44.guided_states(w,base,option,s,cap=2):
            menu,logits=local_baseline(snap,phase,base,option)
            legal=[j for j in range(9) if menu[j].legal]
            if len(legal)<2:continue
            x=torch.from_numpy(action_features(snap,menu)[legal]);rt=np.asarray([executed_suffix(snap,j,steps=20) for j in legal])
            with torch.no_grad():ps=prior(x).numpy()
            newer=logits.copy()
            for j,v in zip(legal,ps-ps.mean()):newer[j]+=float(.8*np.tanh(v))
            newer,_,_,_=style_project(menu,logits,newer,snap.personality,max_shift=.045)
            adv=float(np.dot(probs(newer)[legal]-probs(logits)[legal],rt))
            f=shared_features(x,prior)
            records.append({'graph':info['graph_sha'],'family':fam,'case':case,'person':person,
             'state_sha':sha([snap.pos,snap.goal,snap.history]),'advantage':adv,'label':int(adv>1e-5),
             'pred':{k:float(v(f).sigmoid().item()) for k,v in heads.items()}})
    assert len(set(graphs))==len(graphs)
    return records,set(graphs)


def online(base,option,prior,heads):
    models={'C40-PAIR':prior,'C45-GATE':SelectiveHead(prior,heads['C45-GATE']),
            'C45-SHUFFLE':SelectiveHead(prior,heads['C45-SHUFFLE']),
            'C45-MARGIN':SelectiveHead(prior,rule=True)}
    rows=[];graphs=[];traces=[]
    for cohort,seed in COHORTS:
      for fam in c42.FAMILIES:
       for case in ('detour','forward'):
        for person in (-1,1):
         for switch in (False,True):
          s=seed+person*371+(case=='forward')*673+switch*2039+c42.FAMILIES.index(fam)*47011
          w,info=c42.world(s,fam,case,person);graphs.append(info['graph_sha'])
          for directive,strength in ARMS:
            for model in MODELS:
              mode=model if model in ('C18-A0','C34-RESET') else 'C39-ALIGNED'
              th=[models[model]]*3 if mode=='C39-ALIGNED' else []
              record=play(base,option,th,copy.deepcopy(w),info,s,person,mode,directive,strength,switch,78,False)
              rows.append({'cohort':cohort,'model':model,'graph':info['graph_sha'],'family':fam,
                  'case':case,'person':person,'switch':switch,'directive':directive,'strength':strength,
                  'protected_sha':sha(sorted(w.protected)),**record})
              if cohort=='A' and case=='detour' and person==1 and switch and directive=='backtrack' and strength==1.:
                traced=play(base,option,th,copy.deepcopy(w),info,s,person,mode,directive,strength,switch,78,True)
                assert (traced['actions_sha'],traced['ink_sha'])==(record['actions_sha'],record['ink_sha'])
                traces.append({'cohort':cohort,'model':model,'family':fam,'graph':info['graph_sha'],
                   'initial':w.pos,'protected':sorted(w.protected),'actions_sha':record['actions_sha'],
                   'ink_sha':record['ink_sha'],'goals':record['goals'],'trajectory':traced['trajectory']})
      print('C45 online cohort',cohort,flush=True,file=sys.stderr)
    assert len(set(graphs))==len(graphs)
    return rows,set(graphs),traces,{k:{'eligible':v.eligible,'enabled':v.enabled} for k,v in models.items() if isinstance(v,SelectiveHead)}


def summarize(records,blind_rows,online_rows,graphs,trainstats,checkpoint_checks,selection):
    tr,bl,on=graphs
    assert tr.isdisjoint(bl) and tr.isdisjoint(on) and bl.isdisjoint(on)
    key=lambda r:(r['graph'],r['directive'],r['strength'])
    baseline={key(r):r for r in online_rows if r['model']=='C18-A0'}
    guided=[r for r in online_rows if r['directive']!='none'];nonguided=[r for r in online_rows if r['directive']=='none']
    assert all(r['host_rejections']==0 for r in online_rows)
    cond=collections.defaultdict(lambda:collections.Counter());pergraph=collections.defaultdict(collections.Counter)
    for r in guided:
       cond[(r['cohort'],r['family'],r['case'],r['switch'],r['directive'],r['strength'])][r['model']]+=r['goals']
       pergraph[r['graph']][r['model']]+=r['goals']
    summary={'schema':'driv3r-c45-counterfactual-selective-gate/1','graphs':{'train':len(tr),'blind':len(bl),'online':len(on)},
       'episodes':len(online_rows),'guided_episodes_per_policy':len(guided)//len(MODELS),'training':trainstats,
       'checkpoint_sha256':checkpoint_checks,'selection':selection,'models':{},'paired':{},'blind':{}}
    for name in MODELS:
       g=[r for r in guided if r['model']==name];u=[r for r in nonguided if r['model']==name]
       assert all((r['actions_sha'],r['ink_sha'])==(baseline[key(r)]['actions_sha'],baseline[key(r)]['ink_sha']) for r in u)
       summary['models'][name]={'goal_events':sum(r['goals'] for r in g),
         'unguided_exact':len(u),'condition_regressions_vs_RESET':sum(v[name]<v['C34-RESET'] for v in cond.values()),
         'raw_illegal_top_choice':sum(r['raw_illegal_top1'] for r in g),
         'preference_fraction':sum(r['preferred'] for r in g)/max(1,sum(r['preferred']+r['opposed'] for r in g)),
         'by_family':{f:sum(r['goals'] for r in g if r['family']==f) for f in c42.FAMILIES}}
    for name in ('C40-PAIR','C45-GATE','C45-SHUFFLE','C45-MARGIN'):
      for control in ('C34-RESET','C40-PAIR','C45-SHUFFLE'):
        if name==control:continue
        d=np.asarray([v[name]-v[control] for v in pergraph.values()]);rng=np.random.default_rng(450011+MODELS.index(name)*29+MODELS.index(control));
        picks=rng.integers(len(d),size=(4000,len(d)))
        summary['paired'][f'{name}__{control}']={'delta':int(d.sum()),'positive_graphs':int((d>0).sum()),'negative_graphs':int((d<0).sum()),
           '95pct_descriptive_graph_cluster_bootstrap':[float(v) for v in np.quantile(d[picks].sum(axis=1),[.025,.975])]}
    for name in ('C45-GATE','C45-SHUFFLE'):
       labels=np.asarray([r['label'] for r in blind_rows]);prob=np.asarray([r['pred'][name] for r in blind_rows]);
       summary['blind'][name]={'states':len(labels),'positive_fraction':float(labels.mean()),'accuracy':float(np.mean((prob>=.5)==labels)),
           'brier':float(np.mean((prob-labels)**2)),'gate_rate':float((prob>=.5).mean())}
    return summary


def run(root,out):
    torch.set_num_threads(1);root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base,option,prior=c44.prerequisites(root)
    records,receipts,tr=collect(base,option,prior)
    models,stats=train(records)
    repeat,again=train(records)
    assert stats==again and all(all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict()) for a,b in zip(models,repeat))
    heads={'C45-GATE':models[0],'C45-SHUFFLE':models[1]};checks={}
    for name,model in heads.items():
        path=out/(name+'.safetensors');save_file({k:v.contiguous() for k,v in model.state_dict().items()},str(path))
        checks[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    blind_rows,bl=blind(base,option,prior,heads)
    online_rows,on,traces,selection=online(base,option,prior,heads)
    summary=summarize(records,blind_rows,online_rows,(tr,bl,on),stats,checks,selection)
    summary['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary['training_receipts_sha256']=sha(receipts)
    summary['limitations']=['synthetic fixture, not CyberSand native ABI','approximate RESET phase at sampled states, not complete recurrent state',
       '20-action executed suffix proxies are not long-horizon goal success','fixed .5 gate threshold; no online tuning',
       'per-graph bootstrap intervals descriptive, not scientific proof','whole-trajectory preference proxy confounded by state distribution',
       'checkpoints local; immutable Actions artifact custody absent']
    for filename,obj in [('C45-summary.json',summary),('C45-blind.json',blind_rows),('C45-online.json',online_rows),('C45-traces.json',traces),('C45-training-receipts.json',receipts)]:
        (out/filename).write_text(json.dumps(obj,sort_keys=True,indent=2 if filename=='C45-summary.json' else None)+'\n')
    print(json.dumps({'graphs':summary['graphs'],'training':{k:v for k,v in stats.items() if k not in ('ALIGNED','SHUFFLED')},
     'goals':{k:v['goal_events'] for k,v in summary['models'].items()},'regression':{k:v['condition_regressions_vs_RESET'] for k,v in summary['models'].items()},
     'blind':summary['blind'],'paired':summary['paired'],'selection':selection},indent=2),flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('out');a=parser.parse_args();run(a.root,a.out)

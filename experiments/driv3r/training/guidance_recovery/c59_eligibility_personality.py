"""C59: observable-evidence eligibility gate, remote-goal safety and personality.

DRIV3R synthetic host only. Frozen C58 action head, learned *eligibility*
head, deterministic evidence-gate control; all movement freshly authorized.
No hidden maze knowledge or native CyberSand integration.
"""
from __future__ import annotations
import collections, copy, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file
import c58_dual_memory_recovery as c58
from experiment import sha
from c28_temporal_option import scores,probs
from c34_independent_maze import ReArmedWatch

SEED=590118
DROPS=(0.,.5,1.)
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
POLICIES=('C18-A0','C34-RESET','C58-ALIGNED','C59-STRICT','C59-SOFT','C59-LEARNED')
HORIZON=78

def evidence_features(w,ledger):
    """Features from retained observed menus, executed trace and current legal goal.

    Crucially does not consume experiment goaltype (observed/remote), full graph,
    latent shortest path, or future outcomes.
    """
    goal=tuple(w.goal)
    known={tuple(t) for neighbors in ledger.neighbors.values() for t in neighbors}
    r=ledger.route(w.pos,goal)
    return np.asarray([float(goal in known),float(r is not None),
        min(1.,len(ledger.neighbors)/24.),min(1.,len(ledger.actions)/24.),
        min(1.,ledger.visits[tuple(w.pos)]/8.),
        min(1.,w.distance()/32.)],dtype=np.float32)

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(6,1)
    def forward(self,x):return self.linear(x).squeeze(-1)

def train_gate(base,trainw):
    states=[]
    for gt in ('observed','remote'):
        ss,_=c58.c56.states_for_goal(base,trainw,gt)
        states.extend((gt,x) for x in ss)
    rows=[]
    for goaltype,st in states:
        snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
        w=copy.deepcopy(snap);w.goal=tuple(target)
        for drop in DROPS:
            led=c58.damaged(log,info['graph_sha'],tick,drop)
            # The teacher uses only the observed directed graph, never a maze oracle.
            x=evidence_features(w,led)
            y=int(led.route(w.pos,w.goal) is not None)
            rows.append((info['graph_sha'],goaltype,drop,x,y))
    assert rows and len({r[0] for r in rows})>=8
    xx=torch.from_numpy(np.stack([r[3] for r in rows]));yy=torch.tensor([r[4] for r in rows],dtype=torch.float32)
    torch.manual_seed(SEED)
    model=Gate();opt=torch.optim.AdamW(model.parameters(),lr=.012,weight_decay=.001)
    positives=int(yy.sum().item())
    assert 0<positives<len(rows), 'gate training requires positive and negative examples'
    weights=torch.where(yy>0,.5/max(1,positives),.5/max(1,len(rows)-positives))*len(rows)
    for step in range(240):
        opt.zero_grad(); loss=nn.functional.binary_cross_entropy_with_logits(model(xx),yy,weight=weights)
        loss.backward();opt.step()
    model.eval()
    with torch.no_grad(): pp=torch.sigmoid(model(xx)).numpy()
    pred=(pp>=.5).astype(int)
    return model,{'rows':len(rows),'graphs':len({r[0] for r in rows}),'positives':positives,
        'adamw_steps':240,'training_accuracy':float(np.mean(pred==yy.numpy())),
        'training_brier':float(np.mean((pp-yy.numpy())**2)),
        'positive_predictions':int(pred.sum()),'final_bce':float(loss.item())}

def movement_metrics(points, actions):
    """First 20 executed steps only; no unequal-duration path comparisons."""
    if len(actions)<20:return None
    positions=[tuple(x) for x in points[:21]]
    dirs=[]
    for a,b in zip(positions,positions[1:]):
        dx,dy=b[0]-a[0],b[1]-a[1]
        if abs(dx)+abs(dy)==1:dirs.append((dx,dy))
    if len(dirs)<5:return None
    turn=sum(dirs[i]!=dirs[i-1] for i in range(1,len(dirs)))
    reverse=sum(dirs[i]==(-dirs[i-1][0],-dirs[i-1][1]) for i in range(1,len(dirs)))
    visited=len(set(positions))/len(positions)
    counts=collections.Counter(dirs)
    entropy=-sum((n/len(dirs))*math.log2(n/len(dirs)) for n in counts.values())/2.
    return {'turn_rate':turn/max(1,len(dirs)-1),'reverse_rate':reverse/max(1,len(dirs)-1),
      'unique_ratio':visited,'direction_entropy':entropy,
      'paint_fraction':sum(a[0][0]=='INK' for a in actions[:20])/20.}

def execute(base,opt,snap,rngstate,log,tick,target,arm,name,trained,gate,graph,drop,trace=False):
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c58.c55.cue(w,arm);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rngstate)
    led=c58.damaged(log,graph,tick,drop)
    ctrl=None if name=='C18-A0' or arm[0]=='none' else ReArmedWatch(opt,None)
    path=[];points=[tuple(w.pos)];acts=[];counts=collections.Counter()
    for t in range(tick,HORIZON):
        if w.stopped or w.distance()==0:break
        led.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        if ctrl is not None and name not in ('C18-A0','C34-RESET'):
            legal,nxt,mode=c58.c57.recommendation(w,led,menu,advice)
            if legal:
                use=True
                if name in ('C59-STRICT','C59-SOFT'):
                    use=led.route(w.pos,w.goal) is not None
                elif name=='C59-LEARNED':
                    p=float(1/(1+np.exp(-(float(evidence_features(w,led)@gate[0]+gate[1])))))
                    counts['gate_positive']+=int(p>=.5)
                    use=p>=.5
                if use:
                    counts['intervention_turns']+=1
                    counts['intervention_'+mode]+=1
                    xx=np.asarray([c58.c55.feature(w,led,menu,c,nxt) for c in menu[:8]],np.float32)
                    z=np.tanh(xx@trained[0].T+trained[1])@trained[2].T+trained[3]
                    z=z[:,0]
                    gain=1.5 if name=='C59-SOFT' else 3.
                    delta=np.zeros(len(menu),np.float32)
                    delta[:8]=gain*arm[1]*np.tanh(np.clip(z,-20,20)/2.)
                    logits=np.asarray(logits,dtype=np.float32)+delta
                else:counts['abstentions']+=1
        c=menu[int(rng.choice(len(menu),p=probs(logits)))];before=tuple(w.pos)
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        led.after(before,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w);acts.append((c.action,c.target));points.append(tuple(w.pos))
        if c.action[0] in ('INK','LIFT'):
            preference=float(c.features[13]-c.features[14])*w.personality
            counts['personality_preferred']+=int(preference>0)
            counts['personality_opposed']+=int(preference<0)
        if trace:path.append({'t':t,'p':list(w.pos),'a':c.action[0]})
    return {'goal':int(w.distance()==0),'steps':len(acts),'distance':w.distance(),
      'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),'protected_sha':sha(sorted(w.protected)),
      'host_rejections':w.invalid_attempts,'counts':dict(counts),'personality20':movement_metrics(points,acts),
      'path':path if trace else None}

def run(prior,output,train_reps=2,online_reps=2):
    prior=Path(prior);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c58.c55.TICKS=(9,)
    base,opt,_=c58.c55.c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    historical=set()
    for name in ('C56-summary.json','C57-summary.json','C58-summary.json'):
        old=json.loads((prior/name).read_text())
        for k in ('train','online'): historical.update(x['graph'] for x in old['cohorts'][k])
    tr,trec=c58.c55.c54.c52.worlds(c58.c55.c54.c52.TRAIN_FAMILIES,
        c58.c55.c54.c52.c50.c47.c41.world,SEED+31013,train_reps,exclude=historical)
    gate,fit=train_gate(base,tr)
    gate_np=(gate.linear.weight.detach().numpy().reshape(-1),float(gate.linear.bias.detach().numpy()[0]))
    outfile=output/'C59-GATE.safetensors';save_file(gate.state_dict(),str(outfile))
    frozen=c58.c55.Head();frozen.load_state_dict(load_file(str(prior/'C58-ALIGNED.safetensors')));frozen.eval()
    W=frozen.net[0].weight.detach().numpy();b=frozen.net[0].bias.detach().numpy()
    V=frozen.net[2].weight.detach().numpy();c=frozen.net[2].bias.detach().numpy()
    frozen_np=(W,b,V,c)
    excluded=historical|{r['graph'] for r in trec}
    unseen,ur=c58.c55.c54.c52.worlds(c58.c55.c54.c52.ONLINE_FAMILIES,
       c58.c55.c54.c52.c50.c47.c42.world,SEED+915101,online_reps,exclude=excluded)
    fresh,fr=c58.c55.c54.c52.worlds(c58.c55.c54.c52.TRAIN_FAMILIES,
       c58.c55.c54.c52.c50.c47.c41.world,SEED+815101,online_reps,
       exclude=excluded|{r['graph'] for r in ur})
    states=[];skips={}
    for gt in ('observed','remote'):
        ss,sk=c58.c56.states_for_goal(base,unseen+fresh,gt);states.extend((gt,x) for x in ss);skips[gt]=sk
    records=[];traces=[];counts=collections.defaultdict(collections.Counter);matched=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    unguided=0;gate_confusion=collections.Counter();pers=collections.defaultdict(list)
    for gt,st in states:
      snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
      graph=info['graph_sha']
      for drop in DROPS:
        w=copy.deepcopy(snap);w.goal=tuple(target);led=c58.damaged(log,graph,tick,drop)
        x=evidence_features(w,led)
        p=float(1/(1+np.exp(-(float(x@gate_np[0]+gate_np[1])))))
        y=int(led.route(w.pos,w.goal) is not None)
        gate_confusion[(y,int(p>=.5))]+=1
        for arm in ARMS:
          result={name:execute(base,opt,snap,rngstate,log,tick,target,arm,name,frozen_np,gate_np,graph,drop)
                  for name in POLICIES}
          assert all(z['host_rejections']==0 and z['protected_sha']==receipt['protected_sha'] for z in result.values())
          if arm[0]=='none':
              assert len({(z['action_sha'],z['ink_sha']) for z in result.values()})==1
              unguided+=1
          else:
              for name,z in result.items():
                  c=counts[(gt,drop,fam,name)];c['episodes']+=1;c['goals']+=z['goal']
                  c['abstentions']+=z['counts'].get('abstentions',0)
                  c['interventions']+=z['counts'].get('intervention_turns',0)
                  c['preferred']+=z['counts'].get('personality_preferred',0)
                  c['opposed']+=z['counts'].get('personality_opposed',0)
                  if z['personality20'] is not None:pers[(gt,drop,arm[1],graph,name)].append(z['personality20'])
                  matched[(gt,drop,fam)][graph][name]+=z['goal']
          records.append({'graph':graph,'family':fam,'case':case,'seed':seed,'tick':tick,
              'goaltype':gt,'target':list(target),'drop':drop,'arm':arm,
              'prefix_sha':receipt['prefix_sha'],'prefix_ink_sha':receipt['ink_sha'],
              'gate_pre':{'label':y,'prediction':p},'outcomes':result})
          if arm==('backtrack',1.) and len(traces)<24 and drop in (0.,.5,1.):
              for name in ('C34-RESET','C58-ALIGNED','C59-STRICT','C59-LEARNED'):
                  if len(traces)>=24:break
                  z=execute(base,opt,snap,rngstate,log,tick,target,arm,name,frozen_np,gate_np,graph,drop,True)
                  assert (z['action_sha'],z['ink_sha'])==(result[name]['action_sha'],result[name]['ink_sha'])
                  traces.append({'graph':graph,'family':fam,'seed':seed,'case':case,'goaltype':gt,
                     'drop':drop,'tick':tick,'target':list(target),'prefix':receipt,'policy':name,'outcome':z})
    # Censor-safe matched personality: only pairs with >=20 executed steps in both runs.
    personality={}
    for name in POLICIES:
      if name=='C34-RESET':continue
      deltas=collections.defaultdict(list)
      for (gt,drop,strength,graph,n),items in pers.items():
        if n!=name:continue
        baseline=pers.get((gt,drop,strength,graph,'C34-RESET'))
        if not baseline:continue
        for key in ('turn_rate','reverse_rate','unique_ratio','direction_entropy','paint_fraction'):
            deltas[(gt,drop,key)].append(items[0][key]-baseline[0][key])
      personality[name]={str(k):{'n_matched':len(v),'mean_delta':float(np.mean(v))} for k,v in deltas.items()}
    paired={}
    for k,g in matched.items():
      for name in POLICIES:
        if name=='C34-RESET':continue
        dif=np.asarray([v[name]-v['C34-RESET'] for v in g.values()],dtype=int)
        boot=np.random.default_rng(SEED+len(name)*37+int(k[1]*10)+len(k[2])).choice(dif,size=(2000,len(dif)),replace=True).sum(axis=1)
        paired[str((k,name))]={'delta':int(dif.sum()),'positive_graphs':int(sum(dif>0)),
             'negative_graphs':int(sum(dif<0)),'descriptive_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    s={'schema':'driv3r-c59-observable-eligibility/1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'frozen_c58_sha256':hashlib.sha256((prior/'C58-ALIGNED.safetensors').read_bytes()).hexdigest(),
       'checkpoint_sha256':hashlib.sha256(outfile.read_bytes()).hexdigest(),
       'training':fit,'train_graphs':len(trec),'online_graphs':len(ur)+len(fr),'skips':skips,
       'unguided_parity':unguided,'record_count':len(records),'trace_replays':len(traces),
       'gate_confusion':{str(k):v for k,v in gate_confusion.items()},
       'cohorts':{'train':trec,'online':ur+fr},
       'models':{str(k):dict(v) for k,v in counts.items()},'paired':paired,'personality20':personality,
       'caveats':['Synthetic host only; not native CyberSand or production ABI',
       'Eligibility head imitates a visible-route test; not autonomous navigation training',
       'Observed frontier waypoint task easier than unobserved remote goal',
       'Personality metrics are first-20-step movement proxies and censor short runs',
       'Cluster intervals unadjusted after extensive adaptive prior experiments',
       'No immutable GitHub Actions model artifact custody']}
    for name,obj in [('C59-summary.json',s),('C59-online.json',records),('C59-traces.json',traces)]:
      (output/name).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'training':fit,'online_graphs':s['online_graphs'],'parity':unguided,
       'gate':s['gate_confusion'],'goals':{k:v['goals'] for k,v in s['models'].items()}},indent=2),flush=True)
    return s

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('output');p.add_argument('--train-reps',type=int,default=2);p.add_argument('--online-reps',type=int,default=2)
 a=p.parse_args();run(a.prior,a.output,a.train_reps,a.online_reps)
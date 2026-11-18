"""C35: cross-generator executed-return calibration and cue-constrained recovery.

External synthetic CyberSand-like host only. All optimizer targets come from
legal host-executed counterfactual suffixes. Held-out Wilson mazes are never
used for fitting or threshold selection. No native/game/ABI changes.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import World, DIRECTIONS, sha
from signed_feedback import HostAdvice
from c28_temporal_option import OptionHead, scores, probs
from c32_offstack_recovery import SearchPolicy, fresh
from c34_independent_maze import maze as dfs_maze, shortest, obs, executed_suffix, ReArmedWatch, RetreatHead
from c29_probe_recovery import PREREQS
from personality_retention_core import load_c12
from c19_return_anchored import a0

SCHEMA='driv3r-c35-cross-generator-calibration/1'
TRAIN_FAMILIES=('dfs','prim')
TEST_FAMILIES=('wilson','wilson_braid')
CASES=('detour','forward')
DIRECTIVES=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',.35),('continue',1.))


def grid_edges():
    nodes={(x,y) for x in range(1,14,2) for y in range(1,14,2)}
    return nodes


def graph(seed,family):
    """Independent randomized Prim and Wilson spanning trees on lattice.

    DFS is retained only as an inherited training-family control. Adding
    extra links creates a separate withheld graph distribution with cycles.
    """
    if family=='dfs':return dfs_maze(seed,'tree')
    nodes=grid_edges();rng=np.random.default_rng(seed)
    def neighbors(p):
        return sorted((p[0]+dx,p[1]+dy) for dx,dy in ((-2,0),(2,0),(0,-2),(0,2))
                      if (p[0]+dx,p[1]+dy) in nodes)
    tree={(1,1)};edges=[]
    if family=='prim':
        fringe=[((1,1),q) for q in neighbors((1,1))]
        while fringe:
            ix=int(rng.integers(len(fringe)));p,q=fringe.pop(ix)
            if q in tree:continue
            edges.append((p,q));tree.add(q)
            fringe.extend((q,v) for v in neighbors(q) if v not in tree)
    elif family.startswith('wilson'):
        # Wilson's loop-erased random walks sample a structurally distinct tree.
        for root in sorted(nodes-{(1,1)}):
            if root in tree:continue
            walk=[root];indices={root:0}
            while walk[-1] not in tree:
                nxt=neighbors(walk[-1])[int(rng.integers(len(neighbors(walk[-1]))))]
                if nxt in indices:
                    walk=walk[:indices[nxt]+1];indices={p:i for i,p in enumerate(walk)}
                else:indices[nxt]=len(walk);walk.append(nxt)
            for p,q in zip(walk,walk[1:]):
                edges.append((p,q));tree.add(p);tree.add(q)
    else:raise ValueError(family)
    assert len(tree)==49 and len(edges)==48,(family,len(tree),len(edges))
    allowed=set(nodes)
    for p,q in edges:allowed.add(((p[0]+q[0])//2,(p[1]+q[1])//2))
    if family=='wilson_braid':
        possible=[]
        for p in sorted(nodes):
            for dx,dy in ((2,0),(0,2)):
                q=(p[0]+dx,p[1]+dy)
                mid=(p[0]+dx//2,p[1]+dy//2)
                if q in nodes and mid not in allowed:possible.append(mid)
        rng.shuffle(possible);allowed.update(possible[:14])
    assert len(allowed)==(111 if family=='wilson_braid' else 97)
    return allowed


def _world_once(seed,family,case,person):
    allowed=graph(seed,family);start=(1,1)
    longest=max((shortest(allowed,start,p) for p in allowed),key=len)
    if len(longest)<25:return None
    prefix=longest[:25]
    w=World(seed,'islands',person)
    w.protected={(x,y) for x in range(15) for y in range(15) if (x,y) not in allowed}
    w.pos=start;w.goal=prefix[-1];w.inked={start}
    for nxt in prefix[1:]:
        m=[c for c in w.candidates(0.)[:4] if c.legal and c.target==nxt]
        assert len(m)==1 and w.apply(m[0]) and w.invalid_attempts==0
    candidates=[]
    for p in sorted(allowed-set(prefix)):
        route=shortest(allowed,w.pos,p)
        if route and len(route)>=6 and (route[1]==w.history[-1])==(case=='detour'):
            candidates.append((p,route))
    if not candidates:return None
    best=max(len(r) for _,r in candidates)
    choices=[(p,r) for p,r in candidates if len(r)>=max(6,best-8)]
    rng=np.random.default_rng(seed+3881+(1 if case=='detour' else 2))
    goal,route=choices[int(rng.integers(len(choices)))];w.goal=goal
    w.goal_change=True;w.goals_seen=2;w.stuck=2
    other=sorted((p for p in allowed-set(prefix) if p!=goal),
                 key=lambda p:(-len(shortest(allowed,goal,p)),p))
    assert other
    info={'graph_sha':sha(sorted(allowed)),'prefix_sha':sha(prefix),'prefix_len':24,
          'initial_shortest':len(route)-1,'second_goal':other[0],
          'nodes':len(allowed),'family':family,'initial_retreat':route[1]==w.history[-1]}
    return w,info


def make_world(seed,family,case,person):
    assert family in TRAIN_FAMILIES+TEST_FAMILIES and case in CASES
    for retry in range(100):
        s=seed+retry*7919;v=_world_once(s,family,case,person)
        if v is not None:
            w,info=v;info['seed']=s;info['retry']=retry;return w,info
    raise AssertionError(('fixture generation failed',seed,family,case))


def features(w,menu):
    """15 strictly host-visible features, no shortest-path or graph oracle."""
    f=fresh(w,menu)
    progress=[np.clip(w.distance()-w.distance(menu[j].target),-1.,1.)
              for j,c in enumerate(menu[:8]) if c.legal and c.action[0] in ('INK','LIFT')]
    novelty=[j for j,c in enumerate(menu[:8]) if c.legal and c.target not in w.history]
    back_on_stack=float(bool(w.history and menu[8].legal))
    return np.concatenate([obs(w,menu),np.asarray([
        max(progress,default=-1.),min(progress,default=1.),
        len(novelty)/8.,min(len(w.inked),100)/100.,
        back_on_stack, float(len(f)>1)],np.float32)])


def samples(seed,family,count,steps=4):
    rows=[];xx=[];yy=[]
    for i in range(count):
        for person in (-1,1):
            for case in CASES:
                w,info=make_world(seed+17011*i+109*person+(case=='forward')*37,family,case,person)
                for t in range(steps):
                    menu=w.candidates(.35)
                    forward=[j for j in range(8) if menu[j].legal and menu[j].target not in w.history]
                    if not menu[8].legal or not forward:break
                    j=max(forward,key=lambda k:(w.distance()-w.distance(menu[k].target),-k))
                    back=executed_suffix(w,8,8);fwd=executed_suffix(w,j,8)
                    delta=float(np.clip(back-fwd,-1.5,1.5))
                    xx.append(features(w,menu));yy.append(delta)
                    rows.append({'family':family,'graph_sha':info['graph_sha'],
                        'state_sha':sha([w.pos,w.history,w.goal]),
                        'legal_menu_sha':sha([(c.action,c.target,c.legal) for c in menu]),
                        'back':back,'forward':fwd,'delta':delta})
                    assert w.apply(menu[j]) and w.invalid_attempts==0
    return np.stack(xx),np.asarray(yy,np.float32),rows


class CalHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(15,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def train(seed=351011,epochs=16):
    train_chunks=[];val_chunks=[];receipts=[]
    for family in TRAIN_FAMILIES:
        x,y,r=samples(seed+(family=='prim')*110001,family,24)
        train_chunks.append((x,y));receipts.extend(r)
        a,b,_=samples(seed+900011+(family=='prim')*100019,family,8)
        val_chunks.append((a,b))
    X=np.concatenate([x for x,_ in train_chunks]);Y=np.concatenate([y for _,y in train_chunks])
    VX=np.concatenate([x for x,_ in val_chunks]);VY=np.concatenate([y for _,y in val_chunks])
    x=torch.from_numpy(X);y=torch.from_numpy(Y);vx=torch.from_numpy(VX);vy=torch.from_numpy(VY)
    models=[];fit=[];snapshots=[]
    for replica in range(3):
        torch.manual_seed(seed+replica*557);torch.set_num_threads(1)
        head=CalHead();opt=torch.optim.AdamW(head.parameters(),lr=.008,weight_decay=.02)
        rng=np.random.default_rng(seed+replica*557)
        last=[];updates=0
        for epoch in range(epochs):
            ix=rng.permutation(len(Y));epoch_loss=0.
            for begin in range(0,len(ix),48):
                ids=ix[begin:begin+48]
                loss=nn.functional.smooth_l1_loss(head(x[ids]),y[ids]);opt.zero_grad(set_to_none=True)
                loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                updates+=1;epoch_loss+=float(loss.detach())
            last.append(round(epoch_loss,6))
        head.eval()
        with torch.no_grad():
            pred=head(vx).numpy();mae=float(np.mean(abs(pred-VY)));train_mae=float(torch.abs(head(x)-y).mean())
        models.append(head);fit.append({'replica':replica,'updates':updates,'train_mae':train_mae,
            'validation_mae':mae,'losses':last})
    details={'train_n':len(Y),'val_n':len(VY),'positive':int((Y>0).sum()),
        'negative':int((Y<0).sum()),'zero':int((Y==0).sum()),
        'train_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
        'validation_sha256':hashlib.sha256(VX.tobytes()+VY.tobytes()).hexdigest(),
        'heads':fit}
    return models,details,receipts


def calibrate(models,seed=357001,count=18):
    out={}
    for fam in TRAIN_FAMILIES+TEST_FAMILIES:
        x,y,rows=samples(seed+100003*(TRAIN_FAMILIES+TEST_FAMILIES).index(fam),fam,count)
        tx=torch.from_numpy(x)
        with torch.no_grad():pred=np.stack([m(tx).numpy() for m in models])
        mean=pred.mean(axis=0);spread=pred.std(axis=0)
        # Actionable sign excludes near-ties. Other rows count for MAE.
        nonzero=abs(y)>.15
        out[fam]={'n':len(y),'near_tie_n':int((~nonzero).sum()),'signed_n':int(nonzero.sum()),
            'signed_accuracy':float(np.mean(np.sign(mean[nonzero])==np.sign(y[nonzero]))) if nonzero.any() else None,
            'mae':float(np.mean(abs(mean-y))),'baseline_zero_mae':float(np.mean(abs(y))),
            'prediction_sha256':hashlib.sha256(mean.tobytes()+spread.tobytes()).hexdigest(),
            'host_receipts_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
            'empirical_positive_rate':float(np.mean(y>.15)),
            'predicted_positive_rate':float(np.mean(mean>.15)),
            'predicted_std_mean':float(spread.mean()),
            'dataset_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()}
    return out


class C35Policy(ReArmedWatch):
    def __init__(self,option,heads,variant):
        super().__init__(option,None)
        self.heads=heads;self.variant=variant
        self.head_decisions=0;self.modified=0;self.uncertainty_veto=0
    def modify(self,base,w,advice,arm):
        menu,logits,bad,target=super().modify(base,w,advice,arm)
        if self.phase in ('watch','probe','retrace') and advice.directive==1 and menu[8].legal:
            with torch.no_grad():vals=np.asarray([float(h(torch.from_numpy(features(w,menu)))) for h in self.heads])
            self.head_decisions+=1;avg=float(vals.mean());sd=float(vals.std())
            if self.variant=='GUARD':
                if avg<=.15 or sd>.35 or (vals<=0).any():
                    self.uncertainty_veto+=1;change=0.
                else:change=min(1.5,avg*1.25)
            else:change=float(np.clip(avg*1.25,-1.5,1.5))
            logits[8]+=change;self.modified+=int(abs(change)>1e-8)
        return menu,logits,bad,target


def play(base,option,heads,w,info,seed,arm,directive,strength,model,switch=False,horizon=85,trace=False):
    cue=HostAdvice(w)
    if directive!='none':getattr(cue,'issue_'+directive)(w,strength=strength,ttl=9)
    if model.startswith('C35-'):policy=C35Policy(option,heads,model.split('-')[1])
    elif model=='C34-RESET':policy=ReArmedWatch(option,None)
    else:policy=None
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+
                              {'none':0,'backtrack':411,'continue':977}[directive])
    protected=sha(sorted(w.protected));baseline=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    trace_rows=[];actions=[];rejected=0;first=0;second=0;ink=0;lift=0;back=0;unique=set()
    for t in range(horizon):
        if switch and t==50:
            first=int(first or w.distance()==0);w.goal=info['second_goal'];w.goal_change=True
            w.goals_seen+=1;w.stuck=2
            if directive!='none':getattr(cue,'issue_'+directive)(w,strength=strength,ttl=9)
        if w.stopped:break
        if w.distance()==0:
            if t<50 or not switch:first=1
            else:second=1
            if not switch or t>=50:break
            continue
        if policy is None:menu,logits,bad,_=scores(base,w,cue,arm)
        else:menu,logits,bad,_=policy.modify(base,w,cue,arm)
        rejected+=bad
        selected=menu[int(rng.choice(len(menu),p=probs(logits)))]
        assert selected.legal and w.apply(selected) and w.invalid_attempts==0
        if policy is not None:policy.after(selected)
        cue.advance(w);actions.append((selected.action,selected.target));unique.add(tuple(w.pos))
        ink+=selected.action[0]=='INK';lift+=selected.action[0]=='LIFT';back+=selected.action[0]=='BACKTRACK'
        if trace:trace_rows.append({'t':t,'pos':w.pos,'goal':w.goal,'action':selected.action[0],
                   'distance':w.distance(),'phase':policy.phase if policy is not None else 'baseline'})
        if t<50 and w.distance()==0:first=1
        if switch and t>=50 and w.distance()==0:second=1
    if not switch:first=int(first or w.distance()==0)
    assert protected==sha(sorted(w.protected))
    return {'first_success':first,'second_success':second,'total_success':first+second if switch else first,
      'steps':len(actions),'actions_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
      'prefix_sha':baseline,'raw_illegal_top1':rejected,'host_rejections':w.invalid_attempts,
      'ink_actions':int(ink),'lift_actions':int(lift),'backtrack_actions':int(back),
      'unique_visited':len(unique),'head_decisions':policy.head_decisions if isinstance(policy,C35Policy) else 0,
      'head_modified':policy.modified if isinstance(policy,C35Policy) else 0,
      'uncertainty_veto':policy.uncertainty_veto if isinstance(policy,C35Policy) else 0,
      'path':trace_rows if trace else None}


def evaluate(base,option,heads,seed,n=2,horizon=85):
    models=('C18-A0','C34-RESET','C35-SIGNED','C35-GUARD')
    rows=[];graphs={}
    for family in TRAIN_FAMILIES+TEST_FAMILIES:
     for case in CASES:
      for switch in (False,True):
       for i in range(n):
        for person in (-1,1):
         for arm in (.35,1.):
          for directive,strength in DIRECTIVES:
           s=(seed+10007*i+379*person+int(arm*100)*199+sum(map(ord,case))*411+
              (TRAIN_FAMILIES+TEST_FAMILIES).index(family)*17291+int(switch)*12373)
           w0,info=make_world(s,family,case,person);graphs[info['graph_sha']]=info
           for model in models:
            w=copy.deepcopy(w0)
            record=play(base,option,heads,w,info,s,arm,directive,strength,model,switch,horizon,
                trace=i==0 and person==1 and arm==1. and directive=='backtrack' and strength==1.)
            rows.append({'model':model,'family':family,'case':case,'switch':switch,'i':i,
                'personality':person,'arm':arm,'directive':directive,'strength':strength,
                'seed':s,'graph_sha':info['graph_sha'],**record})
    return rows,graphs


def summarize(rows):
    names=sorted({r['model'] for r in rows});result={}
    key=lambda r:(r['family'],r['case'],r['switch'],r['i'],r['personality'],r['arm'],r['directive'],r['strength'])
    base={key(r):r for r in rows if r['model']=='C18-A0'}
    group=lambda r:(r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    for name in names:
        rr=[r for r in rows if r['model']==name];guided=[r for r in rr if r['directive']!='none']
        groups={str(k):sum(r['total_success'] for r in guided if group(r)==k) for k in sorted({group(r) for r in guided})}
        no=[r for r in rr if r['directive']=='none']
        result[name]={'guided_goals':sum(r['total_success'] for r in guided),
          'guided_episodes':len(guided),'first':sum(r['first_success'] for r in guided),
          'second':sum(r['second_success'] for r in guided),'groups':groups,
          'by_family':{f:sum(r['total_success'] for r in guided if r['family']==f)
                         for f in TRAIN_FAMILIES+TEST_FAMILIES},
          'unguided_identical':sum(r['actions_sha']==base[key(r)]['actions_sha'] and
                               r['ink_sha']==base[key(r)]['ink_sha'] for r in no),
          'unguided_n':len(no),'host_rejections':sum(r['host_rejections'] for r in rr),
          'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr),
          'head_decisions':sum(r['head_decisions'] for r in rr),
          'head_modified':sum(r['head_modified'] for r in rr),
          'uncertainty_veto':sum(r['uncertainty_veto'] for r in rr),
          'personality':{str(p):{'ink':sum(r['ink_actions'] for r in guided if r['personality']==p),
              'lift':sum(r['lift_actions'] for r in guided if r['personality']==p),
              'back':sum(r['backtrack_actions'] for r in guided if r['personality']==p),
              'steps':sum(r['steps'] for r in guided if r['personality']==p)} for p in (-1,1)}}
    for name in names:
        for control in ('C18-A0','C34-RESET'):
            result[name]['regressed_vs_'+control]={k:v-result[name]['groups'][k]
                for k,v in result[control]['groups'].items() if result[name]['groups'][k]<v}
    return result


def run(root,out,seed=351701,n=1,horizon=70):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    heads,fit,receipts=train();checkpoints={}
    for idx,head in enumerate(heads):
        fn=out/f'C35-CAL-{idx}.safetensors';save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(fn))
        checkpoints[str(idx)]=hashlib.sha256(fn.read_bytes()).hexdigest()
    cal=calibrate(heads);output={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fit':fit,'calibration':cal,'checkpoints':checkpoints,'cohorts':{},'topologies':{}}
    for cohort,s in (('A',seed),('B',seed+200003)):
        rows,graphs=evaluate(base,option,heads,s,n,horizon)
        output['cohorts'][cohort]=summarize(rows)
        output['topologies'][cohort]={'unique':len(graphs),'by_family':{f:sum(g['family']==f for g in graphs.values()) for f in TRAIN_FAMILIES+TEST_FAMILIES},
           'min_shortest':min(g['initial_shortest'] for g in graphs.values()),
           'max_shortest':max(g['initial_shortest'] for g in graphs.values())}
        (out/f'cohort_{cohort}.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
        (out/f'topologies_{cohort}.json').write_text(json.dumps(graphs,sort_keys=True)+'\n')
    (out/'C35-metrics.json').write_text(json.dumps(output,sort_keys=True,indent=2)+'\n')
    (out/'training_receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    return output

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('out')
    parser.add_argument('--n',type=int,default=1);parser.add_argument('--seed',type=int,default=351701)
    parser.add_argument('--horizon',type=int,default=70)
    args=parser.parse_args();print(json.dumps(run(args.root,args.out,args.seed,args.n,args.horizon),sort_keys=True))

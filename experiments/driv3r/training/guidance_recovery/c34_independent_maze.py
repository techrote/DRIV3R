"""C34 — independent randomized maze graphs, repeated host goal changes and cues.

Synthetic CyberSand-like host fixture only. Train data contains executed
counterfactual suffixes; inference sees only current legal menu, executed
history and an explicit signed host cue. No graph oracle is read by policies.
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
from c29_probe_recovery import PREREQS
from personality_retention_core import load_c12
from c19_return_anchored import a0

SCHEMA='driv3r-c34-independent-maze/1'
FAMILIES=('tree','braid')
CASES=('detour','forward')


def maze(seed, family):
    """Independent perfect-maze DFS over a 7x7 odd-cell grid, optional cycles.

    Unlike C32/C33, no U trail, rigid corridors, or U-derived prefix is used.
    """
    if family not in FAMILIES:raise ValueError(family)
    rng=np.random.default_rng(seed);start=(1,1);visited={start};stack=[start];allowed={start}
    while stack:
        p=stack[-1]
        choices=[(p[0]+2*dx,p[1]+2*dy) for dx,dy in DIRECTIONS
                 if 1<=p[0]+2*dx<=13 and 1<=p[1]+2*dy<=13 and
                    (p[0]+2*dx,p[1]+2*dy) not in visited]
        if not choices:stack.pop();continue
        q=choices[int(rng.integers(len(choices)))];mid=((p[0]+q[0])//2,(p[1]+q[1])//2)
        allowed.update((q,mid));visited.add(q);stack.append(q)
    assert len(visited)==49 and len(allowed)==97
    if family=='braid':
        # Open additional independent walls to form loops and competing routes.
        possible=[]
        for p in sorted(visited):
            for dx,dy in ((1,0),(0,1)):
                q=(p[0]+2*dx,p[1]+2*dy)
                mid=(p[0]+dx,p[1]+dy)
                if q in visited and mid not in allowed:possible.append(mid)
        rng.shuffle(possible);allowed.update(possible[:14])
    return allowed


def shortest(allowed,start,goal):
    queue=collections.deque([start]);parent={start:None}
    while queue:
        p=queue.popleft()
        if p==goal:
            path=[goal]
            while parent[path[-1]] is not None:path.append(parent[path[-1]])
            return path[::-1]
        for dx,dy in DIRECTIONS:
            q=(p[0]+dx,p[1]+dy)
            if q in allowed and q not in parent:parent[q]=p;queue.append(q)
    return None


def _make_world_once(seed,family,case,personality):
    """Host executes a generated 24-cell prefix then changes to unseen goal."""
    if case not in CASES:raise ValueError(case)
    allowed=maze(seed,family);start=(1,1)
    path=max((shortest(allowed,start,q) for q in allowed),key=len)
    assert len(path)>=25,('prefix too short',seed,len(path))
    prefix=path[:25];w=World(seed,'islands',personality)
    w.protected={(x,y) for x in range(15) for y in range(15) if (x,y) not in allowed}
    w.pos=start;w.goal=prefix[-1];w.inked={start}
    for nxt in prefix[1:]:
        menu=w.candidates(0.)
        moves=[c for c in menu[:4] if c.legal and c.target==nxt]
        assert len(moves)==1 and w.apply(moves[0])
    assert len(w.history)==24 and not w.invalid_attempts
    candidates=[]
    for p in sorted(allowed-set(prefix)):
        route=shortest(allowed,w.pos,p)
        if not route or len(route)<6:continue
        is_retreat=(route[1]==w.history[-1])
        # Use an independent graph oracle only for episode selection/evaluation.
        if (case=='detour' and is_retreat) or (case=='forward' and not is_retreat):
            candidates.append((p,route))
    if not candidates:
        # Still preserve classification of available legal path; never teleport.
        for p in sorted(allowed-set(prefix)):
            route=shortest(allowed,w.pos,p)
            if route and len(route)>=6:candidates.append((p,route))
    assert candidates
    rng=np.random.default_rng(seed+3881+(1 if case=='detour' else 2))
    # Sample from long routes, without giving their length to the model.
    best=max(len(r) for _,r in candidates)
    choices=[(p,r) for p,r in candidates if len(r)>=max(6,best-8)]
    goal,route=choices[int(rng.integers(len(choices)))];w.goal=goal;w.goal_change=True
    w.goals_seen=2;w.stuck=2
    second=[p for p in allowed if p not in set(prefix) and p!=goal and p!=w.pos]
    second.sort(key=lambda p:(-(len(shortest(allowed,goal,p)) or []),p))
    later=second[0]
    info={'graph_sha':sha(sorted(allowed)),'prefix_sha':sha(prefix),'prefix_len':len(prefix)-1,
          'goal':goal,'second_goal':later,'shortest_initial':len(route)-1,
          'graph_nodes':len(allowed),'junctions':sum(sum((x+dx,y+dy) in allowed for dx,dy in DIRECTIONS)>=3 for x,y in allowed),
          'initial_retreat':route[1]==w.history[-1]}
    return w,info


def make_world(seed,family,case,personality):
    # Rejection sample ONLY for challenge construction, never at inference.
    for retry in range(40):
        used_seed=seed+retry*7919
        w,info=_make_world_once(used_seed,family,case,personality)
        if info['initial_retreat']==(case=='detour'):
            info['seed']=used_seed;info['retries']=retry
            return w,info
    raise AssertionError('could not generate required challenge geometry')


def obs(w,menu):
    eligible=fresh(w,menu)
    return np.asarray([min(len(w.history),50)/50.,min(w.distance(),28)/28.,
      min(w.stuck,10)/10.,float(bool(eligible)),min(len(eligible),4)/4.,
      float(menu[8].legal),float(menu[8].features[22]),
      float(sum(c.legal for c in menu[:8]))/8.,float(w.goal_change)],np.float32)


def executed_suffix(w,first,steps=8):
    """Fixed policy executes each suffix legally; offline targets only."""
    v=copy.deepcopy(w);start=v.distance()
    for t in range(steps):
        if v.distance()==0 or v.stopped:break
        menu=v.candidates(.35)
        if t==0:j=first
        else:
            inds=[j for j,c in enumerate(menu[:8]) if c.legal and c.target not in v.history and c.target not in v.inked]
            if inds:j=max(inds,key=lambda k:(v.distance()-v.distance(menu[k].target),-k))
            elif menu[8].legal:j=8
            else:break
        if not menu[j].legal:break
        assert v.apply(menu[j]) and v.invalid_attempts==0
    return float(np.clip((start-v.distance())/steps,-1.,1.))+(1. if v.distance()==0 else 0.)


class RetreatHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(9,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def train(seed=340111,epochs=14):
    torch.manual_seed(seed);torch.set_num_threads(1)
    xs=[];ys=[];receipts=[]
    for i in range(20):
        for person in (-1,1):
            for case in CASES:
                s=seed+17011*i+109*person+int(case=='forward')*37
                w,info=make_world(s,'tree',case,person)
                for t in range(4):
                    menu=w.candidates(.35)
                    legal=[j for j,c in enumerate(menu[:8]) if c.legal and c.target not in w.history]
                    if not legal or not menu[8].legal:break
                    j=max(legal,key=lambda k:w.distance()-w.distance(menu[k].target))
                    back=executed_suffix(w,8);fwd=executed_suffix(w,j)
                    xs.append(obs(w,menu));ys.append(float(np.clip(back-fwd,-1.5,1.5)))
                    receipts.append({'graph':info['graph_sha'],'state':sha([w.pos,w.history,w.goal]),
                      'menu':sha([(c.action,c.target,c.legal) for c in menu]),
                      'back_return':back,'forward_return':fwd,'delta':ys[-1]})
                    assert w.apply(menu[j]) and w.invalid_attempts==0
    X=np.stack(xs);Y=np.asarray(ys,np.float32)
    x=torch.from_numpy(X);y=torch.from_numpy(Y)
    head=RetreatHead();opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    rng=np.random.default_rng(seed);updates=0;losses=[]
    for _ in range(epochs):
        total=0.;perm=rng.permutation(len(Y))
        for start in range(0,len(Y),32):
            k=perm[start:start+32]
            loss=nn.functional.smooth_l1_loss(head(x[k]),y[k]);opt.zero_grad(set_to_none=True)
            loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            total+=float(loss.detach());updates+=1
        losses.append(round(total,6))
    head.eval()
    with torch.no_grad():mae=float(torch.abs(head(x)-y).mean())
    return head,{'examples':len(Y),'updates':updates,'train_mae':mae,
                 'delta_positive':int((Y>0).sum()),'delta_negative':int((Y<0).sum()),
                 'data_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),'losses':losses},receipts


class ReArmedWatch(SearchPolicy):
    """C32-WATCH reset on host-issued new goal, with optional neural bias."""
    def __init__(self,option,head=None):
        super().__init__(option,None,'WATCH');self.head=head
        self.rearms=0;self.head_decisions=0;self.modified=0
    def modify(self,base,w,advice,arm):
        if self.phase!='idle' and self.goal is not None and advice.active(w)==1 and tuple(w.goal)!=self.goal:
            self.phase='idle';self.rearms+=1
        menu,logits,bad,target=super().modify(base,w,advice,arm)
        if self.head is not None and self.phase in ('watch','probe','retrace') and advice.directive==1 and menu[8].legal:
            with torch.no_grad():value=float(self.head(torch.from_numpy(obs(w,menu))))
            self.head_decisions+=1
            # Bounded cue-gated relative-return bias; never change legal actions.
            change=float(np.clip(value*2.,-1.5,1.5));logits[8]+=change
            self.modified+=int(abs(change)>1e-6)
        return menu,logits,bad,target


def play(base,option,head,w,info,seed,arm,directive,strength,model,switch=False,horizon=115,trace=False):
    cue=HostAdvice(w)
    if directive!='none':getattr(cue,'issue_'+directive)(w,strength=strength,ttl=9)
    policy=(ReArmedWatch(option,head if model=='C34-HEAD' else None)
            if model.startswith('C34-') else SearchPolicy(option,None,'WATCH') if model=='C32-WATCH' else None)
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+
                              {'none':0,'backtrack':411,'continue':977}[directive])
    baseline=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    protected=sha(sorted(w.protected));actions=[];path=[];illegal=0;reached_first=0;reached_second=0
    changes=0
    for t in range(horizon):
        if switch and t==50:
            reached_first=int(reached_first or w.distance()==0)
            w.goal=info['second_goal'];w.goal_change=True;w.goals_seen+=1;w.stuck=2
            if directive!='none':getattr(cue,'issue_'+directive)(w,strength=strength,ttl=9)
            changes+=1
        if w.stopped:break
        if w.distance()==0:
            if t<50 or not switch:reached_first=1
            else:reached_second=1
            if not switch or t>=50:break
            continue
        if policy is None:menu,logits,bad,_=scores(base,w,cue,arm)
        else:menu,logits,bad,_=policy.modify(base,w,cue,arm)
        illegal+=bad
        chosen=int(rng.choice(len(menu),p=probs(logits)));c=menu[chosen]
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        if policy is not None:policy.after(c)
        cue.advance(w);actions.append((c.action,c.target))
        if trace:path.append({'t':t,'pos':w.pos,'goal':w.goal,'action':c.action[0],
                              'distance':w.distance(),'phase':policy.phase if policy is not None else 'baseline'})
        if t<50 and w.distance()==0:reached_first=1
        if switch and t>=50 and w.distance()==0:reached_second=1
    if not switch:reached_first=int(reached_first or w.distance()==0)
    assert protected==sha(sorted(w.protected))
    return {'first_success':reached_first,'second_success':reached_second,'total_success':reached_first+reached_second if switch else reached_first,
            'switches':changes,'steps':len(actions),'actions_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
            'prefix_sha':baseline,'host_rejections':w.invalid_attempts,'raw_illegal_top1':illegal,
            'head_decisions':policy.head_decisions if isinstance(policy,ReArmedWatch) else 0,
            'head_modified':policy.modified if isinstance(policy,ReArmedWatch) else 0,
            'rearms':policy.rearms if isinstance(policy,ReArmedWatch) else 0,
            'path':path if trace else None}


def evaluate(base,option,head,seed,n=2,horizon=115):
    rows=[];graphs={};models=('C18-A0','C32-WATCH','C34-RESET','C34-HEAD')
    for fam in FAMILIES:
      for case in CASES:
       for switch in (False,True):
        for i in range(n):
         for person in (-1,1):
          for arm in (.35,1.):
           for directive,strength in (('none',0.),('backtrack',.35),('backtrack',1.),('continue',.35),('continue',1.)):
            s=(seed+10007*i+379*person+int(arm*100)*199+sum(map(ord,case))*411+
              int(fam=='braid')*17291+int(switch)*12373)
            w0,info=make_world(s,fam,case,person);graphs[info['graph_sha']]=info
            for model in models:
                w=copy.deepcopy(w0)
                record=play(base,option,head,w,info,s,arm,directive,strength,model,
                   switch=switch,horizon=horizon,trace=i==0 and person==1 and arm==1. and directive=='backtrack' and strength==1.)
                rows.append({'model':model,'family':fam,'case':case,'switch':switch,'i':i,'personality':person,
                             'arm':arm,'directive':directive,'strength':strength,'seed':s,
                             'graph_sha':info['graph_sha'],**record})
    return rows,graphs


def summarize(rows):
    result={};names=sorted({r['model'] for r in rows})
    k=lambda r:(r['family'],r['case'],r['switch'],r['i'],r['personality'],r['arm'],r['directive'],r['strength'])
    baseline={k(r):r for r in rows if r['model']=='C18-A0'}
    g=lambda r:(r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    for m in names:
        rr=[r for r in rows if r['model']==m];guided=[r for r in rr if r['directive']!='none']
        groups={str(key):sum(r['total_success'] for r in guided if g(r)==key) for key in sorted({g(r) for r in guided})}
        unguided=[r for r in rr if r['directive']=='none']
        result[m]={'guided_total':sum(r['total_success'] for r in guided),
            'guided_episodes':len(guided),'first':sum(r['first_success'] for r in guided),
            'second':sum(r['second_success'] for r in guided),
            'tree':sum(r['total_success'] for r in guided if r['family']=='tree'),
            'braid':sum(r['total_success'] for r in guided if r['family']=='braid'),
            'groups':groups,
            'unguided_identical':sum(r['actions_sha']==baseline[k(r)]['actions_sha'] and
                                    r['ink_sha']==baseline[k(r)]['ink_sha'] for r in unguided),
            'unguided_n':len(unguided),'host_rejections':sum(r['host_rejections'] for r in rr),
            'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr),
            'head_decisions':sum(r['head_decisions'] for r in rr),
            'head_modified':sum(r['head_modified'] for r in rr),'rearms':sum(r['rearms'] for r in rr)}
    for m in names:
        result[m]['regressed_vs_A0']={key:v-result[m]['groups'][key]
            for key,v in result['C18-A0']['groups'].items() if result[m]['groups'][key]<v}
        result[m]['regressed_vs_C32']={key:v-result[m]['groups'][key]
            for key,v in result['C32-WATCH']['groups'].items() if result[m]['groups'][key]<v}
    return result


def run(root,out,seed=340701,n=2,horizon=115):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS])
    base=a0(prior);option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    head,fit,receipts=train()
    save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(out/'C34-RETREAT.safetensors'))
    result={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'checkpoint_sha256':hashlib.sha256((out/'C34-RETREAT.safetensors').read_bytes()).hexdigest(),
           'fit':fit,'cohorts':{},'graph_counts':{}}
    for label,s in [('A',seed),('B',seed+200003)]:
        rows,graphs=evaluate(base,option,head,s,n,horizon)
        result['cohorts'][label]=summarize(rows)
        result['graph_counts'][label]={'unique':len(graphs),'tree':sum(v['graph_nodes']==97 for v in graphs.values()),
                'braid':sum(v['graph_nodes']>97 for v in graphs.values()),
                'first_retreat_count':sum(v['initial_retreat'] for v in graphs.values()),
                'min_shortest':min(v['shortest_initial'] for v in graphs.values()),
                'max_shortest':max(v['shortest_initial'] for v in graphs.values())}
        (out/f'cohort_{label}.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
        (out/f'topologies_{label}.json').write_text(json.dumps(graphs,sort_keys=True,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'training_receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('out');ap.add_argument('--seed',type=int,default=340701)
    ap.add_argument('--n',type=int,default=2);ap.add_argument('--horizon',type=int,default=115)
    args=ap.parse_args();print(json.dumps(run(args.root,args.out,args.seed,args.n,args.horizon),sort_keys=True))

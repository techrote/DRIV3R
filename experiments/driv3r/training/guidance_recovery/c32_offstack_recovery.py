"""C32: history-guided recovery when the new goal is NOT on the executed stack.

Synthetic DRIV3R experiment; no production CyberSand contract changes.
Training targets are post-hoc host-executed short-horizon outcomes. At inference
only current legal menus, executed history and an explicitly issued cue are used.
The returned policy remains stochastic and host legality is rechecked per action.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from experiment import World,sha,DIRECTIONS
from c29_probe_recovery import PREREQS,ProbeHead,custom_rollout
from c28_temporal_option import OptionHead,scores,probs
from c31_observed_progress import C31Policy,ProgressHead,rollout as c31_rollout
from personality_retention_core import load_c12
from c19_return_anchored import a0
from signed_feedback import HostAdvice

SCHEMA='driv3r-c32-offstack-multi-turn/1'

def world(index,scenario,personality,seed):
    """Execute a long U trail first; expose only host-legal moves thereafter.

    'off-early' and 'off-mid' require retracing away from new goal before
    exploring a branch not recorded in history. 'shortcut' is a legitimate
    six-cell direct route. 'lure{2,3,4}' opens incomplete, locally goalward
    exits that must be abandoned after observed lack of legal progress.
    """
    if scenario not in ('off-early','off-mid','off-late','shortcut','lure2','lure3','lure4'):
        raise ValueError(scenario)
    trail=[(x,11) for x in range(2,12)]+[(11,y) for y in range(10,1,-1)]+[(x,2) for x in range(10,1,-1)]
    shift=int(index>=8)
    def tr(p):
        x,y=p
        if index%2:x=14-x
        if (index//2)%2:y=14-y
        if (index//4)%2:x,y=y,x
        return (x+shift,y)
    coords=set(trail)
    anchor=2 if scenario in ('off-early','shortcut','lure2','lure3','lure4') else (6 if scenario=='off-mid' else 9)
    branch=[(anchor,y) for y in (10,9,8)]
    coords.update(branch)
    if scenario=='shortcut':coords.update((2,y) for y in range(3,9))
    if scenario.startswith('lure'):coords.update((2,y) for y in range(3,3+int(scenario[-1])))
    target=tr((anchor,8))
    allow={tr(p) for p in coords}
    w=World(seed,'islands',personality)
    w.protected={(x,y) for x in range(15) for y in range(15) if (x,y) not in allow}
    w.pos=tr(trail[0]);w.goal=tr(trail[-1]);w.inked={w.pos}
    for nxt in trail[1:]:
        menu=w.candidates(0.);valid=[c for c in menu[:4] if c.legal and c.target==tr(nxt)]
        assert len(valid)==1,(index,scenario,w.pos,nxt)
        assert w.apply(valid[0])
    assert len(w.history)==27
    w.goal=target;w.goal_change=True;w.goals_seen=2;w.stuck=2
    assert target not in w.history and target!=w.pos and not w.invalid_attempts
    return w

def bfs_length(w):
    seen={w.pos};queue=collections.deque([(w.pos,0)])
    while queue:
        p,d=queue.popleft()
        if p==w.goal:return d
        for dx,dy in DIRECTIONS:
            q=p[0]+dx,p[1]+dy
            if q not in seen and w.allowed(q):seen.add(q);queue.append((q,d+1))
    return None

def fresh(w,menu):
    return [j for j,c in enumerate(menu[:8]) if c.legal and
            c.action[0] in ('INK','LIFT') and c.target not in w.history and
            c.target not in w.inked and w.distance(c.target)<w.distance()]

def sample_features(w,menu,steps):
    f=fresh(w,menu)
    return np.asarray([float(bool(f)),min(len(f),3)/3.,
        min(len(w.history),40)/40.,min(w.stuck,8)/8.,
        min(steps,8)/8.,
        max((float(menu[j].features[22]) for j in f),default=1.)],np.float32)

class BranchHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(6,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def executed_suffix_probe(w,n=8):
    """Offline label: legal and actually executed goalward suffix, no teleport.

    This is a training-only lookahead, never called by the deployed policy.
    """
    m=w.candidates(.35);f=fresh(w,m)
    if not f:return 0
    ww=copy.deepcopy(w)
    for _ in range(n):
        mm=ww.candidates(.35);ff=fresh(ww,mm)
        if not ff:break
        # Tie-breaking on host-current information only, fixed before outcome.
        c=mm[max(ff,key=lambda j:float(mm[j].features[22]))]
        assert ww.apply(c) and ww.invalid_attempts==0
        if ww.distance()==0:return 1
    return 0

def train(seed=320011,epochs=20):
    torch.manual_seed(seed);torch.set_num_threads(1)
    X=[];Y=[];rows=[]
    for i in range(8):
      for person in (-1,1):
       for scenario in ('off-early','off-mid','shortcut','lure2','lure3','lure4'):
        w=world(i,scenario,person,seed+i*97+person*11)
        for step in range(6):
            m=w.candidates(.35);f=fresh(w,m)
            X.append(sample_features(w,m,step));Y.append(executed_suffix_probe(w,8))
            rows.append({'i':i,'personality':person,'scenario':scenario,'step':step,
                 'menu_sha':sha([(c.action,c.target,c.legal) for c in m]),
                 'prefix_sha':sha([w.pos,w.history,sorted(w.protected)]),'label':Y[-1]})
            if not f:break
            c=m[f[0]];assert w.apply(c) and w.invalid_attempts==0
    x=torch.from_numpy(np.stack(X));y=torch.tensor(Y,dtype=torch.float32)
    net=BranchHead();opt=torch.optim.AdamW(net.parameters(),lr=.018,weight_decay=.01)
    rng=np.random.default_rng(seed);updates=0;losses=[]
    for epoch in range(epochs):
        total=0.;permutation=rng.permutation(len(x))
        for a in range(0,len(x),48):
            inds=permutation[a:a+48]
            loss=nn.functional.binary_cross_entropy_with_logits(net(x[inds]),y[inds]);
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step();updates+=1;total+=float(loss.detach())
        losses.append(round(total,6))
    net.eval()
    with torch.no_grad():acc=float(((net(x)>0)==(y>0)).float().mean())
    return net,{'updates':updates,'examples':len(y),'positives':int(y.sum()),
        'train_accuracy':acc,'losses':losses,
        'features_sha256':hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()},rows

class SearchPolicy:
    def __init__(self,option,branch,variant='RULE'):
        self.option=option;self.branch=branch;self.variant=variant
        self.phase='idle';self.goal=None;self.started_at=None
        self.probe_steps=0;self.probes=0;self.retreats=0;self.releases=0
    def _release(self):
        self.phase='released';self.releases+=1
    def modify(self,base,w,advice,arm):
        menu,logits,bad,_=scores(base,w,advice,arm)
        if self.phase in ('watch','probe','retrace') and (tuple(w.goal)!=self.goal or advice.directive==0):self._release()
        if self.phase=='idle' and advice.active(w)==1 and len(w.history)>=14 and w.goal not in w.history:
            self.goal=tuple(w.goal);self.started_at=w.steps
            if self.variant=='SAFE' and fresh(w,menu):self._release()
            elif self.variant=='WATCH' and fresh(w,menu):self.phase='watch'
            else:self.phase='probe' if fresh(w,menu) else 'retrace'
        if self.phase in ('watch','probe','retrace') and w.steps-self.started_at>=105:self._release()
        if self.phase=='watch' and w.steps>self.started_at and not fresh(w,menu):
            self.phase='retrace'
        if self.phase=='retrace' and fresh(w,menu):self.phase='probe';self.probe_steps=0
        if self.phase=='probe' and not fresh(w,menu):self.phase='retrace'
        target=None
        if self.phase=='probe':
            f=fresh(w,menu);j=max(f,key=lambda k:logits[k])
            if self.variant=='LEARNED':
                with torch.no_grad():strength=float(torch.sigmoid(self.branch(torch.from_numpy(sample_features(w,menu,self.probe_steps)))).item())
                bonus=4.+4.*strength
            else:bonus=6.
            logits[j]+=bonus;target=j;self.probes+=1
            # At least 12% of probability stays on other legal actions.
            rest=np.delete(logits,j);rest=rest[rest>-1e7]
            cap=rest.max()+.8*(math.log(.88/.12)+math.log(np.exp((rest-rest.max())/.8).sum()))
            logits[j]=min(logits[j],cap)
        elif self.phase=='retrace' and menu[8].legal:
            with torch.no_grad():
                z=torch.tensor([len(w.history)/40.,1.,min((w.steps-self.started_at)/40.,2.),1.,advice.strength],dtype=torch.float32)
                bonus=float(self.option(z))
            logits[8]+=bonus;target=8;self.retreats+=1
            rest=np.delete(logits,8);rest=rest[rest>-1e7]
            cap=rest.max()+.8*(math.log(.985/.015)+math.log(np.exp((rest-rest.max())/.8).sum()))
            logits[8]=min(logits[8],cap)
        return menu,logits,bad,target
    def after(self,selected):
        if self.phase=='probe' and selected.action[0] in ('INK','LIFT'):self.probe_steps+=1

def run_episode(base,option,branch,w,seed,arm,directive,strength,variant='RULE',horizon=125,trace=False):
    cue=HostAdvice(w)
    if directive!='none':getattr(cue,'issue_'+directive)(w,strength=strength,ttl=9)
    pol=SearchPolicy(option,branch,variant)
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+
       {'none':0,'backtrack':411,'continue':977}[directive])
    prefix=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    material=sha(sorted(w.protected));actions=[];path=[];bad=0;prob_max=0
    for t in range(horizon):
        if w.stopped or w.distance()==0:break
        menu,logits,raw,target=pol.modify(base,w,cue,arm);bad+=raw
        p=probs(logits);index=int(rng.choice(len(menu),p=p));selected=menu[index]
        assert selected.legal and w.apply(selected) and w.invalid_attempts==0
        pol.after(selected);cue.advance(w)
        prob_max=max(prob_max,float(p[target]) if target is not None else 0.)
        actions.append((selected.action,selected.target))
        if trace:path.append({'pos':w.pos,'action':selected.action[0],'phase':pol.phase,'distance':w.distance()})
    assert sha(sorted(w.protected))==material
    return {'success':int(w.distance()==0),'steps':len(actions),'prefix_sha':prefix,
        'actions_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
        'host_rejected':w.invalid_attempts,'raw_illegal_top1':bad,
        'probes':pol.probes,'retreats':pol.retreats,'max_modified_probability':prob_max,'path':path if trace else None}

def evaluate(prior,option,branch,seed,n=2,horizon=125,models=('C18-A0','C31-DIRECT','C32-RULE','C32-LEARNED','C32-SAFE','C32-WATCH')):
    base=a0(prior);rows=[];dummy=ProgressHead()
    for i in range(n):
      for person in (-1,1):
       for scenario in ('off-early','off-mid','off-late','shortcut','lure2','lure3','lure4'):
        for arm in (.35,1.):
         for directive in ('none','backtrack','continue'):
          for strength in ((0.,) if directive=='none' else (.35,1.)):
           ss=seed+i*10009+person*313+int(arm*100)*127+sum(ord(ch) for ch in scenario)*811
           kw={'i':i,'personality':person,'scenario':scenario,'arm':arm,'directive':directive,'strength':strength,'seed':ss}
           for model in models:
            w=world(i+8,scenario,person,ss)
            trace=i==0 and person==1 and strength==1 and arm==1
            if model=='C18-A0':
                h=ProbeHead();res=custom_rollout(base,option,h,i,scenario,person,arm,directive,strength,ss,horizon,trace,world_override=w,baseline_variant='C18-A0')
                # standardize hashes
                res['actions_sha']=res['action_sha256'];res['ink_sha']=res['ink_sha256']
            elif model=='C31-DIRECT':
                res=c31_rollout(base,option,dummy,w,ss,arm,directive,strength,'DIRECT',horizon,trace)
                res['actions_sha']=res['action_sha256'];res['ink_sha']=res['ink_sha256']
            else:res=run_episode(base,option,branch,w,ss,arm,directive,strength,model.split('-')[1],horizon,trace)
            rows.append(dict(kw,model=model,**res))
    return rows

def summarize(rows):
    cohorts={};d={};baseline={}
    k=lambda r:(r['i'],r['personality'],r['scenario'],r['arm'],r['directive'],r['strength'])
    for r in rows:
        if r['model']=='C18-A0':baseline[k(r)]=r
    for name in sorted({r['model'] for r in rows}):
        rr=[r for r in rows if r['model']==name];guided=[r for r in rr if r['directive']!='none']
        no=[r for r in rr if r['directive']=='none']
        groups={f'{sc}:{dr}:{st}':sum(r['success'] for r in guided if (r['scenario'],r['directive'],r['strength'])==(sc,dr,st)) for sc in sorted({r['scenario'] for r in rr}) for dr in ('backtrack','continue') for st in (.35,1.)}
        d[name]={'guided':sum(r['success'] for r in guided),'guided_total':len(guided),
             'by_scenario':{sc:sum(r['success'] for r in guided if r['scenario']==sc) for sc in sorted({r['scenario'] for r in rr})},
             'groups':groups,'no_cue_identical':sum(r['actions_sha']==baseline[k(r)]['actions_sha'] and r['ink_sha']==baseline[k(r)]['ink_sha'] for r in no),
             'no_cue_total':len(no),'host_rejected':sum(r['host_rejected'] for r in rr),
             'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr)}
    for name in d:
        d[name]['regressed_groups']={k:v-d[name]['groups'][k] for k,v in d['C18-A0']['groups'].items() if d[name]['groups'][k]<v}
    return d

def run(root,out,seed=322221,n=2,models=None):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    branch,fit,receipts=train()
    save_file({k:v.detach().cpu().contiguous() for k,v in branch.state_dict().items()},str(out/'C32-BRANCH.safetensors'))
    metrics={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256((out/'C32-BRANCH.safetensors').read_bytes()).hexdigest(),
        'fit':fit,'cohorts':{},'oracle_shortest_paths':{s:bfs_length(world(8,s,1,seed)) for s in ('off-early','off-mid','off-late','shortcut','lure2','lure3','lure4')}}
    for c,cohort_seed in (('A',seed),('B',seed+200003)):
        rows=evaluate(prior,option,branch,cohort_seed,n=n,models=models) if models is not None else evaluate(prior,option,branch,cohort_seed,n=n)
        metrics['cohorts'][c]=summarize(rows)
        (out/f'cohort_{c}.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (out/'train_receipts.json').write_text(json.dumps(receipts,indent=2,sort_keys=True)+'\n')
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--n',type=int,default=2);p.add_argument('--seed',type=int,default=322221);p.add_argument('--models',nargs='+',default=None);a=p.parse_args()
    print(json.dumps(run(a.root,a.out,a.seed,a.n,a.models),sort_keys=True))

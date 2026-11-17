"""C27: held-out long-horizon trap recovery with separately trained mistake detector and cue acceptance.

Synthetic external host only; not CyberSand's native host ABI. Legal actions, painting,
protected materials and execution remain owned by the immutable World fixture.
No game repository or shared fixture is modified.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file,load_file
import c26_stuck_interventions as previous
from experiment import World,N,ARMS,sha
from c19_return_anchored import a0
from personality_retention_core import load_c12
from attested_stack_cue import KEYS

SCHEMA='driv3r-guidance-c27-deep-trap-detector/1'

def snake_path(rows):
    # Consecutive horizontal corridors are separated by protected rock; only
    # alternating end connectors allow passage. Nearness is misleading.
    p=[]
    for k,y in enumerate(rows):
        xs=range(2,13) if k%2==0 else range(12,1,-1)
        if k:
            x=12 if k%2 else 2
            for yy in range(rows[k-1]+1,y):p.append((x,yy))
        p.extend((x,y) for x in xs)
    assert len(p)==len(set(p)) and all(abs(x-x2)+abs(y-y2)==1 for (x,y),(x2,y2) in zip(p,p[1:]))
    return p

def long_world(i,outcome,person,seed,train):
    # Training uses 3-barrier family; withheld uses 4-barrier geometry. 15x15.
    # Reverse: target lies on executed-history stack but Euclidean descent
    # is blocked by intervening protected materials.
    rows=(3,9) if train else (2,6,10)
    p=snake_path(rows)
    rot=(i%4 if train else (i+1)%4)
    mirror=bool((i//4)%2)
    def tr(q):
        x,y=q
        for _ in range(rot):x,y=14-y,x
        if mirror:x=14-x
        return x,y
    p=[tr(q) for q in p]
    allowed=set(p)
    # A few dead-end branch affordances; no cross-wall bypass is allowed.
    # Branches chosen deterministically, rejecting anything adjacent to a
    # distant path position to avoid unintentional shortcuts.
    for j,pt in enumerate(p):
        if j<4 or j>len(p)-4 or j%9!=5:continue
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            q=(pt[0]+dx,pt[1]+dy)
            if not (1<=q[0]<=13 and 1<=q[1]<=13) or q in allowed:continue
            adj=[k for k,z in enumerate(p) if abs(z[0]-q[0])+abs(z[1]-q[1])==1]
            if adj==[j]:allowed.add(q);break
    w=World(seed,'islands',person)
    w.protected={(x,y) for x in range(N) for y in range(N)}-allowed
    w.history=list(p[:-1]);w.pos=p[-1]
    if outcome=='reverse':w.goal=p[0] if train else p[10]
    elif outcome=='ahead':
        # A local legal endpoint near the start, discouraging unnecessary reversal
        w.goal=p[-4]
    elif outcome=='shortcut':w.goal=p[-6]
    else:w.goal=p[0] if (i%2==0) else p[-6]
    w.inked=set(p);w.last_dir=(p[-1][0]-p[-2][0],p[-1][1]-p[-2][1]);w.steps=8
    w.stuck=2;w.goal_change=True;w.goals_seen=2
    assert len(p)>=20 and all(w.allowed(z) for z in p) and all(not w.allowed(z) for z in w.protected)
    return w

def oracle_geometry_audit(train):
    """Ground truth used ONLY for an offline solvability audit, never inference."""
    w=long_world(0,'reverse',1,270001,train)
    start=w.pos;goal=w.goal
    frontier=collections.deque([(start,0,None)])
    seen={start};length=None;first=None
    while frontier:
        u,d,initial=frontier.popleft()
        if u==goal:length=d;first=initial;break
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            v=(u[0]+dx,u[1]+dy)
            if v in seen or not w.allowed(v):continue
            seen.add(v);frontier.append((v,d+1,v if initial is None else initial))
    assert length is not None
    direct=w.distance()
    return {'shortest_legal_moves':length,'manhattan':direct,'detour_ratio':round(length/direct,3),
      'first_shortest_step_increases_manhattan':w.distance(first)>direct,
      'protected_cells':len(w.protected),'history_depth':len(w.history),
      'start':start,'goal':goal}

def oracle_retrace(train):
    """Offline host-legal feasibility witness, not a trained/control policy."""
    w=long_world(0,'reverse',1,270001,train)
    steps=0
    while w.distance() and steps<100:
        menu=w.candidates(0.)
        assert menu[8].legal and w.apply(menu[8]) and not w.invalid_attempts
        steps+=1
    return {'success':w.distance()==0,'steps':steps,'host_rejected':w.invalid_attempts}

class DetectorHead(nn.Module):
    """Classify induced nonprogress from past executed events, not the host cue."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(4,8),nn.Tanh(),nn.Linear(8,1))
    def forward(self,x):return self.net(x).squeeze(-1)

class DualHead(previous.InterventionHead):
    def __init__(self,prior,mode='joint',scale=.85):
        super().__init__(prior,trigger=False,scale=scale)
        self.detector=DetectorHead()
        self.mode=mode
    def forward(self,x):
        old=self.base(x[...,:32]);out=old.clone();q=x[...,8,:]
        f=torch.cat((q[...,26:28],q[...,29:31],q[...,32:40]),-1)
        delta=self.scale*torch.tanh(self.head(f).squeeze(-1))
        # Two independent judgments: observed mistake likelihood vs whether
        # active cue warrants a particular legal action. The detector cannot
        # change legality, material masks, or advice provenance.
        if self.mode=='detector':
            gate=torch.sigmoid(self.detector(q[...,36:40]))
        else:gate=torch.ones_like(delta)
        active=q[...,2]*q[...,4]*(q[...,26].abs()>.5).to(q.dtype)*q[...,27]*q[...,29]
        out[...,8]=old[...,8]+active*gate*delta
        return out

def evaluate_targeted(models,prior,n,seed,horizon):
    """Fully paired held-out episodes; deep reverse and false cues are primary."""
    rows=[];base=a0(prior)
    for i in range(n):
        for outcome in ('reverse','ahead','shortcut','bridge'):
            for person in (-1,1):
                for arm in (.35,1.):
                    for mode in ('badturn',):
                        ss=seed+i*10117+('reverse','ahead','shortcut','bridge').index(outcome)*101111+person*131+int(arm*100)*17+227
                        for directive in ('none','backtrack','continue'):
                            for strength in ((0.,) if directive=='none' else (.35,1.)):
                                for name,model in models.items():
                                    rec=previous.rollout(model,base,i,outcome,person,arm,directive,strength,mode,ss,horizon,False)
                                    rec.update({'model':name,'index':i,'outcome':outcome,'personality':person,
                                        'arm':float(arm),'mode':mode,'directive':directive,'strength':float(strength),'seed':ss})
                                    rows.append(rec)
    return rows

def experiment(archive,out,train_n=8,eval_n=5,epochs=8,horizon=90,seed=270500):
    torch.set_num_threads(1)
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[archive/'prerequisites'/f'guidance_recovery_{s}.safetensors' for s in ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer','persistent_feedback','signed_feedback','selective_continue')]
    assert all(p.exists() for p in paths)
    prior=load_c12(paths);base=a0(prior)
    original=previous.seed_world
    previous.seed_world=long_world
    try:
        X,A,G,receipts,triggers=previous.collect(prior,n=train_n,seed=seed,horizon=30)
        assert len(X)>0
        # Explicit detector supervision: induced vs neutral legal prefixes.
        y=np.asarray([int(r['mode']=='badturn') for r in receipts],np.float32)
        x=torch.from_numpy(X[:,8,36:40]);target=torch.from_numpy(y)
        torch.manual_seed(270016);det=DetectorHead();opt=torch.optim.AdamW(det.parameters(),lr=.008)
        detect_losses=[];det_updates=0
        for e in range(max(1,epochs*2)):
            ix=np.random.default_rng(270016+e).permutation(len(y))
            for k in range(0,len(ix),64):
                b=ix[k:k+64];z=det(x[b]);loss=nn.functional.binary_cross_entropy_with_logits(z,target[b]);loss.backward();opt.step();opt.zero_grad();det_updates+=1
            detect_losses.append(float(nn.functional.binary_cross_entropy_with_logits(det(x),target).detach()))
        model_joint=DualHead(prior,mode='joint')
        model_detect=DualHead(prior,mode='detector')
        model_detect.detector.load_state_dict(det.state_dict())
        for p in model_detect.detector.parameters():p.requires_grad_(False)
        models={};updates={};losses={}
        xx=torch.from_numpy(X);aa=torch.from_numpy(A).clamp(-3,3)
        with torch.no_grad():p0=previous.softmax(previous.Control(prior,True)(xx),xx)[:,8]
        group_ct=collections.Counter(G);wt=torch.tensor([1/group_ct[g] for g in G],dtype=torch.float32);wt/=wt.mean()
        for name,model in [('C27-JOINT',model_joint),('C27-DETECT',model_detect)]:
            torch.manual_seed(270027);rng=np.random.default_rng(270027)
            opt=torch.optim.AdamW(model.head.parameters(),lr=.0012,weight_decay=.01)
            hist=[];num=0
            for ep in range(epochs):
                order=rng.permutation(len(X));tot=0.
                for start in range(0,len(X),64):
                    ix=order[start:start+64]
                    p=previous.softmax(model(xx[ix]),xx[ix])[:,8]
                    loss=-(wt[ix]*aa[ix]*(p-p0[ix])).mean()+2.5*(p-p0[ix]).square().mean()
                    loss+=.002*sum(z.square().mean() for z in model.head.parameters())
                    loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.)
                    opt.step();opt.zero_grad();num+=1;tot+=float(loss.detach())*len(ix)
                hist.append(tot/len(X))
            models[name]=model.eval();updates[name]=num;losses[name]=hist
        c26=previous.InterventionHead(prior,True)
        c26.head.load_state_dict(load_file(str(archive/'full_a/C26-TRIGGER.safetensors')))
        controls={'C12':previous.Control(prior,False),'C18-A0':previous.Control(prior,True),'C26-TRIGGER':c26.eval()}
        rows=evaluate_targeted({**controls,**models},prior,n=eval_n,seed=seed+800000,horizon=horizon)
        # A second OOD seed cohort for condition-level gate, not a training split.
        rows2=evaluate_targeted({**controls,**models},prior,n=eval_n,seed=seed+900000,horizon=horizon)
        digest=hashlib.sha256(X.tobytes()+A.tobytes()+json.dumps(G).encode()).hexdigest()
        source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        trained={}
        for name,m in models.items():
            checkpoint=out/f'{name}.safetensors'
            # Store BOTH detector and acceptance heads (training is reconstructible).
            weights={f'head.{k}':v.detach().cpu().contiguous() for k,v in m.head.state_dict().items()}
            weights.update({f'detector.{k}':v.detach().cpu().contiguous() for k,v in m.detector.state_dict().items()})
            save_file(weights,str(checkpoint),metadata={'provenance':json.dumps({'schema':SCHEMA,'source_sha256':source_sha,'training_sha256':digest},sort_keys=True,separators=(',',':'))})
            trained[name]={'updates':updates[name],'loss_first':losses[name][0],'loss_last':losses[name][-1],'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        def summarize(rows):
            d={}
            for name in controls|models:
                item=[r for r in rows if r['model']==name]
                d[name]={'guided_goals':sum(r['success'] for r in item if r['directive']!='none'),
                  'guided_n':sum(r['directive']!='none' for r in item),
                  'deep_reverse_goals':sum(r['success'] for r in item if r['outcome']=='reverse' and r['directive']!='none'),
                  'deep_reverse_n':sum(r['outcome']=='reverse' and r['directive']!='none' for r in item),
                  'premature_stop':sum(r['steps_after_cue']<horizon and not r['success'] for r in item),'deep_reverse_median_first_net':float(np.median([r['first_below_intervention'] for r in item if r['outcome']=='reverse' and r['first_below_intervention'] is not None])) if any(r['outcome']=='reverse' and r['first_below_intervention'] is not None for r in item) else None,
                  'protected_errors':sum(r['host_rejected'] for r in item),'illegal_raw_top1':sum(r['illegal_raw_top1'] for r in item),
                  'preferred_ink':sum(r['personality_ink'] for r in item),'opportunities':sum(r['personality_opportunities'] for r in item)}
            return d
        gates={}
        for name in models:
            regress=[];cueoff=0
            for cohort in (rows,rows2):
                by={(r['model'],r['index'],r['outcome'],r['personality'],r['arm'],r['mode'],r['directive'],r['strength']):r for r in cohort}
                for r in cohort:
                    if r['model']!=name:continue
                    key=(r['index'],r['outcome'],r['personality'],r['arm'],r['mode'],r['directive'],r['strength'])
                    if r['directive']=='none':
                        c=by[('C12',*key)]
                        cueoff+=int(r['actions_sha256']==c['actions_sha256'] and r['ink_sha256']==c['ink_sha256'])
                groups=previous.group_summary(cohort)
                regress.extend([k for k,v in groups.items() if '/none/' not in k and name in v and v[name]['goals']<v['C18-A0']['goals']])
            n_none=sum(r['directive']=='none' and r['model']==name for c in (rows,rows2) for r in c)
            gates[name]={'strict_noregression':not regress,'regressed_condition_groups':regress,'cue_off_match':cueoff,'cue_off_n':n_none}
        result={'schema':SCHEMA,'source_sha256':source_sha,'data_sha256':digest,
          'train_menus':len(X),'counterfactuals':len(X)*2,'detector_updates':det_updates,'detector_loss_first':detect_losses[0],
          'detector_loss_last':detect_losses[-1],'detector_badturn_rate':float(y.mean()),'training_triggered':triggers,
          'trained':trained,'cohort_a':summarize(rows),'cohort_b':summarize(rows2),'qualification':gates,
          'episodes':len(rows)+len(rows2),'topology_train':oracle_geometry_audit(True),'topology_heldout':oracle_geometry_audit(False),'oracle_train':oracle_retrace(True),'oracle_heldout':oracle_retrace(False),'prerequisites_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          'limitations':['Synthetic protected-cell grid; not native CyberSand ABI','Preloaded historical trail is not proof of host-executed history','Detector labels are experimental intervention modes, not actual future mistakes','Geometry favors mandatory retracing; shortcut adversaries are simplified']}
        for filename,data in [('training_receipts.json',receipts),('heldout_a.json',rows),('heldout_b.json',rows2),('metrics.json',result)]:
            (out/filename).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
        return result
    finally: previous.seed_world=original

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out');p.add_argument('--train-n',type=int,default=8);p.add_argument('--eval-n',type=int,default=5);p.add_argument('--epochs',type=int,default=8);p.add_argument('--horizon',type=int,default=90);p.add_argument('--seed',type=int,default=270500)
    a=p.parse_args();result=experiment(a.archive,a.out,a.train_n,a.eval_n,a.epochs,a.horizon,a.seed)
    print(json.dumps({k:result[k] for k in ('train_menus','detector_updates','trained','cohort_a','cohort_b','qualification','episodes')},sort_keys=True))
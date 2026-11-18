"""C31: stochastic, host-executed goalward probing over previously visited cells.

Synthetic DRIV3R research only. Training targets are current legal-menu actions,
never future topology or shortest-path oracle outputs. Independent withheld
U-corridor families explicitly distinguish long rollback from a real tunnel.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import World, sha
from c29_probe_recovery import (PREREQS, ProbeHead, Recoverer, custom_rollout,
                                fresh_indices, obs_probe)
from c28_temporal_option import (OptionHead, TemporalOption, history_evidence,
                                 scores, probs, executed_world)
from personality_retention_core import load_c12
from c19_return_anchored import a0
from signed_feedback import HostAdvice

SCHEMA = 'driv3r-c31-visited-progress/1'


def u_world(index, kind, personality, seed, depth=3):
    """A truly host-executed 27-step U prefix, not a preloaded history.

    Legal local observations are identical through the first depth-1 moves
    for a real 9-step tunnel and a length-depth decoy. The goal is on
    the previously executed stack. The oracle is only for offline checks.
    """
    if kind not in ('decoy','tunnel'): raise ValueError(kind)
    if not 1 <= depth <= 6: raise ValueError(depth)
    trail = ([(2,y) for y in (11,)] + [(x,11) for x in range(3,12)]
            + [(11,y) for y in range(10,1,-1)]
            + [(x,2) for x in range(10,1,-1)])
    def transform(p):
        # Evaluation coordinates are shifted, not present in training.
        x,y=p
        if index>=8:x+=1
        if index%2: x=14-x
        if (index//2)%2: y=14-y
        if (index//4)%2: x,y=y,x
        return (x,y)
    trail=[transform(p) for p in trail]
    spur=[transform((2,y)) for y in range(3,12 if kind=='tunnel' else 3+depth)]
    permitted=set(trail)|set(spur)
    w=World(seed,'islands',personality)
    w.protected={(x,y) for x in range(15) for y in range(15) if (x,y) not in permitted}
    w.pos=trail[0];w.goal=trail[-1];w.inked={w.pos}
    for nxt in trail[1:]:
        menu=w.candidates(0.)
        valid=[c for c in menu[:4] if c.legal and c.target==nxt]
        assert len(valid)==1,(index,kind,w.pos,nxt)
        assert w.apply(valid[0])
    assert len(w.history)==27 and w.pos==trail[-1]
    w.goal=trail[0];w.goal_change=True;w.goals_seen=2;w.stuck=2
    assert not w.invalid_attempts
    return w


def legal_progress(w,menu,include_visited=True):
    return [j for j,c in enumerate(menu[:8]) if c.legal
            and c.action[0] in ('INK','LIFT') and
            w.distance(c.target)<w.distance() and
            (include_visited or (c.target not in w.history and c.target not in w.inked))]


def move_features(w,c,probe_depth):
    """All inputs are current host observations or executed history."""
    return np.asarray([1.,float(c.target not in w.history and c.target not in w.inked),
                       float(c.target in w.history),float(c.target in w.inked),
                       float(c.features[22]), min(1.,probe_depth/8.)],np.float32)


class ProgressHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(6,8),nn.Tanh(),nn.Linear(8,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def training_menus(seed=310011, n=8):
    X=[];Y=[];receipts=[]
    for i in range(n):
      for kind in ('decoy','tunnel'):
       for depth in (1,3):
        w=u_world(i,kind,(-1 if i%2 else 1),seed+i*137+depth*47,depth)
        # Teacher executes legal goalward actions, without accessing full map.
        for t in range(9):
          menu=w.candidates(.35)
          for j,c in enumerate(menu[:8]):
            if c.legal:
              X.append(move_features(w,c,t))
              Y.append(int(j in legal_progress(w,menu)))
          receipts.append({'index':i,'kind':kind,'depth':depth,'step':t,
             'menu_hash':sha([(c.action,c.target,c.legal) for c in menu]),
             'history_hash':sha(w.history),'legal_progress':len(legal_progress(w,menu))})
          good=legal_progress(w,menu)
          if not good:break
          assert w.apply(menu[good[0]])
        assert w.invalid_attempts==0
    return np.asarray(X,np.float32), np.asarray(Y,np.float32),receipts


def train(seed=310011,epochs=20):
    torch.set_num_threads(1);torch.manual_seed(seed)
    X,Y,receipts=training_menus(seed)
    net=ProgressHead();optimizer=torch.optim.AdamW(net.parameters(),lr=.025,weight_decay=.01)
    x=torch.from_numpy(X);y=torch.from_numpy(Y);rng=np.random.default_rng(seed)
    losses=[];steps=0
    for epoch in range(epochs):
       idx=rng.permutation(len(Y));total=0.
       for start in range(0,len(Y),64):
          sl=idx[start:start+64]
          loss=nn.functional.binary_cross_entropy_with_logits(net(x[sl]),y[sl])
          optimizer.zero_grad(set_to_none=True);loss.backward()
          nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
          total+=float(loss.detach());steps+=1
       losses.append(round(total,7))
    net.eval()
    with torch.no_grad(): accuracy=float(((net(x)>0)==(y>0)).float().mean())
    return net,{'train_menus':len(receipts),'train_actions':len(Y),'updates':steps,
                'losses':losses,'in_sample_accuracy':accuracy,
                'feature_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest()},receipts


class C31Policy:
    def __init__(self,option_head,progress_head,variant='LEARNED',max_probe=10):
        self.phase='idle';self.goal=None;self.option=TemporalOption(option_head,True,veto=False)
        self.head=progress_head;self.variant=variant;self.max_probe=max_probe
        self.moves=0;self.gated=0;self.releases=0
    def release(self):
        self.phase='released';self.option.active=False;self.releases+=1
    def modify(self,base,w,advice,arm):
        menu,logits,bad,_=scores(base,w,advice,arm)
        if self.phase in ('probe','retrace') and (tuple(w.goal)!=self.goal or advice.directive==0):self.release()
        if self.phase=='idle' and advice.active(w)!=0:
            ev=history_evidence(w)
            if ev and ev['stack_moves']>=14 and ev['detour_ratio']>=2.:
                self.goal=tuple(w.goal)
                self.phase='probe' if legal_progress(w,menu,True) else 'retrace'
        bonus=0.
        if self.phase=='probe':
            prog=legal_progress(w,menu,self.variant!='FRESH_ONLY')
            if not prog and self.moves>0:self.phase='retrace'
            elif not prog:self.release()
            elif self.moves>=self.max_probe:self.release()
            else:
                with torch.no_grad():
                    values=[float(self.head(torch.from_numpy(move_features(w,menu[j],self.moves))).item()) for j in prog]
                j=(max(prog,key=lambda idx:logits[idx]) if self.variant=='DIRECT'
                   else prog[int(np.argmax(values))])
                if self.variant=='LEARNED':
                    # Bounded learned bonus; no action is forced.
                    bonus=8*float(torch.sigmoid(torch.tensor(values[int(np.argmax(values))])).item())
                else:bonus=6.
                logits[j]+=bonus;self.gated+=1
                finite=np.delete(logits,j);finite=finite[finite>-1e7]
                cap=.8*math.log(.88/.12)+.8*math.log(np.exp((finite-finite.max())/.8).sum())+float(finite.max())
                logits[j]=min(logits[j],cap)
        if self.phase=='retrace' and self.option.observe(w,advice,menu):
            bonus=self.option.bonus(w,advice);logits[8]+=bonus
            other=np.delete(logits,8);other=other[other>-1e7]
            cap=.8*math.log(.985/.015)+.8*math.log(np.exp((other-other.max())/.8).sum())+float(other.max())
            logits[8]=min(logits[8],cap)
        return menu,logits,bad,bonus
    def after(self,w,selected):
        if self.phase=='probe':
            if selected.action[0] in ('INK','LIFT'):self.moves+=1
            elif selected.action[0]=='STOP':self.release()
        if self.phase=='retrace':self.option.step(w,selected)


def rollout(base,option,head,w,seed,arm,directive,strength,variant='LEARNED',horizon=90,trace=False):
    a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    pol=C31Policy(option,head,variant)
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+
        {'none':0,'backtrack':411,'continue':977}[directive])
    start_hash=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    protected_hash=sha(sorted(w.protected));actions=[];path=[];illegal=0;maxback=0.
    for t in range(horizon):
        if w.distance()==0 or w.stopped:break
        menu,logits,bad,_=pol.modify(base,w,a,arm)
        illegal+=bad;distribution=probs(logits);j=int(rng.choice(len(menu),p=distribution));c=menu[j]
        assert c.legal
        if pol.phase=='retrace':maxback=max(maxback,float(distribution[8]))
        assert w.apply(c) and w.invalid_attempts==0
        pol.after(w,c);a.advance(w)
        actions.append((c.action,c.target))
        if trace:path.append({'step':t+1,'pos':w.pos,'action':c.action[0],
                               'phase':pol.phase,'distance':w.distance()})
    assert sha(sorted(w.protected))==protected_hash
    return {'success':int(w.distance()==0),'steps':len(actions),'host_rejected':w.invalid_attempts,
            'raw_illegal_top1':illegal,'action_sha256':sha(actions),'ink_sha256':sha(sorted(w.inked)),
            'prefix_sha256':start_hash,'probe_moves':pol.moves,'probe_decisions':pol.gated,
            'released':pol.releases,'max_back_probability':maxback,'path':path if trace else None}


def evaluate(prior,option,head,seed,n=2,horizon=90):
    base=a0(prior);rows=[]
    for i in range(n):
      for personality in (-1,1):
       for (kind,depth) in (('decoy',2),('decoy',3),('decoy',4),('tunnel',4)):
        for arm in (.35,1.):
         for directive in ('none','backtrack','continue'):
          for strength in ((0.,) if directive=='none' else (.35,1.)):
           ss=seed+10009*i+313*personality+int(arm*100)*127+depth*811+(17 if kind=='tunnel' else 0)
           params={'seed':ss,'i':i,'personality':personality,'outcome':f'{kind}{depth}',
                   'arm':arm,'directive':directive,'strength':strength}
           trace=i==0 and personality==1 and arm==1 and strength==1
           for model in ('C18-A0','C29-GUARD','C29-PROBE','C31-FRESH_ONLY','C31-DIRECT','C31-RULE','C31-LEARNED'):
             w=u_world(i+8,kind,personality,ss,depth)
             if model.startswith('C31-'):
                 variant='LEARNED' if model=='C31-LEARNED' else ('RULE' if model=='C31-RULE' else ('DIRECT' if model=='C31-DIRECT' else 'FRESH_ONLY'))
                 res=rollout(base,option,head,w,ss,arm,directive,strength,variant,horizon,trace)
             else:
                 h=ProbeHead();h.load_state_dict(load_file(str(_CHECKPOINT_)))
                 res=custom_rollout(base,option,h,i,kind,personality,arm,directive,strength,ss,
                                    horizon,trace,safe_start=(model=='C29-GUARD'),
                                    world_override=w,baseline_variant=('C18-A0' if model=='C18-A0' else None))
             rows.append(dict(res,model=model,**params))
    return rows

_CHECKPOINT_=None

def summary(rows):
    out={};key=lambda r:(r['i'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength'])
    groups=collections.defaultdict(dict)
    for r in rows:
        if r['directive']!='none':
            k=(r['outcome'],r['directive'],r['strength']);groups[(k,r['model'])]=groups.get((k,r['model']),0)+r['success']
    b={key(r):r for r in rows if r['model']=='C18-A0'}
    for name in sorted(set(r['model'] for r in rows)):
        allr=[r for r in rows if r['model']==name];guided=[r for r in allr if r['directive']!='none']
        regress=[{'group':str(k),'lost':v-groups.get((k,name),0)} for (k,m),v in groups.items() if m=='C18-A0' and groups.get((k,name),0)<v]
        off=[r for r in allr if r['directive']=='none'];exact=sum(r['action_sha256']==b[key(r)]['action_sha256'] and r['ink_sha256']==b[key(r)]['ink_sha256'] for r in off)
        out[name]={'guided_goals':sum(r['success'] for r in guided),'guided_total':len(guided),
                   'by_outcome':{k:sum(r['success'] for r in guided if r['outcome']==k) for k in sorted(set(r['outcome'] for r in guided))},
                   'regressed_groups':regress,'no_cue_exact':exact,'no_cue_total':len(off),
                   'host_rejected':sum(r['host_rejected'] for r in allr),'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in allr)}
    return out


def run(root,out,seed=311111,n=2):
    global _CHECKPOINT_
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    _CHECKPOINT_=root/'results/C29-PROBE.safetensors'
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{name}.safetensors' for name in PREREQS])
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    head,fit,receipts=train()
    save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(out/'C31-PROGRESS.safetensors'))
    result={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256((out/'C31-PROGRESS.safetensors').read_bytes()).hexdigest(),
            'training':fit,'cohorts':{}}
    for name,s in (('a',seed),('b',seed+100000)):
       rows=evaluate(prior,option,head,s,n)
       result['cohorts'][name]=summary(rows)
       (out/f'cohort_{name}.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
    (out/'metrics.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'training_receipts.json').write_text(json.dumps(receipts,sort_keys=True,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--seed',type=int,default=311111);p.add_argument('--n',type=int,default=2)
    a=p.parse_args();print(json.dumps(run(a.root,a.out,a.seed,a.n),indent=2))

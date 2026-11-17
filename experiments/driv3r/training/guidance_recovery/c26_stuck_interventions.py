"""C26: intervention-time trust after *executed* mistakes and sustained stuckness.

Research-only synthetic CyberSand-like sandbox. This is not the canonical native ABI.
The host owns legal menus and material protections. Frozen C12 / C18-A0 remain
unchanged. Test includes independently seeded OOD topologies, wrong advice,
matched cue-off arms, and controlled induced mistakes. Only past host-executed
steps, not a hidden map or oracle label, enter the policy.
"""
from __future__ import annotations
import argparse, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import ARMS,sha
from signed_feedback import HostAdvice
from attested_stack_cue import KEYS,observe32
from c19_return_anchored import a0,run_suffix
from c24_history_trust import seed_world
from c25_online_recovery import online_feedback,softmax, Baseline
from personality_retention_core import load_c12
from c22_branch_personality import fav_opposition

SCHEMA='driv3r-guidance-c26-stuck-intervention/1'
OUTCOMES=('reverse','ahead','shortcut','bridge')
CUES=('none','backtrack','continue')


def host_stuckness(w):
    """4 measured *past-only* observables from host-EXECUTED actions.

    Stagnation is sustained lack of distance reduction. Revisit is measured
    from the recorded positions; source and target geometry are never read.
    Existing seeded history is not treated as observed execution.
    """
    events=[e for e in w.events if e[3]==w.goal]
    if not events:return np.zeros(4,np.float32)
    recent=events[-6:]
    nonprogress=0
    for _,old,new,goal in reversed(recent):
        d0=abs(old[0]-goal[0])+abs(old[1]-goal[1]);d1=abs(new[0]-goal[0])+abs(new[1]-goal[1])
        if d1>=d0:nonprogress+=1
        else:break
    positions=[recent[0][1]]+[e[2] for e in recent]
    revisits=sum(positions[i] in positions[:i] for i in range(1,len(positions)))
    g=recent[-1][3]
    start=abs(positions[0][0]-g[0])+abs(positions[0][1]-g[1])
    end=abs(positions[-1][0]-g[0])+abs(positions[-1][1]-g[1])
    return np.array([nonprogress/6,revisits/max(1,len(recent)),np.clip((start-end)/6,-1,1),len(recent)/6],np.float32)


def intervention_trigger(f):
    # This is a deterministic observation gate, NOT a mandatory action.
    return bool(f[3]>=.5 and (f[0]>=.5 or f[1]>=.33))


def observe40(w,arm,advice,stuck_override=None):
    cs,x=observe32(w,arm,advice)
    online=online_feedback(w,advice)
    stuck=host_stuckness(w) if stuck_override is None else np.asarray(stuck_override,np.float32)
    assert stuck.shape==(4,) and np.isfinite(stuck).all() and (abs(stuck)<=1+1e-5).all()
    z=np.concatenate([x,np.tile(online,(10,1)),np.tile(stuck,(10,1))],axis=1)
    assert z.shape==(10,40) and np.array_equal(z[:,:32],x)
    return cs,z


class InterventionHead(nn.Module):
    """Frozen A0 + learned bounded BACKTRACK-only residual while host cues active."""
    def __init__(self,prior,trigger=True,scale=.85):
        super().__init__()
        self.base=a0(prior)
        for p in self.base.parameters():p.requires_grad_(False)
        self.trigger=bool(trigger);self.scale=float(scale)
        self.head=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        assert x.shape[-2:]==(10,40)
        old=self.base(x[...,:32]);out=old.clone();q=x[...,8,:]
        features=torch.cat((q[...,26:28],q[...,29:31],q[...,32:40]),dim=-1)
        change=self.scale*torch.tanh(self.head(features).squeeze(-1))
        active=q[...,2]*q[...,4]*(q[...,26].abs()>.5).to(q.dtype)*q[...,27]*q[...,29]
        if self.trigger:
            # Stuckness observed BEFORE or during cue; still advisory and stochastic.
            use=(q[...,39]>=.5)&((q[...,36]>=.5)|(q[...,37]>=.33))
            active=active*use.to(q.dtype)
        out[...,8]=old[...,8]+active*change
        return out


class Control(nn.Module):
    def __init__(self,prior,a0_enabled):
        super().__init__();self.base=Baseline(prior,a0_enabled)
    def forward(self,x):return self.base(x[...,:36])


def choose(model,w,a,arm,rng):
    cs,x=observe40(w,arm,a)
    with torch.no_grad():scores=model(torch.from_numpy(x)).numpy().astype(np.float64)
    legal=np.asarray([c.legal for c in cs]);raw_bad=int(not legal[scores.argmax()])
    scores=np.where(legal,scores,-1e8)
    p=np.exp((scores-scores.max())/.8);p/=p.sum()
    j=int(rng.choice(len(cs),p=p));return cs[j],j,x,raw_bad


def induce(w,base,arm,seed,steps=5,mode='badturn'):
    """Execute a source-independent prefix once, then clone it for all policies.

    In the induced arm, the host chooses two explicitly bad but legal forward
    actions, followed by alternating policy and non-progressing legal moves.
    The independent neutral arm samples the baseline without disturbances.
    The host's exact legal menu is respected in both arms.
    """
    rng=np.random.default_rng(seed);trace=[]
    for t in range(steps):
        if w.stopped or w.distance()==0:break
        cs,x=observe32(w,arm,HostAdvice(w));legal=[j for j,c in enumerate(cs[:8]) if c.legal]
        if not legal:break
        with torch.no_grad():scores=base(torch.from_numpy(x)).numpy().astype(np.float64)
        if mode=='badturn' and (t<2 or t%2==0):
            # Target distance is visible through each legal action's local
            # progress feature. No full-map topology or hidden goal routes.
            best=max((w.distance(cs[j].target)-w.distance(),int(cs[j].target in [v[2] for v in w.events]), -j) for j in legal)
            j=next(j for j in legal if (w.distance(cs[j].target)-w.distance(),int(cs[j].target in [v[2] for v in w.events]), -j)==best)
        else:
            masked=np.where([c.legal for c in cs],scores,-1e8)
            pr=np.exp((masked-masked.max())/.8);pr/=pr.sum();j=int(rng.choice(10,p=pr))
        c=cs[j];assert c.legal and w.apply(c) and not w.invalid_attempts
        trace.append([c.action[0],list(w.pos),w.distance()])
    return w,trace


def collect(prior,n=7,seed=260126,horizon=20):
    torch.set_num_threads(1);base=a0(prior)
    xx=[];advantages=[];groups=[];receipts=[];triggers=0
    for i in range(n):
      for outcome in OUTCOMES:
       for person in (-1,1):
        for arm in ARMS:
         for mode in ('badturn','neutral'):
          s=seed+i*7919+OUTCOMES.index(outcome)*131071+person*97+int(arm*100)*31+int(mode=='badturn')*227
          w=seed_world(i,outcome,person,s,True)
          w,prefix=induce(w,base,arm,s+119,5,mode)
          if w.distance()==0 or w.stopped:continue
          for directive in ('backtrack','continue'):
           for strength in (.35,1.):
            v=copy.deepcopy(w);a=HostAdvice(v);getattr(a,'issue_'+directive)(v,strength=strength,ttl=9)
            cs,x=observe40(v,arm,a)
            forward=[j for j,c in enumerate(cs[:8]) if c.legal]
            if not cs[8].legal or not forward:continue
            with torch.no_grad():scores=base(torch.from_numpy(x[:,:32])).numpy()
            fwd=max(forward,key=lambda j:float(scores[j]))
            rs=s+int(strength*100)*11+int(directive=='continue')*199
            rb,_,b1,_=run_suffix(base,v,a,arm,8,rs,horizon)
            rf,_,b2,_=run_suffix(base,v,a,arm,fwd,rs,horizon)
            assert b1==b2==0
            adv=float(rb-rf)
            xx.append(x);advantages.append(adv)
            trig=intervention_trigger(x[8,36:40]);triggers+=int(trig)
            group=f'{mode}/{outcome}/{directive}/{strength:.2f}/{int(trig)}';groups.append(group)
            receipts.append({'seed':s,'group':group,'mode':mode,'arm':float(arm),'strength':strength,
               'prefix':prefix,'state_sha256':sha({'pos':v.pos,'trail':v.history,'goal':v.goal,'protected':sorted(v.protected)}),
               'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
               'stuckness':x[8,36:].tolist(),'advantage':adv,'forced_back_return':rb,
               'forced_forward_return':rf,'first_action_back':cs[8].action,'first_action_forward':cs[fwd].action})
    return np.asarray(xx,np.float32),np.asarray(advantages,np.float32),groups,receipts,triggers


def fit(prior,X,A,G,trigger=True,epochs=12,seed=260211):
    torch.set_num_threads(1);torch.manual_seed(seed)
    m=InterventionHead(prior,trigger).train()
    xx=torch.from_numpy(X);aa=torch.from_numpy(A).clamp(-3,3)
    with torch.no_grad():p0=softmax(Control(prior,True)(xx),xx)[:,8]
    counts={g:G.count(g) for g in set(G)}
    weights=torch.tensor([1/counts[g] for g in G],dtype=torch.float32);weights/=weights.mean()
    optimizer=torch.optim.AdamW(m.head.parameters(),lr=.0012,weight_decay=.01)
    rng=np.random.default_rng(seed);losses=[];updates=0
    for ep in range(epochs):
        order=rng.permutation(len(xx));tot=0.
        for start in range(0,len(xx),64):
            ix=order[start:start+64]
            p=softmax(m(xx[ix]),xx[ix])[:,8]
            objective=-(weights[ix]*aa[ix]*(p-p0[ix])).mean()
            drift=(p-p0[ix]).square().mean()
            loss=objective+2.5*drift+.002*sum(z.square().mean() for z in m.head.parameters())
            loss.backward();nn.utils.clip_grad_norm_(m.head.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            updates+=1;tot+=float(loss.detach())*len(ix)
        losses.append(round(tot/len(xx),7))
    return m.eval(),updates,losses


def rollout(model,base,i,outcome,person,arm,directive,strength,mode,seed,horizon=40,trace=False):
    # Identical source and disturbance prefix for every policy.
    w=seed_world(i,outcome,person,seed,False);w,prefix=induce(w,base,arm,seed+119,5,mode)
    pre=sha({'pos':w.pos,'trail':w.history,'goal':w.goal,'protected':sorted(w.protected)})
    pre_distance=w.distance();initial_events=len(w.events);f=host_stuckness(w)
    a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    r=np.random.default_rng(seed+int(arm*100)*29+int(strength*100)*71+{'none':0,'backtrack':411,'continue':977}[directive]);actions=[];path=[]
    initial_ink=set(w.inked);protected=sha(sorted(w.protected));raw_bad=0
    opportunities=preferred=0;first_progress=None;first_net=None;revisits=0;visited={tuple(w.pos)}
    initial_distance=w.distance();backtracks=0;cue_exposed=0
    for tick in range(horizon):
        if w.stopped or w.distance()==0:break
        cs,x=observe40(w,arm,a)
        c,j,_,bad=choose(model,w,a,arm,r);raw_bad+=bad
        pref,opp=fav_opposition(x)
        if a.active(w) and pref.any() and opp.any():
            opportunities+=1;preferred+=int(j<8 and pref[j])
        olddist=w.distance();assert c.legal and w.apply(c) and not w.invalid_attempts
        if first_progress is None and w.distance()<olddist:first_progress=tick+1
        if first_net is None and w.distance()<initial_distance:first_net=tick+1
        backtracks+=int(c.action[0]=='BACKTRACK')
        revisits+=int(tuple(w.pos) in visited);visited.add(tuple(w.pos))
        cue_exposed+=int(a.active(w)!=0);a.advance(w)
        actions.append((c.action,c.target))
        if trace:path.append({'step':tick,'pos':w.pos,'action':c.action,'distance':w.distance(),
                              'host_cue':directive,'stuck':x[8,36:].tolist()})
        assert initial_ink<=w.inked and sha(sorted(w.protected))==protected
    return {'success':int(w.distance()==0),'steps_after_cue':len(actions),'end_distance':w.distance(),
      'first_distance_decrease':first_progress,'first_below_intervention':first_net,
      'revisits_after_cue':revisits,'backtracks':backtracks,'personality_ink':preferred,
      'personality_opportunities':opportunities,'host_rejected':w.invalid_attempts,
      'illegal_raw_top1':raw_bad,'intervention_trigger':intervention_trigger(f),
      'prefix_events':initial_events,'prefix_distance':pre_distance,'prefix_sha256':pre,
      'actions_sha256':sha(actions),'ink_sha256':sha(sorted(w.inked)),
      'prefix':prefix if trace else None,'path':path,
      'geometry':{'start':prefix[0][1] if prefix else w.pos,'goal':w.goal,'protected':sorted(w.protected)} if trace else None}


def evaluate(models,prior,n=6,seed=260999,horizon=40,trace_each=12):
    rows=[];base=a0(prior)
    for i in range(n):
     for outcome in OUTCOMES:
      for person in (-1,1):
       for arm in ARMS:
        for mode in ('badturn','neutral'):
         s=seed+i*10117+OUTCOMES.index(outcome)*101111+person*131+int(arm*100)*17+int(mode=='badturn')*227
         for directive in CUES:
          for strength in ((0.,) if directive=='none' else (.35,1.)):
           for name,model in models.items():
            trace=sum(r['model']==name and bool(r['path']) for r in rows)<trace_each
            rec=rollout(model,base,i,outcome,person,arm,directive,strength,mode,s,horizon,trace)
            rec.update({'model':name,'index':i,'outcome':outcome,'personality':person,
             'arm':float(arm),'mode':mode,'directive':directive,'strength':float(strength),'seed':s})
            rows.append(rec)
    return rows


def group_summary(rows):
    out={}
    for mode in ('badturn','neutral'):
     for outcome in OUTCOMES:
      for directive in CUES:
       for strength in ((0.,) if directive=='none' else (.35,1.)):
        key=f'{mode}/{outcome}/{directive}/{strength:.2f}'
        items=[r for r in rows if r['mode']==mode and r['outcome']==outcome and r['directive']==directive and r['strength']==strength]
        out[key]={name:{'goals':sum(z['success'] for z in items if z['model']==name),
            'n':sum(z['model']==name for z in items),
            'triggered':sum(z['intervention_trigger'] for z in items if z['model']==name),
            'personality_ink':sum(z['personality_ink'] for z in items if z['model']==name),
            'opportunities':sum(z['personality_opportunities'] for z in items if z['model']==name),
            'median_first_net':float(np.median([z['first_below_intervention'] for z in items if z['model']==name and z['first_below_intervention'] is not None])) if any(z['model']==name and z['first_below_intervention'] is not None for z in items) else None}
            for name in sorted({r['model'] for r in items})}
    return out


def qualify(rows):
    gs=group_summary(rows);out={}
    refs={(r['index'],r['outcome'],r['personality'],r['arm'],r['mode'],r['directive'],r['strength']):r
          for r in rows if r['model']=='C12' and r['directive']=='none'}
    for name in ('C26-TRIGGER','C26-FULL'):
        failures=[];improvements=[]
        for group,models in gs.items():
            if '/none/' in group:continue
            a=models['C18-A0']['goals'];b=models[name]['goals']
            if b<a:failures.append([group,a,b])
            if b>a:improvements.append([group,a,b])
        no=[r for r in rows if r['model']==name and r['directive']=='none']
        matched=all(r['actions_sha256']==refs[(r['index'],r['outcome'],r['personality'],r['arm'],r['mode'],r['directive'],r['strength'])]['actions_sha256'] and
                    r['ink_sha256']==refs[(r['index'],r['outcome'],r['personality'],r['arm'],r['mode'],r['directive'],r['strength'])]['ink_sha256'] for r in no)
        out[name]={'strict_nonregression':not failures,'regressions':failures,'improvements':improvements,'cue_off_exact_C12':matched}
    assert not any(r['host_rejected'] or r['illegal_raw_top1'] for r in rows)
    assert all(v['cue_off_exact_C12'] for v in out.values())
    return out


def run(archive,out,train_n=7,eval_n=6,epochs=12,horizon=40,eval_seed=260999):
    torch.set_num_threads(1);archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')
    paths=[archive/'prerequisites'/f'guidance_recovery_{s}.safetensors' for s in names]
    assert all(p.is_file() for p in paths)
    prior=load_c12(paths)
    X,A,G,receipts,triggers=collect(prior,train_n)
    digest=hashlib.sha256(X.tobytes()+A.tobytes()+json.dumps(G).encode()).hexdigest()
    src_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    models={'C12':Control(prior,False),'C18-A0':Control(prior,True)};trained={}
    for trigger,name in ((True,'C26-TRIGGER'),(False,'C26-FULL')):
        m,updates,losses=fit(prior,X,A,G,trigger,epochs)
        models[name]=m
        cp=out/f'{name}.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in m.head.state_dict().items()},str(cp),
           metadata={'provenance':json.dumps({'schema':SCHEMA,'training_data_sha256':digest,'source_sha256':src_sha,
            'gated':trigger,'prerequisites':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in zip(KEYS,paths)}},sort_keys=True)})
        trained[name]={'updates':updates,'loss_first':losses[0],'loss_last':losses[-1],
                       'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
    rows=evaluate(models,prior,eval_n,eval_seed,horizon)
    results={'schema':SCHEMA,'source_sha256':src_sha,'training_data_sha256':digest,
      'training_menus':len(X),'training_triggered_menus':triggers,'counterfactual_suffixes':len(X)*2,
      'advantage_distribution':{'positive':int((A>.05).sum()),'negative':int((A<-.05).sum()),'tied':int((abs(A)<=.05).sum())},
      'trained':trained,'eval_seed':eval_seed,'heldout_episodes':len(rows),'heldout_groups':group_summary(rows),
      'qualification':qualify(rows),'prerequisites':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in zip(KEYS,paths)},
      'limitations':['Synthetic host only; no native CyberSand qualification','Unseen rotated/shifted topologies are synthetic, not physical gameplay',
       'Forced bad-turn prefixes induce errors deliberately; not natural worm mistakes',
       'Offline labels use host-run counterfactual futures; no future state enters inference']}
    for name,data in [('training_receipts.json',receipts),('heldout_trajectories.json',rows),('metrics.json',results)]:
        (out/name).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out')
    p.add_argument('--train-n',type=int,default=7);p.add_argument('--eval-n',type=int,default=6)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--horizon',type=int,default=40)
    p.add_argument('--eval-seed',type=int,default=260999)
    a=p.parse_args();res=run(a.archive,a.out,a.train_n,a.eval_n,a.epochs,a.horizon,a.eval_seed)
    print(json.dumps({'training_menus':res['training_menus'],'training_triggered_menus':res['training_triggered_menus'],
       'heldout_episodes':res['heldout_episodes'],'trained':res['trained'],'qualification':res['qualification']},sort_keys=True))

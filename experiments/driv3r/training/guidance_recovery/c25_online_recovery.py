"""C25: replay-free, online feedback trust for CyberSand-like synthetic gameplay.

Only past executed movements, past observable goal distances, live host advice,
legal action menu, and C18 host-attested trail facts reach the policy. Offline
counterfactual continuations provide TRAINING LABELS, not inference features.
Do not treat this synthetic contract as CyberSand's authoritative native ABI.
"""
from __future__ import annotations
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import ARMS,sha
from attested_stack_cue import observe32,KEYS
from c19_return_anchored import a0,run_suffix
from c24_history_trust import seed_world,Baseline36
from signed_feedback import HostAdvice
from personality_retention_core import load_c12

SCHEMA='driv3r-guidance-c25-online-feedback/1'
OUTCOMES=('reverse','ahead','shortcut','bridge')
CUES=('none','backtrack','continue')


def online_feedback(w,advice):
    """Four bounded causal past-only measurements, with no future/topology labels.

    Current goal and events are host observed. A changed goal invalidates older
    progress samples. Do not mistake the synthetic old goal for a new one.
    No retrospective source reputation or per-world hidden correctness labels.
    """
    if not advice.active(w) or not advice.age:return np.zeros(4,np.float32)
    recent=[e for e in w.events[-min(6,advice.age):] if e[3]==w.goal]
    if not recent:return np.zeros(4,np.float32)
    sign=advice.active(w);progress=[];follow=[];visited=[]
    for kind,old,new,g in recent:
        d0=abs(old[0]-g[0])+abs(old[1]-g[1]);d1=abs(new[0]-g[0])+abs(new[1]-g[1])
        progress.append(max(-1.,min(1.,(d0-d1))))
        follow.append(int((kind=='BACKTRACK') == (sign==1)))
        visited.append(new)
    count=len(recent)
    # Recall locations actually visited in this cue window, not a complete map.
    revisit=sum(visited[:i].count(p)>0 for i,p in enumerate(visited)) / count
    directional_adv=sum(progress[i] for i in range(count) if follow[i]) / max(1,sum(follow))
    # No hindsight beyond executed steps; update after each host-confirmed action.
    return np.array([np.mean(progress),revisit,sum(follow)/count,directional_adv],np.float32)


def observe36(w,arm,advice,feedback=None):
    menu,x=observe32(w,arm,advice)
    f=online_feedback(w,advice) if feedback is None else np.asarray(feedback,np.float32)
    assert f.shape==(4,) and np.isfinite(f).all() and (np.abs(f)<=1+1e-6).all()
    xx=np.concatenate((x,np.tile(f,(10,1))),axis=1)
    assert xx.shape==(10,36) and np.array_equal(x,xx[:,:32])
    return menu,xx


class OnlineResidual(nn.Module):
    """C18-A0 plus a bounded BACKTRACK-only residual under host advice.

    Offline labels learn whether the cue helps; at inference the only memory is
    online audit of real executed actions. No cues => byte-exact C12 logits.
    """
    def __init__(self,prior,history=True,scale=.85):
        super().__init__();self.base=a0(prior)
        for p in self.base.parameters():p.requires_grad_(False)
        self.history=bool(history);self.scale=float(scale)
        self.head=nn.Sequential(nn.Linear(8,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        assert x.shape[-2:]==(10,36)
        z=self.base(x[...,:32]);out=z.clone();q=x[...,8,:]
        mem=q[...,32:36] if self.history else torch.zeros_like(q[...,32:36])
        feat=torch.cat((q[...,26:28],q[...,29:31],mem),dim=-1)
        delta=self.scale*torch.tanh(self.head(feat).squeeze(-1))
        active=q[...,2]*q[...,4]*(q[...,26].abs()>.5).to(q.dtype)*q[...,27]*q[...,29]
        out[...,8]=z[...,8]+active*delta
        return out


class Baseline(nn.Module):
    def __init__(self,prior,gate):super().__init__();self.base=a0(prior) if gate else Baseline36(prior,False)
    def forward(self,x):return self.base(x[...,:32])


def sample_prefix(base,w,advice,arm,steps,seed):
    """Advance actual synthetic host; do not fabricate synthetic feedback state."""
    rng=np.random.default_rng(seed)
    for t in range(steps):
        if w.stopped or not w.distance() or not advice.active(w):break
        menu,x=observe32(w,arm,advice)
        with torch.no_grad():s=base(torch.from_numpy(x)).numpy().astype(np.float64)
        legal=np.array([c.legal for c in menu]);s=np.where(legal,s,-1e8)
        p=np.exp((s-s.max())/.8);p/=p.sum()
        c=menu[int(rng.choice(len(menu),p=p))]
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        advice.advance(w)
    return w,advice


def collect(prior,n=8,seed=250117,horizon=18,repeats=1):
    torch.set_num_threads(1)
    base=a0(prior);xs=[];adv=[];groups=[];receipts=[];feedback_seen=0;bad=0
    for i in range(n):
      for outcome in OUTCOMES:
       for personality in (-1,1):
        for arm in ARMS:
         for directive in ('backtrack','continue'):
          for strength in (.35,1.):
           s=seed+i*7919+OUTCOMES.index(outcome)*111007+personality*73+int(arm*100)*53+int(strength*100)*7+int(directive=='continue')*13
           w=seed_world(i,outcome,personality,s,True)
           a=HostAdvice(w);getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
           for pre in (0,3):
            v,b=copy.deepcopy((w,a));sample_prefix(base,v,b,arm,pre,s+1234)
            if not b.active(v) or not v.distance():continue
            cs,x=observe36(v,arm,b)
            forward=[j for j,c in enumerate(cs[:8]) if c.legal]
            if not cs[8].legal or not forward:continue
            with torch.no_grad():scores=base(torch.from_numpy(x[:,:32])).numpy()
            fwd=max(forward,key=lambda j:float(scores[j]));back_returns=[];forward_returns=[]
            for r in range(repeats):
             rs=s+pre*103+r*3719
             rb,_,err1,_=run_suffix(base,v,b,arm,8,rs,horizon)
             rf,_,err2,_=run_suffix(base,v,b,arm,fwd,rs,horizon)
             bad+=err1+err2;back_returns.append(rb);forward_returns.append(rf)
            advantage=float(np.mean(back_returns)-np.mean(forward_returns))
            feedback_seen+=int(np.any(x[8,32:]))
            xs.append(x);adv.append(advantage);groups.append(outcome+'/'+directive+'/'+str(pre))
            receipts.append({'source_sha256':sha({'pos':v.pos,'trail':v.history,'goal':v.goal,'protected':sorted(v.protected)}),
              'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'feedback':x[8,32:].tolist(),
              'group':groups[-1],'arm':float(arm),'personality':personality,'strength':strength,
              'forced_back_return':back_returns,'forced_forward_return':forward_returns,
              'host_rejected':v.invalid_attempts})
    assert not bad
    return np.asarray(xs,np.float32),np.asarray(adv,np.float32),groups,receipts,feedback_seen


def softmax(z,x):
    legal=x[:,:,4]>.5;z=z.masked_fill(~legal,-1e7)
    return torch.softmax(z/.8,dim=-1)


def optimize(prior,X,A,G,history,epochs=12,seed=250201):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=OnlineResidual(prior,history)
    xx=torch.from_numpy(X);aa=torch.from_numpy(A).clamp(-3,3)
    with torch.no_grad():p0=softmax(Baseline(prior,True)(xx),xx)[:,8]
    counts={g:G.count(g) for g in set(G)}
    wt=torch.tensor([1/counts[g] for g in G],dtype=torch.float32);wt/=wt.mean()
    opt=torch.optim.AdamW(model.head.parameters(),lr=.0015,weight_decay=.01)
    rng=np.random.default_rng(seed);losses=[];updates=0
    for ep in range(epochs):
        order=rng.permutation(len(xx));summed=0.
        for st in range(0,len(order),64):
            indices=order[st:st+64];p=softmax(model(xx[indices]),xx[indices])[:,8]
            improve=-(wt[indices]*aa[indices]*(p-p0[indices])).mean()
            drift=(p-p0[indices]).square().mean()
            loss=improve+2.*drift+.002*sum(z.square().mean() for z in model.head.parameters())
            loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            opt.step();opt.zero_grad(set_to_none=True);updates+=1
            summed+=float(loss.detach())*len(indices)
        losses.append(round(summed/len(xx),7))
    return model.eval(),updates,losses


def rollout(model,i,outcome,personality,arm,directive,strength,seed,horizon=42,trace=False):
    w=seed_world(i,outcome,personality,seed,False);a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    prot=sha(sorted(w.protected));initial=set(w.inked);initial_position=tuple(w.pos);initial_distance=w.distance()
    source=sha({'pos':w.pos,'goal':w.goal,'trail':w.history,'protected':sorted(w.protected)})
    rng=np.random.default_rng(seed+int(arm*100)*29+int(strength*100)*71+{'none':0,'backtrack':411,'continue':977}[directive]);actions=[];path=[]
    bad=0;opportunities=0;fav=0;feedback_seen=0;first_progress=None;first_distance_below_initial=None;revisit_count=0;seen_positions={tuple(w.pos)}
    for step in range(horizon):
        cs,x=observe36(w,arm,a)
        feedback_seen+=int(np.any(x[8,32:]))
        with torch.no_grad():logits=model(torch.from_numpy(x)).numpy().astype(np.float64)
        legal=np.asarray([c.legal for c in cs]);bad+=int(not legal[int(np.argmax(logits))])
        logits=np.where(legal,logits,-1e8);p=np.exp((logits-logits.max())/.8);p/=p.sum()
        from c22_branch_personality import fav_opposition
        f,o=fav_opposition(x)
        if a.active(w) and f.any() and o.any():opportunities+=1
        k=int(rng.choice(len(cs),p=p));c=cs[k]
        if a.active(w) and f.any() and o.any():fav+=int(k<8 and f[k])
        before_distance=w.distance()
        assert c.legal and w.apply(c) and not w.invalid_attempts
        if first_progress is None and w.distance()<before_distance:first_progress=step+1
        if first_distance_below_initial is None and w.distance()<initial_distance:first_distance_below_initial=step+1
        revisit_count+=int(tuple(w.pos) in seen_positions);seen_positions.add(tuple(w.pos))
        a.advance(w);actions.append((c.action,c.target))
        assert initial<=w.inked and sha(sorted(w.protected))==prot
        if trace:path.append({'step':step,'pos':w.pos,'action':c.action,'goal_distance':w.distance(),
             'feedback':x[8,32:].tolist(),'directive':directive})
        if w.distance()==0 or w.stopped:break
    return {'success':int(w.distance()==0),'steps':step+1,'end_distance':w.distance(),
      'host_rejected':w.invalid_attempts,'illegal_raw_top1':bad,'source_sha256':source,
      'actions_sha256':sha(actions),'ink_sha256':sha(sorted(w.inked)),
      'personality_ink':fav,'personality_opportunities':opportunities,
      'nonzero_feedback_steps':feedback_seen,'first_goal_progress_step':first_progress,
      'first_net_recovery_step':first_distance_below_initial,'revisit_steps':revisit_count,'trajectory':path,
      'geometry':{'start':initial_position,'goal':w.goal,'protected':sorted(w.protected)} if trace else None}


def evaluate(models,n=6,seed=250999,horizon=42,trace_each=6):
    rows=[]
    for i in range(n):
     for outcome in OUTCOMES:
      for person in (-1,1):
       for arm in ARMS:
        for directive in CUES:
         for strength in ((0.,) if directive=='none' else (.35,1.)):
          s=seed+i*10117+OUTCOMES.index(outcome)*101111+person*131
          for name,model in models.items():
           trace=sum(int(r['model']==name and bool(r['trajectory'])) for r in rows)<trace_each
           rec=rollout(model,i,outcome,person,arm,directive,strength,s,horizon,trace)
           rec.update({'model':name,'family':'withheld-hook-zigzag-or-transformed-shortcut-bridge',
             'index':i,'outcome':outcome,'personality':person,'arm':float(arm),
             'directive':directive,'strength':float(strength),'seed':s})
           rows.append(rec)
    return rows


def groups(rows):
    ans={}
    for outcome in OUTCOMES:
     for cue in CUES:
      for strength in ((0.,) if cue=='none' else (.35,1.)):
       k=f'{outcome}/{cue}/{strength:.2f}';matching=[r for r in rows if (r['outcome'],r['directive'],r['strength'])==(outcome,cue,strength)]
       ans[k]={n:{'completions':sum(r['success'] for r in matching if r['model']==n),
                   'n':sum(r['model']==n for r in matching),'mean_steps_on_success':round(float(np.mean([r['steps'] for r in matching if r['model']==n and r['success']])),3) if any(r['model']==n and r['success'] for r in matching) else None,
                   'personality_ink':sum(r['personality_ink'] for r in matching if r['model']==n),
                   'personality_opportunities':sum(r['personality_opportunities'] for r in matching if r['model']==n),
                   'median_first_progress':float(np.median([r['first_goal_progress_step'] for r in matching if r['model']==n and r['first_goal_progress_step'] is not None])) if any(r['model']==n and r['first_goal_progress_step'] is not None for r in matching) else None}
              for n in sorted({r['model'] for r in matching})}
    return ans


def qualify(rows):
    agg=groups(rows);results={}
    for name in ('C25-ONLINE','C25-NOMEM'):
        reg=[];improve=[]
        for k,by in agg.items():
            if '/none/' in k:continue
            ref=by['C18-A0']['completions'];cur=by[name]['completions']
            if cur<ref:reg.append([k,ref,cur])
            if cur>ref:improve.append([k,ref,cur])
        baseline={(r['index'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength']):r for r in rows if r['model']=='C12' and r['directive']=='none'}
        unguided=[r for r in rows if r['model']==name and r['directive']=='none']
        exact=all(r['actions_sha256']==baseline[(r['index'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength'])]['actions_sha256'] and r['ink_sha256']==baseline[(r['index'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength'])]['ink_sha256'] for r in unguided)
        results[name]={'strict_nonregression':not reg,'regressions':reg,'improvements':improve,'cue_off_exact_C12':exact}
    assert not any(r['host_rejected'] or r['illegal_raw_top1'] for r in rows)
    assert all(v['cue_off_exact_C12'] for v in results.values())
    return results


def run(archive,out,train_n=8,eval_n=6,epochs=12,horizon=42,repeats=1,eval_seed=250999):
    torch.set_num_threads(1)
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')
    paths=[archive/'prerequisites'/('guidance_recovery_'+s+'.safetensors') for s in names]
    assert all(p.is_file() for p in paths)
    prior=load_c12(paths)
    X,A,G,training_receipts,seen=collect(prior,n=train_n,repeats=repeats)
    datahash=hashlib.sha256(X.tobytes()+A.tobytes()+json.dumps(G).encode()).hexdigest()
    sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    models={'C12':Baseline(prior,False),'C18-A0':Baseline(prior,True)};train_stats={}
    for use_history,name in ((True,'C25-ONLINE'),(False,'C25-NOMEM')):
        mdl,updates,losses=optimize(prior,X,A,G,use_history,epochs)
        models[name]=mdl
        path=out/(name+'.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in mdl.head.state_dict().items()},str(path),
           metadata={'provenance':json.dumps({'schema':SCHEMA,'trained_data_sha256':datahash,'source_sha256':sourcehash,
             'history':use_history,'prerequisite_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)}},sort_keys=True)})
        train_stats[name]={'updates':updates,'loss_first':losses[0],'loss_last':losses[-1],
                           'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    rows=evaluate(models,n=eval_n,seed=eval_seed,horizon=horizon)
    results={'schema':SCHEMA,'training_menus':len(X),'training_nonzero_feedback_menus':seen,
             'training_counterfactual_suffixes':len(X)*2*repeats,'training_advantage_groups':{
              'positive':int((A>.05).sum()),'negative':int((A<-.05).sum()),'near_tied':int((np.abs(A)<=.05).sum())},
             'source_sha256':sourcehash,'training_data_sha256':datahash,
             'prerequisite_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)},
             'training':train_stats,'eval_seed':eval_seed,'heldout_episodes':len(rows),
             'heldout_groups':groups(rows),'qualifications':qualify(rows),
             'nonzero_feedback_eval_steps':sum(r['nonzero_feedback_steps'] for r in rows),
             'no_illegal_or_protected_changes':True,'limitations':[
             'Offline counterfactual labels use host-simulated futures, not deployed outcomes',
             'Online input measures only executed steps, not future or full-map pathfinding',
             'Synthetic research host is not native CyberSand ABI; no human GPU acceptance']}
    for name,data in [('training_receipts.json',training_receipts),('heldout_trajectories.json',rows),('metrics.json',results)]:
        (out/name).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out')
    p.add_argument('--train-n',type=int,default=8);p.add_argument('--eval-n',type=int,default=6)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--horizon',type=int,default=42)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--eval-seed',type=int,default=250999)
    a=p.parse_args();r=run(a.archive,a.out,a.train_n,a.eval_n,a.epochs,a.horizon,a.repeats,a.eval_seed)
    print(json.dumps({'training_menus':r['training_menus'],'training':r['training'],
         'heldout_episodes':r['heldout_episodes'],'qualification':r['qualifications']},sort_keys=True))

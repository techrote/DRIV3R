"""C29: bounded stochastic probe-and-reconsider recovery, synthetic host only.

All executed motion goes through World.apply and freshly computed legal menus;
no hidden topology, shortest-path result, or future outcome enters inference.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import (executed_world, history_evidence, OptionHead,
    TemporalOption, scores as base_scores, probs, rollout as c28_rollout)
from personality_retention_core import load_c12
from c19_return_anchored import a0

SCHEMA='driv3r-c29-probe-reconsider/1'
PREREQS=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
         'persistent_feedback','signed_feedback','selective_continue')

class ProbeHead(nn.Module):
    def __init__(self):
        super().__init__();self.probe=nn.Linear(4,1);self.reconsider=nn.Linear(4,1)
        with torch.no_grad():
            self.probe.weight.zero_();self.probe.bias.zero_()
            self.reconsider.weight.zero_();self.reconsider.bias.zero_()
    def probe_bonus(self,z):return 8.*torch.sigmoid(self.probe(z)).squeeze(-1)
    def reconsider_prob(self,z):return torch.sigmoid(self.reconsider(z)).squeeze(-1)


def fresh_indices(w,menu):
    return [j for j,c in enumerate(menu[:8]) if c.legal and
        c.action[0] in ('INK','LIFT') and c.target not in w.inked and
        c.target not in w.history and w.distance(c.target)<w.distance()]


def obs_probe(w,menu):
    ev=history_evidence(w)
    return np.array([float(bool(fresh_indices(w,menu))),
       min(1.,len(fresh_indices(w,menu))/3.),
       min(2.,(ev['detour_ratio'] if ev else 0.)/4.),
       min(1.,float(w.stuck)/6.)],np.float32)


def obs_reconsider(w,menu,probe_steps):
    onward=fresh_indices(w,menu)
    return np.array([float(bool(onward)),min(1.,len(onward)/3.),
       min(1.,probe_steps/3.),min(1.,float(w.stuck)/6.)],np.float32)


def train_probe(seed=291101,worlds=16,epochs=16):
    torch.set_num_threads(1);torch.manual_seed(seed)
    head=ProbeHead();X=[];Y=[];R=[];T=[];receipts=[]
    for i in range(worlds):
      for person in (-1,1):
       for outcome in ('reverse','tunnel','decoy'):
        w=executed_world(i,outcome,person,seed+i*377+person*29,True)
        advice=HostAdvice(w);advice.issue_backtrack(w,strength=.65,ttl=9)
        m=w.candidates(.35);inds=fresh_indices(w,m)
        X.append(obs_probe(w,m));Y.append(float(bool(inds)))
        receipts.append({'i':i,'outcome':outcome,'personality':person,
          'prefix_sha256':sha([w.pos,w.history,sorted(w.protected)]),
          'menu_sha256':sha([(c.action,c.target,c.legal) for c in m]),
          'progress_candidates':len(inds)})
        if inds:
          # Supervision from *executed* legal probe, not a peek at the full map.
          c=m[inds[0]];assert w.apply(c) and w.invalid_attempts==0
          advice.advance(w)
          mm=w.candidates(.35)
          R.append(obs_reconsider(w,mm,1));T.append(float(not bool(fresh_indices(w,mm))))
    xt=torch.as_tensor(np.array(X),dtype=torch.float32);yt=torch.tensor(Y,dtype=torch.float32)
    rt=torch.as_tensor(np.array(R),dtype=torch.float32);tt=torch.tensor(T,dtype=torch.float32)
    optim=torch.optim.AdamW(head.parameters(),lr=.04,weight_decay=.01)
    hist=[];updates=0;rng=np.random.default_rng(seed)
    for e in range(epochs):
      ix=rng.permutation(len(xt));ir=rng.permutation(len(rt));loss_total=0.
      # Two separate bounded AdamW steps for distinct supervision tasks.
      for idx,features,targets,which in ((ix,xt,yt,'probe'),(ir,rt,tt,'reconsider')):
        for start in range(0,len(idx),64):
          z=features[idx[start:start+64]];y=targets[idx[start:start+64]]
          if which=='probe':predict=head.probe(z).flatten()
          else:predict=head.reconsider(z).flatten()
          loss=nn.functional.binary_cross_entropy_with_logits(predict,y)
          loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.)
          optim.step();optim.zero_grad(set_to_none=True)
          updates+=1;loss_total+=float(loss.detach())
      hist.append(round(loss_total,6))
    with torch.no_grad():
      return head.eval(),{'updates':updates,'epoch_losses':hist,'train_probe_menus':len(X),
       'train_executed_probes':len(R),'probe_accuracy':float(((head.probe(xt).flatten()>0)==yt.bool()).float().mean()),
       'reconsider_accuracy':float(((head.reconsider(rt).flatten()>0)==tt.bool()).float().mean()),
       'train_sha256':hashlib.sha256(xt.numpy().tobytes()+rt.numpy().tobytes()+yt.numpy().tobytes()+tt.numpy().tobytes()).hexdigest()},receipts


class Recoverer:
    """A host-cued bounded search phase with policy-controlled, legal actions.

    A verified fresh improvement is probed. Once a visited probe branch
    exhibits no further fresh improvement, a trained classifier may initiate
    the frozen C28 temporal correction option. After 3 forward steps, abandon
    probe and use the original model. No off-cue actions are modified.
    """
    def __init__(self,option_head,probe_head,max_probe=3,safe_start=False):
        self.safe_start=safe_start
        self.option=TemporalOption(option_head,True,veto=False)
        self.head=probe_head;self.max_probe=max_probe
        self.phase='idle';self.goal=None;self.probe_steps=0
        self.original_goal_dist=0;self.begin_steps=0
        self.probe_decisions=0;self.reconsider_decisions=0;self.released=0
    def _abort(self):
        self.phase='released';self.option.active=False;self.released+=1
    def modify(self,base,w,advice,arm):
        menu,logits,illegal,_=base_scores(base,w,advice,arm)
        if self.phase in ('probe','retrace') and (tuple(w.goal)!=self.goal or advice.directive==0):self._abort()
        if self.phase=='idle' and advice.active(w)!=0:
            ev=history_evidence(w)
            if ev is not None and ev['stack_moves']>=14 and ev['detour_ratio']>=2.:
                self.goal=tuple(w.goal);self.original_goal_dist=w.distance();self.begin_steps=w.steps
                fresh=fresh_indices(w,menu)
                if self.safe_start and fresh:
                    # One-step candidate obstruction is host-provided local
                    # information; don't send probes down an open tunnel.
                    # This deliberately exposes C28's one-step-decoy limitation.
                    candidate=max(fresh,key=lambda j:menu[j].features[22])
                    local_deadend=menu[candidate].features[22]>=.75
                    if local_deadend and ev['stack_moves']>=20 and advice.age==0:
                        self.phase='probe'
                    else:self.phase='released'
                else:self.phase='probe' if fresh else 'retrace' 
        bonus=0.
        if self.phase=='probe':
            improving=fresh_indices(w,menu)
            if self.probe_steps>=1:
                with torch.no_grad():reconsider=float(self.head.reconsider_prob(torch.from_numpy(obs_reconsider(w,menu,self.probe_steps))).item())
                self.reconsider_decisions+=1
                if not improving and reconsider>=.5:self.phase='retrace'
                elif self.probe_steps>=self.max_probe:self._abort()
                elif not improving:self._abort()
            if self.phase=='probe' and improving:
                # Only one locally best legal fresh candidate gets an exploratory
                # logit increment, and the sampling floor remains nonzero.
                j=max(improving,key=lambda idx:logits[idx])
                with torch.no_grad():b=float(self.head.probe_bonus(torch.from_numpy(obs_probe(w,menu))).item())
                bonus=b;logits[j]+=b;self.probe_decisions+=1
                # Logit cap reserves 12% of sampling mass for other legal moves.
                other=np.delete(logits,j);finite=other[other>-1e7]
                cap=.8*math.log(.88/.12)+.8*math.log(np.exp((finite-finite.max())/.8).sum())+float(finite.max())
                logits[j]=min(logits[j],cap)
        if self.phase=='retrace' and self.option.observe(w,advice,menu):
            b=self.option.bonus(w,advice);bonus=b
            logits[8]+=b
            finite=np.delete(logits,8);finite=finite[finite>-1e7]
            cap=.8*math.log(.985/.015)+.8*math.log(np.exp((finite-finite.max())/.8).sum())+float(finite.max())
            logits[8]=min(logits[8],cap)
        return menu,logits,illegal,bonus
    def after(self,before,w,c):
        if self.phase=='probe':
            if c.action[0] in ('INK','LIFT') and w.distance()<before and w.pos not in w.history[:-1]:
                self.probe_steps+=1
            elif c.action[0]=='STOP':self._abort()
        if self.phase=='retrace':self.option.step(w,c)


def custom_rollout(base,option_head,probe_head,i,outcome,person,arm,directive,strength,seed,
                   horizon=90,trace=False,max_probe=3,safe_start=False,world_override=None,baseline_variant=None):
    w=world_override if world_override is not None else executed_world(i,outcome,person,seed,False)
    before_hash=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    protection=sha(sorted(w.protected));a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    recovery=Recoverer(option_head,probe_head,max_probe,safe_start) if baseline_variant is None else None
    legacy=TemporalOption(option_head,True,veto=True) if baseline_variant=='VETO' else None
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+{'none':0,'backtrack':411,'continue':977}[directive])
    actions=[];path=[];illegal_top=0;first_retrace=None;prob_max=0.;probes=0;recons=0
    for tick in range(horizon):
        if w.distance()==0 or w.stopped:break
        menu,logits,bad,bonus=(recovery.modify(base,w,a,arm) if recovery else base_scores(base,w,a,arm,legacy))
        illegal_top+=bad;dist=probs(logits);j=int(rng.choice(10,p=dist));c=menu[j]
        assert c.legal
        if recovery and recovery.phase=='retrace':prob_max=max(prob_max,float(dist[8]))
        olddist=w.distance()
        if c.action[0]=='BACKTRACK' and first_retrace is None:first_retrace=tick+1
        assert w.apply(c) and not w.invalid_attempts
        if recovery:recovery.after(olddist,w,c)
        elif legacy:legacy.step(w,c)
        a.advance(w)
        actions.append((c.action,c.target))
        if trace:path.append({'pos':w.pos,'step':tick+1,'action':c.action[0],
              'distance':w.distance(),'phase':recovery.phase if recovery else baseline_variant,'prob_back':round(float(dist[8]),4)})
    assert sha(sorted(w.protected))==protection
    return {'success':int(w.distance()==0),'steps':len(actions),'end_distance':w.distance(),
       'stopped':bool(w.stopped),'backtracks':sum(k[0][0]=='BACKTRACK' for k in actions),
       'host_rejected':w.invalid_attempts,'raw_illegal_top1':illegal_top,
       'action_sha256':sha(actions),'ink_sha256':sha(sorted(w.inked)),
       'prefix_sha256':before_hash,'first_backtrack':first_retrace,
       'probe_steps':recovery.probe_steps if recovery else 0,'probe_decisions':recovery.probe_decisions if recovery else 0,
       'reconsider_decisions':recovery.reconsider_decisions if recovery else 0,'released':recovery.released if recovery else 0,
       'max_back_probability':prob_max,'path':path if trace else None,
       'goal':w.goal,'protected':sorted(w.protected) if trace else None}


def evaluate(prior,option_head,probe_head,seed=282900,n=2,horizon=72,max_probe=3,baseline_rows=None):
    base=a0(prior);rows=[]
    if baseline_rows is not None:
        rows=[dict(r,i=r['index']) for r in baseline_rows if r['index']<n and r['model'] in ('C12','C18-A0','VETO')]
    for i in range(n):
      for person in (-1,1):
       for outcome in ('reverse','short','near','tunnel','decoy'):
        for arm in (.35,1.):
         for directive in ('none','backtrack','continue'):
          for strength in ((0.,) if directive=='none' else (.35,1.)):
           ss=seed+i*8017+person*313+('reverse','short','near','tunnel','decoy').index(outcome)*3319+int(arm*100)*127
           trace=(i==0 and person==1 and arm==1. and strength==1.)
           for name in (('PROBE','GUARD') if baseline_rows is not None else ('C12','C18-A0','VETO','PROBE','GUARD')):
             if name in ('PROBE','GUARD'):
               r=custom_rollout(base,option_head,probe_head,i,outcome,person,arm,directive,strength,ss,horizon,trace,max_probe,name=='GUARD')
             else:
               r=c28_rollout(prior if name=='C12' else base,option_head,i,outcome,person,arm,
                            directive,strength,ss,False,horizon,name,trace)
             r.update({'model':name,'i':i,'personality':person,'outcome':outcome,
                'arm':arm,'directive':directive,'strength':strength,'seed':ss});rows.append(r)
    return rows


def summarize(rows):
    table={};groups=collections.defaultdict(lambda:collections.Counter())
    for r in rows:
       if r['directive']!='none':groups[(r['outcome'],r['directive'],r['strength'])][r['model']]+=r['success']
    for model in ('C12','C18-A0','VETO','PROBE','GUARD'):
       rr=[r for r in rows if r['model']==model];gg=[r for r in rr if r['directive']!='none']
       table[model]={'guided_success':sum(r['success'] for r in gg),'guided_n':len(gg),
          'by_outcome':{outcome: [sum(r['success'] for r in gg if r['outcome']==outcome),
                sum(r['outcome']==outcome for r in gg)] for outcome in ('reverse','short','near','tunnel','decoy')},
          'illegal_raw_top1':sum(r['raw_illegal_top1'] for r in rr),
          'host_rejected':sum(r['host_rejected'] for r in rr),
          'probe_steps':sum(r.get('probe_steps',0) for r in rr),
          'max_back_probability':max(r.get('max_back_probability',0) for r in rr)}
    quals={}
    for name in ('PROBE','GUARD'):
      regressed=[];improved=[]
      for g,t in groups.items():
        if t[name]<t['C18-A0']:regressed.append({'group':str(g),'baseline':t['C18-A0'],'candidate':t[name]})
        if t[name]>t['C18-A0']:improved.append({'group':str(g),'baseline':t['C18-A0'],'candidate':t[name]})
      quals[name]={'strict_nonregression':not regressed,'regressed':regressed,'improved':improved}
    table['qualification']=quals
    paired={(r['model'],r['i'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength']):r for r in rows}
    for name in ('PROBE','GUARD'):
      off=[(r,paired['C12',r['i'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength']])
           for r in rows if r['model']==name and r['directive']=='none']
      table['qualification'][name]['cueoff_exact_n']=sum(a['action_sha256']==b['action_sha256'] and
                                   a['ink_sha256']==b['ink_sha256'] for a,b in off)
      table['qualification'][name]['cueoff_total']=len(off)
    return table


def run(archive,out,worlds=16,epochs=16,n=2,horizon=72,seed=282900):
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[archive/'prerequisites'/f'guidance_recovery_{x}.safetensors' for x in PREREQS]
    prior=load_c12(paths)
    option_head=OptionHead();option_head.load_state_dict(load_file(str(archive/'results'/'C28-TEMPORAL.safetensors')))
    head,train,receipts=train_probe(seed-300,worlds,epochs)
    cached=[json.loads((archive/'results'/f'cohort_{cohort}.json').read_text()) for cohort in ('a','b')] if (seed==282900 and horizon==72 and n<=4) else [None,None]
    rows=evaluate(prior,option_head,head,seed,n,horizon,baseline_rows=cached[0])
    other=evaluate(prior,option_head,head,seed+100000,n,horizon,baseline_rows=cached[1])
    cp=out/'C29-PROBE.safetensors';save_file({k:v.contiguous().detach().cpu() for k,v in head.state_dict().items()},str(cp))
    metrics={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'train':train,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
        'C28_checkpoint_sha256':hashlib.sha256((archive/'results'/'C28-TEMPORAL.safetensors').read_bytes()).hexdigest(),
        'cohort_a':summarize(rows),'cohort_b':summarize(other),'heldout_episodes':len(rows)+len(other),
        'caveats':['Synthetic host, not CyberSand native acceptance',
            'Probe start and retrace gate depend on host-attested cue and trail',
            'Preceding trail is host-executed but artificially scripted',
            'Only one-step decoys and local tunnel comparators in evaluation',
            'Training labels are observed immediate legality, not counterfactual full trajectory returns',
            'No active-guidance personality guarantee']}
    for name,obj in (('metrics.json',metrics),('train_receipts.json',receipts),('cohort_a.json',rows),('cohort_b.json',other)):
      (out/name).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out');p.add_argument('--worlds',type=int,default=16)
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--n',type=int,default=2)
    p.add_argument('--horizon',type=int,default=72);p.add_argument('--seed',type=int,default=281900)
    args=p.parse_args();print(json.dumps(run(args.archive,args.out,args.worlds,args.epochs,args.n,args.horizon,args.seed),indent=2))

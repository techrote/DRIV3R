"""C36: state-matched guidance personality-budget and return-head control.

Only legal synthetic-host menus; no topology oracle at inference. Training targets
use separately executed counterfactual suffixes. Research-only, not native ABI.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import OptionHead, scores, probs
from c29_probe_recovery import PREREQS
from c34_independent_maze import make_world, ReArmedWatch, RetreatHead, train as train_return
from personality_retention_core import load_c12
from c19_return_anchored import a0

SCHEMA='driv3r-c36-state-matched-personality-budget/1'
ARMS=('C18-A0','C34-RESET','C36-STYLE','C36-LEARNED','C36-RANDOM')
DIRECTIVES=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',.35),('continue',1.))


def persona_score(menu, logits,person):
    """Signed left-versus-right action probability conditional on legal movement."""
    m=np.asarray(logits[:8],np.float64);weights=np.exp((m-m.max())/.8)
    weights/=weights.sum()
    turns=np.asarray([float(c.features[13]-c.features[14])*person for c in menu[:8]])
    return float(np.dot(weights,turns))


def style_project(menu, original, proposal, person, max_shift=.045):
    """Per-state trust region on signed movement-turn preference, not on goal.

    Keep recovery's total movement probability and its BACKTRACK/STOP mass;
    shrink the within-movement directional change only when it exceeds budget.
    This does NOT force sampled actions or guarantee trajectory-level personality.
    """
    orig=np.asarray(original,dtype=np.float64);out=np.asarray(proposal,dtype=np.float64).copy()
    if not any(c.legal and c.features[13] or c.legal and c.features[14] for c in menu[:8]):
        return out,0.,0.,1.
    start=persona_score(menu,orig,person)
    delta=persona_score(menu,out,person)-start
    if abs(delta)<=max_shift:return out,abs(delta),abs(delta),1.
    target=float(np.log(np.exp((out[:8]-out[:8].max())/.8).sum())* .8+out[:8].max())
    lo,hi=0.,1.
    for _ in range(18):
        mid=(lo+hi)*.5
        mix=(1.-mid)*orig[:8]+mid*out[:8]
        score=persona_score(menu,np.concatenate([mix,out[8:]]),person)
        if abs(score-start)>max_shift:hi=mid
        else:lo=mid
    mix=(1.-lo)*orig[:8]+lo*out[:8]
    lse=float(np.log(np.exp((mix-mix.max())/.8).sum())*.8+mix.max())
    out[:8]=mix+(target-lse)
    return out,abs(delta),abs(persona_score(menu,out,person)-start),lo


class BudgetPolicy(ReArmedWatch):
    def __init__(self,option,head,style):
        super().__init__(option,head)
        self.style=style;self.metric=[]
    def modify(self,base,w,advice,arm):
        # This state-matched comparison reads exactly the SAME legal observation.
        m0,l0,_,_=scores(base,w,advice,arm)
        menu,logits,bad,target=super().modify(base,w,advice,arm)
        assert [(c.action,c.target,c.legal) for c in m0]==[(c.action,c.target,c.legal) for c in menu]
        if advice.directive!=0 and self.style:
            logits,pre,post,alpha=style_project(menu,l0,logits,w.personality)
            self.metric.append((pre,post,alpha))
        return menu,logits,bad,target


def run_one(base,option,learned,random_head,w,info,seed,arm,directive,strength,model,
            switch,horizon=88,trace=False):
    advice=HostAdvice(w)
    if directive!='none':getattr(advice,'issue_'+directive)(w,strength=strength,ttl=9)
    if model=='C18-A0':policy=None
    else:
        head=learned if model=='C36-LEARNED' else random_head if model=='C36-RANDOM' else None
        policy=BudgetPolicy(option,head,model!='C34-RESET')
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+
          {'none':0,'backtrack':411,'continue':977}[directive])
    protected=sha(sorted(w.protected));actions=[];path=[]
    goal1=0;goal2=0;bad=0;pref=0;opp=0;neutral=0;ink=0;lift=0;back=0
    for t in range(horizon):
        if switch and t==50:
            goal1=int(goal1 or w.distance()==0);w.goal=info['second_goal'];w.goal_change=True
            w.goals_seen+=1;w.stuck=2
            if directive!='none':getattr(advice,'issue_'+directive)(w,strength=strength,ttl=9)
        if w.stopped:break
        if w.distance()==0:
            if not switch or t<50:goal1=1
            else:goal2=1
            if not switch or t>=50:break
            continue
        if policy is None:menu,logits,raw,_=scores(base,w,advice,arm)
        else:menu,logits,raw,_=policy.modify(base,w,advice,arm)
        bad+=raw
        j=int(rng.choice(len(menu),p=probs(logits)));selected=menu[j]
        assert selected.legal and w.apply(selected) and not w.invalid_attempts
        if policy is not None:policy.after(selected)
        advice.advance(w);actions.append((selected.action,selected.target))
        kind=selected.action[0]
        ink+=kind=='INK';lift+=kind=='LIFT';back+=kind=='BACKTRACK'
        if kind in ('INK','LIFT'):
            turn=float(selected.features[13]-selected.features[14])*w.personality
            pref+=turn>0;opp+=turn<0;neutral+=turn==0
        if trace:path.append({'t':t,'pos':w.pos,'goal':w.goal,'kind':kind,'phase':policy.phase if policy else 'baseline'})
        if t<50 and w.distance()==0:goal1=1
        if switch and t>=50 and w.distance()==0:goal2=1
    if not switch:goal1=int(goal1 or w.distance()==0)
    assert sha(sorted(w.protected))==protected
    metrics=policy.metric if policy is not None else []
    return {'goals':int(goal1+goal2 if switch else goal1),'goal1':goal1,'goal2':goal2,
      'action_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),'steps':len(actions),
      'host_rejections':w.invalid_attempts,'raw_illegal_top1':bad,
      'preferred':int(pref),'opposed':int(opp),'neutral':int(neutral),
      'ink':int(ink),'lift':int(lift),'back':int(back),
      'style_pre_sum':sum(x[0] for x in metrics),'style_post_sum':sum(x[1] for x in metrics),
      'style_n':len(metrics),'style_projected':sum(x[2]<.999 for x in metrics),
      'trace':path if trace else None}


def eval_cohort(base,option,learned,random_head,seed,n=1,horizon=88):
    out=[];graphs={}
    for fam in ('tree','braid'):
     for case in ('detour','forward'):
      print('C36 eval',seed,fam,case, 'time',time.monotonic(),flush=True,file=sys.stderr)
      for switch in (False,True):
       for i in range(n):
        for person in (-1,1):
         for arm in (.35,1.):
          for directive,strength in DIRECTIVES:
           s=(seed+10007*i+379*person+int(arm*100)*199+sum(map(ord,case))*411+
              int(fam=='braid')*17291+int(switch)*12373)
           w,info=make_world(s,fam,case,person)
           graphs[info['graph_sha']]=info
           for model in ARMS:
            r=run_one(base,option,learned,random_head,copy.deepcopy(w),info,s,arm,
                      directive,strength,model,switch,horizon,
                      trace=i==0 and person==1 and arm==1. and directive=='backtrack' and strength==1.)
            out.append({'model':model,'family':fam,'case':case,'switch':switch,'i':i,
                       'personality':person,'arm':arm,'directive':directive,'strength':strength,
                       'seed':s,'graph_sha':info['graph_sha'],**r})
    return out,graphs


def summarize(rows):
    result={};key=lambda r:(r['family'],r['case'],r['switch'],r['i'],r['personality'],r['arm'],r['directive'],r['strength'])
    baseline={key(r):r for r in rows if r['model']=='C18-A0'}
    group=lambda r:(r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    for model in ARMS:
        rr=[r for r in rows if r['model']==model];gg=[r for r in rr if r['directive']!='none']
        n0=[r for r in rr if r['directive']=='none']
        p={str(sign):{'preferred':sum(r['preferred'] for r in gg if r['personality']==sign),
           'opposed':sum(r['opposed'] for r in gg if r['personality']==sign),
           'ink':sum(r['ink'] for r in gg if r['personality']==sign),
           'lift':sum(r['lift'] for r in gg if r['personality']==sign)} for sign in (-1,1)}
        groups={str(k):sum(r['goals'] for r in gg if group(r)==k) for k in sorted({group(r) for r in gg})}
        result[model]={'goals':sum(r['goals'] for r in gg),'guided_n':len(gg),
             'groups':groups,'by_family':{f:sum(r['goals'] for r in gg if r['family']==f) for f in ('tree','braid')},
             'unguided_exact':sum(r['action_sha']==baseline[key(r)]['action_sha'] and
                  r['ink_sha']==baseline[key(r)]['ink_sha'] for r in n0),'unguided_n':len(n0),
             'safety_host_rejections':sum(r['host_rejections'] for r in rr),
             'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr),
             'style_before_mean':sum(r['style_pre_sum'] for r in gg)/max(1,sum(r['style_n'] for r in gg)),
             'style_after_mean':sum(r['style_post_sum'] for r in gg)/max(1,sum(r['style_n'] for r in gg)),
             'style_states':sum(r['style_n'] for r in gg),
             'style_interventions':sum(r['style_projected'] for r in gg),
             'personality':p}
    for m in ARMS:
        for control in ('C18-A0','C34-RESET'):
            result[m]['regressed_vs_'+control]=[k for k,v in result[control]['groups'].items()
                if result[m]['groups'][k]<v]
    return result


def run(root,out,seed=360701,n=1,horizon=88,epochs=8):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS])
    base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    trained,train_summary,receipts=train_return(seed=360013,epochs=epochs)
    torch.manual_seed(360999);random_head=RetreatHead().eval()
    save_file({k:v.detach().contiguous() for k,v in trained.state_dict().items()},str(out/'C36-LEARNED.safetensors'))
    save_file({k:v.detach().contiguous() for k,v in random_head.state_dict().items()},str(out/'C36-RANDOM.safetensors'))
    summary={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'train':train_summary,'train_receipts_sha256':sha(receipts),
         'checkpoints':{fn:hashlib.sha256((out/fn).read_bytes()).hexdigest() for fn in ('C36-LEARNED.safetensors','C36-RANDOM.safetensors')},'cohorts':{},'graphs':{}}
    for label,s in (('A',seed),('B',seed+200003)):
        rows,graphs=eval_cohort(base,option,trained,random_head,s,n,horizon)
        (out/f'C36-cohort-{label}.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
        summary['cohorts'][label]=summarize(rows)
        summary['graphs'][label]={'count':len(graphs),'sha256':sha(graphs)}
    (out/'C36-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--seed',type=int,default=360701)
    p.add_argument('--n',type=int,default=1);p.add_argument('--horizon',type=int,default=88);p.add_argument('--epochs',type=int,default=8)
    a=p.parse_args();s=run(a.root,a.out,a.seed,a.n,a.horizon,a.epochs)
    print(json.dumps({'train':s['train'],'results':{k:{m:{z:v for z,v in q.items() if z in ('goals','unguided_exact','regressed_vs_C34-RESET','style_before_mean','style_after_mean')} for m,q in x.items()} for k,x in s['cohorts'].items()}},indent=2))

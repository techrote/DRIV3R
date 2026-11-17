"""C38: matched trained-vs-label-shuffled return-sign calibration.

Independent synthetic-host research. A genuine CPU AdamW update trains from
counterfactual legal-action suffixes. Held-out graphs/decisions are excluded
from training; all deployed policies receive only the current host legal menu,
executed history and signed host cue. C37 is frozen as an external control.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from c37_shuffle_replication import collect, COHORTS, summarize as c37_summary
from c34_independent_maze import make_world, executed_suffix, obs, RetreatHead
from c36_personality_budget import run_one, PREREQS, load_c12, a0, OptionHead, DIRECTIVES
from experiment import sha

SCHEMA='driv3r-c38-return-sign/1'
ARMS=('C18-A0','C34-RESET','C37-LEARNED','C37-SHUFFLED','C38-SIGN','C38-PERMUTED')

class SignHead(nn.Module):
    """Predict backtrack advantage sign, then bounded symmetric logit bias."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,16),nn.Tanh(),nn.Linear(16,1))
    def logits(self,x):return self.net(x).squeeze(-1)
    def forward(self,x):return 1.5*torch.tanh(self.logits(x))


def train(seed=381013,epochs=16):
    torch.set_num_threads(1)
    X,Y,receipts=collect(seed)
    sel=Y!=0
    assert sel.sum()>=150 and np.any(Y[sel]>0) and np.any(Y[sel]<0)
    x=torch.from_numpy(X[sel].copy())
    yy=torch.from_numpy((Y[sel]>0).astype(np.float32))
    rng=np.random.default_rng(seed+811)
    shuf=yy[rng.permutation(len(yy))].clone()
    assert not torch.equal(shuf,yy)
    torch.manual_seed(seed+991)
    parent=SignHead(); heads=(copy.deepcopy(parent),copy.deepcopy(parent))
    orders=[rng.permutation(len(yy)) for _ in range(epochs)]
    def fit(head,target):
        optimizer=torch.optim.AdamW(head.parameters(),lr=.012,weight_decay=.01)
        record=[];updates=0
        for order in orders:
            t=0.
            for start in range(0,len(order),32):
                j=order[start:start+32]
                l=nn.functional.binary_cross_entropy_with_logits(head.logits(x[j]),target[j])
                optimizer.zero_grad(set_to_none=True);l.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
                t+=float(l.detach());updates+=1
            record.append(round(t,8))
        head.eval()
        return {'updates':updates,'loss_curve':record}
    trained=fit(heads[0],yy);shuffled=fit(heads[1],shuf)
    with torch.no_grad():
        acc=[float(((h.logits(x)>0)==(yy>0)).float().mean()) for h in heads]
        brier=[float(torch.square(torch.sigmoid(h.logits(x))-yy).mean()) for h in heads]
    stats={'train_graphs':len({r['graph'] for r in receipts}),'train_samples':len(X),
           'nonzero_targets':int(sel.sum()),'positive_sign':int(yy.sum()),
           'dataset_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
           'shuffled_label_sha256':hashlib.sha256(shuf.numpy().tobytes()).hexdigest(),
           'executed_receipts_sha256':sha(receipts),'sign_accuracy':acc,
           'training_brier':brier,'fit':{'sign':trained,'permuted':shuffled}}
    return heads,stats,receipts


def heldout_labels(heads, seed=890011, graphs_per_family=12):
    # These full generation seeds are disjoint from training and evaluation.
    rows=[]
    for family in ('tree','braid'):
      for case in ('detour','forward'):
       for person in (-1,1):
        for i in range(graphs_per_family):
            s=seed+i*11117+int(family=='braid')*17113+int(case=='forward')*391+person*73
            w,info=make_world(s,family,case,person)
            for t in range(4):
                menu=w.candidates(.35)
                legal=[j for j,c in enumerate(menu[:8]) if c.legal and c.target not in w.history]
                if not legal or not menu[8].legal:break
                j=max(legal,key=lambda k:w.distance()-w.distance(menu[k].target))
                b=executed_suffix(w,8);f=executed_suffix(w,j)
                with torch.no_grad():v=[float(h(torch.from_numpy(obs(w,menu))).item()) for h in heads]
                rows.append({'family':family,'case':case,'personality':person,'graph_sha':info['graph_sha'],
                             'state_sha':sha([w.pos,w.history,w.goal]),'actual_delta':float(np.clip(b-f,-1.5,1.5)),
                             'predictions':v,'menu_sha':sha([(c.action,c.target,c.legal) for c in menu])})
                assert w.apply(menu[j]) and not w.invalid_attempts
    return rows


def score_labels(rows):
    nonzero=[r for r in rows if r['actual_delta']!=0]
    scores={}
    for j,name in enumerate(('C38-SIGN','C38-PERMUTED','C37-LEARNED','C37-SHUFFLED')):
        vals=[r for r in nonzero]
        scores[name]={'accuracy_nonzero':sum((r['predictions'][j]>0)==(r['actual_delta']>0) for r in vals)/len(vals),
                      'mae_all':float(np.mean([abs(r['predictions'][j]-r['actual_delta']) for r in rows])),
                      'accuracy_tree':sum((r['predictions'][j]>0)==(r['actual_delta']>0) for r in vals if r['family']=='tree')/sum(r['family']=='tree' for r in vals),
                      'accuracy_braid':sum((r['predictions'][j]>0)==(r['actual_delta']>0) for r in vals if r['family']=='braid')/sum(r['family']=='braid' for r in vals)}
    return {'graph_count':len({r['graph_sha'] for r in rows}),'states':len(rows),'nonzero_sign_states':len(nonzero),'scores':scores}


def evaluate(base,option,heads,seed,horizon=78):
    result=[];graphs=set()
    for family in ('tree','braid'):
     for case in ('detour','forward'):
      for switch in (True,):
       for person in (-1,1):
        for arm in (1.,):
         for directive,strength in DIRECTIVES:
            if directive!='none' and strength!=1.:continue
            s=(seed+379*person+int(arm*100)*199+sum(map(ord,case))*411+
               int(family=='braid')*17291+int(switch)*12373)
            w,info=make_world(s,family,case,person)
            graphs.add(info['graph_sha'])
            for name in ARMS:
                if name=='C18-A0':model='C18-A0';head=heads[0]
                elif name=='C34-RESET':model='C34-RESET';head=heads[0]
                else:
                    model='C36-LEARNED';head=heads[{'C37-LEARNED':2,'C37-SHUFFLED':3,'C38-SIGN':0,'C38-PERMUTED':1}[name]]
                r=run_one(base,option,head,heads[1],copy.deepcopy(w),info,s,arm,directive,
                          strength,model,switch,horizon,
                          trace=(person==1 and arm==1. and directive=='backtrack' and strength==1.))
                result.append({'model':name,'family':family,'case':case,'switch':switch,'i':0,
                    'personality':person,'arm':arm,'directive':directive,'strength':strength,
                    'seed':s,'graph_sha':info['graph_sha'],**r})
    return result,graphs


def summary(rows):
    groups=lambda r:(r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    key=lambda r:(r['family'],r['case'],r['switch'],r['personality'],r['arm'],r['directive'],r['strength'])
    controls={key(r):r for r in rows if r['model']=='C18-A0'}
    models={}
    for name in ARMS:
        rr=[r for r in rows if r['model']==name]
        guided=[r for r in rr if r['directive']!='none']
        idle=[r for r in rr if r['directive']=='none']
        cond={str(k):sum(r['goals'] for r in guided if groups(r)==k)
              for k in sorted(set(groups(r) for r in guided))}
        models[name]={'guided_goal_events':sum(r['goals'] for r in guided),
            'guided_n':len(guided),'condition_goals':cond,
            'unguided_exact':sum((r['action_sha'],r['ink_sha'])==(controls[key(r)]['action_sha'],controls[key(r)]['ink_sha']) for r in idle),
            'unguided_n':len(idle),'host_rejections':sum(r['host_rejections'] for r in rr),
            'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in rr),
            'preferred_ink':sum(r['preferred'] for r in guided),
            'opposed_ink':sum(r['opposed'] for r in guided),
            'style_states':sum(r['style_n'] for r in guided),
            'style_shift_before':sum(r['style_pre_sum'] for r in guided),
            'style_shift_after':sum(r['style_post_sum'] for r in guided)}
    for name in ARMS:
        for control in ('C34-RESET','C37-LEARNED','C37-SHUFFLED'):
            models[name]['regressed_vs_'+control]=[k for k,v in models[control]['condition_goals'].items() if models[name]['condition_goals'][k]<v]
    return models


def run(root,out,seed=381013,horizon=78):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    a,trainstats,receipts=train(seed)
    b,repeat,_=train(seed)
    assert trainstats==repeat
    checkpoints={}
    for label,i in (('C38-SIGN',0),('C38-PERMUTED',1)):
        for tag,obj in (('',a[i]),('-REPLICA',b[i])):
            fp=out/(label+tag+'.safetensors')
            save_file({k:v.detach().cpu().contiguous() for k,v in obj.state_dict().items()},str(fp))
        assert (out/(label+'.safetensors')).read_bytes()==(out/(label+'-REPLICA.safetensors')).read_bytes()
        checkpoints[label]=hashlib.sha256((out/(label+'.safetensors')).read_bytes()).hexdigest()
    c37=[]
    for label in ('C37-LEARNED','C37-SHUFFLED'):
        h=RetreatHead();h.load_state_dict(load_file(str(root/'c37-checkpoints'/f'{label}.safetensors')));h.eval()
        c37.append(h)
    heads=[*a,*c37]
    labels=heldout_labels(heads)
    labelscore=score_labels(labels)
    (out/'C38-heldout-labels.json').write_text(json.dumps(labels,sort_keys=True)+'\n')
    (out/'C38-training-receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    results={};meta={}
    for label,evseed in COHORTS:
        print('C38 policy cohort',label,flush=True,file=sys.stderr)
        if (out/f'C38-cohort-{label}.json').exists():
            rows=json.loads((out/f'C38-cohort-{label}.json').read_text())
            graphs={r['graph_sha'] for r in rows}
        else:
            rows,graphs=evaluate(base,option,heads,evseed,horizon)
            (out/f'C38-cohort-{label}.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
        results[label]=summary(rows);meta[label]={'graphs':len(graphs),'graph_set_sha256':sha(sorted(graphs))}
    report={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training':trainstats,'training_reproduction_identical':True,
            'checkpoint_sha256':checkpoints,'heldout_labels':labelscore,'policy':results,'graphs':meta,
            'c37_checkpoints':{label:hashlib.sha256((root/'c37-checkpoints'/f'{label}.safetensors').read_bytes()).hexdigest() for label in ('C37-LEARNED','C37-SHUFFLED')}}
    (out/'C38-summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--seed',type=int,default=381013);p.add_argument('--horizon',type=int,default=78)
    x=p.parse_args();r=run(x.root,x.out,x.seed,x.horizon)
    print(json.dumps({'training':r['training'],'heldout':r['heldout_labels'],
        'cohorts':{c:{m:(v['guided_goal_events'],len(v['regressed_vs_C34-RESET'])) for m,v in s.items()} for c,s in r['policy'].items()}},indent=2))

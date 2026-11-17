"""C37: paired neural-vs-shuffled-vs-random return-head replication.

Research-only synthetic host. C34 independent maze generator and C36 legal-menu,
state-matched active guidance budget are held fixed. All training targets come
from genuinely executed legal continuations, never from runtime topology or an
inference oracle. No CyberSand/native changes or canonical ABI claim.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file
from experiment import sha
from c34_independent_maze import (CASES, make_world, executed_suffix, obs, RetreatHead)
from c36_personality_budget import (run_one, PREREQS, load_c12, a0, OptionHead,
                                     DIRECTIVES)

SCHEMA='driv3r-c37-shuffled-return-replication/1'
ARMS=('C18-A0','C34-RESET','C37-STYLE','C37-LEARNED','C37-SHUFFLED','C37-RANDOM')
COHORTS=(('A',370701),('B',570707),('C',770713))


def collect(seed=370113):
    """20 maze roots per goal case/personality; 4 executed continuation states each."""
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
                    back=executed_suffix(w,8)
                    forward=executed_suffix(w,j)
                    delta=float(np.clip(back-forward,-1.5,1.5))
                    xs.append(obs(w,menu));ys.append(delta)
                    receipts.append({'graph':info['graph_sha'],
                      'state_sha':sha([w.pos,w.history,w.goal]),
                      'menu_sha':sha([(c.action,c.target,c.legal) for c in menu]),
                      'back_return':back,'forward_return':forward,'target':delta})
                    assert w.apply(menu[j]) and w.invalid_attempts==0
    X=np.stack(xs).astype(np.float32);Y=np.asarray(ys,np.float32)
    return X,Y,receipts


def train_controls(seed=370113,epochs=10):
    """Identical initial weights/updates; SHUFFLED destroys feature-label alignment."""
    torch.set_num_threads(1)
    X,Y,receipts=collect(seed)
    rng=np.random.default_rng(seed+88811)
    Yshuffle=Y[rng.permutation(len(Y))].copy()
    assert len(Y)>=250 and not np.array_equal(Y,Yshuffle)
    # Same exact initial model weights for learned, shuffled and random controls.
    torch.manual_seed(seed+9986)
    random_head=RetreatHead().eval()
    learned=copy.deepcopy(random_head);shuffled=copy.deepcopy(random_head)
    x=torch.from_numpy(X);y=torch.from_numpy(Y);ys=torch.from_numpy(Yshuffle)
    permutation_rng=np.random.default_rng(seed+1003)
    orders=[permutation_rng.permutation(len(Y)) for _ in range(epochs)]
    def optimize(head,labels):
        opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
        loss_curve=[];updates=0
        for order in orders:
            total=0.
            for start in range(0,len(order),32):
                ids=order[start:start+32]
                loss=nn.functional.smooth_l1_loss(head(x[ids]),labels[ids])
                opt.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                total+=float(loss.detach());updates+=1
            loss_curve.append(round(total,7))
        head.eval();return updates,loss_curve
    updates,curve=optimize(learned,y)
    su,scurve=optimize(shuffled,ys)
    assert updates==su
    with torch.no_grad():
        learned_mae=float(torch.mean(torch.abs(learned(x)-y)))
        shuffle_mae=float(torch.mean(torch.abs(shuffled(x)-y)))
        random_mae=float(torch.mean(torch.abs(random_head(x)-y)))
        # Explicitly evaluate every head against *actual unshuffled targets*.
        head_difference=float(torch.mean(torch.abs(learned(x)-shuffled(x))))
    stats={'samples':len(Y),'unique_train_graphs':len({r['graph'] for r in receipts}),
           'updates_per_trained_arm':updates,'epochs':epochs,
           'dataset_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
           'permuted_targets_sha256':hashlib.sha256(Yshuffle.tobytes()).hexdigest(),
           'receipts_sha256':sha(receipts),'target_positive':int((Y>0).sum()),
           'target_negative':int((Y<0).sum()),
           'actual_target_mae':{'learned':learned_mae,'shuffled':shuffle_mae,'random':random_mae},
           'head_difference_mae':head_difference,'losses':{'learned':curve,'shuffled':scurve}}
    return (learned,shuffled,random_head),stats,receipts


def run_policy(base,option,heads,w,info,seed,arm,directive,strength,name,switch,horizon,trace):
    # C36.run_one is unchanged: use its identical legal-menu budget and neural-head
    # hook for each of these THREE matched neural candidates.
    model={'C18-A0':'C18-A0','C34-RESET':'C34-RESET',
           'C37-STYLE':'C36-STYLE'}.get(name,'C36-LEARNED')
    picked={'C37-LEARNED':heads[0], 'C37-SHUFFLED':heads[1],
            'C37-RANDOM':heads[2]}.get(name,heads[0])
    return run_one(base,option,picked,heads[2],w,info,seed,arm,directive,
                   strength,model,switch,horizon,trace)


def evaluate(base,option,heads,seed,replicates=1,horizon=72):
    rows=[];graphs={}
    for family in ('tree','braid'):
      for case in ('detour','forward'):
       for switch in (False,True):
        for i in range(replicates):
         for person in (-1,1):
          for arm in (.35,1.):
           for directive,strength in DIRECTIVES:
            s=(seed+10007*i+379*person+int(arm*100)*199+
               sum(map(ord,case))*411+int(family=='braid')*17291+int(switch)*12373)
            w,info=make_world(s,family,case,person)
            graphs[info['graph_sha']]=info
            # A single common host seed, geometry, cue, history and action RNG
            # across all six models. The only intervention is policy head/control.
            for name in ARMS:
                r=run_policy(base,option,heads,copy.deepcopy(w),info,s,arm,
                     directive,strength,name,switch,horizon,
                     trace=(i==0 and person==1 and arm==1. and
                            directive=='backtrack' and strength==1.))
                rows.append({'model':name,'family':family,'case':case,
                   'switch':switch,'i':i,'personality':person,'arm':arm,
                   'directive':directive,'strength':strength,'seed':s,
                   'graph_sha':info['graph_sha'],**r})
    return rows,graphs


def summarize(rows):
    key=lambda r:(r['family'],r['case'],r['switch'],r['i'],r['personality'],
                  r['arm'],r['directive'],r['strength'])
    conditions=lambda r:(r['family'],r['case'],r['switch'],r['directive'],r['strength'])
    baseline={key(r):r for r in rows if r['model']=='C18-A0'}
    models={}
    for model in ARMS:
      allrows=[r for r in rows if r['model']==model]
      guided=[r for r in allrows if r['directive']!='none']
      unguided=[r for r in allrows if r['directive']=='none']
      groups={str(k):sum(r['goals'] for r in guided if conditions(r)==k)
             for k in sorted({conditions(r) for r in guided})}
      models[model]={
          'guided_goal_events':sum(r['goals'] for r in guided),
          'guided_n':len(guided),
          'condition_goals':groups,
          'by_family':{f:sum(r['goals'] for r in guided if r['family']==f)
                       for f in ('tree','braid')},
          'by_personality':{str(p):{'ink':sum(r['ink'] for r in guided if r['personality']==p),
              'lift':sum(r['lift'] for r in guided if r['personality']==p),
              'preferred':sum(r['preferred'] for r in guided if r['personality']==p),
              'opposed':sum(r['opposed'] for r in guided if r['personality']==p),
              'back':sum(r['back'] for r in guided if r['personality']==p)} for p in (-1,1)},
          'unguided_exact':sum(r['action_sha']==baseline[key(r)]['action_sha'] and
                    r['ink_sha']==baseline[key(r)]['ink_sha'] for r in unguided),
          'unguided_n':len(unguided),
          'host_rejections':sum(r['host_rejections'] for r in allrows),
          'raw_illegal_top1':sum(r['raw_illegal_top1'] for r in allrows),
          'style_before_mean':sum(r['style_pre_sum'] for r in guided)/max(1,sum(r['style_n'] for r in guided)),
          'style_after_mean':sum(r['style_post_sum'] for r in guided)/max(1,sum(r['style_n'] for r in guided)),
          'style_states':sum(r['style_n'] for r in guided)}
    for m in models:
      for control in ('C34-RESET','C37-SHUFFLED','C37-RANDOM'):
        models[m]['regressed_vs_'+control]=[k for k,v in models[control]['condition_goals'].items()
                  if models[m]['condition_goals'][k]<v]
    return models


def run(root,out,replicates=1,horizon=72,epochs=10):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS])
    base=a0(prior)
    option=OptionHead();option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')))
    heads,fit,receipts=train_controls(epochs=epochs)
    # Second independent optimizer execution should reconstruct byte-identical bytes.
    replica,repfit,_=train_controls(epochs=epochs)
    assert fit==repfit
    names=('C37-LEARNED','C37-SHUFFLED','C37-RANDOM')
    checkpoint={}
    for name,head,again in zip(names,heads,replica):
        tensors={k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}
        fp=out/(name+'.safetensors');save_file(tensors,str(fp))
        rep=out/(name+'-REPLICA.safetensors');save_file(
            {k:v.detach().cpu().contiguous() for k,v in again.state_dict().items()},str(rep))
        assert fp.read_bytes()==rep.read_bytes()
        checkpoint[name]=hashlib.sha256(fp.read_bytes()).hexdigest()
    summary={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'training':fit,'training_replica_identical':True,'checkpoints':checkpoint,
             'cohorts':{},'graphs':{},'controls':list(ARMS)}
    (out/'C37-training-receipts.json').write_text(json.dumps(receipts,sort_keys=True)+'\n')
    for label,seed in COHORTS:
        print('C37 cohort',label,seed,flush=True,file=sys.stderr)
        rows,graphs=evaluate(base,option,heads,seed,replicates,horizon)
        (out/f'C37-cohort-{label}.json').write_text(json.dumps(rows,sort_keys=True)+'\n')
        summary['cohorts'][label]=summarize(rows)
        summary['graphs'][label]={'unique':len(graphs),'sha256':sha(graphs),
             'tree':sum(x['graph_nodes']==97 for x in graphs.values()),
             'braid':sum(x['graph_nodes']>97 for x in graphs.values())}
    (out/'C37-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('root');ap.add_argument('out')
    ap.add_argument('--replicates',type=int,default=1)
    ap.add_argument('--horizon',type=int,default=72)
    ap.add_argument('--epochs',type=int,default=10)
    args=ap.parse_args()
    result=run(args.root,args.out,args.replicates,args.horizon,args.epochs)
    print(json.dumps({'training':result['training'],
          'results':{c:{m:{'goals':v['guided_goal_events'],
          'regressions':len(v['regressed_vs_C34-RESET'])}
          for m,v in s.items()} for c,s in result['cohorts'].items()}},indent=2))

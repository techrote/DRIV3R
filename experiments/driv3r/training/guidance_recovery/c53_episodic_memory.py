"""C53 — observed branch-memory test of host-issued recovery cues.

Synthetic host research only. A private shortest-route oracle supplies TRAIN/BLIND
labels; inference sees only the current legal menu, host cue and executed history.
Frozen C18/C34/C40/C52 sources are unchanged. No native CyberSand ABI claim.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file
import c52_shortest_step_cue as c52
from experiment import sha
from c28_temporal_option import scores, probs
from c34_independent_maze import ReArmedWatch

SEED=530118
FEATURES=24
VARIANTS=('C53-MEM','C53-SHUFFLED','C53-LOCAL')
POLICIES=('C18-A0','C34-RESET','C40-PAIR','C52-PREV','C53-MEM','C53-SHUFFLED','C53-LOCAL','C53-MASKED','C53-RANDOM')
HORIZON=78

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(FEATURES,16),nn.Tanh(),nn.Linear(16,8),nn.Tanh(),nn.Linear(8,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def observed_memory(w):
    """Eight bounded features from actual prior host-executed actions ONLY.

    Do not query w.protected or w.allowed at remote coordinates, and do not use
    teacher labels/goal outcomes. w.events is the host-observed movement ledger.
    """
    trail=[tuple(e[1]) for e in w.events]+[tuple(w.pos)]
    counts=collections.Counter(trail)
    menu=w.candidates(0.)  # observed legal neighbouring menu, no distant geometry
    neighbours=[tuple(c.target) for c in menu[:4] if c.legal]
    known_edges=collections.defaultdict(set)
    for event in w.events:
        if event[0] in ('INK','LIFT','BACKTRACK'):
            a,b=tuple(event[1]),tuple(event[2]);known_edges[a].add(b);known_edges[b].add(a)
    back=tuple(menu[8].target) if menu[8].legal else tuple(w.pos)
    last_back=max((i for i,p in enumerate(trail[:-1]) if p==back),default=-1)
    repeat_recent=sum(counts[p]>1 for p in trail[-16:])/max(1,min(16,len(trail)))
    observed_forks=sum(len(v)>=3 for v in known_edges.values())
    a=[min(1.,(counts[w.pos]-1)/5.),
       sum(counts[p]>0 for p in neighbours)/max(1,len(neighbours)),
       min(1.,counts[back]/5.),
       0. if last_back<0 else min(1.,(len(trail)-last_back)/28.),
       min(1.,len(counts)/70.),
       repeat_recent,
       sum(counts[p]==0 for p in neighbours)/max(1,len(neighbours)),
       min(1.,observed_forks/12.)]
    assert len(a)==8 and np.isfinite(a).all()
    return np.asarray(a,dtype=np.float32)


def features(w,advice):
    return np.concatenate([c52.features(w,advice),observed_memory(w)]).astype(np.float32)


def teacher(base,option,worldset):
    """Executed legally visited states; oracle label reads hidden geometry ONLY here."""
    rows=[];skips=collections.Counter();prefixes=[]
    for w,info,seed,family,case in worldset:
      for tick in c52.CUE_TICKS:
        got=c52.c50.prefix(base,w,seed,tick)
        if got is None:skips['early_terminal']+=1;continue
        snap,rng_state,receipt=got
        target=c52.c50.resolved_goal(snap,info)
        if target is None:skips['goal_on_history']+=1;continue
        prefixes.append((info['graph_sha'],tick,receipt['action_sha']))
        for arm in (('backtrack',.35),('backtrack',1.)):
          for name in ('C18-A0','C34-RESET'):
            z=copy.deepcopy(snap);z.goal=target;z.goal_change=True;z.goals_seen+=1;z.stuck=2
            rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
            adv=c52.issue(z,arm)
            ctrl=ReArmedWatch(option,None) if name=='C34-RESET' else None
            for age in range(9):
                if z.stopped or z.distance()==0:break
                if adv.active(z)==1:
                    label=c52.label_state(z)
                    if label is not None:
                        rows.append({'graph':info['graph_sha'],'family':family,'case':case,
                          'personality':z.personality,'cue_tick':tick,'age':age,'arm':arm[1],
                          'x':features(z,adv).tolist(),'y':label,
                          'state_sha':sha([z.pos,z.goal,z.history,sorted(z.inked),z.events])})
                if ctrl is None:menu,logits,_,_=scores(base,z,adv,1.)
                else:menu,logits,_,_=ctrl.modify(base,z,adv,1.)
                choice=menu[int(rng.choice(len(menu),p=probs(logits)))]
                assert choice.legal and z.apply(choice) and z.invalid_attempts==0
                if ctrl is not None:ctrl.after(choice)
                adv.advance(z)
    return rows,dict(skips),prefixes


def fit(records,epochs=16):
    torch.set_num_threads(1)
    X=np.asarray([r['x'] for r in records],np.float32)
    y=np.asarray([r['y'] for r in records],np.float32)
    shuffled,changed=c52.shuffle_graph_labels(records,SEED)
    mean=X.mean(0);scale=np.maximum(X.std(0),.08)
    assert len(mean)==FEATURES and len(set(y.tolist()))>1 and changed>0
    init_seed=SEED+71;torch.manual_seed(init_seed)
    init=Head();heads={name:copy.deepcopy(init) for name in VARIANTS}
    order=[np.random.default_rng(SEED+101+i).permutation(len(records)) for i in range(epochs)]
    curves={};updates={}
    for name in VARIANTS:
        z=X.copy()
        if name=='C53-LOCAL':z[:,16:]=0.0
        XX=torch.from_numpy(np.clip((z-mean)/scale,-6,6).astype(np.float32))
        labels=shuffled if name=='C53-SHUFFLED' else y
        Y=torch.from_numpy(labels)
        p=float(np.mean(labels));weights=torch.where(Y>0,.5/max(p,.03),.5/max(1-p,.03))
        head=heads[name];opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.02)
        losses=[];steps=0
        for idx in order:
            s=0.
            for k in range(0,len(idx),64):
                take=idx[k:k+64];logits=head(XX[take]);loss=(nn.functional.binary_cross_entropy_with_logits(logits,Y[take],reduction='none')*weights[take]).mean()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                s+=float(loss.detach())*len(take);steps+=1
            losses.append(round(s/len(records),6))
        head.eval();curves[name]=losses;updates[name]=steps
    return heads,mean,scale,{'examples':len(records),'graphs':len(set(r['graph'] for r in records)),
       'aligned_positive':int(y.sum()),'shuffled_label_changes':changed,
       'features_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
       'labels_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
       'shuffled_labels_sha256':hashlib.sha256(shuffled.tobytes()).hexdigest(),
       'updates':updates,'losses':curves}


def logit(head,x,mean,scale):
    x=torch.from_numpy(np.clip((x-mean)/scale,-6,6).astype(np.float32))
    with torch.no_grad():return float(head(x).item())


def suffix(base,option,prior,snap,rng_state,tick,target,arm,name,models,means,scales,trace=False):
    if name in ('C18-A0','C34-RESET','C40-PAIR'):
        z=c52.c50.suffix(base,option,prior,snap,rng_state,0,name,arm,tick,target,trace)
        z['neural_states']=0
        return z
    w=copy.deepcopy(snap);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
    w.goal=target;w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c52.issue(w,arm);actions=[];path=[];pref=opp=state_count=0;reached=False;end=None
    for t in range(tick,HORIZON):
        if w.stopped:break
        if w.distance()==0:reached=True;break
        menu,logits,_,_=scores(base,w,advice,1.)
        if advice.active(w)==1 and menu[8].legal:
            if name=='C53-RANDOM':v=1. if rng.random()<.5 else -1.
            elif name=='C52-PREV':v=logit(models[name],c52.features(w,advice),means[name],scales[name])
            else:
                x=features(w,advice)
                if name=='C53-LOCAL':x[16:]=0.0
                if name=='C53-MASKED':x[16:]=means['C53-MEM'][16:]
                v=logit(models['C53-MEM' if name=='C53-MASKED' else name],x,
                        means['C53-MEM' if name=='C53-MASKED' else name],scales['C53-MEM' if name=='C53-MASKED' else name])
            logits[8]+=float(np.clip(v*2.5,-3.5,3.5))*float(advice.strength)
            state_count+=1
        choice=menu[int(rng.choice(len(menu),p=probs(logits)))]
        assert choice.legal and w.apply(choice) and w.invalid_attempts==0
        advice.advance(w);actions.append((choice.action,choice.target));end=t
        if choice.action[0] in ('INK','LIFT'):
            turn=float(choice.features[13]-choice.features[14])*w.personality
            pref+=int(turn>0);opp+=int(turn<0)
        if trace:path.append({'t':t,'pos':w.pos,'action':choice.action[0]})
        if w.distance()==0:reached=True;break
    assert sha(sorted(w.protected))==sha(sorted(snap.protected))
    return {'goal':int(reached),'steps_after_cue':len(actions),'success_tick':end if reached else None,
      'action_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
      'protected_sha':sha(sorted(w.protected)),'host_rejections':w.invalid_attempts,
      'preferred_ink':pref,'opposed_ink':opp,'neural_states':state_count,
      'path':path if trace else None}


def evaluate(base,option,prior,worldset,models,means,scales,trace_limit=18):
    rows=[];traces=[];skips=collections.Counter()
    for w,info,seed,family,case in worldset:
      for tick in c52.CUE_TICKS:
        got=c52.c50.prefix(base,w,seed,tick)
        if got is None:skips['early_terminal']+=1;continue
        snap,rng_state,receipt=got;target=c52.c50.resolved_goal(snap,info)
        if target is None:skips['goal_on_history']+=1;continue
        for arm in c52.ARMS:
            if arm[0] in ('none','continue'):
                z=suffix(base,option,prior,snap,rng_state,tick,target,arm,'C18-A0',models,means,scales)
                outcomes={m:copy.deepcopy(z) for m in POLICIES}
            else:outcomes={m:suffix(base,option,prior,snap,rng_state,tick,target,arm,m,models,means,scales) for m in POLICIES}
            assert all(z['host_rejections']==0 and z['protected_sha']==receipt['protected_sha'] for z in outcomes.values())
            if arm[0]=='none':assert len({(z['action_sha'],z['ink_sha']) for z in outcomes.values()})==1
            rows.append({'graph':info['graph_sha'],'family':family,'case':case,'personality':w.personality,
              'seed':seed,'tick':tick,'arm':arm[0],'strength':arm[1],
              'prefix_sha':receipt['action_sha'],'prefix_ink_sha':receipt['ink_sha'],
              'protected_sha':receipt['protected_sha'],'target':target,'outcomes':outcomes})
            if arm==('backtrack',1.) and len(traces)<trace_limit:
                for m in ('C34-RESET','C52-PREV','C53-MEM','C53-SHUFFLED','C53-LOCAL','C53-MASKED'):
                    if len(traces)>=trace_limit:break
                    z=suffix(base,option,prior,snap,rng_state,tick,target,arm,m,models,means,scales,True)
                    assert (z['action_sha'],z['ink_sha'])==(outcomes[m]['action_sha'],outcomes[m]['ink_sha'])
                    traces.append({'graph':info['graph_sha'],'family':family,'case':case,'model':m,
                      'seed':seed,'personality':w.personality,'start':receipt['initial'],
                      'prefix_positions':receipt['positions'],'protected':sorted(snap.protected),
                      'target':target,'outcome':z})
    return rows,traces,dict(skips)


def report(train,blind,online,receipts,skips,stats,checkpoints):
    trainids,blindids,onlineids=({r['graph'] for r in receipts[c]} for c in ('train','blind','online'))
    assert not (trainids&blindids or trainids&onlineids or blindids&onlineids)
    guided=[r for r in online if r['arm']!='none']
    groups=collections.defaultdict(collections.Counter);graphs=collections.defaultdict(collections.Counter)
    models={}
    for r in guided:
        group=(r['family'],r['case'],r['tick'],r['arm'],r['strength'])
        for m,z in r['outcomes'].items():groups[group][m]+=z['goal'];graphs[r['graph']][m]+=z['goal']
    unguided=[r for r in online if r['arm']=='none']
    for m in POLICIES:
        data=[r['outcomes'][m] for r in guided]
        models[m]={'guided_goals':sum(z['goal'] for z in data),'guided_n':len(data),
          'by_family':{f:sum(r['outcomes'][m]['goal'] for r in guided if r['family']==f) for f in c52.ONLINE_FAMILIES},
          'condition_regressions_vs_RESET':sum(v[m]<v['C34-RESET'] for v in groups.values()),
          'zero_guidance_hash_parity':sum((r['outcomes'][m]['action_sha'],r['outcomes'][m]['ink_sha'])==
                   (r['outcomes']['C18-A0']['action_sha'],r['outcomes']['C18-A0']['ink_sha']) for r in unguided),
          'host_rejections':sum(r['outcomes'][m]['host_rejections'] for r in online),
          'preferred_ink':sum(z['preferred_ink'] for z in data),'opposed_ink':sum(z['opposed_ink'] for z in data),
          'active_neural_states':sum(z.get('neural_states',0) for z in data)}
    paired={}
    for m in ('C53-MEM','C53-SHUFFLED','C53-LOCAL','C53-MASKED','C52-PREV'):
      for baseline in ('C34-RESET','C53-LOCAL','C53-SHUFFLED'):
        if m==baseline:continue
        differences=np.asarray([g[m]-g[baseline] for g in graphs.values()],np.int32)
        draws=differences[np.random.default_rng(SEED+len(m)*7+len(baseline)*13).integers(len(differences),size=(3000,len(differences)))].sum(1)
        paired[m+'__'+baseline]={'delta':int(differences.sum()),'favorable_graphs':int((differences>0).sum()),
          'unfavorable_graphs':int((differences<0).sum()),'descriptive_cluster_bootstrap95':list(map(float,np.quantile(draws,[.025,.975])))}
    blind_stats={}
    yy=np.asarray([r['y'] for r in blind],np.float32)
    for m in ('C53-MEM','C53-SHUFFLED','C53-LOCAL','C53-MASKED','C52-PREV'):
        pp=np.asarray([r['predictions'][m] for r in blind])
        blind_stats[m]={'accuracy':float(np.mean((pp>=.5)==yy)),
          'brier':float(np.mean((pp-yy)**2))}
    memory_gain=paired['C53-MEM__C53-LOCAL']
    gates={'heldout_goals_above_RESET':models['C53-MEM']['guided_goals']>models['C34-RESET']['guided_goals'],
      'beats_shuffled':models['C53-MEM']['guided_goals']>models['C53-SHUFFLED']['guided_goals'],
      'beats_local_ablation':memory_gain['delta']>0,
      'both_ood_families_positive_vs_RESET':all(models['C53-MEM']['by_family'][f]>models['C34-RESET']['by_family'][f] for f in c52.ONLINE_FAMILIES),
      'zero_condition_regressions':models['C53-MEM']['condition_regressions_vs_RESET']==0,
      'positive_cluster_CI_vs_RESET':paired['C53-MEM__C34-RESET']['descriptive_cluster_bootstrap95'][0]>0,
      'exact_no_guidance':all(v['zero_guidance_hash_parity']==len(unguided) for v in models.values()),
      'host_legal':all(v['host_rejections']==0 for v in models.values())}
    return {'schema':'driv3r-c53-observed-episodic-branch-memory/1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'cohort_graphs':{c:len({r['graph'] for r in receipts[c]}) for c in ('train','blind','online')},
      'cohort_graph_sha':{c:sha(sorted({r['graph'] for r in receipts[c]})) for c in ('train','blind','online')},
      'train':stats,'checkpoints_sha256':checkpoints,'train_records':len(train),'blind_records':len(blind),
      'online_rows':len(online),'guided_episodes_per_policy':len(guided),
      'independent_executed_suffixes':sum(1 if r['arm'] in ('none','continue') else len(POLICIES) for r in online),
      'models':models,'paired':paired,'blind_classification':blind_stats,'gates':gates,
      'promotion_eligible':all(gates.values()),'world_generation':receipts,'skipped':skips,
      'caveats':['Synthetic host only, not native CyberSand acceptance',
      'Teacher BFS sees hidden protected geometry during training/blind scoring only',
      'Graph-independent sample size is graph count, not correlated decision-state count',
      'History features use host executed event ledger including observed visited positions, not remote legal geometry',
      'Historical d_common does not supersede CyberSand host action ABI',
      'Matched guidance and no-guidance are synthetic only; full trajectory personality not validated',
      'External generalization limited to Wilson/Kruskal in fixed 15x15 simulation grid',
      'Bootstrap descriptive only, uncorrected for extensive prior research selection',
      'C53-RANDOM uses extra RNG draws; only randomized negative control',
      'Checkpoints local archives only, not immutable GitHub Actions artifact custody']}


def main(prior,out,train_reps=2,blind_reps=2,online_reps=3):
    prior=Path(prior);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    base,opt,frozen=c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    prior_52=Path(__file__).resolve().parent/'source_and_evidence'
    prior_sum=json.loads((prior_52/'C52-summary.json').read_text())
    prior_check=prior_sum['checkpoints_sha256']['C52-ALIGNED']
    ck=Path(__file__).resolve().parent/'results/C52-ALIGNED.safetensors'
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==prior_check
    old=c52.Head();old.load_state_dict(load_file(str(ck)));old.eval()
    old_norm=json.loads((Path(__file__).resolve().parent/'results/C52-normalization.json').read_text())
    old_mean=np.asarray(old_norm['mean'],np.float32);old_scale=np.asarray(old_norm['scale'],np.float32)
    previous=set()
    for fname in ('C52-training.json','C52-blind.json','C52-online.json'):
       previous.update(r['graph'] for r in json.loads((Path(__file__).resolve().parent/'results'/fname).read_text()))
    c51=json.loads((Path(__file__).resolve().parent/'c51/output_c51/C51-online.json').read_text())
    previous.update(r['graph'] for r in c51)
    trainw,tr=c52.worlds(c52.TRAIN_FAMILIES,c52.c50.c47.c41.world,SEED+17,train_reps,exclude=previous)
    records,skip_train,_=teacher(base,opt,trainw)
    print('C53 train',len(trainw),'graphs',len(records),'states',flush=True)
    heads,mean,scale,stats=fit(records)
    checks={}
    for name,head in heads.items():
        ckpath=out/(name+'.safetensors');save_file(head.state_dict(),str(ckpath))
        checks[name]=hashlib.sha256(ckpath.read_bytes()).hexdigest()
    train_ids={r['graph'] for r in tr}
    blindw,bl=c52.worlds(c52.ONLINE_FAMILIES,c52.c50.c47.c42.world,SEED+201131,blind_reps,exclude=previous|train_ids)
    onlinew,on=c52.worlds(c52.ONLINE_FAMILIES,c52.c50.c47.c42.world,SEED+403117,online_reps,
                       exclude=previous|train_ids|{r['graph'] for r in bl})
    blind,skip_blind,_=teacher(base,opt,blindw)
    models={**heads,'C52-PREV':old}
    means={**{m:mean for m in heads},'C52-PREV':old_mean}
    scales={**{m:scale for m in heads},'C52-PREV':old_scale}
    for r in blind:
      x=np.asarray(r['x'],np.float32)
      r['predictions']={}
      for name in ('C53-MEM','C53-SHUFFLED','C53-LOCAL','C53-MASKED','C52-PREV'):
        q=x.copy()
        if name=='C53-LOCAL':q[16:]=0.
        if name=='C53-MASKED':q[16:]=mean[16:]
        if name=='C52-PREV':q=q[:16]
        model=models['C53-MEM'] if name=='C53-MASKED' else models[name]
        mu=means['C53-MEM'] if name=='C53-MASKED' else means[name]
        sig=scales['C53-MEM'] if name=='C53-MASKED' else scales[name]
        r['predictions'][name]=1/(1+math.exp(-logit(model,q,mu,sig)))
    print('C53 blind',len(blindw),'graphs',len(blind),'states',flush=True)
    online,traces,skip_online=evaluate(base,opt,frozen,onlinew,models,means,scales)
    result=report(records,blind,online,{'train':tr,'blind':bl,'online':on},
           {'train':skip_train,'blind':skip_blind,'online':skip_online},stats,checks)
    result['previous_C52_ALIGNED_sha256']=prior_check
    for name,obj in [('C53-training.json',records),('C53-blind.json',blind),('C53-online.json',online),
        ('C53-selected-traces.json',traces),('C53-summary.json',result)]:
        (out/name).write_text(json.dumps(obj,separators=(',',':'),sort_keys=True)+'\n')
    (out/'C53-normalization.json').write_text(json.dumps({'mean':mean.tolist(),'scale':scale.tolist()},sort_keys=True)+'\n')
    print(json.dumps({'guided_goals':{m:z['guided_goals'] for m,z in result['models'].items()},
        'paired_memory_vs_LOCAL':result['paired']['C53-MEM__C53-LOCAL'],
        'blind':result['blind_classification'],'gates':result['gates']},indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('out');
    p.add_argument('--train-reps',type=int,default=2);p.add_argument('--blind-reps',type=int,default=2)
    p.add_argument('--online-reps',type=int,default=3);a=p.parse_args()
    main(a.prior,a.out,a.train_reps,a.blind_reps,a.online_reps)

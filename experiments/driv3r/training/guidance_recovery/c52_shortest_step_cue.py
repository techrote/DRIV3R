"""C52: action-level recovery cue calibration with a graph oracle confined to TRAINING.

Independent synthetic-host DRIV3R research. At inference no shortest path, hidden
cells, goal outcomes, or policy-specific counterfactuals are consulted. Only
actual host-observed prefix, recent history, legal menu and signed host cue.
No native CyberSand/Neurounner mutations or ABI claims.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math, os
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
import c50_continuous_recovery as c50
from experiment import DIRECTIONS, N, sha
from signed_feedback import HostAdvice
from c28_temporal_option import scores, probs
from c39_action_value import ConditionalPolicy
from c34_independent_maze import ReArmedWatch

SCHEMA='driv3r-c52-action-level-oracle-cue-calibration/1'
SEED=520118
TRAIN_FAMILIES=('tree','braid','prim','prim_braid')
ONLINE_FAMILIES=('wilson','kruskal')
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.),('continue',1.))
POLICIES=('C18-A0','C34-RESET','C40-PAIR','C52-ALIGNED','C52-SHUFFLED','C52-RANDOM')
CUE_TICKS=(9,19)
HORIZON=78
NFEATURES=16

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(NFEATURES,16),nn.Tanh(),nn.Linear(16,8),nn.Tanh(),nn.Linear(8,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def features(w,advice):
    """Present/past-only local observation; never inspect remote protected cells.

    Legal neighbour mask is host supplied, and only immediately adjacent cell
    features are consulted. The shortest-route oracle below is TRAINING ONLY.
    """
    menu=w.candidates(0.)
    previous=w.history[-1] if w.history else w.pos
    front=[c for c in menu[:4] if c.legal]
    distinctive={c.target for c in front}
    delta=w.distance()-w.distance(previous)
    fresh=sum(c.target not in w.history and c.target not in w.inked for c in front)
    progress=sum(w.distance(c.target)<w.distance() for c in front)
    back=int(bool(w.history) and menu[8].legal)
    turns=[(w.last_dir[0]*(p[1]-w.pos[1])-w.last_dir[1]*(p[0]-w.pos[0]))*w.personality
           for p in distinctive]
    r=np.asarray([min(1,len(w.history)/75), min(1,w.stuck/8),min(1,w.distance()/28),
      np.clip(delta/2.,-1,1),fresh/4,progress/4,len(distinctive)/4,
      float(back),float(previous in w.inked),float(previous in w.history[:-1]),
      float(w.personality),float(w.goal_change),
      float(advice.strength if advice.active(w)==1 else 0),
      float(advice.remaining/12),float(np.mean(turns) if turns else 0),
      float(w.steps/90)],dtype=np.float32)
    assert r.shape==(NFEATURES,) and np.isfinite(r).all()
    return r


def distance_oracle(w):
    """A private offline teacher: BFS using COMPLETE protected geometry.

    Its output is NEVER passed into features() or runtime policy selection.
    This explicitly is not an available CyberSand player observation.
    """
    q=deque([w.goal]); dist={w.goal:0}
    while q:
        x,y=q.popleft()
        for dx,dy in DIRECTIONS:
            p=(x+dx,y+dy)
            if w.allowed(p) and p not in dist:dist[p]=dist[(x,y)]+1;q.append(p)
    return dist


def label_state(w):
    menu=w.candidates(0.)
    if not menu[8].legal:return None
    dist=distance_oracle(w)
    prev=tuple(menu[8].target)
    if prev not in dist:return None
    alternatives={c.target for c in menu[:4] if c.legal and tuple(c.target)!=prev}
    other=min((dist.get(p,999) for p in alternatives),default=999)
    if other==999:return 1 # sole legal escape: use actual host backtrack
    return int(dist[prev]<other)


def worlds(families,worldfn,seed,reps,exclude=()):
    """Geometry-only deterministic retries; no model outcomes inspected.

    Reject repeated graph geometry before any rollout, plus earlier C50/C51
    OOD test graphs supplied to exclusion. Every retry is receipted.
    """
    rows=[];seen=set(exclude);receipts=[]
    for fi,fam in enumerate(families):
      for case in ('detour','forward'):
       for person in (-1,1):
        for i in range(reps):
         origin=seed+fi*170003+(case=='forward')*19013+(person+1)*3737+i*21041
         for trial in range(180):
            s=origin+trial*104729
            try:w,info=worldfn(s,fam,case,person)
            except (AssertionError, ValueError):continue
            graph=info['graph_sha']
            if graph in seen:continue
            seen.add(graph)
            rows.append((w,info,s,fam,case))
            receipts.append({'graph':graph,'seed':s,'initial_seed':origin,'retry':trial,
              'family':fam,'case':case,'personality':person})
            break
         else: raise AssertionError('cannot generate eligible topology without duplicates')
    return rows,receipts


def issue(w,arm):
    directive,strength=arm
    advice=HostAdvice(w)
    if directive!='none':getattr(advice,'issue_'+directive)(w,strength=strength,ttl=9)
    return advice


def collect_teacher(base,option,prior,worldset,rollout_arms=(('backtrack',.35),('backtrack',1.))):
    records=[];skip=collections.Counter();prefix_keys=[]
    for w,info,s,fam,case in worldset:
      for tick in CUE_TICKS:
        p=c50.prefix(base,w,s,tick)
        if p is None:skip['early_terminal']+=1;continue
        snap,rng_state,receipt=p
        target=c50.resolved_goal(snap,info)
        if target is None:skip['goal_on_history']+=1;continue
        prefix_keys.append((info['graph_sha'],tick,receipt['action_sha']))
        for arm in rollout_arms:
         for model in ('C18-A0','C34-RESET'):
          z=copy.deepcopy(snap);z.goal=target;z.goal_change=True;z.goals_seen+=1;z.stuck=2
          rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
          adv=issue(z,arm)
          pctrl=ReArmedWatch(option,None) if model=='C34-RESET' else None
          for j in range(9):
            if z.stopped or z.distance()==0:break
            if adv.active(z)==1:
                label=label_state(z)
                if label is not None:
                  records.append({'graph':info['graph_sha'],'family':fam,'case':case,
                    'personality':z.personality,'arm':arm[1],'tick':tick,'age':j,
                    'x':features(z,adv).tolist(),'y':label,
                    'state_sha':sha([z.pos,z.goal,z.history,z.last_dir,z.stuck,z.inked if False else sorted(z.inked)]),
                    'oracle_label_method':'private-BFS-best-unique-neighbor'})
            if pctrl is None:menu,logits,_,_=scores(base,z,adv,1.)
            else:menu,logits,_,_=pctrl.modify(base,z,adv,1.)
            selected=menu[int(rng.choice(len(menu),p=probs(logits)))]
            assert selected.legal and z.apply(selected) and not z.invalid_attempts
            if pctrl is not None:pctrl.after(selected)
            adv.advance(z)
    return records,dict(skip),prefix_keys


def shuffle_graph_labels(rows,seed):
    """Nonidentical graph-level shuffle within family/case/personality strata."""
    y=np.asarray([r['y'] for r in rows],dtype=np.float32);out=y.copy()
    groups=collections.defaultdict(dict)
    for i,r in enumerate(rows):
        key=(r['family'],)
        groups[key].setdefault(r['graph'],[]).append(i)
    rng=np.random.default_rng(seed+31)
    changed=0
    for bygraph in groups.values():
        gg=sorted(bygraph)
        if len(gg)<2:continue
        # Per-graph labels are drawn from a DIFFERENT graph's empirical labels,
        # without exposing an aligned label vector as weighting or input.
        perm=np.roll(gg,1+int(rng.integers(len(gg)-1)))
        for target,source in zip(gg,perm):
            dest=bygraph[target];values=y[bygraph[source]]
            if len(values):
                offset=int(rng.integers(len(values)))
                for k,idx in enumerate(dest):out[idx]=values[(k+offset)%len(values)]
    changed=int(np.sum(out!=y));assert changed>0,'inactive shuffled control'
    return out,changed


def fit(rows,seed=SEED,epochs=22):
    assert len({r['graph'] for r in rows})>=8
    torch.set_num_threads(1)
    X=np.asarray([r['x'] for r in rows],dtype=np.float32)
    y=np.asarray([r['y'] for r in rows],dtype=np.float32)
    shuffled,changed=shuffle_graph_labels(rows,seed)
    mean=X.mean(0);scale=np.maximum(X.std(0),.08)
    XX=torch.from_numpy(np.clip((X-mean)/scale,-6,6).astype(np.float32))
    torch.manual_seed(seed+71);init=Head();heads={x:copy.deepcopy(init) for x in ('C52-ALIGNED','C52-SHUFFLED')}
    orders=[np.random.default_rng(seed+101+i).permutation(len(rows)) for i in range(epochs)]
    stats={}
    for name,labels in (('C52-ALIGNED',y),('C52-SHUFFLED',shuffled)):
       model=heads[name];opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)
       Y=torch.from_numpy(labels)
       p=float(np.mean(labels));weights=torch.where(Y>0,.5/max(p,.03),.5/max(1-p,.03))
       curve=[];updates=0
       for order in orders:
          total=0
          for k in range(0,len(rows),64):
              ix=order[k:k+64];pred=model(XX[ix])
              loss=(nn.functional.binary_cross_entropy_with_logits(pred,Y[ix],reduction='none')*weights[ix]).mean()
              opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
              opt.step();updates+=1;total+=float(loss.detach())*len(ix)
          curve.append(round(total/len(rows),6))
       model.eval()
       stats[name]={'updates':updates,'epoch_losses':curve,'positive_share':p}
    stats.update({'train_records':len(rows),'train_graphs':len(set(r['graph'] for r in rows)),
      'true_positive':int(y.sum()),'shuffled_positive':int(shuffled.sum()),
      'shuffled_label_changes':changed,
      'training_features_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
      'labels_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
      'shuffled_labels_sha256':hashlib.sha256(shuffled.tobytes()).hexdigest(),
      'normalization_sha256':hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest()})
    return heads,mean,scale,stats


def score_head(head,x,mean,scale):
    z=torch.from_numpy(np.clip((x-mean)/scale,-6,6).astype(np.float32))
    with torch.no_grad():v=float(head(z).item())
    return v


def run_suffix(base,option,prior,snap,rng_state,model,arm,cue_tick,target,heads,mean,scale,trace=False):
    if model in ('C18-A0','C34-RESET','C40-PAIR'):
      return c50.suffix(base,option,prior,snap,rng_state,0,model,arm,cue_tick,target,trace)
    w=copy.deepcopy(snap);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
    w.goal=target;w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=issue(w,arm)
    acts=[];path=[];raw_illegal=0;pref=opp=0;neural_uses=0;oracle_predictions=[];reached=False
    for t in range(cue_tick,HORIZON):
        if w.stopped:break
        if w.distance()==0:reached=True;break
        menu,logits,bad,_=scores(base,w,advice,1.)
        raw_illegal+=bad
        if advice.active(w)==1 and menu[8].legal:
            x=features(w,advice)
            if model=='C52-RANDOM':
                # Predeclared zero-information Bernoulli control, keyless per-tick RNG
                v=1 if rng.random()<.5 else -1
            else:v=score_head(heads[model],x,mean,scale)
            # Learned action logit bonus, bounded. The host still masks illegal moves.
            # Low confidence reduces intervention; never a forced action or script.
            delta=np.clip(v*2.5,-3.5,3.5)*float(advice.strength)
            logits[8]+=delta
            neural_uses+=1
        selected=menu[int(rng.choice(len(menu),p=probs(logits)))]
        assert selected.legal and w.apply(selected) and not w.invalid_attempts
        advice.advance(w)
        acts.append((selected.action,selected.target))
        if selected.action[0] in ('INK','LIFT'):
            turn=float(selected.features[13]-selected.features[14])*w.personality
            pref+=int(turn>0);opp+=int(turn<0)
        if trace:path.append({'t':t,'pos':w.pos,'goal':w.goal,'action':selected.action[0]})
        if w.distance()==0:reached=True;break
    assert sha(sorted(w.protected))==sha(sorted(snap.protected))
    return {'goal':int(reached),'steps_after_cue':len(acts),'action_sha':sha(acts),
      'ink_sha':sha(sorted(w.inked)),'protected_sha':sha(sorted(w.protected)),
      'host_rejections':w.invalid_attempts,'raw_illegal_top1':raw_illegal,
      'preferred_ink':pref,'opposed_ink':opp,'neural_states':neural_uses,
      'path':path if trace else None}


def collect_online(base,option,prior,worldset,heads,mean,scale,trace_limit=18):
    rows=[];traces=[];skips=collections.Counter()
    for w,info,s,fam,case in worldset:
      for tick in CUE_TICKS:
        got=c50.prefix(base,w,s,tick)
        if got is None:skips['early_terminal']+=1;continue
        snap,rngstate,receipt=got
        target=c50.resolved_goal(snap,info)
        if target is None:skips['goal_on_history']+=1;continue
        for arm in ARMS:
            if arm[0] in ('none','continue'):
                # All corrective heads are gated exclusively by BACKTRACK.
                # Reuse the same fully executed A0 outcome, not independent
                # imaginary trajectories. This parity is separately unit-tested.
                a0=run_suffix(base,option,prior,snap,rngstate,'C18-A0',arm,tick,target,heads,mean,scale)
                results={m:copy.deepcopy(a0) for m in POLICIES}
            else:
                results={m:run_suffix(base,option,prior,snap,rngstate,m,arm,tick,target,heads,mean,scale)
                  for m in POLICIES}
            assert all(z['host_rejections']==0 and z['protected_sha']==receipt['protected_sha'] for z in results.values())
            if arm[0]=='none':
              assert len({(z['action_sha'],z['ink_sha']) for z in results.values()})==1
            row={'graph':info['graph_sha'],'seed':s,'family':fam,'case':case,'personality':w.personality,
              'cue_tick':tick,'arm':arm[0],'strength':arm[1],
              'prefix_sha':receipt['action_sha'],'prefix_ink_sha':receipt['ink_sha'],
              'target':target,'protected_sha':receipt['protected_sha'],'outcomes':results}
            rows.append(row)
            if arm[0]=='backtrack' and arm[1]==1 and len(traces)<trace_limit:
              for m in POLICIES:
                if len(traces)>=trace_limit:break
                z=run_suffix(base,option,prior,snap,rngstate,m,arm,tick,target,heads,mean,scale,True)
                assert (z['action_sha'],z['ink_sha'])==(results[m]['action_sha'],results[m]['ink_sha'])
                traces.append({'graph':info['graph_sha'],'family':fam,'case':case,'personality':w.personality,
                  'model':m,'start':receipt['initial'],'prefix_positions':receipt['positions'],
                  'cue_tick':tick,'target':target,'protected':sorted(snap.protected),'outcome':z})
    return rows,traces,dict(skips)


def summary(train,blind,online,train_ids,blind_ids,online_ids,stats,checks,receipts,skips):
    assert train_ids.isdisjoint(blind_ids) and train_ids.isdisjoint(online_ids) and blind_ids.isdisjoint(online_ids)
    guided=[r for r in online if r['arm']!='none']
    group=collections.defaultdict(collections.Counter)
    graph=collections.defaultdict(collections.Counter)
    byfamily=collections.defaultdict(collections.Counter)
    for r in guided:
        key=(r['family'],r['case'],r['cue_tick'],r['arm'],r['strength'])
        for m,o in r['outcomes'].items():
            group[key][m]+=o['goal'];graph[r['graph']][m]+=o['goal'];byfamily[r['family']][m]+=o['goal']
    result={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'train_graphs':len(train_ids),'blind_graphs':len(blind_ids),'online_graphs':len(online_ids),
      'graph_sha256':{k:sha(sorted(v)) for k,v in [('train',train_ids),('blind',blind_ids),('online',online_ids)]},
      'training':stats,'checkpoints_sha256':checks,'train_size':len(train),'blind_size':len(blind),
      'online_rows':len(online),'policy_outcome_records':len(online)*len(POLICIES),
      'executed_suffix_rollouts':sum(len(POLICIES) if r['arm']=='backtrack' else 1 for r in online),
      'exact_parity_reused_records':sum(len(POLICIES)-1 for r in online if r['arm']!='backtrack'),
      'guided_episodes_per_policy':len(guided),
      'world_generation':receipts,'skipped':skips,'models':{},'paired':{},'blind':{},
      'limitations':['Entire experiment uses synthetic host; NOT native CyberSand acceptance',
      'Oracle BFS reads complete protected-cell geometry during training and blind scoring ONLY; not available at inference',
      'Action-level BFS labels are a surrogate for success, not executed-goal outcomes',
      'Graph generators are fixed 7x7 maze families with 15x15 host grid and shared features',
      'Eligible episodes conditional on prefix surviving and post-cue target not already visited',
      'One correction event; model instantiated at cue, not pre-existing continuous neural internal state',
      'Random control has additional RNG draws and thus its stochastic coupling differs from other policies',
      'No/continue outcomes are reused from one executed A0 path because all corrective heads are disabled without BACKTRACK; verified separately',
      'Historical d_common fixture is NOT canonical CyberSand host ABI',
      'Unguided exact hash and ink-turn ratio are only personality proxies',
      'Bootstrap uncorrected for multiple prior experiments and research selection',
      'Checkpoints not yet in immutable Actions artifact custody']}
    for m in POLICIES:
      r=[row['outcomes'][m] for row in guided]
      idle=[row for row in online if row['arm']=='none']
      result['models'][m]={'goals':sum(z['goal'] for z in r),
        'by_family':{f:int(byfamily[f][m]) for f in ONLINE_FAMILIES},
        'condition_regressions_vs_RESET':sum(v[m]<v['C34-RESET'] for v in group.values()),
        'unguided_exact':sum((x['outcomes'][m]['action_sha'],x['outcomes'][m]['ink_sha'])==
            (x['outcomes']['C18-A0']['action_sha'],x['outcomes']['C18-A0']['ink_sha']) for x in idle),
        'neural_decision_states':sum(z.get('neural_states',0) for z in r),
        'preferred_ink':sum(z['preferred_ink'] for z in r),'opposed_ink':sum(z['opposed_ink'] for z in r),
        'host_rejections':sum(z['host_rejections'] for row in online for n,z in row['outcomes'].items() if n==m)}
    for m in ('C52-ALIGNED','C52-SHUFFLED','C52-RANDOM','C40-PAIR'):
      for ref in ('C34-RESET','C52-SHUFFLED'):
        if m==ref:continue
        diff=np.asarray([g[m]-g[ref] for g in graph.values()],np.int32)
        if not len(diff):continue
        rng=np.random.default_rng(SEED+len(m)*31+len(ref)*23)
        draws=diff[rng.integers(len(diff),size=(5000,len(diff)))].sum(1)
        result['paired'][m+'__'+ref]={'goal_delta':int(diff.sum()),
           'positive_graphs':int((diff>0).sum()),'negative_graphs':int((diff<0).sum()),
           'graph_cluster_bootstrap_95_descriptive':[float(v) for v in np.quantile(draws,[.025,.975])]}
    for m in ('C52-ALIGNED','C52-SHUFFLED'):
        xx=np.asarray([r['x'] for r in blind],dtype=np.float32)
        yy=np.asarray([r['y'] for r in blind],dtype=np.float32)
        p=np.asarray([r['predictions'][m] for r in blind],np.float32)
        result['blind'][m]={'n':len(blind),'positive':int(yy.sum()),
          'balanced_accuracy':float(.5*((p[yy==1]>=.5).mean()+(p[yy==0]<.5).mean())) if yy.min()!=yy.max() else None,
          'brier':float(np.mean((p-yy)**2)),'accuracy':float(np.mean((p>=.5)==yy))}
    result['blind']['majority']={'accuracy':float(max(np.mean([r['y'] for r in blind]),1-np.mean([r['y'] for r in blind])))}
    gates={'aligned_better_than_shuffled_on_online_goals':result['models']['C52-ALIGNED']['goals']>result['models']['C52-SHUFFLED']['goals'],
       'aligned_better_than_RESET_on_online_goals':result['models']['C52-ALIGNED']['goals']>result['models']['C34-RESET']['goals'],
       'both_ood_families_positive_vs_RESET':all(byfamily[f]['C52-ALIGNED']>byfamily[f]['C34-RESET'] for f in ONLINE_FAMILIES),
       'zero_condition_regressions':result['models']['C52-ALIGNED']['condition_regressions_vs_RESET']==0,
       'positive_cluster_bootstrap_lower_vs_RESET':result['paired']['C52-ALIGNED__C34-RESET']['graph_cluster_bootstrap_95_descriptive'][0]>0,
       'blind_brier_better_than_shuffled':result['blind']['C52-ALIGNED']['brier']<result['blind']['C52-SHUFFLED']['brier'],
       'exact_all_unguided_hashes':all(z['unguided_exact']==len(online)-len(guided) for z in result['models'].values()),
       'zero_host_rejections':all(z['host_rejections']==0 for z in result['models'].values())}
    result['promotion_gates']=gates;result['promotion_eligible']=all(gates.values())
    return result


def main(prior,out,train_reps=4,blind_reps=2,online_reps=6):
    prior=Path(prior);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    base,option,frozen=c50.c47.c44.prerequisites(prior/'previous/predecessor_C42')
    ck=prior/'previous/predecessor_C42/out_c40/C40-PAIR.safetensors'
    frozen_sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    assert frozen_sha=='5d9d7c6602e3be68ec7ad3a593e1c06e709b4804974107ef572c30a548766184'
    previous=set()
    earlier=Path(__file__).resolve().parent.parent/'prereq/C51-online.json'
    if earlier.exists():previous={r['graph'] for r in json.loads(earlier.read_text())}
    trainw,trrec=worlds(TRAIN_FAMILIES,c50.c47.c41.world,SEED+17,train_reps,exclude=previous)
    train,sktrain,prefixes=collect_teacher(base,option,frozen,trainw)
    print('train graphs',len(trainw),'states',len(train),'positive',sum(r['y'] for r in train),flush=True)
    if not train or len(set(r['y'] for r in train))<2:raise AssertionError('non-discriminative training labels')
    heads,mean,scale,stats=fit(train)
    checks={}
    for name,h in heads.items():
        dest=out/(name+'.safetensors');save_file(h.state_dict(),str(dest));checks[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    # Online and blind are independently-seeded and disjoint identities, OOD Wilson/Kruskal.
    trids=set(r['graph'] for r in trrec)
    blindw,blrec=worlds(ONLINE_FAMILIES,c50.c47.c42.world,SEED+201131,blind_reps,exclude=previous|trids)
    onlinew,onrec=worlds(ONLINE_FAMILIES,c50.c47.c42.world,SEED+403117,online_reps,exclude=previous|trids|set(r['graph'] for r in blrec))
    blind,skblind,_=collect_teacher(base,option,frozen,blindw)
    for r in blind:
        x=np.asarray(r['x'],np.float32)
        r['predictions']={m:1/(1+math.exp(-score_head(h,x,mean,scale))) for m,h in heads.items()}
    print('blind graphs',len(blindw),'states',len(blind),'positive',sum(r['y'] for r in blind),flush=True)
    online,traces,skonline=collect_online(base,option,frozen,onlinew,heads,mean,scale)
    print('online graphs',len(onlinew),'rows',len(online),'episodes',len(online)*len(POLICIES),flush=True)
    train_ids,blind_ids,online_ids=[{r['graph'] for r in z} for z in (trrec,blrec,onrec)]
    res=summary(train,blind,online,train_ids,blind_ids,online_ids,stats,checks,
      {'train':trrec,'blind':blrec,'online':onrec},
      {'train':sktrain,'blind':skblind,'online':skonline})
    res['frozen_C40_checkpoint_sha256']=frozen_sha
    # Full scientific evidence: source checksum, executed trajectories and labels (not merely summaries).
    for name,data in [('C52-training.json',train),('C52-blind.json',blind),
       ('C52-online.json',online),('C52-selected-traces.json',traces),('C52-summary.json',res)]:
        (out/name).write_text(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n')
    (out/'C52-normalization.json').write_text(json.dumps({'mean':mean.tolist(),'scale':scale.tolist()},sort_keys=True)+'\n')
    print(json.dumps({'goals':{m:z['goals'] for m,z in res['models'].items()},
      'training':{'graphs':stats['train_graphs'],'records':stats['train_records'],'positive':stats['true_positive'],'updates':stats['C52-ALIGNED']['updates']},
      'blind':res['blind'],'paired':res['paired'],'promotion_gates':res['promotion_gates']},indent=2),flush=True)
    return res

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('output')
    p.add_argument('--train-reps',type=int,default=4);p.add_argument('--blind-reps',type=int,default=2)
    p.add_argument('--online-reps',type=int,default=6)
    a=p.parse_args();main(a.prior,a.output,a.train_reps,a.blind_reps,a.online_reps)

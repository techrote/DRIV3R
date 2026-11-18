"""C60: online, host-legal remote-goal exploration with bounded policy-gradient training.

External synthetic CyberSand-like host only, NOT the production ABI. Original
C59 and predecessors imported unchanged. No hidden graph, protected-cell map,
shortest-path oracle, or future rollouts in policy features or action selection.
"""
from __future__ import annotations
import collections, copy, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file
from experiment import sha
from c28_temporal_option import scores, probs
from c34_independent_maze import ReArmedWatch
import c59_eligibility_personality as c59

SEED=600118
HORIZON=78
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0., .5)
POLICIES=('C18-A0','C34-RESET','C60-NOVELTY','C60-FRONTIER','C60-ALIGNED','C60-SHUFFLED')
NFEAT=12
BONUS=1.8


def action_features(w,ledger,menu):
    """Only CURRENT legal candidates, visited menus/actions, known host goal.

    No access to world.protected, world.allowed, hidden edges or future state.
    Goal distance uses host-published position, not geodesic maze distance.
    """
    d=float(max(1,w.distance())); here=tuple(w.pos)
    recent=[tuple(a[1]) for a in ledger.actions[-10:]]
    output=[]
    for c in menu:
        k=c.action[0]; p=tuple(c.target)
        move=k in ('INK','LIFT','BACKTRACK') and p!=here
        dx=p[0]-here[0];dy=p[1]-here[1]
        dist=abs(p[0]-w.goal[0])+abs(p[1]-w.goal[1])
        turn=w.last_dir[0]*dy-w.last_dir[1]*dx
        # Limited to the host currently offered action and history; never
        # inspect neighbors of a remote cell from the full simulator graph.
        f=[int(move),int(p not in ledger.visits),
           min(1.,ledger.visits[p]/6.), int(p in recent),
           np.clip((d-dist)/3.,-1,1),
           int(k=='BACKTRACK'),int(k=='STOP'),int(k=='INK'),
           int(turn*w.personality>0),int(turn*w.personality<0),
           min(1.,ledger.visits[here]/6.),
           min(1.,len(ledger.frontiers())/16.)]
        output.append(f)
    return np.asarray(output,np.float32)


def softmax(x):
    a=np.asarray(x,np.float64);a-=a.max();z=np.exp(a);return z/z.sum()


def explored_route(w,ledger):
    """Plan toward a genuinely observed but unvisited frontier only."""
    candidates=[]
    for p in ledger.frontiers():
        r=ledger.route(w.pos,p)
        if r is not None and len(r)>1:
            candidates.append((len(r)-1 + .25*w.distance(p),p,r))
    return min(candidates,key=lambda r:(r[0],r[1])) if candidates else None


def exploration_delta(w,ledger,menu,features,policy,weights,strength):
    if policy=='C60-NOVELTY':
        return strength*(2.0*features[:,1]-.8*features[:,2] + .3*features[:,4])*features[:,0]
    if policy=='C60-FRONTIER':
        # Soft multi-step option, not a forced target or hidden graph path.
        dest=explored_route(w,ledger)
        if dest:
            next_pos=tuple(dest[2][1]);delta=np.asarray([int(tuple(c.target)==next_pos and c.action[0] in ('INK','LIFT','BACKTRACK')) for c in menu],np.float32)
            return strength*2.2*delta
        return strength*(1.8*features[:,1]-.6*features[:,2])*features[:,0]
    if policy in ('C60-ALIGNED','C60-SHUFFLED'):
        return strength*BONUS*np.clip(features@weights,-1.0,1.0)
    return np.zeros(len(menu),np.float32)


def execute(base,opt,st,arm,policy,weights,drop=0.,collect=False,trace=False):
    snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c59.c58.c55.cue(w,arm)
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rngstate)
    led=c59.c58.damaged(log,info['graph_sha'],tick,drop)
    ctrl=None if policy=='C18-A0' or arm[0]=='none' else ReArmedWatch(opt,None)
    sampled=[];acts=[];points=[tuple(w.pos)];path=[]
    counts=collections.Counter();visited=set(led.visits)
    # Set of previously reached positions is not a secret: prefix host visits only.
    for t in range(tick,HORIZON):
        if w.stopped or w.distance()==0:break
        led.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        fs=action_features(w,led,menu)
        bs=np.asarray(logits,np.float64)
        if ctrl is not None and policy not in ('C18-A0','C34-RESET'):
            delta=exploration_delta(w,led,menu,fs,policy,weights,arm[1])
            bs+=delta
            counts['biased_ticks']+=int(np.any(delta!=0))
        prob=softmax(bs)
        j=int(rng.choice(len(menu),p=prob))
        c=menu[j];before=tuple(w.pos);prior_dist=w.distance()
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        led.after(before,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w)
        acts.append((c.action,c.target));points.append(tuple(w.pos))
        newly=tuple(w.pos) not in visited
        visited.add(tuple(w.pos))
        counts['new_sites']+=int(newly);counts['revisits']+=int(not newly)
        goal=int(w.distance()==0)
        # Local rewards depend only on executed transitions and published goal;
        # future outcomes influence returns but are never runtime observations.
        reward= .65*newly+.12*np.clip(prior_dist-w.distance(),-1,1)+8.*goal-.025
        if c.action[0]=='STOP' and not goal:reward-=1.
        if tuple(w.pos)==before:reward-=.06
        if collect:
            sampled.append({'features':fs,'logits':np.asarray(logits,np.float32),
              'action':j,'reward':float(reward),'graph':info['graph_sha'],
              'family':fam,'strength':arm[1]})
        if trace:path.append({'t':t,'p':list(w.pos),'a':c.action[0]})
    return {'goal':int(w.distance()==0),'steps':len(acts),'distance':int(w.distance()),
      'unique_sites':len(visited),'new_sites':counts['new_sites'],
      'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),'protected_sha':sha(sorted(w.protected)),
      'host_rejections':w.invalid_attempts,'counts':dict(counts),
      'personality20':c59.movement_metrics(points,acts),
      'start_pos':list(snap.pos),'target':list(target),
      'path':path if trace else None},sampled


def policy_fit(episodes,base,opt,mode):
    """Genuine on-policy optimizer: CPU AdamW actor update with delayed return.

    Matched graph-shuffle negative control replaces reward targets within an
    epoch while retaining each action's observed feature stream. This is NOT
    imitation of an oracle route and never uses held-out graph outcomes.
    """
    torch.manual_seed(SEED+77)
    model=nn.Linear(NFEAT,1,bias=False);nn.init.zeros_(model.weight)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.035,weight_decay=.015)
    rng=np.random.default_rng(SEED+(0 if mode=='aligned' else 1009))
    updates=0;seen=0;goals=0;stats=[]
    for epoch in range(4):
        order=rng.permutation(len(episodes));batch=[]
        for idx in order:
            st,arm=episodes[int(idx)];weights=model.weight.detach().numpy().reshape(-1).copy()
            outcome,dec=execute(base,opt,st,arm,'C60-ALIGNED',weights,0.,collect=True)
            if not dec:continue
            rewards=np.asarray([r['reward'] for r in dec],np.float32)
            returns=np.zeros(len(dec),np.float32);carry=0.
            for i in range(len(dec)-1,-1,-1):
                carry=rewards[i]+.92*carry;returns[i]=carry
            batch.append((dec,returns,st[6]['graph_sha']))
            goals+=outcome['goal'];seen+=1
        if mode=='shuffled':
            # Shuffling within the epoch across independent graph instances,
            # deliberately corrupts delayed return/action associations.
            all_returns=np.concatenate([ret for _,ret,_ in batch]);rng.shuffle(all_returns)
            offset=0;tmp=[]
            for dec,ret,graph in batch:
                n=len(ret);tmp.append((dec,all_returns[offset:offset+n].copy(),graph));offset+=n
            batch=tmp
        flat=[(decision,float(ret)) for decisions,rr,_ in batch for decision,ret in zip(decisions,rr)]
        if not flat:raise AssertionError('no actual training decisions')
        # Standardize advantage only from the training epoch, not held-out data.
        values=np.asarray([rr for _,rr in flat],np.float64)
        center,spread=float(np.mean(values)),max(float(np.std(values)),1e-4)
        for batchstart in range(0,len(flat),512):
            data=flat[batchstart:batchstart+512]
            feats=torch.from_numpy(np.stack([e[0]['features'] for e in data]))
            frozen=torch.from_numpy(np.stack([e[0]['logits'] for e in data]))
            acts=torch.tensor([e[0]['action'] for e in data],dtype=torch.long)
            adv=torch.tensor([(r-center)/spread for _,r in data],dtype=torch.float32)
            strength=torch.tensor([e[0]['strength'] for e in data],dtype=torch.float32)
            # clip bounded impact of neural head: prevents deterministic puppetry
            added=BONUS*strength[:,None]*torch.clamp(model(feats).squeeze(-1),-1.,1.)
            logits=frozen+added
            logp=torch.log_softmax(logits,dim=1)
            probs_now=torch.softmax(logits,dim=1)
            entropy=-(probs_now*logp).sum(-1).mean()
            baseline_log=torch.log_softmax(frozen,dim=1)
            kl=(probs_now*(logp-baseline_log)).sum(-1).mean()
            pg=-(logp.gather(1,acts[:,None]).squeeze(1)*adv).mean()
            loss=pg + .04*kl - .008*entropy
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.8);optimizer.step()
            updates+=1
        stats.append({'epoch':epoch,'trajectories':len(batch),'actions':len(flat),
           'mean_return':float(values.mean()),'goal_completions_cumulative':goals,
           'weight_l2':float(torch.linalg.vector_norm(model.weight).item())})
    return model,{'mode':mode,'updates':updates,'episodes':seen,'goals_on_training_rollouts':goals,'epochs':stats}


def worlds():
    c59.c58.c55.TICKS=(9,)
    fn=c59.c58.c55.c54.c52.worlds
    fam=c59.c58.c55.c54.c52.TRAIN_FAMILIES
    ood=c59.c58.c55.c54.c52.ONLINE_FAMILIES
    basefn=c59.c58.c55.c54.c52.c50.c47.c41.world
    oodfn=c59.c58.c55.c54.c52.c50.c47.c42.world
    # Explicitly frozen historical geometry exclusion set, committed within
    # this laboratory. No reliance on any sibling worker's mutable files.
    exclusion_path=Path(__file__).with_name('C60-historical-graph-exclusions.json')
    snapshot=json.loads(exclusion_path.read_text())
    assert snapshot['schema']=='driv3r-c60-graph-exclusions/1'
    old=set(snapshot['excluded_graphs'])
    assert len(old)==len(snapshot['excluded_graphs']) and len(old)>=60
    tr,trec=fn(fam,basefn,SEED+31111,1,exclude=old)
    ex=old|{r['graph'] for r in trec}
    un,unrec=fn(ood,oodfn,SEED+91111,2,exclude=ex)
    fresh,frecrec=fn(fam,basefn,SEED+81111,1,exclude=ex|{r['graph'] for r in unrec})
    return (tr,trec),(un+fresh,unrec+frecrec),old


def run(prior,output):
    prior=Path(prior);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base,opt,_=c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    (trainw,trrec),(onlinew,onrec),history=worlds()
    trainst,sktrain=c59.c58.c56.states_for_goal(base,trainw,'remote')
    onst,skon=c59.c58.c56.states_for_goal(base,onlinew,'remote')
    assert len(trainst)>8 and len(onst)>16 and not {x[6]['graph_sha'] for x in trainst}&{x[6]['graph_sha'] for x in onst}
    episodes=[(s,arm) for s in trainst for arm in ARMS[1:]]
    aligned,fit=policy_fit(episodes,base,opt,'aligned')
    shuffled,fitsh=policy_fit(episodes,base,opt,'shuffled')
    rng=np.random.default_rng(SEED+17)
    random=np.asarray(rng.normal(0,.18,NFEAT),np.float32)
    weights={'C60-ALIGNED':aligned.weight.detach().numpy().reshape(-1).copy(),
      'C60-SHUFFLED':shuffled.weight.detach().numpy().reshape(-1).copy(),
      'C60-RANDOM':random}
    checkpoints={}
    for mode,model in [('ALIGNED',aligned),('SHUFFLED',shuffled)]:
        p=output/f'C60-{mode}.safetensors';save_file(model.state_dict(),str(p))
        checkpoints[mode]=hashlib.sha256(p.read_bytes()).hexdigest()
    records=[];traces=[];ag=collections.defaultdict(collections.Counter)
    personality=collections.defaultdict(list);parity=0
    for st in onst:
        graph=st[6]['graph_sha'];fam=st[8]
        for drop in DROPS:
            for arm in ARMS:
                results={}
                for pol in POLICIES:
                    z,_=execute(base,opt,st,arm,pol,weights.get(pol,np.zeros(NFEAT,np.float32)),drop,trace=False)
                    results[pol]=z
                    if arm[0]!='none':
                        agg=ag[(fam,drop,pol)];agg['episodes']+=1;agg['goals']+=z['goal'];agg['new_sites']+=z['new_sites']
                        if z['personality20'] is not None:
                            personality[(graph,drop,arm[1],pol)].append(z['personality20'])
                assert all(z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha'] for z in results.values())
                if arm[0]=='none':
                    assert len({(z['action_sha'],z['ink_sha']) for z in results.values()})==1
                    parity+=1
                records.append({'graph':graph,'family':fam,'case':st[9],'seed':st[7],
                    'tick':st[4],'target':list(st[5]),'drop':drop,'arm':arm,
                    'prefix_sha':st[3]['prefix_sha'],'prefix_ink_sha':st[3]['ink_sha'],
                    'outcomes':results})
                if arm==('backtrack',1.) and drop in (0.,.5) and len(traces)<20:
                    for pol in ('C34-RESET','C60-FRONTIER','C60-ALIGNED','C60-SHUFFLED'):
                        if len(traces)>=20:break
                        z,_=execute(base,opt,st,arm,pol,weights.get(pol,np.zeros(NFEAT,np.float32)),drop,trace=True)
                        assert (z['action_sha'],z['ink_sha'])==(results[pol]['action_sha'],results[pol]['ink_sha'])
                        traces.append({'graph':graph,'family':fam,'policy':pol,'drop':drop,
                          'seed':st[7],'case':st[9],'prefix':st[3],'target':list(st[5]),'outcome':z})
    comparisons={}
    for drop in DROPS:
        for fam in sorted({s[8] for s in onst}):
            rs=[r for r in records if r['drop']==drop and r['family']==fam and r['arm'][0]!='none']
            if not rs:continue
            for pol in POLICIES:
                comparisons[str((drop,fam,pol))]={'goals':sum(r['outcomes'][pol]['goal'] for r in rs),'episodes':len(rs),
                'paired_wins':sum(r['outcomes'][pol]['goal']>r['outcomes']['C34-RESET']['goal'] for r in rs),
                'paired_losses':sum(r['outcomes'][pol]['goal']<r['outcomes']['C34-RESET']['goal'] for r in rs)}
    deltas={}
    for pol in POLICIES[2:]:
        for drop in DROPS:
            bygraph=collections.defaultdict(int)
            for r in records:
                if r['drop']==drop and r['arm'][0]!='none':
                    bygraph[r['graph']]+=r['outcomes'][pol]['goal']-r['outcomes']['C34-RESET']['goal']
            ds=np.asarray(list(bygraph.values()),np.int32)
            rng=np.random.default_rng(SEED+int(100*drop)+len(pol))
            bootstrap=ds[rng.integers(0,len(ds),size=(2000,len(ds)))].sum(axis=1)
            deltas[str((pol,drop))]={'difference':int(ds.sum()),'positive_graphs':int((ds>0).sum()),
              'negative_graphs':int((ds<0).sum()),'exploratory_cluster95':np.quantile(bootstrap,[.025,.975]).tolist()}
    psych={}
    for pol in POLICIES[2:]:
        metrics=collections.defaultdict(list)
        for (graph,drop,strength,which),v in personality.items():
            if which!=pol:continue
            ctl=personality.get((graph,drop,strength,'C34-RESET'))
            if not ctl:continue
            for k in ('turn_rate','reverse_rate','unique_ratio','direction_entropy','paint_fraction'):
                metrics[(drop,k)].append(v[0][k]-ctl[0][k])
        psych[pol]={str(k):{'n':len(v),'mean_delta':float(np.mean(v))} for k,v in metrics.items()}
    summary={'schema':'driv3r-c60-online-remote-frontier-rl/1','seed':SEED,
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'historical_excluded_graphs':len(history),'training_graphs':len(trrec),
      'online_graphs':len(onrec),'eligible_train_graphs':len(trainst),
      'eligible_online_graphs':len(onst),'training_skips':sktrain,'online_skips':skon,
      'training':fit,'shuffled_control_training':fitsh,'checkpoint_sha256':checkpoints,
      'weights':{k:v.tolist() for k,v in weights.items()},
      'records':len(records),'policy_suffixes':len(records)*len(POLICIES),
      'unguided_parity_conditions':parity,'replayed_traces':len(traces),
      'aggregate':{str(k):dict(v) for k,v in ag.items()},'comparisons':comparisons,
      'clustered':deltas,'personality20':psych,
      'cohorts':{'train':trrec,'online':onrec},
      'caveats':['Synthetic host only; not native CyberSand','Previously unseen remote goal known by coordinate but hidden maze inaccessible',
      'Policy-gradient reward is shaped for legal novelty and goal progress; not an optimal-control oracle',
      'First-20 movement proxies do not establish artistic/meandering personality acceptance',
      'Exploratory intervals are unadjusted after adaptive prior C-series research',
      'No immutable Actions checkpoint custody; checkpoints retained only in local owner archive']}
    for name,obj in [('C60-summary.json',summary),('C60-online.json',records),('C60-traces.json',traces)]:
        (output/name).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'graphs':len(onrec),'eligible':len(onst),'train':len(trainst),'suffixes':len(records)*len(POLICIES),
       'parity':parity,'fits':(fit['updates'],fitsh['updates']),
       'goals':{str((fam,dr,pol)):c['goals'] for (fam,dr,pol),c in ag.items() if dr==.5}},indent=2),flush=True)
    return summary

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('prior');parser.add_argument('output')
    args=parser.parse_args();run(args.prior,args.output)

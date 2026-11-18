"""C56: controlled host-observed memory erasure and recovery transfer.

External synthetic-host science, not CyberSand's native ABI. All policy actions
come from fresh host-legal menus and go through World.apply; test goals are
host-assigned only. Memory loss is pre-registered and policy-independent.
"""
from __future__ import annotations
import collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
import c55_observed_waypoint_distillation as c55
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import scores, probs
from c34_independent_maze import ReArmedWatch

SEED = 560118
LEVELS = (0., .50)
POLICIES = ('C18-A0','C34-RESET','C55-ROUTE','C55-ALIGNED',
            'C56-AUGMENTED','C56-SHUFFLED')
HORIZON = 78


def erase(log, graph, tick, drop):
    """Deterministically forget complete *previously observed* local menus.

    Leaves current node available for contemporary host re-observation. Never
    invents a legal connection; deletion alone cannot expose hidden geometry.
    """
    ledger = copy.deepcopy(log)
    if drop == 0.: return ledger
    keep = {}
    for p, nbrs in ledger.neighbors.items():
        if ledger.actions and tuple(p) == tuple(ledger.actions[-1][1]):
            keep[p] = nbrs; continue
        h = hashlib.sha256(f'{SEED}/{graph}/{tick}/{p[0]},{p[1]}'.encode()).digest()
        uniform = int.from_bytes(h[:8],'big') / (1<<64)
        if uniform >= drop:keep[p] = nbrs
    ledger.neighbors = keep
    # Recorded visits and action history remain; only past local menus are lost.
    return ledger


def next_step(w, log, advice, menu):
    if advice.active(w)!=1:return ([],None)
    route=log.route(w.pos,w.goal)
    if not route or len(route)<2:return ([],None)
    nxt=route[1]
    legal=[i for i,c in enumerate(menu[:8]) if c.legal and tuple(c.target)==nxt]
    return legal,nxt


def augmented_teacher(base, option, states):
    rows=[];counts=collections.Counter()
    for snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance in states:
      for drop in LEVELS:
        for arm in c55.ARMS[1:]:
          w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
          advice=c55.cue(w,arm)
          rr=np.random.default_rng();rr.bit_generator.state=copy.deepcopy(rngstate)
          ledger=erase(log,info['graph_sha'],tick,drop)
          ctrl=ReArmedWatch(option,None)
          for j in range(9):
            if w.stopped or w.distance()==0:break
            menu=ledger.see(w);legal,nxt=next_step(w,ledger,advice,menu)
            counts['visited_states']+=1
            if legal:
              counts['reachable_teacher_states']+=1
              for k,c in enumerate(menu[:8]):
                if c.legal:
                  x=c55.feature(w,ledger,menu,c,nxt)
                  rows.append({'graph':info['graph_sha'],'family':fam,'case':case,
                    'tick':tick,'arm':arm[1],'drop':drop,'j':j,'x':x.tolist(),
                    'y':int(k in legal), 'state':sha([w.pos,w.goal,ledger.actions[-5:],w.stuck,drop]),
                    'kind':c.action[0]})
            menu,logits,_,_=ctrl.modify(base,w,advice,1.)
            c=menu[int(rr.choice(len(menu),p=probs(logits)))]; start=tuple(w.pos)
            assert c.legal and w.apply(c) and not w.invalid_attempts
            ledger.after(start,c);ctrl.after(c);advice.advance(w)
    return rows,dict(counts)


def rollout(base,option,snap,rngstate,log,tick,target,arm,policy,heads,graph,drop,trace=False):
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c55.cue(w,arm)
    r=np.random.default_rng();r.bit_generator.state=copy.deepcopy(rngstate)
    ledger=erase(log,graph,tick,drop)
    ctrl=None if policy=='C18-A0' or arm[0]=='none' else ReArmedWatch(option,None)
    visited=[];actions=[];guided=followed=available=missing=preferred=opposed=0
    path=[]
    for t in range(tick,HORIZON):
        if w.stopped or w.distance()==0:break
        ledger.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        legal,nxt=next_step(w,ledger,advice,menu)
        if advice.active(w)==1:
            if legal: available+=1
            else: missing+=1
        if legal and policy not in ('C18-A0','C34-RESET'):
            guided+=1
            if policy=='C55-ROUTE':
                delta=np.zeros(len(menu),np.float32);delta[legal]=4.2*arm[1]
            else:
                with torch.no_grad():
                    xs=np.asarray([c55.feature(w,ledger,menu,c,nxt) for c in menu[:8]],np.float32)
                    z=heads[policy](torch.from_numpy(xs)).numpy()
                delta=np.zeros(len(menu),np.float32)
                delta[:8]=4.2*arm[1]*(2/(1+np.exp(-np.clip(z,-20,20)))-1)
            logits=np.asarray(logits,dtype=np.float32)+delta
        c=menu[int(r.choice(len(menu),p=probs(logits)))]; start=tuple(w.pos)
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        ledger.after(start,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w)
        actions.append((c.action,c.target));visited.append(tuple(w.pos))
        followed+=int(bool(legal) and tuple(c.target)==nxt)
        if c.action[0] in ('INK','LIFT'):
            turn=float(c.features[13]-c.features[14])*w.personality
            preferred+=int(turn>0);opposed+=int(turn<0)
        if trace:path.append({'t':t,'p':list(w.pos),'a':c.action[0],'route':int(bool(legal))})
    assert sha(sorted(w.protected))==sha(sorted(snap.protected))
    return {'goal':int(w.distance()==0),'steps':len(actions), 'final_distance':w.distance(),
        'action_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
        'protected_sha':sha(sorted(w.protected)),'host_rejections':w.invalid_attempts,
        'guided_turns':guided,'followed':followed,'route_available':available,
        'route_missing':missing,'preferred_ink':preferred,'opposed_ink':opposed,
        'path':path if trace else None}


def states_for_goal(base,worlds,goal_type):
    saved=[];skip=collections.Counter()
    for w,info,seed,fam,case in worlds:
      for tick in c55.TICKS:
        got=c55.c54.prefix(base,w,seed,tick)
        if got is None:skip['early_terminal']+=1;continue
        snap,rng,log,receipt=got
        if goal_type=='observed': target=c55.waypoint(log,snap.pos)
        else: target=c55.c54.c52.c50.resolved_goal(snap,info)
        if target is None:skip['no_goal']+=1;continue
        if goal_type=='remote' and tuple(target) in log.neighbors:skip['target_already_observed']+=1;continue
        route=log.route(snap.pos,target)
        saved.append((snap,rng,log,receipt,tick,target,info,seed,fam,case,
                      len(route)-1 if route else None))
    return saved,dict(skip)


def run(prior,output,train_reps=2,online_reps=2):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c55.TICKS=(9,) # bounded, fully matched prefix-tick replication
    base,opt,_=c55.c54.c52.c50.c47.c44.prerequisites(Path(prior)/'predecessor_C42')
    historical=set()
    previous=Path(prior)/'previous_C55'
    for name in ('C55-training-records.json','C55-blind-records.json','C55-online.json','C55-external-online.json'):
        historical.update(x['graph'] for x in json.loads((previous/name).read_text()))
    trainw,trec=c55.c54.c52.worlds(c55.c54.c52.TRAIN_FAMILIES,c55.c54.c52.c50.c47.c41.world,
                      SEED+10031,train_reps,exclude=historical)
    trstates,trskip=c55.states(base,trainw)
    teacher,cov=augmented_teacher(base,opt,trstates)
    assert len(teacher)>300
    trained,stats=c55.train(teacher,epochs=9)
    heads={'C56-AUGMENTED':trained['C55-ALIGNED'],'C56-SHUFFLED':trained['C55-SHUFFLED']}
    for name in ('C55-ALIGNED','C55-SHUFFLED'):
        m=c55.Head();m.load_state_dict(load_file(str(Path(prior)/'checkpoints'/(name+'.safetensors'))));m.eval();heads[name]=m
    checkpoints={}
    for name in ('C56-AUGMENTED','C56-SHUFFLED'):
        p=output/(name+'.safetensors');save_file(heads[name].state_dict(),str(p))
        checkpoints[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    train_ids={r['graph'] for r in trec}
    onlineU,recU=c55.c54.c52.worlds(c55.c54.c52.ONLINE_FAMILIES,c55.c54.c52.c50.c47.c42.world,
                      SEED+901177,online_reps,exclude=historical|train_ids)
    onlineF,recF=c55.c54.c52.worlds(c55.c54.c52.TRAIN_FAMILIES,c55.c54.c52.c50.c47.c41.world,
                      SEED+801177,online_reps,exclude=historical|train_ids|{r['graph'] for r in recU})
    receipts={'train':trec,'online':recU+recF}
    assert not train_ids.intersection({r['graph'] for r in receipts['online']})
    allstates=[];skips={}
    for goaltype in ('observed','remote'):
        ss,sskip=states_for_goal(base,onlineU+onlineF,goaltype)
        allstates.extend((goaltype,x) for x in ss);skips[goaltype]=sskip
    results=[];traces=[];modelcounts=collections.defaultdict(collections.Counter)
    conds=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.Counter()))
    for goaltype,st in allstates:
        snap,rng,log,receipt,tick,target,info,seed,fam,case,distance=st
        graph=info['graph_sha']
        for drop in LEVELS:
          initial=erase(log,graph,tick,drop)
          retained=int(initial.route(snap.pos,target) is not None)
          for arm in c55.ARMS:
            outcomes={}
            for name in POLICIES:
                z=rollout(base,opt,snap,rng,log,tick,target,arm,name,heads,graph,drop)
                outcomes[name]=z
                if arm[0]!='none':
                    key=(goaltype,drop,fam,case,tick,arm[1]);mc=modelcounts[(goaltype,drop,name)]
                    mc['goals']+=z['goal'];mc['episodes']+=1
                    mc['guided_turns']+=z['guided_turns'];mc['route_available']+=z['route_available'];mc['route_missing']+=z['route_missing']
                    mc['followed']+=z['followed'];mc['preferred_ink']+=z['preferred_ink'];mc['opposed_ink']+=z['opposed_ink']
                    conds[key][name]['goals']+=z['goal']
            assert all(z['host_rejections']==0 and z['protected_sha']==receipt['protected_sha'] for z in outcomes.values())
            if arm[0]=='none':assert len({(z['action_sha'],z['ink_sha']) for z in outcomes.values()})==1
            results.append({'goal_type':goaltype,'drop':drop,'graph':graph,'family':fam,'case':case,
                 'seed':seed,'tick':tick,'arm':arm,'target':target,'route_retained':retained,
                 'prefix_sha':receipt['prefix_sha'],'prefix_ink_sha':receipt['ink_sha'],
                 'outcomes':outcomes})
            if arm==('backtrack',1.) and goaltype=='observed' and len(traces)<24 and drop in (0.,.5):
                for name in ('C34-RESET','C55-ROUTE','C55-ALIGNED','C56-AUGMENTED'):
                    if len(traces)>=24:break
                    z=rollout(base,opt,snap,rng,log,tick,target,arm,name,heads,graph,drop,trace=True)
                    assert (z['action_sha'],z['ink_sha'])==(outcomes[name]['action_sha'],outcomes[name]['ink_sha'])
                    traces.append({'goal_type':goaltype,'drop':drop,'graph':graph,'family':fam,'case':case,
                       'seed':seed,'tick':tick,'initial':receipt['initial'],
                       'positions':receipt['positions'],'target':target,'protected':sorted(snap.protected),
                       'policy':name,'outcome':z})
    aggregate={}
    for (goaltype,drop,name),c in sorted(modelcounts.items()):
        aggregate.setdefault(goaltype,{}).setdefault(str(drop),{})[name]=dict(c)
    regressions={}
    for key,by in conds.items():
        for name in POLICIES:
            if by[name]['goals'] < by['C34-RESET']['goals']:
                regressions[(key[0],key[1],name)] = regressions.get((key[0],key[1],name),0)+1
    by_graph=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    for r in results:
        if r['arm'][0]=='none':continue
        key=(r['goal_type'],r['drop'])
        for name,z in r['outcomes'].items(): by_graph[key][r['graph']][name]+=z['goal']
    paired={}
    for key,g in by_graph.items():
        for a,b in (('C56-AUGMENTED','C34-RESET'),('C56-AUGMENTED','C55-ALIGNED'),
                    ('C56-AUGMENTED','C56-SHUFFLED'),('C55-ALIGNED','C34-RESET'),('C55-ROUTE','C34-RESET')):
            ds=np.asarray([v[a]-v[b] for v in g.values()],np.int32)
            random=np.random.default_rng(SEED+len(a)*11+len(b)*17+int(key[1]*100))
            boot=ds[random.integers(len(ds),size=(2000,len(ds)))].sum(1)
            paired[f'{key[0]}-{key[1]}:{a}__{b}']={'delta':int(ds.sum()),'positive_graphs':int(sum(ds>0)),
                      'negative_graphs':int(sum(ds<0)),'descriptive_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    retention=collections.defaultdict(lambda:collections.Counter())
    for r in results:
        if r['arm'][0]!='none':continue
        rr=retention[(r['goal_type'],str(r['drop']))];rr['retained']+=r['route_retained'];rr['episodes']+=1
    s={'schema':'driv3r-c56-partial-memory-recovery/1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'historical_excluded_graphs':len(historical),'train_graphs':len(trec),'online_graphs':len(receipts['online']),
       'train_rows':len(teacher),'training':stats,'teacher_coverage':cov,'train_skips':trskip,
       'checkpoint_sha256':checkpoints,'evaluation_rows':len(results),
       'guided_per_policy':sum(r['arm'][0]!='none' for r in results),'no_guidance_per_policy':sum(r['arm'][0]=='none' for r in results),
       'models':aggregate,'paired':paired,'condition_regressions':{f'{g}/{d}/{n}':v for (g,d,n),v in regressions.items()},
       'route_retention':{f'{g}/{d}':dict(v) for (g,d),v in retention.items()},'skip':skips,
       'cohorts':receipts,'scientific_caveats':['Independent synthetic host, not CyberSand native ABI',
           'Host-selected observed-waypoint goals are easier than remote goals; report separately',
           'All neural variants require a surviving observed route before applying their learned bias',
           'Memory deletion is experimental, not a validated naturalistic memory failure model',
           'Descriptive unadjusted graph-cluster bootstrap after a long adaptive research sequence',
           'No-guidance hash parity does not qualify guided personality preservation',
           'No GitHub Actions immutable checkpoint artifact custody']}
    for name,obj in [('C56-training.json',teacher),('C56-online.json',results),
                     ('C56-traces.json',traces),('C56-summary.json',s)]:
        (output/name).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'training':stats,'evaluation_rows':len(results),
                      'goal_events':{g:{d:{n:v['goals'] for n,v in names.items()}
                      for d,names in drops.items()} for g,drops in aggregate.items()},
                      'retention':s['route_retention']},indent=2),flush=True)
    return s

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('output')
    p.add_argument('--train-reps',type=int,default=2);p.add_argument('--online-reps',type=int,default=2)
    a=p.parse_args();run(a.prior,a.output,a.train_reps,a.online_reps)

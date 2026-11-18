"""C55 — CPU waypoint-following distillation on executable host-observed routes.

Research-only synthetic host. Goal changes are explicitly selected from contemporaneously
observed legal menus; no remote graph map enters deployed policy/teacher. Every act is
revalidated by World.apply. Separate unseen Wilson/Kruskal generator is held out.
The synthetic test asks about an *observed and reachable waypoint*, not the harder
unknown far goals on which C54 failed. Do not conflate these task distributions.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
import c54_delayed_branch_option as c54
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import scores, probs
from c34_independent_maze import ReArmedWatch

SEED=550118
HORIZON=78
TICKS=(9,19)
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
POLICIES=('C18-A0','C34-RESET','C55-ROUTE','C55-ALIGNED','C55-SHUFFLED','C55-RANDOM')
FEAT=13

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(FEAT,20),nn.Tanh(),nn.Linear(20,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def waypoint(log, start):
    """Select a deliberate branch from information recorded by the host along a prefix.

    This is synthetic goal-construction authority, NOT a deployed world-map oracle.
    At deployment the host merely publishes target goal and the observed ledger
    is maintained without inspecting any remote cell.
    """
    offered=[]
    for q in sorted(log.frontiers()):
        route=log.route(start,q)
        if route and 3<=len(route)<=15:
            offered.append((len(route)-1,q,route))
    if not offered:return None
    offered.sort(key=lambda v:(-v[0],v[1]))
    # Prefer 3-8 legal edges away; constructed route is *already observed*.
    reasonable=[v for v in offered if v[0]<=8]
    return (reasonable or offered)[0][1]


def states(base, worlds):
    made=[];skips=collections.Counter()
    for w,info,seed,family,case in worlds:
        for tick in TICKS:
            prior=c54.prefix(base,w,seed,tick)
            if prior is None:skips['early_terminal']+=1;continue
            snap,rng,log,receipt=prior
            target=waypoint(log,snap.pos)
            if target is None:skips['no_observed_nonlocal_goal']+=1;continue
            assert target not in log.neighbors # unvisited target; only legally observed as frontier
            route=log.route(snap.pos,target)
            assert route and len(route)>2
            made.append((snap,rng,log,receipt,tick,target,info,seed,family,case,len(route)-1))
    return made,dict(skips)


def cue(world,arm):
    advice=HostAdvice(world)
    if arm[0]!='none':getattr(advice,'issue_'+arm[0])(world,strength=arm[1],ttl=9)
    return advice


def desired(log,w,advice,menu):
    """Current suggestion from host-observed route only and FRESH legal candidates."""
    if advice.active(w)!=1:return ([],None)
    route=log.route(w.pos,w.goal)
    if route is None or len(route)<2:return ([],None)
    nxt=route[1]
    legal=[i for i,c in enumerate(menu[:8]) if c.legal and tuple(c.target)==nxt]
    return legal,nxt


def feature(w,log,menu,c,nxt):
    """No graph oracle, no remote protected/allowed/candidate reads.

    Candidate target and legal flag are supplied by CURRENT host action menu.
    The waypoint is derived only from previously observed host-legal edges.
    """
    vec=(nxt[0]-w.pos[0],nxt[1]-w.pos[1])
    goal=(w.goal[0]-w.pos[0],w.goal[1]-w.pos[1])
    step=(c.target[0]-w.pos[0],c.target[1]-w.pos[1])
    return np.asarray([step[0]/2.,step[1]/2.,vec[0]/2.,vec[1]/2.,
       goal[0]/14.,goal[1]/14.,float(c.action[0]=='INK'),float(c.action[0]=='LIFT'),
       float(c.legal),float(w.personality), min(1.,log.visits[tuple(w.pos)]/8.),
       float(w.stuck/10.),min(1.,len(log.neighbors)/50.)],np.float32)


def collect_teacher(base,option,worlds,max_steps=9):
    """Dense on-policy host-executed teacher examples, no hidden shortest paths."""
    ss,skip=states(base,worlds);examples=[];one_step=[]
    for snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance in ss:
      for arm in ARMS[1:]:
        w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
        advice=cue(w,arm);rr=np.random.default_rng();rr.bit_generator.state=copy.deepcopy(rngstate)
        ledger=copy.deepcopy(log);controller=ReArmedWatch(option,None)
        for j in range(max_steps):
          if w.stopped or w.distance()==0:break
          menu=ledger.see(w);allowed,nxt=desired(ledger,w,advice,menu)
          if allowed:
            for k,c in enumerate(menu):
              if c.legal and k<8:
                x=feature(w,ledger,menu,c,nxt)
                examples.append({'graph':info['graph_sha'],'family':fam,'case':case,
                   'tick':tick,'arm':arm[1],'j':j,'x':x.tolist(),'y':int(k in allowed),
                   'state':sha([w.pos,w.goal,ledger.actions[-5:],w.stuck]),
                   'kind':c.action[0]})
            one_step.append({'graph':info['graph_sha'],'family':fam,'n_correct':len(allowed),
                             'legal_actions':sum(c.legal for c in menu[:8])})
          menu,logits,_,_=controller.modify(base,w,advice,1.)
          c=menu[int(rr.choice(len(menu),p=probs(logits)))];start=tuple(w.pos)
          assert c.legal and w.apply(c) and not w.invalid_attempts
          ledger.after(start,c);controller.after(c);advice.advance(w)
    return examples,one_step,ss,skip


def train(rows,epochs=14):
    torch.use_deterministic_algorithms(True);torch.set_num_threads(1)
    assert len({r['graph'] for r in rows})>=8
    X=np.asarray([r['x'] for r in rows],np.float32);Y=np.asarray([r['y'] for r in rows],np.float32)
    assert len(X)>300 and 0<Y.mean()<.7
    # Matched graph-label shuffle. Trains a genuine negative control with identical
    # optimizer, initialization and feature stream, not a random untrained net.
    by_graph=collections.defaultdict(list)
    for i,r in enumerate(rows):by_graph[r['graph']].append(i)
    groups=sorted(by_graph)
    rng=np.random.default_rng(SEED+31);shuf=Y.copy()
    for shift in range(1,len(groups)):
        perm=groups[shift:]+groups[:shift]
        if any(x!=y for x,y in zip(groups,perm)):break
    for dest,src in zip(groups,perm):
        for j,idx in enumerate(by_graph[dest]):shuf[idx]=Y[by_graph[src][j%len(by_graph[src])]]
    shuffled_changes=int((Y!=shuf).sum())
    assert shuffled_changes>0
    torch.manual_seed(SEED+17);init=Head()
    heads={k:copy.deepcopy(init) for k in ('C55-ALIGNED','C55-SHUFFLED')}
    orders=[np.random.default_rng(SEED+101+i).permutation(len(rows)) for i in range(epochs)]
    losses={};updates={}
    X=torch.from_numpy(X)
    for name,targets in (('C55-ALIGNED',Y),('C55-SHUFFLED',shuf)):
        m=heads[name];opt=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.01)
        y=torch.from_numpy(targets)
        # Balanced classification so majority-class baseline cannot win by abstaining.
        positive=max(.01,float(targets.mean()));wt=torch.where(y>0,.5/positive,.5/(1-positive))
        curve=[];n=0
        for order in orders:
            loss_ep=0.
            for batch in (order[k:k+64] for k in range(0,len(order),64)):
                pred=m(X[batch]);loss=(wt[batch]*nn.functional.binary_cross_entropy_with_logits(pred,y[batch],reduction='none')).mean()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt.step();loss_ep+=float(loss)*len(batch);n+=1
            curve.append(round(loss_ep/len(X),5))
        m.eval();losses[name]=curve;updates[name]=n
    return heads,{'n_states':len(set(r['state'] for r in rows)),'n_examples':len(rows),
       'positive_examples':int(Y.sum()),'positive_rate':round(float(Y.mean()),4),
       'training_graphs':len(groups),'shuffled_changes':shuffled_changes,
       'optimizer_updates':updates,'training_losses':losses,
       'X_sha256':hashlib.sha256(X.numpy().tobytes()).hexdigest(),
       'Y_sha256':hashlib.sha256(Y.tobytes()).hexdigest()}


def blind_rank(rows,heads):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['graph'],r['tick'],r['arm'],r['j'],r['state'])].append(r)
    metrics={}
    for name,head in heads.items():
        ok=total=0;predicted=[]
        with torch.no_grad():
            for sample in groups.values():
                x=torch.tensor(np.asarray([r['x'] for r in sample],np.float32))
                z=head(x).numpy();k=int(np.argmax(z));ok+=int(sample[k]['y']==1);total+=1
                predicted.append(float(z.max()))
        metrics[name]={'top1_legal_route_accuracy':ok/total if total else None,'states':total,
                       'mean_top_logit':float(np.mean(predicted)) if predicted else None}
    return metrics


def rollout(base,option,snap,rngstate,log,tick,target,arm,policy,heads=None,trace=False):
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=cue(w,arm);r=np.random.default_rng();r.bit_generator.state=copy.deepcopy(rngstate)
    ctrl=None if policy=='C18-A0' else ReArmedWatch(option,None)
    ledger=copy.deepcopy(log);acts=[];turns=0;matched=0;compliant=0;pref=opp=0;path=[];goal=False
    for t in range(tick,HORIZON):
        if w.stopped:break
        if w.distance()==0:goal=True;break
        ledger.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        legal,nxt=desired(ledger,w,advice,menu)
        if legal and policy not in ('C18-A0','C34-RESET'):
            turns+=1
            if policy=='C55-ROUTE':
                delta=np.zeros(len(menu),np.float32)
                delta[legal]=4.2*arm[1]
            elif policy=='C55-RANDOM':
                rr=np.random.default_rng(SEED+len(acts)+w.seed+int(w.steps*73))
                shuffled=sorted(i for i in range(8) if menu[i].legal)
                delta=np.zeros(len(menu),np.float32)
                delta[shuffled[int(rr.integers(len(shuffled)))]]=4.2*arm[1]
            else:
                with torch.no_grad():
                    xs=np.asarray([feature(w,ledger,menu,c,nxt) for c in menu[:8]],np.float32)
                    z=heads[policy](torch.from_numpy(xs)).numpy()
                delta=np.zeros(len(menu),np.float32)
                delta[:8]=4.2*arm[1]*(2/(1+np.exp(-np.clip(z,-20,20)))-1)
            logits=np.asarray(logits,dtype=np.float32)+delta
        c=menu[int(r.choice(len(menu),p=probs(logits)))];start=tuple(w.pos)
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        ledger.after(start,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w);acts.append((c.action,c.target))
        if legal:
            matched+=int(tuple(c.target)==nxt)
            compliant+=int(policy not in ('C18-A0','C34-RESET') and tuple(c.target)==nxt)
        if c.action[0] in ('INK','LIFT'):
            turn=float(c.features[13]-c.features[14])*w.personality
            pref+=int(turn>0);opp+=int(turn<0)
        if trace:path.append({'t':t,'p':tuple(w.pos),'a':c.action[0],'hinted':int(bool(legal))})
        if w.distance()==0:goal=True;break
    assert sha(sorted(w.protected))==sha(sorted(snap.protected))
    return {'goal':int(goal),'steps':len(acts),'final_distance':w.distance(),
       'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),
       'host_rejections':w.invalid_attempts,'protected_sha':sha(sorted(w.protected)),
       'route_opportunities':turns,'next_waypoint_followed':matched,'preferred_ink':pref,'opposed_ink':opp,
       'path':path if trace else None}


def evaluate(base,option,worlds,heads,trace_limit=18):
    ss,skips=states(base,worlds);rows=[];gallery=[]
    for snap,rng,log,receipt,tick,target,info,seed,fam,case,length in ss:
      for arm in ARMS:
        out={}
        for p in POLICIES:
          if arm[0]=='none' and p!='C18-A0':out[p]=copy.deepcopy(out['C18-A0'])
          else:out[p]=rollout(base,option,snap,rng,log,tick,target,arm,p,heads)
        assert all(o['host_rejections']==0 and o['protected_sha']==receipt['protected_sha'] for o in out.values())
        if arm[0]=='none':assert len({(o['action_sha'],o['ink_sha']) for o in out.values()})==1
        row={'graph':info['graph_sha'],'family':fam,'case':case,'seed':seed,'tick':tick,
             'arm':arm,'path_len':length,'target':target,'prefix_sha':receipt['prefix_sha'],
             'prefix_ink_sha':receipt['ink_sha'],'outcomes':out}
        rows.append(row)
        if arm==('backtrack',1.) and len(gallery)<trace_limit:
          for p in ('C34-RESET','C55-ROUTE','C55-ALIGNED'):
            if len(gallery)>=trace_limit:break
            v=rollout(base,option,snap,rng,log,tick,target,arm,p,heads,True)
            assert (v['action_sha'],v['ink_sha'])==(out[p]['action_sha'],out[p]['ink_sha'])
            gallery.append({'graph':info['graph_sha'],'family':fam,'case':case,'seed':seed,
               'tick':tick,'initial':receipt['initial'],'positions':receipt['positions'],
               'target':target,'protected':sorted(snap.protected),'policy':p,'outcome':v})
    return rows,gallery,skips


def summary(rows,train,blind,online,checkpoints,skips,receipts):
    graph_ids={x:{r['graph'] for r in rec} for x,rec in receipts.items()}
    assert not(graph_ids['train']&graph_ids['blind'] or graph_ids['train']&graph_ids['online'] or graph_ids['blind']&graph_ids['online'])
    guided=[r for r in rows if r['arm'][0]!='none'];none=[r for r in rows if r['arm'][0]=='none']
    models={}
    groups=collections.defaultdict(collections.Counter);graphs=collections.defaultdict(collections.Counter)
    for row in guided:
        cond=(row['family'],row['case'],row['tick'],row['arm'])
        for p,o in row['outcomes'].items():groups[cond][p]+=o['goal'];graphs[row['graph']][p]+=o['goal']
    for p in POLICIES:
        zz=[r['outcomes'][p] for r in guided]
        models[p]={'goal_events':sum(o['goal'] for o in zz),'guided_episodes':len(zz),
           'condition_regressions_vs_RESET':sum(g[p]<g['C34-RESET'] for g in groups.values()),
           'by_family':{fam:sum(r['outcomes'][p]['goal'] for r in guided if r['family']==fam) for fam in sorted({r['family'] for r in guided})},
           'route_opportunities':sum(o['route_opportunities'] for o in zz),
           'next_waypoint_followed':sum(o['next_waypoint_followed'] for o in zz),
           'preferred_ink':sum(o['preferred_ink'] for o in zz),'opposed_ink':sum(o['opposed_ink'] for o in zz),
           'no_guidance_parity':sum((r['outcomes'][p]['action_sha'],r['outcomes'][p]['ink_sha']) ==
                  (r['outcomes']['C18-A0']['action_sha'],r['outcomes']['C18-A0']['ink_sha']) for r in none)}
    paired={}
    for a in ('C55-ROUTE','C55-ALIGNED','C55-SHUFFLED','C55-RANDOM'):
      for b in ('C34-RESET','C55-ROUTE','C55-SHUFFLED'):
        if a==b:continue
        ds=np.array([v[a]-v[b] for v in graphs.values()],np.int32)
        rng=np.random.default_rng(SEED+len(a)*7+len(b)*29)
        draws=ds[rng.integers(len(ds),size=(2000,len(ds)))].sum(axis=1)
        paired[f'{a}__{b}']={'goal_delta':int(ds.sum()),'positive_graphs':int(sum(ds>0)),
           'negative_graphs':int(sum(ds<0)),'descriptive_95':np.quantile(draws,[.025,.975]).tolist()}
    gates={'beats_RESET':models['C55-ALIGNED']['goal_events']>models['C34-RESET']['goal_events'],
       'beats_shuffled':models['C55-ALIGNED']['goal_events']>models['C55-SHUFFLED']['goal_events'],
       'beats_route':models['C55-ALIGNED']['goal_events']>models['C55-ROUTE']['goal_events'],
       'zero_condition_regressions':models['C55-ALIGNED']['condition_regressions_vs_RESET']==0,
       'positive_cluster_lower':paired['C55-ALIGNED__C34-RESET']['descriptive_95'][0]>0,
       'exact_unguided_parity':all(z['no_guidance_parity']==len(none) for z in models.values())}
    return {'schema':'driv3r-c55-observed-waypoint-distillation/1',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'training':train,'blind':blind,'cohorts':{k:len(v) for k,v in graph_ids.items()},
      'cohort_graph_hashes':{k:sha(sorted(v)) for k,v in graph_ids.items()},
      'evaluation_rows':len(rows),'guided_per_policy':len(guided),'no_guidance_per_policy':len(none),
      'models':models,'paired':paired,'promotion_gates':gates,'promotion_eligible':all(gates.values()),
      'checkpoint_sha256':checkpoints,'skipped':skips,'receipts':receipts,
      'caveats':['Synthetic host only, not authoritative CyberSand native ABI',
       'Prospectively constructed reachable host-observed waypoint is intentionally easier than C54 far-goal transfer',
       'Training supervision is route-step imitation from host-observed route, not full-goal return optimization',
       'World factory starts with a separately host-executed 24-cell prefix not reconstructed in this study',
       'No-cue parity does not establish guided whole-trajectory personality preservation',
       'Unadjusted exploratory graph-cluster CI; research sequence has selection effects',
       'No GitHub Actions immutable checkpoint custody or native acceptance']}


def run(out,prior,train_reps=4,blind_reps=2,online_reps=4,familiar_reps=2):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base,option,_=c54.c52.c50.c47.c44.prerequisites(Path(prior)/'predecessor_C42')
    exclusions=set()
    # Historical C54 records are excluded by graph identity, not guessed seeds.
    # This is a fresh replication, not train/test split over previously seen worlds.
    hist=Path(prior)/'C54-summary.json'
    if hist.exists():
       meta=json.loads(hist.read_text())
       for block in meta['worlds'].values():exclusions.update(r['graph'] for r in block)
    train_w,train_rec=c54.c52.worlds(c54.c52.TRAIN_FAMILIES,c54.c52.c50.c47.c41.world,
                                SEED+103,train_reps,exclude=exclusions)
    rows,training_steps,_,train_skip=collect_teacher(base,option,train_w)
    heads,train_stats=train(rows)
    print('C55 training',train_stats['training_graphs'],'graphs',len(rows),'legal candidate labels',flush=True)
    checks={}
    for name,h in heads.items():
        dest=out/(name+'.safetensors');save_file(h.state_dict(),str(dest))
        checks[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    tr={r['graph'] for r in train_rec}
    blind_w,blind_rec=c54.c52.worlds(c54.c52.ONLINE_FAMILIES,c54.c52.c50.c47.c42.world,
                       SEED+210103,blind_reps,exclude=exclusions|tr)
    blind_rows,steps,_,blind_skip=collect_teacher(base,option,blind_w)
    rank=blind_rank(blind_rows,heads)
    br={r['graph'] for r in blind_rec}
    online_u,rec_u=c54.c52.worlds(c54.c52.ONLINE_FAMILIES,c54.c52.c50.c47.c42.world,
                   SEED+410103,online_reps,exclude=exclusions|tr|br)
    online_f,rec_f=c54.c52.worlds(c54.c52.TRAIN_FAMILIES,c54.c52.c50.c47.c41.world,
                   SEED+710103,familiar_reps,exclude=exclusions|tr|br|{r['graph'] for r in rec_u})
    eval_rows,traces,eval_skip=evaluate(base,option,online_u+online_f,heads)
    s=summary(eval_rows,train_stats,rank,eval_rows,checks,
              {'train':train_skip,'blind':blind_skip,'online':eval_skip},
              {'train':train_rec,'blind':blind_rec,'online':rec_u+rec_f})
    for name,obj in [('C55-training-records.json',rows),('C55-blind-records.json',blind_rows),
                     ('C55-online.json',eval_rows),('C55-traces.json',traces),('C55-summary.json',s)]:
        (out/name).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'training':train_stats,'blind_rank':rank,
      'goal_events':{p:z['goal_events'] for p,z in s['models'].items()},
      'paired':s['paired']['C55-ALIGNED__C34-RESET'],
      'gates':s['promotion_gates']},indent=2),flush=True)
    return s

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('prior');ap.add_argument('out')
    ap.add_argument('--train-reps',type=int,default=4)
    ap.add_argument('--blind-reps',type=int,default=2)
    ap.add_argument('--online-reps',type=int,default=4)
    ap.add_argument('--familiar-reps',type=int,default=2)
    a=ap.parse_args();run(a.out,a.prior,a.train_reps,a.blind_reps,a.online_reps,a.familiar_reps)

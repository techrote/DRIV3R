"""C54 — causal delayed-goal test of a host-observed branch option.

Research-only synthetic host. Every waypoint comes from a host-legal menu
recorded while actually at that position; full map and oracle never enter the
option or fitted head. The host revalidates every single action. No CyberSand
native ABI or game-side change is implied.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import sha
from signed_feedback import HostAdvice
from c28_temporal_option import scores, probs
from c34_independent_maze import ReArmedWatch
import c52_shortest_step_cue as c52

SEED = 540118
HORIZON = 78
CUE_TICKS = (9, 19)
ARMS = (('none', 0.), ('backtrack', .35), ('backtrack', 1.))
POLICIES = ('C18-A0', 'C34-RESET', 'C54-ROUTE', 'C54-ALIGNED', 'C54-SHUFFLED', 'C54-RANDOM')
FCOUNT = 16


class ObservedLedger:
    """Known local neighbourhoods and traversals, never remote map inspection."""
    def __init__(self):
        self.neighbors = {}
        self.visits = collections.Counter()
        self.actions = []

    def see(self, world):
        p = tuple(world.pos)
        # Only the current host-authorised action menu is observed. No reading
        # world.protected or world.allowed for other cells, even former positions.
        menu = world.candidates(0.)
        self.neighbors[p] = tuple(sorted({tuple(x.target) for x in menu[:4] if x.legal}))
        self.visits[p] += 1
        return menu

    def after(self, start, choice):
        self.actions.append((tuple(start), tuple(choice.target), choice.action[0]))

    def frontiers(self):
        return {n for here, nbrs in self.neighbors.items() for n in nbrs if n not in self.neighbors}

    def route(self, origin, destination):
        """BFS exclusively through locally observed edges, not global geometry."""
        origin = tuple(origin); destination = tuple(destination)
        path = {origin:None}; queue = collections.deque([origin])
        while queue:
            p = queue.popleft()
            if p == destination: break
            for q in self.neighbors.get(p, ()):
                if q not in path:
                    path[q] = p
                    queue.append(q)
        if destination not in path: return None
        out = [destination]
        while path[out[-1]] is not None:out.append(path[out[-1]])
        return list(reversed(out))

    def choose(self, world):
        """Select one observed unvisited branch, stable tie order, no oracle."""
        scored = []
        for p in self.frontiers():
            route = self.route(world.pos, p)
            if route is None or len(route)<2:continue
            # Favor observed close/frontier branches and visible Manhattan goal;
            # this is a deliberately weak local heuristic, NOT hidden geometry.
            score = (len(route)-1) + .35 * world.distance(p)
            scored.append((score, len(route), p))
        if not scored:return None
        return min(scored)[2]


class BranchOption:
    """Bounded, cancellable intention; all execution stays stochastic and legal."""
    def __init__(self, ledger, active, max_ticks=18):
        self.ledger=ledger
        self.active=active
        self.end_after=max_ticks
        self.target=None
        self.goal=None
        self.age=0
        self.prompts=0
        self.completed=0
        self.aborted=0
        self.deviation=0

    def prepare(self, w, advice, menu):
        if not self.active:return None
        if advice.directive==0 or tuple(w.goal)!=self.goal and self.goal is not None:
            self.active=False;self.aborted+=1;return None
        if self.goal is None:self.goal=tuple(w.goal)
        if self.age>=self.end_after:
            self.active=False;self.aborted+=1;return None
        if self.target is None:
            self.target=self.ledger.choose(w)
            if self.target is None:
                self.active=False;self.aborted+=1;return None
        # We must not teleport to a remembered point. Compute observed route,
        # and require its next edge in the FRESH host menu.
        route=self.ledger.route(w.pos,self.target)
        if route is None:
            self.active=False;self.aborted+=1;return None
        if len(route)==1:
            self.completed+=1;self.active=False;return None
        next_pos=route[1]
        legal=[i for i,c in enumerate(menu[:8]) if c.legal and tuple(c.target)==next_pos]
        if not legal:
            self.active=False;self.aborted+=1;return None
        self.age+=1;self.prompts+=1
        return legal

    def after(self, selected):
        if self.target is not None and tuple(selected.target)!=self.target and self.active:
            self.deviation+=1


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(FCOUNT,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def features(world, ledger, arm):
    frontier=ledger.frontiers(); routes=[]
    for p in frontier:
        route=ledger.route(world.pos,p)
        if route and len(route)>1:routes.append((len(route)-1,world.distance(p)))
    routes.sort()
    known = len(ledger.neighbors)
    visited = ledger.visits[tuple(world.pos)]
    legal_here = ledger.neighbors.get(tuple(world.pos),())
    recent=[b for a,b,k in ledger.actions[-12:]]
    x=[float(len(frontier)>0), min(1.,len(frontier)/10.),
       min(1.,(routes[0][0] if routes else 40)/20.),
       min(1.,(routes[0][1] if routes else 40)/28.),
       min(1.,(routes[-1][0] if routes else 40)/20.),
       min(1.,known/30.), min(1.,visited/6.),
       min(1.,len(legal_here)/4.),
       min(1.,len(set(recent))/12.),
       float(world.personality), min(1.,world.distance()/28.),
       min(1.,world.stuck/8.), min(1.,world.steps/100.),
       float(world.goal_change), float(arm[1]),
       float(arm[0]=='backtrack')]
    out=np.asarray(x,np.float32)
    assert out.shape==(FCOUNT,) and np.isfinite(out).all()
    return out


def prefix(base,world,seed,tick):
    """Record menus contemporaneously, including host execution, not retrospectively."""
    w=copy.deepcopy(world); advice=HostAdvice(w)
    rng=np.random.default_rng(seed+99203)
    log=ObservedLedger(); pos=[tuple(w.pos)];actions=[]
    protected_sha=sha(sorted(w.protected))
    for _ in range(tick):
        if w.stopped or w.distance()==0:break
        log.see(w)
        menu,logits,_,_=scores(base,w,advice,1.)
        choice=menu[int(rng.choice(len(menu),p=probs(logits)))]
        start=tuple(w.pos)
        assert choice.legal and w.apply(choice) and w.invalid_attempts==0
        log.after(start,choice);advice.advance(w)
        actions.append((choice.action,choice.target));pos.append(tuple(w.pos))
    if len(actions)!=tick or w.stopped or w.distance()==0:return None
    log.see(w)
    assert protected_sha==sha(sorted(w.protected))
    return w,copy.deepcopy(rng.bit_generator.state),log,{
        'prefix_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
        'protected_sha':protected_sha,'positions':pos,'initial':pos[0]}


def suffix(base, option, snap, rng_state, log, tick, target, arm, policy,
           model=None, mean=None, scale=None, trace=False):
    w=copy.deepcopy(snap);ledger=copy.deepcopy(log)
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rng_state)
    w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c52.issue(w,arm)
    ctrl=ReArmedWatch(option,None) if policy!='C18-A0' else None
    use=False;gate_logit=None
    if arm[0]=='backtrack' and ledger.frontiers():
        if policy=='C54-ROUTE':use=True
        elif policy=='C54-RANDOM':use=bool(rng.random()<.5)
        elif policy in ('C54-ALIGNED','C54-SHUFFLED'):
            x=features(w,ledger,arm)
            z=torch.from_numpy(np.clip((x-mean)/scale,-6.,6.).astype(np.float32))
            with torch.no_grad():gate_logit=float(model(z).item())
            use=gate_logit>0.
    active_option=BranchOption(ledger,use)
    acts=[]; path=[]; preferred=opposed=0; option_turns=0; reached=False; end=tick
    for t in range(tick,HORIZON):
        if w.stopped:break
        if w.distance()==0:reached=True;break
        ledger.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        picked=active_option.prepare(w,advice,menu)
        if picked:
            option_turns+=1
            # Preserve randomness and preference: modest bonus to INK and LIFT,
            # not a hard action override. STOP, backtrack remain available.
            for i in picked:logits[i]+=3.0
        choice=menu[int(rng.choice(len(menu),p=probs(logits)))]
        start=tuple(w.pos)
        assert choice.legal and w.apply(choice) and w.invalid_attempts==0
        ledger.after(start,choice);active_option.after(choice)
        if ctrl is not None:ctrl.after(choice)
        advice.advance(w)
        acts.append((choice.action,choice.target));end=t
        if choice.action[0] in ('INK','LIFT'):
            turn=float(choice.features[13]-choice.features[14])*w.personality
            preferred+=int(turn>0);opposed+=int(turn<0)
        if trace:path.append({'t':t,'pos':tuple(w.pos),'action':choice.action[0]})
        if w.distance()==0:reached=True;break
    assert sha(sorted(w.protected))==sha(sorted(snap.protected))
    return {'goal':int(reached),'steps':len(acts),'success_tick':end if reached else None,
       'final_manhattan':w.distance(),'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),
       'protected_sha':sha(sorted(w.protected)),'host_rejections':w.invalid_attempts,
       'preferred_ink':preferred,'opposed_ink':opposed,
       'gate_logit':gate_logit,'option_started':int(use),'option_turns':option_turns,
       'option_finished':active_option.completed,'option_abort':active_option.aborted,
       'path':path if trace else None}


def snapshots(base,worlds):
    made=[]; skip=collections.Counter()
    for w,info,seed,fam,case in worlds:
        for tick in CUE_TICKS:
            got=prefix(base,w,seed,tick)
            if got is None:skip['prefix_terminated']+=1;continue
            snap,rng,ledger,receipt=got
            target=c52.c50.resolved_goal(snap,info)
            if target is None:skip['goal_visited']+=1;continue
            made.append((snap,rng,ledger,receipt,tick,target,info,seed,fam,case))
    return made,dict(skip)


def collect_pair(base,option,worlds):
    examples=[];recs=[];skips={};seen=set()
    states,skips=snapshots(base,worlds)
    for snap,rng,ledger,receipt,tick,target,info,seed,fam,case in states:
      for arm in ARMS[1:]:
        xy=features(snap,ledger,arm)
        raw=suffix(base,option,snap,rng,ledger,tick,target,arm,'C34-RESET')
        route=suffix(base,option,snap,rng,ledger,tick,target,arm,'C54-ROUTE')
        # Only route-vs-RESET matched fully executed suffixes create targets.
        advantage=route['goal']-raw['goal']
        row={'graph':info['graph_sha'],'family':fam,'case':case,'personality':snap.personality,
             'tick':tick,'strength':arm[1], 'prefix_sha':receipt['prefix_sha'],
             'x':xy.tolist(),'goal_diff':advantage,
             'distance_diff':raw['final_manhattan']-route['final_manhattan'],
             'reset_goal':raw['goal'],'route_goal':route['goal'],
             'route_option_turns':route['option_turns']}
        examples.append(row)
    return examples,skips


def fit(rows,epochs=24):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    X=np.asarray([r['x'] for r in rows],np.float32)
    Y=np.asarray([r['goal_diff'] for r in rows],np.float32)
    graphs=sorted({r['graph'] for r in rows})
    assert len(graphs)>=8 and np.isfinite(X).all()
    mean=X.mean(0);scale=np.maximum(X.std(0),.08)
    norm=torch.from_numpy(np.clip((X-mean)/scale,-6,6).astype(np.float32))
    # Graph-level shuffled outcome labels, leaving observation/candidate counts intact.
    group=collections.defaultdict(list)
    for i,r in enumerate(rows):group[r['graph']].append(i)
    rng=np.random.default_rng(SEED+31);order_graphs=np.asarray(graphs)[rng.permutation(len(graphs))]
    shuffle=Y.copy()
    for t,source in zip(graphs,order_graphs):
        src=Y[group[source]]
        for i,idx in enumerate(group[t]):shuffle[idx]=src[i%len(src)]
    changed=int(np.count_nonzero(shuffle!=Y))
    # Targets are signed actual GOAL deltas (0 ties), not an oracle or short-return.
    # Add small regularizer on ties; no bootstrapping based on evaluation outcomes.
    torch.manual_seed(SEED+17)
    init=Head();heads={name:copy.deepcopy(init) for name in ('C54-ALIGNED','C54-SHUFFLED')}
    orders=[np.random.default_rng(SEED+99+e).permutation(len(rows)) for e in range(epochs)]
    curves={};updates={}
    for name,yy in (('C54-ALIGNED',Y),('C54-SHUFFLED',shuffle)):
        head=heads[name];opt=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
        targets=torch.from_numpy(yy);weights=torch.where(targets.abs()>0,5.,1.)
        losses=[];n=0
        for order in orders:
            s=0.
            for k in range(0,len(order),32):
                ix=order[k:k+32];pred=head(norm[ix]);loss=(weights[ix]*(pred-targets[ix]).square()).mean()
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
                s+=float(loss.detach())*len(ix);n+=1
            losses.append(round(s/len(order),6))
        head.eval();curves[name]=losses;updates[name]=n
    return heads,mean,scale,{'training_graphs':len(graphs),'train_pairs':len(rows),
      'different_goals':int(np.sum(Y!=0)),'route_wins':int(np.sum(Y>0)),
      'route_losses':int(np.sum(Y<0)),'shuffle_changed':changed,
      'feature_sha256':hashlib.sha256(X.tobytes()).hexdigest(),
      'target_sha256':hashlib.sha256(Y.tobytes()).hexdigest(),
      'shuffled_target_sha256':hashlib.sha256(shuffle.tobytes()).hexdigest(),
      'training_updates':updates,'training_losses':curves}


def evaluate(base,option,worlds,heads,mean,scale,retain=12):
    rows=[];gallery=[];states,skip=snapshots(base,worlds)
    for snap,rng,ledger,receipt,tick,target,info,seed,fam,case in states:
      for arm in ARMS:
        outcomes={}
        for name in POLICIES:
          if arm[0]=='none' and name!='C18-A0':
              outcomes[name]=copy.deepcopy(outcomes['C18-A0']);continue
          model=heads.get(name)
          z=suffix(base,option,snap,rng,ledger,tick,target,arm,name,model,mean,scale)
          outcomes[name]=z
        assert all(z['host_rejections']==0 and z['protected_sha']==receipt['protected_sha'] for z in outcomes.values())
        if arm[0]=='none':assert len({(v['action_sha'],v['ink_sha']) for v in outcomes.values()})==1
        row={'graph':info['graph_sha'],'family':fam,'case':case,'personality':snap.personality,
             'seed':seed,'tick':tick,'arm':arm[0],'strength':arm[1],
             'initial':receipt['initial'],'prefix_sha':receipt['prefix_sha'],
             'prefix_ink_sha':receipt['ink_sha'],'target':target,'outcomes':outcomes}
        rows.append(row)
        if arm==('backtrack',1.) and len(gallery)<retain:
          for name in ('C34-RESET','C54-ROUTE','C54-ALIGNED','C54-SHUFFLED'):
            if len(gallery)>=retain:break
            extra=suffix(base,option,snap,rng,ledger,tick,target,arm,name,heads.get(name),mean,scale,True)
            assert (extra['action_sha'],extra['ink_sha'])==(outcomes[name]['action_sha'],outcomes[name]['ink_sha'])
            gallery.append({'graph':info['graph_sha'],'family':fam,'case':case,
              'model':name,'personality':snap.personality,'seed':seed,'tick':tick,
              'protected':sorted(snap.protected),'prefix_positions':receipt['positions'],
              'target':target,'outcome':extra})
    return rows,gallery,skip


def summarize(train,blind,online,receipts,checkpoints,stats,skips):
    ids={k:{i['graph'] for i in receipts[k]} for k in ('train','blind','online')}
    assert not (ids['train']&ids['blind'] or ids['train']&ids['online'] or ids['blind']&ids['online'])
    guided=[r for r in online if r['arm']!='none'];none=[r for r in online if r['arm']=='none']
    aggregate={};groups=collections.defaultdict(collections.Counter);graphs=collections.defaultdict(collections.Counter)
    for r in guided:
      key=(r['family'],r['case'],r['tick'],r['arm'],r['strength'])
      for name,z in r['outcomes'].items():
        groups[key][name]+=z['goal'];graphs[r['graph']][name]+=z['goal']
    for name in POLICIES:
      rec=[r['outcomes'][name] for r in guided]
      aggregate[name]={'goal_events':sum(x['goal'] for x in rec),'episodes':len(rec),
        'condition_regressions_vs_RESET':sum(g[name]<g['C34-RESET'] for g in groups.values()),
        'by_family':{f:sum(r['outcomes'][name]['goal'] for r in guided if r['family']==f) for f in sorted({r['family'] for r in guided})},
        'option_started':sum(x['option_started'] for x in rec),'option_turns':sum(x['option_turns'] for x in rec),
        'preferred_ink':sum(x['preferred_ink'] for x in rec),'opposed_ink':sum(x['opposed_ink'] for x in rec),
        'no_cue_hash_parity':sum((r['outcomes'][name]['action_sha'],r['outcomes'][name]['ink_sha'])==
               (r['outcomes']['C18-A0']['action_sha'],r['outcomes']['C18-A0']['ink_sha']) for r in none),
        'host_rejections':sum(r['outcomes'][name]['host_rejections'] for r in online)}
    paired={}
    for name in ('C54-ROUTE','C54-ALIGNED','C54-SHUFFLED','C54-RANDOM'):
      for baseline in ('C34-RESET','C54-ROUTE','C54-SHUFFLED'):
        if name==baseline:continue
        ds=np.asarray([g[name]-g[baseline] for g in graphs.values()],np.int32)
        rng=np.random.default_rng(SEED+len(name)*9+len(baseline)*41)
        draws=ds[rng.integers(len(ds),size=(3000,len(ds)))].sum(1)
        paired[f'{name}__{baseline}']={'goal_delta':int(ds.sum()),'positive_graphs':int((ds>0).sum()),
            'negative_graphs':int((ds<0).sum()),'descriptive_cluster_bootstrap95':np.quantile(draws,[.025,.975]).tolist()}
    gates={'improves_RESET':aggregate['C54-ALIGNED']['goal_events']>aggregate['C34-RESET']['goal_events'],
       'beats_shuffled':aggregate['C54-ALIGNED']['goal_events']>aggregate['C54-SHUFFLED']['goal_events'],
       'beats_route_ablation':aggregate['C54-ALIGNED']['goal_events']>aggregate['C54-ROUTE']['goal_events'],
       'zero_condition_regressions':aggregate['C54-ALIGNED']['condition_regressions_vs_RESET']==0,
       'positive_95_lower':paired['C54-ALIGNED__C34-RESET']['descriptive_cluster_bootstrap95'][0]>0,
       'exact_no_cue_parity':all(a['no_cue_hash_parity']==len(none) for a in aggregate.values()),
       'host_legality':all(a['host_rejections']==0 for a in aggregate.values())}
    return {'schema':'driv3r-c54-host-observed-delayed-branch-options/1',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'cohorts':{k:len(v) for k,v in ids.items()},
      'cohort_graph_hashes':{k:sha(sorted(v)) for k,v in ids.items()},
      'train':stats,'training_pairs':len(train),'blind_pairs':len(blind),
      'online_rows':len(online),'guided_episodes_per_policy':len(guided),
      'models':aggregate,'paired':paired,'gates':gates,'promotion_eligible':all(gates.values()),
      'checkpoints_sha256':checkpoints,'worlds':receipts,'skips':skips,
      'caveats':['Synthetic host only; CyberSand native ABI is separate and authoritative',
      'Option visits only remembered host-visible local legal menus; each step is independently authorised',
      'The baseline fixture has an earlier host-executed 24-step graph prefix whose menus are not reconstructed',
      'Full-goal outcome labels are sparse; training may collapse to always-abstain',
      'No-cue parity is synthetic; whole guided personality is not qualified',
      'Graph-cluster interval descriptive, uncorrected for research sequence',
      'C54-RANDOM has additional RNG draw and is a qualitative negative control',
      'Native CyberSand validation and immutable Actions checkpoint custody outstanding']}


def run(out, prior, train_reps=6, blind_reps=2, online_reps=4, online_in_family_reps=2):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1)
    base,option,_=c52.c50.c47.c44.prerequisites(Path(prior)/'predecessor_C42')
    exclusions=set()
    # Explicitly exclude all C53 train, blind, online topologies, plus earlier
    # C52/C51 cohorts, to avoid repeat-testing previously reported graphs.
    here=Path(__file__).resolve().parent
    for fn in ('C53-training.json','C53-blind.json','C53-online.json'):
        exclusions.update(r['graph'] for r in json.loads((here/'output'/fn).read_text()))
    for fn in ('C52-training.json','C52-blind.json','C52-online.json'):
        exclusions.update(r['graph'] for r in json.loads((here/'results'/fn).read_text()))
    exclusions.update(r['graph'] for r in json.loads((here/'c51/output_c51/C51-online.json').read_text()))
    trainw,trainr=c52.worlds(c52.TRAIN_FAMILIES,c52.c50.c47.c41.world,SEED+119,
                            train_reps,exclude=exclusions)
    train,sktrain=collect_pair(base,option,trainw)
    print('C54 train',len(trainw),'graphs',len(train),'paired full-goal samples',flush=True)
    heads,mean,scale,stats=fit(train)
    checks={}
    for name,model in heads.items():
        path=out/(name+'.safetensors');save_file(model.state_dict(),str(path))
        checks[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    tr={r['graph'] for r in trainr}
    blindw,blindr=c52.worlds(c52.ONLINE_FAMILIES,c52.c50.c47.c42.world,SEED+209117,
                             blind_reps,exclude=exclusions|tr)
    blind,skblind=collect_pair(base,option,blindw)
    onexclude=exclusions|tr|{r['graph'] for r in blindr}
    onlinew_ood,onr_ood=c52.worlds(c52.ONLINE_FAMILIES,c52.c50.c47.c42.world,SEED+409117,
                           online_reps,exclude=onexclude)
    onlinew_id,onr_id=c52.worlds(c52.TRAIN_FAMILIES,c52.c50.c47.c41.world,SEED+699117,
                            online_in_family_reps,exclude=onexclude|{r['graph'] for r in onr_ood})
    onlinew=onlinew_ood+onlinew_id;onr=onr_ood+onr_id
    online,traces,skon=evaluate(base,option,onlinew,heads,mean,scale)
    result=summarize(train,blind,online,{'train':trainr,'blind':blindr,'online':onr},
                     checks,stats,{'train':sktrain,'blind':skblind,'online':skon})
    result['online_family_cohorts']={'held_out_familiar_graphs':len(onr_id),'unseen_family_graphs':len(onr_ood)}
    result['blind_paired_success']={
       'route_goals':sum(r['route_goal'] for r in blind),
       'reset_goals':sum(r['reset_goal'] for r in blind),
       'different_outcomes':sum(r['goal_diff']!=0 for r in blind),
       'route_wins':sum(r['goal_diff']>0 for r in blind),
       'route_losses':sum(r['goal_diff']<0 for r in blind)}
    for fn,obj in [('C54-training.json',train),('C54-blind.json',blind),
                   ('C54-online.json',online),('C54-traces.json',traces),('C54-summary.json',result)]:
        (out/fn).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')
    (out/'C54-norm.json').write_text(json.dumps({'mean':mean.tolist(),'scale':scale.tolist()},sort_keys=True)+'\n')
    print(json.dumps({'training':stats,'blind':result['blind_paired_success'],
        'online_goals':{k:v['goal_events'] for k,v in result['models'].items()},
        'option_turns':{k:v['option_turns'] for k,v in result['models'].items()},
        'paired':result['paired']['C54-ALIGNED__C34-RESET'],
        'gates':result['gates']},indent=2),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('prior');ap.add_argument('out')
    ap.add_argument('--train-reps',type=int,default=6)
    ap.add_argument('--blind-reps',type=int,default=1)
    ap.add_argument('--online-reps',type=int,default=4)
    ap.add_argument('--online-in-family-reps',type=int,default=2)
    a=ap.parse_args();run(a.out,a.prior,a.train_reps,a.blind_reps,a.online_reps,a.online_in_family_reps)

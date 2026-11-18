"""C70 — observed-route-conditioned stochastic actor, synthetic host only.

Causal input: ONLY executed prefix/continuation menu observations and remembered
host-legal edges. The true graph, material mask and hidden shortest paths are
not available to this actor. Training reward is based on executed legal moves,
novelty, published goal distance and completed goals. No model deployment.
"""
from __future__ import annotations
import argparse, base64, collections, copy, hashlib, json, math, os, sys, zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file

HERE=Path(__file__).resolve().parent
DATA_ROOT=Path(os.environ.get('C70_ARCHIVE_ROOT',str(HERE))).resolve()
sys.path.insert(0,str(DATA_ROOT/'frozen/research-source'))
import c60_remote_frontier_rl as c60
from experiment import sha
from c28_temporal_option import scores
from c34_independent_maze import ReArmedWatch
import c69_memory_potency as c69

SEED=700119
FAMILIES=('tree','braid','prim','wilson','kruskal')
DROPS=(0., .5, 1.)
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
BONUS=2.8
HORIZON=95
NFEAT=6
POLICIES=('RESET','FRONTIER','C60-FROZEN','C70-ALIGNED','C70-SHIFTED','STATIC-ROUTE')


def digest(data):return hashlib.sha256(data).hexdigest()

def cohort(base):
    # Exclusion includes all frozen prior graph identities and the immediately
    # predecessor C69 audit. Selection is based on route richness BEFORE suffix.
    prior_ids=c69.known_ids()
    past=json.loads((DATA_ROOT/'output/C69-summary.json').read_text())
    prior_ids.update(x['graph'] for x in past['selected_prefixes'])
    orig_known,orig_seed,orig_quota=c69.known_ids,c69.SEED,c69.FAMILY_QUOTA
    c69.known_ids=lambda: set(prior_ids)
    c69.SEED=SEED;c69.FAMILY_QUOTA=5
    try: states,eligible,receipts,skips,oldcount=c69.prepare_states(base)
    finally:
        c69.known_ids=orig_known;c69.SEED=orig_seed;c69.FAMILY_QUOTA=orig_quota
    # First two per family train, remaining three per family test, all disjoint.
    tr=[];te=[]
    for family in FAMILIES:
        ff=[x for x in states if x[1]['family']==family]
        assert len(ff)==5
        tr.extend(ff[:2]);te.extend(ff[2:])
    assert len(tr)==10 and len(te)==15
    assert not set(x[1]['graph'] for x in tr)&set(x[1]['graph'] for x in te)
    return tr,te,eligible,skips,oldcount


def features(w,led,menu):
    # The only remote 'route' is a route through OBSERVED host menus.
    route=led.route(w.pos,w.goal)
    nxt=tuple(route[1]) if route is not None and len(route)>1 else None
    here=tuple(w.pos)
    dist=max(1,w.distance())
    xs=[]
    for c in menu:
        p=tuple(c.target);k=c.action[0]
        move=k in ('INK','LIFT','BACKTRACK') and p!=here
        progress=float(np.clip((dist-(abs(p[0]-w.goal[0])+abs(p[1]-w.goal[1])))/2.,-1,1))
        turn=(p[0]-here[0])*w.last_dir[1]-(p[1]-here[1])*w.last_dir[0]
        xs.append([int(move and p==nxt),int(move and p not in led.visits),
            progress if move else 0.,int(k=='BACKTRACK'),
            int(move and turn*w.personality>0)-int(move and turn*w.personality<0),
            int(move and led.visits.get(p,0)>2)])
    return np.asarray(xs,np.float32), int(nxt is not None)


def execute(base,opt,st,arm,policy,weight,drop=0.,collect=False,trace=False):
    snap,rngstate,original,receipt,tick,target,info,seed,family,case,distance=st
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c60.c59.c58.c55.cue(w,arm)
    rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rngstate)
    led=c60.c59.c58.damaged(original,info['graph_sha'],tick,drop)
    ctrl=None if policy=='C18-A0' or arm[0]=='none' else ReArmedWatch(opt,None)
    acts=[];decisions=[];path=[];points=[tuple(w.pos)];counts=collections.Counter()
    prior_visit=set(led.visits);route_reads=0;memory_ticks=0
    for t in range(tick,HORIZON):
        if w.stopped or w.distance()==0:break
        led.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        logits=np.asarray(logits,np.float64).copy()
        xx,hasroute=features(w,led,menu)
        route_reads+=1;memory_ticks+=hasroute
        strength=arm[1]
        if ctrl is not None and strength:
            if policy=='FRONTIER':
                f=c60.action_features(w,led,menu)
                logits+=c60.exploration_delta(w,led,menu,f,'C60-FRONTIER',np.zeros(c60.NFEAT,np.float32),strength)
            elif policy=='C60-FROZEN':
                f=c60.action_features(w,led,menu)
                logits+=c60.exploration_delta(w,led,menu,f,'C60-ALIGNED',weight,strength)
            elif policy=='STATIC-ROUTE':
                logits+=3.0*strength*xx[:,0] + .3*strength*xx[:,1]
            elif policy in ('C70-ALIGNED','C70-SHIFTED'):
                logits+=BONUS*strength*np.clip(xx@weight,-1.25,1.25)
        p=c60.softmax(logits);j=int(rng.choice(len(menu),p=p));c=menu[j]
        before=tuple(w.pos);prior=w.distance()
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        led.after(before,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w)
        acts.append((c.action,c.target));points.append(tuple(w.pos))
        novel=tuple(w.pos) not in prior_visit;prior_visit.add(tuple(w.pos))
        goal=int(w.distance()==0)
        # Actual host transitions; never uses hidden shortest-path labels.
        reward=.28*novel + .24*np.clip(prior-w.distance(),-1,1) + 10.*goal - .035
        if c.action[0]=='STOP' and not goal:reward-=1.
        if before==tuple(w.pos):reward-=.06
        if collect:
            decisions.append({'x':xx,'base':np.asarray(logits - (BONUS*strength*np.clip(xx@weight,-1.25,1.25) if policy in ('C70-ALIGNED','C70-SHIFTED') and strength else 0.),np.float32),
               'a':j,'r':float(reward),'strength':strength,'route':hasroute})
        if trace:path.append({'t':t,'p':list(w.pos),'a':c.action[0]})
        counts['novel']+=int(novel);counts['route_matching_action']+=int(xx[j,0]);counts['moves']+=int(tuple(w.pos)!=before)
    return {'goal':int(w.distance()==0),'steps':len(acts),'new_sites':counts['novel'],
       'route_matching_actions':counts['route_matching_action'],'route_reads':route_reads,
       'route_available_ticks':memory_ticks,'distance':int(w.distance()),
       'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),'protected_sha':sha(sorted(w.protected)),
       'host_rejections':w.invalid_attempts,'personality':c69.path_metrics({'start_pos':list(snap.pos),'path':path}) if trace else None,
       'start_pos':list(snap.pos),'target':list(target),'path':path if trace else None},decisions


def train(base,opt,trainstates,mode):
    torch.manual_seed(SEED+303)
    model=nn.Linear(NFEAT,1,bias=False);nn.init.zeros_(model.weight)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.045,weight_decay=.005)
    rng=np.random.default_rng(SEED+410)
    stats=[];updates=0
    # Equal graphs, episode windows, epochs and minibatch update counts.
    # Heads have different policies after initial update, so exposure may differ.
    for epoch in range(3):
        decisions=[];episodes=[]
        for st,rec in trainstates:
            for arm in ARMS[1:]:
                for drop in (0.,1.):
                    weight=model.weight.detach().numpy().reshape(-1).copy()
                    z,steps=execute(base,opt,st,arm,'C70-ALIGNED',weight,drop,collect=True)
                    if not steps:continue
                    returns=[];ret=0.
                    for s in reversed(steps):
                        ret=s['r']+.94*ret;returns.append(ret)
                    returns=returns[::-1]
                    episodes.append((steps,np.asarray(returns,np.float32),rec['graph'],z['goal']))
        if mode=='shifted':
            # Wrong-credit negative control: same trajectories/length distribution,
            # but returns cyclically rolled by a random nonzero offset PER episode.
            corrected=[]
            for steps,rewards,graph,goal in episodes:
                offset=int(rng.integers(1,len(rewards))) if len(rewards)>1 else 0
                corrected.append((steps,np.roll(rewards,offset),graph,goal))
            episodes=corrected
        values=np.concatenate([ret for _,ret,_,_ in episodes]);avg=float(values.mean());sd=max(float(values.std()),1e-4)
        flat=[(s,(float(v)-avg)/sd) for steps,vals,_,_ in episodes for s,v in zip(steps,vals)]
        # Fixed 12 optimizer updates per epoch (36 total) with bounded minibatches.
        for upd in range(12):
            inds=rng.integers(0,len(flat),size=96)
            batch=[flat[int(k)] for k in inds]
            x=torch.tensor(np.stack([a['x'] for a,_ in batch]),dtype=torch.float32)
            frozen=torch.tensor(np.stack([a['base'] for a,_ in batch]),dtype=torch.float32)
            act=torch.tensor([a['a'] for a,_ in batch],dtype=torch.long)
            strength=torch.tensor([a['strength'] for a,_ in batch],dtype=torch.float32)
            adv=torch.tensor([b for _,b in batch],dtype=torch.float32)
            logits=frozen+BONUS*strength[:,None]*torch.clamp(model(x).squeeze(-1),-1.25,1.25)
            logp=torch.log_softmax(logits,dim=1)
            prob=torch.softmax(logits,dim=1)
            entropy=-(prob*logp).sum(-1).mean()
            old=torch.log_softmax(frozen,dim=1)
            kl=(prob*(logp-old)).sum(-1).mean()
            loss=-(logp.gather(1,act[:,None]).squeeze(-1)*adv).mean()+.06*kl-.008*entropy
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
        stats.append({'epoch':epoch,'trajectories':len(episodes),'actions':len(flat),'training_goals':sum(goal for *_,goal in episodes),
                      'weight_norm':float(torch.linalg.vector_norm(model.weight).item())})
    return model,{'mode':mode,'updates':updates,'epochs':stats}


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c60.HORIZON=HORIZON
    base,opt,_=c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(DATA_ROOT/'frozen/predecessor_C42')
    trainst,testst,selection,skips,excluded=cohort(base)
    fit={};heads={};checks={}
    for mode in ('aligned','shifted'):
        model,metrics=train(base,opt,trainst,mode);fit[mode]=metrics
        name='C70-'+mode.upper();p=output/(name+'.safetensors')
        save_file(model.state_dict(),str(p));checks[name]=digest(p.read_bytes())
        heads[name]=model.weight.detach().numpy().reshape(-1).copy()
    weights60=np.asarray(json.loads((DATA_ROOT/'publication/C60-summary.json').read_text())['weights']['C60-ALIGNED'],np.float32)
    rows=[];trace_rows=[];aggregates=collections.defaultdict(lambda:collections.Counter())
    n_actions=0
    for gi,(st,receipt) in enumerate(testst):
        print('C70 test graph',gi+1,'/',len(testst),receipt['family'],flush=True)
        for arm in ARMS:
            for drop in DROPS:
                outs={}
                for name in POLICIES:
                    weight=weights60 if name=='C60-FROZEN' else heads.get(name,np.zeros(NFEAT,np.float32))
                    z,_=execute(base,opt,st,arm,name,weight,drop,trace=True)
                    assert z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha']
                    n_actions+=z['steps'];trace=z.pop('path')
                    outs[name]=z
                    if arm[1]:
                        key=f'{drop}/{arm[1]}/{name}'
                        ag=aggregates[key];ag['goals']+=z['goal'];ag['new_sites']+=z['new_sites']
                        ag['route_matches']+=z['route_matching_actions'];ag['route_ticks']+=z['route_available_ticks']
                    if gi<3 and arm[1]==1. and drop in (0.,1.) and name in ('C60-FROZEN','C70-ALIGNED','STATIC-ROUTE'):
                        trace_rows.append({'family':receipt['family'],'graph':receipt['graph'],'policy':name,'drop':drop,
                                          'start':z['start_pos'],'goal_pos':z['target'],'goal':z['goal'],
                                          'path':trace,'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
                if arm[0]=='none':
                    assert len({(z['action_sha'],z['ink_sha']) for z in outs.values()})==1
                rows.append({'graph':receipt['graph'],'family':receipt['family'],'tick':receipt['tick'],'arm':list(arm),
                             'drop':drop,'prefix_sha':receipt['prefix_sha'],'outcomes':outs})
    summary={'schema':'driv3r-c70-observed-memory-actor/1','seed':SEED,'source_sha256':digest(Path(__file__).read_bytes()),
          'train_graphs':[r for _,r in trainst],'test_graphs':[r for _,r in testst],
          'excluded_graph_count':excluded,'selection':selection,'eligibility_skips':skips,
          'fitness':fit,'checkpoint_sha256':checks,'weights':{k:v.tolist() for k,v in heads.items()},
          'conditions':len(rows),'executed_suffixes':len(rows)*len(POLICIES),'actions_executed':n_actions,
          'aggregate':{k:dict(v) for k,v in aggregates.items()},'gallery_traces':len(trace_rows),
          'limitations':['Synthetic host, not CyberSand native gameplay','Observed waypoint already lies within past explored route; not remote unknown goal',
             'Different on-policy episode lengths mean training action exposure not equal despite identical updates',
             'Shifted-credit control cyclically shifts within each episode; leakage from within-episode reward autocorrelation remains',
             'Small topology-stratified heldout cohort and exploratory paired statistics',
             'Only simple movement/ink behavioral metrics, not independent worm-personality acceptance']}
    ledger={'schema':'driv3r-c70-executed-ledger/1','summary':summary,'rows':rows,'gallery':trace_rows}
    (output/'C70-summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (output/'C70-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(json.dumps(ledger,sort_keys=True,separators=(',',':')).encode(),9)).decode()+'\n')
    c69.gallery(trace_rows,output/'C70-gallery.svg')
    print(json.dumps({'fit':fit,'strong_intact':{p:summary['aggregate'].get(f'0.0/1.0/{p}') for p in POLICIES},
                      'strong_erased':{p:summary['aggregate'].get(f'1.0/1.0/{p}') for p in POLICIES},
                      'suffixes':summary['executed_suffixes'],'actions':n_actions},indent=2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output');args=ap.parse_args();run(args.output)

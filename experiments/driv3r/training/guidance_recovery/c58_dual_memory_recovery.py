"""C58 dual-memory-loss guidance: host-legal, synthetic-only DRIV3R experiment.

No CyberSand native integration or replacement ABI. Never queries unseen geometry
for training labels or action selection. Frozen parent code is imported untouched.
"""
from __future__ import annotations
import collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
import c57_trace_restoration as c57
import c56_memory_recovery as c56
import c55_observed_waypoint_distillation as c55
from c28_temporal_option import scores,probs
from c34_independent_maze import ReArmedWatch
from experiment import sha

SEED=580118
DROPS=(0.,.5,1.)
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
POLICIES=('C18-A0','C34-RESET','C57-TRACE','C57-HEURISTIC','C58-ALIGNED','C58-SHUFFLED')
HORIZON=78


def damaged(log,graph,tick,drop):
    """Erase both remembered menus and executed-action entries independently.

    Previously observed node visits remain as counts, not traversable edges.
    The current menu is always freshly observed and all executed actions are
    host revalidated. Neither map nor graph oracle is read in this function.
    """
    ledger=c56.erase(log,graph,tick,drop)
    if drop:
        ledger.actions=[a for i,a in enumerate(ledger.actions)
          if int.from_bytes(hashlib.sha256(f'c58-actions/{graph}/{tick}/{i}'.encode()).digest()[:8],'big')/(1<<64)>=drop]
    return c57.restore(ledger)


def teacher(base,opt,states):
    rows=[];opps=collections.Counter()
    for gt,st in states:
      snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
      for drop in DROPS:
       for arm in ARMS[1:]:
        w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
        advice=c55.cue(w,arm);led=damaged(log,info['graph_sha'],tick,drop)
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rngstate)
        reset=ReArmedWatch(opt,None)
        for j in range(7):
            if w.stopped or w.distance()==0:break
            menu=led.see(w)
            valid,nxt,mode=c57.recommendation(w,led,menu,advice)
            opps[mode]+=1
            if valid:
                for i,c in enumerate(menu[:8]):
                    if c.legal:
                        rows.append({'graph':info['graph_sha'],'goaltype':gt,'drop':drop,'arm':arm[1],
                          'x':c55.feature(w,led,menu,c,nxt).tolist(),'y':int(i in valid),
                          'state':sha([w.pos,w.goal,led.actions[-5:],drop,arm,j]),
                          'mode':mode,'kind':c.action[0]})
            menu,logits,_,_=reset.modify(base,w,advice,1.)
            c=menu[int(rng.choice(len(menu),p=probs(logits)))];before=tuple(w.pos)
            assert c.legal and w.apply(c) and w.invalid_attempts==0
            led.after(before,c);reset.after(c);advice.advance(w)
    return rows,dict(opps)


def execute(base,opt,snap,rngstate,log,tick,target,arm,name,heads,graph,drop,trace=False):
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c55.cue(w,arm);rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(rngstate)
    led=damaged(log,graph,tick,drop)
    retained={'menus':len(led.neighbors),'actions':len(led.actions),
              'route':int(led.route(w.pos,w.goal) is not None)}
    ctrl=None if name=='C18-A0' or arm[0]=='none' else ReArmedWatch(opt,None)
    path=[];acts=[];counts=collections.Counter()
    for t in range(tick,HORIZON):
        if w.stopped or w.distance()==0:break
        led.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        if ctrl is not None and name not in ('C18-A0','C34-RESET'):
            legal,nxt,mode=c57.recommendation(w,led,menu,advice,allow_fallback=name!='C57-TRACE')
            if legal:
                counts['offered_'+mode]+=1
                delta=np.zeros(len(menu),np.float32)
                if name in ('C57-TRACE','C57-HEURISTIC'):
                    delta[legal]=3.*arm[1]
                else:
                    with torch.no_grad():
                        x=torch.from_numpy(np.asarray([c55.feature(w,led,menu,c,nxt) for c in menu[:8]],np.float32))
                        logits_model=heads[name](x).numpy()
                    delta[:8]=3.*arm[1]*np.tanh(np.clip(logits_model,-20,20)/2.)
                logits=np.asarray(logits,dtype=np.float32)+delta
        c=menu[int(rng.choice(len(menu),p=probs(logits)))];before=tuple(w.pos)
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        led.after(before,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w);acts.append((c.action,c.target))
        if c.action[0] in ('INK','LIFT'):
            pref=float(c.features[13]-c.features[14])*w.personality
            counts['personality_preferred']+=int(pref>0)
            counts['personality_opposed']+=int(pref<0)
            counts['ink_actions']+=int(c.action[0]=='INK')
            counts['lift_actions']+=int(c.action[0]=='LIFT')
        if trace:path.append({'t':t,'p':list(w.pos),'a':c.action[0]})
    return {'goal':int(w.distance()==0),'steps':len(acts),'distance':w.distance(),
      'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),'protected_sha':sha(sorted(w.protected)),
      'host_rejections':w.invalid_attempts,'retained':retained,'counts':dict(counts),
      'path':path if trace else None}


def run(prior,out,train_reps=2,online_reps=1):
    prior=Path(prior);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c55.TICKS=(9,)
    base,opt,_=c55.c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    # Frozen historical geometry IDs from C56 and C57, not their outcomes.
    old=set()
    for p in (prior/'C56-summary.json',prior/'C57-summary.json'):
        prev=json.loads(p.read_text())
        for k in ('train','online'):
            old.update(x['graph'] for x in prev['cohorts'][k])
    tr,trec=c55.c54.c52.worlds(c55.c54.c52.TRAIN_FAMILIES,c55.c54.c52.c50.c47.c41.world,SEED+9021,train_reps,exclude=old)
    trainstates=[]
    for gt in ('observed','remote'):
        ss,_=c56.states_for_goal(base,tr,gt)
        trainstates.extend((gt,x) for x in ss)
    rows,opps=teacher(base,opt,trainstates)
    assert len(rows)>300 and len({r['graph'] for r in rows})>=8
    models,fit=c55.train(rows,epochs=8)
    heads={'C58-ALIGNED':models['C55-ALIGNED'],'C58-SHUFFLED':models['C55-SHUFFLED']}
    checks={}
    for name,model in heads.items():
        path=out/(name+'.safetensors');save_file(model.state_dict(),str(path));checks[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    excluded=old|{x['graph'] for x in trec}
    uns,ur=c55.c54.c52.worlds(c55.c54.c52.ONLINE_FAMILIES,c55.c54.c52.c50.c47.c42.world,SEED+902188,online_reps,exclude=excluded)
    seen,fr=c55.c54.c52.worlds(c55.c54.c52.TRAIN_FAMILIES,c55.c54.c52.c50.c47.c41.world,SEED+802188,online_reps,exclude=excluded|{x['graph'] for x in ur})
    states=[];skips={}
    for gt in ('observed','remote'):
        ss,skip=c56.states_for_goal(base,uns+seen,gt);states.extend((gt,x) for x in ss);skips[gt]=skip
    allrows=[];traces=[];agg=collections.defaultdict(collections.Counter);paired=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
    parity=0;regress=collections.Counter();observed_corrupt=collections.defaultdict(list)
    for gt,st in states:
      snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
      for drop in DROPS:
       for arm in ARMS:
        z={name:execute(base,opt,snap,rngstate,log,tick,target,arm,name,heads,info['graph_sha'],drop)
           for name in POLICIES}
        assert all(a['host_rejections']==0 and a['protected_sha']==receipt['protected_sha'] for a in z.values())
        if arm[0]=='none':
            assert len({(v['action_sha'],v['ink_sha']) for v in z.values()})==1
            parity+=1
        else:
            condition=(gt,drop,fam,arm[1]); gc=paired[(gt,drop,arm[1],fam)][info['graph_sha']]
            for name,v in z.items():
                ac=agg[(gt,drop,fam,name)];ac['goals']+=v['goal'];ac['episodes']+=1
                for k,n in v['counts'].items():ac[k]+=n
                gc[name]+=v['goal'];regress[(gt,drop,name)]+=int(v['goal']<z['C34-RESET']['goal'])
            observed_corrupt[(gt,drop)].append(z['C57-TRACE']['retained'])
        allrows.append({'graph':info['graph_sha'],'family':fam,'case':case,'seed':seed,'tick':tick,
          'personality':int(snap.personality),'goaltype':gt,'target':list(target),'drop':drop,
          'arm':arm,'prefix_sha':receipt['prefix_sha'],'prefix_ink_sha':receipt['ink_sha'],
          'outcomes':z})
        if arm==('backtrack',1.) and gt=='observed' and len(traces)<18 and drop in (0.,.5,1.):
            for name in ('C34-RESET','C57-HEURISTIC','C58-ALIGNED'):
                if len(traces)>=18:break
                v=execute(base,opt,snap,rngstate,log,tick,target,arm,name,heads,info['graph_sha'],drop,True)
                assert (v['action_sha'],v['ink_sha'])==(z[name]['action_sha'],z[name]['ink_sha'])
                traces.append({'graph':info['graph_sha'],'seed':seed,'family':fam,'case':case,
                   'tick':tick,'target':list(target),'drop':drop,'policy':name,'prefix':receipt,'outcome':v})
    pairs={}
    for key,bygraph in paired.items():
        for name in POLICIES[2:]:
            delta=np.asarray([v[name]-v['C34-RESET'] for v in bygraph.values()],int)
            rng=np.random.default_rng(SEED+len(delta)+len(name))
            boot=delta[rng.integers(len(delta),size=(1000,len(delta)))].sum(axis=1)
            pairs[f'{key}/{name}']={'goal_delta':int(delta.sum()),'positive_graphs':int(sum(delta>0)),
              'negative_graphs':int(sum(delta<0)),'exploratory_graph_bootstrap95':np.quantile(boot,[.025,.975]).tolist()}
    summary={'schema':'driv3r-c58-joint-memory/1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'train_graphs':len(trec),'online_graphs':len(ur)+len(fr),'training':fit,'opportunities':opps,
       'checkpoints':checks,'unguided_parity':parity,'online_rows':len(allrows),
       'cohorts':{'train':trec,'online':ur+fr},'skips':skips,
       'models':{str(k):dict(v) for k,v in agg.items()},'regressions':{str(k):v for k,v in regress.items()},
       'paired':pairs,'retention':{str(k):{'route_retained':sum(v['route'] for v in vals),
          'total':len(vals),'mean_actions':np.mean([v['actions'] for v in vals]).item() if vals else None,
          'mean_menus':np.mean([v['menus'] for v in vals]).item() if vals else None} for k,vals in observed_corrupt.items()},
       'caveats':['External synthetic host only; no native CyberSand acceptance',
         'Training target imitates a local accessible heuristic; no global shortest path oracle',
         'Joint erasure removes historical neighbor menus and logged action edges, retains visited-node counts',
         'Fresh legal host menu retained by design; no-policy may edit protected geometry',
         'Target selection intentionally separates observed waypoint versus previously unseen remote goal',
         'Adaptive experiment series makes confidence intervals exploratory',
         'Personality measured as directional action preference only; whole-trajectory personality unproven',
         'No immutable Actions checkpoint artifact custody']}
    for name,data in [('C58-summary.json',summary),('C58-training.json',rows),('C58-online.json',allrows),('C58-traces.json',traces)]:
        (out/name).write_text(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'train':fit,'online_graphs':summary['online_graphs'],'parity':parity,
      'goals':{k:v['goals'] for k,v in summary['models'].items()}},indent=2),flush=True)
    return summary

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('output');p.add_argument('--train-reps',type=int,default=2);p.add_argument('--online-reps',type=int,default=1)
 a=p.parse_args();run(a.prior,a.output,a.train_reps,a.online_reps)

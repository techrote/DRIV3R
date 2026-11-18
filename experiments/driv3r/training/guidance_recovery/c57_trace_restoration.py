"""C57: host-executed action-history restoration and partial-memory recovery.

DRIV3R synthetic-only research, not native CyberSand compatibility proof. The
immutable historical d_common fixture and all other worker files are untouched.
The goal is assigned by the host. Recovered edges are derived EXCLUSIVELY from
previously executed actions, and the host revalidates each attempted next action.
"""
from __future__ import annotations
import collections, copy, hashlib, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
import c55_observed_waypoint_distillation as c55
import c56_memory_recovery as c56
from experiment import sha
from c28_temporal_option import scores, probs
from c34_independent_maze import ReArmedWatch

SEED=570118
ARMS=(('none',0.),('backtrack',.35),('backtrack',1.))
DROPS=(0.,.5)
POLICIES=('C18-A0','C34-RESET','C57-TRACE','C57-HEURISTIC','C57-ALIGNED','C57-SHUFFLED')
HORIZON=78


def restore(ledger):
    """Recover directed edges proved by executed actions, never inverse edges.

    This specifically tests C56's memory-loss model: local neighbor-menu maps
    were erased, but the independent action log was deliberately retained.
    Historic authorization is NOT assumed current; each move is revalidated.
    """
    out=copy.deepcopy(ledger)
    for a,b,k in out.actions:
        a,b=tuple(a),tuple(b)
        if abs(a[0]-b[0])+abs(a[1]-b[1])!=1:continue
        out.neighbors[a]=tuple(sorted(set(out.neighbors.get(a,()))|{b}))
    return out


def recommendation(w,ledger,menu,advice,allow_fallback=True):
    if advice.active(w)!=1:return [],None,'inactive'
    route=ledger.route(w.pos,w.goal)
    if route is not None and len(route)>1:
        nxt=route[1]
        valid=[i for i,c in enumerate(menu[:8]) if c.legal and tuple(c.target)==nxt]
        if valid:return valid,nxt,'trace-route'
    if not allow_fallback:return [],None,'missing-route'
    valid=[(i,c) for i,c in enumerate(menu[:8]) if c.legal]
    if not valid:return [],None,'no-legals'
    # No hidden map reads: only supplied goal, current host-legal menu, own visits.
    recent={tuple(b) for a,b,k in ledger.actions[-6:]}
    def rank(ic):
        i,c=ic; q=tuple(c.target)
        return (float(w.distance(q))+.32*min(8,ledger.visits[q])+
                .24*int(q in recent),int(c.action[0]=='LIFT'),i)
    winner=min(valid,key=rank)[1].target
    return [i for i,c in valid if tuple(c.target)==tuple(winner)],tuple(winner),'local-fallback'


def teacher_rows(base,option,states):
    rows=[];opps=collections.Counter()
    for goaltype,st in states:
      snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
      for drop in DROPS:
       for arm in ARMS[1:]:
        w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
        advice=c55.cue(w,arm);ledger=restore(c56.erase(log,info['graph_sha'],tick,drop))
        rr=np.random.default_rng();rr.bit_generator.state=copy.deepcopy(rngstate)
        ctrl=ReArmedWatch(option,None)
        for j in range(8):
            if w.stopped or w.distance()==0:break
            menu=ledger.see(w)
            valid,nxt,mode=recommendation(w,ledger,menu,advice)
            opps[mode]+=1
            if valid:
                for i,c in enumerate(menu[:8]):
                    if not c.legal:continue
                    rows.append({'graph':info['graph_sha'],'family':fam,'case':case,'goaltype':goaltype,
                      'drop':drop,'arm':arm[1],'j':j,'mode':mode,'x':c55.feature(w,ledger,menu,c,nxt).tolist(),
                      'y':int(i in valid),'state':sha([w.pos,w.goal,ledger.actions[-5:],drop,arm]),
                      'kind':c.action[0]})
            menu,logits,_,_=ctrl.modify(base,w,advice,1.)
            c=menu[int(rr.choice(len(menu),p=probs(logits)))];start=tuple(w.pos)
            assert c.legal and w.apply(c) and not w.invalid_attempts
            ledger.after(start,c);ctrl.after(c);advice.advance(w)
    return rows,dict(opps)


def rollout(base,option,snap,rngstate,log,tick,target,arm,policy,heads,graph,drop,trace=False):
    w=copy.deepcopy(snap);w.goal=tuple(target);w.goal_change=True;w.goals_seen+=1;w.stuck=2
    advice=c55.cue(w,arm);r=np.random.default_rng();r.bit_generator.state=copy.deepcopy(rngstate)
    ledger=restore(c56.erase(log,graph,tick,drop))
    # Distinguish no guidance from implicit RESET (no control when none issued).
    ctrl=None if policy=='C18-A0' or arm[0]=='none' else ReArmedWatch(option,None)
    acts=[];path=[];counts=collections.Counter()
    for t in range(tick,HORIZON):
        if w.stopped or w.distance()==0:break
        ledger.see(w)
        if ctrl is None:menu,logits,_,_=scores(base,w,advice,1.)
        else:menu,logits,_,_=ctrl.modify(base,w,advice,1.)
        if policy not in ('C18-A0','C34-RESET'):
            valid,nxt,mode=recommendation(w,ledger,menu,advice,allow_fallback=policy!='C57-TRACE')
            if valid:
                counts[mode]+=1
                if policy in ('C57-TRACE','C57-HEURISTIC'):
                    delta=np.zeros(len(menu),np.float32);delta[valid]=3.*arm[1]
                else:
                    with torch.no_grad():
                        xs=np.asarray([c55.feature(w,ledger,menu,c,nxt) for c in menu[:8]],np.float32)
                        z=heads[policy](torch.from_numpy(xs)).numpy()
                    delta=np.zeros(len(menu),np.float32)
                    delta[:8]=3.*arm[1]*np.tanh(np.clip(z,-20.,20.)/2.)
                logits=np.asarray(logits,dtype=np.float32)+delta
        c=menu[int(r.choice(len(menu),p=probs(logits)))]; start=tuple(w.pos)
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        ledger.after(start,c)
        if ctrl is not None:ctrl.after(c)
        advice.advance(w);acts.append((c.action,c.target))
        if c.action[0] in ('INK','LIFT'):
            turn=float(c.features[13]-c.features[14])*w.personality
            counts['preferred_ink']+=int(turn>0);counts['opposed_ink']+=int(turn<0)
        if trace:path.append({'t':t,'p':list(w.pos),'a':c.action[0]})
    assert sha(sorted(w.protected))==sha(sorted(snap.protected))
    return {'goal':int(w.distance()==0),'steps':len(acts),'distance':w.distance(),
       'action_sha':sha(acts),'ink_sha':sha(sorted(w.inked)),
       'protected_sha':sha(sorted(w.protected)),'host_rejections':w.invalid_attempts,
       'counts':dict(counts),'path':path if trace else None}


def run(prior,output,train_reps=2,online_reps=2):
    prior=Path(prior);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    c55.TICKS=(9,)
    base,opt,_=c55.c54.c52.c50.c47.c44.prerequisites(prior/'predecessor_C42')
    history=set()
    for name in ('C55-training-records.json','C55-blind-records.json','C55-online.json','C55-external-online.json'):
        history.update(r['graph'] for r in json.loads((prior/'previous_C55'/name).read_text()))
    for r in json.loads((prior/'C56-summary.json').read_text())['cohorts']['train'] + json.loads((prior/'C56-summary.json').read_text())['cohorts']['online']:
        history.add(r['graph'])
    trainw,trainrec=c55.c54.c52.worlds(c55.c54.c52.TRAIN_FAMILIES,c55.c54.c52.c50.c47.c41.world,SEED+1823,train_reps,exclude=history)
    trainst=[]
    for gt in ('observed','remote'):
        states,_=c56.states_for_goal(base,trainw,gt)
        trainst.extend((gt,s) for s in states)
    rows,opps=teacher_rows(base,opt,trainst)
    assert len(rows)>300 and len({r['graph'] for r in rows})>=8
    heads,training=c55.train(rows,epochs=8)
    neural={'C57-ALIGNED':heads['C55-ALIGNED'],'C57-SHUFFLED':heads['C55-SHUFFLED']}
    check={}
    for name,head in neural.items():
        file=output/(name+'.safetensors');save_file(head.state_dict(),str(file))
        check[name]=hashlib.sha256(file.read_bytes()).hexdigest()
    excluded=history|{r['graph'] for r in trainrec}
    onlineU,ou=c55.c54.c52.worlds(c55.c54.c52.ONLINE_FAMILIES,c55.c54.c52.c50.c47.c42.world,SEED+900071,online_reps,exclude=excluded)
    onlineF,of=c55.c54.c52.worlds(c55.c54.c52.TRAIN_FAMILIES,c55.c54.c52.c50.c47.c41.world,SEED+800071,online_reps,exclude=excluded|{r['graph'] for r in ou})
    states=[];skips={}
    for gt in ('observed','remote'):
        ss,sk=c56.states_for_goal(base,onlineU+onlineF,gt)
        states.extend((gt,s) for s in ss);skips[gt]=sk
    outcomes=[];traces=[];summary=collections.defaultdict(lambda:collections.Counter())
    pergraph=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.Counter()))
    parity=0;regress=collections.Counter()
    for gt,st in states:
      snap,rngstate,log,receipt,tick,target,info,seed,fam,case,distance=st
      for drop in DROPS:
       for arm in ARMS:
        z={name:rollout(base,opt,snap,rngstate,log,tick,target,arm,name,neural,info['graph_sha'],drop)
           for name in POLICIES}
        assert all(t['host_rejections']==0 and t['protected_sha']==receipt['protected_sha'] for t in z.values())
        if arm[0]=='none':
            assert len({(v['action_sha'],v['ink_sha']) for v in z.values()})==1
            parity+=1
        else:
            key=(gt,str(drop),arm[1]); by=pergraph[key][info['graph_sha']]
            for name,val in z.items():
                summary[(gt,str(drop),name)]['goals']+=val['goal']
                summary[(gt,str(drop),name)]['episodes']+=1
                summary[(gt,str(drop),name)]['trace_turns']+=val['counts'].get('trace-route',0)
                summary[(gt,str(drop),name)]['fallback_turns']+=val['counts'].get('local-fallback',0)
                summary[(gt,str(drop),name)]['preferred_ink']+=val['counts'].get('preferred_ink',0)
                summary[(gt,str(drop),name)]['opposed_ink']+=val['counts'].get('opposed_ink',0)
                by[name]+=val['goal']
                regress[(gt,str(drop),name)]+=int(val['goal']<z['C34-RESET']['goal'])
        outcomes.append({'goal_type':gt,'drop':drop,'graph':info['graph_sha'],'family':fam,
           'case':case,'seed':seed,'tick':tick,'arm':arm,'target':target,'prefix_sha':receipt['prefix_sha'],
           'prefix_ink_sha':receipt['ink_sha'],'outcomes':z})
        if arm==('backtrack',1.) and len(traces)<18 and gt=='observed':
            for name in ('C34-RESET','C57-TRACE','C57-HEURISTIC'):
                if len(traces)>=18:break
                tr=rollout(base,opt,snap,rngstate,log,tick,target,arm,name,neural,info['graph_sha'],drop,True)
                assert (tr['action_sha'],tr['ink_sha'])==(z[name]['action_sha'],z[name]['ink_sha'])
                traces.append({'graph':info['graph_sha'],'family':fam,'case':case,'seed':seed,'tick':tick,
                   'target':target,'drop':drop,'policy':name,'prefix':receipt,'outcome':tr})
    pair={}
    for key,graphs in pergraph.items():
        for name in ('C57-TRACE','C57-HEURISTIC','C57-ALIGNED','C57-SHUFFLED'):
            diffs=np.array([c[name]-c['C34-RESET'] for c in graphs.values()])
            rand=np.random.default_rng(SEED+len(graphs)+len(name)+int(key[2]*100))
            boots=diffs[rand.integers(len(diffs),size=(1000,len(diffs)))].sum(1)
            pair[f'{key}:{name}']={'delta':int(diffs.sum()),'positive_graphs':int(sum(diffs>0)),
              'negative_graphs':int(sum(diffs<0)), 'descriptive_bootstrap95':np.quantile(boots,[.025,.975]).tolist()}
    report={'schema':'driv3r-c57-trace-restoration/1','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'train_graphs':len(trainrec),'online_graphs':len(ou)+len(of),'historical_excluded_graphs':len(history),
       'train_rows':len(rows),'training':training,'train_opportunities':opps,
       'checkpoint_sha256':check,'online_rows':len(outcomes),'unguided_parity':parity,
       'models':{f'{a}/{b}/{name}':dict(z) for (a,b,name),z in summary.items()},
       'paired':pair,'regressions':{f'{a}/{b}/{name}':int(v) for (a,b,name),v in regress.items()},
       'skips':skips,'cohorts':{'train':trainrec,'online':ou+of},
       'caveats':['Synthetic host only, not CyberSand native ABI','Erasure removes historical menu maps but preserves action log',
        'Recovered directed edges were legal when executed but require fresh host validation',
        'Host-assigned observed frontiers are easier than previously unseen remote goals',
        'Training labels derive from a transparent local heuristic, not solved remote geometry',
        'Repeated adaptive experiments make descriptive intervals exploratory',
        'No-guidance parity alone does not prove guided personality retention',
        'No immutable GitHub Actions checkpoint custody']}
    for name,obj in [('C57-train.json',rows),('C57-online.json',outcomes),('C57-traces.json',traces),('C57-summary.json',report)]:
        (output/name).write_text(json.dumps(obj,separators=(',',':'),sort_keys=True)+'\n')
    print(json.dumps({'training':training,'online_graphs':report['online_graphs'],'goals':{k:v['goals'] for k,v in report['models'].items()},
      'unguided_parity':parity},indent=2),flush=True)
    return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('prior');p.add_argument('output');p.add_argument('--train-reps',type=int,default=2);p.add_argument('--online-reps',type=int,default=2)
 a=p.parse_args();run(a.prior,a.output,a.train_reps,a.online_reps)

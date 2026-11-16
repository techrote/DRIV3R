"""SIDE-EXIT-COST-001: matched supervised cost-to-recovery from host rollouts.

External synthetic research only. The one-cell menu, material exclusion and
BACKTRACK/RELOCATE_START operations are supplied by the immutable imported World.
A learned student sees raw remaining budget, path length and walk-back distance
for the next open side exit; never a feasibility comparison or future reward.
Training labels come from two actual counterfactual host rollouts, not a hard-
coded cap >= distance rule. Evaluation opens DEVELOPMENT families only.
No native execution, held-out worlds, paid services or game changes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np
from . import policy
from .return_generalization_lab import fixture, force_error, TRAIN_DEPTHS, DEV_DEPTHS, TRAIN_DIRECTIONS, DEV_DIRECTIONS
from .train import _png, sha_file
from .world import MOVE, World, digest, specs

VERSION='side-exit-cost/1'
STAGES=(36,72,128)
HORIZONS=(128,160)
ARMS=('budget_only','linear_geometry','nonlinear_geometry')
TRAIN_BACK=(1,3,5)
DEV_BACK=(2,4)
TRAIN_CAP=(2,4,6,9)
DEV_CAP=(3,5,7,10)
RETURNS=('BACKTRACK','RELOCATE_START')
INPUT=5


def make_world(sp,depth,direction,back,side):
    ds,dirs,ks=(TRAIN_DEPTHS,TRAIN_DIRECTIONS,TRAIN_BACK) if sp.split=='train' else (DEV_DEPTHS,DEV_DIRECTIONS,DEV_BACK) if sp.split=='development' else ((),(),())
    if depth not in ds or direction not in dirs or back not in ks or back>=depth or side not in (-1,1):
        raise ValueError('split, horizon or side-exit partition violation')
    w=fixture(sp,depth,direction)
    dx,dy=MOVE['MOVE_'+direction]
    # A side gap in the wall, invisible from the dead-end cursor unless the
    # policy explicitly walks its OWN ink history backwards. Goal is beyond it.
    sx,sy=-dy*side,dx*side
    anchor=(48+(depth-back)*dx,48+(depth-back)*dy)
    for m in range(1,7):
        p=(anchor[0]+m*sx,anchor[1]+m*sy)
        w.materials.pop(p,None);w.protected.discard(p)
        # A physically closed branch pocket: no external shortcut from START
        # after relocation. The only legal opening is through own live path.
        for sign in (-1,1):
            wall=(p[0]+sign*dx,p[1]+sign*dy)
            if wall != anchor:
                w.materials[wall]=37;w.protected.discard(wall)
    end=(anchor[0]+7*sx,anchor[1]+7*sy)
    w.materials[end]=1;w.protected.add(end)
    goal=(anchor[0]+3*sx,anchor[1]+3*sy)
    if goal in w.materials or goal in w.protected:
        raise ValueError('goal blocked')
    return w,goal


def nearest_exit(w:World):
    # State observable by the agent: scan committed live path, not rollout
    # future or teacher labels; no engineered cap >= distance indicator.
    for distance in range(1,len(w.path)):
        p=w.path[-1-distance]
        if any(not w.occupied((p[0]+dx,p[1]+dy)) for dx,dy in MOVE.values()):
            return distance
    return len(w.path)-1


def raw_features(w,cap,arm):
    if arm not in ARMS or cap<1 or cap>12:raise ValueError('arm/cap')
    budget=(cap-w.recoveries)/12.
    depth=(len(w.path)-1)/12.
    dist=nearest_exit(w)/12.
    # Explicitly no subtraction, threshold bit, hidden reward or A* oracle.
    return np.asarray([budget,depth,dist if arm!='budget_only' else 0.,
                       w.steps/160.,1.],dtype=np.float64)


def step_controller(w,cap,goal,stage,rng):
    menu=w.menu(cap)
    if not menu:raise ValueError('invalid stopped policy continuation')
    if any(a in MOVE for a in menu):
        # Host-observable spatial goal: if the actual goal lies along an open
        # cardinal move, prioritize reaching it; otherwise use imperfect
        # source-defined plural painting teachers, never hidden geometry edits.
        preferred=[a for a,(dx,dy) in MOVE.items() if a in menu and
                   abs(w.cursor[0]+dx-goal[0])+abs(w.cursor[1]+dy-goal[1]) <
                   abs(w.cursor[0]-goal[0])+abs(w.cursor[1]-goal[1])]
        if preferred and abs(w.cursor[0]-goal[0])+abs(w.cursor[1]-goal[1]) <= 3:
            return preferred[0]
        f=np.stack([policy.features(w,a,stage) for a in menu])
        scores=policy.teacher_scores(w,menu,f)
        scores += np.asarray([rng.uniform(-.09,.09) for _ in menu])
        return menu[int(scores.argmax())]
    if 'BACKTRACK' in menu:return 'BACKTRACK'
    if 'RELOCATE_START' in menu:return 'RELOCATE_START'
    return 'STOP'


def play(w,goal,cap,stage,first,steps,seed):
    """Counterfactual on a deep copy, fully host-menu mediated."""
    if first not in RETURNS or first not in w.menu(cap):raise ValueError('illegal fork')
    initial_ink=len(w.ink)
    w=deepcopy(w)
    rows=[w.apply(first,cap)]
    rng=random.Random(seed)
    reached=goal in w.ink
    for _ in range(steps-1):
        if w.stopped or reached:break
        selected=step_controller(w,cap,goal,stage,rng)
        rows.append(w.apply(selected,cap))
        reached=goal in w.ink
    # Reward actual goal completion, additional ink, survival and resource cost.
    # The 8-to-1 goal weight is a declared task objective, NOT a feature.
    reward=(8. if reached else 0.) + .12*(len(w.ink)-initial_ink)
    reward += .1*sum(bool(e['painted']) for e in rows)
    reward -= 2. if w.stopped else 0.
    reward -= .04*(w.recoveries)
    return {'reward':round(reward,4),'reached_goal':reached,'stopped':w.stopped,
            'actions':dict(sorted(Counter(e['chosen'] for e in rows).items())),
            'end_sha':w.state_sha(),'events':rows}


def counterfactual(w,goal,cap,stage,seed):
    arms={a:play(w,goal,cap,stage,a,12,seed) for a in RETURNS}
    # True observed 12-step rollout advantage; never the engineered inequality.
    target=int(arms['BACKTRACK']['reward']>arms['RELOCATE_START']['reward'])
    return target,arms


def corpus(seed,stage,quick=False):
    if stage not in STAGES:raise ValueError('training stage')
    train_depths=TRAIN_DEPTHS[:1] if stage==36 else TRAIN_DEPTHS[:2] if stage==72 else TRAIN_DEPTHS
    ks=TRAIN_BACK[:1] if stage==36 else TRAIN_BACK[:2] if stage==72 else TRAIN_BACK
    rows=[];label_counts=Counter();meta=[];forks=Counter()
    train_specs=specs('train',1)[:1 if quick else 2]
    for sp in train_specs:
        for direction in TRAIN_DIRECTIONS[:1 if quick else 2]:
            for depth in train_depths:
                for back in ks:
                    if back>=depth:continue
                    for cap in TRAIN_CAP[:2 if quick else None]:
                        for side in (-1,1)[:1 if quick else None]:
                            w,goal=make_world(sp,depth,direction,back,side)
                            force_error(w,depth,direction,cap)
                            if set(w.menu(cap))!=set((*RETURNS,'STOP')):raise ValueError('host fork not guaranteed')
                            observed=nearest_exit(w)
                            if observed!=back:raise ValueError('observed return distance differs')
                            outcome,cf=counterfactual(w,goal,cap,stage,seed*17+sp.seed*19+depth*31+back*41+cap*43+side)
                            f={arm:raw_features(w,cap,arm) for arm in ARMS}
                            weight=1+min(3.,abs(cf['BACKTRACK']['reward']-cf['RELOCATE_START']['reward'])/6.)
                            rows.append((f,outcome,weight));label_counts[outcome]+=1
                            forks[f'd{depth}-b{back}-c{cap}-label{outcome}']+=1
                            meta.append((sp.ident,direction,depth,back,side,cap,round(cf['BACKTRACK']['reward'],3),round(cf['RELOCATE_START']['reward'],3)))
    X={a:np.stack([r[0][a] for r in rows]) for a in ARMS}
    return X,np.asarray([r[1] for r in rows]),np.asarray([r[2] for r in rows]),{
        'stage':stage,'examples':len(rows),'positive_labels':label_counts[1],
        'negative_labels':label_counts[0],'label_forks':dict(sorted(forks.items())),
        'counterfactual_digest':digest(meta),'split':'train',
        'depths':list(train_depths),'exit_distances':list(ks)}


class CostRanker:
    """Two genuine CPU-Adam students, plus budget-only input ablation."""
    def __init__(self,seed,arm):
        if arm not in ARMS:raise ValueError('arm')
        self.seed=seed;self.arm=arm;self.updates=0
        rng=np.random.default_rng(seed)
        if arm=='nonlinear_geometry':
            self.p={'W':rng.normal(0,.15,(INPUT,12)),'b':np.zeros(12),
                    'v':rng.normal(0,.15,12),'c':np.zeros(1)}
        else:
            self.p={'v':rng.normal(0,.15,INPUT),'c':np.zeros(1)}
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}

    def logits(self,x):
        if 'W' in self.p:
            h=np.tanh(x@self.p['W']+self.p['b'])
            return h@self.p['v']+self.p['c'][0]
        return x@self.p['v']+self.p['c'][0]

    def probs(self,x):
        logits=np.clip(self.logits(x),-30.,30.)
        return 1./(1.+np.exp(-logits))

    def decide(self,w,cap):
        p=float(self.probs(raw_features(w,cap,self.arm)))
        # Record both options; STOP still in actual host menu and can be chosen
        # later by the continuation, but the specific fork experiment compares
        # these two declared legal recovery actions.
        return ('BACKTRACK' if p>.5 else 'RELOCATE_START'),{'BACKTRACK':p,'RELOCATE_START':1.-p}

    def train(self,X,Y,W,updates,seed):
        rng=np.random.default_rng(seed);loss=[];p=self.p
        for _ in range(updates):
            ix=rng.choice(len(Y),size=min(64,len(Y)),replace=True,p=W/W.sum())
            xx=X[ix];yy=Y[ix];n=len(ix)
            if 'W' in p:
                hidden=np.tanh(xx@p['W']+p['b'])
                logits=hidden@p['v']+p['c'][0]
            else:logits=xx@p['v']+p['c'][0]
            pp=1./(1.+np.exp(-np.clip(logits,-30,30)))
            loss.append(float((-yy*np.log(np.maximum(pp,1e-9))-(1.-yy)*np.log(np.maximum(1.-pp,1e-9))).mean()))
            delta=(pp-yy)/n
            if 'W' in p:
                dh=delta[:,None]*p['v'][None,:]*(1.-hidden**2)
                grads={'W':xx.T@dh,'b':dh.sum(axis=0),'v':hidden.T@delta,'c':np.asarray([delta.sum()])}
            else:grads={'v':xx.T@delta,'c':np.asarray([delta.sum()])}
            norm=sum(float((g*g).sum()) for g in grads.values())**.5
            scale=min(1.,3./max(1e-12,norm));self.updates+=1
            for key,g in grads.items():
                g*=scale
                self.m[key]=.9*self.m[key]+.1*g
                self.v[key]=.999*self.v[key]+.001*g*g
                p[key]-=.025*(self.m[key]/(1-.9**self.updates))/(np.sqrt(self.v[key]/(1-.999**self.updates))+1e-8)
        return {'updates':updates,'initial_loss':round(loss[0],5),'last_10_loss':round(statistics.mean(loss[-10:]),5)}

    def save(self,path):
        np.savez_compressed(path,**self.p,seed=self.seed,arm=self.arm,updates=self.updates,version=VERSION)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            if str(data['version'])!=VERSION:raise ValueError('wrong checkpoint version')
            r=cls(int(data['seed']),str(data['arm']))
            for k in r.p:
                a=data[k]
                if a.shape!=r.p[k].shape or not np.isfinite(a).all():raise ValueError('corrupt weights')
                r.p[k]=a.copy()
            r.updates=int(data['updates'])
        return r


def rollout(sp,model,depth,direction,back,side,cap,horizon):
    if sp.split!='development' or depth not in DEV_DEPTHS or direction not in DEV_DIRECTIONS or back not in DEV_BACK or cap not in DEV_CAP or horizon not in HORIZONS:
        raise ValueError('development partition breach')
    w,goal=make_world(sp,depth,direction,back,side)
    initial=w.base_sha();rows=[]
    for ev in force_error(w,depth,direction,cap):
        rows.append({'chosen':ev['chosen'],'probabilities':None,'intervention':'forced_legal_error','event':ev})
    truth,cf=counterfactual(w,goal,cap,horizon,model.seed+sp.seed)
    st=time.perf_counter_ns();choice,ps=model.decide(w,cap);lat=[(time.perf_counter_ns()-st)/1e6]
    if choice not in w.menu(cap):raise ValueError('illegal neural fork')
    rows.append({'chosen':choice,'probabilities':{k:round(v,8) for k,v in ps.items()},'intervention':None,'event':w.apply(choice,cap)})
    rng=random.Random(model.seed+sp.seed+horizon)
    for t in range(depth+1,horizon):
        if w.stopped:break
        st=time.perf_counter_ns()
        candidate=step_controller(w,cap,goal,horizon,rng)
        lat.append((time.perf_counter_ns()-st)/1e6)
        legal=w.menu(cap)
        actual=candidate;intervention=None
        if t in (37,79,109):
            alternatives=[a for a in legal if a in MOVE and a!=candidate]
            if alternatives:
                actual=alternatives[(model.seed+t)%len(alternatives)]
                intervention={'from':candidate,'to':actual}
        ev=w.apply(actual,cap)
        rows.append({'chosen':candidate,'probabilities':None,'intervention':intervention,'event':ev})
    return {'version':VERSION,'world':sp.ident,'split':'development','seed':model.seed,
            'arm':model.arm,'depth':depth,'direction':direction,'back':back,'side':side,
            'cap':cap,'horizon':horizon,'base_sha':initial,'goal':list(goal),
            'fork':choice,'oracle_label':truth,'fork_correct':int(choice=='BACKTRACK')==truth,
            'counterfactual_rewards':{k:cf[k]['reward'] for k in RETURNS},
            'counterfactual_regret':round(max(cf[k]['reward'] for k in RETURNS)-cf[choice]['reward'],4),
            'goal_reached':goal in w.ink,'rows':rows,'trace_sha':digest([r['event'] for r in rows]),
            'final_sha':w.state_sha(),'ink':[list(p) for p in sorted(w.ink)],
            'metrics':w.metrics(horizon),'action_counts':dict(sorted(Counter(r['event']['chosen'] for r in rows).items())),
            'decision_ms_mean':round(statistics.mean(lat),6),'native_receipts':0}


def replay_episode(ep,model,sp):
    if ep['version']!=VERSION or ep['seed']!=model.seed or ep['arm']!=model.arm or ep['split']!='development' or ep['world']!=sp.ident or ep['native_receipts']!=0:
        raise ValueError('episode identity')
    w,goal=make_world(sp,ep['depth'],ep['direction'],ep['back'],ep['side']);cap=ep['cap'];h=ep['horizon']
    if cap not in DEV_CAP or h not in HORIZONS or ep['base_sha']!=w.base_sha() or ep['goal']!=list(goal):raise ValueError('fixture identity')
    evs=force_error(w,ep['depth'],ep['direction'],cap)
    if [r['event'] for r in ep['rows'][:ep['depth']]]!=evs:raise ValueError('forced source tamper')
    choice,ps=model.decide(w,cap)
    row=ep['rows'][ep['depth']]
    if (row['chosen']!=choice or row['intervention'] is not None or
        set(row['probabilities'])!=set(ps) or
        any(abs(row['probabilities'][k]-p)>2e-8 for k,p in ps.items()) or
        w.apply(choice,cap)!=row['event']):raise ValueError('checkpoint mismatch')
    rng=random.Random(model.seed+sp.seed+h)
    for t,row in enumerate(ep['rows'][ep['depth']+1:],start=ep['depth']+1):
        if w.stopped:raise ValueError('continued after stop')
        candidate=step_controller(w,cap,goal,h,rng)
        legal=w.menu(cap)
        actual=candidate;intervention=None
        if t in (37,79,109):
            options=[a for a in legal if a in MOVE and a!=candidate]
            if options:
                actual=options[(model.seed+t)%len(options)]
                intervention={'from':candidate,'to':actual}
        if (candidate!=row['chosen'] or intervention!=row['intervention'] or
            w.apply(actual,cap)!=row['event']):raise ValueError('host transition mismatch')
    truth,cf=counterfactual(make_forced(sp,ep['depth'],ep['direction'],ep['back'],ep['side'],cap)[0],goal,cap,h,model.seed+sp.seed)
    if not w.stopped and len(ep['rows'])!=h:raise ValueError('truncated nonterminal trajectory')
    if (ep['oracle_label']!=truth or ep['fork_correct']!=(int(choice=='BACKTRACK')==truth) or
        ep['counterfactual_rewards']!={k:cf[k]['reward'] for k in RETURNS} or
        abs(ep['counterfactual_regret']-round(max(cf[k]['reward'] for k in RETURNS)-cf[choice]['reward'],4))>1e-6 or
        ep['goal_reached']!=(goal in w.ink) or ep['metrics']!=w.metrics(h) or
        ep['final_sha']!=w.state_sha() or ep['trace_sha']!=digest([r['event'] for r in ep['rows']]) or
        ep['action_counts']!=dict(sorted(Counter(r['event']['chosen'] for r in ep['rows']).items())) or
        ep['ink']!=[list(p) for p in sorted(w.ink)]):raise ValueError('trajectory metric tamper')
    return len(ep['rows'])


def make_forced(sp,depth,direction,back,side,cap):
    w,goal=make_world(sp,depth,direction,back,side);force_error(w,depth,direction,cap);return w,goal


def aggregate(episodes):
    return {'episodes':len(episodes),'correct_fork':sum(e['fork_correct'] for e in episodes),
            'goals_reached':sum(e['goal_reached'] for e in episodes),
            'early_stop':sum(e['metrics']['stopped_early'] for e in episodes),
            'mean_ink':round(statistics.mean(e['metrics']['unique_painted_cells'] for e in episodes),3),
            'mean_counterfactual_regret':round(statistics.mean(e['counterfactual_regret'] for e in episodes),4),
            'mean_decision_ms':round(statistics.mean(e['decision_ms_mean'] for e in episodes),6)}


def dependencies():
    p=Path(__file__).parent
    return {x:sha_file(p/x) for x in ('side_exit_cost_lab.py','return_generalization_lab.py','world.py','policy.py','train.py')}


def run(out,seed=600,updates=30,quick=False):
    if out.exists():raise FileExistsError('immutable evidence path')
    if not 0<=seed<100000 or not 1<=updates<=120:raise ValueError('optimizer bound')
    out.mkdir(parents=True);start=time.perf_counter();models={a:CostRanker(seed,a) for a in ARMS};train=[]
    for stage in (STAGES[:1] if quick else STAGES):
        X,Y,W,meta=corpus(seed,stage,quick)
        train.append({'stage':stage,'meta':meta,'fit':{a:m.train(X[a],Y,W,updates,seed+stage*19) for a,m in models.items()}})
    checkpoints={}
    for arm,m in models.items():
        name=f'{arm}-{seed}.npz';m.save(out/name);c=CostRanker.load(out/name)
        if c.updates!=m.updates or any(not np.array_equal(c.p[k],m.p[k]) for k in m.p):raise ValueError('restore failed')
        checkpoints[arm]={'name':name,'sha256':sha_file(out/name),'parameters':sum(x.size for x in m.p.values()),'updates':m.updates}
    dev=specs('development',1)[:1]
    (out/'traces').mkdir();episodes=[];decisions=0
    for sp in dev:
        directions=DEV_DIRECTIONS[:1] if quick else DEV_DIRECTIONS
        for direction in directions:
            for depth in DEV_DEPTHS[:1 if quick else None]:
                for back in DEV_BACK[:1 if quick else None]:
                    for cap in (DEV_CAP[:2] if quick else (3,7)):
                        side=(-1 if (sp.seed+depth+cap)%2 else 1)
                        for horizon in (HORIZONS[:1] if quick else HORIZONS[-1:]):
                            for arm,m in models.items():
                                ep=rollout(sp,m,depth,direction,back,side,cap,horizon)
                                decisions+=replay_episode(ep,m,sp)
                                name=f'{arm}-{sp.ident}-{direction}-d{depth}-b{back}-c{cap}-h{horizon}'
                                (out/'traces'/(name+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
                                if depth==DEV_DEPTHS[-1] and back==DEV_BACK[-1] and horizon==160 and cap in (3,10) and sp==dev[0] and direction=='W':
                                    image_world,_=make_world(sp,depth,direction,back,side)
                                    for row in ep['rows']:image_world.apply(row['event']['chosen'],cap)
                                    _png(out/'traces'/(name+'.png'),image_world)
                                episodes.append(ep)
    summary={arm:aggregate([e for e in episodes if e['arm']==arm]) for arm in ARMS}
    margins={str(k):{arm:aggregate([e for e in episodes if e['arm']==arm and e['cap']-e['back']==k]) for arm in ARMS} for k in sorted({e['cap']-e['back'] for e in episodes})}
    result={'version':VERSION,'seed':seed,'updates_per_stage':updates,'train':train,
            'checkpoints':checkpoints,'episodes':len(episodes),'decisions':decisions,
            'summary':summary,'budget_minus_distance':margins,'source_sha256':dependencies(),
            'train_depths':list(TRAIN_DEPTHS),'dev_depths':list(DEV_DEPTHS),
            'train_directions':list(TRAIN_DIRECTIONS),'dev_directions':list(DEV_DIRECTIONS),
            'train_exit_distances':list(TRAIN_BACK),'dev_exit_distances':list(DEV_BACK),
            'elapsed_s':round(time.perf_counter()-start,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'quick':quick,'heldout_uninspected':True,'native_receipts':0,
            'paid_compute':False,'synthetic_not_native':True}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    manifest={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file()}
    (out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return result


def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=dependencies() or not r['heldout_uninspected'] or r['native_receipts']!=0:raise ValueError('source identity')
    manifest=json.loads((out/'SHA256SUMS.json').read_text())
    actual={str(p.relative_to(out)):sha_file(p) for p in out.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
    if actual!=manifest:raise ValueError('artifact manifest mismatch')
    models={arm:CostRanker.load(out/c['name']) for arm,c in r['checkpoints'].items()}
    if set(models)!=set(ARMS):raise ValueError('checkpoint arm missing')
    for arm,m in models.items():
        c=r['checkpoints'][arm]
        if m.seed!=r['seed'] or m.arm!=arm or m.updates!=c['updates'] or c['parameters']!=sum(x.size for x in m.p.values()) or sha_file(out/c['name'])!=c['sha256']:raise ValueError('checkpoint identity')
    dev={sp.ident:sp for sp in specs('development',1)[:1]}
    episodes=[];decisions=0
    for f in (out/'traces').glob('*.json'):
        ep=json.loads(f.read_text());sp=dev.get(ep['world'])
        if sp is None:raise ValueError('unrecognized dev world')
        decisions+=replay_episode(ep,models[ep['arm']],sp);episodes.append(ep)
    if (len(episodes),decisions)!=(r['episodes'],r['decisions']):raise ValueError('episode count')
    for arm in ARMS:
        actual=aggregate([ep for ep in episodes if ep['arm']==arm]);expected=dict(r['summary'][arm])
        actual.pop('mean_decision_ms');expected.pop('mean_decision_ms')
        if actual!=expected:raise ValueError('aggregate tamper')
    return {'episodes_replayed':len(episodes),'decisions_replayed':decisions,
            'files_verified':len(manifest),'checkpoints_restored':len(models)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--seed',type=int,default=600);p.add_argument('--updates',type=int,default=30)
    p.add_argument('--quick',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.seed,a.updates,a.quick),sort_keys=True))

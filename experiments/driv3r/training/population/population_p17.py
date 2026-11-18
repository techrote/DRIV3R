"""P17: bounded CPU population study of multistage painting/recovery missions.

All environments are SYNTHETIC P0-derived host-legal menus. No game imports.
Optimizer-trained MLPs are tiny scratch policies, not SLMs or native agents.
The privileged shortest-route demonstrator is NOT an autonomous learned planner.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math, time, itertools
from pathlib import Path
import numpy as np
from population_lab import World, DIRECTIONS, NICHES, FEATURES, stable_seed, utility, softmax

ACTIONS=('N','E','S','W','RETRACE','STOP')
EXTRA=('relative_goal_distance','phase','remaining','material_sand','material_stone',
       'material_metal','material_clay','recent_retrace','goal_dx','goal_dy')
ALL_FEATURES=FEATURES+EXTRA
ARMS=('goal_aware_curriculum','goal_aware_static','goal_blind_curriculum')


def sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class MissionWorld(World):
    """Finite P0 legal catalog, 3 changing goals, protected cells, materials and debris."""
    def __init__(self, seed, holdout=False, disturbed=True, max_steps=96):
        super().__init__(seed=seed,max_steps=max_steps)
        self.cursor=(3,11);self.painted={self.cursor}
        self.stage=0;self.stopped=False;self.switched=False
        self.disturbed=disturbed;self.disturbance_count=0;self.recovery_count=0
        self.materials={};self.material_hazard_paints=0
        rng=np.random.default_rng(stable_seed('p17',seed,'world',int(holdout)))
        # Nonrectangular protected regions and random blocking debris, disjoint from the starting cell.
        self.protected={(x,y) for x in range(9,12) for y in range(8,14)}
        self.protected|={(x,9) for x in range(14,17)}
        self.obstacles={(x,y) for x in range(1,21) for y in range(1,21)
                        if (x,y) not in self.protected and (x,y)!=self.cursor and rng.random()<(.085 if not holdout else .12)}
        wall=7+(seed%3 if not holdout else (seed+1)%3)
        gap=5+(seed%11)
        for y in range(2,20):
            if abs(y-gap)>1 and (wall,y) not in self.protected:
                self.obstacles.add((wall,y))
        # Three directed, independently traversable scaffolds; open tiles are NOT prepainted.
        # Branches separate near the origin so stage switching requires retracing.
        paths=[]
        paths.append([(3,y) for y in range(11,3,-1)]+[(x,4) for x in range(4,18)]+[(17,y) for y in range(5,7)])
        paths.append([(2,11)]+[(2,y) for y in range(12,18)]+[(x,17) for x in range(3,6)])
        paths.append([(x,11) for x in range(4,7)]+[(6,y) for y in range(12,19)]+[(x,18) for x in range(7,18)]+[(17,y) for y in range(17,14,-1)])
        for path in paths:
            for pos in path:
                self.protected.discard(pos);self.obstacles.discard(pos)
        # The three target locations and orders are controlled independently of the random walls.
        anchors=[(17,6),(5,17),(17,15)]
        if holdout:
            order=((1,0,2),(2,0,1),(1,2,0),(2,1,0))[seed%4]
        else:
            order=((0,1,2),(0,2,1))[seed%2]
        self.goals=[anchors[i] for i in order]
        self.goal=self.goals[0];self.second_goal=self.goals[1]
        self.materials={(x,y):('sand','stone','metal','clay')[int(rng.integers(4))]
                        for x in range(1,21) for y in range(1,21)}
        self.holdout=holdout

    def legal(self):
        if self.stage==len(self.goals) or self.stopped or self.steps>=self.max_steps:
            return ['STOP']
        return super().legal()

    def feature(self,action):
        base=super().feature(action)  # host-legal, no future route information
        if action in DIRECTIONS:
            dx,dy=DIRECTIONS[action];p=(self.cursor[0]+dx,self.cursor[1]+dy)
        elif action=='RETRACE':p=self.parent[self.cursor]
        else:p=self.cursor
        mat=self.materials.get(p,'sand')
        dist=(abs(p[0]-self.goal[0])+abs(p[1]-self.goal[1]))/40
        extras=[dist,self.stage/3,max(0,(self.max_steps-self.steps)/self.max_steps)]
        extras += [float(action in DIRECTIONS and mat==m) for m in ('sand','stone','metal','clay')]
        extras += [float(len(self.history)>0 and self.history[-1]['action']=='RETRACE'),
                   (self.goal[0]-self.cursor[0])/20,(self.goal[1]-self.cursor[1])/20]
        return np.concatenate((base,np.asarray(extras,dtype=np.float64)))

    def step(self,action):
        if action not in self.legal():raise ValueError('invalid host action')
        target=None
        if action in DIRECTIONS:
            dx,dy=DIRECTIONS[action];target=(self.cursor[0]+dx,self.cursor[1]+dy)
        if action=='RETRACE':self.recovery_count+=1
        if target is not None and self.materials[target] in ('metal','stone'):
            self.material_hazard_paints+=1
        # Delegate physical movement to the P0 synthetic host; override its unconditional halfway goal switch.
        previous_stage=self.stage
        super().step(action)
        if action!='STOP' and self.cursor==self.goals[self.stage]:
            self.stage+=1
            self.switched=self.stage>0
            if self.stage<len(self.goals):self.goal=self.goals[self.stage]
        else:
            # Restore goal if P0's arbitrary halfway switch fired.
            self.goal=self.goals[min(self.stage,len(self.goals)-1)]
        # The inherited P0 half-horizon switch is NOT the mission's switch condition.
        self.switched=self.stage>0
        # Dynamic debris, placed only in currently open cells: host must recompute legal menu.
        if self.disturbed and self.disturbance_count==0 and self.steps>=8 and self.stage<len(self.goals):
            adjacent=[(self.cursor[0]+dx,self.cursor[1]+dy) for dx,dy in DIRECTIONS.values()]
            candidates=[p for p in adjacent if self.open(p) and p not in self.goals]
            if candidates:
                # Deterministic perturbation, one newly blocked prospective paint cell.
                p=candidates[stable_seed('debris',self.seed,self.steps)%len(candidates)]
                self.obstacles.add(p);self.disturbance_count=1
        self.history[-1]['stage_before']=previous_stage
        self.history[-1]['stage_after']=self.stage
        self.history[-1]['goal_before']=list(self.goals[previous_stage]) if previous_stage<3 else None
        self.history[-1]['new_obstacle_count']=self.disturbance_count
        if self.painted & (self.protected|self.obstacles):
            raise AssertionError('protected/obstacle paint')


def expert_action(w):
    """Privileged shortest-path teacher over host-legal moves and parent retrace."""
    actions=w.legal()
    if w.stage==3:return 'STOP'
    start=w.cursor;target=w.goal
    # BFS state: position and hypothetical last parent. Forward edges are empty cells only.
    # Parent edges exist only on the *already painted* branch tree.
    queue=collections.deque([(start,None)])
    seen={start}
    while queue:
        p,first=queue.popleft()
        if p==target:return first or 'STOP'
        opts=[]
        if p in w.parent:opts.append(('RETRACE',w.parent[p]))
        for action,(dx,dy) in DIRECTIONS.items():
            q=(p[0]+dx,p[1]+dy)
            if w.open(q):opts.append((action,q))
        for action,q in opts:
            if q not in seen:
                seen.add(q);queue.append((q,action if first is None else first))
    return 'RETRACE' if 'RETRACE' in actions else 'STOP'


class SmallPolicy:
    """Real manually implemented Adam-trained 30-feature two-layer MLP."""
    def __init__(self,seed,hidden=32,lr=.006):
        rng=np.random.default_rng(seed)
        d=len(ALL_FEATURES)
        self.p=dict(W1=rng.normal(0,.13,(d,hidden)),b1=np.zeros(hidden),
                    W2=rng.normal(0,.13,hidden),b2=np.zeros(1))
        self.m={k:np.zeros_like(v) for k,v in self.p.items()}
        self.v={k:np.zeros_like(v) for k,v in self.p.items()}
        self.lr=lr;self.updates=0;self.hidden=hidden
    def logits(self,X):
        return np.tanh(X@self.p['W1']+self.p['b1'])@self.p['W2']+self.p['b2'][0]
    def choice(self,X):return int(np.argmax(self.logits(X)))
    def teach(self,X,target):
        H=np.tanh(X@self.p['W1']+self.p['b1'])
        pred=softmax(H@self.p['W2']+self.p['b2'][0]);d=pred-target
        dh=d[:,None]*self.p['W2'][None,:]*(1-H*H)
        g=dict(W1=X.T@dh,b1=dh.sum(0),W2=H.T@d,b2=np.array([d.sum()]))
        norm=math.sqrt(sum(float((v*v).sum()) for v in g.values()))
        if norm>8:g={k:v*8/norm for k,v in g.items()}
        self.updates+=1
        for k,grad in g.items():
            self.m[k]=.9*self.m[k]+.1*grad
            self.v[k]=.999*self.v[k]+.001*grad*grad
            m=self.m[k]/(1-.9**self.updates)
            v=self.v[k]/(1-.999**self.updates)
            self.p[k]-=self.lr*m/(np.sqrt(v)+1e-8)
    def serialized(self):
        return dict(format='p17-mlp-v1',features=list(ALL_FEATURES),hidden=self.hidden,
                    updates=self.updates,lr=self.lr,parameters={k:v.tolist() for k,v in self.p.items()})


def observations(w,arm):
    acts=w.legal()
    X=np.stack([w.feature(a) for a in acts])
    if arm=='goal_blind_curriculum':
        for feature in ('goal_progress','guidance','goal_switched',
                        'relative_goal_distance','phase','goal_dx','goal_dy'):
            X[:,ALL_FEATURES.index(feature)]=0
    return acts,X


def train(policy,niche,train_seeds,arm,updates,epsilon=.22):
    rng=np.random.default_rng(stable_seed('p17','trainrng',train_seeds[0],niche,arm))
    losses=[];episodes=0;w=None
    while policy.updates<updates:
        if w is None or w.stopped or w.steps>=w.max_steps or w.stage==3:
            s=train_seeds[episodes%len(train_seeds)];episodes+=1
            w=MissionWorld(s,disturbed=(arm!='goal_aware_static'))
        acts,X=observations(w,arm)
        teacher=expert_action(w)
        if teacher not in acts:raise AssertionError('privileged teacher outside host legal set')
        target=np.zeros(len(acts),dtype=float)
        # Strong supervised route target plus small niche-conditioned style preference.
        vals=np.asarray([utility(niche,w.feature(a)[:len(FEATURES)],w) for a in acts])
        target+=.16*softmax(vals/.7);target[acts.index(teacher)]+=.84
        pred=softmax(policy.logits(X));losses.append(-float(np.sum(target*np.log(np.maximum(pred,1e-12)))))
        policy.teach(X,target)
        if rng.random()<.6:
            action=teacher
        elif rng.random()<epsilon:
            action=acts[int(rng.integers(len(acts)))]
        else:action=acts[policy.choice(X)]
        w.step(action)
    return dict(optimizer_updates=policy.updates,train_episodes=episodes,
                final_mean_cross_entropy=round(float(np.mean(losses[-min(100,len(losses)):])),5))


def rollout(policy,niche,seed,arm,holdout=True,disturbed=True,full_trace=False):
    w=MissionWorld(seed,holdout=holdout,disturbed=disturbed)
    trace=[];legal=0
    while not w.stopped and w.steps<w.max_steps:
        acts,X=observations(w,arm)
        if policy=='expert':action=expert_action(w)
        elif policy=='nonstop_random':
            non=[a for a in acts if a!='STOP']
            action=non[stable_seed('random',seed,w.steps,niche)%len(non)] if non else 'STOP'
        elif policy=='persistent':
            action=w.heading if w.heading in acts else next((a for a in acts if a in DIRECTIONS),
                                   'RETRACE' if 'RETRACE' in acts else 'STOP')
        else:action=acts[policy.choice(X)]
        if action not in acts:raise AssertionError('illegal model action')
        legal+=1
        before_stage=w.stage
        w.step(action)
        if full_trace:trace.append(dict(action=action,pos=list(w.cursor),stage_before=before_stage,
                                       stage_after=w.stage,legal_actions=acts,goal=list(w.goal)))
    outcome=dict(completed=(w.stage==3),success=(w.stage==3 and w.stopped),
        painted=len(w.painted)-1,retraces=w.recovery_count,debris=w.disturbance_count,
        hazard_paints=w.material_hazard_paints,steps=w.steps,early_stop=(w.stopped and w.stage!=3),
        stage=w.stage,legal_actions=legal,legal_rate=1.,descriptor=w.descriptor())
    if full_trace:
        outcome.update(seed=seed,niche=niche,arm=arm,synthetic=True,native=False,
                       start=[3,11],goals=[list(p) for p in w.goals],
                       history=trace,painted=[list(p) for p in sorted(w.painted)],
                       obstacles=[list(p) for p in sorted(w.obstacles)],
                       protected=[list(p) for p in sorted(w.protected)])
    return outcome


def summarize(policy,niche,seeds,arm,disturbed=True):
    outcomes=[rollout(policy,niche,s,arm,disturbed=disturbed) for s in seeds]
    return {k:round(float(np.mean([o[k] for o in outcomes])),5)
            for k in ('completed','success','painted','retraces','debris','hazard_paints','steps','early_stop','legal_rate')}


def svg_trace(w):
    """Trajectory plot composed from actual synthetic host trace, no generated image claims."""
    cell=13; s=['<svg xmlns="http://www.w3.org/2000/svg" width="286" height="286" viewBox="0 0 286 286">',
               '<rect width="286" height="286" fill="#19212a"/>']
    for x,y in sorted(w.obstacles):s.append(f'<rect x="{13*x}" y="{13*y}" width="12" height="12" fill="#586573"/>')
    for x,y in sorted(w.protected):s.append(f'<rect x="{13*x}" y="{13*y}" width="12" height="12" fill="#ad4a4f"/>')
    for x,y in sorted(w.painted):s.append(f'<rect x="{13*x+2}" y="{13*y+2}" width="9" height="9" fill="#51c9a9"/>')
    for i,(x,y) in enumerate(w.goals):s.append(f'<circle cx="{13*x+6}" cy="{13*y+6}" r="4" fill="{("#fff0a0","#efa975","#80b8ff")[i]}"/>')
    s.append(f'<circle cx="{13*w.cursor[0]+6}" cy="{13*w.cursor[1]+6}" r="4" fill="#ffffff"/>')
    s.append('</svg>')
    return '\n'.join(s)+'\n'


def write_manifest(folder):
    folder=Path(folder)
    paths=sorted(p for p in folder.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    data=''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(folder).as_posix()}\n' for p in paths)
    (folder/'SHA256SUMS').write_text(data)
    return len(paths)


def run(output,seeds=(672,673),train_worlds=12,selection_worlds=5,heldout_worlds=8,
        updates=120,hidden=32,offspring=2,gallery_limit=2):
    if (not 1<=len(seeds)<=4 or len(set(seeds))!=len(seeds) or
        not 4<=train_worlds<=40 or not 3<=selection_worlds<=20 or
        not 4<=heldout_worlds<=32 or not 40<=updates<=1500 or not 8<=hidden<=64 or
        not 1<=offspring<=4 or not 0<=gallery_limit<=12):
        raise ValueError('outside bounded P17 CPU research envelope')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir();(output/'gallery').mkdir()
    records=[];trace_records=[];start=time.monotonic();model_count=0
    for seed in seeds:
        split={name:[stable_seed('p17',seed,name,i) for i in range(count)]
               for name,count in [('train',train_worlds),('selection',selection_worlds),('heldout',heldout_worlds)]}
        if len(set(sum(split.values(),[])))!=sum(len(v) for v in split.values()):raise AssertionError('split collision')
        for niche in NICHES:
            for arm in ARMS:
                options=[]
                for child in range(offspring):
                    ident=f'{seed}-{niche}-{arm}-{child}'
                    p=SmallPolicy(stable_seed('init',seed,niche,child),hidden,
                                  lr=(.004,.008,.012,.006)[child])
                    train_info=train(p,niche,split['train'],arm,updates)
                    selection=summarize(p,niche,split['selection'],arm)
                    options.append((selection['success'],selection['completed'],ident,p,train_info,selection,child))
                    ck=output/'checkpoints'/f'{ident}.json'
                    ck.write_text(json.dumps(p.serialized(),separators=(',',':'))+'\n')
                    records.append(dict(id=ident,seed=seed,niche=niche,arm=arm,child=child,
                         parent='scratch',lineage_seed=stable_seed('init',seed,niche,child),
                         train_seed_sha256=sha(split['train']),selection_seed_sha256=sha(split['selection']),
                         heldout_seed_sha256=sha(split['heldout']),train=train_info,
                         selection=selection,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest()))
                    model_count+=1
                # Pre-registered success-first selection, no heldout model shopping.
                winner=max(options,key=lambda x:(x[0],x[1],-x[6]))
                _,_,id,p,_,_,_=winner
                held=summarize(p,niche,split['heldout'],arm)
                for r in records:
                    if r['id']==id:r['selected']=True;r['heldout']=held
                for s in split['heldout'][:gallery_limit]:
                    t=rollout(p,niche,s,arm,full_trace=True)
                    path=output/'gallery'/f'{id}-{s}'
                    (path.with_suffix('.json')).write_text(json.dumps(t,indent=2)+'\n')
                    # Regenerate host world from actual selected policy for SVG.
                    w=MissionWorld(s,holdout=True)
                    while not w.stopped and w.steps<w.max_steps:
                        acts,X=observations(w,arm);w.step(acts[p.choice(X)])
                    path.with_suffix('.svg').write_text(svg_trace(w))
                trace_records.append(dict(seed=seed,niche=niche,arm=arm,winner=id,
                                          selection=winner[5],heldout=held))
        (output/f'splits-{seed}.json').write_text(json.dumps(dict(split=split,split_hashes={k:sha(v) for k,v in split.items()},
                    geometry='P0-derived synthetic host; seed-specific walls, protected zones, material tiles'),indent=2)+'\n')
    diversity={}
    for arm in ARMS:
        pairwise=[];jaccard=[]
        for seed in seeds:
            winners={r['niche']:r['winner'] for r in trace_records if r['seed']==seed and r['arm']==arm}
            for world_seed in [stable_seed('p17',seed,'heldout',i) for i in range(min(gallery_limit,heldout_worlds))]:
                outcomes=[]
                for niche in NICHES:
                    file=output/'gallery'/f'{winners[niche]}-{world_seed}.json'
                    outcomes.append(json.loads(file.read_text()))
                for i,one in enumerate(outcomes):
                    for other in outcomes[i+1:]:
                        a=[r['action'] for r in one['history']];b=[r['action'] for r in other['history']]
                        pairwise.append(sum(x!=y for x,y in itertools.zip_longest(a,b,fillvalue='PAD'))/max(1,len(a),len(b)))
                        aset={tuple(x) for x in one['painted']};bset={tuple(x) for x in other['painted']}
                        jaccard.append(1-len(aset&bset)/len(aset|bset))
        diversity[arm]=dict(action_distance=round(float(np.mean(pairwise)),5) if pairwise else None,
                            paint_jaccard_distance=round(float(np.mean(jaccard)),5) if jaccard else None,
                            compared_pairs=len(pairwise))
    aggregate={}
    for arm in ARMS:
        rows=[r for r in trace_records if r['arm']==arm]
        aggregate[arm]={k:round(float(np.mean([r['heldout'][k] for r in rows])),5)
                        for k in ('completed','success','painted','retraces','hazard_paints','early_stop','legal_rate')}
    # Original host-controlled baselines; heldout never available to selection or training.
    controls={}
    for seed in seeds:
        hold=[stable_seed('p17',seed,'heldout',i) for i in range(heldout_worlds)]
        controls[str(seed)]={n:{c:summarize(c,n,hold,'goal_aware_curriculum') for c in ('expert','nonstop_random','persistent')}
                             for n in NICHES}
    result=dict(format='population-p17-autonomous-mission-v1',native=False,slm=False,
        source='synthetic P0 derived external host, not certified CyberSand ABI',
        seeds=list(seeds),arms=ARMS,niches=NICHES,train_worlds=train_worlds,
        selection_worlds=selection_worlds,heldout_worlds=heldout_worlds,
        updates_per_candidate=updates,candidates=model_count,total_adam_updates=model_count*updates,
        offspring=offspring,aggregate=aggregate,diversity=diversity,controls=controls,champions=trace_records,
        elapsed_s=round(time.monotonic()-start,3),
        interpretation=['Synthetic scaffold guarantees routes to three distinct changing-goal targets',
         '100% legal action rate is guaranteed by host-side action menu, not learned legality',
         'Expert is privileged BFS; models imitate labels from a planner, not learned general problem solving',
         'Material labels are synthetic and not tied to CyberSand engine reaction dynamics',
         'Finite six-niche teacher utilities are subjective proxies, not independently measured artistic personalities',
         'Goal-blind is an observation ablation, not a matched-information architectural comparator',
         'No native gameplay, no trained pretrained SLM, no claimed source-faithful engine material physics'])
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'lineage.json').write_text(json.dumps(records,indent=2)+'\n')
    write_manifest(output)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[672,673]);ap.add_argument('--updates',type=int,default=120)
    ap.add_argument('--offspring',type=int,default=2)
    a=ap.parse_args();r=run(**vars(a))
    print(json.dumps(dict(candidates=r['candidates'],updates=r['total_adam_updates'],aggregate=r['aggregate']),sort_keys=True))

if __name__=='__main__':main()

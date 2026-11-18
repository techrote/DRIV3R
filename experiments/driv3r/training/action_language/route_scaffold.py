"""GAL-015: bounded route-to-frontier scaffold and trained paint-action scorer.

External synthetic material worlds only. Route construction uses only radius-4
observations and accepted host receipts. All choices are from host-issued leased
IDs; no hidden goal/corridor or native game mutation is used by inference.
Scaffolded LIFTs are deterministic host-authorized actions, so `model_calls`
counts paint-selection calls, *not* actual SLM invocation or token usage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from frontier_memory import Ledger, Scorer, features, choose_teacher
from material_corridors import visible_material, digest
from material_corridors import CorridorWorld, neighbors, inside, make_corridor
from lab import W, H, World
HORIZONS = (32, 48, 64)
KINDS = ('path','tree','comb','deep_tree')
from lab import DECISION_CAP, START, INK, png, sha
from frontier_memory import FEATURES, Scorer

def load_frozen(path):
    cp=json.loads(Path(path).read_text())
    if cp.get('schema')!='gal-011-memory-frontier/1' or cp.get('features')!=list(FEATURES) or cp.get('mode')!='trained':
        raise ValueError('incompatible GAL-011 baseline checkpoint')
    m=Scorer();m.weights=list(cp['weights'])
    return m

def extended_tree(seed: int, kind: str, count: int):
    """Host-owned induced tree. No hidden target is consumed by the policy.

    comb = intentionally long return-to-trunk paths with widely separated limbs;
    deep_tree = random tree that strongly prefers extending the furthest tips.
    """
    if kind not in KINDS or count not in (33, 49, 65):
        raise ValueError('unknown family or corridor size')
    if count == 33 and kind in ('path', 'tree'):
        return make_corridor(seed, kind)
    for attempt in range(240):
        rng = random.Random(f'gal012:{kind}:{count}:{seed}:{attempt}')
        cells = {START};current = START;prev = START
        if kind == 'comb':
            # A horizontal spine from the fixed starting point. Limbs alternate
            # north/south, separated by 3 columns to avoid accidental cycles.
            base_len = 17 if count == 49 else 23 if count == 65 else 12
            spine_len = base_len+rng.choice((-1,0,1))
            trunk = [(START[0]+i, START[1]) for i in range(spine_len)]
            cells.update(trunk)
            attach = list(range(rng.choice((1,2)),spine_len-1,3))
            flip=rng.choice((-1,1))
            limbs = []
            for j,offset in enumerate(attach):
                sign = flip if j%2 else -flip
                limbs.append([(START[0]+offset,START[1]+sign*i) for i in range(1,15)])
            grown = [0]*len(limbs)
            while len(cells) < count:
                choices = [i for i,limb in enumerate(limbs) if grown[i] < len(limb)]
                if not choices:break
                # Unequal branch lengths create distinct return-to-trunk
                # workloads without introducing cycles or unreachable cells.
                weights=[1/(1+grown[j])**.4 for j in choices]
                i=rng.choices(choices,weights=weights,k=1)[0]
                c = limbs[i][grown[i]]
                if not inside(c) or sum(n in cells for n in neighbors(c)) != 1:break
                cells.add(c);grown[i]+=1
            if len(cells)==count:return frozenset(cells),attempt
            continue
        distances = {START: 0}
        while len(cells) < count:
            frontier=[]
            roots = [current] if kind == 'path' else sorted(cells)
            for a in roots:
                for b in neighbors(a):
                    if inside(b) and b not in cells and sum(n in cells for n in neighbors(b))==1:
                        frontier.append((a,b))
            if not frontier:break
            if kind=='path':
                direction=(current[0]-prev[0],current[1]-prev[1])
                straight=[z for z in frontier if (z[1][0]-z[0][0],z[1][1]-z[0][1])==direction]
                a,b=rng.choice(straight if straight and rng.random()<.72 else frontier)
                prev,current=current,b
            elif kind=='deep_tree':
                # Force some width before committing to far tips. This tests
                # recovery over long remembered corridors, not easy local fill.
                ranked=sorted(frontier,key=lambda z:(distances[z[0]],rng.random()),reverse=True)
                a,b=ranked[0] if rng.random()<.83 else rng.choice(frontier)
            else:
                a,b=rng.choice(frontier)
            distances[b]=distances[a]+1
            cells.add(b)
        deg=[sum(n in cells for n in neighbors(p)) for p in cells]
        if len(cells)==count and (kind=='path' or sum(x>=3 for x in deg)>=2):
            if kind=='path' and max(deg)>2:raise AssertionError('non-induced path')
            return frozenset(cells),attempt
    raise RuntimeError(f'generator exhausted {kind} {count} {seed}')


class HorizonWorld(CorridorWorld):
    def __init__(self,seed:int,kind:str,arm:str,lease:str,ink_target:int):
        if ink_target not in HORIZONS:raise ValueError('bad target')
        super().__init__(seed,'tree' if kind in ('comb','deep_tree') else kind,arm,lease)
        self.kind=kind
        self.corridor,self.attempt=extended_tree(seed,kind,ink_target+1)
        self.goal=self.corridor-{START}
        self.material=[3]*(W*H);self.protected=set()
        for y in range(H):
            for x in range(W):
                if x in (0,W-1) or y in (0,H-1):
                    self.material[self.idx((x,y))]=5;self.protected.add((x,y))
        for p in self.corridor:self.material[self.idx(p)]=0
        self.original=self.material.copy()
        self.ink_target=ink_target
        assert len(self.goal)==ink_target

    def legal(self):
        offered=World.legal(self)
        remaining=self.ink_target-len(self.own)
        return {n:p for n,p in offered.items() if not n.startswith(('INK_','JOG_')) or len(p)<=remaining}




SCHEMA='gal-015-route-scaffold/1'
TRAIN_BASE=1940000
EVAL_BASE=2040000
TRAIN_KINDS=('path','tree')
EVAL_KINDS=('path','tree','comb','deep_tree')
EVAL_TARGETS=(32,48,64)
MODES=('frozen_sample','frozen_greedy','route_untrained','route_trained','route_unguarded','observation_teacher')
# Explicit task semantics: START excluded from every goal; no privileged map.
PAINT_FEATURES=('novel_len','frozen_logit','align_heading','target_frontier','leftover', 'near_known_open')

def get_paint_candidates(world, ledger, menu):
    return [(n,p) for n,p in menu.items()
        if n.startswith(('INK_','JOG_')) and p and START not in p
        and all(q not in ledger.recorded_paints for q in p)]

def route_actions(world,ledger,menu,allow_endpoint=False):
    path=ledger.route(world.cursor)
    if not path:return []
    matches=[]
    for n,p in menu.items():
        if n.startswith(('LIFT_','RETRACE_')) and p and (len(p)<=len(path) if allow_endpoint else len(p)<len(path)) and tuple(path[:len(p)])==p:
            matches.append((n,p))
    return matches

def _features(world,ledger,model,name,path,all_paints):
    # All features depend only on the visible crop, accepted receipts and
    # frozen scorer (not privileged geometry or hidden target state).
    logit=sum(a*b for a,b in zip(features(world,ledger,name,path,ledger.route(world.cursor)),model.weights))
    crop=visible_material(world,'material')
    terminal=path[-1]
    direction=name.split('_')[1] if name.startswith('INK_') else world.heading
    left=max((len(p) for _,p in all_paints),default=1)
    neighbors=((terminal[0]+1,terminal[1]),(terminal[0]-1,terminal[1]),
               (terminal[0],terminal[1]+1),(terminal[0],terminal[1]-1))
    return (len(path)/4,logit/10,float(direction==world.heading),
        float(terminal in ledger.targets()),(left-len(path))/4,
        sum(crop[q]==0 for q in neighbors if q in crop)/4)

class PaintScorer:
    def __init__(self,weights=None):
        self.weights=[0.]*len(PAINT_FEATURES) if weights is None else list(weights)
        if len(self.weights)!=len(PAINT_FEATURES):raise ValueError('paint checkpoint features mismatch')
    def probabilities(self,vs):
        s=[sum(a*b for a,b in zip(v,self.weights)) for v in vs]
        hi=max(s); e=[math.exp(max(-40.,v-hi)) for v in s];z=sum(e)
        return [x/z for x in e]
    def choose(self,world,ledger,base,paints):
        vecs=[_features(world,ledger,base,name,path,paints) for name,path in paints]
        scores=[sum(w*v for w,v in zip(self.weights,vec)) for vec in vecs]
        return max(range(len(paints)),key=lambda i:(scores[i],paints[i][0])),vecs
    def update(self,vs,target,rate=.035):
        probs=self.probabilities(vs)
        exp=[sum(p*v[k] for p,v in zip(probs,vs)) for k in range(len(PAINT_FEATURES))]
        for k in range(len(self.weights)):
            self.weights[k]=max(-8.,min(8.,self.weights[k]+rate*(vs[target][k]-exp[k])))

def choose(world,ledger,base,adapter,mode,rng):
    menu,token=world.offer()
    if mode=='observation_teacher':return choose_teacher(world,ledger,menu),token,0,'teacher'
    if mode.startswith('frozen'):
        route=ledger.route(world.cursor)
        vecs=[features(world,ledger,n,p,route) for n,p in menu.items()]
        logits=[sum(a*b for a,b in zip(v,base.weights)) for v in vecs]
        if mode=='frozen_greedy':i=max(range(len(logits)),key=lambda j:(logits[j],list(menu)[j]))
        else:
            hi=max(logits);prob=[math.exp(max(-40.,v-hi)) for v in logits];z=sum(prob)
            draw=rng.random()*z; i=len(prob)-1
            for j,v in enumerate(prob):
                draw-=v
                if draw<=0:i=j;break
        return list(menu)[i],token,1,'model'
    paints=get_paint_candidates(world,ledger,menu)
    if paints:
        if mode in ('route_untrained','route_unguarded'):
            # Untrained but deterministic rule, isolates scaffold contribution.
            i=max(range(len(paints)),key=lambda j:(len(paints[j][1]),paints[j][0]))
        else:i,_=adapter.choose(world,ledger,base,paints)
        return paints[i][0],token,1,'model'
    travel=route_actions(world,ledger,menu,allow_endpoint=mode=='route_unguarded')
    if travel:
        return max(travel,key=lambda x:(len(x[1]),x[0]))[0],token,0,'route'
    return 'STOP',token,0,'stop'

def episode(seed,kind,target,mode,base,adapter,save_trace=False):
    if mode not in MODES:raise ValueError('mode unknown')
    world=HorizonWorld(seed,kind,'lift',f'gal015:{mode}:{target}:{kind}:{seed}',target)
    led=Ledger();rng=random.Random(f'gal015:matched:{kind}:{target}:{seed}')
    calls=route_steps=0
    while not world.stopped and world.decisions<DECISION_CAP and world.goal-world.own:
        led.observe(world)
        action,token,call,reason=choose(world,led,base,adapter,mode,rng)
        opts,_=world.offer();path=opts[action]
        receipt=world.apply(action,token,original_response=f'gal015:{mode}:{reason}:{action}')
        led.accept(action,path,receipt)
        calls+=call;route_steps+=reason=='route'
    complete=world.goal.issubset(world.own)
    correct=len(world.own&world.goal)
    assert not(world.own&world.protected)
    assert all(world.material[world.idx(p)]==world.original[world.idx(p)] for p in world.protected)
    assert all(r['native_receipts']==0 and r['action_id'] for r in world.trace)
    row={'seed':seed,'kind':kind,'target':target,'mode':mode,'complete':complete,
         'correct':correct,'coverage':round(correct/target,6),
         'decisions':world.decisions,'model_calls_proxy':calls,'route_steps':route_steps,
         'ink':len(world.own),'ink_per_decision':round(len(world.own)/max(1,world.decisions),6),
         'stopped_early':world.stopped and not complete,
         'material_sha256':sha(world.material),'receipts_sha256':sha(world.trace),
         'native_receipts':0,'resident_slms':0}
    return (world,row) if save_trace else (None,row)

def train(base,worlds=48):
    if not (8<=worlds<=96):raise ValueError('bounded training worlds')
    adapter=PaintScorer();updates=0;first=0
    for i in range(worlds):
        seed=TRAIN_BASE+i;kind=TRAIN_KINDS[i%len(TRAIN_KINDS)]
        target=32 if i%3 else 48
        world=HorizonWorld(seed,kind,'lift',f'gal015:train:{i}',target)
        led=Ledger();roll=random.Random(f'gal015:roll:{seed}')
        while not world.stopped and world.decisions<DECISION_CAP and world.goal-world.own:
            led.observe(world);menu,lease=world.offer();paints=get_paint_candidates(world,led,menu)
            if paints:
                expert=choose_teacher(world,led,menu)
                # The teacher chooses from a larger catalog, but if its action
                # is not a safe fresh-paint candidate we still enforce the
                # exact trained head's scope.
                if expert not in [n for n,_ in paints]:
                    expert=max(paints,key=lambda x:(len(x[1]),x[0]))[0]
                pick,vecs=adapter.choose(world,led,base,paints)
                desired=next(j for j,(name,_) in enumerate(paints) if name==expert)
                adapter.update(vecs,desired);updates+=1
                action=expert if roll.random()<(.9 if i<worlds//2 else .45) else paints[pick][0]
                first+=action==expert
            else:
                travel=route_actions(world,led,menu)
                action=max(travel,key=lambda x:(len(x[1]),x[0]))[0] if travel else 'STOP'
            receipt=world.apply(action,lease,original_response=f'gal015:train:{action}')
            led.accept(action,menu[action],receipt)
    return adapter,{'worlds':worlds,'updates':updates,'teacher_followed_on_paint':first,
                    'training_kinds':TRAIN_KINDS,'training_targets':(32,48),
                    'training_seeds':[TRAIN_BASE,TRAIN_BASE+worlds-1]}

def aggregate(rows):
    return {'episodes':len(rows),'completions':sum(x['complete'] for x in rows),
        'mean_coverage':round(statistics.mean(x['coverage'] for x in rows),5),
        'mean_decisions':round(statistics.mean(x['decisions'] for x in rows),4),
        'mean_model_calls_proxy':round(statistics.mean(x['model_calls_proxy'] for x in rows),4),
        'mean_route_steps':round(statistics.mean(x['route_steps'] for x in rows),4),
        'ink_per_decision':round(sum(x['ink'] for x in rows)/sum(x['decisions'] for x in rows),5),
        'early_stops':sum(x['stopped_early'] for x in rows)}



def make_svg_gallery(out,base,adapter):
    """Compact original 2x3 raster gallery; no native/render screenshots."""
    from html import escape
    modes=('frozen_greedy','route_untrained','route_trained')
    seeds=(EVAL_BASE,EVAL_BASE+1)
    cell=3; panel_w=172;panel_h=175
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="516" height="350" viewBox="0 0 516 350">',
        '<rect width="516" height="350" fill="#15202b"/>']
    for ri,seed in enumerate(seeds):
        for ci,mode in enumerate(modes):
            world,row=episode(seed,'deep_tree',64,mode,base,adapter,save_trace=True)
            x0=ci*panel_w+13;y0=ri*panel_h+23
            parts.append(f'<text x="{x0}" y="{ri*panel_h+13}" font-size="10" fill="white" font-family="monospace">{escape(mode)} {row["correct"]}/64 · {row["decisions"]}d</text>')
            parts.append(f'<rect x="{x0}" y="{y0}" width="144" height="144" fill="#68717c"/>')
            for (x,y) in sorted(world.corridor):
                color='#f1bb5c' if (x,y) in world.own else '#101820'
                parts.append(f'<rect x="{x0+x*cell}" y="{y0+y*cell}" width="3" height="3" fill="{color}"/>')
            parts.append(f'<rect x="{x0+START[0]*cell}" y="{y0+START[1]*cell}" width="3" height="3" fill="#46c9d3"/>')
            x,y=world.cursor
            parts.append(f'<rect x="{x0+x*cell}" y="{y0+y*cell}" width="3" height="3" fill="#e83f61"/>')
    parts.append('</svg>')
    target=Path(out)/'GAL015_trajectory_gallery.svg'
    target.write_text('\n'.join(parts)+'\n')
    return digest(target)

def audit_evaluation(out,frozen_checkpoint):
    """Independently regenerate each episode using frozen checkpoint bytes."""
    out=Path(out);cp=json.loads((out/'trained_paint_scorer.json').read_text())
    assert cp['features']==list(PAINT_FEATURES) and cp['schema']==SCHEMA
    base=load_frozen(frozen_checkpoint);adapter=PaintScorer(cp['weights'])
    rows=json.loads((out/'episodes.json').read_text());matched=0;counts={}
    for r in rows:
        _,again=episode(r['seed'],r['kind'],r['target'],r['mode'],base,adapter)
        if again!=r:raise AssertionError(f'episode replay mismatch {r["mode"]} {r["seed"]}')
        matched+=1;counts[r['mode']]=counts.get(r['mode'],0)+1
    summary={'schema':SCHEMA,'replayed_episodes':matched,'all_episode_rows_and_receipt_hashes_match':True,
             'by_mode':counts,'trained_checkpoint_sha256':digest(out/'trained_paint_scorer.json'),
             'baseline_checkpoint_sha256':digest(frozen_checkpoint)}
    (out/'audit.json').write_text(json.dumps(summary,sort_keys=True,indent=2)+'\n')
    return summary

def run(out, frozen_checkpoint, train_worlds=48, eval_per_panel=12, gallery=2):
    if not (4<=eval_per_panel<=32 and 0<=gallery<=4):raise ValueError('bounded evaluation')
    base=load_frozen(frozen_checkpoint)
    adapter,training=train(base,train_worlds)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'gallery').mkdir(exist_ok=True)
    cp={'schema':SCHEMA,'features':PAINT_FEATURES,'weights':adapter.weights,
        'trained':training,'source':'external synthetic host','native_acceptance':False}
    cp_path=out/'trained_paint_scorer.json'
    cp_path.write_text(json.dumps(cp,sort_keys=True,indent=2)+'\n')
    rows=[];receipts={}
    for kind in EVAL_KINDS:
        for target in EVAL_TARGETS:
            for i in range(eval_per_panel):
                seed=EVAL_BASE+i
                for mode in MODES:
                    w,r=episode(seed,kind,target,mode,base,adapter,save_trace=i<gallery)
                    rows.append(r)
                    if w is not None and kind in ('tree','deep_tree') and target==64:
                        k=f'{mode}-{kind}-{target}-{seed}'
                        png(out/'gallery'/f'{k}.png',w,scale=5)
                        receipts[k]=w.trace
    panels={}
    for kind in EVAL_KINDS:
        for target in EVAL_TARGETS:
            for mode in MODES:
                subset=[r for r in rows if r['kind']==kind and r['target']==target and r['mode']==mode]
                panels[f'{kind}:{target}:{mode}']=aggregate(subset)
    totals={mode:aggregate([r for r in rows if r['mode']==mode]) for mode in MODES}
    results={'schema':SCHEMA,'train':training,'eval_seeds':[EVAL_BASE,EVAL_BASE+eval_per_panel-1],
        'heldout_morphologies':['comb','deep_tree'],'eval_worlds_per_panel':eval_per_panel,
        'checkpoint_sha256':digest(cp_path),'baseline_checkpoint_sha256':digest(frozen_checkpoint),
        'panels':panels,'totals':totals,'evidence_class':'external_synthetic_only',
        'native_receipts':0,'resident_slms':0,
        'model_calls_proxy_not_real_inference':True, 'unsafe_ablation':'route_unguarded crosses into unpainted target cell with LIFT, still host-legal but task-incorrect'}
    results['gallery_sha256']=make_svg_gallery(out,base,adapter)
    for name,val in [('metrics.json',results),('episodes.json',rows),('gallery_receipts.json',receipts)]:
        (out/name).write_text(json.dumps(val,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'training':training,'totals':totals,'checkpoint_sha256':digest(cp_path)},indent=2))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    p.add_argument('--frozen',required=True)
    p.add_argument('--train-worlds',type=int,default=48);p.add_argument('--eval',type=int,default=12)
    p.add_argument('--audit',action='store_true')
    args=p.parse_args()
    if args.audit:print(json.dumps(audit_evaluation(args.out,args.frozen),indent=2))
    else:run(args.out,args.frozen,args.train_worlds,args.eval)

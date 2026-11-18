"""TOPOLOGY-TRANSFER-001: CPU imitation and actual recovery transfer in synthetic worlds.

Research-only generator modifies a fresh `World` instance, not World code or the
CyberSand game. TRAIN motifs and DEVELOPMENT motifs are disjoint. Historical
World host-legal menus, material protection and backtracking semantics remain
unchanged. Native CyberSand acceptance is NOT claimed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import resource
import statistics
import time
import numpy as np

from .goal_conditioning_lab import GoalRanker, matrix, MAX_ACTIONS
from .goal_observability_lab import strict_step
from .detour_recovery_lab import prefix_plan
from .world import World, MOVE, Spec, specs, digest, MATERIAL_Z, SOLID
from .train import sha_file

VERSION='topology-transfer/1'
TRAIN_MOTIFS=('ribs','shelf')
DEV_MOTIFS=('stagger','lattice')
DEPTHS=(3,5)
ROTATIONS=(0,1,2,3)
TARGETS=((15,-9),(15,9))
SEED_PAIRS=((821,711),(822,712))
ARMS=('new','frozen','masked_new','prefix_oracle')
SOURCE=('topology_transfer_lab.py','world.py','goal_conditioning_lab.py',
        'goal_observability_lab.py','detour_recovery_lab.py','policy.py','train.py')


def rot(p, quarter):
    x,y=p
    for _ in range(quarter%4): x,y=-y,x
    return x,y


def convert(p, quarter):
    x,y=rot(p,quarter)
    return 48+x,48+y


def terrain(motif, variation):
    blocks=set(); protected=set()
    def bar_x(x, lo,hi, gaps=()):
        for y in range(lo,hi+1):
            if y not in gaps:blocks.add((x,y))
    def bar_y(y,lo,hi,gaps=()):
        for x in range(lo,hi+1):
            if x not in gaps:blocks.add((x,y))
    if motif=='ribs':
        bar_x(9,-10,10, (-6,-5,5,6))
        bar_x(12,-10,10, (-9,-8,8,9))
    elif motif=='shelf':
        bar_y(-4,7,16,(9+variation,10+variation))
        bar_y(4,7,16,(13-variation,14-variation))
        for x in (8,12,16):blocks.add((x,0))
    elif motif=='stagger':
        bar_x(9,-11,3,(-4-variation,-3-variation))
        bar_x(13,-3,11,(5+variation,6+variation))
        bar_y(0,10,12,(11,))
    elif motif=='lattice':
        for x0,y0 in ((8,-9),(12,-2),(7,5)):
            for x in range(x0,x0+3):
                for y in range(y0,y0+4):blocks.add((x,y))
        bar_y(-5,12,16,(14+variation,))
        for x in range(11,14):protected.add((x,3+variation))
    else:raise ValueError('unexpected motif')
    # Central eastbound cul-de-sac and no painting through its flanks.
    for x in range(1,8):
        blocks.add((x,-1));blocks.add((x,1))
    blocks.add((8,0))
    blocks.add((7,0))
    for x in range(1,7):blocks.discard((x,0))
    return blocks, protected


def fixture(sp, motif, depth, quarter, goal_index):
    if sp.split not in ('train','development') or sp.challenge!='normal':
        raise ValueError('No structural HELDOUT or unsupported base challenge')
    if motif not in (TRAIN_MOTIFS if sp.split=='train' else DEV_MOTIFS):
        raise ValueError('Motif split leakage')
    if depth not in DEPTHS or quarter not in ROTATIONS or goal_index not in (0,1):
        raise ValueError('Out-of-bounds fixture params')
    w=World.create(sp)
    # Explicit controlled local overwrite; outside original geometry is retained.
    local={(x,y) for x in range(29,68) for y in range(29,68)}
    cleared_materials=sum(p in local for p in w.materials)
    cleared_protected=len(w.protected & local)
    for p in local:
        w.materials.pop(p,None)
        w.protected.discard(p)
    variation=(sp.seed%2)
    blocks,protected=terrain(motif,variation)
    for p in blocks:w.materials[convert(p,quarter)]=SOLID if (p[0]+p[1])%3 else MATERIAL_Z
    for p in protected:w.protected.add(convert(p,quarter))
    goal=convert(TARGETS[goal_index],quarter)
    assert not w.occupied(goal) and not w.occupied((48,48))
    dir_action=('MOVE_E','MOVE_S','MOVE_W','MOVE_N')[quarter]
    for _ in range(depth):
        assert dir_action in w.menu(12)
        assert strict_step(w,dir_action) is not None
    assert w.logical_moves==depth
    assert not w.ink & (set(w.materials) | w.protected)
    return w,goal,{'cleared_source_material_cells':cleared_materials,
                   'cleared_source_protected_cells':cleared_protected,
                   'new_obstacle_cells':len(blocks),'new_protected_cells':len(protected)}


def oracle(w,goal):
    p=prefix_plan(w,goal)
    return p[0] if p else 'STOP'


def action_cost(w,a):
    return len(w.path)-1 if a=='RELOCATE_START' else (1 if a in MOVE or a=='BACKTRACK' else 0)


def execute(w,a):
    if a not in w.menu(12):raise AssertionError('Not host legal')
    if w.logical_moves+action_cost(w,a)>64:return None
    return w.apply(a,12)


def train_corpus(seed, stage):
    X=[];M=[];Y=[];acts=Counter();forced=0;tasks=0
    rng=random.Random(seed*31+stage)
    for sp in specs('train',1):
        for motif in TRAIN_MOTIFS:
            for quarter in (0,2):
                for depth in DEPTHS:
                    for gi in (0,1):
                        w,goal,_=fixture(sp,motif,depth,quarter,gi)
                        assert prefix_plan(w,goal) is not None
                        tasks+=1
                        for tick in range(min(stage,26,64-depth)):
                            if w.cursor==goal or w.stopped or w.logical_moves>=64:break
                            menu,x=matrix(w,goal,64,False)
                            a=oracle(w,goal)
                            mask=np.zeros(MAX_ACTIONS,dtype=bool);mask[:len(menu)]=True
                            X.append(x);M.append(mask);Y.append(menu.index(a));acts[a]+=1
                            candidates=[b for b in menu if b in MOVE and b!=a]
                            if candidates and rng.random()<.10 and w.logical_moves+1<64:
                                a=rng.choice(candidates);forced+=1
                            if execute(w,a) is None:break
    return np.stack(X),np.stack(M),np.asarray(Y,dtype=np.int64),{
        'tasks':tasks,'examples':len(Y),'teacher_action_counts':dict(sorted(acts.items())),
        'forced_bad_moves':forced,'stage':stage,'max_actual_continuation_training_steps':26,'motifs':list(TRAIN_MOTIFS)}


def fit(seed,steps):
    model=GoalRanker(seed)
    stages=[]
    for stage in (36,72,128):
        X,M,Y,meta=train_corpus(seed,stage)
        training=model.fit(X,M,Y,steps,seed*100+stage)
        stages.append({'data':meta,'optimizer':training})
    return model,stages


def episode(sp,motif,depth,quarter,gi,arm,seed,model):
    w,goal,terrain_info=fixture(sp,motif,depth,quarter,gi)
    origin=w.state_sha()
    base=w.base_sha()
    prelude=list(w.decisions)
    events=[];probs=[];latencies=[];why='64_gate'
    for tick in range(64):
        if w.cursor==goal:why='goal_arrived';break
        if w.stopped:why='stopped';break
        if w.logical_moves>=64:why='64_gate';break
        t=time.perf_counter_ns()
        if arm=='prefix_oracle':a,p=oracle(w,goal),None
        else:a,p=model.choose(w,goal,64,arm=='masked_new')
        latencies.append((time.perf_counter_ns()-t)/1e6)
        if a not in w.menu(12):raise AssertionError('model ignored host menu')
        if p is not None:
            probs.append({'tick':tick,'state_sha256':w.state_sha(),'chosen':a,'distribution':p})
        e=execute(w,a)
        if e is None:why='64_gate';break
        events.append(e)
    else:why='decision_guard'
    counts=dict(sorted(Counter(e['chosen'] for e in events).items()))
    row={'spec':sp.ident,'family':sp.family,'world_seed':sp.seed,
         'motif':motif,'depth':depth,'quarter':quarter,'goal_index':gi,
         'seed':seed,'arm':arm,'goal':list(goal),'prelude_sha256':origin,'base_sha256':base,
         'arrived':w.cursor==goal,'stop':why,'actions':len(events),'logical_moves':w.logical_moves,
         'protected_or_material_edits':len(w.ink & (set(w.materials)|w.protected)),
         'synthetic_relocations':counts.get('RELOCATE_START',0),
         'counts':counts,'last_cursor':list(w.cursor),'trace_sha256':digest(w.decisions),
         'final_sha256':w.state_sha(),'median_select_ms':round(statistics.median(latencies),6) if latencies else None,
         'terrain':terrain_info}
    trace={'row':row,'spec':{'split':sp.split,'family':sp.family,'seed':sp.seed,
                           'mode':sp.mode,'challenge':sp.challenge},
           'prelude':prelude,'events':events,'probabilities':probs}
    return row,trace


def verify_trace(trace,model):
    r=trace['row'];s=Spec(**trace['spec'])
    w,goal,info=fixture(s,r['motif'],r['depth'],r['quarter'],r['goal_index'])
    assert trace['prelude']==w.decisions and w.state_sha()==r['prelude_sha256']
    assert r['terrain']==info and w.base_sha()==r['base_sha256']
    probs=iter(trace['probabilities']);events=iter(trace['events'])
    count=0;nprob=0
    for tick in range(64):
        if w.cursor==goal:why='goal_arrived';break
        if w.stopped:why='stopped';break
        if w.logical_moves>=64:why='64_gate';break
        if r['arm']=='prefix_oracle':a,p=oracle(w,goal),None
        else:a,p=model.choose(w,goal,64,r['arm']=='masked_new')
        if p is not None:
            rec=next(probs)
            assert rec=={'tick':tick,'state_sha256':w.state_sha(),'chosen':a,'distribution':p}
            nprob+=1
        if w.logical_moves+action_cost(w,a)>64:why='64_gate';break
        rec=next(events)
        assert rec==w.apply(a,12)
        count+=1
    else:why='decision_guard'
    assert next(events,None) is None and next(probs,None) is None
    assert (w.cursor==goal)==r['arrived'] and why==r['stop']
    assert digest(w.decisions)==r['trace_sha256'] and w.state_sha()==r['final_sha256']
    assert w.logical_moves==r['logical_moves'] and count==r['actions']
    assert r['protected_or_material_edits']==0
    return count+len(trace['prelude']),nprob


def summary(rows):
    out={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm]
        cnt=Counter()
        for r in rr:cnt.update(r['counts'])
        out[arm]={'episodes':len(rr),'arrivals':sum(x['arrived'] for x in rr),
                  'mean_actions':round(statistics.mean(x['actions'] for x in rr),3),
                  'median_select_ms':round(statistics.median([x['median_select_ms'] for x in rr if x['median_select_ms'] is not None]),6),
                  'by_motif':{m:{'episodes':sum(r['motif']==m for r in rr),
                                 'arrivals':sum(r['arrived'] for r in rr if r['motif']==m)} for m in DEV_MOTIFS},
                  'action_counts':dict(sorted(cnt.items()))}
    return out


def run(out,old_cp,updates=64):
    if out.exists():raise FileExistsError('immutable output exists')
    if not 1<=updates<=160:raise ValueError('unbounded optimizer')
    out.mkdir(parents=True);(out/'traces').mkdir()
    t=time.perf_counter();models={};training={};checkpoints={}
    for new_seed,old_seed in SEED_PAIRS:
        m,stages=fit(new_seed,updates);models[(new_seed,'new')]=m
        models[(new_seed,'masked_new')]=m
        training[str(new_seed)]=stages
        p=out/f'{new_seed}-topology.npz';m.save(p)
        rt=GoalRanker.load(p)
        assert all(np.array_equal(m.p[k],rt.p[k]) for k in m.p)
        checkpoints[f'{new_seed}-topology']={'sha256':sha_file(p),'params':sum(x.size for x in m.p.values()),'updates':m.updates}
        q=old_cp/f'{old_seed}-conditioned.npz';assert q.is_file()
        frozen=GoalRanker.load(q);assert frozen.updates==120
        models[(new_seed,'frozen')]=frozen
        checkpoints[f'{old_seed}-frozen']={'sha256':sha_file(q),'params':sum(x.size for x in frozen.p.values()),'updates':frozen.updates}
    rows=[];rec_count=0;vec_count=0
    for new_seed,_ in SEED_PAIRS:
        for sp in specs('development',1):
            for motif in DEV_MOTIFS:
                for depth in DEPTHS:
                    quarter=(sp.seed+depth)%4
                    for gi in (0,1):
                        w,goal,_=fixture(sp,motif,depth,quarter,gi)
                        assert prefix_plan(w,goal) is not None
                        for arm in ARMS:
                            model=models.get((new_seed,arm))
                            r,tr=episode(sp,motif,depth,quarter,gi,arm,new_seed,model)
                            receipts,vectors=verify_trace(tr,model)
                            rec_count+=receipts;vec_count+=vectors;rows.append(r)
                            name=f'{new_seed}-{sp.family}-{sp.seed}-{motif}-{depth}-{quarter}-{gi}-{arm}.json'
                            (out/'traces'/name).write_text(json.dumps(tr,sort_keys=True,separators=(',',':'))+'\n')
    ev={'version':VERSION,'date':'2026-11-18','split':'development','heldout_opened':False,
        'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in SOURCE},
        'checkpoints':checkpoints,'training':training,'updates_per_stage':updates,
        'episodes':len(rows),'verified_action_receipts_including_preludes':rec_count,
        'verified_neural_probability_vectors':vec_count,'summary':summary(rows),
        'per_episode':rows,'wall_seconds':round(time.perf_counter()-t,3),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'native_game_receipts':0,'synthetic_64_logical_gate':True,
        'limitations':['Locally modified central 39x39 terrain patch; base World families remain but central original terrain is replaced.',
          'Two new DEVELOPMENT overlay motifs share construction primitives with TRAIN motifs; this is not sealed structural HELDOUT.',
          'Oracle solves only backtrack-prefix then empty-cell path class, not native/global optimal planning.',
          'No native CyberSand paint/actor receipts, model ABI, or real GPU model comparison.',
          'Checkpoint files are local ZIP payloads pending immutable Actions artifacts.']}
    (out/'evidence.json').write_text(json.dumps(ev,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return ev


def verify(out,old_cp):
    e=json.loads((out/'evidence.json').read_text());sums=json.loads((out/'SHA256SUMS.json').read_text())
    assert sums=={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}
    assert e['source_sha256']=={n:sha_file(Path(__file__).parent/n) for n in SOURCE}
    models={}
    for new_seed,old_seed in SEED_PAIRS:
        p=out/f'{new_seed}-topology.npz';q=old_cp/f'{old_seed}-conditioned.npz'
        assert sha_file(p)==e['checkpoints'][f'{new_seed}-topology']['sha256']
        assert sha_file(q)==e['checkpoints'][f'{old_seed}-frozen']['sha256']
        m=GoalRanker.load(p);models[(new_seed,'new')]=m;models[(new_seed,'masked_new')]=m
        models[(new_seed,'frozen')]=GoalRanker.load(q)
    rows=[];receipts=0;vectors=0
    for p in sorted((out/'traces').glob('*.json')):
        tr=json.loads(p.read_text());r=tr['row']
        n,k=verify_trace(tr,models.get((r['seed'],r['arm'])))
        receipts+=n;vectors+=k;rows.append(r)
    assert len(rows)==e['episodes']
    assert receipts==e['verified_action_receipts_including_preludes']
    assert vectors==e['verified_neural_probability_vectors']
    assert summary(rows)==e['summary']
    return {'episodes':len(rows),'receipts':receipts,'vectors':vectors}



def train_one(out, seed, updates):
    if seed not in dict(SEED_PAIRS) or not 1<=updates<=160:
        raise ValueError('bad seed/update bound')
    out.mkdir(parents=True,exist_ok=True)
    p=out/f'{seed}-topology.npz';meta=out/f'{seed}-training.json'
    if p.exists() or meta.exists():raise FileExistsError('immutable training output')
    m,stages=fit(seed,updates);m.save(p)
    assert all(np.array_equal(m.p[k],GoalRanker.load(p).p[k]) for k in m.p)
    meta.write_text(json.dumps({'seed':seed,'updates_per_stage':updates,'stages':stages,
                       'checkpoint_sha256':sha_file(p)},sort_keys=True,indent=2)+'\n')
    return {'seed':seed,'updates':m.updates,'sha256':sha_file(p),'stages':len(stages)}


def eval_one(out,old_cp,new_seed,family):
    if new_seed not in dict(SEED_PAIRS) or family not in [s.family for s in specs('development',1)]:
        raise ValueError('invalid split/seed family')
    trace_dir=out/'traces';trace_dir.mkdir(parents=True,exist_ok=True)
    old_seed=dict(SEED_PAIRS)[new_seed]
    models={'new':GoalRanker.load(out/f'{new_seed}-topology.npz'),
            'frozen':GoalRanker.load(old_cp/f'{old_seed}-conditioned.npz')}
    models['masked_new']=models['new']
    rows=[];receipts=vectors=0;t=time.perf_counter()
    for sp in specs('development',1):
        if sp.family!=family:continue
        for motif in DEV_MOTIFS:
            for depth in DEPTHS:
                quarter=(sp.seed+depth)%4
                for gi in (0,1):
                    w,goal,_=fixture(sp,motif,depth,quarter,gi)
                    assert prefix_plan(w,goal) is not None
                    for arm in ARMS:
                        model=models.get(arm)
                        r,tr=episode(sp,motif,depth,quarter,gi,arm,new_seed,model)
                        n,k=verify_trace(tr,model)
                        receipts+=n;vectors+=k;rows.append(r)
                        name=f'{new_seed}-{sp.family}-{sp.seed}-{motif}-{depth}-{quarter}-{gi}-{arm}.json'
                        pp=trace_dir/name
                        if pp.exists():raise FileExistsError(pp)
                        pp.write_text(json.dumps(tr,sort_keys=True,separators=(',',':'))+'\n')
    meta=out/f'{new_seed}-{family}-eval.json'
    if meta.exists():raise FileExistsError(meta)
    data={'rows':rows,'receipts':receipts,'vectors':vectors,'wall_seconds':round(time.perf_counter()-t,3),
          'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    meta.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return {'seed':new_seed,'family':family,'episodes':len(rows),'receipts':receipts,
            'vectors':vectors,'wall_seconds':data['wall_seconds']}


def finalize(out,old_cp):
    if (out/'evidence.json').exists():raise FileExistsError('evidence sealed')
    source_dir=Path(__file__).parent
    training={};checkpoints={};records=[];receipts=vectors=0;wall=0.;max_rss=0
    for new_seed,old_seed in SEED_PAIRS:
        meta=json.loads((out/f'{new_seed}-training.json').read_text())
        p=out/f'{new_seed}-topology.npz';q=old_cp/f'{old_seed}-conditioned.npz'
        assert meta['checkpoint_sha256']==sha_file(p)
        m=GoalRanker.load(p);frozen=GoalRanker.load(q)
        assert m.updates==meta['updates_per_stage']*3 and frozen.updates==120
        training[str(new_seed)]=meta['stages']
        checkpoints[f'{new_seed}-topology']={'sha256':sha_file(p),'params':sum(v.size for v in m.p.values()),'updates':m.updates}
        checkpoints[f'{old_seed}-frozen']={'sha256':sha_file(q),'params':sum(v.size for v in frozen.p.values()),'updates':frozen.updates}
        for family in [s.family for s in specs('development',1)]:
            detail=json.loads((out/f'{new_seed}-{family}-eval.json').read_text())
            records.extend(detail['rows']);receipts+=detail['receipts'];vectors+=detail['vectors']
            wall+=detail['wall_seconds'];max_rss=max(max_rss,detail['peak_rss_kib'])
    all_paths=list((out/'traces').glob('*.json'))
    assert len(all_paths)==len(records)==len(SEED_PAIRS)*len(specs('development',1))*len(DEV_MOTIFS)*len(DEPTHS)*2*len(ARMS)
    ev={'version':VERSION,'date':'2026-11-18','split':'development','heldout_opened':False,
        'source_sha256':{n:sha_file(source_dir/n) for n in SOURCE},
        'checkpoints':checkpoints,'training':training,
        'episodes':len(records),'verified_action_receipts_including_preludes':receipts,
        'verified_neural_probability_vectors':vectors,'summary':summary(records),
        'per_episode':records,'evaluation_cpu_seconds_sum':round(wall,3),
        'per_slice_peak_rss_kib_max':max_rss,
        'native_game_receipts':0,'synthetic_64_logical_gate':True,
        'limitations':['Locally modified central 39x39 terrain patch; base World families remain but central original terrain is replaced.',
          'Two new DEVELOPMENT overlay motifs share construction primitives with TRAIN motifs; this is not sealed structural HELDOUT.',
          'Oracle solves only backtrack-prefix then empty-cell path class, not native/global optimal planning.',
          'No native CyberSand paint/actor receipts, model ABI, or real GPU model comparison.',
          'Checkpoint files are local ZIP payloads pending immutable Actions artifacts.']}
    (out/'evidence.json').write_text(json.dumps(ev,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return {'episodes':len(records),'receipts':receipts,'vectors':vectors,'summary':ev['summary']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path)
    ap.add_argument('--old-checkpoints',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=64);ap.add_argument('--verify',type=Path)
    ap.add_argument('--train-one',type=int);ap.add_argument('--eval-one',type=int)
    ap.add_argument('--family');ap.add_argument('--finalize',action='store_true')
    args=ap.parse_args()
    if args.train_one is not None:result=train_one(args.out,args.train_one,args.updates)
    elif args.eval_one is not None:result=eval_one(args.out,args.old_checkpoints,args.eval_one,args.family)
    elif args.finalize:result=finalize(args.out,args.old_checkpoints)
    elif args.verify:result=verify(args.verify,args.old_checkpoints)
    else:result=run(args.out,args.old_checkpoints,args.updates)
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_episode','training','source_sha256')},indent=2))

if __name__=='__main__':main()

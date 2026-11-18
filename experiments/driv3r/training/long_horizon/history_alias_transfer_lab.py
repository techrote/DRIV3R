"""HISTORY-ALIAS-TRANSFER-001: rotated and recovery-budget-shifted alias stress.

Frozen checkpoints only; no heldout structural families, native execution, game
mutation, training, or native-ABI claims. All actions are synthetic host-legal.
"""
from __future__ import annotations
import argparse, copy, json, resource, statistics, time
from collections import Counter
from pathlib import Path
import numpy as np
from . import history_alias_lab as old
from .world import Spec, World, digest, replay
from .train import sha_file

VERSION = 'history-alias-transfer/1'
# Disjoint from old 3007–3307 DEVELOPMENT seeds, and distinct from TRAIN.
FAMILIES = ('pickets', 'ridges', 'islands', 'zigzag')
ROTATIONS = (0, 1, 2, 3)
BUDGETS = (2, 3, 4)
CASES = 20
HORIZON = 12
SOURCES = ('history_alias_transfer_lab.py','history_alias_lab.py',
           'history_destination_lab.py','goal_recovery_lab.py','goal_conditioning_lab.py',
           'world.py','policy.py','train.py','test_history_alias_transfer_lab.py')
EXTRA_DETOUR = ((0,-1),)

def rotate(q, k):
    x,y=q
    for _ in range(k):x,y=-y,x
    return x,y

def make_state(sp, path, rotation, remaining):
    if remaining not in BUDGETS:raise ValueError('invalid recovery budget')
    w=World.create(sp)
    # 12 - (detour length + one terminal recovery) = requested remainder.
    # For 2/3/4, use detour length 9/8/7 respectively.
    detour=old.DETOUR+EXTRA_DETOUR
    prior=(0,0)
    for q in detour[:11-remaining]:
        after=rotate(q,rotation);before=rotate(prior,rotation)
        a=old.legal_move(before,after)
        if a not in w.menu(12):raise ValueError(('illegal detour', sp,rotation,remaining,a,w.menu(12)))
        w.apply(a,12);prior=q
    for _ in detour[:11-remaining]:
        if 'BACKTRACK' not in w.menu(12):raise ValueError('missing legal recovery')
        w.apply('BACKTRACK',12)
    for a,b in zip(path,path[1:]):
        a,b=rotate(a,rotation),rotate(b,rotation)
        act=old.legal_move(a,b)
        if act not in w.menu(12):raise ValueError(('illegal path',sp,rotation,remaining,act,w.menu(12)))
        w.apply(act,12)
    if 'BACKTRACK' not in w.menu(12):raise ValueError('missing terminal recovery')
    w.apply('BACKTRACK',12)
    if 12-w.recoveries != remaining:raise AssertionError('budget was not consumed through host actions')
    if w.logical_moves>64:raise AssertionError('synthetic M3 move proxy exceeded')
    return w

def cases():
    originals=old.proposals()
    originals=[r for r in originals if r[0]>=13][:CASES]
    assert len(originals)==CASES
    for i,(_,sig,pos,neg,g) in enumerate(originals):
        family=FAMILIES[i%len(FAMILIES)]
        sp=Spec('development',family,3507+(i%4)*100+i//4,
                ('wall','route','motif')[i%3])
        for rot in ROTATIONS:
            for remaining in BUDGETS:
                a=make_state(sp,pos,rot,remaining)
                b=make_state(sp,neg,rot,remaining)
                goal=(48+rotate(g,rot)[0],48+rotate(g,rot)[1])
                if not (a.ink==b.ink and a.materials==b.materials and
                    a.protected==b.protected and a.cursor==b.cursor and
                    a.heading==b.heading and a.recoveries==b.recoveries and
                    a.steps==b.steps and a.last_action==b.last_action and
                    a.menu()==b.menu()):
                    raise AssertionError('nonmatched controlled pair')
                if not np.array_equal(blind_matrix(a,goal)[1], blind_matrix(b,goal)[1]):
                    raise AssertionError('goal-only observation leaks history')
                for kind,w in (('positive',a),('negative',b)):
                    d=len(w.path)-1-w.path.index(goal) if goal in w.path else 100
                    correct='BACKTRACK' if d<=remaining else 'STOP'
                    if correct not in w.menu(12):raise AssertionError('teacher action is not host-legal')
                    yield {'id':sig,'spec':sp,'rotation':rot*90,
                           'remaining':remaining,'kind':kind,'world':w,'goal':goal,
                           'depth':d,'label':correct}

def blind_matrix(w,goal):
    menu,x=old.matrix(w,goal,128,True)
    x[:,old.prior.prior.goal.FEATURE_DIM:old.prior.WIDTH]=0
    return menu,x

def model_matrix(w,goal,arm):
    if arm=='blind':return blind_matrix(w,goal)
    menu,x=old.matrix(w,goal,128)
    if arm=='raw':x[:,-2:]=0
    return menu,x

def rollout(c, model, arm, seed, full=False):
    w=copy.deepcopy(c['world']);g=c['goal'];original_len=len(w.decisions)
    records=[];first='';hit=0 if w.cursor==g else None;counts=Counter();inference=[]
    for t in range(HORIZON):
        if w.stopped or hit is not None:break
        menu,x=model_matrix(w,g,arm)
        if menu!=w.menu(12):raise AssertionError('host menu discrepancy')
        t0=time.perf_counter_ns();p=model.probs(x,len(menu));inference.append((time.perf_counter_ns()-t0)/1e6)
        if not np.isfinite(p).all() or abs(float(p.sum())-1)>1e-8:raise AssertionError('invalid probabilities')
        choice=menu[int(p.argmax())];first=first or choice
        event=w.apply(choice,12);counts[choice]+=1
        if full:records.append({'menu':menu,'probs':[round(float(z),10) for z in p],
                                'chosen':choice,'receipt':event})
        if w.cursor==g:hit=t+1
    return {'case':c['id'],'spec':c['spec'].__dict__,'rotation':c['rotation'],
            'remaining':c['remaining'],'kind':c['kind'],'depth':c['depth'],
            'goal':list(g),'seed':seed,'arm':arm,'label':c['label'],
            'first':first,'first_correct':first==c['label'],'hit_step':hit,
            'decisions':w.steps-c['world'].steps,'executed_actions':dict(sorted(counts.items())),
            'origin_sha':c['world'].state_sha(),'end_sha':w.state_sha(),
            'trace_sha':digest(w.decisions[original_len:]),'rows':records,
            '_inference':inference}

def summaries(rows):
    result={}
    for arm in ('blind','raw','depth'):
        subset=[r for r in rows if r['arm']==arm]
        table={'all':subset}
        table.update({f'rotation_{v}':[r for r in subset if r['rotation']==v] for v in (0,90,180,270)})
        table.update({f'budget_{v}':[r for r in subset if r['remaining']==v] for v in BUDGETS})
        result[arm]={}
        for k,rs in table.items():
            action=Counter();[action.update(r['executed_actions']) for r in rs]
            result[arm][k]={'n':len(rs),'first_correct':sum(r['first_correct'] for r in rs),
                       'reachable':sum(r['label']=='BACKTRACK' for r in rs),
                       'reached':sum(r['hit_step'] is not None for r in rs if r['label']=='BACKTRACK'),
                       'unreachable_immediate_stop':sum(r['first']=='STOP' for r in rs if r['label']=='STOP'),
                       'actions':dict(sorted(action.items())),
                       'mean_decisions':round(sum(r['decisions'] for r in rs)/max(1,len(rs)),4)}
    return result

def load_models(ckpt):
    return {s:{a:old.Ranker.load(ckpt/f'{s}-{a}.npz') for a in ('blind','raw','depth')}
            for s in (619,620)}

def evaluate(models,full=False):
    rows=[];stored=[];lat=[]
    for i,c in enumerate(cases()):
        for seed,mm in models.items():
            for arm,m in mm.items():
                sample=full and i<36 and c['rotation'] in (90,180) and c['remaining']==3
                r=rollout(c,m,arm,seed,sample)
                lat+=r.pop('_inference')
                if sample:stored.append(copy.deepcopy(r))
                r.pop('rows');rows.append(r)
    return rows,stored,lat

def run(out,ckpt):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    t=time.perf_counter()
    models=load_models(ckpt)
    checkpoint_sha={f'{s}-{a}':sha_file(ckpt/f'{s}-{a}.npz') for s in models for a in models[s]}
    rows,samples,lat=evaluate(models,True)
    # Verify that training originals have not been transformed into train evidence.
    original_source={p:sha_file(Path(__file__).parent/p) for p in SOURCES}
    (out/'episodes.json').write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    (out/'samples.json').write_text(json.dumps(samples,sort_keys=True,separators=(',',':'))+'\n')
    result={'version':VERSION,'source_sha256':original_source,'checkpoint_sha256':checkpoint_sha,
      'partitions':{'training':'unchanged original path motifs lengths 10-12',
        'development':'20 length-13/14 motifs; unseen rotated orientations and world seeds 3507-3807+',
        'structural_HELDOUT':'sealed'},'case_count':len(list(cases())),
      'episode_count':len(rows),'decisions':sum(r['decisions'] for r in rows),
      'summary':summaries(rows),'elapsed_s':round(time.perf_counter()-t,3),
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'inference_latency_ms':{'median':round(statistics.median(lat),6),
          'p95':round(float(np.quantile(lat,.95)),6),'scope':'model probabilities only'},
      'native_execution':False,'optimizer_steps':0,'paid_compute':False}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(p.relative_to(out)):sha_file(p)
       for p in sorted(out.rglob('*')) if p.is_file()},sort_keys=True,indent=2)+'\n')
    return result

def verify(out,ckpt):
    data=json.loads((out/'report.json').read_text())
    if data['source_sha256']!={p:sha_file(Path(__file__).parent/p) for p in SOURCES}:
        raise AssertionError('source drift')
    sums=json.loads((out/'SHA256SUMS.json').read_text())
    if sums!={str(p.relative_to(out)):sha_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'}:
        raise AssertionError('evidence drift')
    models=load_models(ckpt)
    if data['checkpoint_sha256']!={f'{s}-{a}':sha_file(ckpt/f'{s}-{a}.npz') for s in models for a in models[s]}:
        raise AssertionError('checkpoint drift')
    rows,_,_=evaluate(models,False)
    if rows!=json.loads((out/'episodes.json').read_text()):raise AssertionError('episode replay drift')
    if summaries(rows)!=data['summary']:raise AssertionError('summary replay drift')
    sample=json.loads((out/'samples.json').read_text())
    lookup={(c['id'],c['rotation'],c['remaining'],c['kind']):c for c in cases()}
    for recorded in sample:
        c=lookup[(recorded['case'],recorded['rotation'],recorded['remaining'],recorded['kind'])]
        restored=replay(c['spec'],c['world'].decisions)
        if restored.state_sha()!=c['world'].state_sha():raise AssertionError('prehistory receipt reconstruction')
        for event in recorded['rows']:
            if event['menu']!=restored.menu(12) or event['receipt']!=restored.apply(event['chosen'],12):
                raise AssertionError('independent host receipt reconstruction')
        res=rollout(c,models[recorded['seed']][recorded['arm']],recorded['arm'],recorded['seed'],True)
        res.pop('_inference')
        if res!=recorded:raise AssertionError('sample probability/receipt replay drift')
    return {'episodes_replayed':len(rows),'full_probability_receipt_replays':len(sample),'hashes_verified':True}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--checkpoints',required=True,type=Path);ap.add_argument('--verify',action='store_true')
    a=ap.parse_args();print(json.dumps(verify(a.output,a.checkpoints) if a.verify else run(a.output,a.checkpoints),indent=2))

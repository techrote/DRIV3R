"""LOOP-GUARD-001: bounded observation-derived memory ablation.

Frozen SEQUENTIAL-PROBE-001 checkpoints, unchanged neural probabilities and
host-legal action menus. All interventions are research-side inference only.
No native ABI, production game, paid training or structural heldout changes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import zipfile

from . import sequential_probe_lab as base
from . import contrastive_fork_lab as rotate
from .world import MOVE, Spec, specs
from .train import sha_file

VERSION='loop-guard/1'
ARMS=('raw','anti_reverse','conserve','sham_conserve')
KIND=('trap','open')
HORIZON=128
SEEDS=(811,814)
SOURCE=('loop_guard_lab.py','sequential_probe_lab.py','sensing_probe_lab.py','menu_alias_lab.py',
        'contrastive_fork_lab.py','recovery_fork_lab.py','goal_recovery_lab.py',
        'goal_conditioning_lab.py','goal_retarget_lab.py','world.py','policy.py','train.py')

def candidate(w, action):
    if action in MOVE:
        dx,dy=MOVE[action]
        return (w.cursor[0]+dx,w.cursor[1]+dy)
    if action=='BACKTRACK' and len(w.path)>1:return w.path[-2]
    if action=='RELOCATE_START':return w.path[0]
    return w.cursor

def decision(w, model, q, state, arm, prev, seen_trap):
    if arm not in ARMS:raise ValueError('arm')
    obs=base.observe(w,'sequential7',q['forward'],q['retreat'],state)
    menu,features=base.features(w,q['goal'],HORIZON,q['forward'],q['retreat'],state,obs)
    probs=model.probs(features,len(menu))
    probability={a:round(float(probs[i]),8) for i,a in enumerate(menu)}
    found=bool(obs['hit'] and obs['reads']==3 and state['retreating'])
    seen=seen_trap or found
    active=(seen if arm!='sham_conserve' else state['spent']==7)
    rejected=[]
    available=[a for a in menu if a in MOVE]
    if active and not state['retreating']:
        if arm in ('conserve','sham_conserve') and available:
            rejected.extend(a for a in menu if a not in MOVE)
        if arm in ('anti_reverse','conserve','sham_conserve') and prev is not None:
            other=[a for a in available if candidate(w,a)!=prev]
            if other:
                rejected.extend(a for a in available if candidate(w,a)==prev)
    accepted=[i for i,a in enumerate(menu) if a not in rejected]
    if not accepted:raise ValueError('guard suppressed entire legal menu')
    selected=menu[max(accepted,key=lambda i:float(probs[i]))]
    if selected not in w.menu(12):raise ValueError('not host legal')
    return selected,probability,obs,seen,sorted(set(rejected)),menu

def episode(sp,turns,kind,arm,model):
    if sp.split!='development' or turns not in rotate.ROTATION or kind not in KIND or arm not in ARMS:
        raise ValueError('DEVELOPMENT only, sealed HELDOUT')
    q=base.fixture(sp,turns,kind)
    w=q['world'];state=base.fresh_state();previous=None;detected=False
    initial=w.state_sha();prefix=q['prefix'];old_ink=len(w.ink)
    rows=[];lat=[];recovery_on_detection=None
    for t in range(HORIZON):
        if w.stopped:break
        tick=time.perf_counter_ns()
        action,prob,obs,detected_next,rejected,menu=decision(w,model,q,state,arm,previous,detected)
        lat.append((time.perf_counter_ns()-tick)/1e6)
        if recovery_on_detection is None and detected_next:recovery_on_detection=w.recoveries
        before=w.cursor
        event=w.apply(action,12)
        base.advance_state(w,q['forward'],q['retreat'],state,action,before)
        rows.append({'action':action,'probabilities':prob,'observation':obs,
                     'rejected':rejected,'menu':menu,'detected':detected_next,'event':event})
        previous=before;detected=detected_next
    return {'version':VERSION,'split':'development','spec':sp.__dict__,'turns':turns,
            'kind':kind,'arm':arm,'horizon':HORIZON,'initial_sha256':initial,'prefix':prefix,
            'base_sha256':w.base_sha(),'final_sha256':w.state_sha(),'rows':rows,
            'sensor_cells_read':state['spent'],'recovery_on_detection':recovery_on_detection,
            'painted':len(w.ink)-old_ink,'recoveries':w.recoveries,
            'early_stop':int(len(rows)<HORIZON),'detect':detected,
            'mean_latency_ms':round(statistics.mean(lat),6) if lat else 0.}

def verify_episode(ep,model):
    if ep['version']!=VERSION or ep['split']!='development' or ep['horizon']!=HORIZON:
        raise ValueError('episode identity')
    sp=Spec(**ep['spec']);q=base.fixture(sp,ep['turns'],ep['kind']);w=q['world']
    if ep['initial_sha256']!=w.state_sha() or ep['prefix']!=q['prefix'] or ep['base_sha256']!=w.base_sha():
        raise ValueError('world provenance')
    state=base.fresh_state();previous=None;detected=False;cost=None;old_ink=len(w.ink)
    for row in ep['rows']:
        if w.stopped:raise ValueError('post STOP actions')
        action,prob,obs,seen,rejected,menu=decision(w,model,q,state,ep['arm'],previous,detected)
        if (row['action']!=action or row['probabilities']!=prob or row['observation']!=obs
                or row['rejected']!=rejected or row['menu']!=menu or row['detected']!=seen):
            raise ValueError('neural probability / legal menu / observation mismatch')
        if cost is None and seen:cost=w.recoveries
        before=w.cursor;ev=w.apply(action,12)
        if row['event']!=ev:raise ValueError('synthetic host event mismatch')
        base.advance_state(w,q['forward'],q['retreat'],state,action,before)
        previous=before;detected=seen
    if (len(ep['rows'])>HORIZON or (not w.stopped and len(ep['rows'])!=HORIZON)
        or ep['sensor_cells_read']!=state['spent'] or state['spent']>7
        or ep['recovery_on_detection']!=cost or ep['detect']!=detected
        or ep['painted']!=len(w.ink)-old_ink or ep['recoveries']!=w.recoveries
        or ep['early_stop']!=int(len(ep['rows'])<HORIZON) or ep['final_sha256']!=w.state_sha()):
        raise ValueError('terminal evidence mismatch')
    return len(ep['rows'])

def summarize(episodes):
    report={}
    for kind in KIND:
        for arm in ARMS:
            eps=[ep for ep in episodes if ep['kind']==kind and ep['arm']==arm]
            if not eps:continue
            action_count=Counter(row['action'] for ep in eps for row in ep['rows'])
            report[kind+'/'+arm]={'n':len(eps),'mean_paint':round(statistics.mean(e['painted'] for e in eps),3),
                'early_stop':sum(e['early_stop'] for e in eps),
                'mean_recoveries':round(statistics.mean(e['recoveries'] for e in eps),3),
                'mean_decisions':round(statistics.mean(len(e['rows']) for e in eps),3),
                'detected':sum(e['detect'] for e in eps),
                'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in eps),6),
                'guarded_actions':sum(bool(r['rejected']) for e in eps for r in e['rows']),
                'action_counts':dict(sorted(action_count.items()))}
    return report

def source_shas():return {name:sha_file(Path(__file__).parent/name) for name in SOURCE}

def run(out,prior_zip,quick=False,validation=False):
    if out.exists():raise FileExistsError('immutable evidence directory')
    with zipfile.ZipFile(prior_zip) as z:
        prior={}
        for seed in SEEDS[:1 if quick else 2]:
            checkpoint=f'final-{seed}/sequential7.npz'
            blob=z.read(checkpoint)
            record=json.loads(z.read(f'final-{seed}/report.json'))
            if hashlib.sha256(blob).hexdigest()!=record['checkpoints']['sequential7']['sha256']:
                raise ValueError('prior checkpoint provenance')
            prior[seed]=(blob,record)
    out.mkdir(parents=True)
    (out/'traces').mkdir();(out/'checkpoints').mkdir()
    episodes=[];cps={};start=time.perf_counter()
    for seed,(blob,prior_report) in prior.items():
        cp=out/'checkpoints'/f'sequential7-{seed}.npz';cp.write_bytes(blob)
        model=base.ProbeRanker.load(cp)
        cps[str(seed)]={'sha256':sha_file(cp),'parameters':sum(v.size for v in model.p.values()),
                        'source_campaign':prior_report['version'],'prior_source_sha256':prior_report['source_sha256']}
        worlds=(specs('development',1)[:1] if quick else
                [x for x in specs('development',2) if (x.seed%100 == (1 if validation else 0))])
        for sp in worlds:
            for turns in ((0,) if quick else ((2,3) if validation else (0,1))):
                base.matched(sp,turns)
                for kind in KIND:
                    for arm in ARMS:
                        ep=episode(sp,turns,kind,arm,model)
                        verify_episode(ep,model)
                        name=f'{seed}-{sp.family}-{sp.seed}-{turns}-{kind}-{arm}.json'
                        (out/'traces'/name).write_text(json.dumps(ep,sort_keys=True)+'\n')
                        episodes.append(ep)
    report={'version':VERSION,'source_sha256':source_shas(),'prior_evidence_zip_sha256':sha_file(prior_zip),
            'models':cps,'episodes':len(episodes),'decisions':sum(len(x['rows']) for x in episodes),
            'summary':summarize(episodes),'sealed_heldout_opened':False,'native_receipts':0,
            'checkpoint_training_this_experiment':False,
            'experiment':'inference-only action-selection controls with unchanged checkpoints',
        'fixture_partition':'development-confirmation' if validation else 'development-pilot',
            'wall_seconds':round(time.perf_counter()-start,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(x.relative_to(out)):sha_file(x)
            for x in out.rglob('*') if x.is_file()},indent=2,sort_keys=True)+'\n')
    return report

def verify(out):
    r=json.loads((out/'report.json').read_text())
    if r['version']!=VERSION or r['source_sha256']!=source_shas():raise ValueError('source identity')
    files={str(x.relative_to(out)):sha_file(x) for x in out.rglob('*')
           if x.is_file() and x.name!='SHA256SUMS.json'}
    if files!=json.loads((out/'SHA256SUMS.json').read_text()):raise ValueError('manifest mismatch')
    episodes=[];n=0
    models={int(k):base.ProbeRanker.load(out/'checkpoints'/f'sequential7-{k}.npz') for k in r['models']}
    for k,m in models.items():
        if sha_file(out/'checkpoints'/f'sequential7-{k}.npz')!=r['models'][str(k)]['sha256']:
            raise ValueError('checkpoint')
    for f in sorted((out/'traces').glob('*.json')):
        e=json.loads(f.read_text());seed=int(f.name.split('-')[0]);n+=verify_episode(e,models[seed]);episodes.append(e)
    if len(episodes)!=r['episodes'] or n!=r['decisions'] or summarize(episodes)!=r['summary']:
        raise ValueError('replay summary')
    return {'verified_episodes':len(episodes),'verified_decisions':n,'verified_files':len(files)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--verify',type=Path)
    p.add_argument('--prior-zip',type=Path);p.add_argument('--quick',action='store_true')
    p.add_argument('--validation',action='store_true')
    a=p.parse_args();print(json.dumps(verify(a.verify) if a.verify else run(a.out,a.prior_zip,a.quick,a.validation),sort_keys=True))

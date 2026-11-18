"""RECOVERY-DEBT-001: physical, host-legal recovery-budget counterfactuals.

No optimizer or game modification. Explicit TRAIN/DEVELOPMENT partition; structural
HELDOUT is never opened. Existing eight restored suffix checkpoints are immutable.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import statistics
import tempfile
import time
import zipfile
from . import retrace_choice_lab as prior, sequential_probe_lab as seq
from .world import SOLID, World, specs, Spec
from .train import sha_file

ARMS = ('frozen','always_move','contrastive','outcome')
SEEDS = (811,814)
SOURCE = ('recovery_debt_audit.py','retrace_choice_lab.py','sequential_probe_lab.py','world.py','policy.py','train.py')
VERSION = 'recovery-debt/1'

def restored(archive, seed):
    models, hashes = {}, {}
    with zipfile.ZipFile(archive) as z, tempfile.TemporaryDirectory() as td:
        original = json.loads(z.read('evidence/report.json'))['checkpoints'][str(seed)]
        for arm in ARMS:
            raw = z.read(f'evidence/checkpoints/{seed}-{arm}.npz')
            h = hashlib.sha256(raw).hexdigest()
            if h != original[arm]['sha256']:
                raise ValueError('checkpoint provenance mismatch')
            path = Path(td)/f'{arm}.npz'
            path.write_bytes(raw)
            models[arm] = seq.ProbeRanker.load(path)
            hashes[arm] = h
    return models, hashes

def fixture(sp, spent):
    if sp.split not in ('train','development') or sp.challenge != 'normal' or spent not in (10,11):
        raise ValueError('invalid split/budget; HELDOUT sealed')
    # Deliberate source-faithful world edits BEFORE the first legal action.
    # Exit is north from the previous corridor cell; side pocket dead-ends.
    w = World.create(replace(sp, challenge='trap_corridor'))
    for x in range(49,71):
        w.materials.pop((x,48), None); w.protected.discard((x,48))
    for x in range(49,56):
        for y in (47,49): w.materials[(x,y)] = SOLID
    w.materials[(55,48)] = SOLID; w.protected.add((55,48))
    for x in range(48,53):
        for y in (47,49):
            w.materials.pop((x,y),None); w.protected.discard((x,y))
    for pt in ((53,47),(54,49)):
        w.materials.pop(pt,None); w.protected.discard(pt)
    w.materials[(53,49)] = SOLID
    w.materials[(47,48)] = SOLID
    if spent == 11:
        w.materials.pop((47,48),None); w.protected.discard((47,48))
    prefix = []
    # Every pre-spent recovery is a real legal BACKTRACK, preceded by a
    # distinct painted move; no synthetic counter injection or history rewrite.
    if spent == 11:
        prefix.extend((w.apply('MOVE_W',12), w.apply('BACKTRACK',12)))
    for x in range(48,55):
        if x < 53:
            for action in ('MOVE_N','MOVE_S'):
                prefix.extend((w.apply(action,12),w.apply('BACKTRACK',12)))
        if x < 54: prefix.append(w.apply('MOVE_E',12))
    if w.cursor != (54,48) or w.recoveries != spent or len(prefix) != 6 + spent*2:
        raise ValueError('prefix/recovery receipt mismatch')
    menu = w.menu(12)
    if menu != ['BACKTRACK','MOVE_S','RELOCATE_START','STOP']:
        raise ValueError('legal action menu drift: '+str(menu))
    return {'world':w,'prefix':prefix,'side':'MOVE_S','goal':(53,42),
            'spec':sp.__dict__,'spent':spent,'turns':0,'kind':'budget_trap'}

def continuation(sp, spent, seed, arm, model, first, full=True):
    q=fixture(sp,spent); w=q['world']; initial=w.state_sha(); old=len(w.ink)
    if first not in ('MOVE_S','BACKTRACK') or first not in w.menu(12):
        raise ValueError('first action not host legal')
    rows=[];counts=Counter();lat=[]
    for step in range(128):
        if w.stopped: break
        menu,x=prior.inputs(q,128)
        tic=time.perf_counter_ns();probs=model.probs(x,len(menu))
        latency=(time.perf_counter_ns()-tic)/1e6
        action=first if step == 0 else menu[int(probs.argmax())]
        event=w.apply(action,12)
        counts[action]+=1;lat.append(latency)
        if full:
            rows.append({'menu':menu,'p':[round(float(v),8) for v in probs],
                         'action':action,'event':event})
    return {'seed':seed,'arm':arm,'split':sp.split,'spec':sp.__dict__,'spent':spent,
        'first':first,'initial_sha256':initial,'base_sha256':w.base_sha(),
        'prefix':q['prefix'] if full else None,'rows':rows if full else None,
        'final_sha256':w.state_sha(),'painted':len(w.ink)-old,'stopped':w.stopped,
        'decisions':sum(counts.values()),'recoveries':w.recoveries,
        'actions':dict(sorted(counts.items())),
        'latency_ms_mean':round(statistics.mean(lat),6) if lat else 0.}

def concise(e):
    return {k:v for k,v in e.items() if k not in ('rows','prefix')}

def run(archive, out):
    if out.exists(): raise FileExistsError('refuse to overwrite original evidence')
    out.mkdir(parents=True);start=time.perf_counter();cases=[];identities={};all_eps=[]
    counts={s:{a:Counter() for a in ARMS} for s in ('train','development')}
    for seed in SEEDS:
        models,identities[str(seed)]=restored(archive,seed)
        for split in ('train','development'):
            for sp in specs(split,1):
                for spent in (11,):
                    for arm,m in models.items():
                        move=continuation(sp,spent,seed,arm,m,'MOVE_S')
                        back=continuation(sp,spent,seed,arm,m,'BACKTRACK')
                        if move['initial_sha256']!=back['initial_sha256']:
                            raise ValueError('different counterfactual initial states')
                        delta=move['painted']-back['painted']
                        case={'split':split,'family':sp.family,'spent':spent,'seed':seed,'arm':arm,
                              'move_minus_backtrack':delta,'move':concise(move),'backtrack':concise(back)}
                        cases.append(case)
                        all_eps.extend((move,back))
                        counts[split][arm].update(move['actions']);counts[split][arm].update(back['actions'])
    for i,e in enumerate(all_eps):
        (out/f'{i:03d}.json').write_text(json.dumps(e,sort_keys=True,separators=(',',':'))+'\n')
    matrix={}
    for split in ('train','development'):
        matrix[split]={}
        for sp in specs(split,1):
            for spent in (11,):
                name=f'{sp.family}/spent{spent}'
                group=[c for c in cases if c['split']==split and c['family']==sp.family and c['spent']==spent]
                deltas=[c['move_minus_backtrack'] for c in group]
                matrix[split][name]={'deltas':{f"{c['seed']}/{c['arm']}":c['move_minus_backtrack'] for c in group},
                      'robust_backtrack_margin8':all(v<=-8 for v in deltas),
                      'sign_conflict':min(deltas)<0<max(deltas),
                      'backtrack_better':sum(v<0 for v in deltas)}
    summary={s:{'cases':len(matrix[s]),'comparisons':sum(c['split']==s for c in cases),
        'robust_backtrack_margin8':sum(v['robust_backtrack_margin8'] for v in matrix[s].values()),
        'sign_conflicts':sum(v['sign_conflict'] for v in matrix[s].values()),
        'by_arm':{a:{'n':len(v:=[c for c in cases if c['split']==s and c['arm']==a]),
                'backtrack_wins':sum(c['move_minus_backtrack']<0 for c in v),
                'mean_delta':round(statistics.mean(c['move_minus_backtrack'] for c in v),3),
                'action_counts':dict(sorted(counts[s][a].items())),
                'mean_latency_ms':round(statistics.mean(z['latency_ms_mean'] for c in v for z in (c['move'],c['backtrack'])),6)}
                for a in ARMS}} for s in ('train','development')}
    report={'version':VERSION,'source_sha256':{n:sha_file(Path(__file__).parent/n) for n in SOURCE},
      'archive_sha256':sha_file(archive),'checkpoints_sha256':identities,'sampling':{'families':4,
      'recovery_spent':[11],'seeds':list(SEEDS),'suffix_arms':list(ARMS),'horizon':128},
      'matrix':matrix,'summary':summary,'comparisons':len(cases),'trajectories':len(all_eps),
      'decisions':sum(e['decisions'] for e in all_eps),'elapsed_s':round(time.perf_counter()-start,3),
      'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'optimizer_updates':0,'heldout_opened':False,'native_receipts':0,'paid_compute':False}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    sums={p.name:sha_file(p) for p in out.glob('*.json') if p.name!='SHA256SUMS.json'}
    (out/'SHA256SUMS.json').write_text(json.dumps(sums,sort_keys=True,indent=2)+'\n')
    return report

def verify(archive,out):
    r=json.loads((out/'report.json').read_text());sums=json.loads((out/'SHA256SUMS.json').read_text())
    if {p.name:sha_file(p) for p in out.glob('*.json') if p.name!='SHA256SUMS.json'}!=sums:
        raise ValueError('manifest tampering')
    if r['archive_sha256']!=sha_file(archive) or r['source_sha256']!={n:sha_file(Path(__file__).parent/n) for n in SOURCE}:
        raise ValueError('provenance mismatch')
    episodes=[]
    for seed in SEEDS:
        models,ident=restored(archive,seed)
        if ident!=r['checkpoints_sha256'][str(seed)]:raise ValueError('checkpoint drift')
        for p in sorted(out.glob('???.json')):
            e=json.loads(p.read_text())
            if e['seed']!=seed:continue
            q=fixture(Spec(**e['spec']),e['spent']);w=q['world']
            if e['initial_sha256']!=w.state_sha() or e['prefix']!=q['prefix'] or e['base_sha256']!=w.base_sha():
                raise ValueError('prefix/state mismatch')
            old=len(w.ink)
            for idx,row in enumerate(e['rows']):
                menu,x=prior.inputs(q,128); pp=models[e['arm']].probs(x,len(menu))
                dist=[round(float(v),8) for v in pp]
                if row['menu']!=menu or row['p']!=dist or row['action'] not in menu:
                    raise ValueError('probability or action mismatch')
                if (idx==0 and row['action']!=e['first']) or (idx>0 and row['action']!=menu[int(pp.argmax())]):
                    raise ValueError('suffix policy drift')
                if row['event']!=w.apply(row['action'],12):raise ValueError('receipt mismatch')
            if e['final_sha256']!=w.state_sha() or e['painted']!=len(w.ink)-old or e['recoveries']!=w.recoveries or e['stopped']!=w.stopped or e['decisions']!=len(e['rows']) or e['actions']!=dict(sorted(Counter(row['action'] for row in e['rows']).items())):
                raise ValueError('terminal state mismatch')
            episodes.append(e)
    if len(episodes)!=r['trajectories'] or sum(e['decisions'] for e in episodes)!=r['decisions']:
        raise ValueError('report mismatch')
    return {'verified_trajectories':len(episodes),'decisions':sum(e['decisions'] for e in episodes),
            'checkpoint_identities':8,'manifest_files':len(sums)}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint-zip',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    print(json.dumps(verify(a.checkpoint_zip,a.out) if a.verify else run(a.checkpoint_zip,a.out),sort_keys=True))

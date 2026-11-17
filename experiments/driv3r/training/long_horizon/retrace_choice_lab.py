"""RETRACE-CHOICE-001: train matched recover-vs-progress forks using host-legal menus.

Research synthetic World only. Source fixture and historical d_common remain unchanged;
no native ABI or sealed structural heldout claims. TRAIN supervision, DEVELOPMENT replay.
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
import numpy as np
from . import sequential_probe_lab as seq, contrastive_fork_lab as rot
from . import goal_recovery_lab as host, policy
from .world import MOVE, Spec, specs
from .train import sha_file

VERSION='retrace-choice/1'
ARMS=('frozen','always_move','always_retrace','contrastive')
KINDS=('blocked_spur','open_spur')
STAGES=(36,72,128)
SOURCE=('retrace_choice_lab.py','sequential_probe_lab.py','sensing_probe_lab.py',
        'menu_alias_lab.py','contrastive_fork_lab.py','recovery_fork_lab.py',
        'goal_recovery_lab.py','goal_conditioning_lab.py','goal_retarget_lab.py',
        'world.py','policy.py','train.py')

def fork(sp,turns,kind):
    if sp.split not in ('train','development') or sp.challenge!='normal' or turns not in rot.ROTATION or kind not in KINDS:
        raise ValueError('sealed heldout or invalid fork')
    q=seq.fixture(sp,turns,'trap');w=q['world']
    # Static fixture specialization before neural decisions. An unblocked south
    # entry is on the original legal menu in both cases; only *beyond* it differs.
    def p(x,y):return rot.rotate_xy((x,y),turns)
    w.materials.pop(p(54,49),None);w.protected.discard(p(54,49))
    for y in range(50,64):
        if kind=='open_spur':
            w.materials.pop(p(54,y),None);w.protected.discard(p(54,y))
    if kind=='blocked_spur':
        w.materials[p(54,50)]=1;w.protected.add(p(54,50))
    # Advance five genuine legal steps along existing prefix to the fork.
    prefix=list(q['prefix'])
    for _ in range(5):
        ev=w.apply(q['forward'],12);prefix.append(ev)
    if w.cursor!=p(54,48) or w.recoveries!=0:raise ValueError('fork prefix')
    menu=w.menu(12)
    side=rot.rotate_action('MOVE_S',turns)
    if side not in menu or 'BACKTRACK' not in menu or 'RELOCATE_START' not in menu:
        raise ValueError('matched legal menu missing')
    if w.occupied(p(54,49)):raise ValueError('side is not host legal')
    return {'world':w,'spec':sp.__dict__,'turns':turns,'kind':kind,
            'prefix':prefix,'side':side,'goal':p(54,59)}

def paired(sp,turns):
    a=fork(sp,turns,'blocked_spur');b=fork(sp,turns,'open_spur')
    if a['world'].menu(12)!=b['world'].menu(12):raise ValueError('menu leak')
    # Same cursor, heading, action menu and recovery state; different reachable
    # second step of the offered side MOVE, known to the synthetic feature sensor.
    for q in (a,b):
        if not q['world'].occupied(rot.rotate_xy((54,48),turns)):
            raise ValueError('prefix ink missing')
    return a,b

def inputs(q,horizon):
    w=q['world'];state=seq.fresh_state()
    obs={'hit':0.,'free_ratio':0.,'spent':0,'reads':0,'retreating':False}
    return seq.features(w,q['goal'],horizon,q['side'],'BACKTRACK',state,obs)

def teacher(q,arm):
    if arm=='always_move':return q['side']
    if arm=='always_retrace':return 'BACKTRACK'
    if arm=='contrastive':return 'BACKTRACK' if q['kind']=='blocked_spur' else q['side']
    raise ValueError('not a training arm')

def examples(seed,stage,natural=24,repeats=8):
    if stage not in STAGES or not 4<=natural<=128 or not 1<=repeats<=24:
        raise ValueError('training budget')
    natural_x,masks,labels,meta=host.corpus(seed,stage)
    rng=np.random.default_rng(seed*101+stage)
    ii=rng.choice(len(labels),size=min(natural,len(labels)),replace=False)
    n_x=np.concatenate([natural_x[ii],np.zeros((len(ii),policy.ACTION_MAX,seq.EXTRA))],axis=2)
    X={a:list(n_x) for a in ARMS[1:]};M={a:list(masks[ii]) for a in X};Y={a:list(labels[ii]) for a in X}
    ledger=[]
    for sp in specs('train',1):
        for turns in rot.ROTATION:
            for q in paired(sp,turns):
                menu,x=inputs(q,stage)
                mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                for arm in X:
                    label=teacher(q,arm)
                    if label not in menu:raise ValueError('non-legal TRAIN label')
                    for _ in range(repeats):
                        X[arm].append(x.copy());M[arm].append(mask.copy());Y[arm].append(menu.index(label))
                    ledger.append({'split':'train','seed':seed,'stage':stage,'spec':sp.__dict__,
                                   'turns':turns,'kind':q['kind'],'arm':arm,
                                   'menu':menu,'label':label,'initial_sha':q['world'].state_sha(),
                                   'world_sha':q['world'].base_sha(),
                                   'features_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    return ({a:np.stack(X[a]) for a in X},{a:np.stack(M[a]) for a in X},
            {a:np.array(Y[a],np.int64) for a in X},ledger,
            {'natural':len(ii),'targeted_pairs':len(ledger)//len(X),
             'targeted_repetitions':repeats,'natural_split':meta['split']})

def verify_train(ledger):
    for row in ledger:
        sp=Spec(**row['spec'])
        if sp.split!='train' or row['split']!='train' or row['stage'] not in STAGES:raise ValueError('TRAIN leak')
        q=fork(sp,row['turns'],row['kind']);menu,x=inputs(q,row['stage'])
        if (row['menu']!=menu or row['label']!=teacher(q,row['arm']) or
            row['initial_sha']!=q['world'].state_sha() or row['world_sha']!=q['world'].base_sha() or
            row['features_sha256']!=hashlib.sha256(x.tobytes()).hexdigest()):
            raise ValueError('TRAIN label/state tampering')
    return len(ledger)

def adapt(seed,pretrained_zip,updates=55):
    if not 1<=updates<=120:raise ValueError('optimizer budget')
    with zipfile.ZipFile(pretrained_zip) as z:
        cp=z.read(f'final-{seed}/sequential7.npz')
        record=json.loads(z.read(f'final-{seed}/report.json'))
        if hashlib.sha256(cp).hexdigest()!=record['checkpoints']['sequential7']['sha256']:
            raise ValueError('pretrained provenance mismatch')
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'frozen.npz';p.write_bytes(cp)
        frozen=seq.ProbeRanker.load(p)
    models={'frozen':frozen}
    for arm in ARMS[1:]:models[arm]=copy.deepcopy(frozen)
    history=[];ledger=[]
    for stage in STAGES:
        X,M,Y,rows,meta=examples(seed,stage)
        for arm in X:
            history.append({'stage':stage,'arm':arm,'optimizer':models[arm].fit(X[arm],M[arm],Y[arm],updates,seed*71+stage),'corpus':meta})
        ledger.extend(rows)
    return models,history,ledger,hashlib.sha256(cp).hexdigest()

def decide(model,q,horizon):
    menu,x=inputs(q,horizon)
    pp=model.probs(x,len(menu))
    return menu[int(pp.argmax())],{a:round(float(pp[i]),8) for i,a in enumerate(menu)}

def episode(sp,turns,kind,arm,model,horizon=128):
    if sp.split!='development' or horizon not in (36,72,128) or arm not in ARMS:
        raise ValueError('DEVELOPMENT only')
    q=fork(sp,turns,kind);w=q['world'];initial=w.state_sha();before=len(w.ink)
    rows=[];lat=[];first_prob={};first=None
    for step in range(horizon):
        if w.stopped:break
        tic=time.perf_counter_ns()
        menu,x=inputs(q,horizon)
        probabilities=model.probs(x,len(menu));action=menu[int(probabilities.argmax())]
        lat.append((time.perf_counter_ns()-tic)/1e6)
        prob={a:round(float(probabilities[i]),8) for i,a in enumerate(menu)}
        if first is None:first=action;first_prob=prob
        event=w.apply(action,12)
        rows.append({'action':action,'probabilities':prob,'menu':menu,'event':event})
    return {'version':VERSION,'split':'development','spec':sp.__dict__,'turns':turns,
            'kind':kind,'arm':arm,'horizon':horizon,'prefix':q['prefix'],
            'initial_sha256':initial,'base_sha256':w.base_sha(),
            'first_action':first,'first_probs':first_prob,'rows':rows,
            'painted':len(w.ink)-before,'recovery':w.recoveries,
            'early_stop':int(len(rows)<horizon),'final_sha256':w.state_sha(),
            'mean_latency_ms':round(statistics.mean(lat),6) if lat else 0.}

def verify_episode(e,model):
    if e['version']!=VERSION or e['split']!='development':raise ValueError('identity')
    sp=Spec(**e['spec']);q=fork(sp,e['turns'],e['kind']);w=q['world']
    if (e['initial_sha256']!=w.state_sha() or e['prefix']!=q['prefix'] or
        e['base_sha256']!=w.base_sha()):raise ValueError('fixture replay')
    old=len(w.ink)
    for i,row in enumerate(e['rows']):
        if w.stopped:raise ValueError('post STOP')
        m,x=inputs(q,e['horizon']);p=model.probs(x,len(m));a=m[int(p.argmax())]
        dist={z:round(float(p[j]),8) for j,z in enumerate(m)}
        if (row['menu']!=m or row['action']!=a or row['probabilities']!=dist or
            (i==0 and (e['first_action']!=a or e['first_probs']!=dist))):
            raise ValueError('prediction corruption')
        if row['event']!=w.apply(a,12):raise ValueError('host event corruption')
    if (len(e['rows'])>e['horizon'] or (len(e['rows'])<e['horizon'] and not w.stopped) or
        e['painted']!=len(w.ink)-old or e['recovery']!=w.recoveries or
        e['final_sha256']!=w.state_sha() or e['early_stop']!=int(len(e['rows'])<e['horizon'])):
        raise ValueError('terminal mismatch')
    return len(e['rows'])

def summarize(eps):
    report={}
    for kind in KINDS:
        for arm in ARMS:
            es=[e for e in eps if e['kind']==kind and e['arm']==arm]
            if not es:continue
            correct=('BACKTRACK' if kind=='blocked_spur' else None)
            actions=Counter(r['action'] for e in es for r in e['rows'])
            report[kind+'/'+arm]={'episodes':len(es),
                'correct_first':sum((e['first_action']=='BACKTRACK' if correct else e['first_action']==rot.rotate_action('MOVE_S',e['turns'])) for e in es),
                'mean_painted':round(statistics.mean(e['painted'] for e in es),3),
                'early_stop':sum(e['early_stop'] for e in es),
                'mean_recovery':round(statistics.mean(e['recovery'] for e in es),3),
                'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in es),6),
                'actions':dict(sorted(actions.items()))}
    return report

def shas():return {n:sha_file(Path(__file__).parent/n) for n in SOURCE}

def run(out,prior_zip,seeds=(811,814),quick=False,updates=55):
    if out.exists():raise FileExistsError('immutable output exists')
    if any(s not in (811,814) for s in seeds):raise ValueError('bounded seed set')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();(out/'traces').mkdir()
    start=time.perf_counter();episodes=[];ledger=[];models_by_seed={};history={};cp_info={}
    for seed in seeds:
        models,hist,rows,sha=adapt(seed,prior_zip,updates);models_by_seed[seed]=models
        history[str(seed)]=hist;ledger.extend(rows)
        cp_info[str(seed)]={'source_sha256':sha}
        for arm,m in models.items():
            p=out/'checkpoints'/f'{seed}-{arm}.npz';m.save(p)
            mm=seq.ProbeRanker.load(p)
            if any(not np.array_equal(m.p[k],mm.p[k]) for k in m.p):raise ValueError('checkpoint restoration')
            cp_info[str(seed)][arm]={'sha256':sha_file(p),'updates':m.updates,'parameters':sum(v.size for v in m.p.values())}
        for sp in specs('development',1)[:1 if quick else 2]:
            for turns in ((0,) if quick else (0,1)):
                paired(sp,turns)
                for kind in KINDS:
                    for arm in ARMS:
                        ep=episode(sp,turns,kind,arm,models[arm],36 if quick else 128)
                        verify_episode(ep,models[arm]);episodes.append((seed,ep))
                        name=f'{seed}-{sp.family}-{sp.seed}-{turns}-{kind}-{arm}.json'
                        (out/'traces'/name).write_text(json.dumps(ep,sort_keys=True)+'\n')
    (out/'training.json').write_text(json.dumps(ledger,sort_keys=True)+'\n')
    result={'version':VERSION,'source_sha256':shas(),'original_zip_sha256':sha_file(prior_zip),
            'checkpoint_info':cp_info,'train_history':history,'train_rows':len(ledger),
            'summary':summarize([ep for _,ep in episodes]),
            'episodes':len(episodes),'decisions':sum(len(ep['rows']) for _,ep in episodes),
            'elapsed_seconds':round(time.perf_counter()-start,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'split':'development','sealed_heldout_opened':False,'native_receipts':0,'paid_compute':False}
    (out/'report.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file()},sort_keys=True,indent=2)+'\n')
    return result

def verify(out):
    report=json.loads((out/'report.json').read_text())
    if report['version']!=VERSION or report['source_sha256']!=shas():raise ValueError('source identity')
    files={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'}
    if files!=json.loads((out/'SHA256SUMS.json').read_text()):raise ValueError('manifest mismatch')
    verify_train(json.loads((out/'training.json').read_text()))
    eps=[];n=0;models={}
    for seed,info in report['checkpoint_info'].items():
        models[int(seed)]={}
        for arm in ARMS:
            p=out/'checkpoints'/f'{seed}-{arm}.npz'
            if sha_file(p)!=info[arm]['sha256']:raise ValueError('checkpoint hash')
            models[int(seed)][arm]=seq.ProbeRanker.load(p)
    for f in sorted((out/'traces').glob('*.json')):
        seed=int(f.name.split('-')[0]);e=json.loads(f.read_text())
        n+=verify_episode(e,models[seed][e['arm']]);eps.append(e)
    summaries=summarize(eps)
    for key in summaries:
        summaries[key].pop('mean_latency_ms');report['summary'][key].pop('mean_latency_ms')
    if summaries!=report['summary'] or len(eps)!=report['episodes'] or n!=report['decisions']:
        raise ValueError('aggregate disagreement')
    return {'checkpoints':sum(len(v) for v in models.values()),'episodes':len(eps),'decisions':n,'files':len(files),'train_rows':len(json.loads((out/'training.json').read_text()))}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);ap.add_argument('--prior-zip',type=Path);ap.add_argument('--verify',type=Path)
    ap.add_argument('--quick',action='store_true');ap.add_argument('--updates',type=int,default=55)
    ns=ap.parse_args();print(json.dumps(verify(ns.verify) if ns.verify else run(ns.out,ns.prior_zip,(811,) if ns.quick else (811,814),ns.quick,ns.updates),sort_keys=True))

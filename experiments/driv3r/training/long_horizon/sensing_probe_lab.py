"""SENSE-RANGE-001: equal-menu hidden traps and bounded local occupancy sensing.

External synthetic research; sensor is NOT an approved native CyberSand ABI.
TRAIN-only fitting, DEVELOPMENT-only evaluation; sealed heldout never opened.
Never alter host menus or grant probing/teleport actions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np

from . import menu_alias_lab as alias, contrastive_fork_lab as rotated
from . import goal_recovery_lab as host, policy
from .world import MOVE, Spec, specs
from .train import sha_file, _png

VERSION = 'sense-range/1'
STAGES = (36, 72, 128)
ARMS = ('frozen', 'blind', 'range3', 'range7')
ADAPT = ARMS[1:]
RANGES = {'blind':0, 'range3':3, 'range7':7}
KINDS = alias.KINDS
EVAL_SEED = 791
EXTRA = 2
WIDTH = host.WIDTH + EXTRA
SOURCE = ('sensing_probe_lab.py', 'menu_alias_lab.py', 'contrastive_fork_lab.py',
          'recovery_fork_lab.py', 'goal_recovery_lab.py','goal_conditioning_lab.py',
          'goal_retarget_lab.py','world.py','policy.py','train.py')


def ray(w, action, radius):
    """Only occupancy within radius from CURRENT cell; no distant oracle.

    Radius is a research parameter, not a claim that CyberSand exposes this
    sightline. Count includes the action's first cell; all actual movements
    remain subject to w.menu(12), regardless of sensor output.
    """
    if radius not in (0,3,7): raise ValueError('radius not bounded')
    if action not in MOVE or radius == 0: return (0.,0.)
    dx,dy = MOVE[action]; cx,cy = w.cursor
    free = 0
    for step in range(1,radius+1):
        if w.occupied((cx+dx*step,cy+dy*step)):break
        free += 1
    return (float(free<radius),float(free)/radius)


def matrix(w, goal, horizon, radius):
    menu, base = host.matrix(w,goal,horizon,False)
    x = np.zeros((policy.ACTION_MAX,WIDTH),np.float64)
    x[:,:host.WIDTH] = base
    for i,a in enumerate(menu):x[i,-EXTRA:] = ray(w,a,radius)
    return menu,x


class SensorRanker(host.Ranker):
    def __init__(self, seed, hidden=24):
        super().__init__(seed,hidden)
        extra=np.random.default_rng(seed+8191).normal(0,.09,(EXTRA,hidden))
        self.p['w1'] = np.concatenate((self.p['w1'],extra),axis=0)
        self.m['w1'] = np.zeros_like(self.p['w1'])
        self.v['w1'] = np.zeros_like(self.p['w1'])

    @classmethod
    def from_base(cls, base):
        m=cls(base.seed,base.hidden)
        for k in ('b1','w2','b2'):m.p[k]=base.p[k].copy()
        m.p['w1'][:host.WIDTH]=base.p['w1'].copy()
        return m

    def choose_sensed(self,w,goal,horizon,radius):
        menu,x=matrix(w,goal,horizon,radius)
        p=self.probs(x,len(menu))
        return menu[int(p.argmax())],{a:round(float(p[i]),8) for i,a in enumerate(menu)}


def predict(model,arm,w,goal,horizon):
    return model.choose(w,goal,horizon,False) if arm=='frozen' else model.choose_sensed(w,goal,horizon,RANGES[arm])


def train_cases(seed,stage, natural_limit=64, variants=1):
    if stage not in STAGES or not 16<=natural_limit<=128 or variants not in (1,2):
        raise ValueError('unbounded training fixture')
    raw,masks,labels,prior=host.corpus(seed,stage)
    rng=np.random.default_rng(seed*31+stage)
    ind=np.sort(rng.choice(len(labels),size=min(natural_limit,len(labels)),replace=False))
    zeros=np.zeros((len(ind),policy.ACTION_MAX,EXTRA))
    base=np.concatenate((raw[ind],zeros),axis=2)
    X={arm:list(base) for arm in ADAPT}; M=list(masks[ind]); Y=list(labels[ind]); ledger=[]
    for sp in specs('train',1):
        for variant in range(variants):
            for turns in rotated.ROTATION:
                cases,feature=alias.matched(sp,seed,variant,turns,stage)
                for kind in KINDS:
                    q=cases[kind]; w=q['world']; menu=w.menu(12)
                    choice=q['forward'] if kind=='open' else q['retreat']
                    if choice not in menu:raise ValueError('illegal teacher decision')
                    mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                    M.append(mask);Y.append(menu.index(choice))
                    for arm in ADAPT:
                        legal,x=matrix(w,q['goal'],stage,RANGES[arm])
                        if legal!=menu:raise ValueError('menu changed during sensing')
                        X[arm].append(x)
                    ledger.append({'split':'train','spec':q['spec'],'seed':seed,'variant':variant,
                                   'turns':turns,'kind':kind,'stage':stage,
                                   'state_sha256':w.state_sha(),'base_sha256':w.base_sha(),
                                   'prefix':q['prefix'],'menu':menu,'goal':list(q['goal']),
                                   'label':choice,'witness':q['witness'],
                                   'rays':{a:list(ray(w,q['forward'],RANGES[a])) for a in ADAPT}})
    return ({a:np.stack(x) for a,x in X.items()},np.stack(M),np.asarray(Y,np.int64),ledger,
            {'natural':len(ind),'matched_supplemental':len(ledger),'natural_sensor_missing_zero_padded':True,
             'source':prior,'train_only':True})


def verify_training_row(row):
    if row['split']!='train' or row['kind'] not in KINDS or row['stage'] not in STAGES:
        raise ValueError('training split/kind')
    s0=Spec(**row['spec']);sp=Spec(s0.split,s0.family,s0.seed,s0.mode,'normal');q=alias.construct(sp,row['seed'],row['variant'],row['turns'],row['kind'])
    w=q['world']; label=q['forward'] if row['kind']=='open' else q['retreat']
    rays={a:list(ray(w,q['forward'],RANGES[a])) for a in ADAPT}
    if (row['state_sha256']!=w.state_sha() or row['base_sha256']!=w.base_sha()
        or row['prefix']!=q['prefix'] or row['goal']!=list(q['goal'])
        or row['menu']!=w.menu(12) or row['label']!=label
        or row['witness']!=q['witness'] or row['rays']!=rays):
        raise ValueError('training source replay mismatch')
    return True


def train(seed, base_updates=18, adapt_updates=48, natural_limit=64, variants=1):
    if not 1<=base_updates<=60 or not 1<=adapt_updates<=100:
        raise ValueError('optimizer out of bounds')
    base,pre=host.train(seed,base_updates)
    frozen=base['goal_recovery']; models={'frozen':copy.deepcopy(frozen)}
    models.update({a:SensorRanker.from_base(frozen) for a in ADAPT})
    history=[];ledger=[]
    for stage in STAGES:
        X,M,Y,rows,meta=train_cases(seed,stage,natural_limit,variants)
        hist={a:models[a].fit(X[a],M,Y,adapt_updates,seed*811+stage) for a in ADAPT}
        history.append({'stage':stage,'adaptation':hist,'corpus':meta});ledger.extend(rows)
    return models,{'baseline':pre,'adaptation':history},ledger


def episode(sp,seed,turns,kind,arm,model,horizon=128,png=None):
    if sp.split!='development' or arm not in ARMS or kind not in KINDS or horizon not in (72,128):
        raise ValueError('development bounds')
    q=alias.construct(sp,seed,0,turns,kind);w=q['world'];before=len(w.ink)
    start_state=w.state_sha();rows=[];lat=[]
    for t in range(horizon):
        if w.stopped:break
        clock=time.perf_counter_ns();a,p=predict(model,arm,w,q['goal'],horizon)
        lat.append((time.perf_counter_ns()-clock)/1e6)
        rays={r:list(ray(w,q['forward'],r)) for r in (3,7)} if t==0 else None
        rows.append({'requested':a,'probabilities':p,'event':w.apply(a,12),
                     **({'initial_forward_rays':rays} if rays is not None else {})})
    if png is not None:_png(png,w)
    return {'version':VERSION,'split':'development','spec':q['spec'],'seed':seed,
            'turns':turns,'kind':kind,'arm':arm,'horizon':horizon,'goal':list(q['goal']),
            'first_menu':q['world'].menu(12) if not rows else sorted((q['forward'],q['retreat'],'STOP')),
            'forward':q['forward'],'retreat':q['retreat'],'prefix':q['prefix'],
            'initial_sha256':start_state,'base_sha256':q['world'].base_sha(),
            'witness':q['witness'],'first_action':rows[0]['requested'],
            'rows':rows,'new_paint':len(w.ink)-before,'final_sha256':w.state_sha(),
            'metrics':w.metrics(horizon+len(q['prefix'])),
            'latency_ms':round(statistics.mean(lat),6)}


def verify_episode(ep,model):
    if ep['version']!=VERSION or ep['split']!='development' or ep['arm'] not in ARMS:
        raise ValueError('episode version/split')
    s0=Spec(**ep['spec']);sp=Spec(s0.split,s0.family,s0.seed,s0.mode,'normal');q=alias.construct(sp,ep['seed'],0,ep['turns'],ep['kind'])
    w=q['world'];before=len(w.ink)
    if (ep['goal']!=list(q['goal']) or ep['first_menu']!=w.menu(12)
        or ep['prefix']!=q['prefix'] or ep['initial_sha256']!=w.state_sha()
        or ep['base_sha256']!=w.base_sha() or ep['witness']!=q['witness']
        or ep['forward']!=q['forward'] or ep['retreat']!=q['retreat']):
        raise ValueError('initial state provenance mismatch')
    for i,row in enumerate(ep['rows']):
        if w.stopped:raise ValueError('post-stop move')
        a,p=predict(model,ep['arm'],w,q['goal'],ep['horizon'])
        if a!=row['requested'] or p!=row['probabilities']:
            raise ValueError('checkpoint action or distribution mismatch')
        if i==0:
            rays={r:list(ray(w,q['forward'],r)) for r in (3,7)}
            if row.get('initial_forward_rays')!={str(k):v for k,v in rays.items()} and row.get('initial_forward_rays')!=rays:
                raise ValueError('range sensor evidence mismatch')
        elif 'initial_forward_rays' in row:raise ValueError('unexpected sensor evidence')
        if row['event']!=w.apply(a,12):raise ValueError('host-legal state mismatch')
    if (len(ep['rows'])>ep['horizon'] or ep['first_action']!=ep['rows'][0]['requested']
        or ep['new_paint']!=len(w.ink)-before or ep['final_sha256']!=w.state_sha()
        or ep['metrics']!=w.metrics(ep['horizon']+len(q['prefix']))
        or (len(ep['rows'])<ep['horizon'] and not w.stopped)):
        raise ValueError('episode end mismatch')
    return len(ep['rows'])


def summarize(eps):
    out={}
    for kind in KINDS:
        for arm in ARMS:
            es=[e for e in eps if e['kind']==kind and e['arm']==arm]
            if not es:continue
            acts=Counter(row['event']['chosen'] for e in es for row in e['rows'])
            right=lambda e:e['forward'] if kind=='open' else e['retreat']
            out[kind+'/'+arm]={'n':len(es),'correct_first':sum(e['first_action']==right(e) for e in es),
                'early_stop':sum(len(e['rows'])<e['horizon'] for e in es),
                'mean_new_paint':round(statistics.mean(e['new_paint'] for e in es),3),
                'mean_turns':round(statistics.mean(len(e['rows']) for e in es),3),
                'mean_latency_ms':round(statistics.mean(e['latency_ms'] for e in es),6),
                'action_distribution':dict(sorted(acts.items()))}
    return out


def source_sha():return {f:sha_file(Path(__file__).parent/f) for f in SOURCE}


def run(out,seed=711,base_updates=18,adapt_updates=48,natural_limit=64,variants=1,quick=False):
    if out.exists():raise FileExistsError('immutable evidence already exists')
    out.mkdir(parents=True);start=time.perf_counter();gc.collect();enabled=gc.isenabled()
    try:
        if enabled:gc.disable()
        models,hist,ledger=train(seed,base_updates,adapt_updates,natural_limit,variants)
        checkpoints={}
        for arm,m in models.items():
            f=out/(arm+'.npz');m.save(f)
            restored=(host.Ranker if arm=='frozen' else SensorRanker).load(f)
            if any(not np.array_equal(m.p[k],restored.p[k]) for k in m.p):raise ValueError('checkpoint restore')
            checkpoints[arm]={'sha256':sha_file(f),'parameters':sum(x.size for x in m.p.values()),
                              'updates':m.updates}
        models={a:(host.Ranker if a=='frozen' else SensorRanker).load(out/(a+'.npz')) for a in ARMS}
        (out/'traces').mkdir();(out/'png').mkdir();eps=[];alias_rows=[]
        for sp in specs('development',1)[:1 if quick else 4]:
            for turns in ((0,) if quick else (0,1)):
                cases,feat=alias.matched(sp,EVAL_SEED,0,turns)
                if not np.array_equal(feat['far_trap'],feat['open']):raise ValueError('alias changed')
                far=cases['far_trap'];open_=cases['open']
                # Radius-3 remains aliased; radius-7 must resolve it.
                r3=ray(far['world'],far['forward'],3); r3o=ray(open_['world'],open_['forward'],3)
                r7=ray(far['world'],far['forward'],7); r7o=ray(open_['world'],open_['forward'],7)
                if r3!=r3o or r7==r7o:raise ValueError('sensor intervention is not isolating range')
                alias_rows.append({'family':sp.family,'turns':turns,'far':far['world'].state_sha(),
                                   'open':open_['world'].state_sha(),
                                   'r3_same':True,'r7_different':True,'rays':[list(r3),list(r7),list(r7o)]})
                for kind in KINDS:
                    for arm in ARMS:
                        name=f'{sp.family}-{sp.seed}-{turns}-{kind}-{arm}'
                        png=out/'png'/(name+'.png') if not quick and turns==0 and arm in ('blind','range7') else None
                        ep=episode(sp,EVAL_SEED,turns,kind,arm,models[arm],72 if quick else 128,png)
                        verify_episode(ep,models[arm]);eps.append(ep)
                        (out/'traces'/(name+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
    finally:
        if enabled:gc.enable()
    for row in ledger:verify_training_row(row)
    (out/'training.json').write_text(json.dumps(ledger,sort_keys=True)+'\n')
    (out/'aliases.json').write_text(json.dumps(alias_rows,sort_keys=True)+'\n')
    rep={'version':VERSION,'seed':seed,'curriculum':STAGES,'base_updates':base_updates,
         'adapt_updates':adapt_updates,'natural_limit':natural_limit,'variants':variants,
         'history':hist,'checkpoints':checkpoints,'summary':summarize(eps),
         'episodes':len(eps),'decisions':sum(len(e['rows']) for e in eps),
         'training_cases':len(ledger),'aliases':len(alias_rows),'source_sha256':source_sha(),
         'split':'development','sensor_is_non_native_research_proposal':True,
         'sealed_heldout_opened':False,'native_receipts':0,'paid_compute':False,
         'elapsed_s':round(time.perf_counter()-start,3),
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f)
      for f in out.rglob('*') if f.is_file()},indent=2,sort_keys=True)+'\n')
    return rep


def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=source_sha():
        raise ValueError('source identity mismatch')
    actual={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*') if f.is_file() and f.name!='SHA256SUMS.json'}
    if actual!=json.loads((out/'SHA256SUMS.json').read_text()):
        raise ValueError('manifest mismatch')
    models={a:(host.Ranker if a=='frozen' else SensorRanker).load(out/(a+'.npz')) for a in ARMS}
    for a,m in models.items():
        if sha_file(out/(a+'.npz'))!=rep['checkpoints'][a]['sha256'] or m.updates!=rep['checkpoints'][a]['updates']:
            raise ValueError('checkpoint identity mismatch')
    ledger=json.loads((out/'training.json').read_text())
    for r in ledger:verify_training_row(r)
    eps=[]
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text());verify_episode(ep,models[ep['arm']]);eps.append(ep)
    aliases=json.loads((out/'aliases.json').read_text())
    for r in aliases:
        sp=next(s for s in specs('development',1) if s.family==r['family'])
        cases,_=alias.matched(sp,EVAL_SEED,0,r['turns'])
        far=cases['far_trap'];open_=cases['open']
        if (r['far']!=far['world'].state_sha() or r['open']!=open_['world'].state_sha()
            or r['rays']!=[list(ray(far['world'],far['forward'],3)),list(ray(far['world'],far['forward'],7)),
                           list(ray(open_['world'],open_['forward'],7))]):
            raise ValueError('alias witness mismatch')
    summed=summarize(eps)
    for k in summed:
        summed[k].pop('mean_latency_ms');rep['summary'][k].pop('mean_latency_ms')
    if (summed!=rep['summary'] or len(eps)!=rep['episodes'] or len(ledger)!=rep['training_cases']
        or len(aliases)!=rep['aliases'] or sum(len(e['rows']) for e in eps)!=rep['decisions']):
        raise ValueError('aggregate replay mismatch')
    return {'checkpoints':len(models),'training_cases':len(ledger),'episodes':len(eps),
            'decisions':sum(len(e['rows']) for e in eps),'aliases':len(aliases),'hashes':len(actual)}


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--out',type=Path);cli.add_argument('--verify',type=Path)
    cli.add_argument('--seed',type=int,default=711);cli.add_argument('--base-updates',type=int,default=18)
    cli.add_argument('--adapt-updates',type=int,default=48);cli.add_argument('--natural-limit',type=int,default=64)
    cli.add_argument('--variants',type=int,default=1);cli.add_argument('--quick',action='store_true')
    a=cli.parse_args();print(json.dumps(verify(a.verify) if a.verify else
        run(a.out,a.seed,a.base_updates,a.adapt_updates,a.natural_limit,a.variants,a.quick),sort_keys=True))

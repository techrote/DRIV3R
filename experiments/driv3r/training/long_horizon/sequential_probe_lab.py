"""SEQUENTIAL-PROBE-001: equal 7-cell observation budget, static vs motion probe.

A research-only CyberSand-style synthetic World experiment. No native sensing,
new host actions, paid compute, sealed HELDOUT, or mutation of canonical ABI.
TRAIN-only supervised optimization; separate DEVELOPMENT replay evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from dataclasses import replace
import gc
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np

from . import sensing_probe_lab as sense, goal_recovery_lab as host, policy
from . import contrastive_fork_lab as rotate
from .world import World, Spec, SOLID, MOVE, specs
from .train import sha_file, _png

VERSION = 'sequential-probe/1'
STAGES = (36,72,128)
ARMS = ('blind','static7','sequential7')
EXTRA = 3
WIDTH = host.WIDTH + EXTRA
SPLITS = ('train','development')
KINDS = ('trap','open')
SEED_EVAL = 803
# 1 initial forward read, 3 after first actual move, 3 after third move.
# The probe CANNOT see the obstacle until it has paid three real MOVE actions.
PROBE_PLAN = {0:1,1:3,3:3}
SOURCE = ('sequential_probe_lab.py','sensing_probe_lab.py','menu_alias_lab.py',
          'contrastive_fork_lab.py','recovery_fork_lab.py','goal_recovery_lab.py',
          'goal_conditioning_lab.py','goal_retarget_lab.py','world.py','policy.py','train.py')


def fixture(sp,turns,kind):
    """Build a new recoverable-risk corridor; apply only legal actions after setup.

    Unlike prior exhausted-recovery fork, this leaves enough BACKTRACK budget
    to reverse the probe. Static material edits are the *world design*, not
    an unapproved runtime mutation. Open/trap initial menu and base neural
    features are equal; only obstacle x55 differs six cells ahead.
    """
    if sp.split not in SPLITS or sp.challenge!='normal' or turns not in rotate.ROTATION or kind not in KINDS:
        raise ValueError('invalid split/fixture (HELDOUT is sealed)')
    w=World.create(replace(sp,challenge='trap_corridor'))
    for x in range(49,71):
        w.materials.pop((x,48),None);w.protected.discard((x,48))
    for x in range(49,56):
        for y in (47,49):w.materials[(x,y)]=SOLID
    if kind=='trap':
        w.materials[(55,48)]=SOLID
        w.protected.add((55,48))
    # source terrain rotate BEFORE any movement, not a mid-simulation edit
    w.materials={rotate.rotate_xy(p,turns):v for p,v in w.materials.items()}
    w.protected={rotate.rotate_xy(p,turns) for p in w.protected}
    forward=rotate.rotate_action('MOVE_E',turns)
    retreat=rotate.rotate_action('MOVE_W',turns)
    first=w.apply(forward,12)
    if w.recoveries!=0 or w.cursor!=rotate.rotate_xy((49,48),turns):
        raise ValueError('non-legal synthetic prefix')
    if not {forward,retreat,'BACKTRACK','STOP'}.issubset(w.menu(12)):
        raise ValueError('incompatible legal menu')
    # The initial goal is deliberately common to both pairs and points east.
    goal=rotate.rotate_xy((65,48),turns)
    return {'world':w,'spec':sp.__dict__,'kind':kind,'turns':turns,
            'goal':goal,'forward':forward,'retreat':retreat,'prefix':[first],
            'obstacle':list(rotate.rotate_xy((55,48),turns))}


def matched(sp,turns):
    a=fixture(sp,turns,'trap');b=fixture(sp,turns,'open')
    ma,xa=host.matrix(a['world'],a['goal'],128,False)
    mb,xb=host.matrix(b['world'],b['goal'],128,False)
    if ma!=mb or not np.array_equal(xa,xb):raise ValueError('initial host observation mismatch')
    if scan(a['world'],a['forward'],7)[0]==scan(b['world'],b['forward'],7)[0]:
        raise ValueError('static ray cannot distinguish')
    if scan(a['world'],a['forward'],1)!=scan(b['world'],b['forward'],1):
        raise ValueError('probe has early information leakage')
    return {'trap':a,'open':b}


def scan(w,action,count):
    """Return (blocked, free_ratio); charge EXACTLY count read-only cells.

    All probed cells are ahead of current cursor; reads include cells beyond
    the first obstacle, but their only exposed feature is contiguous free run.
    No viability or goal label is consulted. Reading is synthetic, non-native.
    """
    if action not in MOVE or count not in (1,3,7):raise ValueError('unbounded scan')
    dx,dy=MOVE[action];x,y=w.cursor
    blocked=[w.occupied((x+i*dx,y+i*dy)) for i in range(1,count+1)]
    first=next((i for i,b in enumerate(blocked) if b),count)
    return float(first<count),float(first/count)


def fresh_state():
    return {'spent':0,'seen':[],'retreating':False,'forward_steps':0,'started':False}


def observe(w,arm,forward,retreat,state):
    """One observation request per host decision; persistent read budget.

    Probe-stage index depends on *actual legal movement* since fork. If the
    model deviates from the route, unspent reads are not magically reassigned.
    Memory can become set only by a genuine 3-cell blocked observation.
    """
    if arm not in ARMS:raise ValueError('arm')
    count=0;hit=free=0.
    if arm=='static7' and not state['started']:
        count=7
    elif arm=='sequential7' and not state['retreating']:
        stage=state['forward_steps']; pos=stage
        if pos in PROBE_PLAN and pos not in state['seen'] and state['spent']+PROBE_PLAN[pos]<=7:
            count=PROBE_PLAN[pos]
    if count:
        hit,free=scan(w,forward,count)
        state['seen'].append(state['forward_steps'])
        state['spent']+=count
        if arm=='sequential7' and state['forward_steps']==3 and hit:
            state['retreating']=True
    state['started']=True
    if state['spent']>7:raise ValueError('information budget exceeded')
    return {'reads':count,'hit':hit,'free_ratio':free,'spent':state['spent'],
            'retreating':state['retreating']}


def features(w,goal,horizon,forward,retreat,state,obs):
    menu,base=host.matrix(w,goal,horizon,False)
    x=np.zeros((policy.ACTION_MAX,WIDTH),np.float64)
    x[:,:host.WIDTH]=base
    for i,act in enumerate(menu):
        if act==forward:
            x[i,-3: -1]=(obs['hit'],obs['free_ratio'])
        if state['retreating'] and act in ('BACKTRACK',retreat):
            x[i,-1]=1.
    return menu,x


class ProbeRanker(host.Ranker):
    def __init__(self,seed,hidden=24):
        super().__init__(seed,hidden)
        self.p['w1']=np.concatenate([self.p['w1'],np.random.default_rng(seed+1549).normal(0,.09,(EXTRA,hidden))],axis=0)
        self.m['w1']=np.zeros_like(self.p['w1'])
        self.v['w1']=np.zeros_like(self.p['w1'])

    @classmethod
    def from_base(cls,base):
        m=cls(base.seed,base.hidden)
        for k in ('b1','w2','b2'):m.p[k]=base.p[k].copy()
        m.p['w1'][:host.WIDTH]=base.p['w1'].copy()
        return m


def choose(model,w,goal,horizon,forward,retreat,state,arm):
    obs=observe(w,arm,forward,retreat,state)
    menu,x=features(w,goal,horizon,forward,retreat,state,obs)
    p=model.probs(x,len(menu))
    act=menu[int(p.argmax())]
    return act,{a:round(float(p[i]),8) for i,a in enumerate(menu)},obs


def sensor_mask_control(model,w,goal,horizon,forward,retreat,state,obs):
    """Counterfactual decision on SAME state with the 3-cell report suppressed.

    The masked prediction is diagnostic only and never executed, so it cannot
    alter game state or teacher labels. It checks whether the trained model
    responds to evidence rather than merely recognizing progress along path.
    """
    masked=copy.deepcopy(state);masked['retreating']=False
    empty=dict(obs,hit=0.,free_ratio=0.,retreating=False)
    menu,x=features(w,goal,horizon,forward,retreat,masked,empty)
    prob=model.probs(x,len(menu))
    return {'masked_action':menu[int(prob.argmax())],
            'masked_probabilities':{a:round(float(prob[i]),8) for i,a in enumerate(menu)}}


def advance_state(w,forward,retreat,state,act,before):
    if act==forward and w.cursor != before:
        state['forward_steps']+=1
    # When the model deliberately turns, no repeated sensor scan at a seen stage.
    # A trap retreat memory persists until actually leaving the fork westward.
    if state['retreating'] and act==retreat:
        state['retreating']=False


def instruction(w,kind,arm,forward,retreat,state,step):
    # All outputs are checked against original host-legal menu below.
    if arm=='static7':
        expected=retreat if kind=='trap' else forward
    elif arm=='sequential7':
        expected=(('BACKTRACK' if state['retreating'] and len(w.path)>1 else retreat)
                  if kind=='trap' and state['retreating'] else forward)
    else:
        expected=forward
    if expected not in w.menu(12):
        if state['retreating'] and 'BACKTRACK' in w.menu(12):return 'BACKTRACK'
        return next((a for a in (retreat,forward,'BACKTRACK','RELOCATE_START') if a in w.menu(12)), 'STOP')
    return expected


def train_cases(seed,stage,natural_limit=48):
    if stage not in STAGES or not 16<=natural_limit<=128:raise ValueError('bound')
    original,masks,y,prior=host.corpus(seed,stage)
    rng=np.random.default_rng(seed*7+stage)
    indices=np.sort(rng.choice(len(y),size=min(len(y),natural_limit),replace=False))
    natural=np.concatenate([original[indices],np.zeros((len(indices),policy.ACTION_MAX,EXTRA))],axis=2)
    X={a:list(natural) for a in ARMS};M={a:list(masks[indices]) for a in ARMS}
    Y={a:list(y[indices]) for a in ARMS};ledger=[]
    for sp in specs('train',1):
        for turns in rotate.ROTATION:
            cases=matched(sp,turns)
            for kind,q in cases.items():
                for arm in ARMS:
                    w=copy.deepcopy(q['world']);state=fresh_state()
                    for step in range(10):
                        if w.stopped:break
                        obs=observe(w,arm,q['forward'],q['retreat'],state)
                        menu,mat=features(w,q['goal'],stage,q['forward'],q['retreat'],state,obs)
                        label=instruction(w,kind,arm,q['forward'],q['retreat'],state,step)
                        mask=np.zeros(policy.ACTION_MAX,bool);mask[:len(menu)]=True
                        X[arm].append(mat);M[arm].append(mask);Y[arm].append(menu.index(label))
                        pre=w.state_sha();ev=w.apply(label,12)
                        advance_state(w,q['forward'],q['retreat'],state,label,tuple(ev['cursor_before']))
                        ledger.append({'split':'train','seed':seed,'stage':stage,
                            'spec':sp.__dict__,'turns':turns,'kind':kind,'arm':arm,'step':step,
                            'pre':pre,'post':w.state_sha(),'menu':menu,'obs':obs,
                            'label':label,'event':ev})
    return ({a:np.stack(X[a]) for a in ARMS},{a:np.stack(M[a]) for a in ARMS},
            {a:np.asarray(Y[a],np.int64) for a in ARMS},ledger,
            {'natural_examples_per_arm':len(indices),'targeted_examples_per_arm':len(ledger)//len(ARMS),
             'natural_sensor_zero_padded':True,'source':prior,'train_only':True})



def verify_train_ledger(ledger):
    """Independently regenerate labeled, legal TRAIN paths and sensor charges."""
    blocks={}
    for row in ledger:
        if row['split']!='train' or row['stage'] not in STAGES or row['arm'] not in ARMS:
            raise ValueError('invalid training provenance')
        spec=Spec(**row['spec'])
        if spec.split!='train':raise ValueError('non-TRAIN training state')
        key=(row['seed'],row['stage'],spec.ident,row['turns'],row['kind'],row['arm'])
        blocks.setdefault(key,[]).append(row)
    for key,rows in blocks.items():
        seed,stage,ident,turns,kind,arm=key
        if [x['step'] for x in rows]!=list(range(len(rows))) or len(rows)!=10:
            raise ValueError('training sequence incomplete')
        q=fixture(Spec(**rows[0]['spec']),turns,kind);w=q['world'];state=fresh_state()
        for row in rows:
            obs=observe(w,arm,q['forward'],q['retreat'],state)
            menu,_=features(w,q['goal'],stage,q['forward'],q['retreat'],state,obs)
            label=instruction(w,kind,arm,q['forward'],q['retreat'],state,row['step'])
            pre=w.state_sha();event=w.apply(label,12)
            advance_state(w,q['forward'],q['retreat'],state,label,tuple(event['cursor_before']))
            if (row['obs']!=obs or row['menu']!=menu or row['label']!=label or
                row['pre']!=pre or row['post']!=w.state_sha() or row['event']!=event):
                raise ValueError('training source/replay mismatch')
    return len(blocks),len(ledger)


def train(seed,base_updates=12,adapt_updates=45,natural_limit=48):
    if not 1<=base_updates<=50 or not 1<=adapt_updates<=100:raise ValueError('optimizer budget')
    base,pre=host.train(seed,base_updates);frozen=base['goal_recovery']
    models={a:ProbeRanker.from_base(frozen) for a in ARMS}
    hist=[];ledger=[]
    for stage in STAGES:
        X,M,Y,rows,meta=train_cases(seed,stage,natural_limit)
        history={a:models[a].fit(X[a],M[a],Y[a],adapt_updates,seed*819+stage) for a in ARMS}
        hist.append({'stage':stage,'optimizer':history,'data':meta})
        ledger.extend(rows)
    return models,{'baseline':pre,'adaptation':hist},ledger


def episode(sp,turns,kind,arm,model,horizon=128,png=None):
    if sp.split!='development' or kind not in KINDS or arm not in ARMS or horizon not in (36,72,128):
        raise ValueError('development only')
    q=fixture(sp,turns,kind);w=q['world'];state=fresh_state();start=w.state_sha();before=len(w.ink)
    rows=[];lat=[];reached_safe=False;first_resolve=None
    for t in range(horizon):
        if w.stopped:break
        tic=time.perf_counter_ns()
        a,p,obs=choose(model,w,q['goal'],horizon,q['forward'],q['retreat'],state,arm)
        lat.append((time.perf_counter_ns()-tic)/1e6)
        control=(sensor_mask_control(model,w,q['goal'],horizon,q['forward'],q['retreat'],state,obs)
                 if arm=='sequential7' and obs['spent']==7 and obs['reads']==3 else None)
        ev=w.apply(a,12)
        advance_state(w,q['forward'],q['retreat'],state,a,tuple(ev['cursor_before']))
        # Reach/exit witnesses (not used to drive the policy).
        root=rotate.rotate_xy((48,48),turns)
        safe=rotate.rotate_xy((56,48),turns)
        if first_resolve is None and ((kind=='trap' and w.cursor==root) or (kind=='open' and w.cursor==safe)):
            first_resolve=t+1
        rows.append({'requested':a,'probabilities':p,'observation':obs,'event':ev,
                     'sensor_mask_control':control})
    if png is not None:_png(png,w)
    return {'version':VERSION,'split':'development','spec':sp.__dict__,'turns':turns,'kind':kind,
            'arm':arm,'goal':list(q['goal']),'forward':q['forward'],'retreat':q['retreat'],
            'prefix':q['prefix'],'initial_sha256':start,'base_sha256':w.base_sha(),
            'horizon':horizon,'rows':rows,'sensor_cells_read':state['spent'],
            'first_action':rows[0]['requested'],'first_resolution_step':first_resolve,
            'new_paint':len(w.ink)-before,'final_sha256':w.state_sha(),
            'metrics':w.metrics(horizon+len(q['prefix'])),
            'mean_latency_ms':round(statistics.mean(lat),6) if lat else 0.}


def verify_episode(ep,model):
    if ep['version']!=VERSION or ep['split']!='development':raise ValueError('episode identity')
    sp=Spec(**ep['spec']);q=fixture(sp,ep['turns'],ep['kind']);w=q['world'];state=fresh_state()
    if ep['initial_sha256']!=w.state_sha() or ep['base_sha256']!=w.base_sha() or ep['goal']!=list(q['goal']) or ep['prefix']!=q['prefix']:
        raise ValueError('world source mismatch')
    before=len(w.ink);resolve=None
    for t,row in enumerate(ep['rows']):
        a,p,obs=choose(model,w,q['goal'],ep['horizon'],q['forward'],q['retreat'],state,ep['arm'])
        if a!=row['requested'] or p!=row['probabilities'] or obs!=row['observation']:
            raise ValueError('neural/observation replay mismatch')
        expected_control=(sensor_mask_control(model,w,q['goal'],ep['horizon'],q['forward'],q['retreat'],state,obs)
                          if ep['arm']=='sequential7' and obs['spent']==7 and obs['reads']==3 else None)
        if expected_control!=row.get('sensor_mask_control'):
            raise ValueError('masked sensor counterfactual mismatch')
        ev=w.apply(a,12)
        if ev!=row['event']:raise ValueError('host-legal event mismatch')
        advance_state(w,q['forward'],q['retreat'],state,a,tuple(ev['cursor_before']))
        if resolve is None and ((ep['kind']=='trap' and w.cursor==rotate.rotate_xy((48,48),ep['turns'])) or
                                (ep['kind']=='open' and w.cursor==rotate.rotate_xy((56,48),ep['turns']))):
            resolve=t+1
    if (state['spent']!=ep['sensor_cells_read'] or len(ep['rows'])>ep['horizon'] or
        (len(ep['rows'])<ep['horizon'] and not w.stopped) or
        ep['first_action']!=ep['rows'][0]['requested'] or resolve!=ep['first_resolution_step'] or
        ep['new_paint']!=len(w.ink)-before or ep['final_sha256']!=w.state_sha() or
        ep['metrics']!=w.metrics(ep['horizon']+len(q['prefix']))):
        raise ValueError('end state or budget mismatch')
    return len(ep['rows'])


def summarize(eps):
    result={}
    for kind in KINDS:
        for arm in ARMS:
            es=[e for e in eps if e['kind']==kind and e['arm']==arm]
            if not es:continue
            acts=Counter(r['event']['chosen'] for e in es for r in e['rows'])
            controls=[(r['requested'],r['sensor_mask_control']['masked_action'])
                      for ep in es for r in ep['rows'] if r.get('sensor_mask_control')]
            result[kind+'/'+arm]={
                'episodes':len(es),'initial_forward':sum(e['first_action']==e['forward'] for e in es),
                'mask_controls':len(controls),
                'sensor_mask_action_flips':sum(a!=b for a,b in controls),
                'resolved':sum(e['first_resolution_step'] is not None for e in es),
                'early_stop':sum(len(e['rows'])<e['horizon'] for e in es),
                'mean_paint':round(statistics.mean(e['new_paint'] for e in es),3),
                'mean_latency_ms':round(statistics.mean(e['mean_latency_ms'] for e in es),6),
                'mean_reads':round(statistics.mean(e['sensor_cells_read'] for e in es),3),
                'mean_decisions':round(statistics.mean(len(e['rows']) for e in es),3),
                'actions':dict(sorted(acts.items()))}
    return result


def source_sha():return {f:sha_file(Path(__file__).parent/f) for f in SOURCE}


def run(out,seed=811,base_updates=12,adapt_updates=45,natural_limit=48,quick=False):
    if out.exists():raise FileExistsError('immutable evidence directory exists')
    if seed not in (811,812,813,814):raise ValueError('seed set bounded')
    out.mkdir(parents=True);start=time.perf_counter();gc.collect();old=gc.isenabled()
    if old:gc.disable()
    try:
        models,hist,ledger=train(seed,base_updates,adapt_updates,natural_limit)
        cps={}
        for arm,m in models.items():
            f=out/(arm+'.npz');m.save(f)
            mm=ProbeRanker.load(f)
            if any(not np.array_equal(m.p[k],mm.p[k]) for k in m.p):raise ValueError('checkpoint restore')
            cps[arm]={'sha256':sha_file(f),'parameters':sum(z.size for z in m.p.values()),'updates':m.updates}
        models={a:ProbeRanker.load(out/(a+'.npz')) for a in ARMS}
        (out/'traces').mkdir();(out/'png').mkdir();eps=[]
        for sp in specs('development',1)[:1 if quick else 4]:
            for turns in ((0,) if quick else (0,1)):
                matched(sp,turns)
                for kind in KINDS:
                    for arm in ARMS:
                        fn=f'{sp.family}-{sp.seed}-{turns}-{kind}-{arm}'
                        png=(out/'png'/(fn+'.png')) if not quick and turns==0 else None
                        ep=episode(sp,turns,kind,arm,models[arm],36 if quick else 128,png)
                        verify_episode(ep,models[arm]);eps.append(ep)
                        (out/'traces'/(fn+'.json')).write_text(json.dumps(ep,sort_keys=True)+'\n')
    finally:
        if old:gc.enable()
    (out/'training.json').write_text(json.dumps(ledger,sort_keys=True)+'\n')
    report={'version':VERSION,'seed':seed,'source_sha256':source_sha(),
            'split':'development','sealed_heldout_opened':False,'native_receipts':0,
            'paid_compute':False,'curriculum':STAGES,'base_updates':base_updates,
            'adapt_updates':adapt_updates,'natural_limit':natural_limit,
            'training_history':hist,'checkpoints':cps,'summary':summarize(eps),
            'training_rows':len(ledger),'episodes':len(eps),
            'decisions':sum(len(x['rows']) for x in eps),
            'elapsed_seconds':round(time.perf_counter()-start,3),
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    (out/'SHA256SUMS.json').write_text(json.dumps({str(f.relative_to(out)):sha_file(f)
         for f in out.rglob('*') if f.is_file()},sort_keys=True,indent=2)+'\n')
    return report


def verify(out):
    rep=json.loads((out/'report.json').read_text())
    if rep['version']!=VERSION or rep['source_sha256']!=source_sha():raise ValueError('source SHA mismatch')
    files={str(f.relative_to(out)):sha_file(f) for f in out.rglob('*')
           if f.is_file() and f.name!='SHA256SUMS.json'}
    if files!=json.loads((out/'SHA256SUMS.json').read_text()):raise ValueError('manifest mismatch')
    models={a:ProbeRanker.load(out/(a+'.npz')) for a in ARMS}
    for a,m in models.items():
        if rep['checkpoints'][a]['sha256']!=sha_file(out/(a+'.npz')) or rep['checkpoints'][a]['updates']!=m.updates:
            raise ValueError('checkpoint hash/update mismatch')
    train_rows=json.loads((out/'training.json').read_text())
    verify_train_ledger(train_rows)
    episodes=[]
    for f in sorted((out/'traces').glob('*.json')):
        ep=json.loads(f.read_text());verify_episode(ep,models[ep['arm']]);episodes.append(ep)
    summary=summarize(episodes)
    for k in summary:
        summary[k].pop('mean_latency_ms');rep['summary'][k].pop('mean_latency_ms')
    if (summary!=rep['summary'] or len(episodes)!=rep['episodes'] or
        sum(len(x['rows']) for x in episodes)!=rep['decisions'] or
        len(train_rows)!=rep['training_rows']):
        raise ValueError('aggregate mismatch')
    return {'checkpoints':len(models),'episodes':len(episodes),
            'decisions':sum(len(x['rows']) for x in episodes),
            'training_rows':rep['training_rows'],'verified_files':len(files)}


if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--out',type=Path)
    cli.add_argument('--verify',type=Path);cli.add_argument('--seed',type=int,default=811)
    cli.add_argument('--base-updates',type=int,default=12);cli.add_argument('--adapt-updates',type=int,default=45)
    cli.add_argument('--natural-limit',type=int,default=48);cli.add_argument('--quick',action='store_true')
    a=cli.parse_args();print(json.dumps(verify(a.verify) if a.verify else
        run(a.out,a.seed,a.base_updates,a.adapt_updates,a.natural_limit,a.quick),sort_keys=True))

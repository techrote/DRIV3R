#!/usr/bin/env python3
"""QL-019: genuinely replayed W0 on-policy trajectories using held-out surrogate policies.

NOT original QL-017 neural policies: refit their archived score responses on four
source world seeds, validate on two disjoint held-out world seeds, and only
then run fresh W0 worlds with/without the immutable QL-018 Q4 correction.
This is an intervention robustness test on surrogate models, not promotion or
proof of original-model on-policy gains, nor GGUF quantization.
"""
import argparse
import collections
import hashlib
import json
import math
import random
import resource
import statistics
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor

ACTIONS = ('STRAIGHT','LEFT','RIGHT','RETRACE','BRANCH','STOP')
FAMILIES = ('GREEBLES','PIPEWORK','ARCHITECTURE')
DIRS = ((0,-1),(1,0),(0,1),(-1,0))
SIZE = 25
B = ACTIONS.index('BRANCH'); R = ACTIONS.index('RETRACE'); S = ACTIONS.index('STOP')
FITTING_SEEDS = (10611,10623,10647,10651)
VALIDATION_SEEDS = (10667,10679)
PROSPECTIVE_SEEDS = (10711,10723,10747,10751,10767,10779)
POLICIES = ('recovery-q1-branch12','recovery-q1-branch08')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_verified(path, manifest_file):
    with zipfile.ZipFile(path) as z:
        manifest=json.loads(z.read(manifest_file))
        entries=manifest['entries'] if isinstance(manifest,dict) else manifest
        for item in entries:
            data=z.read(item['path'])
            assert sha(data)==item['sha256'] and len(data)==item['bytes'],item['path']
        return {p:z.read(p) for p in z.namelist()},len(entries)


class WormWorld:
    """Copied W0 fixture mechanics verbatim in semantics; validated against archived steps."""
    def __init__(self,seed,family):
        self.seed=seed;self.family=family
        r=random.Random(seed)
        self.grid=np.zeros((SIZE,SIZE),np.uint8)
        for y in range(SIZE):
            for x in range(SIZE):
                v=r.random()
                self.grid[y,x]=1 if v<.085 else 2 if v<.115 else 0
        k=SIZE//2
        self.grid[k-2:k+3,k-2:k+3]=0
        self.pos=(k,k);self.heading=r.randrange(4)
        self.path=[self.pos];self.visited={self.pos};self.ink={self.pos}
        self.branches=0;self.stopped=False;self.actions=[]

    def neighbor(self,delta):
        dx,dy=DIRS[(self.heading+delta)%4]
        return self.pos[0]+dx,self.pos[1]+dy

    def state(self,xy):
        x,y=xy
        if not(0<=x<SIZE and 0<=y<SIZE):return -1
        if self.grid[y,x]==2:return -2
        if self.grid[y,x]==1:return -1
        return 0 if xy in self.visited else 1

    def legal(self):
        if self.stopped:return ('STOP',)
        legal=[a for d,a in ((0,'STRAIGHT'),(-1,'LEFT'),(1,'RIGHT')) if self.state(self.neighbor(d))>=0]
        if len(self.path)>1:legal.append('RETRACE')
        if len(self.ink)>=4 and sum(self.state(self.neighbor(d))==1 for d in (0,-1,1))>=2:
            legal.append('BRANCH')
        legal.append('STOP')
        return tuple(legal)

    def features(self,tick):
        local=[self.state(self.neighbor(d)) for d in (0,-1,1)]
        x,y=self.pos
        values=[]
        for st in local:
            values.extend(float(st==k) for k in (-2,-1,0,1))
        values.extend(float(self.heading==k) for k in range(4))
        values.extend(float(self.family==k) for k in FAMILIES)
        values.extend([min(tick,64)/64,min(len(self.ink),64)/64,min(len(self.path),64)/64,
                       min(self.branches,16)/16,(x-SIZE/2)/SIZE,(y-SIZE/2)/SIZE,
                       float(len(self.legal())-1)/5,
                       float(bool(self.actions) and self.actions[-1]=='RETRACE')])
        assert len(values)==27
        return np.asarray(values,dtype=np.float32)

    def apply(self,choice):
        assert choice in self.legal()
        if choice=='STOP':self.stopped=True
        elif choice=='RETRACE':
            self.path.pop();self.pos=self.path[-1]
        else:
            if choice=='BRANCH':
                options=[d for d in (-1,1,0) if self.state(self.neighbor(d))==1]
                assert options
                delta=options[self.branches%len(options)];self.branches+=1
            else:delta={'STRAIGHT':0,'LEFT':-1,'RIGHT':1}[choice]
            self.heading=(self.heading+delta)%4
            self.pos=self.neighbor(0)
            assert self.state(self.pos)>=0
            self.path.append(self.pos);self.visited.add(self.pos);self.ink.add(self.pos)
        self.actions.append(choice)


def observation_features(ep):
    w=WormWorld(ep['seed'],ep['family'])
    rng=random.Random(1000003*ep['seed'])
    assert ep['first_pass']
    out=[];positions=[]
    for i,s in enumerate(ep['steps']):
        assert i==s['tick'] and tuple(s['position'])==w.pos
        assert tuple(s['legal'])==w.legal() and 'STOP' in w.legal()
        f=w.features(i)
        v=np.asarray(s['scores'],np.float32)
        assert v.shape==(6,) and np.isfinite(v).all()
        masked=np.asarray([a in w.legal() for a in ACTIONS],bool)
        assert ACTIONS[int(np.argmax(np.where(masked,v,-1e9)))]==s['masked_top']
        is_override=rng.random()<ep['override_rate']
        assert is_override==s['host_override']
        chosen=rng.choice(w.legal()) if is_override else s['masked_top']
        assert chosen==s['chosen']
        assert chosen in w.legal()
        ctx=context_from_positions(w,i,positions)
        out.append((f,v,masked,ctx))
        positions.append(w.pos)
        w.apply(chosen)
        assert w.pos==tuple(s['next_position'])
    assert len(out)==len(ep['steps'])
    return out


def context(w,tick):
    prior=w.actions
    repeats=1-len(set(w.path))/max(1,len(w.path))
    # QL-018 used unique visited coordinates / all prior positions, not current live path.
    return np.asarray([1.,min(tick/96.,1.),repeats,
        prior[-8:].count('BRANCH')/8.,prior[-8:].count('RETRACE')/8.,
        float('RETRACE' in w.legal())],dtype=np.float32)


def context_from_positions(w,tick,positions):
    # Correct the feature definition to match QL-018 archived extraction exactly.
    seen=len(set(positions+[w.pos])); repeats=1-seen/(len(positions)+1)
    prior=w.actions
    return np.asarray([1.,min(tick/96.,1.),repeats,
        prior[-8:].count('BRANCH')/8.,prior[-8:].count('RETRACE')/8.,
        float('RETRACE' in w.legal())],dtype=np.float32)


def load_q4(payload,expected):
    assert len(payload)==7
    scale=struct.unpack('<f',payload[3:])[0]
    code=[]
    for i in range(6):
        off=i*4;v=(payload[off//8]>>(off%8))&15
        code.append(v-7)
    out=np.asarray(code,np.float32)*scale
    assert np.allclose(out,expected,atol=2e-6,rtol=2e-6)
    return out


def render_world(w,path,scale=8):
    colors={0:(243,239,221),1:(79,82,94),2:(220,89,96)}
    im=Image.new('RGB',(SIZE,SIZE)); pix=im.load()
    for y in range(SIZE):
        for x in range(SIZE):
            xy=(x,y)
            pix[x,y]=(246,181,65) if xy==w.pos else (46,103,142) if xy in w.ink else colors[int(w.grid[y,x])]
    im.resize((SIZE*scale,SIZE*scale),Image.Resampling.NEAREST).save(path)


def rollout(model,wts,seed,family,rate,tag,out,horizon=96):
    w=WormWorld(seed,family);rng=random.Random(1000003*seed)
    positions=[];steps=[]
    for tick in range(horizon):
        if w.stopped:break
        f=w.features(tick)
        menu=w.legal()
        score=np.asarray(model.predict(f[None])[0],np.float64)
        if wts is not None:
            x=context_from_positions(w,tick,positions)
            score[B]+=float(x@wts)
        legal=np.asarray([a in menu for a in ACTIONS],bool)
        choice=ACTIONS[int(np.argmax(np.where(legal,score,-1e9)))]
        override=rng.random()<rate
        action=rng.choice(menu) if override else choice
        assert action in menu
        pos=w.pos
        w.apply(action)
        positions.append(pos)
        steps.append({'tick':tick,'pos':pos,'next':w.pos,'legal':menu,
            'greedy':choice,'chosen':action,'host_override':override})
    fname=f'{tag}-{family.lower()}-{seed}-{int(rate*100)}pct.png'
    png=out/'gallery'/fname
    render_world(w,png)
    hist=collections.Counter(w.actions)
    return {'policy':tag,'seed':seed,'family':family,'rate':rate,
        'turns':len(steps),'unique_cells':len(w.ink),
        'branch':hist['BRANCH'],'retrace':hist['RETRACE'],'stop':hist['STOP'],
        'host_override_stops':sum(s['chosen']=='STOP' and s['host_override'] for s in steps),
        'png':str(png.relative_to(out)),'png_sha256':sha(png.read_bytes())},steps


def summarise(eps):
    return {'episodes':len(eps),'mean_cells':round(statistics.mean(e['unique_cells'] for e in eps),4),
            'mean_turns':round(statistics.mean(e['turns'] for e in eps),4),
            'total_retrace':sum(e['retrace'] for e in eps),
            'total_branch':sum(e['branch'] for e in eps),
            'total_stop':sum(e['stop'] for e in eps),
            'override_stops':sum(e['host_override_stops'] for e in eps),
            'steps':sum(e['turns'] for e in eps),
            'per_seed':{str(s):{'mean_cells':round(statistics.mean(e['unique_cells'] for e in eps if e['seed']==s),4)} for s in PROSPECTIVE_SEEDS}}


def run(args):
    root=Path(args.out);(root/'gallery').mkdir(parents=True,exist_ok=True)
    z17,n17=read_verified(args.ql017,'MANIFEST.json')
    z18,n18=read_verified(args.ql018,'EVIDENCE_MANIFEST.json')
    q17=json.loads(z17['results.json']);q18=json.loads(z18['ql018-results.json'])
    assert q17['schema']=='ql017-separated-recovery-branch-v1'
    assert q18['schema']=='ql018-archived-score-shadow-calibration-v1'
    assert sha(Path(args.ql017).read_bytes())==q18['input_ql017_archive_sha256']
    assert q18['train_seeds']==list(FITTING_SEEDS)
    assert q18['test_seeds']==list(VALIDATION_SEEDS)
    raw=json.loads(z17['raw_decisions.json'])
    replayed=0;records={p:{'train':[],'valid':[]} for p in POLICIES}
    for ep in raw:
        steps=observation_features(ep)
        replayed+=len(steps)
        if ep['policy'] in POLICIES:
            group='train' if ep['seed'] in FITTING_SEEDS else 'valid'
            records[ep['policy']][group].extend(steps)
    assert replayed==12449
    assert not set(PROSPECTIVE_SEEDS)&set(FITTING_SEEDS+VALIDATION_SEEDS)
    reports={};trajectories=[];metrics={}
    for policy in POLICIES:
        train=records[policy]['train'];valid=records[policy]['valid']
        X=np.stack([v[0] for v in train]);Y=np.stack([v[1] for v in train])
        Xv=np.stack([v[0] for v in valid]);Yv=np.stack([v[1] for v in valid]);Lm=np.stack([v[2] for v in valid])
        model=ExtraTreesRegressor(n_estimators=20,max_depth=16,min_samples_leaf=2,
                                   random_state=19019,n_jobs=1)
        t0=time.perf_counter();model.fit(X,Y);fit_time=time.perf_counter()-t0
        pred=model.predict(Xv)
        truechoice=np.where(Lm,Yv,-1e9).argmax(1)
        predchoice=np.where(Lm,pred,-1e9).argmax(1)
        act_agree=float(np.mean(truechoice==predchoice))
        margin_error=float(np.mean(abs((pred[:,B]-pred[:,R])-(Yv[:,B]-Yv[:,R]))))
        val={'train_n':len(train),'valid_n':len(valid),
            'heldout_masked_action_agreement':round(act_agree,5),
            'heldout_full_logit_mae':round(float(np.mean(abs(pred-Yv))),5),
            'heldout_branch_minus_retrace_margin_mae':round(margin_error,5),
            'heldout_true_branch':int(np.sum(truechoice==B)),
            'heldout_surrogate_branch':int(np.sum(predchoice==B)),
            'fit_seconds':round(fit_time,4)}
        q4=load_q4(z18[f'ql018-models/{policy}-context-q4.bin'],
                   q18['results'][policy]['context-q4']['weights'])
        Z=np.stack([v[3] for v in valid])
        direct=Yv.copy();direct[:,B]+=(Z@q4)
        approx=pred.copy();approx[:,B]+=(Z@q4)
        true_cal=np.where(Lm,direct,-1e9).argmax(1)
        sur_cal=np.where(Lm,approx,-1e9).argmax(1)
        val['heldout_q4_action_agreement']=round(float(np.mean(true_cal==sur_cal)),5)
        val['heldout_true_q4_branch']=int(np.sum(true_cal==B))
        val['heldout_surrogate_q4_branch']=int(np.sum(sur_cal==B))
        val['heldout_true_q4_flips']=int(np.sum(true_cal!=truechoice))
        val['heldout_surrogate_q4_flips']=int(np.sum(sur_cal!=predchoice))
        val['q4_flip_recall_on_stored_scores']=round(float(
            np.mean((sur_cal!=predchoice)[true_cal!=truechoice])
            if np.any(true_cal!=truechoice) else 1.),5)
        ztrain=np.stack([v[3] for v in train]);mean_shift=float((ztrain@q4).mean())
        constant=np.asarray([mean_shift,0,0,0,0,0],np.float32)
        val['mean_training_q4_branch_logit_shift']=round(mean_shift,6)
        # DO NOT use any prospective scores in fitting. All prospective scores
        # are emitted by the fitted surrogate, not the true archived model.
        per={}
        for intervention,weights in (('uncorrected',None),('q4',q4),('q4-constant-ablation',constant)):
            tag=policy+'-'+intervention
            for rate in (0.,.10):
                eps=[]
                start=time.perf_counter_ns()
                for seed in PROSPECTIVE_SEEDS:
                    for fam in FAMILIES:
                        e,tr=rollout(model,weights,seed,fam,rate,tag,root)
                        eps.append(e);trajectories.append({'episode':e,'steps':tr})
                elapsed=(time.perf_counter_ns()-start)/1e9
                cohort=summarise(eps)
                cohort['total_rollout_elapsed_seconds_including_png']=round(elapsed,4)
                per[f'{tag}-{int(rate*100)}pct']=cohort
        # measured same-process per-decision end-to-end surrogate + correction
        tm={}
        for key,weights in (('surrogate_only',None),('surrogate_plus_q4',q4)):
            times=[]
            for i in range(min(40,len(Xv))):
                t=time.perf_counter_ns();score=model.predict(Xv[i:i+1])[0]
                if weights is not None:
                    score[B]+=float(np.dot(np.array([1,0,0,0,0,1.],np.float32),weights))
                np.argmax(np.where(Lm[i],score,-1e9))
                times.append((time.perf_counter_ns()-t)/1e6)
            tm[key]={'median_ms':round(float(np.median(times)),5),
                     'p95_ms':round(float(np.quantile(times,.95)),5),
                     'decisions_per_sec_reference':round(1000/float(np.mean(times)),1)}
        reports[policy]={'surrogate_validation':val,'q4_packed_bytes':len(z18[f'ql018-models/{policy}-context-q4.bin']),
                         'q4_packed_sha256':sha(z18[f'ql018-models/{policy}-context-q4.bin']),
                         'latency':tm,'prospective':per}
    report={'schema':'ql019-w0-surrogate-onpolicy-v1',
        'ql017_zip_sha256':sha(Path(args.ql017).read_bytes()),
        'ql018_zip_sha256':sha(Path(args.ql018).read_bytes()),
        'ql017_source_sha':q17['source_sha'],
        'verified_ql017_files':n17,'verified_ql018_files':n18,
        'replayed_archived_legal_steps':replayed,
        'surrogate_fit_seeds':FITTING_SEEDS,'surrogate_validation_seeds':VALIDATION_SEEDS,
        'prospective_untouched_seeds':PROSPECTIVE_SEEDS,
        'surrogate_type':'ExtraTreesRegressor 20 trees, depth 16, minimum leaf 2, fixed seed',
        'models':reports,'rss_peak_kib_shared_process':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'limitations':[
            'The candidate policies are NEW fitted surrogate emulators of archived QL-017 scores, not the original QL-017 model weights.',
            'True original-model on-policy Q4 intervention remains UNTESTED; neural teacher and QL-017 checkpoint graph unavailable to this execution.',
            'Held-out validation is disjoint by world seed; outcomes from six fresh world seeds are exploratory and not a confidence interval.',
            'Actual on-policy W0 state/action transitions and new PNGs produced for both surrogate and corrected surrogate.',
            'Q4 correction was loaded byte-for-byte from QL-018; no retraining and no test-world leakage.',
            'Mean-shift-only constant ablation derived from fit-seed states; no additional training.',
            'Printed bytes refer to six-weight residual only; ExtraTrees surrogate has much larger memory than packed original policy.',
            'Reference inference uses CPU ExtraTrees, not actual GGUF IQ1/IQ2/IQ3 nor native CyberSand.',
            'STOP always legal; host-random override can cause early STOP and confound trajectory coverage.',
            'DRIV3R-only research; no native CyberSand changes or paid GPU/API usage.'
        ]}
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'trajectories.json').write_text(json.dumps(trajectories,separators=(',',':'))+'\n')
    inv=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='MANIFEST.json':
            b=p.read_bytes();inv.append({'path':str(p.relative_to(root)),'bytes':len(b),'sha256':sha(b)})
    (root/'MANIFEST.json').write_text(json.dumps(inv,indent=2)+'\n')
    print(json.dumps({'status':'QL019_W0_SURROGATE','replayed':replayed,
        'files':len(inv),'png':len(list((root/'gallery').glob('*.png'))),
        'models':{k:{'validation':v['surrogate_validation'],
            'prospective':{name:{m:cv[m] for m in ('mean_cells','total_branch','total_retrace','steps')} for name,cv in v['prospective'].items()}}
            for k,v in reports.items()}},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ql017',required=True)
    ap.add_argument('--ql018',required=True)
    ap.add_argument('--out',required=True)
    run(ap.parse_args())

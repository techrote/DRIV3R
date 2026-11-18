#!/usr/bin/env python3
"""QL-020: finite low-bit Q4 gain/novelty collar intervention study.

Uses original QL-017 recorded neural logits for source-matched shadow tests
and separately refitted QL-019 surrogate for genuinely new W0 rollouts.
No counterfactual observed action is called an original neural rollout.
Does not modify a game ABI, model checkpoint, or production source.
"""
import argparse, collections, hashlib, json, os, random, pickle, resource, statistics, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from ql019_onpolicy_surrogate import (ACTIONS,FAMILIES,B,R,S,WormWorld,observation_features,
  context_from_positions,read_verified,load_q4,render_world,FITTING_SEEDS,VALIDATION_SEEDS,
  POLICIES,sha)

STUDY_SEEDS=(10811,10823,10847,10851,10867,10879)
MODES=('parent','q4-raw','q4-half','q4-cap030','q4-fresh-collar','constant')
DIR=Path(__file__).resolve().parent

def delta_for(mode,w,ctx,features,constant):
    if mode=='parent':return 0.
    d=float(np.dot(ctx,w)) if mode!='constant' else float(constant)
    if mode=='q4-half':d*=.5
    if mode=='q4-cap030':d=float(np.clip(d,-.30,.30))
    if mode=='q4-fresh-collar':
        d=float(np.clip(d,-.60,.60))
        fresh=int(round(sum(features[j] for j in (3,7,11))))
        if fresh>=2 and ctx[3]<1/8 and d<-.20:d=-.20
        if ctx[3]>=2/8 and d>.20:d=.20
    return d

def shadow(original_rows,w,const,mode):
    n=branch0=branch1=retrace0=retrace1=stop0=stop1=flips=0
    fdelta=[]; n_fresh=0; branches_fresh=0
    for f,s,menu,ctx in original_rows:
        assert bool(menu[S]);mask=np.where(menu,s,-1e9)
        d=delta_for(mode,w,ctx,f,const)
        adjusted=mask.copy();adjusted[B]+=d if menu[B] else 0.
        p,q=int(np.argmax(mask)),int(np.argmax(adjusted));n+=1
        branch0+=(p==B);branch1+=(q==B);retrace0+=(p==R);retrace1+=(q==R)
        stop0+=(p==S);stop1+=(q==S);flips+=(p!=q)
        assert menu[q]
        fresh=sum(f[j] for j in (3,7,11))
        if fresh>=2 and menu[B]:n_fresh+=1;branches_fresh+=(q==B)
        fdelta.append(d)
    return {'n':n,'before_branch':branch0,'after_branch':branch1,
            'before_retrace':retrace0,'after_retrace':retrace1,
            'before_stop':stop0,'after_stop':stop1,'flips':flips,
            'fresh_eligible':n_fresh,'fresh_branch':branches_fresh,
            'mean_branch_delta':round(statistics.mean(fdelta),5)}

def rollout(model,w,constant,mode,seed,family,rate,horizon=96):
    world=WormWorld(seed,family);rng=random.Random(1000003*seed)
    positions=[];deltas=[];choices=[];flips=0;host_stops=0
    for t in range(horizon):
        if world.stopped:break
        f=world.features(t);ctx=context_from_positions(world,t,positions)
        sc=np.asarray(model.predict(f[None])[0],dtype=np.float64);legal=world.legal()
        m=np.asarray([a in legal for a in ACTIONS],dtype=bool)
        base=int(np.argmax(np.where(m,sc,-1e9)))
        d=delta_for(mode,w,ctx,f,constant)
        if m[B]:sc[B]+=d
        corrected=int(np.argmax(np.where(m,sc,-1e9)))
        flips+=(base!=corrected)
        override=rng.random()<rate
        chosen=rng.choice(legal) if override else ACTIONS[corrected]
        assert chosen in legal
        if chosen=='STOP' and override:host_stops+=1
        positions.append(world.pos);deltas.append(d);choices.append(chosen)
        world.apply(chosen)
    return world,{'seed':seed,'family':family,'rate':rate,'mode':mode,'turns':len(choices),
        'unique':len(world.ink),'branch':choices.count('BRANCH'),'retrace':choices.count('RETRACE'),
        'stop':choices.count('STOP'),'override_stop':host_stops,'neural_flip':flips,
        'mean_delta':round(float(np.mean(deltas)),5) if deltas else 0.}

def latency(model,valid,w,constant,mode):
    t=[]
    for _ in range(2):
        for f,score,mask,ctx in valid[:8]:
            begin=time.perf_counter_ns()
            logits=np.asarray(model.predict(f[None])[0],np.float64)
            d=delta_for(mode,w,ctx,f,constant)
            if mask[B]:logits[B]+=d
            assert mask[int(np.argmax(np.where(mask,logits,-1e9)))]
            if _ == 1:t.append((time.perf_counter_ns()-begin)/1e6)
    return {"median_ms":round(float(np.median(t)),5),"p95_ms":round(float(np.quantile(t,.95)),5),
      "decisions_per_second_reference":round(1000/float(np.mean(t)),1)}

def stats(eps):
    return {'n':len(eps),'mean_unique':round(statistics.mean(e['unique'] for e in eps),3),
      'mean_turns':round(statistics.mean(e['turns'] for e in eps),3),
      'total_branch':sum(e['branch'] for e in eps),'total_retrace':sum(e['retrace'] for e in eps),
      'total_stop':sum(e['stop'] for e in eps),'random_override_stop':sum(e['override_stop'] for e in eps),
      'total_neural_flips':sum(e['neural_flip'] for e in eps)}

def run(args):
    z17,n17=read_verified(args.ql017,'MANIFEST.json')
    z18,n18=read_verified(args.ql018,'EVIDENCE_MANIFEST.json')
    r17=json.loads(z17['results.json']);r18=json.loads(z18['ql018-results.json'])
    assert r17['source_sha']=='cddb2e4c3fad7a03a9df6a4605dfdf48a419a369'
    assert sha(Path(args.ql017).read_bytes())==r18['input_ql017_archive_sha256']
    assert set(STUDY_SEEDS).isdisjoint(FITTING_SEEDS+VALIDATION_SEEDS)
    raw=json.loads(z17['raw_decisions.json']);replayed=0
    rows={p:{'train':[],'valid':[]} for p in POLICIES}
    for ep in raw:
        observations=observation_features(ep);replayed+=len(observations)
        if ep['policy'] in POLICIES:
            part='train' if ep['seed'] in FITTING_SEEDS else 'valid'
            rows[ep['policy']][part]+=observations
    assert replayed==12449
    out=Path(args.out);(out/'gallery').mkdir(parents=True,exist_ok=True)
    records=[];reports={};t0=time.perf_counter()
    for policy in POLICIES:
        train,valid=rows[policy]['train'],rows[policy]['valid']
        x=np.stack([a[0] for a in train]);y=np.stack([a[1] for a in train])
        model=ExtraTreesRegressor(n_estimators=20,max_depth=16,min_samples_leaf=2,
                                  random_state=19019,n_jobs=1)
        model.fit(x,y)
        xv=np.stack([a[0] for a in valid]);yv=np.stack([a[1] for a in valid]);mv=np.stack([a[2] for a in valid])
        agreement=float(np.mean(np.argmax(np.where(mv,model.predict(xv),-1e9),axis=1)==np.argmax(np.where(mv,yv,-1e9),axis=1)))
        assert agreement>.90
        q4=z18[f'ql018-models/{policy}-context-q4.bin']; assert len(q4)==7
        w=load_q4(q4,r18['results'][policy]['context-q4']['weights'])
        constant=float(np.mean([np.dot(a[3],w) for a in train]))
        per={}
        for mode in MODES:
            sh=shadow(valid,w,constant,mode)
            cohorts={}
            for rate in (0.,.10):
                eps=[]
                for seed in STUDY_SEEDS:
                    for family in FAMILIES:
                        world,e=rollout(model,w,constant,mode,seed,family,rate)
                        eps.append(e);records.append({'policy':policy,**e})
                        png=out/'gallery'/f'{policy}-{mode}-{seed}-{family.lower()}-{int(rate*100)}pct.png'
                        render_world(world,png)
                cohorts[f'{int(rate*100)}pct'] = stats(eps)
            per[mode]={'exact_neural_shadow_validation':sh,'fresh_surrogate_w0':cohorts,
                'same_runner_reference_latency':latency(model,valid,w,constant,mode)}
        reports[policy]={'surrogate_validation_action_agreement':round(agreement,6),
           'q4_bytes':len(q4),'surrogate_pickle_bytes':len(pickle.dumps(model)),
           'q4_sha256':sha(q4),'q4_weights':[float(n) for n in w],
           'mean_training_q4_delta':round(constant,6),'modes':per}
    (out/'episode_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    report={'schema':'ql020-q4-margin-robustness-v1','source_sha256':sha(Path(__file__).read_bytes()),
      'ql017_zip_sha256':sha(Path(args.ql017).read_bytes()),
      'ql018_zip_sha256':sha(Path(args.ql018).read_bytes()),
      'validated_ql017_entries':n17,'validated_ql018_entries':n18,'replayed_legal_archived_actions':replayed,
      'source_ql017_sha':r17['source_sha'],'surrogate_training_seeds':FITTING_SEEDS,
      'surrogate_validation_seeds':VALIDATION_SEEDS,'fresh_test_seeds':STUDY_SEEDS,
      'policies':reports,'fresh_episode_n':len(records),'fresh_png_n':len(records),
      'wall_seconds_including_png':round(time.perf_counter()-t0,3),
      'peak_shared_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'limitations':['Archived neural score shadow tests are exact for recorded states only, not neural counterfactual trajectories.',
        'Fresh W0 rollouts use fixed QL-019 score-regression surrogate, NOT original neural model.',
        'The Q4 correction is unchanged from QL-018; collars are non-trained host-side research overlays, not low-bit new trained models.',
        'No quantized kernel speedup, GGUF IQ codecs, native game applicability or independent optimality oracle established.',
        'The small six-seed prospective cohort is exploratory; host random STOP confounds coverage; no statistical significance claim.',
        'Correction delta only affects legal BRANCH score, not action legality or other raw logits.',
        'RSS is shared process and cannot be attributed to individual models.',
        'All research is external to CyberSand; no source outside DRIV3R quant_lab changed.']}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    entries=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='MANIFEST.json':
            b=p.read_bytes();entries.append({'path':str(p.relative_to(out)),'bytes':len(b),'sha256':sha(b)})
    (out/'MANIFEST.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(json.dumps({'status':'QL020_COMPLETE','legal_replayed':replayed,'png':len(records),
        'archive_files':len(entries),'models':{p:{m:v['fresh_surrogate_w0']['10pct'] for m,v in d['modes'].items()} for p,d in reports.items()}},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--ql017',required=True)
    ap.add_argument('--ql018',required=True);ap.add_argument('--out',required=True)
    run(ap.parse_args())

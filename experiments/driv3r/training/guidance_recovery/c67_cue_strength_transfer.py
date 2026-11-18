"""C67: observation-only choice between weak and strong corrective guidance.

Train on the formerly held-out C66 executed graph cohort, compare on entirely
fresh graph identities, and keep precomputed counterfactual episodes separate
from predictions. This does NOT establish CyberSand native compatibility.
"""
from __future__ import annotations
import argparse, base64, collections, hashlib, json, math, zlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
import c65_paired_delayed_credit as c65

SEED = 670118
UPDATES = 192
DROPS = (0., .5)
STRONG = ('backtrack', 1.)
WEAK = ('backtrack', .35)
FEATURES = c65.c63.FEATURES
FAMILIES = ('wilson','kruskal')
ALLOWED = {'WEAK','STRONG','ALIGNED','SHUFFLED','COIN'}

def sha(data):return hashlib.sha256(data).hexdigest()

def prefix_features(state,drop):
    # All features are generated from prefix observations by frozen C63.
    # Pass a neutral cue so the model cannot read arm strength at decision time.
    x=c65.c63.observe_features(state,('backtrack',0.),drop).copy()
    assert x.shape==(FEATURES,) and x[9]==0 and np.isfinite(x).all()
    return x

def graph_cohorts():
    # Reconstruct the exact C66 training cohort and exclude every known prior.
    c65.c60.SEED=660118
    c65.c60.HORIZON=95
    c65.c60.c59.c58.c55.TICKS=(9,)
    _,_,older,_=c65.cohorts()
    historical={r['graph'] for group in older for r in group}
    c65_receipt=json.loads((c65.ROOT/'publication/C65-summary.json').read_text())
    historical.update(r['graph'] for group in c65_receipt['graph_receipts'] for r in group)
    worldgen=c65.c60.c59.c58.c55.c54.c52.worlds
    familiar=c65.c60.c59.c58.c55.c54.c52.TRAIN_FAMILIES
    famfn=c65.c60.c59.c58.c55.c54.c52.c50.c47.c41.world
    otherfn=c65.c60.c59.c58.c55.c54.c52.c50.c47.c42.world
    training_worlds=[]
    for fs,fn,seed in [(FAMILIES,otherfn,660118+503),(familiar,famfn,660118+907)]:
        w,r=worldgen(fs,fn,seed,8,exclude=historical)
        assert not {v['graph'] for v in r}&historical
        historical.update(v['graph'] for v in r);training_worlds+=w
    base,opt,_=c65.c60.c59.c58.c55.c54.c52.c50.c47.c44.prerequisites(c65.ROOT/'frozen/predecessor_C42')
    states,_=c65.c60.c59.c58.c56.states_for_goal(base,training_worlds,'remote')
    groups=collections.defaultdict(list)
    for st in states:groups[st[8]].append(st)
    train=[s for f in sorted(groups) for s in groups[f][:4]]
    expected=json.loads((c65.ROOT/'lab/C66-summary.json').read_text())
    assert [s[6]['graph_sha'] for s in train]==expected['graph_ids']
    historical.update(expected['graph_ids'])
    # C67 test set fixed BEFORE observing any test policy outcomes.
    c65.c60.SEED=SEED
    testworld=[];testreceipts=[]
    for fs,fn,seed,reps in [(FAMILIES,otherfn,SEED+103,16),(familiar,famfn,SEED+907,16)]:
        w,r=worldgen(fs,fn,seed,reps,exclude=historical)
        assert not {v['graph'] for v in r}&historical
        historical.update(v['graph'] for v in r);testworld+=w;testreceipts+=r
    states,skip=c65.c60.c59.c58.c56.states_for_goal(base,testworld,'remote')
    gg=collections.defaultdict(list)
    for st in states:gg[st[8]].append(st)
    chosen=[s for f in sorted(gg) for s in gg[f][:6]]
    assert len(chosen)>=24 and len({s[6]['graph_sha'] for s in chosen})==len(chosen)
    assert not {s[6]['graph_sha'] for s in chosen}&set(expected['graph_ids'])
    return base,opt,train,chosen,testreceipts,skip,len(historical)

def trainer_rows(training):
    led=json.loads(zlib.decompress(base64.b64decode((c65.ROOT/'lab/C66-ledger.zlib.b64').read_text())))
    index={(r['graph'],r['drop'],r['arm'][1]):r for r in led['records']}
    result=[]
    for st in training:
        graph=st[6]['graph_sha']
        for drop in DROPS:
            w=index[graph,drop,WEAK[1]]['outcomes']['C60-FROZEN']
            s=index[graph,drop,STRONG[1]]['outcomes']['C60-FROZEN']
            # Goal return is primary. Exploration is explicitly a weak tie-break.
            delta=12*(w['goal']-s['goal'])+.018*(w['new_sites']-s['new_sites'])
            delta+=.01*(s['distance']-w['distance'])
            delta-=.01*(w['steps']-s['steps'])
            result.append({'x':prefix_features(st,drop),'adv':float(np.clip(delta,-14,14)),
               'graph':graph,'family':st[8],'drop':drop,'weak_goal':w['goal'],'strong_goal':s['goal']})
    assert len(result)==2*len(training)
    return result

def fit(rows,mode):
    # 192 identical minibatches, optimizer settings and initial parameters.
    # Wrong credit: permute advantages within family x memory-loss strata.
    x=torch.tensor(np.asarray([r['x'] for r in rows]),dtype=torch.float32)
    a=np.asarray([r['adv'] for r in rows],np.float32)
    if mode=='shuffled':
        rng=np.random.default_rng(SEED+710)
        for f in sorted({r['family'] for r in rows}):
            for d in DROPS:
                ix=np.asarray([i for i,r in enumerate(rows) if r['family']==f and r['drop']==d])
                a[ix]=a[ix][rng.permutation(len(ix))]
    elif mode!='aligned':raise ValueError(mode)
    y=torch.tensor(.5+.48*np.tanh(a/9.),dtype=torch.float32)
    torch.manual_seed(SEED+22)
    model=nn.Linear(FEATURES,1,bias=True)
    with torch.no_grad():model.weight.zero_();model.bias.zero_()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.014,weight_decay=.03)
    rng=np.random.default_rng(SEED+44);logs=[]
    for step in range(UPDATES):
        ix=torch.tensor(rng.integers(0,len(rows),size=min(32,len(rows))),dtype=torch.int64)
        p=model(x[ix]).squeeze(-1)
        loss=nn.functional.binary_cross_entropy_with_logits(p,y[ix])
        optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step in (0,47,95,143,191):logs.append({'step':step+1,'loss':float(loss.detach())})
    return model,logs

def choice(model,x):
    with torch.no_grad(): return WEAK[1] if float(model(torch.from_numpy(x)).squeeze())>=0 else STRONG[1]

def evaluate(base,opt,states,models):
    prior=np.asarray(json.loads((c65.ROOT/'publication/C60-summary.json').read_text())['weights']['C60-ALIGNED'],np.float32)
    weights={'C60-FROZEN':prior}
    rows=[];samples=[]
    for i,st in enumerate(states):
        if i%6==0:print('C67 heldout',i,'/',len(states),flush=True)
        for drop in DROPS:
            x=prefix_features(st,drop);episodes={}
            for arm in (WEAK,STRONG):
                for p in ('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN'):
                    z=c65.policy(base,opt,st,arm,drop,p,weights,trace=True)
                    assert z['host_rejections']==0 and z['protected_sha']==st[3]['protected_sha']
                    episodes[(arm[1],p)]=z
            # No guidance validates exact policy parity, independent of cue strength.
            if drop==0:
                neutral=[c65.policy(base,opt,st,('none',0),0,p,weights) for p in ('C18-A0','C34-RESET','C60-FRONTIER','C60-FROZEN')]
                assert len({(z['action_sha'],z['ink_sha']) for z in neutral})==1
            graph=st[6]['graph_sha']
            assignments={'WEAK':WEAK[1],'STRONG':STRONG[1],
                'ALIGNED':choice(models['ALIGNED'],x),'SHUFFLED':choice(models['SHUFFLED'],x),
                'COIN':WEAK[1] if int(sha(f'{SEED}:{graph}:{drop}'.encode())[:8],16)%2 else STRONG[1]}
            base_choice=episodes[(WEAK[1],'C34-RESET')]['goal']
            out={k:episodes[(arm,'C60-FROZEN')] for k,arm in assignments.items()}
            row={'graph':graph,'family':st[8],'drop':drop,'choices':assignments,
                 'policies':{k:{key:z[key] for key in ('goal','new_sites','distance','steps','action_sha','ink_sha')}
                     | {'morphology':c65.path_metrics(z)} for k,z in out.items()},
                 'reset_weak':base_choice,
                 'reset_strong':episodes[(STRONG[1],'C34-RESET')]['goal'],
                 'base_weak':episodes[(WEAK[1],'C18-A0')]['goal'],
                 'frontier_weak':episodes[(WEAK[1],'C60-FRONTIER')]['goal']}
            rows.append(row)
            if drop==0 and len(samples)<24:
                for name in ('WEAK','STRONG','ALIGNED','SHUFFLED'):
                    z=out[name];arm=assignments[name]
                    samples.append({'graph':graph,'family':st[8],'policy':name,'arm':arm,
                        'goal':z['goal'],'start':z['start_pos'],'target':z['target'],'path':z['path'],
                        'action_sha':z['action_sha'],'ink_sha':z['ink_sha']})
    return rows,samples

def summarize(rows):
    a={}
    for d in DROPS:
        rr=[r for r in rows if r['drop']==d]
        arm={p:{'goals':sum(r['policies'][p]['goal'] for r in rr),
            'new_sites':sum(r['policies'][p]['new_sites'] for r in rr),
            'weak_selected':sum(r['choices'][p]==WEAK[1] for r in rr),
            'wins_vs_reset_weak':sum(r['policies'][p]['goal']>r['reset_weak'] for r in rr),
            'losses_vs_reset_weak':sum(r['policies'][p]['goal']<r['reset_weak'] for r in rr),
            'mean_turn_rate':float(np.mean([r['policies'][p]['morphology']['turn_rate'] for r in rr])),
            'mean_unique_ratio':float(np.mean([r['policies'][p]['morphology']['unique_ratio'] for r in rr]))}
             for p in ALLOWED}
        a[str(d)]={'conditions':len(rr),'reset_weak_goals':sum(r['reset_weak'] for r in rr),
          'base_weak_goals':sum(r['base_weak'] for r in rr),
          'frontier_weak_goals':sum(r['frontier_weak'] for r in rr),'policies':arm}
    return a

def run(out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base,opt,train,test,receipts,skips,history=graph_cohorts()
    tr=trainer_rows(train)
    models={};logs={}
    for p in ('ALIGNED','SHUFFLED'):
        models[p],logs[p]=fit(tr,p.lower())
        model2,_=fit(tr,p.lower())
        assert all(torch.equal(a,b) for a,b in zip(models[p].parameters(),model2.parameters()))
        save_file(models[p].state_dict(),str(out/(f'C67-{p}.safetensors')))
    rows,samples=evaluate(base,opt,test,models)
    # Each displayed selected outcome must correspond to one previously executed legal arm.
    assert all(r['policies'][p]['action_sha'] in {r['policies']['WEAK']['action_sha'],r['policies']['STRONG']['action_sha']}
         for r in rows for p in ALLOWED)
    assert len({r['graph'] for r in rows})==len(test)
    gallery=out/'C67-gallery.svg';c65.gallery(samples,gallery)
    payload={'schema':'driv3r-c67-cue-strength-transfer/1','seed':SEED,'updates_each':UPDATES,
      'trained_on':'C66 executed outcomes, formerly held-out; C67 is the fresh held-out transfer cohort',
      'train_graph_ids':[s[6]['graph_sha'] for s in train],
      'test_graph_ids':[s[6]['graph_sha'] for s in test],
      'historical_exclusions':history,'source_training_c66_sha256':sha((c65.ROOT/'lab/C66-ledger.zlib.b64').read_bytes()),
      'test_receipts':receipts,'test_skips':skips,'optimizer':logs,'aggregates':summarize(rows),
      'training_advantages':{'nonzero':sum(abs(r['adv'])>1e-8 for r in tr),
            'weak_wins':sum(r['weak_goal']>r['strong_goal'] for r in tr),
            'strong_wins':sum(r['weak_goal']<r['strong_goal'] for r in tr)},
      'suffixes':len(rows)*8+len(test)*4,'rows':len(rows),'gallery_traces':len(samples),
      'checkpoint_sha256':{p:sha((out/(f'C67-{p}.safetensors')).read_bytes()) for p in models},
      'limitations':['C66 outcomes reused for TRAINING, not held-out confirmation',
         'Selections are evaluated by assigning previously executed host counterfactuals',
         'Synthetic CyberSand-like host, not native CyberSand',
         'Only weak vs strong hint strength, no new action-policy learning',
         'Small sample; no confirmatory statistical significance; morphological statistics are proxies']}
    (out/'C67-summary.json').write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
    (out/'C67-ledger.zlib.b64').write_text(base64.b64encode(zlib.compress(json.dumps({'summary':payload,'rows':rows,'samples':samples},sort_keys=True,separators=(',',':')).encode(),9)).decode()+'\n')
    print(json.dumps({'train':len(train),'test':len(test),'rows':len(rows),'suffixes':payload['suffixes'],
      'training_advantages':payload['training_advantages'],'aggregates':payload['aggregates']},indent=2),flush=True)
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();run(a.out)

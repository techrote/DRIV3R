"""C21: opportunity-conditioned guidance research on a frozen synthetic C12 policy.

Owner scope: experiments/driv3r/training/guidance_recovery only.
External CyberSand-like host fixture; never a production action/observation ABI.
All optimizer labels come from host-applied paired counterfactual suffixes.
The non-BACKTRACK policy logits, legal mask, protected geometry, and C12
parameters are invariant. Checkpoints are intentionally NOT committed to Git.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file,load_file
from c19_return_anchored import collect,evaluate,a0,make_world,summary
from return_anchored_core import ReturnAnchoredGate
from attested_stack_cue import Prior32,observe32,KEYS
from personality_retention_core import load_c12
from signed_feedback import HostAdvice
from experiment import ARMS,sha
from adversarial_cues import source_hash

SCHEMA='driv3r-guidance-c21-opportunity-mass/1'
OFFSETS={'reverse':0,'ahead':100000,'shortcut':200000,'bridge':300000}

def candidate_masks(x):
    """Legal personality-favoured INK candidates vs the other turning direction.

    Indices 0, 4, 13, 14 and 21 are historical synthetic fixture features,
    NOT the immutable CyberSand native ABI.
    """
    legal=x[:,:8,4]>.5
    ink=x[:,:8,0]>.5
    fav=legal&ink&(x[:,:8,21]>.5)
    opposing=legal&ink&((x[:,:8,13]>.5)|(x[:,:8,14]>.5))&~fav
    return fav,opposing

def masked_softmax(logits,x,temperature=.8):
    mask=(x[:,:,4] if x.ndim==3 else x[:,4])<.5
    return torch.softmax(logits.masked_fill(mask,-1e7)/temperature,dim=-1)

def optimize(prior,X,A,W,G,kind,seed=210221,epochs=24):
    """Return improvement plus an explicit at-state favourite-INK mass floor.

    Contrary to outcome-based turn-gap, this constraint compares both policies
    on the exact SAME observed candidate menu, avoiding path-divergence bias.
    """
    assert kind in ('return','opportunity')
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=ReturnAnchoredGate(prior,width=12,scale=.8,variant='full')
    xx=torch.from_numpy(X); advantage=torch.from_numpy(A)
    wt=torch.from_numpy(W); groups=torch.from_numpy(G)
    fav,opp=candidate_masks(X)
    eligible=torch.from_numpy(np.any(fav,axis=1))
    favmask=torch.from_numpy(fav)
    with torch.no_grad():
        old=a0(prior)(xx)
        oldp=masked_softmax(old,xx)
        oldback=oldp[:,8]
        oldfav=(oldp[:,:8]*favmask).sum(1)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=.0012,weight_decay=.01)
    rng=np.random.default_rng(seed);losses=[];steps=0
    for _ in range(epochs):
        ids=rng.permutation(len(X));loss_total=0.
        for st in range(0,len(ids),64):
            ix=ids[st:st+64];x=xx[ix];w=wt[ix];adv=advantage[ix];g=groups[ix]
            scores=model(x);p=masked_softmax(scores,x);back=p[:,8]
            gain=-(back-oldback[ix])*adv*w
            worst=torch.stack([gain[g==k].mean() for k in range(8) if bool((g==k).any())]).max()
            drift=(back-oldback[ix]).square().mean()
            residual=model.scale*torch.tanh(model.head(x[:,8,:]).squeeze(-1))
            loss=.70*gain.mean()+.30*worst+1.6*drift+.09*residual.square().mean()
            if kind=='opportunity':
                # A relative *probability mass* floor, not a target left/right
                # action count or a label from hidden map geometry.
                fav_mass=(p[:,:8]*favmask[ix]).sum(1)
                drop=(oldfav[ix]-fav_mass)/oldfav[ix].clamp(min=.1)
                if bool(eligible[ix].any()):
                    loss=loss+10.*torch.relu(drop[eligible[ix]]-.005).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            loss_total+=float(loss.detach())*len(ix);steps+=1
        losses.append(round(loss_total/len(X),7))
    return model.eval(),steps,losses,int(eligible.sum())

def fixed_state_audit(models,rows,seed,cap=2400):
    """Score every candidate model on *identical* held-out reference states.

    Replays actual C18-A0 trajectories and captures eligible menus at each
    visited state, not merely synthetic first-state start positions. Because
    model actions are not applied during this audit, path divergence cannot
    contaminate the probability-mass comparison.
    """
    xs={k:[] for k in ('none','backtrack','continue')};encountered={k:0 for k in xs}
    for row in rows:
        if row['model']!='C18-A0':continue
        i=(row['seed']-seed-OFFSETS[row['outcome']])//7499
        assert row['seed']==seed+i*7499+OFFSETS[row['outcome']]
        w=make_world(i,row['outcome'],row['personality'],row['seed'],train=False)
        advice=HostAdvice(w)
        if row['directive']!='none':getattr(advice,'issue_'+row['directive'])(w,strength=1.,ttl=9)
        for step in row['trace']:
            menu,x=observe32(w,row['arm'],advice)
            fav,opp=candidate_masks(x[None,...])
            if fav.any() and len(xs[row['directive']])<cap:
                xs[row['directive']].append(x)
                encountered[row['directive']]+=1
            match=[c for c in menu if c.legal and tuple(c.action)==tuple(step['action']) and tuple(c.target)==tuple(step['cursor'])]
            assert len(match)==1
            assert w.apply(match[0]);advice.advance(w)
    out={}
    for directive,records in xs.items():
        if not records:
            out[directive]={'eligible_state_menus':0,'models':{}}
            continue
        batch=torch.from_numpy(np.asarray(records,np.float32))
        fav,opp=candidate_masks(batch.numpy());favor=torch.from_numpy(fav)
        table={}
        for label,model in models.items():
            favprob=[];backprob=[]
            for st in range(0,len(batch),256):
                with torch.no_grad():p=masked_softmax(model(batch[st:st+256]),batch[st:st+256])
                favprob.extend((p[:,:8]*favor[st:st+256]).sum(1).tolist())
                backprob.extend(p[:,8].tolist())
            table[label]={'favourite_ink_mass':round(float(np.mean(favprob)),7),
                          'backtrack_mass':round(float(np.mean(backprob)),7)}
        out[directive]={'eligible_state_menus':len(records),
                        'both_turns_available':int(np.sum(opp.any(axis=1))),
                        'models':table}
    return out

def trace_opportunity_audit(rows,seed):
    """Replay saved actions through a fresh synthetic host, checking every step.

    Uses only recorded actions, never candidate policy logits. Reports both
    active-advice and after-advice style opportunities for long-horizon paths.
    """
    stats={};replayed=0
    for row in rows:
        i=(row['seed']-seed-OFFSETS[row['outcome']])//7499
        assert row['seed']==seed+i*7499+OFFSETS[row['outcome']]
        w=make_world(i,row['outcome'],row['personality'],row['seed'],train=False)
        assert source_hash(w)==row['source_sha256']
        a=HostAdvice(w)
        if row['directive']!='none':getattr(a,'issue_'+row['directive'])(w,strength=1.,ttl=9)
        key=row['model']+'/'+row['directive']
        d=stats.setdefault(key,{'episodes':0,'eligible_active':0,'eligible_post':0,
          'preferred_active':0,'preferred_post':0,'opposed_active':0,'opposed_post':0,
          'preferred_available_active':0,'preferred_available_post':0,
          'preferred_chosen_active':0,'preferred_chosen_post':0,'goal_success':0})
        d['episodes']+=1;d['goal_success']+=int(row['reached'])
        for step in row['trace']:
            cs,_=observe32(w,row['arm'],a)
            matching=[c for c in cs if c.legal and tuple(c.action)==tuple(step['action']) and tuple(c.target)==tuple(step['cursor'])]
            assert len(matching)==1,(row['source_sha256'],step)
            choice=matching[0]
            active='active' if a.active(w) else 'post'
            if row['directive']=='none':active='post'
            turn=lambda c:w.last_dir[0]*c.action[2]-w.last_dir[1]*c.action[1]
            fav=lambda c:c.legal and c.action[0]=='INK' and turn(c)*w.personality>0
            opposing=lambda c:c.legal and c.action[0]=='INK' and turn(c)*w.personality<0
            if any(fav(c) for c in cs):
                d['preferred_available_'+active]+=1
                d['preferred_chosen_'+active]+=int(fav(choice))
            if any(fav(c) for c in cs) and any(opposing(c) for c in cs):
                d['eligible_'+active]+=1
                d['preferred_'+active]+=int(fav(choice))
                d['opposed_'+active]+=int(opposing(choice))
            assert w.apply(choice) and w.invalid_attempts==0
            a.advance(w)
        assert bool(w.distance()==0)==row['reached']
        assert sha(sorted(w.inked))==row['ink_sha256']
        replayed+=1
    for d in stats.values():
        for arm in ('active','post'):
            d['preferred_given_available_'+arm]=round(d['preferred_chosen_'+arm]/max(1,d['preferred_available_'+arm]),6)
            d['preferred_given_two_sided_'+arm]=round(d['preferred_'+arm]/max(1,d['eligible_'+arm]),6)
    return {'replayed_episodes':replayed,'strata':stats}

def run(archive,out):
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')
    paths=[archive/'prerequisites'/('guidance_recovery_'+n+'.safetensors') for n in names]
    assert all(p.exists() for p in paths)
    prior=load_c12(paths)
    # Source families distinct from validation and independent cohorts.
    X,A,W,G,receipts,train_stats=collect(prior,n=18,seed=220201,horizon=20,repeats=2)
    source_input=hashlib.sha256(X.tobytes()+A.tobytes()+W.tobytes()+G.tobytes()).hexdigest()
    models={'C12':Prior32(prior),'C18-A0':a0(prior)}
    c20=ReturnAnchoredGate(prior,width=12,scale=.8,variant='full')
    c20.head.load_state_dict(load_file(str(archive/'results/C20-STYLE.safetensors')))
    models['C20-STYLE']=c20.eval()
    results={}
    for variant in ('return','opportunity'):
        trained,updates,losses,eligible=optimize(load_c12(paths),X,A,W,G,variant)
        label='C21-'+variant.upper();models[label]=trained
        provenance={'schema':SCHEMA,'variant':variant,'input_sha256':source_input,
           'predecessors':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)},
           'seed':210221,'optimizer_updates':updates}
        checkpoint=out/(label+'.safetensors')
        save_file({k:v.detach().cpu().contiguous() for k,v in trained.head.state_dict().items()},
                 str(checkpoint),metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
        results[label]={'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'updates':updates,'objective_start':losses[0],'objective_end':losses[-1],
            'fav_ink_training_menus':eligible,'losses':losses}
    val=evaluate(models,n=4,seed=230201,horizon=40)
    test=evaluate(models,n=9,seed=240201,horizon=56)
    for key,rows in (('validation',val),('heldout',test)):
        (out/(key+'_trajectories.json')).write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    tested=trace_opportunity_audit(test,240201)
    fixed=fixed_state_audit(models,test,seed=240201)
    # All C21 cue-off must remain literally identical to C12.
    uncueds={}
    for m in ('C21-RETURN','C21-OPPORTUNITY'):
        a=sorted((r for r in test if r['model']=='C12' and r['directive']=='none'),key=lambda r:(r['source_sha256'],r['personality'],r['arm']))
        b=sorted((r for r in test if r['model']==m and r['directive']=='none'),key=lambda r:(r['source_sha256'],r['personality'],r['arm']))
        uncueds[m]={'pairs':len(a),'exact':sum(r['trace']==s['trace'] and r['ink_sha256']==s['ink_sha256'] for r,s in zip(a,b))}
        assert uncueds[m]['pairs']==uncueds[m]['exact']
    assert all(not r['host_rejected'] and not r['bad'] for r in test)
    metrics={'schema':SCHEMA,'training':{'menus':len(X),'host_executed_suffixes':len(X)*4,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'input_sha256':source_input,'stats':train_stats,'seed':220201},
       'checkpoints':results,'validation_seed':230201,'heldout_seed':240201,
       'validation_episodes':len(val),'heldout_episodes':len(test),
       'validation':summary(val),'heldout':summary(test),
       'fixed_state':fixed,'long_horizon_opportunity':tested,'uncued':uncueds,
       'host_rejected':sum(r['host_rejected'] for r in test),
       'illegal_unmasked_top1':sum(r['bad'] for r in test)}
    (out/'metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'training_menus':len(X),'checkpoint':results,
      'heldout_episodes':len(test),'cue_off_exact':uncueds},sort_keys=True))
    return metrics

if __name__=='__main__':run(sys.argv[1],sys.argv[2])

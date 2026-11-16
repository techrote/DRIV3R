"""C20: bounded guidance correction with an explicit drawing-personality cost.

External, synthetic game-policy research only. Does not edit or load the game.
This standalone experiment uses the archived C19 matched host-executed suffix
rollouts as training labels, and new trajectory families for evaluation.
"""
from __future__ import annotations
import hashlib, json, sys, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file, load_file
from c19_return_anchored import collect, evaluate, a0, summary
from return_anchored_core import ReturnAnchoredGate
from attested_stack_cue import Prior32, KEYS
from personality_retention_core import load_c12

SCHEMA='driv3r-guidance-c20-personality-constrained/1'

def opportunities(x):
    """Count host-legal, personality-favoured INK turns, not future labels."""
    ink = x[:,:4,:]
    return ((ink[:,:,4]>.5)&(ink[:,:,21]>.5)).sum(axis=1).astype(np.float32)

def optimize(prior,X,A,W,G,kind,seed=20261120,epochs=32):
    """Train a BACKTRACK-only residual using observed full-episode returns.

    STYLE differs from RETURN only by a local, non-privileged turn-opportunity
    preference in low-advantage ties; it is not supplied the test maps.
    """
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=ReturnAnchoredGate(prior,width=12,scale=.8,variant='full')
    xx=torch.from_numpy(X);adv=torch.from_numpy(A);wt=torch.from_numpy(W)
    groups=torch.from_numpy(G);opp=torch.from_numpy(opportunities(X))
    # C18-A0 has exactly +1 BACKTRACK logit under the same host-attested cue.
    with torch.no_grad():
        anchor=a0(prior)(xx).masked_fill(xx[:,:,4]<.5,-1e7)
        base_margin=anchor[:,8]-torch.logsumexp(torch.cat((anchor[:,:8],anchor[:,9:]),1),1)
        p0=torch.sigmoid(base_margin)
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=.0012,weight_decay=.01)
    rng=np.random.default_rng(seed);losses=[];updates=0
    for _ in range(epochs):
        order=rng.permutation(len(xx));total=0.
        for start in range(0,len(order),64):
            index=order[start:start+64];x=xx[index];y=adv[index];w=wt[index];g=groups[index]
            scores=model(x).masked_fill(x[:,:,4]<.5,-1e7)
            margin=scores[:,8]-torch.logsumexp(torch.cat((scores[:,:8],scores[:,9:]),1),1)
            p=torch.sigmoid(margin)
            head=model.head(x[:,8,:]).flatten();change=model.scale*torch.tanh(head)
            # On-policy A0 counterfactual return advantage; positive means BACKTRACK.
            improvement=-(p-p0[index])*y*w
            worst=torch.stack([improvement[g==k].mean() for k in range(8) if bool((g==k).any())]).max()
            loss=.75*improvement.mean()+.25*worst+1.7*(p-p0[index]).square().mean()+.08*change.square().mean()
            if kind=='style':
                # Preference to leave room for characteristic turns when returns
                # are near-tied. Strong evidence of recovery overrides this cost.
                tie=torch.exp(-2.5*y.abs())
                signed=(change.clamp(min=0)*(.5+opp[index]/3.))
                loss=loss+.28*(tie*signed.square()).mean()
            loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(index);updates+=1
        losses.append(round(total/len(xx),6))
    return model.eval(),updates,losses

def style_summary(rows):
    """Aggregate signed turns AND opportunity-normalized counts per condition."""
    labels=sorted({r['model'] for r in rows});out={}
    for outcome in ('reverse','ahead','shortcut','bridge'):
      for directive in ('none','backtrack','continue'):
        result={}
        for label in labels:
          group=[r for r in rows if r['model']==label and r['outcome']==outcome and r['directive']==directive]
          person={}
          for p in (-1,1):
            a=[r for r in group if r['personality']==p]
            left=sum(r['left'] for r in a);right=sum(r['right'] for r in a)
            person[str(p)]={'left':left,'right':right,'turn_events':left+right,
               'signed_bias':round((left-right)/max(1,left+right),5)}
          result[label]={'goal_success':sum(r['reached'] for r in group),'n':len(group),
            'mean_steps':round(float(np.mean([r['steps'] for r in group])),3),
            'personality_turn_gap':round(person['1']['signed_bias']-person['-1']['signed_bias'],5),
            'turn_events':sum(z['turn_events'] for z in person.values()),
            'person':person}
        out[f'{outcome}/{directive}']=result
    return out

def run(archive_root,out):
    root=Path(archive_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[root/'prerequisites'/('guidance_recovery_'+k+'.safetensors') for k in ()]
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
            'persistent_feedback','signed_feedback','selective_continue')
    paths=[root/'prerequisites'/('guidance_recovery_'+n+'.safetensors') for n in names]
    assert all(x.is_file() for x in paths)
    prior=load_c12(paths)
    X,A,W,G,receipts,stats=collect(prior,n=20,seed=209100,horizon=24,repeats=2)
    input_sha=hashlib.sha256(X.tobytes()+A.tobytes()+W.tobytes()+G.tobytes()).hexdigest()
    train_meta={'menus':len(X),'counterfactual_suffixes':len(X)*4,'comparison':stats,
         'input_sha256':input_sha,'training_seed':209100,
         'prerequisite_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)}}
    models={'C12':Prior32(prior),'C18-A0':a0(prior)};checkpoints={}
    for kind in ('return','style'):
        trained,updates,losses=optimize(load_c12(paths),X,A,W,G,kind)
        label='C20-'+kind.upper();models[label]=trained
        checkpoint=out/(label+'.safetensors')
        meta={'schema':SCHEMA,'model':label,'training_input_sha256':input_sha,'steps':updates,
            'prerequisite_sha256':train_meta['prerequisite_sha256']}
        save_file({k:v.detach().cpu().contiguous() for k,v in trained.head.state_dict().items()},
            str(checkpoint),metadata={'provenance':json.dumps(meta,sort_keys=True,separators=(',',':'))})
        checkpoints[label]={'updates':updates,'loss_first':losses[0],
             'loss_final':losses[-1],'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    # Validation and independent heldout sources are disjoint from collection.
    validation=evaluate(models,n=3,seed=211000,horizon=40)
    heldout=evaluate(models,n=8,seed=221000,horizon=48)
    for label,data in (('validation',validation),('heldout',heldout)):
        (out/(label+'_trajectories.json')).write_text(json.dumps(data,sort_keys=True,separators=(',',':'))+'\n')
    metrics={'schema':SCHEMA,'training':train_meta,'checkpoints':checkpoints,
       'validation_seed':211000,'heldout_seed':221000,'validation_episodes':len(validation),
       'heldout_episodes':len(heldout),'validation':style_summary(validation),
       'heldout':style_summary(heldout),'host_rejected':sum(r['host_rejected'] for r in heldout),
       'illegal_raw_top1':sum(r['bad'] for r in heldout),
       'cue_off_exact':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for label in ('C20-RETURN','C20-STYLE'):
        old=sorted((r for r in heldout if r['model']=='C12' and r['directive']=='none'),key=lambda r:(r['source_sha256'],r['personality'],r['arm']))
        new=sorted((r for r in heldout if r['model']==label and r['directive']=='none'),key=lambda r:(r['source_sha256'],r['personality'],r['arm']))
        metrics['cue_off_exact'][label]={'pairs':len(old),'matched':sum(a['trace']==b['trace'] and a['ink_sha256']==b['ink_sha256'] for a,b in zip(old,new))}
    (out/'metrics.json').write_text(json.dumps(metrics,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'checkpoints':checkpoints,'heldout_episodes':len(heldout),
        'cue_off_exact':metrics['cue_off_exact']},sort_keys=True))
    return metrics

if __name__=='__main__':
    run(sys.argv[1],sys.argv[2])

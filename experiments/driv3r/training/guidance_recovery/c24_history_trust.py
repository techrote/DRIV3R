"""C24: source-history-conditioned host cue trust, external synthetic CyberSand-like research.

Memory is constructed from *completed, host-executed paired calibration missions*,
never from the query mission's hidden map/goal outcome. A source can issue
mostly helpful or mostly misleading cues. Frozen C12 + C18-A0 is retained;
only BACKTRACK logits under active advisory may change. Host legality wins.
The synthetic audit/replay protocol is NOT the native CyberSand observation ABI.
"""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import ARMS,sha
from attested_stack_cue import observe32,KEYS,Prior32
from signed_feedback import HostAdvice
from personality_retention_core import load_c12
from c19_return_anchored import a0,run_suffix,make_world
from c22_branch_personality import world,softmax,fav_opposition

SCHEMA='driv3r-guidance-c24-history-trust/1'
CUES=('none','backtrack','continue')
MODES=('mostly_correct','mostly_misleading')


def seed_world(i,outcome,person,seed,train):
    if outcome in ('shortcut','bridge'):
        return make_world(i,outcome,person,seed,train=train)
    family=('hairpin','staircase')[i%2] if train else ('hook','zigzag')[i%2]
    return world(seed,family,person,outcome,i%4,bool((i//4)%2),1 if train else 2,13*i)


def source_directive(outcome,mode,rng):
    """Research distribution only: source has latent accuracy, NOT passed to policy."""
    nominal='backtrack' if outcome=='reverse' else 'continue'
    valid=bool(rng.random() < (.8 if mode=='mostly_correct' else .2))
    directive=nominal if valid else ('continue' if nominal=='backtrack' else 'backtrack')
    return directive,valid


def observed_calibration(prior,source_seed,mode,train=True,count=6,horizon=20):
    """No private target labels; host executes BOTH paired historical actions.

    The explicit audit is synthetic replay of earlier completed missions. A
    deployable ABI would have to attest this feedback; no game-side change made.
    """
    baseline=a0(prior);rng=np.random.default_rng(source_seed)
    evidence=[]
    for j in range(count):
        outcome=('reverse','ahead','bridge','shortcut')[j%4]
        w=seed_world(j,outcome,(-1 if j%2 else 1),source_seed+10091*j,train)
        a=HostAdvice(w);directive,nominal=source_directive(outcome,mode,rng)
        getattr(a,'issue_'+directive)(w,strength=1.,ttl=9)
        menu,x=observe32(w,1.,a)
        if not menu[8].legal: continue
        with torch.no_grad():score=baseline(torch.from_numpy(x)).numpy()
        legal_forward=[k for k in range(8) if menu[k].legal]
        if not legal_forward:continue
        forward=max(legal_forward,key=lambda k:score[k])
        pair_seed=source_seed+j*3137+711
        rback,doneback,badback,_=run_suffix(baseline,w,a,1.,8,pair_seed,horizon)
        rfwd,donefwd,badfwd,_=run_suffix(baseline,w,a,1.,forward,pair_seed,horizon)
        assert not (badback or badfwd)
        # Did *following the previous host directive* outperform disregarding it?
        signed=(rback-rfwd)*(1 if directive=='backtrack' else -1)
        evidence.append({'directive':directive,'observed_follow_advantage':round(float(signed),7),
                         'follow_completed':int(doneback if directive=='backtrack' else donefwd),
                         'ignore_completed':int(donefwd if directive=='backtrack' else doneback),
                         'source_state_sha256':sha({'pos':w.pos,'goal':w.goal,'trail':w.history,
                                                    'protected':sorted(w.protected)}),
                         'nominal_valid_generator_only':bool(nominal)})
    vals=np.asarray([e['observed_follow_advantage'] for e in evidence],np.float64)
    per=[]
    for directive in ('backtrack','continue'):
        v=np.asarray([e['observed_follow_advantage'] for e in evidence if e['directive']==directive])
        per.append(float(np.tanh(v.mean())) if len(v) else 0.)
    # Four bounded, audited historical features; target outcome is absent.
    features=np.asarray([float(np.tanh(vals.mean())) if len(vals) else 0.,
                         float((np.sum(vals>0)-np.sum(vals<0))/max(1,len(vals))),
                         per[0],per[1]],np.float32)
    return features,evidence


def observe36(w,arm,a,memory):
    menu,x=observe32(w,arm,a)
    assert np.asarray(memory).shape==(4,)
    z=np.concatenate((x,np.tile(np.asarray(memory,np.float32),(10,1))),axis=-1)
    assert z.shape==(10,36) and np.array_equal(z[:,:32],x)
    return menu,z


class TrustAdapter(nn.Module):
    """Frozen A0 plus bounded correction to BACKTRACK alone during advice.

    A no-memory ablation gets the identical local cue/geometry features but a
    zero history vector. No-cue is EXACT frozen C12, including all INK scores.
    """
    def __init__(self,prior,history=True,scale=1.25):
        super().__init__();self.prior=a0(prior)
        for p in self.prior.parameters():p.requires_grad_(False)
        self.history=history;self.scale=float(scale)
        self.head=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        assert x.shape[-2:]==(10,36)
        baseline=self.prior(x[...,:32]);out=baseline.clone();q=x[...,8,:]
        history=q[...,32:36] if self.history else torch.zeros_like(q[...,32:36])
        f=torch.cat([q[...,26:28],q[...,29:31],history],dim=-1)
        delta=self.scale*torch.tanh(self.head(f).squeeze(-1))
        active=q[...,2]*q[...,4]*(q[...,26].abs()>.5).to(q.dtype)*q[...,27]
        out[...,8]=baseline[...,8]+active*delta
        return out


class Baseline36(nn.Module):
    def __init__(self,prior,use_a0=True):
        super().__init__();self.policy=a0(prior) if use_a0 else Prior32(prior)
    def forward(self,x):return self.policy(x[...,:32])


def fork_targets(prior,n=14,seed=241117,repeats=2,horizon=22):
    base=Baseline36(prior,True);xs=[];adv=[];groups=[];records=[];calibrations={}
    for i in range(n):
      mode=MODES[i%2];source=seed+17011*i
      memory,evidence=observed_calibration(prior,source,mode,train=True)
      calibrations[str(source)]={'source_mode_generator_only':mode,'memory':memory.tolist(),'audits':evidence}
      rng=np.random.default_rng(source+313)
      for outcome in ('reverse','ahead','shortcut','bridge'):
       for person in (-1,1):
        for arm in ARMS:
         directive,nominal=source_directive(outcome,mode,rng)
         w=seed_world(i,outcome,person,source+100000*('reverse','ahead','shortcut','bridge').index(outcome)+person*73,True)
         a=HostAdvice(w);getattr(a,'issue_'+directive)(w,strength=(.35 if person<0 else 1.),ttl=9)
         menu,x=observe36(w,arm,a,memory)
         with torch.no_grad():scores=base(torch.from_numpy(x)).numpy()
         forwards=[k for k in range(8) if menu[k].legal]
         if not menu[8].legal or not forwards:continue
         fwd=max(forwards,key=lambda k:scores[k])
         vals=[]
         for j in range(repeats):
          rs=source+7919*j+('reverse','ahead','shortcut','bridge').index(outcome)*571+int(arm*100)
          rb,_,bad1,_=run_suffix(base.policy,w,a,arm,8,rs,horizon)
          rf,_,bad2,_=run_suffix(base.policy,w,a,arm,fwd,rs,horizon)
          assert bad1==bad2==0
          vals.append(rb-rf)
         xs.append(x);adv.append(float(np.mean(vals)))
         groups.append(f'{mode}/{outcome}/{directive}')
         records.append({'source_seed':source,'world_source_sha256':sha({'pos':w.pos,'goal':w.goal,'protected':sorted(w.protected),'trail':w.history}),
                         'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                         'outcome_generator_only':outcome,'directive':directive,'nominal_valid_generator_only':nominal,
                         'arm':float(arm),'personality':person,'return_back_minus_forward':vals,
                         'memory':memory.tolist()})
    return np.asarray(xs,np.float32),np.asarray(adv,np.float32),groups,records,calibrations


def optimize(prior,x,adv,groups,history=True,epochs=20,seed=241101):
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=TrustAdapter(prior,history=history)
    xx=torch.from_numpy(x);aa=torch.from_numpy(adv).clamp(-3,3)
    with torch.no_grad():
        old=softmax(Baseline36(prior)(xx),xx)
        p0=old[:,8]
    counts={g:groups.count(g) for g in set(groups)}
    wt=torch.tensor([1/max(1,counts[g]) for g in groups],dtype=torch.float32)
    wt=wt/wt.mean()
    opt=torch.optim.AdamW(model.head.parameters(),lr=.002,weight_decay=.01)
    rng=np.random.default_rng(seed);hist=[];updates=0
    for epoch in range(epochs):
      order=rng.permutation(len(xx));total=0
      for st in range(0,len(xx),64):
        sel=order[st:st+64];p=softmax(model(xx[sel]),xx[sel]);pb=p[:,8].clamp(1e-5,1-1e-5)
        # Advantage is from host-EXECUTED futures, not a shortest-path oracle.
        return_loss=-(wt[sel]*aa[sel]*(pb-p0[sel])).mean()
        drift=(pb-p0[sel]).square().mean()
        loss=return_loss+1.5*drift+.0005*sum(z.square().mean() for z in model.head.parameters())
        loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.)
        opt.step();opt.zero_grad(set_to_none=True);updates+=1;total+=float(loss.detach())*len(sel)
      hist.append(round(total/len(xx),7))
    return model.eval(),updates,hist


def rollout(policy,prior,source,mode,outcome,person,arm,directive,strength,seed,train=False,horizon=44,trace=False,mem=None):
    w=seed_world(source%50,outcome,person,seed,train)
    a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    if mem is None:mem,_=observed_calibration(prior,source,mode,train=train)
    rng=np.random.default_rng(seed+int(arm*100)*97+int(strength*100)*61+{'none':0,'backtrack':313,'continue':727}[directive])
    initial=sha({'pos':w.pos,'goal':w.goal,'trail':w.history,'protected':sorted(w.protected)})
    geometry={'start':w.pos,'goal':w.goal,'protected':sorted(w.protected),
              'initial_ink':sorted(w.inked)} if trace else None
    protected=sha(sorted(w.protected));ink=set(w.inked);actions=[];path=[];bad=0;back=0
    fav_chosen=0;fav_opportunities=0;first_favmass=None
    for tick in range(horizon):
        menu,x=observe36(w,arm,a,mem)
        with torch.no_grad():s=policy(torch.from_numpy(x)).numpy().astype(np.float64)
        legal=np.asarray([c.legal for c in menu]);bad+=int(not legal[s.argmax()])
        s[~legal]=-1e8;p=np.exp((s-s.max())/.8);p/=p.sum()
        choice=int(rng.choice(10,p=p));c=menu[choice]
        f,o=fav_opposition(x)
        if tick==0 and a.active(w) and f.any() and o.any():first_favmass=float(p[:8][f].sum())
        if bool(a.active(w)) and f.any() and o.any():
            fav_opportunities+=1;fav_chosen+=int(choice<8 and f[choice])
        assert c.legal and w.apply(c) and not w.invalid_attempts
        assert ink<=w.inked and protected==sha(sorted(w.protected))
        back+=int(c.action[0]=='BACKTRACK')
        actions.append((c.action,c.target));a.advance(w)
        if trace:path.append({'t':tick,'pos':w.pos,'action':c.action,'distance':w.distance(),'advice':directive})
        if w.distance()==0 or w.stopped:break
    return {'reached':int(w.distance()==0),'steps':tick+1,'distance':w.distance(),
            'backtracks':back,'host_rejected':w.invalid_attempts,'bad':bad,
            'source_sha256':initial,'protected_sha256':protected,'actions_sha256':sha(actions),
            'ink_sha256':sha(sorted(w.inked)),'fav_ink':fav_chosen,'fav_opportunities':fav_opportunities,
            'first_favmass':first_favmass,'trace':path,'geometry':geometry}


def evaluate(models,prior,n=6,seed=241999,horizon=44,trace_n=28):
    """Crossed adversarial arms PLUS source-issued matched source-history arms."""
    assert n>=4, 'Evaluation needs both source reliability regimes'
    rows=[];history=[]
    cache=[]
    for i in range(n):
        mode=MODES[(i//2)%2];source=seed+11083*i
        mem,evidence=observed_calibration(prior,source,mode,train=False)
        cache.append((mem,evidence))
    for i in range(n):
      mode=MODES[(i//2)%2];source=seed+11083*i
      mem,evidence=cache[i]
      # Wrong-source audited history is an explicit ablation, chosen from a
      # different latent-quality regime; source mode is NEVER passed to policy.
      wrong_sources=[j for j in range(n) if j!=i and MODES[(j//2)%2]!=mode]
      shuffled_memory=cache[wrong_sources[i%len(wrong_sources)]][0]
      history.append({'source':source,'mode_generator_only':mode,'memory':mem.tolist(),'audits':evidence})
      for outcome in ('reverse','ahead','bridge','shortcut'):
       for person in (-1,1):
        s=source+100000*('reverse','ahead','bridge','shortcut').index(outcome)+131*person
        chosen,nominal=source_directive(outcome,mode,np.random.default_rng(s+9593))
        for arm in ARMS:
         # Fully crossed hostile directives expose condition-level regressions.
         challenges=[('crossed',d,st,None) for d in CUES
                     for st in ((0.,) if d=='none' else (.35,1.))]
         # Realistic source-issued directives are correlated with past audit
         # outcomes, but target outcome/validity is never given to the model.
         challenges += [('source_issued',chosen,st,nominal) for st in (.35,1.)]
         for selection,directive,strength,nominal_valid in challenges:
          for name,p in models.items():
            rec=rollout(p,prior,source,mode,outcome,person,arm,directive,strength,s,
                        train=False,horizon=horizon,trace=len(rows)<trace_n*len(models),
                        mem=shuffled_memory if name=='C24-SHUFFLE' else mem)
            rec.update({'model':name,'source':source,'mode_generator_only':mode,
                        'outcome':outcome,'personality':person,'arm':float(arm),
                        'directive':directive,'strength':strength,'seed':s,'selection':selection,
                        'nominal_valid_generator_only':nominal_valid})
            rows.append(rec)
    return rows,history


def summarize(rows):
    out={}
    for selection in ('crossed','source_issued'):
     for mode in MODES:
      for outcome in ('reverse','ahead','bridge','shortcut'):
       for directive in CUES:
        for strength in ((0.,) if directive=='none' else (.35,1.)):
         key=f'{selection}/{mode}/{outcome}/{directive}/{strength:.2f}'
         grp=[r for r in rows if (r['selection'],r['mode_generator_only'],r['outcome'],r['directive'],r['strength'])==(selection,mode,outcome,directive,strength)]
         if not grp:continue
         out[key]={}
         for name in sorted({r['model'] for r in grp}):
            zz=[r for r in grp if r['model']==name]
            eligible=[r['first_favmass'] for r in zz if r['first_favmass'] is not None]
            out[key][name]={'goals':sum(r['reached'] for r in zz),'n':len(zz),
                            'mean_end_distance':round(float(np.mean([r['distance'] for r in zz])),4),
                            'mean_success_steps':round(float(np.mean([r['steps'] for r in zz if r['reached']])),3) if any(r['reached'] for r in zz) else None,
                            'first_fav_mass':round(float(np.mean(eligible)),6) if eligible else None,
                            'fav_ink':sum(r['fav_ink'] for r in zz),'fav_opportunities':sum(r['fav_opportunities'] for r in zz)}
    return out


def validate(models,prior,rows):
    for name in ('C24-TRUST','C24-NOMEM'):
       prior_rows={(r['source'],r['seed'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength']):r for r in rows if r['model']=='C12' and r['directive']=='none'}
       selected=[r for r in rows if r['model']==name and r['directive']=='none']
       assert all(r['actions_sha256']==prior_rows[(r['source'],r['seed'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength'])]['actions_sha256'] and r['ink_sha256']==prior_rows[(r['source'],r['seed'],r['personality'],r['outcome'],r['arm'],r['directive'],r['strength'])]['ink_sha256'] for r in selected)
    assert not any(r['host_rejected'] or r['bad'] for r in rows)
    grouped=summarize(rows);results={}
    for name in ('C24-TRUST','C24-NOMEM','C24-SHUFFLE'):
      regress=[];better=[]
      for key,by in grouped.items():
       if '/none/' in key:continue
       if by[name]['goals']<by['C18-A0']['goals']:regress.append([key,by['C18-A0']['goals'],by[name]['goals']])
       if by[name]['goals']>by['C18-A0']['goals']:better.append([key,by['C18-A0']['goals'],by[name]['goals']])
      results[name]={'strict_noregression_pass':not regress,'regressions':regress,'improvements':better}
    return results


def run(archive,out,train_n=14,eval_n=6,epochs=20,horizon=44,repeats=2,eval_seed=241999):
    torch.set_num_threads(1)
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')
    paths=[archive/'prerequisites'/('guidance_recovery_'+s+'.safetensors') for s in names]
    assert all(p.exists() for p in paths)
    prior=load_c12(paths)
    x,advantages,groups,trainrec,calibrations=fork_targets(prior,n=train_n,repeats=repeats)
    models={'C12':Baseline36(prior,False),'C18-A0':Baseline36(prior,True)}
    stats={};states={}
    sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    datahash=hashlib.sha256(x.tobytes()+advantages.tobytes()+json.dumps(groups).encode()).hexdigest()
    for history,name in ((True,'C24-TRUST'),(False,'C24-NOMEM')):
       new,updates,losses=optimize(prior,x,advantages,groups,history,epochs)
       models[name]=new
       if history:models['C24-SHUFFLE']=new
       path=out/(name+'.safetensors')
       save_file({k:v.detach().cpu().contiguous() for k,v in new.head.state_dict().items()},str(path),
          metadata={'provenance':json.dumps({'schema':SCHEMA,'source_sha256':sourcehash,
               'data_sha256':datahash,'history':history,'predecessor_sha256':
               {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)}},sort_keys=True)})
       stats[name]={'updates':updates,'loss_first':losses[0],'loss_last':losses[-1],
                    'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    rows,histories=evaluate(models,prior,n=eval_n,seed=eval_seed,horizon=horizon)
    qualifications=validate(models,prior,rows)
    report={'schema':SCHEMA,'source_sha256':sourcehash,'training_sha256':datahash,
            'prerequisite_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)},
            'training_menus':len(x),'paired_training_futures':len(x)*repeats*2,
            'training_advantages':{'positive':int(np.sum(advantages>.05)),
                   'negative':int(np.sum(advantages<-.05)),'near_tied':int(np.sum(np.abs(advantages)<=.05))},
            'training':stats,'eval_seed':eval_seed,'heldout_episodes':len(rows),'heldout_groups':summarize(rows),
            'qualifications':qualifications,'source_modes':MODES,
            'rejection':'Any guided recovery regression vs C18-A0 rejects checkpoint.',
            'limitations':'Synthetic host and synthetic replay audits; not native CyberSand host ABI or live gameplay evidence.'}
    for n,data in [('train_fork_receipts.json',trainrec),('calibration_receipts.json',calibrations),
                   ('heldout_calibration_receipts.json',histories),('heldout_trajectories.json',rows),
                   ('metrics.json',report)]:
        (out/n).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    if len(sys.argv)<3:raise SystemExit('Usage: python c24_history_trust.py ARCHIVE_DIR OUTPUT [train_n eval_n epochs horizon repeats eval_seed]')
    result=run(sys.argv[1],sys.argv[2],*[int(v) for v in sys.argv[3:]])
    print(json.dumps({'train_menus':result['training_menus'],'training':result['training'],
         'heldout_episodes':result['heldout_episodes'],'qualifications':result['qualifications']},sort_keys=True))

"""C23: host-executed return-constrained, opportunity-conditioned personality.

Synthetic CyberSand-like research only. Host owns legality and immutable materials;
no game-side changes. Frozen C12 / C18-A0, C22-STRONG as independent control.
Train a bounded legal INK sidecar on paired *executed* suffix returns and
personality preservation with a held-out-group nonregression acceptance gate.
"""
from __future__ import annotations
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from experiment import ARMS,sha
from signed_feedback import HostAdvice
from attested_stack_cue import observe32,Prior32,KEYS
from personality_retention_core import load_c12
from c19_return_anchored import a0
from c22_branch_personality import (world,StyleSidecar,fav_opposition,softmax,
                                   eval_rollouts,metrics,guards,assemble,state_audit)

SCHEMA='driv3r-guidance-c23-joint-returns/1'


def baseline_c22(prior,archive):
    model=StyleSidecar(a0(prior),scale=1.2)
    model.head.load_state_dict(load_file(str(Path(archive)/'results/C22-STRONG.safetensors')))
    return model.eval()


def legal_sample(policy,w,a,arm,rng):
    menu,x=observe32(w,arm,a)
    with torch.no_grad():s=policy(torch.from_numpy(x)).numpy().astype(np.float64)
    valid=np.asarray([c.legal for c in menu],dtype=bool)
    bad=int(not valid[int(np.argmax(s))]);s[~valid]=-1e8
    p=np.exp((s-s.max())/.8);p/=p.sum()
    j=int(rng.choice(len(menu),p=p))
    assert menu[j].legal and w.apply(menu[j]) and w.invalid_attempts==0
    a.advance(w)
    return bad


def suffix(policy,w,a,arm,first,seed,horizon=24):
    """Matched host-executed return, preserving identical random draws by seed."""
    v,b=copy.deepcopy((w,a));initial=v.distance();startink=set(v.inked)
    prot=sha(sorted(v.protected));menu,_=observe32(v,arm,b)
    assert menu[first].legal and v.apply(menu[first]) and not v.invalid_attempts
    b.advance(v);steps=1;bad=0
    rng=np.random.default_rng(seed)
    while steps<horizon and not (v.distance()==0 or v.stopped):
        bad+=legal_sample(policy,v,b,arm,rng);steps+=1
    assert not bad and v.invalid_attempts==0 and prot==sha(sorted(v.protected))
    assert startink.issubset(v.inked)
    progress=(initial-v.distance())/max(1,initial)
    # Dominant goal completion, progress and bounded step cost; no oracle hints.
    ret=3*int(v.distance()==0)+.65*progress-.2*steps/horizon
    return ret, int(v.distance()==0), v.distance(), steps


def collect(prior,n=7,seed=231117,replicates=2,horizon=24,max_samples=480):
    """Two on-policy, legal INK forks from the *same* synthetic observed state.

    Train only on hairpin/staircase; frozen C18-A0 executes suffixes. Four
    arms (no cue is excluded from training since sidecar is exactly disabled).
    """
    base=a0(prior);xs=[];av=[];groups=[];receipts=[]
    sampled=0;forked=0;fav_wins=0;opp_wins=0
    for i in range(n):
      fam=('hairpin','staircase')[i%2];rot=(i//2)%4;mirror=bool((i//4)%2)
      for outcome in ('reverse','ahead'):
       for person in (-1,1):
        s=seed+7777*i+(100000 if outcome=='ahead' else 0)+(500000 if person==1 else 0)
        for arm in ARMS:
         for directive in ('backtrack','continue'):
          for strength in (.35,1.):
           w=world(s,fam,person,outcome,rot,mirror,1,i*13);a=HostAdvice(w)
           getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
           rng=np.random.default_rng(s+int(arm*100)*47+(0 if directive=='backtrack' else 911))
           captured=0
           for t in range(10):
            menu,x=observe32(w,arm,a)
            fav,opp=fav_opposition(x)
            if a.active(w) and fav.any() and opp.any() and sampled<max_samples and captured<1:
              with torch.no_grad():score=base(torch.from_numpy(x)).numpy()
              jf=max(np.flatnonzero(fav),key=lambda k:score[k]);jo=max(np.flatnonzero(opp),key=lambda k:score[k])
              y=[];done=[]
              for rep in range(replicates):
               rs=s+rep*16127+t*173+int(arm*100)*31
               f=suffix(base,w,a,arm,int(jf),rs,horizon)
               o=suffix(base,w,a,arm,int(jo),rs,horizon)
               y.append(f[0]-o[0]);done.append([f[1],o[1]])
               forked+=2
              advantage=float(np.mean(y));fav_wins+=int(advantage>.05);opp_wins+=int(advantage<-.05)
              xs.append(x);av.append(advantage)
              group=f'{outcome}/{directive}/{strength:.2f}/{person:+d}'
              groups.append(group)
              receipts.append({'seed':s,'family':fam,'outcome':outcome,'personality':person,
               'arm':float(arm),'directive':directive,'strength':strength,'tick':t,
               'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
               'source_sha256':sha({'pos':w.pos,'goal':w.goal,'trail':w.history,
                                   'protected':sorted(w.protected),'ink':sorted(w.inked)}),
               'legal_favourite':int(jf),'legal_opposition':int(jo),
               'return_diffs':y,'goal_reached_forks':done,'advantage':advantage})
              sampled+=1;captured+=1
            if w.distance()==0 or w.stopped:break
            legal_sample(base,w,a,arm,rng)
    assert sampled and len(groups)==len(xs)
    return np.asarray(xs,np.float32),np.asarray(av,np.float32),groups,receipts,{
        'menus':sampled,'paired_suffixes':forked,'positive_advantages':fav_wins,
        'negative_advantages':opp_wins,'zeroish_advantages':sampled-fav_wins-opp_wins}


def optimize(prior,archive,x,advantages,groups,variant='guard',epochs=20,seed=231191):
    assert variant in ('guard','return')
    torch.manual_seed(seed);torch.set_num_threads(1)
    model=baseline_c22(prior,archive)
    # Safety guard: keep the C22 1.2 legal fav INK cap for the new policy.
    for p in model.head.parameters():p.requires_grad_(True)
    xx=torch.from_numpy(x);adv=torch.from_numpy(advantages).clamp(-2,2)
    fav,opp=fav_opposition(xx)
    with torch.no_grad():
       anchor=softmax(a0(prior)(xx),xx)
       oldstyle=softmax(model(xx),xx)
       native=softmax(Prior32(prior)(xx),xx)
       target=torch.minimum((native[:,:8]*fav).sum(1), (anchor[:,:8]*fav).sum(1)+.05)
    opt=torch.optim.AdamW(model.head.parameters(),lr=.001 if variant=='guard' else .002,
                         weight_decay=.015)
    rng=np.random.default_rng(seed);losses=[];updates=0
    # Per-group balancing prevents frequent/locally easy situations dominating.
    g=np.asarray(groups);uni=np.unique(g);weights=np.asarray([1./max(1,(g==v).sum()) for v in g])
    weights=weights/weights.mean()
    ww=torch.from_numpy(weights.astype(np.float32))
    for epoch in range(epochs):
      order=rng.permutation(len(xx));acc=0.
      for st in range(0,len(order),48):
       ix=order[st:st+48];z=xx[ix]
       p=softmax(model(z),z);pf=(p[:,:8]*fav[ix]).sum(1).clamp_min(1e-7)
       po=(p[:,:8]*opp[ix]).sum(1).clamp_min(1e-7)
       # Positive logged fork advantage pushes preferred action; negative
       # advantage penalises unsafe style, including misleading-cue cases.
       ret=-(ww[ix]*adv[ix]*(torch.log(pf)-torch.log(po))).mean()
       style=torch.relu(target[ix]-(p[:,:8]*fav[ix]).sum(1)).square().mean()
       drift=(p-anchor[ix]).square().sum(1).mean()
       prior_kl=(p*(p.clamp_min(1e-7).log()-oldstyle[ix].clamp_min(1e-7).log())).sum(1).mean()
       loss=((.025 if variant=='guard' else .055)*ret +
             (1.5 if variant=='guard' else .45)*style +
             (1.8 if variant=='guard' else .35)*drift +
             (1.5 if variant=='guard' else .35)*prior_kl)
       loss.backward();nn.utils.clip_grad_norm_(model.head.parameters(),1.)
       opt.step();opt.zero_grad(set_to_none=True);updates+=1;acc+=float(loss.detach())*len(ix)
      losses.append(round(acc/len(xx),8))
    return model.eval(),updates,losses


def acceptance(rows,audit):
    """Strict nonregression vs C18-A0 on each cue/outcome at each strength.

    No statistical promotion claims from a single finite held-out cohort.
    """
    grouped=metrics(rows);out={}
    for name in ('C23-GUARD','C23-RETURN'):
        fails=[];improvements=[]
        for key,by in grouped.items():
            if '/none/' in key:continue
            if 'C18-A0' not in by or name not in by:continue
            base=by['C18-A0']['goals'];now=by[name]['goals']
            if now<base:fails.append({'group':key,'baseline':base,'candidate':now})
            if now>base:improvements.append({'group':key,'baseline':base,'candidate':now})
        anchor=audit['models']['C18-A0']['fav_ink_mass'];mass=audit['models'][name]['fav_ink_mass']
        out[name]={'pass':not fails and mass>=anchor,'regressions':fails,
                   'improvements':improvements,'fav_mass_vs_a0':round(mass-anchor,6),
                   'retention_nonregression':mass>=anchor}
    return out


def run(archive,out,n=7,eval_n=8,epochs=20,horizon=48,replicates=2,eval_seed=341117):
    torch.set_num_threads(1);archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
           'persistent_feedback','signed_feedback','selective_continue')
    paths=[archive/'prerequisites'/('guidance_recovery_'+s+'.safetensors') for s in names]
    assert all(p.is_file() for p in paths)
    prior=load_c12(paths)
    x,adv,groups,receipts,stats=collect(prior,n=n,replicates=replicates)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data_hash=hashlib.sha256(x.tobytes()+adv.tobytes()+json.dumps(groups).encode()).hexdigest()
    models={'C12':Prior32(prior),'C18-A0':a0(prior),'C22-STRONG':baseline_c22(prior,archive)}
    trained={}
    for variant in ('guard','return'):
       candidate,updates,hist=optimize(load_c12(paths),archive,x,adv,groups,variant,epochs)
       assert guards(prior,candidate)
       label='C23-'+variant.upper();models[label]=candidate
       weights=out/(label+'.safetensors')
       metadata={'schema':SCHEMA,'variant':variant,'source_sha256':source_hash,'data_sha256':data_hash,
           'predecessors_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)},
           'c22_sha256':hashlib.sha256((archive/'results/C22-STRONG.safetensors').read_bytes()).hexdigest()}
       save_file({k:v.detach().cpu().contiguous() for k,v in candidate.head.state_dict().items()},str(weights),
           metadata={'provenance':json.dumps(metadata,sort_keys=True,separators=(',',':'))})
       trained[label]={'updates':updates,'loss_first':hist[0],'loss_last':hist[-1],
                       'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                       'loss_history':hist}
    held_x,held_receipts=assemble(prior,max(4,eval_n//2),331117,train=False)
    # Disjoint source/topology seeds: hook / zigzag held out, not used for optimization.
    rows=eval_rollouts(models,eval_n,eval_seed,horizon=horizon,trace_limit=32)
    assert not any(r['host_rejected'] or r['bad'] for r in rows)
    pair={}
    baseline={ (r['seed'],r['arm'],r['outcome'],r['personality']):r for r in rows
              if r['model']=='C12' and r['directive']=='none'}
    for name in ('C18-A0','C22-STRONG','C23-GUARD','C23-RETURN'):
       paired=[(baseline[(r['seed'],r['arm'],r['outcome'],r['personality'])],r)
               for r in rows if r['model']==name and r['directive']=='none']
       pair[name]={'exact':sum(a['actions_sha256']==b['actions_sha256'] and
                                a['ink_sha256']==b['ink_sha256'] for a,b in paired),'pairs':len(paired)}
       assert pair[name]['pairs']==pair[name]['exact']
    audit=state_audit(models,held_x);qualification=acceptance(rows,audit)
    report={'schema':SCHEMA,'source_sha256':source_hash,'training_sha256':data_hash,
       'training_stats':stats,'training':trained,'prerequisite_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)},
       'fixed_state':audit,'qualifications':qualification,'grouped_heldout':metrics(rows),
       'heldout_episode_count':len(rows),'heldout_same_state_menus':len(held_x),
       'cue_off_pairwise_exact':pair,'illegal_raw_top1':sum(r['bad'] for r in rows),
       'host_rejected':sum(r['host_rejected'] for r in rows),
       'evaluation_seed':eval_seed,'training_families':['hairpin','staircase'],
       'heldout_families':['hook','zigzag'],
       'scope':'External synthetic host: not native CyberSand ABI or runtime; never promote automatically.'}
    for filename,data in (('training_fork_receipts.json',receipts),('heldout_state_receipts.json',held_receipts),
                          ('heldout_trajectories.json',rows),('metrics.json',report)):
       (out/filename).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')
    return report

if __name__=='__main__':
    if len(sys.argv)<3:raise SystemExit('usage: python c23_constrained_return.py C22_ARCHIVE_DIR OUTPUT [train_n eval_n epochs horizon replicas evaluation_seed]')
    r=run(sys.argv[1],sys.argv[2],*[int(a) for a in sys.argv[3:]])
    print(json.dumps({'training':r['training_stats'],'checkpoints':r['training'],
                      'eval_episodes':r['heldout_episode_count'],'acceptance':r['qualifications']},sort_keys=True))

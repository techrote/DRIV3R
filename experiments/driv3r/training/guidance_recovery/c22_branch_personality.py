"""C22 branch-rich opportunity-conditioned personality under recoverable host advice.

External CyberSand-*like* synthetic world, NOT game ABI. Host owns legality,
materials and execution. Frozen C12 + C18-A0; only bounded legal INK favourite
logits can change, and only during active host guidance on two-sided menus.
Train/evaluate on disjoint topology families with seeded stochastic rollouts.
"""
from __future__ import annotations
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from experiment import ARMS,N,World,DIRECTIONS,sha
from topology_transfer import TRAILS,EXITS,SIDE,transform
from signed_feedback import HostAdvice
from attested_stack_cue import observe32,Prior32,KEYS
from c19_return_anchored import a0
from personality_retention_core import load_c12
from adversarial_cues import source_hash

SCHEMA='driv3r-guidance-c22-branch-personality/1'
FAMILIES_TRAIN=('hairpin','staircase')
FAMILIES_TEST=('hook','zigzag')
DIRECTIONS3=('none','backtrack','continue')

def world(seed, family, persona, outcome, rotation, mirror, breadth, vary):
    """Branch-rich protected-material world, current trail and actual reachable goal.

    Training: one-ring around trail with seeded side pockets. Held-out: two-ring
    with deliberately different barriers, rotations, reflections and shortcuts.
    No topology or forbidden coordinates are injected into policy observations.
    """
    path=TRAILS[family]; ex=EXITS[family]; tr=lambda p:transform(p,rotation,mirror)
    core=set(path)|set(ex)|set(SIDE[family]);allowed=set(core)
    for p in core:
        for dx,dy in DIRECTIONS:
            q=(p[0]+dx,p[1]+dy)
            if 1<=q[0]<=13 and 1<=q[1]<=13:allowed.add(q)
    if breadth==2:
        for p in tuple(allowed):
            for dx,dy in DIRECTIONS:
                q=(p[0]+dx,p[1]+dy)
                if 2<=q[0]<=12 and 2<=q[1]<=12:allowed.add(q)
        # Deterministic withheld slits: keep the original trail/exits open.
        allowed={p for p in allowed if p in core or ((p[0]*13+p[1]*7+vary)%17 not in (1,2))}
    else:
        # Keep the original legal corridor while altering side-pocket affordance.
        allowed={p for p in allowed if p in core or ((p[0]*11+p[1]*5+vary)%13!=0)}
    w=World(seed,'islands',persona)
    w.protected={(x,y) for x in range(N) for y in range(N)}-{tr(p) for p in allowed}
    w.history=[tr(p) for p in path[:-1]];w.pos=tr(path[-1]);w.goal=tr(path[0] if outcome=='reverse' else ex[-1])
    w.inked=set(w.history)|{w.pos};w.last_dir=(w.pos[0]-w.history[-1][0],w.pos[1]-w.history[-1][1])
    w.steps=len(path)+3;w.goal_change=True;w.goals_seen=2;w.stuck=2;w.previous_distance=w.distance()
    assert len(w.history)>3 and w.allowed(w.goal) and w.allowed(w.pos)
    return w

def fav_opposition(x):
    legal=x[...,:8,4]>.5;ink=x[...,:8,0]>.5
    fav=legal&ink&(x[...,:8,21]>.5)
    opposite=legal&ink&((x[...,:8,13]>.5)|(x[...,:8,14]>.5))&~fav
    return fav,opposite

class StyleSidecar(nn.Module):
    """Correction remains C18-A0; only a legal, preferred INK subset shifts.

    Bounded and explicitly stochastic: no hard action rule, legality override,
    material paint or backtracking. No-cue C12 action logits are bit-identical.
    """
    def __init__(self, frozen_prior, scale=.6, width=12):
        super().__init__();self.prior=frozen_prior
        for p in self.prior.parameters():p.requires_grad_(False)
        self.scale=float(scale)
        self.head=nn.Sequential(nn.Linear(32,width),nn.Tanh(),nn.Linear(width,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x):
        if x.shape[-2:]!=(10,32):raise ValueError('C22 expects a 10x32 menu')
        scores=self.prior(x);old=scores.clone()
        fav,op=fav_opposition(x)
        b=x[...,8,:]
        active=(b[...,26].abs()>.5).to(x.dtype)*b[...,27]*(fav.any(-1)&op.any(-1)).to(x.dtype)
        offset=self.scale*torch.tanh(self.head(x[...,:8,:]).squeeze(-1))
        old[...,:8]=scores[...,:8]+active[...,None]*fav.to(x.dtype)*offset
        return old

def softmax(scores,x):
    return torch.softmax(scores.masked_fill(x[...,4]<.5,-1e7)/.8,dim=-1)

def sim(policy,w,arm,directive,strength,rngseed,horizon=54,trace=False):
    a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    rng=np.random.default_rng(rngseed)
    outset=source_hash(w); protected=sha(sorted(w.protected)); ink=set(w.inked)
    steps=[];actions=[];preferred={'active':[0,0,0], 'post':[0,0,0]};bad=0;backtracks=0
    first_eligible=0
    for tick in range(horizon):
        menu,x=observe32(w,arm,a)
        with torch.no_grad():s=policy(torch.from_numpy(x)).numpy().astype(np.float64)
        legal=np.asarray([c.legal for c in menu]);bad+=int(not legal[int(np.argmax(s))])
        s=np.where(legal,s,-1e8);p=np.exp((s-s.max())/.8);p/=p.sum()
        j=int(rng.choice(10,p=p));c=menu[j]
        fav,opp=fav_opposition(x);eligible=bool(fav.any() and opp.any())
        active='active' if a.active(w) else 'post'
        if tick==0:first_eligible=int(eligible)
        if eligible:
            preferred[active][0]+=1
            preferred[active][1]+=int(j<8 and fav[j]);preferred[active][2]+=int(j<8 and opp[j])
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        actions.append((c.action,c.target))
        a.advance(w)
        assert ink<=w.inked and sha(sorted(w.protected))==protected
        backtracks+=int(c.action[0]=='BACKTRACK')
        if trace:steps.append({'t':tick,'action':c.action,'cursor':w.pos,'goal':w.goal,
                 'eligible':eligible,'advice_active':active=='active','chosen_fav':bool(j<8 and fav[j]),
                 'distance':w.distance()})
        if w.distance()==0 or w.stopped:break
    return {'reached':w.distance()==0,'distance':w.distance(),'steps':tick+1,
        'host_rejected':w.invalid_attempts,'bad':bad,'preferred':preferred,
        'backtracks':backtracks,'ink_sha256':sha(sorted(w.inked)),'actions_sha256':sha(actions),
        'source_sha256':outset,'protected_sha256':protected,'trace':steps,
        'first_eligible':first_eligible}

def assemble(prior,n,seed,train=True):
    """Actual visited, host-legal A0 state menus: training and eval disjoint."""
    base=a0(prior);xs=[];metadata=[]
    fams=FAMILIES_TRAIN if train else FAMILIES_TEST
    for i in range(n):
        fam=fams[i%len(fams)];rotation=(i//2)%4;mirror=bool((i//8)%2)
        if not train:rotation=(rotation+2)%4;mirror=not mirror
        for outcome in ('reverse','ahead'):
            for person in (-1,1):
                s=seed+i*5099+(0 if outcome=='reverse' else 100000)
                w=world(s,fam,person,outcome,rotation,mirror,1 if train else 2,i*13)
                for arm in ARMS:
                    for directive in ('backtrack','continue'):
                        for strength in (.35,1.):
                            v=copy.deepcopy(w);a=HostAdvice(v)
                            getattr(a,'issue_'+directive)(v,strength=strength,ttl=9)
                            rng=np.random.default_rng(s+int(arm*100)*23+int(strength*100)*53+(0 if directive=='backtrack' else 9971))
                            for t in range(25 if train else 32):
                                menu,x=observe32(v,arm,a)
                                fav,opp=fav_opposition(x)
                                if fav.any() and opp.any() and bool(a.active(v)):
                                    xs.append(x)
                                    metadata.append({'seed':s,'family':fam,'outcome':outcome,'personality':person,
                                      'arm':arm,'directive':directive,'strength':strength,'tick':t,
                                      'source_sha256':source_hash(v),
                                      'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
                                with torch.no_grad():scores=base(torch.from_numpy(x)).numpy().astype(np.float64)
                                scores=np.where([c.legal for c in menu],scores,-1e8)
                                p=np.exp((scores-scores.max())/.8);p/=p.sum()
                                choice=menu[int(rng.choice(10,p=p))]
                                assert v.apply(choice) and v.invalid_attempts==0
                                a.advance(v)
                                if v.distance()==0 or v.stopped:break
    if not xs:raise RuntimeError('no two-sided guidance menus generated')
    return np.asarray(xs,np.float32),metadata

def optimize(prior,x,kind,seed=220122,epochs=20):
    assert kind in ('bounded','strong')
    torch.set_num_threads(1);torch.manual_seed(seed)
    model=StyleSidecar(a0(prior),scale=.6 if kind=='bounded' else 1.2)
    xx=torch.from_numpy(x);fav,_=fav_opposition(xx)
    with torch.no_grad():
        old=softmax(a0(prior)(xx),xx);basefav=(old[:,:8]*fav).sum(1)
        native=softmax(Prior32(prior)(xx),xx);nativefav=(native[:,:8]*fav).sum(1)
    # Reference is a weakly-restored C12 mass, not a one-step hidden map label.
    target=torch.minimum(torch.maximum(basefav,nativefav),basefav+.045).detach()
    optimizer=torch.optim.AdamW(model.head.parameters(),lr=.002,weight_decay=.01)
    rng=np.random.default_rng(seed);hist=[];steps=0
    for ep in range(epochs):
        order=rng.permutation(len(xx));total=0.
        for st in range(0,len(order),64):
            ix=order[st:st+64];z=xx[ix];p=softmax(model(z),z);favmass=(p[:,:8]*fav[ix]).sum(1)
            # Substantive differentiable policy-distribution training. The
            # stronger setting relaxes trust penalties to test oversteering.
            drift=(p-old[ix]).square().sum(1).mean()
            objective=(favmass-target[ix]).square().mean()+ (2.5 if kind=='bounded' else .25)*drift
            objective=objective+.02*(model.scale*torch.tanh(model.head(z[:,:8,:]))).square().mean()
            objective.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            steps+=1;total+=float(objective.detach())*len(ix)
        hist.append(round(total/len(xx),8))
    return model.eval(),steps,hist

def state_audit(models,xx):
    x=torch.from_numpy(xx);f,o=fav_opposition(x);records={}
    for name,model in models.items():
        ff=[];b=[]
        for k in range(0,len(x),256):
            with torch.no_grad():p=softmax(model(x[k:k+256]),x[k:k+256])
            ff.extend((p[:,:8]*f[k:k+256]).sum(1).tolist());b.extend(p[:,8].tolist())
        records[name]={'fav_ink_mass':round(float(np.mean(ff)),6),'backtrack_mass':round(float(np.mean(b)),6)}
    return {'same_state_menus':len(x),'both_turns_eligible':int(np.sum(o.any(-1).numpy())), 'models':records}

def eval_rollouts(models,n,seed,horizon=54,trace_limit=16):
    rows=[];fams=FAMILIES_TEST
    for i in range(n):
        fam=fams[i%2];rot=((i//2)+2)%4;mir=not bool((i//8)%2)
        for outcome in ('reverse','ahead'):
            for person in (-1,1):
                s=seed+i*5099+(0 if outcome=='reverse' else 100000)
                for arm in ARMS:
                    for directive in DIRECTIONS3:
                        for strength in ((0.,) if directive=='none' else (.35,1.)):
                            for name,policy in models.items():
                                w=world(s,fam,person,outcome,rot,mir,2,i*13)
                                rs=s+int(arm*100)*23+int(strength*100)*53+{'none':0,'backtrack':7911,'continue':9971}[directive]
                                rec=sim(policy,w,arm,directive,strength,rs,horizon,trace=(len(rows)<trace_limit*len(models)))
                                rec.update({'model':name,'seed':s,'family':fam,'outcome':outcome,'personality':person,
                                  'arm':arm,'directive':directive,'strength':strength})
                                rows.append(rec)
    return rows

def metrics(rows):
    result={}
    for outcome in ('reverse','ahead'):
        for directive in DIRECTIONS3:
            for strength in ((0.,) if directive=='none' else (.35,1.)):
                key=f'{outcome}/{directive}/{strength:.2f}'
                group=[r for r in rows if r['outcome']==outcome and r['directive']==directive and r['strength']==strength]
                result[key]={}
                for name in sorted(set(r['model'] for r in group)):
                    z=[r for r in group if r['model']==name]
                    result[key][name]={'goals':sum(r['reached'] for r in z),'n':len(z),
                      'mean_distance':round(float(np.mean([r['distance'] for r in z])),3),
                      'mean_success_steps':round(float(np.mean([r['steps'] for r in z if r['reached']])),3) if any(r['reached'] for r in z) else None,
                      'active_two_sided':sum(r['preferred']['active'][0] for r in z),
                      'active_fav_selected':sum(r['preferred']['active'][1] for r in z),
                      'post_two_sided':sum(r['preferred']['post'][0] for r in z),
                      'post_fav_selected':sum(r['preferred']['post'][1] for r in z),
                      'by_personality':{str(person):{'goals':sum(r['reached'] for r in z if r['personality']==person),
                         'n':sum(r['personality']==person for r in z),
                         'active_fav_selected':sum(r['preferred']['active'][1] for r in z if r['personality']==person),
                         'active_two_sided':sum(r['preferred']['active'][0] for r in z if r['personality']==person)}
                         for person in (-1,1)}}
    return result

def guards(prior,model):
    # No-cue/outside opportunity exactness: only legal favourite INK can shift.
    for fam in FAMILIES_TRAIN+FAMILIES_TEST:
        for i in range(4):
            for person in (-1,1):
                w=world(9000+i,fam,person,'reverse',i%4,bool(i%2),2,13*i)
                for directive in DIRECTIONS3:
                    a=HostAdvice(w);v,a=copy.deepcopy((w,a))
                    if directive!='none':getattr(a,'issue_'+directive)(v,strength=1.,ttl=9)
                    _,x=observe32(v,.35,a);xx=torch.from_numpy(x)
                    with torch.no_grad():old=a0(prior)(xx);new=model(xx)
                    f,_=fav_opposition(x)
                    for j in range(10):
                        if j>=8 or not f[j] or directive=='none':assert torch.equal(old[j],new[j])
                    if directive=='none':assert torch.equal(new,prior(xx[:,:29]))
                    assert torch.equal(new[8:],old[8:])
    return True

def run(archive,out,train_n=8,eval_n=8,epochs=20,horizon=54):
    torch.set_num_threads(1)
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    name_map=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer','persistent_feedback','signed_feedback','selective_continue')
    paths=[archive/'prerequisites'/('guidance_recovery_'+p+'.safetensors') for p in name_map]
    assert all(p.exists() for p in paths)
    prior=load_c12(paths);X,receipts=assemble(prior,train_n,220112,train=True)
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();input_sha=hashlib.sha256(X.tobytes()).hexdigest()
    models={'C12':Prior32(prior),'C18-A0':a0(prior)};training={}
    for kind in ('bounded','strong'):
        mod,steps,loss=optimize(load_c12(paths),X,kind,epochs=epochs)
        assert guards(prior,mod)
        label='C22-'+kind.upper();models[label]=mod
        chk=out/(label+'.safetensors')
        provenance={'schema':SCHEMA,'source_sha256':source,'data_sha256':input_sha,'kind':kind,
                    'predecessor_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in zip(KEYS,paths)}}
        save_file({k:v.detach().cpu().contiguous() for k,v in mod.head.state_dict().items()},str(chk),
             metadata={'provenance':json.dumps(provenance,sort_keys=True,separators=(',',':'))})
        training[label]={'updates':steps,'loss_first':loss[0],'loss_last':loss[-1],
                         'checkpoint_sha256':hashlib.sha256(chk.read_bytes()).hexdigest()}
    heldX,heldreceipts=assemble(prior,max(4,eval_n//2),330212,train=False)
    held=eval_rollouts(models,eval_n,340212,horizon)
    # No-guidance exactness verified from actual host-applied matched trajectories.
    comparisons={}
    for name in ('C18-A0','C22-BOUNDED','C22-STRONG'):
        a=sorted((r for r in held if r['model']=='C12' and r['directive']=='none'),key=lambda z:(z['seed'],z['arm'],z['personality'],z['outcome']))
        b=sorted((r for r in held if r['model']==name and r['directive']=='none'),key=lambda z:(z['seed'],z['arm'],z['personality'],z['outcome']))
        assert len(a)==len(b)
        comparisons[name]={'pairs':len(a),'exact':sum(x['actions_sha256']==y['actions_sha256'] and x['ink_sha256']==y['ink_sha256'] for x,y in zip(a,b))}
        assert comparisons[name]['exact']==len(a)
    assert sum(r['host_rejected'] for r in held)==0
    assert all(r['bad']==0 for r in held)
    for name,rows in (('training_receipts',receipts),('heldout_receipts',heldreceipts),('heldout_trajectories',held)):
        (out/(name+'.json')).write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n')
    metric={'schema':SCHEMA,'source_sha256':source,'training_input_sha256':input_sha,
         'training_menus':len(X),'heldout_same_state_menus':len(heldX),
         'training':training,'fixed_state':state_audit(models,heldX),'heldout':metrics(held),
         'heldout_episodes':len(held),'cue_off':comparisons,'host_rejected':0,
         'illegal_raw_top1':sum(r['bad'] for r in held),
         'training_family':FAMILIES_TRAIN,'heldout_family':FAMILIES_TEST,
         'disclosure':'Synthetic host fixtures only; no CyberSand native inference or physical acceptance.'}
    (out/'metrics.json').write_text(json.dumps(metric,sort_keys=True,indent=2)+'\n')
    return metric

if __name__=='__main__':
    if len(sys.argv)<3:raise SystemExit('usage: python c22_branch_personality.py C21_archive_dir OUTPUT [train_n eval_n epochs horizon]')
    result=run(sys.argv[1],sys.argv[2],*[int(v) for v in sys.argv[3:]])
    print(json.dumps({'trained_menus':result['training_menus'],'heldout_episodes':result['heldout_episodes'],
                      'updates':result['training'],'fixed_state':result['fixed_state']},sort_keys=True))

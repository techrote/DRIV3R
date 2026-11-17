"""C28: trained, stochastic, host-verified multi-step recovery option.

External synthetic game-policy research only. No CyberSand/native ABI changes.
C27 paths are EXECUTED through World.apply before the goal switch, rather
than silently preloading the trail. The saved option is a policy intent,
not an action script; each step is independently selected from a freshly
host-authorised menu. Historical d_common is not modified.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from experiment import World, ACTIONS, ARMS, sha
from signed_feedback import HostAdvice
from attested_stack_cue import observe32,KEYS
from personality_retention_core import load_c12
from c19_return_anchored import a0,build_bridge
from c27_deep_recovery import long_world
from attested_stack_cue import AttestedGate

SCHEMA='driv3r-guidance-c28-executed-temporal-option/1'


def executed_world(i,outcome,person,seed,train):
    """Replay the entire corridor into the host before changing the goal.

    The initial path is a deterministic research prefix (not model behaviour),
    but *every move* is executed under the host's ordinary safety checks.
    No future shortest-path data enter any policy feature.
    """
    fixture=long_world(i,'reverse',person,seed,train)
    trail=[*fixture.history,fixture.pos]
    w=World(seed,'islands',person)
    w.protected=set(fixture.protected)
    if outcome in ('tunnel','decoy'):
        goal=trail[0] if train else trail[10]
        xx,yy=trail[-1]
        if outcome=='tunnel':
            # Withheld counterexample: a real direct path exists despite the
            # historically long route; constructed before any host action.
            while xx!=goal[0]:
                xx+=1 if goal[0]>xx else -1;w.protected.discard((xx,yy))
            while yy!=goal[1]:
                yy+=1 if goal[1]>yy else -1;w.protected.discard((xx,yy))
        else:
            # Withheld *false* local improvement: create one tempting fresh
            # adjacent cell that has no other legal neighbour. It is NOT a
            # shortcut; the host-legal shortest path remains the long trail.
            old_allowed={(x,y) for x in range(15) for y in range(15)}-w.protected
            choices=[]
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                q=(xx+dx,yy+dy)
                if q in old_allowed or not (1<=q[0]<=13 and 1<=q[1]<=13):continue
                if abs(q[0]-goal[0])+abs(q[1]-goal[1])>=abs(xx-goal[0])+abs(yy-goal[1]):continue
                adjacent={(q[0]+ex,q[1]+ey) for ex,ey in ((1,0),(-1,0),(0,1),(0,-1))} & old_allowed
                if adjacent=={(xx,yy)}:choices.append(q)
            assert choices, ('cannot make dead-end decoy',i,train,trail[-1],goal)
            w.protected.discard(choices[0])
    w.pos=trail[0];w.goal=trail[-1];w.inked={trail[0]}
    for nxt in trail[1:]:
        menu=w.candidates(0.)
        matched=[c for c in menu[:4] if c.target==nxt and c.legal]
        assert len(matched)==1,(nxt,w.pos)
        assert w.apply(matched[0]) and w.invalid_attempts==0
    assert w.pos==trail[-1] and w.history==trail[:-1]
    assert len(w.events)==len(trail)-1
    if outcome in ('reverse','tunnel','decoy'):w.goal=trail[0] if train else trail[10]
    elif outcome=='short':w.goal=trail[-6]
    elif outcome=='near':w.goal=trail[-4]
    else:raise ValueError(outcome)
    w.goal_change=True;w.goals_seen=2;w.stuck=2
    return w


def history_evidence(w):
    """Verified host trail index, not a visibility-through-walls oracle."""
    found=[k for k,p in enumerate(w.history) if p==w.goal]
    if not found:return None
    dist=len(w.history)-found[-1]
    direct=max(1,w.distance())
    return {'stack_moves':dist,'direct_distance':direct,'detour_ratio':dist/direct,
            'stack_fraction':dist/max(1,len(w.history))}


class OptionHead(nn.Module):
    """Learn a bounded stochastic BACKTRACK logit increment, no other logits."""
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(5,1)
        with torch.no_grad():
            self.fc.weight.zero_();self.fc.bias.fill_(0.05)
    def forward(self,z):
        return 11.*torch.sigmoid(self.fc(z)).squeeze(-1)


def features(w,advice,origin):
    ev=history_evidence(w)
    if ev is None:return None
    return np.array([ev['stack_moves']/40.,min(2.,ev['detour_ratio'])/2.,
          np.clip((w.steps-origin)/40.,0.,2.),
          float(advice.directive), float(advice.strength)],np.float32)


class TemporalOption:
    """A finite, verifiable recovery *intent*, retaining stochastic choice.

    Only a live actual host cue may initialise it. Continued operation after
    cue expiry requires the same goal on the legal history stack; every tick
    rechecks permissions. No model writes cells or invents legal actions.
    Explicit cancellation and goal change invalidate it immediately.
    """
    def __init__(self,head,persistent,probe=False,veto=False):
        self.head=head;self.persistent=persistent;self.probe=probe;self.veto=veto
        self.veto_latched=False;self.active=False
        self.goal=None;self.origin=0;self.last_depth=None;self.intent_source=None
        self.original_strength=0.;self.retreats=0;self.rejections=0
    def observe(self,w,advice,menu):
        ev=history_evidence(w)
        active_cue=advice.active(w)!=0
        if self.veto_latched:return False
        if self.active:
            if (tuple(w.goal)!=self.goal or advice.directive==0 or not menu[8].legal
                or ev is None or ev['stack_moves']<1
                or (self.last_depth is not None and ev['stack_moves']>self.last_depth+3)
                or w.steps-self.origin>110 or (not self.persistent and not active_cue)):
                self.active=False
        if not self.active and active_cue and menu[8].legal and ev is not None:
            # Strong past-only evidence, deliberately not blind cue acceptance.
            # Shortest paths and hidden protected-cell geometry are not inspected.
            fresh_progress=any(c.legal and c.action[0] in ('INK','LIFT')
                      and c.target not in w.history and c.target not in w.inked
                      and w.distance(c.target)<w.distance() for c in menu[:8])
            if self.veto and ev['stack_moves']>=14 and ev['detour_ratio']>=2.0 and fresh_progress:
                # Conservative, episode-long abstention. A locally promising
                # alternative may be a real shortcut; do not persistently
                # override it until a new host intervention is issued.
                self.veto_latched=True
                return False
            if ev['stack_moves']>=14 and ev['detour_ratio']>=2.0 and not (self.probe and fresh_progress):
                self.active=True;self.goal=tuple(w.goal);self.origin=w.steps
                self.original_strength=advice.strength
                self.intent_source=advice.directive;self.retreats=0;self.rejections=0
        self.last_depth=ev['stack_moves'] if ev is not None else None
        return self.active and (self.persistent or active_cue)
    def bonus(self,w,advice):
        f=features(w,advice,self.origin)
        # Retain issuance metadata once the cue expires, not stale cue claims.
        f[3]=self.intent_source or 0
        f[4]=self.original_strength
        with torch.no_grad():return float(self.head(torch.from_numpy(f)).item())
    def step(self,w,selected):
        if self.active:
            if selected.action[0]=='BACKTRACK':self.retreats+=1
            else:self.rejections+=1


def scores(base,w,advice,arm,option=None):
    menu,x=observe32(w,arm,advice)
    with torch.no_grad():
        feed=x if isinstance(base,AttestedGate) else x[:,:29]
        logits=base(torch.from_numpy(feed)).numpy().astype(np.float64)
    raw_illegal=int(not menu[int(np.argmax(logits))].legal)
    bonus=0.
    if option is not None and option.observe(w,advice,menu):
        bonus=option.bonus(w,advice)
        logits[8]+=bonus
    legal=np.array([c.legal for c in menu])
    logits=np.where(legal,logits,-1e8)
    # Entropy floor: even during the option at least 1.5% probability
    # remains on host-legal alternatives (no forced action scripts).
    if bonus>0:
        other=np.concatenate([logits[:8],logits[9:]])
        other=other[np.isfinite(other) & (other>-1e7)]
        odds=math.log(.985/.015)*.8
        desired=odds+.8*math.log(float(np.exp((other-np.max(other))/.8).sum()))+float(np.max(other))
        if logits[8]>desired:logits[8]=desired
    return menu,logits,raw_illegal,bonus


def probs(logits,temp=.8):
    z=np.exp((logits-np.max(logits))/temp);return z/z.sum()


def collect_training(base,head,n=12,seed=281011):
    """Executed legal rollback states + actual frozen-model logits.

    Uses no hidden shortest-path oracle; teaching labels are a successful
    sequence of actual host BACKTRACK steps to a goal on executed history.
    """
    f=[];baseline=[];receipts=[];off_targets=[]
    for i in range(n):
        for person in (-1,1):
            for arm in (.35,1.):
                for cue in ('backtrack','continue'):
                    w=executed_world(i,'reverse',person,seed+i*7919+person*73,True)
                    a=HostAdvice(w);getattr(a,'issue_'+cue)(w,strength=arm,ttl=9)
                    start=w.steps
                    for tick in range(36):
                        if w.distance()==0:break
                        menu,logits,raw,_=scores(base,w,a,arm)
                        assert menu[8].legal and raw==0
                        ev=history_evidence(w)
                        assert ev is not None
                        if ev['stack_moves']>=1:
                            z=features(w,a,start);z[3]=a.directive;z[4]=arm
                            f.append(z);baseline.append(logits)
                            receipts.append({'world':i,'personality':person,'arm':arm,'cue':cue,
                                'step':tick,'stack_moves':ev['stack_moves'],
                                'world_sha256':sha([w.pos,w.goal,w.history,sorted(w.protected)]),
                                'legal_backtrack':bool(menu[8].legal)})
                        # Real host, not a teacher-written future observation.
                        assert w.apply(menu[8]) and not w.invalid_attempts
                        a.advance(w)
                    assert w.distance()==0
    return np.asarray(f,np.float32),np.asarray(baseline,np.float32),receipts


def train(base,n=12,epochs=18,seed=281011):
    torch.set_num_threads(1);torch.manual_seed(seed)
    head=OptionHead();x,logits,receipts=collect_training(base,head,n=n,seed=seed)
    assert len(x)>0
    # Soft 0.985 backtrack target: keep stochastic alternatives available.
    z=torch.from_numpy(x); raw=torch.from_numpy(logits)
    optim=torch.optim.AdamW(head.parameters(),lr=.018,weight_decay=.01)
    rng=np.random.default_rng(seed);hist=[];updates=0
    for ep in range(epochs):
        order=rng.permutation(len(z));total=0.
        for st in range(0,len(z),96):
            ix=order[st:st+96];l=raw[ix].clone()
            bonus=head(z[ix]);l[:,8]+=bonus
            p=torch.softmax(l/.8,dim=-1)[:,8].clamp(1e-8,1-1e-8)
            desired=torch.full_like(p,.985)
            loss=nn.functional.binary_cross_entropy(p,desired)
            # Penalise excessive evidence-conditioned intervention to avoid
            # simply hardcoding 100%-BACKTRACK or overwriting the base logits.
            loss+=.0025*bonus.square().mean()
            loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.)
            optim.step();optim.zero_grad(set_to_none=True)
            total+=float(loss.detach())*len(ix);updates+=1
        hist.append(round(total/len(z),6))
    with torch.no_grad():
        plus=head(z);p=torch.softmax(torch.cat((raw[:,:8],(raw[:,8]+plus)[:,None],raw[:,9:]),axis=1)/.8,axis=1)[:,8]
    return head.eval(),{'updates':updates,'losses':hist,'train_menus':len(z),
       'mean_p_backtrack':float(p.mean()),'min_p_backtrack':float(p.min()),
       'max_p_backtrack':float(p.max()),'train_inputs_sha256':hashlib.sha256(x.tobytes()+logits.tobytes()).hexdigest()},receipts


def rollout(base,head,i,outcome,person,arm,directive,strength,seed,train,horizon,variant,trace=False):
    w=executed_world(i,outcome,person,seed,train)
    pre=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    pprotected=sha(sorted(w.protected));inked=set(w.inked)
    a=HostAdvice(w)
    if directive!='none':getattr(a,'issue_'+directive)(w,strength=strength,ttl=9)
    option=TemporalOption(head,variant in ('PERSIST','PROBE','VETO'),probe=(variant=='PROBE'),veto=(variant=='VETO')) if variant in ('PERSIST','TTL','PROBE','VETO') else None
    rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+{'none':0,'backtrack':411,'continue':977}[directive])
    actions=[];positions=[];bad=0;bonuses=0;first_retreat=None
    options_nonback=0;option_max_p=0.;option_total_entropy=0.
    for tick in range(horizon):
        if w.distance()==0 or w.stopped:break
        menu,l,illegal,bonus=scores(base,w,a,arm,option)
        bad+=illegal;bonuses+=int(bonus>0)
        distribution=probs(l)
        j=int(rng.choice(10,p=distribution))
        c=menu[j];assert c.legal
        if bonus>0:
            option_max_p=max(option_max_p,float(distribution[8]))
            option_total_entropy+=float(-np.sum(distribution[distribution>0]*np.log(distribution[distribution>0])))
            options_nonback+=int(c.action[0]!='BACKTRACK')
        if option:option.step(w,c)
        if c.action[0]=='BACKTRACK' and first_retreat is None:first_retreat=tick+1
        assert w.apply(c) and w.invalid_attempts==0
        a.advance(w)
        actions.append((c.action,c.target))
        if trace:positions.append({'step':tick+1,'pos':w.pos,'action':c.action[0],
                                   'distance':w.distance(),'history':len(w.history),
                                   'bonus':round(bonus,3)})
    assert sha(sorted(w.protected))==pprotected and inked<=w.inked
    return {'success':int(w.distance()==0),'steps':len(actions),'end_distance':w.distance(),
     'stopped':bool(w.stopped),'backtracks':sum(k[0][0]=='BACKTRACK' for k in actions),
     'first_backtrack':first_retreat,'option_steps':bonuses,
     'option_nonbacktrack':options_nonback,'option_max_back_probability':option_max_p,
     'option_mean_entropy':option_total_entropy/max(1,bonuses),
     'host_rejected':w.invalid_attempts,'raw_illegal_top1':bad,
     'action_sha256':sha(actions),'ink_sha256':sha(sorted(w.inked)),
     'prefix_sha256':pre,'path':positions if trace else None,
     'goal':w.goal,'protected':sorted(w.protected) if trace else None}


def evaluate(prior,head,seed=281900,n=4,horizon=100,train=False):
    torch.set_num_threads(1)
    frozen=a0(prior)
    rows=[]
    for i in range(n):
        for person in (-1,1):
          for outcome in ('reverse','short','near','tunnel','decoy'):
            for arm in (.35,1.):
              for directive in ('none','backtrack','continue'):
                for strength in ((0.,) if directive=='none' else (.35,1.)):
                  ss=seed+i*8017+person*313+('reverse','short','near','tunnel','decoy').index(outcome)*3319+int(arm*100)*127
                  for variant in ('C12','C18-A0','TTL','PERSIST','VETO'):
                    base=prior if variant=='C12' else frozen
                    record=rollout(base,head,i,outcome,person,arm,directive,strength,ss,
                                  train,horizon,variant,trace=(i==0 and person==1 and arm==1. and strength==1.))
                    record.update({'model':variant,'index':i,'outcome':outcome,
                                  'personality':person,'arm':arm,'directive':directive,
                                  'strength':strength,'seed':ss})
                    rows.append(record)
    return rows


def summarize(rows):
    result={}
    for variant in ('C12','C18-A0','TTL','PERSIST','VETO'):
        v=[r for r in rows if r['model']==variant]
        u=[r for r in v if r['directive']!='none']
        deep=[r for r in u if r['outcome']=='reverse']
        result[variant]={'guided_goals':sum(r['success'] for r in u),
              'guided_n':len(u),'deep_reverse_goals':sum(r['success'] for r in deep),
              'deep_reverse_n':len(deep),
              'mean_deep_backtracks':round(float(np.mean([r['backtracks'] for r in deep])),2),
              'option_steps':sum(r['option_steps'] for r in v),
              'option_nonbacktrack':sum(r['option_nonbacktrack'] for r in v),
              'max_back_probability':max(r['option_max_back_probability'] for r in v),
              'tunnel_goals':sum(r['success'] for r in u if r['outcome']=='tunnel'),
              'tunnel_n':sum(r['outcome']=='tunnel' for r in u),
              'decoy_goals':sum(r['success'] for r in u if r['outcome']=='decoy'),
              'decoy_n':sum(r['outcome']=='decoy' for r in u),
              'mean_tunnel_steps_success':round(float(np.mean([r['steps'] for r in u if r['outcome']=='tunnel' and r['success']])),2) if any(r['outcome']=='tunnel' and r['success'] for r in u) else None,
              'mean_deep_first_backtrack':round(float(np.mean([r['first_backtrack'] for r in deep if r['first_backtrack'] is not None])),2) if any(r['first_backtrack'] is not None for r in deep) else None,
              'host_rejected':sum(r['host_rejected'] for r in v),
              'illegal_raw_top1':sum(r['raw_illegal_top1'] for r in v)}
    return result


def qualify(rows):
    lookup={(r['model'],r['index'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength']):r for r in rows}
    out={}
    for name in ('TTL','PERSIST','VETO'):
        cueoff=[];regressed=[];improved=[]
        for r in rows:
            if r['model']!=name:continue
            k=(r['index'],r['outcome'],r['personality'],r['arm'],r['directive'],r['strength'])
            ref=lookup['C12',*k] if r['directive']=='none' else lookup['C18-A0',*k]
            if r['directive']=='none':
                cueoff.append(r['action_sha256']==ref['action_sha256'] and r['ink_sha256']==ref['ink_sha256'])
        groups=collections.defaultdict(lambda:collections.Counter())
        for r in rows:
            if r['directive']=='none':continue
            g=(r['outcome'],r['directive'],r['strength'])
            groups[g][r['model']]+=r['success']
        for g,d in groups.items():
            if d[name]<d['C18-A0']:regressed.append([str(g),d['C18-A0'],d[name]])
            if d[name]>d['C18-A0']:improved.append([str(g),d['C18-A0'],d[name]])
        out[name]={'cue_off_identical':all(cueoff),'cue_off_n':len(cueoff),
                'regressed_groups':regressed,'improved_groups':improved,
                'strict_nonregression':len(regressed)==0}
    return out


def run(archive,out,train_n=12,eval_n=4,epochs=18,horizon=100,seed=281900):
    torch.set_num_threads(1)
    archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[archive/'prerequisites'/f'guidance_recovery_{s}.safetensors' for s in
           ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
            'persistent_feedback','signed_feedback','selective_continue')]
    prior=load_c12(paths)
    frozen=a0(prior)
    head,train,receipts=train_model(frozen,train_n,epochs,seed-111)
    rows=evaluate(prior,head,seed,eval_n,horizon,False)
    rows2=evaluate(prior,head,seed+100000,eval_n,horizon,False)
    source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    weights={k:v.detach().contiguous().cpu() for k,v in head.state_dict().items()}
    cp=out/'C28-TEMPORAL.safetensors'
    save_file(weights,str(cp),metadata={'provenance':json.dumps({'schema':SCHEMA,
      'source_sha256':source_sha,'train_sha256':train['train_inputs_sha256']},sort_keys=True)})
    metrics={'schema':SCHEMA,'source_sha256':source_sha,
        'train':train,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
        'prerequisites_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'cohort_a':summarize(rows),'cohort_b':summarize(rows2),
        'qualification_a':qualify(rows),'qualification_b':qualify(rows2),
        'heldout_episodes':len(rows)+len(rows2),
        'limitations':['External synthetic host, not CyberSand native ABI',
          'Training uses supervised successful executed BACKTRACK sequences, not on-policy RL',
          'Detour ratio derived from verified history depth and Manhattan distance; not shortest-path topology',
          'Short/near on-trail adversaries are easier than unrestricted natural-world branching',
          'VETO abstains from recovery on promising local dead ends; rare goal reversals with forward decoys remain unresolved',
          'Prefix path is host-executed but synthetic predetermined, not independent worm navigation']}
    for filename,obj in [('metrics.json',metrics),('train_receipts.json',receipts),('cohort_a.json',rows),('cohort_b.json',rows2)]:
        (out/filename).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
    return metrics

train_model=train
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('out')
    parser.add_argument('--train-n',type=int,default=12);parser.add_argument('--eval-n',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=18);parser.add_argument('--horizon',type=int,default=100)
    parser.add_argument('--seed',type=int,default=281900);a=parser.parse_args()
    print(json.dumps(run(a.archive,a.out,a.train_n,a.eval_n,a.epochs,a.horizon,a.seed),indent=2))

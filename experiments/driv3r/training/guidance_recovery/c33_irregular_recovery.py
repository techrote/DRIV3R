"""C33: out-of-family irregular grafts and host-executed cue response.

Read-only synthetic CyberSand-like World; no native ABI claim. A second network
trains on ACTUALLY EXECUTED short continuation returns but is admitted to
policy only when an explicit positive host recovery cue activates C32-WATCH.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from experiment import sha, DIRECTIONS
from c32_offstack_recovery import (world as base_world, SearchPolicy, OptionHead,
    probs, scores, sample_features, fresh, run_episode)
from c29_probe_recovery import PREREQS, ProbeHead, custom_rollout
from personality_retention_core import load_c12
from c19_return_anchored import a0
from signed_feedback import HostAdvice

SCHEMA='driv3r-c33-irregular-branches/1'
SCENARIOS=('off-early','off-mid','shortcut','lure4')
FAMILIES=('twigs','braid')


def graft_world(i,scenario,family,personality,seed):
    """Different graph edges, junctions, cycles: not just reflected U-corridors.

    Every added node is reachable from a legal current cell. Evaluation uses
    family=braid, never observed by training; no shape label is policy input.
    """
    if family not in FAMILIES: raise ValueError(family)
    w=base_world(i,scenario,personality,seed)
    permitted={(x,y) for x in range(15) for y in range(15) if (x,y) not in w.protected}
    original=set(permitted)
    rng=np.random.default_rng(seed ^ (i*7919+1991))
    if family=='twigs':
        # Random walks from distinct original corridor anchors; mostly tree spurs.
        anchors=list(sorted(original))
        for k in range(5):
            p=anchors[int(rng.integers(len(anchors)))]; prev=None
            for step in range(2+int(rng.integers(2,7))):
                neigh=[(p[0]+dx,p[1]+dy) for dx,dy in DIRECTIONS]
                neigh=[q for q in neigh if 1<=q[0]<=13 and 1<=q[1]<=13 and q not in original and q!=prev]
                if not neigh:break
                new=[q for q in neigh if q not in permitted]
                if new and rng.random()<.85:neigh=new
                q=neigh[int(rng.integers(len(neigh)))];permitted.add(q);prev,p=p,q
    else:
        # Unseen graph family: irregular Manhattan braids linking old nodes;
        # retains loops and multiple competing, often locally attractive exits.
        anchors=list(sorted(original))
        for _ in range(5):
            a=anchors[int(rng.integers(len(anchors)))];b=anchors[int(rng.integers(len(anchors)))]
            if abs(a[0]-b[0])+abs(a[1]-b[1])<5:continue
            p=a
            while p!=b:
                axes=[]
                if p[0]!=b[0]:axes.append((p[0]+(1 if b[0]>p[0] else -1),p[1]))
                if p[1]!=b[1]:axes.append((p[0],p[1]+(1 if b[1]>p[1] else -1)))
                p=axes[int(rng.integers(len(axes)))];permitted.add(p)
    w.protected={(x,y) for x in range(15) for y in range(15) if (x,y) not in permitted}
    assert not w.invalid_attempts and w.goal not in w.history and w.allowed(w.goal)
    assert len(permitted-original)>=1
    deg=collections.Counter(sum((x+dx,y+dy) in permitted for dx,dy in DIRECTIONS) for x,y in permitted)
    info={'family':family,'added':len(permitted-original),'junctions':sum(v for k,v in deg.items() if k>=3),
        'branch_sha256':sha(sorted(permitted))}
    return w,info


def features(w,menu,j):
    c=menu[j]
    return np.array([float(c.target not in w.history),float(c.target not in w.inked),
        float(c.target in w.inked),float(c.features[22]),
        float(w.distance()-w.distance(c.target))/2.,
        float(c.features[21]),min(len(w.history),50)/50.,min(w.stuck,8)/8.],np.float32)


def continuation(w,j,depth=10):
    """Execute candidate then greedy local choices; never peek at BFS or future map.

    This offline short-horizon teacher is fixed before training. All returns
    are observed from World.apply after host independently checks legality.
    """
    v=copy.deepcopy(w);start=v.distance();mm=v.candidates(.35)
    assert mm[j].legal and v.apply(mm[j]);seen={w.pos,v.pos};initial_ink=set(v.inked)
    for _ in range(depth-1):
        if v.distance()==0:break
        menu=v.candidates(.35)
        good=[k for k,c in enumerate(menu[:8]) if c.legal and c.action[0]=='INK' and c.target not in seen]
        if not good:
            good=[k for k,c in enumerate(menu[:8]) if c.legal and c.action[0]=='INK']
        if good:
            k=max(good,key=lambda x:(v.distance()-v.distance(menu[x].target),-menu[x].features[22]))
        elif menu[8].legal:k=8
        else:break
        assert v.apply(menu[k]);seen.add(v.pos)
    assert not v.invalid_attempts
    # Executed return (not oracle optimality), bounded and allowed to be negative.
    return float(np.clip((start-v.distance())/depth,-1.,1.)) + (1. if v.distance()==0 else 0.)


class ReturnHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def train(seed=330071,epochs=12):
    torch.set_num_threads(1);torch.manual_seed(seed)
    xx=[];yy=[];receipts=[]
    # Only twigs: braid is withheld at all training stages.
    for i in range(8):
      for scenario in SCENARIOS:
       for personality in (-1,1):
        w,info=graft_world(i,scenario,'twigs',personality,seed+379*i+41*personality+sum(map(ord,scenario)))
        for t in range(3):
            menu=w.candidates(.35)
            js=[j for j,c in enumerate(menu[:8]) if c.legal and c.action[0]=='INK']
            for j in js:
                xx.append(features(w,menu,j));yy.append(continuation(w,j))
            receipts.append({'scenario':scenario,'index':i,'step':t,'personality':personality,
                'prefix_sha256':sha([w.pos,w.goal,w.history,sorted(w.protected)]),
                'menu_sha256':sha([(c.action,c.target,c.legal) for c in menu]),
                'returns':{str(j):yy[-len(js)+k] for k,j in enumerate(js)}})
            if not js:break
            j=max(js,key=lambda k:(w.distance()-w.distance(menu[k].target)))
            assert w.apply(menu[j]) and not w.invalid_attempts
    X=np.stack(xx);Y=np.array(yy,np.float32)
    x=torch.from_numpy(X);y=torch.from_numpy(Y);head=ReturnHead()
    opt=torch.optim.AdamW(head.parameters(),lr=.012,weight_decay=.01)
    rng=np.random.default_rng(seed);updates=0;losses=[]
    for e in range(epochs):
        perm=rng.permutation(len(y));epoch_loss=0.
        for k in range(0,len(y),48):
            z=perm[k:k+48];err=nn.functional.smooth_l1_loss(head(x[z]),y[z]);opt.zero_grad(set_to_none=True)
            err.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
            updates+=1;epoch_loss+=float(err.detach())
        losses.append(round(epoch_loss,6))
    head.eval()
    with torch.no_grad():mae=float(torch.mean(torch.abs(head(x)-y)))
    return head,{'examples':len(y),'updates':updates,'train_mae':mae,
        'positive_returns':int((Y>0).sum()),'losses':losses,
        'data_sha256':hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest()},receipts


class AuditedPolicy(SearchPolicy):
    """WATCH plus optional probe scoring. C33-FIXED is capacity control."""
    def __init__(self,option,head,variant):
        super().__init__(option,None,'WATCH');self.head=head;self.variant=variant;self.rank_changes=0
    def modify(self,base,w,advice,arm):
        menu,logits,bad,target=super().modify(base,w,advice,arm)
        if self.phase=='watch' and advice.directive==1:
            options=fresh(w,menu)
            if options:
                default=max(options,key=lambda j:logits[j])
                if self.variant=='HEAD':
                    with torch.no_grad():
                        values={j:float(self.head(torch.from_numpy(features(w,menu,j)))) for j in options}
                    j=max(options,key=lambda j:(values[j],-j))
                else:j=default
                if j!=default:self.rank_changes+=1
                logits[j]+=2.5
                rest=np.delete(logits,j);rest=rest[rest>-1e7]
                cap=rest.max()+.8*(math.log(.88/.12)+math.log(np.exp((rest-rest.max())/.8).sum()))
                logits[j]=min(logits[j],cap);target=j
        return menu,logits,bad,target


def play(base,option,head,w,seed,arm,directive,strength,model,horizon=125,trace=False):
    if model=='C18-A0':
        return custom_rollout(base,option,ProbeHead(),0,'off-early',w.personality,arm,directive,strength,seed,horizon,trace,world_override=w,baseline_variant='C18-A0')
    if model=='C32-WATCH':
        return run_episode(base,option,None,w,seed,arm,directive,strength,'WATCH',horizon,trace)
    cue=HostAdvice(w)
    if directive!='none':getattr(cue,'issue_'+directive)(w,strength=strength,ttl=9)
    p=AuditedPolicy(option,head,model.split('-')[1]);rng=np.random.default_rng(seed+int(arm*100)*17+int(strength*100)*31+
       {'none':0,'backtrack':411,'continue':977}[directive])
    origin=sha([w.pos,w.goal,w.history,sorted(w.protected),list(w.events)])
    protected=sha(sorted(w.protected));actions=[];path=[];bad=0
    for t in range(horizon):
        if w.stopped or w.distance()==0:break
        menu,logits,raw,target=p.modify(base,w,cue,arm);bad+=raw
        j=int(rng.choice(len(menu),p=probs(logits)));c=menu[j]
        assert c.legal and w.apply(c) and w.invalid_attempts==0
        p.after(c);cue.advance(w);actions.append((c.action,c.target))
        if trace:path.append({'pos':w.pos,'distance':w.distance(),'phase':p.phase,'action':c.action[0]})
    assert sha(sorted(w.protected))==protected
    return {'success':int(w.distance()==0),'steps':len(actions),'prefix_sha':origin,
       'actions_sha':sha(actions),'ink_sha':sha(sorted(w.inked)),
       'host_rejected':w.invalid_attempts,'raw_illegal_top1':bad,
       'rank_changes':p.rank_changes,'path':path if trace else None}


def eval_cohort(base,option,head,seed,n=2,horizon=125):
    rows=[];info={};models=('C18-A0','C32-WATCH','C33-FIXED','C33-HEAD')
    for family in FAMILIES:
     for sc in SCENARIOS:
      for i in range(n):
       for person in (-1,1):
        for arm in (.35,1.):
         for directive in ('none','backtrack','continue'):
          for strength in ((0.,) if directive=='none' else (.35,1.)):
           ss=seed+i*10009+313*person+int(arm*100)*127+sum(ord(x) for x in sc)*811+int(family=='braid')*4337
           for model in models:
            w,topo=graft_world(i+8,sc,family,person,ss)
            info[topo['branch_sha256']]=topo
            trace=i==0 and person==1 and strength==1 and arm==1 and directive=='backtrack'
            r=play(base,option,head,w,ss,arm,directive,strength,model,horizon,trace)
            if model=='C18-A0':
                r['actions_sha']=r['action_sha256'];r['ink_sha']=r['ink_sha256']
            rows.append({'model':model,'family':family,'scenario':sc,'i':i,'personality':person,
              'arm':arm,'directive':directive,'strength':strength,'seed':ss,
              'topology_sha':topo['branch_sha256'],**r})
    return rows,info


def summary(rows):
    names=sorted({r['model'] for r in rows});groups={}
    key=lambda r:(r['family'],r['scenario'],r['directive'],r['strength'])
    key_no=lambda r:(r['family'],r['scenario'],r['i'],r['personality'],r['arm'],r['seed'])
    ref={key_no(r):r for r in rows if r['model']=='C18-A0' and r['directive']=='none'}
    for m in names:
        mr=[r for r in rows if r['model']==m];guided=[r for r in mr if r['directive']!='none']
        group={':'.join(map(str,k)):sum(r['success'] for r in guided if key(r)==k) for k in sorted({key(r) for r in guided})}
        no=[r for r in mr if r['directive']=='none'];match=sum(r['actions_sha']==ref[key_no(r)]['actions_sha'] and r['ink_sha']==ref[key_no(r)]['ink_sha'] for r in no)
        groups[m]={'guided_success':sum(r['success'] for r in guided),'guided_count':len(guided),
          'by_family':{f:sum(r['success'] for r in guided if r['family']==f) for f in FAMILIES},
          'by_scenario':{sc:sum(r['success'] for r in guided if r['scenario']==sc) for sc in SCENARIOS},
          'groups':group,'unguided_identical':match,'unguided_total':len(no),
          'host_rejections':sum(r['host_rejected'] for r in mr),
          'illegal_top1':sum(r['raw_illegal_top1'] for r in mr),
          'rank_changes':sum(r.get('rank_changes',0) for r in mr)}
    for m in names:
        groups[m]['regression_groups']={k:groups['C18-A0']['groups'][k]-v for k,v in groups[m]['groups'].items() if v<groups['C18-A0']['groups'][k]}
    return groups


def run(root,out,seed=330577,n=2,horizon=125):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{p}.safetensors' for p in PREREQS]);option=OptionHead()
    option.load_state_dict(load_file(str(root/'results/C28-TEMPORAL.safetensors')));base=a0(prior)
    head,fit,receipts=train();save_file({k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()},str(out/'C33-RETURN.safetensors'))
    all_metrics={'schema':SCHEMA,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'checkpoint_sha256':hashlib.sha256((out/'C33-RETURN.safetensors').read_bytes()).hexdigest(),
      'fit':fit,'cohorts':{},'topologies':{}}
    for group,s in (('A',seed),('B',seed+200003)):
        rows,info=eval_cohort(base,option,head,s,n,horizon)
        all_metrics['cohorts'][group]=summary(rows)
        all_metrics['topologies'][group]={'distinct':len(info),'twigs':sum(1 for v in info.values() if v['family']=='twigs'),
           'braid':sum(1 for v in info.values() if v['family']=='braid'),
           'added_min':min(x['added'] for x in info.values()),'added_max':max(x['added'] for x in info.values()),
           'junction_min':min(x['junctions'] for x in info.values()),'junction_max':max(x['junctions'] for x in info.values())}
        (out/f'cohort_{group}.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    (out/'metrics.json').write_text(json.dumps(all_metrics,indent=2,sort_keys=True)+'\n')
    (out/'train_receipts.json').write_text(json.dumps(receipts,indent=2,sort_keys=True)+'\n')
    return all_metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('out');p.add_argument('--n',type=int,default=2);p.add_argument('--horizon',type=int,default=125);p.add_argument('--seed',type=int,default=330577);a=p.parse_args()
    print(json.dumps(run(a.root,a.out,a.seed,a.n,a.horizon),sort_keys=True))
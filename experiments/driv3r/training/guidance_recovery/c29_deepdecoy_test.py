"""Withheld 2-step misleading dead-ends; no topology oracle at inference."""
from __future__ import annotations
import argparse,json,hashlib
from collections import deque,Counter,defaultdict
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
from c28_temporal_option import executed_world, OptionHead
from c29_probe_recovery import (fresh_indices,custom_rollout,ProbeHead,PREREQS)
from personality_retention_core import load_c12
from c19_return_anchored import a0

def deep_world(i,person,seed):
    # Odd rotations preserve all prior executed history and permit an isolated
    # two-step false-progress spur that has no exit to the old path.
    w=executed_world(2*i+1,'decoy',person,seed,False)
    m=w.candidates(.35);fresh=fresh_indices(w,m)
    q=m[fresh[0]].target
    candidates=[]
    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
        r=(q[0]+dx,q[1]+dy)
        if not (1<=r[0]<=13 and 1<=r[1]<=13) or r not in w.protected or w.distance(r)>=w.distance(q):continue
        neighbours={(r[0]+ex,r[1]+ey) for ex,ey in ((1,0),(-1,0),(0,1),(0,-1)) if w.allowed((r[0]+ex,r[1]+ey))}
        if neighbours=={q}:candidates.append(r)
    assert candidates,(i,person,q)
    r=sorted(candidates)[0];w.protected.remove(r)
    # Genuine two-step legal false progress: q now appears as traversable as
    # the first cell of the actual shortcut tunnel in candidate adjacency.
    m=w.candidates(.35)
    assert m[fresh[0]].features[22]==.5
    assert w.allowed(q) and w.allowed(r)
    return w

def shortest(w):
    q=deque([(w.pos,0)]);seen={w.pos}
    while q:
        p,n=q.popleft()
        if p==w.goal:return n
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            r=(p[0]+dx,p[1]+dy)
            if r not in seen and w.allowed(r):seen.add(r);q.append((r,n+1))
    return None

def evaluate(prior,option,probe,seed,n=2,horizon=72):
    base=a0(prior);rows=[]
    for i in range(n):
      for person in (-1,1):
       for arm in (.35,1.):
        for directive in ('none','backtrack','continue'):
         for strength in ((0.,) if directive=='none' else (.35,1.)):
          ss=seed+i*8017+person*313+int(arm*100)*127+90311
          for name in ('C12','C18-A0','VETO','PROBE','GUARD'):
             w=deep_world(i,person,ss)
             assert shortest(w)==28
             r=custom_rollout(prior if name=='C12' else base,option,probe,i,'deepdecoy',person,
                  arm,directive,strength,ss,horizon,trace=(i==0 and person==1 and arm==1. and strength==1.),
                  safe_start=(name=='GUARD'),world_override=w,
                  baseline_variant=name if name in ('C12','C18-A0','VETO') else None)
             r.update({'model':name,'i':i,'personality':person,'arm':arm,'directive':directive,
                       'strength':strength,'seed':ss,'outcome':'deepdecoy'});rows.append(r)
    return rows

def report(rows):
    out={}
    for name in ('C12','C18-A0','VETO','PROBE','GUARD'):
        r=[x for x in rows if x['model']==name];guided=[x for x in r if x['directive']!='none']
        out[name]={'goals':sum(x['success'] for x in guided),'guided_n':len(guided),
            'host_rejected':sum(x['host_rejected'] for x in r),
            'raw_illegal_top1':sum(x['raw_illegal_top1'] for x in r),
            'probe_steps':sum(x['probe_steps'] for x in r),
            'mean_goal_steps':round(float(np.mean([x['steps'] for x in guided if x['success']])),2) if any(x['success'] for x in guided) else None}
    for name in ('PROBE','GUARD'):
      paired={(x['model'],x['i'],x['personality'],x['arm'],x['directive'],x['strength']):x for x in rows}
      checks=[(x,paired['C12',x['i'],x['personality'],x['arm'],x['directive'],x['strength']])
          for x in rows if x['model']==name and x['directive']=='none']
      out[name]['cueoff_exact']=sum(a['action_sha256']==b['action_sha256'] and a['ink_sha256']==b['ink_sha256'] for a,b in checks)
      out[name]['cueoff_n']=len(checks)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out');p.add_argument('--n',type=int,default=2)
    a=p.parse_args();root=Path(a.archive);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    prior=load_c12([root/'prerequisites'/f'guidance_recovery_{x}.safetensors' for x in PREREQS])
    option=OptionHead();option.load_state_dict(load_file(str(root/'results'/'C28-TEMPORAL.safetensors')))
    probe=ProbeHead();probe.load_state_dict(load_file(str(out/'C29-PROBE.safetensors')))
    cohorts=[]
    for letter,seed in (('a',282900+190000),('b',382900+190000)):
        rows=evaluate(prior,option,probe,seed,a.n)
        (out/f'deepdecoy_{letter}.json').write_text(json.dumps(rows,sort_keys=True,indent=2)+'\n')
        cohorts.append(report(rows))
    result={'a':cohorts[0],'b':cohorts[1],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'deepdecoy_metrics.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2))

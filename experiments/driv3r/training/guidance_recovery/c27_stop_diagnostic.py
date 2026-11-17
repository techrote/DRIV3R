"""C27 research-only causal STOP-score ablation; never a deployable policy.

Penalising STOP changes unguided behavior and deliberately violates the no-cue
preservation gate. It is evaluated solely as a diagnostic to identify
premature-termination as a possible confound for deeper recovery.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import c26_stuck_interventions as c26
from c27_deep_recovery import long_world
from attested_stack_cue import KEYS
from personality_retention_core import load_c12
from c19_return_anchored import a0

class StopPenalty(nn.Module):
    def __init__(self,policy,penalty):
        super().__init__();self.policy=policy;self.penalty=float(penalty)
    def forward(self,x):
        out=self.policy(x).clone()
        out[...,9]-=self.penalty
        return out

def run(archive,out,n=2,horizon=90,seed=270500):
    torch.set_num_threads(1);archive=Path(archive);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[archive/'prerequisites'/f'guidance_recovery_{s}.safetensors' for s in ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer','persistent_feedback','signed_feedback','selective_continue')]
    prior=load_c12(paths)
    base=a0(prior)
    models={'C18-A0':c26.Control(prior,True),'STOP-3-DIAGNOSTIC':StopPenalty(c26.Control(prior,True),3.),'STOP-6-DIAGNOSTIC':StopPenalty(c26.Control(prior,True),6.)}
    old=c26.seed_world;c26.seed_world=long_world
    try:
        rows=[]
        for cohort in ('A','B'):
            ss=seed+(800000 if cohort=='A' else 900000)
            for i in range(n):
                for person in (-1,1):
                    for arm in (.35,1.):
                        for directive in ('backtrack','continue'):
                            for strength in (.35,1.):
                                scene=ss+i*10117+person*131+int(arm*100)*17+227
                                for name,model in models.items():
                                    rec=c26.rollout(model,base,i,'reverse',person,arm,directive,strength,'badturn',scene,horizon,True)
                                    rec.update({'model':name,'cohort':cohort,'index':i,'personality':person,'arm':arm,'directive':directive,'strength':strength,'seed':scene})
                                    rows.append(rec)
        summary={}
        for cohort in ('A','B'):
            summary[cohort]={}
            for name in models:
                rs=[r for r in rows if r['cohort']==cohort and r['model']==name]
                summary[cohort][name]={'goals':sum(r['success'] for r in rs),'n':len(rs),
                  'premature_stops':sum(not r['success'] and r['steps_after_cue']<horizon for r in rs),
                  'median_steps_on_success':float(np.median([r['steps_after_cue'] for r in rs if r['success']])) if any(r['success'] for r in rs) else None,
                  'host_rejected':sum(r['host_rejected'] for r in rs),'illegal_raw_top1':sum(r['illegal_raw_top1'] for r in rs)}
        (out/'diagnostic_trajectories.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
        metrics={'schema':'driv3r-c27-stop-ablation/1','results':summary,'episodes':len(rows),'nonpromotable_reason':'Explicit STOP suppression violates cue-off bit identity and is not a trained recovery model',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out/'diagnostic_metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
        return metrics
    finally:c26.seed_world=old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('out');p.add_argument('--n',type=int,default=2);p.add_argument('--horizon',type=int,default=90);a=p.parse_args()
    print(json.dumps(run(a.archive,a.out,a.n,a.horizon)['results'],sort_keys=True))
"""C26 same-state policy audit: avoid conflating divergence with personality."""
import argparse,hashlib,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from c26_gallery import models
from c26_stuck_interventions import OUTCOMES,observe40,induce,intervention_trigger
from c24_history_trust import seed_world
from signed_feedback import HostAdvice
from c19_return_anchored import a0
from c22_branch_personality import fav_opposition


def audit(archive,checkpoints,rows,out):
    torch.set_num_threads(1)
    prior,policies=models(archive,checkpoints)
    entries=json.loads(Path(rows).read_text());seen=set();audit_rows=[]
    for r in entries:
        k=(r['index'],r['outcome'],r['personality'],r['arm'],r['mode'],r['directive'],r['strength'],r['seed'])
        if k in seen:continue
        seen.add(k)
        idx,outcome,person,arm,mode,cue,strength,seed=k
        w=seed_world(idx,outcome,person,seed,False)
        w,_=induce(w,a0(prior),arm,seed+119,5,mode)
        assert w.invalid_attempts==0
        a=HostAdvice(w)
        if cue!='none':getattr(a,'issue_'+cue)(w,strength=strength,ttl=9)
        candidates,x=observe40(w,arm,a)
        legal=np.array([c.legal for c in candidates]);pref,opp=fav_opposition(x)
        admitted=bool(pref.any() and opp.any())
        for name,policy in policies.items():
            with torch.no_grad():z=policy(torch.from_numpy(x)).numpy().astype(np.float64)
            assert legal[z.argmax()]
            z=np.where(legal,z,-1e8);probs=np.exp((z-z.max())/.8);probs/=probs.sum()
            assert 1-1e-8<=probs.sum()<=1+1e-8
            audit_rows.append({'id':k,'model':name,'cue':cue,'strength':strength,'mode':mode,
                 'trigger':int(intervention_trigger(x[8,36:])),
                 'opportunity':int(admitted),
                 'fav_probability':float(probs[:8][pref].sum()) if admitted else None,
                 'opp_probability':float(probs[:8][opp].sum()) if admitted else None,
                 'backtrack_probability':float(probs[8]),
                 'entropy':float(-(probs[probs>0]*np.log(probs[probs>0])).sum()),
                 'observation_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    result={'same_state_menus':len(seen),'rows':len(audit_rows),'by_model':{}}
    for name in sorted(policies):
        result['by_model'][name]={}
        for arm in ('cue_off','cue_on','cue_on_triggered'):
            matched=[r for r in audit_rows if r['model']==name and (r['cue']=='none' if arm=='cue_off'
                   else (r['cue']!='none' if arm=='cue_on' else r['cue']!='none' and r['trigger']))]
            suitable=[r for r in matched if r['opportunity']]
            result['by_model'][name][arm]={
                'n_menus':len(matched),'n_personality_opportunities':len(suitable),
                'mean_fav_probability':round(float(np.mean([r['fav_probability'] for r in suitable])),6) if suitable else None,
                'mean_backtrack_probability':round(float(np.mean([r['backtrack_probability'] for r in matched])),6) if matched else None,
                'mean_entropy_nats':round(float(np.mean([r['entropy'] for r in matched])),6) if matched else None}
    assert len(audit_rows)==len(seen)*len(policies)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'same_state_audit.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    # Raw per-state audit is source-checkable but remains external evidence.
    (out/'same_state_menus.json').write_text(json.dumps(audit_rows,sort_keys=True,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('checkpoints');p.add_argument('rows');p.add_argument('out')
    a=p.parse_args();d=audit(a.archive,a.checkpoints,a.rows,a.out)
    print(json.dumps(d,sort_keys=True))

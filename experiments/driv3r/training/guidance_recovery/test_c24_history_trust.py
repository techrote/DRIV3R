"""C24 regressions: history boundaries, host-only legality and deterministic optimizer."""
import unittest
import numpy as np
import torch
from c24_history_trust import (observe36,source_directive,observed_calibration,
    seed_world,TrustAdapter,Baseline36,fork_targets,optimize)
from signed_feedback import HostAdvice
from attested_stack_cue import observe32
from personality_retention_core import load_c12
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=tuple(ROOT/'prerequisites'/('guidance_recovery_'+n+'.safetensors') for n in
    ('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
     'persistent_feedback','signed_feedback','selective_continue'))

class C24Test(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
    torch.set_num_threads(1);cls.prior=load_c12(FILES)

 def test_observation_only_extends_immutable_history(self):
    w=seed_world(2,'reverse',1,54321,True);a=HostAdvice(w);a.issue_continue(w,.35)
    before,old=observe32(w,.35,a)
    mem=np.asarray((.1,-.4,.3,.2),np.float32);after,ext=observe36(w,.35,a,mem)
    self.assertTrue(np.array_equal(old,ext[:,:32]))
    self.assertTrue(np.all(ext[:,32:]==mem))
    self.assertEqual([(c.action,c.target,c.legal) for c in before],[(c.action,c.target,c.legal) for c in after])

 def test_no_cue_exact_c12_and_zero_memory_response(self):
    w=seed_world(1,'bridge',1,54322,True);a=HostAdvice(w)
    _,x=observe36(w,0,a,np.asarray((-.3,.2,0.,.6),np.float32))
    for history in (False,True):
        model=TrustAdapter(self.prior,history)
        with torch.no_grad():
            t=model(torch.from_numpy(x));v=self.prior(torch.from_numpy(x[:,:29]))
        self.assertTrue(torch.equal(t,v))

 def test_only_backtrack_under_active_cue(self):
    w=seed_world(1,'reverse',1,55422,True);a=HostAdvice(w);a.issue_backtrack(w,1.)
    _,x=observe36(w,1,a,np.ones(4,np.float32))
    model=TrustAdapter(self.prior,True)
    with torch.no_grad():
        model.head[-1].bias.fill_(1.)
        raw=model.prior(torch.from_numpy(x[:,:32]));new=model(torch.from_numpy(x))
    self.assertTrue(torch.equal(raw[:8],new[:8]));self.assertTrue(torch.equal(raw[9],new[9]))
    self.assertLessEqual(float((new[8]-raw[8]).abs()),1.25+1e-6)

 def test_source_calibration_deterministic_and_without_query_labels(self):
    a,rows=observed_calibration(self.prior,471111,'mostly_correct',True,count=3,horizon=12)
    b,rowsb=observed_calibration(self.prior,471111,'mostly_correct',True,count=3,horizon=12)
    self.assertTrue(np.array_equal(a,b));self.assertEqual(rows,rowsb)
    self.assertTrue(np.isfinite(a).all());self.assertTrue((np.abs(a)<=1.00001).all())
    self.assertTrue(all('observed_follow_advantage' in v for v in rows))
    self.assertTrue(all('target_outcome' not in v for v in rows))

 def test_policy_never_proposes_material_edit(self):
    w=seed_world(1,'shortcut',-1,54322,False);a=HostAdvice(w);a.issue_continue(w,1.)
    menu,x=observe36(w,1.,a,np.zeros(4,np.float32))
    with torch.no_grad():scores=TrustAdapter(self.prior)(torch.from_numpy(x))
    self.assertEqual(scores.shape,(10,))
    self.assertEqual(sum(c.action[0]=='BACKTRACK' for c in menu),1)
    self.assertEqual(sum(c.action[0]=='STOP' for c in menu),1)

 def test_cpu_adamw_step_updates_only_head(self):
    x,adv,g,rec,cal=fork_targets(self.prior,n=2,repeats=1,horizon=12)
    assert len(x)>0
    a,n,history=optimize(self.prior,x,adv,g,history=True,epochs=2)
    b,m,historyb=optimize(self.prior,x,adv,g,history=True,epochs=2)
    self.assertEqual((n,history),(m,historyb))
    self.assertTrue(all(torch.equal(a.head.state_dict()[key],b.head.state_dict()[key]) for key in a.head.state_dict()))
    self.assertTrue(all(not p.requires_grad for p in a.prior.parameters()))

if __name__=='__main__':unittest.main()

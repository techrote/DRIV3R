"""Self-contained research-contract regression tests; no checkpoints required."""
import unittest
import numpy as np
import torch
from torch import nn
from c22_branch_personality import world, fav_opposition, StyleSidecar, sim
from experiment import ARMS,sha
from signed_feedback import HostAdvice
from attested_stack_cue import observe32

class DummyPrior(nn.Module):
    def forward(self,x):
        # Deliberately legal-mask-agnostic scores to prove host still owns safety.
        return 0.45*x[...,7] + 0.25*x[...,21] + 0.1*x[...,0]

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_two_sided_protected_world(self):
        for fam in ('hairpin','staircase','hook','zigzag'):
            for outcome in ('reverse','ahead'):
                w=world(12341,fam,1,outcome,0,False,2,19)
                menu,x=observe32(w,.35,HostAdvice(w))
                fav,opp=fav_opposition(x)
                self.assertTrue(fav.any() and opp.any(),(fam,outcome))
                self.assertTrue(all(not c.legal for c in menu if c.target in w.protected and c.action[0] in ('INK','LIFT')))
                self.assertTrue(w.allowed(w.goal))
                self.assertGreater(len(w.protected),100)

    def test_same_state_policy_invariants(self):
        dummy=DummyPrior();model=StyleSidecar(dummy,scale=.8)
        with torch.no_grad():
            model.head[-1].bias.fill_(.5)
        for family in ('hairpin','hook'):
            for person in (-1,1):
                for directive in ('none','backtrack','continue'):
                    w=world(90211,family,person,'reverse',1,False,2,1)
                    a=HostAdvice(w)
                    if directive!='none':getattr(a,'issue_'+directive)(w,strength=.35,ttl=9)
                    cs,x=observe32(w,.35,a);y=torch.from_numpy(x)
                    base=dummy(y[:,:29]);new=model(y);f,_=fav_opposition(x)
                    for j in range(10):
                        if directive=='none' or j>=8 or not f[j]:
                            self.assertTrue(torch.equal(new[j],base[j]))
                    self.assertTrue(torch.equal(base[8:],new[8:]))
                    self.assertEqual([c.legal for c in cs], [bool(v) for v in x[:,4]])

    def test_host_applies_without_unsafe_edits(self):
        w=world(77419,'zigzag',-1,'ahead',3,True,2,7)
        before=sha(sorted(w.protected));initial=set(w.inked)
        r=sim(StyleSidecar(DummyPrior()),w,1.0,'continue',1.,77919,24,trace=True)
        self.assertEqual(r['host_rejected'],0)
        self.assertEqual(before,sha(sorted(w.protected)))
        self.assertTrue(initial.issubset(w.inked))
        self.assertEqual(len(r['trace']),r['steps'])
        self.assertEqual(len(r['actions_sha256']),64)

    def test_hint_arms_do_not_change_action_permissions(self):
        w=world(81711,'hairpin',1,'reverse',0,True,1,3)
        perms=[[c.legal for c in w.candidates(a)] for a in ARMS]
        self.assertEqual(perms[0],perms[1]);self.assertEqual(perms[0],perms[2])

if __name__=='__main__':unittest.main()

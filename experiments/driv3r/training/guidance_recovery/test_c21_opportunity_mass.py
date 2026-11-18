"""C21 bounded, host-legal opportunity-mass guards (offline synthetic research)."""
import unittest
import numpy as np
import torch
from c21_opportunity_mass import candidate_masks,masked_softmax,trace_opportunity_audit
from return_anchored_core import ReturnAnchoredGate
from c19_return_anchored import a0,make_world
from attested_stack_cue import observe32
from signed_feedback import HostAdvice

class TestC21(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # CI does not have experimental external checkpoint binaries.
        # Use a deterministic dummy prior to test *interface* invariants.
        class DummyPrior(torch.nn.Module):
            def forward(self,x):
                return x[...,7]*.4 + x[...,21]*.3 + x[...,16]*.15
        cls.prior=DummyPrior()

    def test_candidate_masks_use_only_legal_ink(self):
        w=make_world(0,'shortcut',1,22101,train=False);a=HostAdvice(w)
        _,x=observe32(w,.65,a);fav,op=candidate_masks(x[None,...])
        self.assertFalse(bool((fav&op).any()))
        self.assertTrue(np.all(x[:8,4][fav[0]]>.5))

    def test_model_changes_only_backtrack_on_attested_active_advice(self):
        model=ReturnAnchoredGate(self.prior,width=12,scale=.8).eval()
        ref=a0(self.prior)
        # Simulate a nonzero trained head: absence of cues must still be exact.
        with torch.no_grad():model.head[-1].bias.fill_(.7)
        for outcome in ('reverse','ahead','shortcut','bridge'):
            for directive in ('none','backtrack','continue'):
                w=make_world(0,outcome,1,284000,train=False);a=HostAdvice(w)
                if directive!='none':getattr(a,'issue_'+directive)(w)
                _,x=observe32(w,.65,a);xt=torch.from_numpy(x)
                with torch.no_grad():got=model(xt);old=self.prior(xt[:,:29]);anch=ref(xt)
                self.assertTrue(torch.equal(got[:8],old[:8]))
                self.assertTrue(torch.equal(got[9:],old[9:]))
                if directive=='none':
                    self.assertTrue(torch.equal(got,old))
                else:
                    self.assertTrue(torch.equal(anch[:8],old[:8]))
                    self.assertTrue(torch.equal(anch[9:],old[9:]))

    def test_preferred_turn_available_without_opposing_turn(self):
        x=np.zeros((1,10,32),np.float32)
        x[0,0,0]=1;x[0,0,4]=1;x[0,0,21]=1
        fav,opp=candidate_masks(x)
        self.assertEqual(int(fav.sum()),1)
        self.assertEqual(int(opp.sum()),0)

    def test_masked_probability_is_normalized_and_never_illegal(self):
        w=make_world(0,'shortcut',-1,22101,train=False);a=HostAdvice(w)
        _,x=observe32(w,.35,a);xt=torch.from_numpy(x)
        with torch.no_grad():p=masked_softmax(self.prior(xt[:,:29]),xt)
        self.assertAlmostEqual(float(p.sum()),1.,places=5)
        self.assertEqual(float(p[xt[:,4]<.5].sum()),0.)

if __name__=='__main__':unittest.main()

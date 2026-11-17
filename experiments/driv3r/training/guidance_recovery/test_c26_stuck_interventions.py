"""Source-local, synthetic-only regression guards for C26."""
import copy
import unittest
import numpy as np
import torch
from experiment import World
from signed_feedback import HostAdvice
from c26_stuck_interventions import (host_stuckness,intervention_trigger,observe40,
        InterventionHead,Control,induce,rollout)
from c24_history_trust import seed_world
from c19_return_anchored import a0
from personality_retention_core import load_c12
from pathlib import Path


class TestC26(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]/'prerequisites'
        names=('cue_calibrated','adversarial_adapter','personality_gate','topology_transfer',
               'persistent_feedback','signed_feedback','selective_continue')
        cls.prior=load_c12([root/f'guidance_recovery_{n}.safetensors' for n in names])
        torch.set_num_threads(1)

    def test_stuckness_follows_only_executed_steps_and_current_goal(self):
        w=World(17,'islands',1)
        # No events: seeded history alone does not count as observed stuckness.
        w.history=[(1,1),(2,1),(2,2)]
        self.assertTrue(np.array_equal(host_stuckness(w),np.zeros(4,np.float32)))
        g=w.goal
        w.events=[('INK',(1,1),(0,1),g),('INK',(0,1),(1,1),g),
                  ('LIFT',(1,1),(0,1),g),('LIFT',(0,1),(1,1),g)]
        f=host_stuckness(w)
        self.assertTrue(intervention_trigger(f))
        self.assertGreaterEqual(f[1],.25)
        w.goal=(7,7)
        self.assertTrue(np.array_equal(host_stuckness(w),np.zeros(4,np.float32)))

    def test_no_cue_and_untriggered_are_exact_baseline(self):
        w=seed_world(0,'reverse',1,777,False)
        a=HostAdvice(w)
        cs,x=observe40(w,1.,a)
        m=InterventionHead(self.prior,True)
        baseline=Control(self.prior,True)
        # Force nonzero learned correction to exercise masks.
        with torch.no_grad():m.head[-1].bias.fill_(2.)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(torch.from_numpy(x)),baseline(torch.from_numpy(x))))
        a.issue_backtrack(w,strength=1.)
        _,y=observe40(w,1.,a)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(torch.from_numpy(y)),baseline(torch.from_numpy(y))))

    def test_nonzero_stuck_cue_only_changes_legal_backtrack(self):
        w=seed_world(0,'reverse',1,1138,False)
        w,_=induce(w,a0(self.prior),.35,1199,5,'badturn')
        a=HostAdvice(w);a.issue_backtrack(w,strength=1.)
        _,x=observe40(w,.35,a)
        m=InterventionHead(self.prior,True);baseline=Control(self.prior,True)
        with torch.no_grad():m.head[-1].bias.fill_(2.)
        with torch.no_grad():
            pred=m(torch.from_numpy(x));old=baseline(torch.from_numpy(x))
        self.assertTrue(torch.equal(pred[:8],old[:8]))
        self.assertTrue(torch.equal(pred[9],old[9]))
        self.assertLessEqual(abs(float(pred[8]-old[8])),.85+1e-5)
        self.assertTrue(np.array_equal(x[:,:32],observe40(w,.35,a)[1][:,:32]))
        self.assertFalse(np.any(~np.isfinite(x)))

    def test_induced_prefix_does_not_touch_protected(self):
        for outcome in ('reverse','ahead','shortcut','bridge'):
            w=seed_world(1,outcome,-1,20018,False)
            before=set(w.protected)
            w,trace=induce(w,a0(self.prior),1.,14009,5,'badturn')
            self.assertEqual(w.protected,before)
            self.assertEqual(w.invalid_attempts,0)
            self.assertEqual(len(trace),len(w.events))

    def test_rollout_legal_and_matched_prefix(self):
        prior=self.prior
        common=dict(i=1,outcome='ahead',person=-1,arm=.35,
                    directive='continue',strength=1.,mode='badturn',seed=3012,horizon=8,trace=True)
        a=rollout(Control(prior,True),a0(prior),**common)
        b=rollout(InterventionHead(prior,True),a0(prior),**common)
        self.assertEqual(a['prefix_sha256'],b['prefix_sha256'])
        self.assertEqual(a['prefix'],b['prefix'])
        self.assertEqual(a['host_rejected'],0)
        self.assertEqual(b['host_rejected'],0)
        self.assertEqual(a['illegal_raw_top1'],0)
        self.assertEqual(b['illegal_raw_top1'],0)


if __name__=='__main__':unittest.main()
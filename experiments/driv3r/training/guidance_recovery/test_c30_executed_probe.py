"""Bounded, offline C30 acceptance tests; no external services or paid jobs."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from c30_executed_probe import (EvidenceHead,AdaptiveRecovery,OptionHead,ProbeHead,
    collect_training,evidence,verified_forward,rollout,train)
from c30_three_step_validation import world,validate_geometry
from c29_probe_recovery import fresh_indices
from signed_feedback import HostAdvice
from c28_temporal_option import executed_world

class C30Test(unittest.TestCase):
    def test_three_step_oracle_solvable_but_deceptive(self):
        geometry=validate_geometry()
        self.assertEqual(len(geometry),16)
        self.assertEqual({g['shortest'] for g in geometry if g['kind']=='three_decoy'},{21})
        self.assertEqual({g['shortest'] for g in geometry if g['kind']=='three_tunnel'},{7})
        for g in geometry:
            self.assertEqual(len(g['prefix'])>=4,True)
            if g['kind']=='three_decoy':
                self.assertEqual(g['prefix'][3]['fresh_n'],0)
                self.assertEqual(g['prefix'][3]['visited_goalward_n'],0)
    def test_training_features_only_executed_host_observations(self):
        x,y,r=collect_training(300011,4)
        self.assertEqual(len(x),len(y))
        self.assertEqual(len(y),len(r))
        self.assertTrue(bool((x>=0).all()))
        self.assertGreater(int(y.sum()),0)
        self.assertTrue(all(entry['menu_sha256'] for entry in r))
    def test_learned_head_and_fixed_seed_optimizer_replicate(self):
        h,a,_=train(300011,2,4);hh,aa,_=train(300011,2,4)
        self.assertEqual(a['training_sha256'],aa['training_sha256'])
        for k,v in h.state_dict().items():self.assertTrue(torch.equal(v,hh.state_dict()[k]))
    def test_host_legality_and_goal_cancellation(self):
        w=world(4,'three_decoy',1,212)
        adv=HostAdvice(w);adv.issue_backtrack(w,strength=1.,ttl=9)
        rec=AdaptiveRecovery(OptionHead(),ProbeHead(),EvidenceHead())
        self.assertEqual(rec.phase,'idle')
        class FakeBase:
            def __call__(self,x):return torch.zeros((1,10))
        # Refuse to depend on base shape here: check explicit state release on goal change.
        rec.phase='probe';rec.goal=tuple(w.goal)
        w.goal=(12,12)
        rec._release()
        self.assertEqual(rec.phase,'released')
        self.assertFalse(rec.option.active)
    def test_fresh_and_old_goalward_distinction(self):
        w=world(0,'three_decoy',1,123)
        menu=w.candidates(.35)
        self.assertTrue(fresh_indices(w,menu))
        self.assertFalse(verified_forward(w,menu))
        for _ in range(3):
            menu=w.candidates(.35);j=fresh_indices(w,menu)[0]
            self.assertTrue(w.apply(menu[j]))
        self.assertFalse(fresh_indices(w,w.candidates(.35)))
        self.assertEqual(w.invalid_attempts,0)

if __name__=='__main__':unittest.main()

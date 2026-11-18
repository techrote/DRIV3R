"""Focused guardrails for FEASIBILITY-SHIELD-001 (no CLI training)."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import unittest
from .feasibility_shield_lab import decide, episode, verify_trace, ARMS, SEEDS
from .detour_recovery_lab import fixture, prefix_plan
from .goal_conditioning_lab import GoalRanker
from .world import Spec, specs

ROOT = Path(__file__).resolve().parents[5] / 'run' / 'checkpoints'

class ShieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp0=specs('development',1)[0]
        cls.sp=Spec(cls.sp0.split,cls.sp0.family,cls.sp0.seed,cls.sp0.mode,'trap_corridor')
        cls.goal=(55,56)
        cls.model=GoalRanker.load(ROOT/'711-conditioned.npz')

    def test_split_guard(self):
        self.assertEqual(self.sp.split,'development')
        self.assertEqual(self.sp0.family,'pickets')
        self.assertNotIn(self.sp.family,('archipelago','staggered_vault'))
        self.assertEqual(set(ARMS),{'neural','planner','late24','shield0','shield8'})

    def test_preflight_planner(self):
        w=fixture(self.sp,3)
        p=prefix_plan(w,self.goal,64)
        self.assertIsNotNone(p)
        self.assertEqual(p[0],'BACKTRACK')
        self.assertEqual(w.logical_moves,3)

    def test_all_arms_receipt_replay(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                row,trace=episode(self.sp,3,self.goal,711,arm,self.model)
                self.assertGreater(verify_trace(trace,self.model)[0],3)
                self.assertLessEqual(row['logical_moves'],64)
                self.assertEqual(row['protected_material_edits'],0)
                if arm in ('planner','late24','shield0','shield8'):
                    self.assertTrue(row['arrived'])

    def test_probabilities_tamper_rejected(self):
        _,trace=episode(self.sp,3,self.goal,711,'neural',self.model)
        bad=deepcopy(trace)
        p=next(e for e in bad['entries'] if e['probabilities'] is not None)['probabilities']
        k=next(iter(p));p[k]=0.9
        with self.assertRaises(AssertionError):verify_trace(bad,self.model)

    def test_illegal_action_tamper_rejected(self):
        _,trace=episode(self.sp,3,self.goal,711,'shield0',self.model)
        bad=deepcopy(trace)
        bad['entries'][0]['action']='MOVE_BAD'
        with self.assertRaises(AssertionError):verify_trace(bad,self.model)

if __name__=='__main__':unittest.main()

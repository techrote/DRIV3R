"""Prespecified trigger, exact action-receipt and probability tamper tests.

Synthetic DEVELOPMENT test only; no HELDOUT and no native CyberSand acceptance.
"""
from __future__ import annotations
import copy
from pathlib import Path
import unittest
from . import topology_adaptive_handoff_lab as lab
from .goal_conditioning_lab import GoalRanker
from .world import specs

CHECKPOINTS = Path(__file__).resolve().parents[4] / 'checkpoints'

class AdaptiveHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp=specs('development',1)[0]
        cls.checkpoint=CHECKPOINTS/'711-conditioned.npz'

    def test_sealed_split_and_distinct_masks(self):
        for split in ('train','heldout'):
            with self.assertRaises(ValueError):lab.overlay_world(specs(split,1)[0],lab.LAYOUTS[0])
        hashes=[lab.overlay_world(self.sp,layout).base_sha() for layout in lab.LAYOUTS]
        self.assertEqual(len(set(hashes)),4)

    def test_prespecified_triggers_are_exclusive(self):
        self.assertEqual(lab.SCHEDULE,(12,30))
        self.assertEqual(lab.CAP,64)
        self.assertEqual(len(set(lab.ARMS)),len(lab.ARMS))
        if not self.checkpoint.exists():self.skipTest('requires immutable local checkpoint')
        w=lab.overlay_world(self.sp,'horizontal_switchback');model=GoalRanker.load(self.checkpoint)
        goal=lab.GOALS[0]
        # A second-switch trigger cannot fire at the first switch.
        _,_,_,_,calls,_=lab.plan_or_neural(w,goal,model,'second_switch',12,1,12,0,None,False)
        self.assertEqual(calls,0)
        _,_,_,_,calls,_=lab.plan_or_neural(w,goal,model,'second_switch',30,2,30,0,None,False)
        self.assertEqual(calls,1)
        _,_,_,_,calls,_=lab.plan_or_neural(w,goal,model,'second_switch_stall3',30,2,30,0,None,False)
        self.assertEqual(calls,0)
        _,_,_,_,calls,_=lab.plan_or_neural(w,goal,model,'second_switch_stall3',30,2,30,3,None,False)
        self.assertEqual(calls,1)
        _,_,_,_,calls,_=lab.plan_or_neural(w,goal,model,'each_switch',12,1,12,0,None,False)
        self.assertEqual(calls,1)
        _,_,_,_,calls,_=lab.plan_or_neural(w,goal,model,'budget_slack10',12,1,12,0,None,False)
        self.assertEqual(calls,0)

    def test_all_controller_replay_and_tamper_rejection(self):
        if not self.checkpoint.exists():self.skipTest('requires immutable local checkpoint')
        model=GoalRanker.load(self.checkpoint)
        for arm in lab.ARMS:
            row,tr=lab.episode(self.sp,'horizontal_switchback',0,711,arm,model)
            c,v=lab.verify_trace(tr,model)
            self.assertEqual(row['decisions'],c)
            self.assertGreaterEqual(v,0)
            self.assertLessEqual(row['logical_moves'],64)
            self.assertEqual(row['illegal_material_edits'],0)
        row,tr=lab.episode(self.sp,'horizontal_switchback',0,711,'second_switch',model)
        self.assertTrue(tr['entries'])
        for field,damage in (('menu',lambda e:e['menu'].append('NOT_LEGAL')),
                             ('receipt',lambda e:e['receipt'].update(cursor_after=[0,0])),
                             ('probabilities',lambda e:e.update(probabilities={'STOP':1}))):
            bad=copy.deepcopy(tr)
            # A neural entry must be chosen for probability tamper.
            entry=next((e for e in bad['entries'] if e['probabilities'] is not None),bad['entries'][0])
            damage(entry)
            with self.assertRaises(AssertionError):lab.verify_trace(bad,model)
        bad=copy.deepcopy(tr)
        self.assertTrue(bad['switches'])
        bad['switches'][0]['next_goal']=[0,0]
        with self.assertRaises(AssertionError):lab.verify_trace(bad,model)

if __name__=='__main__':unittest.main()

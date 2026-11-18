"""Focused source plus optional full local archive qualification."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from .retarget_handoff_lab import episode,fixture,verify_trace,run,verify,SEEDS,ARMS,LAYOUTS,CAP
from .goal_conditioning_lab import GoalRanker
from .world import Spec
from .detour_recovery_lab import GOALS

class RetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cp=os.environ.get('DRIV3R_RETARGET_CHECKPOINTS')
        if not cp: raise unittest.SkipTest('Checkpoint archive not supplied; run source-only syntax separately')
        cls.cp=Path(cp); cls.model=GoalRanker.load(cls.cp/'711-conditioned.npz')

    def test_all_arms_source_replay(self):
        spec=Spec('development','islands',3200,'motif','trap_corridor')
        for layout in ('original','north_ledge'):
            for arm in ARMS:
                with self.subTest(layout=layout,arm=arm):
                    row,trace=episode(spec,layout,GOALS,711,arm,self.model)
                    n,p=verify_trace(trace,self.model)
                    self.assertGreaterEqual(n,5)
                    self.assertLessEqual(row['logical_moves'],CAP)
                    self.assertEqual(row['material_protected_edits'],0)
                    self.assertLessEqual(row['planner_calls'],2)

    def test_tampered_action_rejected(self):
        spec=Spec('development','pickets',3000,'wall','trap_corridor')
        _,trace=episode(spec,'original',GOALS,711,'switch_plan',self.model)
        bad=copy.deepcopy(trace);bad['entries'][0]['action']='STOP'
        with self.assertRaises(AssertionError): verify_trace(bad,self.model)

    def test_stale_target_rejected(self):
        spec=Spec('development','pickets',3000,'wall','trap_corridor')
        _,trace=episode(spec,'original',GOALS,711,'switch_plan',self.model)
        bad=copy.deepcopy(trace)
        event=next(e for e in bad['entries'] if e['tick']==8)
        event['goal']=list(GOALS[0])
        with self.assertRaises(AssertionError):verify_trace(bad,self.model)

    def test_full_archive_if_requested(self):
        folder=os.environ.get('DRIV3R_RETARGET_ARCHIVE')
        if not folder:self.skipTest('Full archive not supplied')
        evidence=verify(Path(folder),self.cp)
        self.assertEqual(evidence['episodes'],len(evidence['rows']))
        self.assertEqual(evidence['structural_heldout_opened'],False)

if __name__=='__main__': unittest.main()

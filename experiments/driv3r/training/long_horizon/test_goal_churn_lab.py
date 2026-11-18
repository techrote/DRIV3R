"""GOAL-CHURN-001 deterministic counterfactual and archive integrity checks."""
import copy
import json
import os
import unittest
from pathlib import Path
from . import goal_churn_lab as g
from .goal_conditioning_lab import GoalRanker
from .world import Spec

class GoalChurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cps=Path(os.environ.get('DRIV3R_CHURN_CHECKPOINTS','/nonexistent'))
        if not all((cls.cps/f'{s}-conditioned.npz').exists() for s in g.SEEDS):
            raise unittest.SkipTest('Frozen checkpoint archive required; set DRIV3R_CHURN_CHECKPOINTS')
        cls.model=GoalRanker.load(cls.cps/'711-conditioned.npz')
        base=g.specs('development',1)[0]
        cls.sp=Spec(base.split,base.family,base.seed,base.mode,'trap_corridor')
        cls.row,cls.trace=g.episode(cls.sp,'original',0,711,'absolute24',(8,24),cls.model)

    def test_two_stage_changes_are_real(self):
        self.assertEqual([s['tick'] for s in self.trace['switches']],[8,24])
        self.assertEqual(self.row['switches_seen'],2)
        self.assertLessEqual(self.row['logical_moves'],64)
        self.assertEqual(self.row['illegal_material_edits'],0)

    def test_entire_trace_recomputes(self):
        self.assertEqual(g.verify_trace(self.trace,self.model)[0],len(self.trace['entries']))

    def test_tamper_probability_rejected(self):
        t=copy.deepcopy(self.trace)
        target=next(x for x in t['entries'] if x['probabilities'] is not None)
        target['probabilities']['STOP']=0.991
        with self.assertRaises(AssertionError):g.verify_trace(t,self.model)

    def test_tamper_switch_state_rejected(self):
        t=copy.deepcopy(self.trace);t['switches'][0]['pre_sha256']='0'*64
        with self.assertRaises(AssertionError):g.verify_trace(t,self.model)

    def test_tamper_action_rejected(self):
        t=copy.deepcopy(self.trace);t['entries'][0]['action']='STOP'
        with self.assertRaises(AssertionError):g.verify_trace(t,self.model)

    def test_goal_switch_invalidates_cached_plan(self):
        # The first goal's cached planner path must not continue into phase 1.
        _,t=g.episode(self.sp,'original',0,711,'plan_every_stage',(8,24),self.model)
        self.assertEqual([s['tick'] for s in t['switches']],[8,24])
        self.assertEqual(sum(e['planner_calls'] for e in t['entries']),3)
        self.assertTrue(all(e['planner_calls']==1 for e in t['entries'] if e['source']=='new_plan'))

    def test_archived_replay(self):
        archive=os.environ.get('DRIV3R_CHURN_RUN')
        if not archive: self.skipTest('Set DRIV3R_CHURN_RUN for full independent archive replay')
        ev=g.verify(Path(archive),self.cps)
        self.assertGreater(ev['verified_action_receipts'],0)

if __name__=='__main__':unittest.main()

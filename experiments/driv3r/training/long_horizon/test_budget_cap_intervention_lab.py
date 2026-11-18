"""BUDGET-CAP-INTERVENTION-001 source-backed, no-remote qualification."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from . import budget_cap_intervention_lab as lab, budget_calibration_lab as cal

class TestCausalBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=lab.select_cases()

    def test_identical_state_menu_and_ink_under_all_caps(self):
        for c in self.cases[:36]:
            w=c['state'];state=w.state_sha();base=w.menu(12)
            for b in lab.BUDGETS:
                for arm in lab.ARMS:
                    menu,x,cap=lab.intervention(w,c['goal'],arm,b)
                    self.assertEqual(menu,base)
                    self.assertEqual(cap,w.recoveries+b)
                    self.assertEqual(w.state_sha(),state)
                    self.assertEqual(x.shape[0],cal.policy.ACTION_MAX)

    def test_exact_original_normalization_at_anchor(self):
        for c in self.cases[:48]:
            for arm in lab.ARMS:
                m,x,cap=lab.intervention(c['state'],c['goal'],arm,c['budget'])
                orig,old=cal.matrix(c['state'],c['goal'],arm)
                self.assertEqual(cap,12)
                self.assertEqual(m,orig)
                np.testing.assert_allclose(x,old,rtol=0,atol=1e-12)

    def test_blind_model_has_no_budget_feature_leak(self):
        for c in self.cases[:24]:
            cols=[lab.intervention(c['state'],c['goal'],'budget_blind',b)[1] for b in lab.BUDGETS]
            for other in cols[1:]:np.testing.assert_array_equal(cols[0],other)

    def test_oracle_monotone_and_path_depth_genuine(self):
        for c in self.cases:
            w=c['state']; goal=c['goal'];depth=c['depth']
            if goal in w.path:self.assertEqual(depth,len(w.path)-1-w.path.index(goal))
            votes=[depth<=b for b in lab.BUDGETS]
            self.assertEqual(votes,sorted(votes))
            self.assertLessEqual(w.logical_moves,64)

    def test_reject_unsupported_budget(self):
        c=self.cases[0]
        with self.assertRaises(ValueError):lab.intervention(c['state'],c['goal'],'calibrated',0)

    def test_integrity_fails_when_checkpoint_changed(self):
        self.assertEqual(len(lab.EXPECTED_CHECKPOINT_SHA),6)
        with self.assertRaises(ValueError):lab.load_models(Path('/nonexistent/checkpoint-directory'))

if __name__=='__main__':unittest.main()

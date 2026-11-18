"""Budget-calibration synthetic-only boundaries and reproducibility tests."""
import unittest
import numpy as np
from . import budget_calibration_lab as lab
from .world import replay

class BudgetCalibrationTests(unittest.TestCase):
    def test_partition_motifs(self):
        a={r[1] for r in lab.TRAIN_MOTIFS};b={r[1] for r in lab.DEVELOPMENT_MOTIFS}
        self.assertTrue(a.isdisjoint(b));self.assertEqual(len(a),14);self.assertEqual(len(b),20)
        self.assertTrue(all(r[0]==12 for r in lab.TRAIN_MOTIFS))
        self.assertTrue(all(r[0]==14 for r in lab.DEVELOPMENT_MOTIFS))
    def test_budgets_disjoint(self):
        self.assertEqual(set(lab.TRAIN_BUDGETS)&{1,5},set())
    def test_matched_legal_states(self):
        for i,c in enumerate(lab.cases('development')):
            self.assertEqual(c['state'].menu(12),lab.matrix(c['state'],c['goal'],'calibrated')[0])
            self.assertEqual(c['budget'],12-c['state'].recoveries)
            self.assertLessEqual(c['state'].logical_moves,64)
            if i in (0,37,239,479):
                w=replay(c['spec'],c['state'].decisions)
                self.assertEqual(w.state_sha(),c['state'].state_sha())
    def test_features_no_explicit_slack(self):
        c=lab.cases('train')[0]
        menu,x=lab.matrix(c['state'],c['goal'],'calibrated')
        _,b=lab.matrix(c['state'],c['goal'],'budget_blind')
        self.assertTrue(np.allclose(x[:,-2:],0));self.assertTrue(np.allclose(b[:,[18,30,34,lab.alias.WIDTH-3]],0))
        self.assertTrue(np.array_equal(x[:,-4],b[:,-4]))
    def test_alias_pair_cannot_be_distinguished_by_goal_only_features(self):
        cs=lab.cases('development')
        # Both path orders paint exactly the same cells; only history differs.
        for i in (0,18,100,238):
            a,b=cs[2*i:2*i+2]
            self.assertEqual(a['state'].ink,b['state'].ink)
            self.assertEqual(a['state'].menu(12),b['state'].menu(12))
            goal_a=lab.alias.prior.prior.goal.matrix(a['state'],a['goal'],128,False)[1]
            goal_b=lab.alias.prior.prior.goal.matrix(b['state'],b['goal'],128,False)[1]
            self.assertTrue(np.array_equal(goal_a,goal_b))

    def test_stack_rule_is_consistent_with_host_actions(self):
        for c in lab.cases('development'):
            self.assertEqual(c['label'], 'BACKTRACK' if c['depth']<=c['budget'] else 'STOP')
            self.assertIn(c['label'], c['state'].menu(12))

    def test_supervised_corpus(self):
        X,M,Y,info=lab.corpus(36,'calibrated')
        self.assertGreater(len(Y),100)
        self.assertGreater(info['actions'].get('BACKTRACK',0),0)
        self.assertGreater(info['actions'].get('STOP',0),0)
        self.assertEqual(len(X),len(Y));self.assertEqual(len(M),len(Y))

if __name__=='__main__':unittest.main()

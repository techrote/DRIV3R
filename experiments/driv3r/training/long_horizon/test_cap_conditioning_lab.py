"""Source-only invariants; optional archived checkpoint/receipt replay via CAP_RUN."""
import os, unittest
import numpy as np
from pathlib import Path
from . import cap_conditioning_lab as cap
from . import budget_calibration_lab as prior

class CapConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training=prior.cases('train')
        cls.development=prior.cases('development')

    def test_disjoint_development_motifs_and_specs(self):
        self.assertEqual((len(self.training),len(self.development)),(336,480))
        a={c['case'] for c in self.training};b={c['case'] for c in self.development}
        self.assertFalse(a & b)
        self.assertTrue(all(c['spec'].split=='train' for c in self.training))
        self.assertTrue(all(c['spec'].split=='development' for c in self.development))

    def test_no_change_to_world_menu_or_noncap_features(self):
        for c in self.training[:12]+self.development[:12]:
            w=c['state'];sha=w.state_sha()
            menu,base=prior.matrix(w,c['goal'],'calibrated')
            for b in range(1,7):
                legal,x,_,oracle=cap.counterfactual(c,b)
                self.assertEqual(menu,legal)
                self.assertEqual(w.state_sha(),sha)
                self.assertEqual(oracle,'BACKTRACK' if c['depth']<=b else 'STOP')
                indices=[i for i in range(base.shape[1]) if i not in cap.INTERVENED]
                np.testing.assert_array_equal(x[:,indices],base[:,indices])

    def test_oracle_budget_order(self):
        for c in self.development:
            labels=[cap.counterfactual(c,b)[-1] for b in range(1,7)]
            self.assertEqual(labels,sorted(labels,key=lambda q:q!='STOP'))  # STOP then BACKTRACK

    def test_blind_cap_invariance(self):
        for c in self.development[:16]:
            arr=[cap.counterfactual(c,b,'cap_blind')[1] for b in (1,3,5)]
            np.testing.assert_array_equal(arr[0],arr[1])
            np.testing.assert_array_equal(arr[0],arr[2])

    def test_balanced_bounded_training_cases(self):
        for arm in cap.ARMS:
            X,M,Y,meta=cap.corpus(36,arm)
            self.assertEqual(len(Y),1008)
            self.assertEqual(len(Y),meta['examples'])
            self.assertEqual(X.shape[1:],(cap.policy.ACTION_MAX,cap.alias.WIDTH))
            self.assertTrue(M.any(axis=1).all())
            self.assertTrue((Y<M.sum(axis=1)).all())

    def test_archive_replay_if_given(self):
        path=os.environ.get('DRIV3R_CAP_RUN')
        if not path:self.skipTest('optional immutable run output not supplied')
        verified=cap.verify(Path(path))
        self.assertEqual(verified['decision_probability_vectors_recomputed'],8640)
        self.assertGreater(verified['legal_continuation_actions_replayed'],0)

if __name__=='__main__':unittest.main()

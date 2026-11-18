"""Source-only tests; optional full checkpoint replay via DRIV3R_COST_RUN."""
import os,unittest
from pathlib import Path
import numpy as np
from . import recovery_cost_sweep_lab as lab
from . import cap_conditioning_lab as cap, budget_calibration_lab as prior

class RecoveryCostSweepTests(unittest.TestCase):
    def test_isolated_splits(self):
        tr=prior.cases('train');dev=prior.cases('development')
        self.assertEqual((len(tr),len(dev)),(336,480))
        self.assertFalse({c['case'] for c in tr}&{c['case'] for c in dev})
        self.assertTrue(all(c['spec'].split=='development' for c in dev))
    def test_cost_resampling_changes_only_weights(self):
        X,M,Y,_=lab.resampled(36,1)
        idx=np.arange(len(Y))
        for factor in (2,4):
            a,b,c,meta=lab.resampled(36,factor)
            np.testing.assert_array_equal(a[:len(Y)],X)
            np.testing.assert_array_equal(b[:len(Y)],M)
            np.testing.assert_array_equal(c[:len(Y)],Y)
            self.assertEqual(meta['resampled_n'],len(Y)+(factor-1)*meta['backtrack_original'])
        self.assertTrue(np.isfinite(X).all())
    def test_controlled_cap_and_menu(self):
        for c in prior.cases('development')[:30]:
            before=c['state'].state_sha()
            for remaining in lab.CAPS:
                menu,x,cap_,oracle=cap.counterfactual(c,remaining)
                self.assertEqual(menu,c['state'].menu(cap_))
                self.assertEqual(oracle,'BACKTRACK' if c['depth']<=remaining else 'STOP')
                self.assertTrue(np.isfinite(x).all())
            self.assertEqual(before,c['state'].state_sha())
    def test_full_receipt_replay_if_available(self):
        path=os.getenv('DRIV3R_COST_RUN')
        if not path:self.skipTest('full local checkpoint evidence not installed')
        proof=lab.verify(Path(path));self.assertEqual(proof['verified_first_vectors'],8640)
        self.assertEqual(proof['verified_continuations'],864)

if __name__=='__main__':unittest.main()

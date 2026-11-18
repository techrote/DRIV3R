"""Focused, no-GPU tests for the controlled path-history alias experiment."""
import unittest
from pathlib import Path
import numpy as np
from . import history_alias_lab as h

class TestHistoryAlias(unittest.TestCase):
    def test_exact_source_partition_and_identical_ink(self):
        train=list(h.pairs('train'));dev=list(h.pairs('development'))
        self.assertEqual((len(train),len(dev)),(36,40))
        self.assertTrue(set(p['case'] for p in train).isdisjoint(p['case'] for p in dev))
        for p in train+dev:
            a,b=p['positive'],p['negative']
            self.assertEqual(a.ink,b.ink)
            self.assertEqual(a.materials,b.materials)
            self.assertEqual(a.protected,b.protected)
            self.assertEqual((a.cursor,a.heading,a.steps,a.recoveries,a.last_action),
                             (b.cursor,b.heading,b.steps,b.recoveries,b.last_action))
            self.assertEqual(a.menu(),b.menu())
            self.assertEqual(a.recoveries,9)
            self.assertGreater(p['unrecoverable_depth'],12-b.recoveries)
            self.assertLessEqual(p['recoverable_depth'],12-a.recoveries)
    def test_true_alias_of_blind_observation(self):
        for p in list(h.pairs('development'))[:8]:
            a,b=p['positive'],p['negative'];g=p['goal']
            aa=h.matrix(a,g,128,True)[1];bb=h.matrix(b,g,128,True)[1]
            aa[:,h.prior.prior.goal.FEATURE_DIM:h.prior.WIDTH]=0
            bb[:,h.prior.prior.goal.FEATURE_DIM:h.prior.WIDTH]=0
            self.assertTrue(np.array_equal(aa,bb))
            self.assertFalse(np.array_equal(h.matrix(a,g,128)[1],h.matrix(b,g,128)[1]))
    def test_balanced_legal_training(self):
        x,m,y,counts=h.examples(36)
        self.assertEqual(counts['BACKTRACK'],counts['STOP'])
        self.assertEqual(x.shape[1:],(7,h.WIDTH))
        self.assertTrue(m[np.arange(len(y)),y].all())
        self.assertEqual(h.STAGES,(36,72,128))
    def test_analytic_path_reachability_and_legal_receipts(self):
        for p in list(h.pairs('development'))[:8]:
            pos=p['positive'];goal=p['goal']
            for _ in range(p['recoverable_depth']):
                self.assertIn('BACKTRACK',pos.menu(12));pos.apply('BACKTRACK',12)
            self.assertEqual(pos.cursor,goal)
            self.assertNotEqual(p['negative'].cursor,goal)
    def test_model_checkpoint_and_budget(self):
        m=h.Ranker(619)
        self.assertEqual(m.p['w1'].shape[0],h.WIDTH)
        self.assertEqual(sum(x.size for x in m.p.values()),1153)
        with self.assertRaises(ValueError):h.run(Path('/tmp/unused-history-alias-budget'),updates=100000)

if __name__=='__main__':unittest.main()

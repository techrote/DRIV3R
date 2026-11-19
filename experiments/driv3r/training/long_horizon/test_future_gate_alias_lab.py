"""CPU-only structural-alias, legality, optimizer and receipt-tamper tests."""
from __future__ import annotations
import copy
import unittest
import numpy as np
from . import future_gate_alias_lab as lab


class FutureGateAliasTests(unittest.TestCase):
    def setUp(self):
        self.early = lab.Spec('development',3000,36,'early')
        self.late = lab.Spec('development',3000,36,'late')

    def test_heldout_remains_sealed(self):
        for split, seed in [('heldout',9000),('development',1000),('train',3000)]:
            with self.assertRaises(ValueError):
                lab.Spec(split,seed,36,'early')

    def test_host_like_legality_and_history(self):
        w = lab.World(self.early)
        self.assertIn('STOP',w.menu())
        self.assertNotIn('BACKTRACK',w.menu())
        w.step('MOVE_E')
        self.assertIn('BACKTRACK',w.menu())
        self.assertNotIn('MOVE_W',w.menu())
        w.step('BACKTRACK')
        self.assertEqual(w.cursor,w.start)
        self.assertEqual(w.logical_moves,1)
        with self.assertRaises(ValueError):
            w.step('MOVE_E')
        w.step('STOP')
        self.assertTrue(w.stopped)

    def test_identical_present_different_future(self):
        ea,la = lab.World(self.early),lab.World(self.late)
        for _ in range(5):
            ea.step('MOVE_E')
            la.step('MOVE_E')
        ea.event(5,'close')
        la.event(5,'close')
        plan = lab.bfs_plan(lab.World(self.early))[5:]
        a = lab.observed_features(ea,plan)
        b = lab.observed_features(la,plan)
        self.assertTrue(np.array_equal(a,b))
        self.assertEqual(ea.menu(),la.menu())
        self.assertTrue(a[1])

    def test_matched_preferred_handoffs(self):
        for h in lab.HORIZONS:
            cap = 128 if h==128 else 64
            for mode,want in [('early','lazy'),('late','overlap')]:
                sp=lab.Spec('development',3000,h,mode)
                a=lab.episode(sp,'lazy',paint_cap=cap)
                b=lab.episode(sp,'overlap',paint_cap=cap)
                result='overlap' if lab.utility(b)>lab.utility(a) else 'lazy'
                self.assertEqual(result,want,(h,mode))

    def test_native_ceiling_not_shadow_success(self):
        sp=lab.Spec('development',3000,128,'early')
        native=lab.episode(sp,'lazy')
        shadow=lab.episode(sp,'lazy',paint_cap=128)
        self.assertFalse(native['arrived'])
        self.assertEqual(native['paint_moves'],64)
        self.assertEqual(native['cohort'],'native64')
        self.assertTrue(shadow['arrived'])
        self.assertGreater(shadow['paint_moves'],64)
        self.assertEqual(shadow['cohort'],'shadow128')
        with self.assertRaises(ValueError):
            lab.episode(lab.Spec('development',3000,72,'early'),'lazy',paint_cap=128)

    def test_replay_and_receipt_tamper(self):
        _,trace=lab.episode(self.late,'overlap',record=True)
        self.assertEqual(lab.replay(trace),len(trace['entries']))
        for key,bad in [('action','STOP'),('pre_sha256','false'),('menu',['STOP'])]:
            mutated=copy.deepcopy(trace)
            mutated['entries'][0][key]=bad
            with self.subTest(key=key),self.assertRaises((AssertionError,ValueError)):
                lab.replay(mutated)
        mutated=copy.deepcopy(trace)
        mutated['entries'][5]['event']['cell']=[99,99]
        with self.assertRaises(AssertionError):
            lab.replay(mutated)

    def test_optimizer_determinism_and_irreducible_alias(self):
        xs,ys,conflicts,_=lab.corpus()
        self.assertEqual(len(xs),432)
        self.assertEqual(len(conflicts),144)
        a,la=lab.train(xs,ys)
        b,lb=lab.train(xs,ys)
        self.assertEqual(a.checkpoint_sha(),b.checkpoint_sha())
        self.assertEqual(la,lb)
        self.assertEqual(la[-1]['steps'],540)
        self.assertTrue(np.all(np.isfinite(a.probability(xs))))

    def test_learned_actions_are_legal(self):
        x,y,_,_=lab.corpus()
        model,_=lab.train(x,y)
        row,trace=lab.episode(self.late,'learned',model,record=True)
        self.assertTrue(row['arrived'])
        self.assertEqual(row['paint_moves'],sum(n for a,n in row['action_counts'].items() if a.startswith('MOVE_')))
        self.assertEqual(lab.replay(trace),row['decisions'])
        self.assertLessEqual(row['paint_moves'],64)


if __name__=='__main__':
    unittest.main()

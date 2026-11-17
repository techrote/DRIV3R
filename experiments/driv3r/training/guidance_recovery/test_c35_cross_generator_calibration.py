"""C35 deterministic, host-legality, no-leakage and replay regression tests."""
from __future__ import annotations
import copy, json, os, sys, unittest
from pathlib import Path
import numpy as np
import torch
import c35_cross_generator_calibration as c
from c35_gallery import replay

class TestC35(unittest.TestCase):
    def test_graph_distribution_and_determinism(self):
        for fam in c.TRAIN_FAMILIES+c.TEST_FAMILIES:
            a=c.graph(351701,fam);b=c.graph(351701,fam)
            self.assertEqual(a,b)
            self.assertEqual(len(a),111 if fam=='wilson_braid' else 97)
            self.assertEqual(len(c.shortest(a,(1,1),(13,13)))>0,True)

    def test_distinct_heldout_generators(self):
        arrays=[c.graph(451701,fam) for fam in c.TRAIN_FAMILIES+c.TEST_FAMILIES]
        self.assertEqual(len({c.sha(sorted(q)) for q in arrays}),4)
        for family in c.TEST_FAMILIES:
            self.assertNotIn(family,c.TRAIN_FAMILIES)

    def test_world_prefix_host_executed_and_goal_unseen(self):
        for family in c.TRAIN_FAMILIES+c.TEST_FAMILIES:
            for case in c.CASES:
                w,info=c.make_world(351701,family,case,1)
                self.assertEqual(len(w.history),24)
                self.assertEqual(w.steps,24)
                self.assertEqual(w.invalid_attempts,0)
                self.assertNotIn(w.goal,w.history)
                self.assertEqual(info['initial_retreat'],case=='detour')
                self.assertFalse(set(w.inked)&w.protected)

    def test_oracle_not_in_features(self):
        w,_=c.make_world(351701,'wilson_braid','forward',1)
        x=c.features(w,w.candidates(.35))
        self.assertEqual(x.shape,(15,))
        self.assertTrue(np.isfinite(x).all())
        v=copy.deepcopy(w)
        remote=(14,14)
        self.assertGreater(abs(remote[0]-w.pos[0])+abs(remote[1]-w.pos[1]),2)
        v.protected.symmetric_difference_update({remote})
        np.testing.assert_array_equal(x,c.features(v,v.candidates(.35)))

    def test_counterfactual_targets_use_legal_execution(self):
        X,Y,receipts=c.samples(371001,'wilson',1)
        self.assertEqual(len(X),len(Y))
        self.assertTrue(np.isfinite(X).all())
        self.assertTrue(np.isfinite(Y).all())
        self.assertTrue(all(-1.5<=r['delta']<=1.5 for r in receipts))
        self.assertTrue(all('legal_menu_sha' in r and 'state_sha' in r for r in receipts))

    def test_head_dimension_and_output(self):
        m=c.CalHead().eval()
        with torch.no_grad():z=m(torch.zeros((3,15)))
        self.assertEqual(tuple(z.shape),(3,))

    def test_gallery_replays_exact_host_actions_and_ink(self):
        path=Path(__file__).resolve().parent.parent/'C35-selected-traces.json'
        if not path.exists():self.skipTest('selected research traces are retained in evidence archive')
        rr=json.loads(path.read_text())
        self.assertGreaterEqual(len(rr),16)
        for r in rr:self.assertEqual(replay(r)[0].invalid_attempts,0)

    def test_snapshot_checkpoint_repeated_identical(self):
        root=Path(__file__).resolve().parent.parent
        a=root/'out_a';b=root/'out_b'
        if not a.exists() or not b.exists():self.skipTest('optimizer replicas retained in evidence archive')
        for i in range(3):
            self.assertEqual((a/f'C35-CAL-{i}.safetensors').read_bytes(),
                             (b/f'C35-CAL-{i}.safetensors').read_bytes())

if __name__=='__main__':unittest.main()

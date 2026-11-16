"""Controlled adversarial tests for SIDE-EXIT-COST-001."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import side_exit_cost_lab as lab
from .world import specs

class SideExitCostTests(unittest.TestCase):
    def test_partitioned_geometry(self):
        self.assertFalse(set(lab.TRAIN_DEPTHS)&set(lab.DEV_DEPTHS))
        self.assertFalse(set(lab.TRAIN_BACK)&set(lab.DEV_BACK))
        self.assertFalse(set(lab.TRAIN_DIRECTIONS)&set(lab.DEV_DIRECTIONS))
        sp=specs('development',1)[0]
        for direction in lab.DEV_DIRECTIONS:
            for depth in lab.DEV_DEPTHS:
                for back in lab.DEV_BACK:
                    for side in (-1,1):
                        w,goal=lab.make_world(sp,depth,direction,back,side)
                        lab.force_error(w,depth,direction,3)
                        self.assertEqual(lab.nearest_exit(w),back)
                        self.assertEqual(set(w.menu(3)),set((*lab.RETURNS,'STOP')))
                        self.assertNotIn(goal,w.protected)
                        self.assertNotIn(goal,w.materials)

    def test_cap_only_is_aliased(self):
        sp=specs('development',1)[0]
        a,_=lab.make_forced(sp,8,'W',2,1,5)
        b,_=lab.make_forced(sp,8,'W',4,1,5)
        self.assertTrue(np.array_equal(lab.raw_features(a,5,'budget_only'),lab.raw_features(b,5,'budget_only')))
        self.assertFalse(np.array_equal(lab.raw_features(a,5,'linear_geometry'),lab.raw_features(b,5,'linear_geometry')))
        self.assertEqual(lab.INPUT,5)

    def test_sealed_rejected(self):
        a=specs('train',1)[0]
        with self.assertRaises(ValueError):lab.make_world(a,6,'E',1,1)
        with self.assertRaises(ValueError):lab.make_world(a,5,'W',1,1)
        with self.assertRaises(ValueError):lab.make_world(replace(a,split='heldout'),5,'E',1,1)

    def test_no_hidden_feasibility_feature(self):
        sp=specs('development',1)[0]
        w,goal=lab.make_forced(sp,8,'W',4,1,3)
        f=lab.raw_features(w,3,'nonlinear_geometry')
        self.assertEqual(f[0],3/12)
        self.assertEqual(f[2],4/12)
        self.assertFalse(any(x in (1.,-1.) for x in f[:3]))
        self.assertEqual(lab.counterfactual(w,goal,3,128,123)[0] in (0,1),True)

    def test_fork_probability_and_host_tamper(self):
        sp=specs('development',1)[0];m=lab.CostRanker(600,'nonlinear_geometry')
        ep=lab.rollout(sp,m,6,'W',2,1,3,128)
        lab.replay_episode(ep,m,sp)
        ep['rows'][6]['probabilities']['BACKTRACK']+=.08
        with self.assertRaises(ValueError):lab.replay_episode(ep,m,sp)
        ep=lab.rollout(sp,m,6,'W',2,1,3,128)
        ep['rows'][6]['event']['legal']=['STOP']
        with self.assertRaises(ValueError):lab.replay_episode(ep,m,sp)

    def test_training_and_original_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'case';r=lab.run(out,600,2,quick=True)
            self.assertEqual(r['episodes'],6)
            self.assertEqual(set(r['checkpoints']),set(lab.ARMS))
            self.assertEqual(lab.verify(out)['episodes_replayed'],6)
            e=next((out/'traces').glob('*.json'))
            o=json.loads(e.read_text());o['goal_reached']=not o['goal_reached'];e.write_text(json.dumps(o))
            with self.assertRaises(ValueError):lab.verify(out)

if __name__=='__main__':unittest.main()

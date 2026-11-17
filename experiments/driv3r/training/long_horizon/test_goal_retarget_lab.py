"""Guardrails for GOAL-RETARGET-001: partition, legal trajectories, independent replay."""
from pathlib import Path
import json
import tempfile
import unittest
from . import goal_retarget_lab as lab
from .world import World, specs

class GoalRetargetTests(unittest.TestCase):
    def test_only_new_development_seeds(self):
        seen=lab.cases()
        self.assertEqual(len(seen),8)
        self.assertEqual({s.seed for s in seen},{3001,3101,3201,3301})
        self.assertTrue(all(s.split=="development" for s in seen))
        self.assertTrue(all(s.family not in ("archipelago","staggered_vault") for s in seen))

    def test_goal_schedule_fixed_before_rollout(self):
        sp=lab.cases()[0]
        g=lab.targets_for(sp,617,"retarget")
        self.assertEqual(g,lab.targets_for(sp,617,"retarget"))
        self.assertEqual(len(g),4)
        w=World.create(sp)
        first=w.menu(12)[0]
        w.apply(first)
        self.assertEqual(g,lab.targets_for(sp,617,"retarget"))
        self.assertEqual([lab.active_phase(t,"retarget") for t in (0,31,32,71,72,111,112)],[0,0,1,1,2,2,3])

    def test_masked_arm_same_weights(self):
        a,_=lab.prior.train(617,1)
        self.assertEqual(sum(x.size for x in a["goal_only"].p.values()),
                         sum(x.size for x in a["goal_recovery"].p.values()))
        w=World.create(lab.cases()[0]);target=lab.targets_for(w.spec,617,"retarget")[0]
        menu,x=lab.prior.matrix(w,target,128,True)
        _,y=lab.prior.matrix(w,target,128,False)
        self.assertTrue((x[:,-8:]==0).all())
        self.assertTrue((y[:,-8:]!=0).any())
        self.assertEqual(menu,w.menu(12))

    def test_short_optimizer_and_replay_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            o=Path(tmp)/"artifacts"
            lab.run(o,seed=617,updates=2,quick=True)
            self.assertEqual(lab.verify(o)["episodes"],6)
            p=next((o/"traces").glob("*.json"))
            ep=json.loads(p.read_text())
            sp={s.ident:s for s in lab.cases()}[ep["spec"]]
            mods={a:lab.prior.Ranker.load(o/(a+".npz")) for a in ("goal_only","goal_recovery")}
            ep["rows"][0]["probabilities"][ep["rows"][0]["requested"]]=0
            with self.assertRaises(ValueError):lab.verify_episode(ep,sp,mods)
            p.write_text(json.dumps(ep))
            with self.assertRaises(ValueError):lab.verify(o)

    def test_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"out";p.mkdir()
            with self.assertRaises(FileExistsError):lab.run(p,seed=617,updates=1,quick=True)

if __name__=="__main__":unittest.main()

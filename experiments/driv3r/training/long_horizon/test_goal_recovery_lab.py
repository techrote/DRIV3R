"""GOAL-RECOVERY-001: guardrails on training, explicit recovery and replay."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import goal_recovery_lab as lab
from .world import World, specs

class GoalRecoveryTests(unittest.TestCase):
    def test_mask_preserves_matched_policy_input(self):
        w=World.create(specs('development',1)[0]);target=(60,48)
        a,x=lab.matrix(w,target,128,True);b,y=lab.matrix(w,target,128,False)
        self.assertEqual(a,b)
        np.testing.assert_array_equal(x[:,:lab.goal.FEATURE_DIM],y[:,:lab.goal.FEATURE_DIM])
        self.assertTrue(np.all(x[:,-8:]==0))
        self.assertTrue(np.any(y[:,-8:]!=0))

    def test_corridor_label_legal(self):
        sp=replace(specs('train',1)[0],challenge='trap_corridor')
        w=World.create(sp)
        for _ in range(6):w.apply('MOVE_E',12)
        lab_label=lab.label(w,(60,48),128,__import__('random').Random(7))
        self.assertIn(lab_label,w.menu(12))
        self.assertEqual(lab_label,'RELOCATE_START')

    def test_no_heldout_corpus(self):
        _,_,_,m=lab.corpus(615,36)
        self.assertEqual(m['split'],'train')
        self.assertGreater(m['examples'],100)

    def test_quick_training_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';lab.run(out,seed=615,updates=2,quick=True)
            verified=lab.verify(out)
            self.assertEqual(verified['episodes'],4)
            p=next((out/'traces').glob('*.json'))
            record=json.loads(p.read_text());record['rows'][0]['probabilities'][record['rows'][0]['requested']]=0.0
            models={a:lab.Ranker.load(out/f'{a}.npz') for a in lab.ARMS}
            with self.assertRaises(ValueError):lab.verify_episode(record, specs('development',1)[0],models)
            (out/'report.json').write_text((out/'report.json').read_text()+' ')
            with self.assertRaises(ValueError):lab.verify(out)

if __name__=='__main__':unittest.main()

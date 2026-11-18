"""Source-faithful GOAL-COND-001 invariants; no native or sealed-heldout tests."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from . import goal_conditioning_lab as lab
from .world import World, specs

class GoalConditionTests(unittest.TestCase):
    def test_training_partition(self):
        _,_,_,m=lab.training_corpus(613,36)
        self.assertEqual(m['world_partition'],'train')
        self.assertGreater(m['examples'],100)

    def test_masked_goal_invariance(self):
        w=World.create(specs('development',1)[0]); m=lab.GoalRanker(613)
        a,x=lab.matrix(w,(60,48),128,True)
        b,y=lab.matrix(w,(36,48),128,True)
        self.assertEqual(a,b);np.testing.assert_array_equal(x,y)
        _,u=lab.matrix(w,(60,48),128,False)
        _,v=lab.matrix(w,(36,48),128,False)
        self.assertFalse(np.array_equal(u,v))

    def test_no_illegal_terrain_edits(self):
        sp=specs('development',1)[0]
        models={arm:lab.GoalRanker(613) for arm in ('blind','conditioned')}
        for arm in models:
            ep=lab.episode(sp,models,arm,613,128)
            self.assertEqual(ep['metrics']['protected_cell_edits'],0)
            self.assertEqual(ep['metrics']['illegal_host_actions'],0)
            self.assertEqual(lab.verify_episode(ep,sp,models),len(ep['rows']))

    def test_quick_artifact_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'proof';lab.run(out,seed=613,updates=2,quick=True)
            self.assertEqual(lab.verify(out)['episodes'],2)
            files=list((out/'traces').glob('*.json'))
            import json
            ep=json.loads(files[0].read_text())
            models={a:lab.GoalRanker.load(out/f'{a}.npz') for a in ('blind','conditioned')}
            sp=specs('development',1)[0]
            ep['rows'][0]['probabilities'][next(iter(ep['rows'][0]['probabilities']))]=.99
            with self.assertRaises(ValueError):lab.verify_episode(ep,sp,models)
            raw=(out/'report.json').read_text();(out/'report.json').write_text(raw+' ')
            with self.assertRaises(ValueError):lab.verify(out)

if __name__=='__main__':unittest.main()

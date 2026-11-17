"""Bounded source-faithful, partition, optimizer and tamper checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import retarget_onpolicy_lab as lab
from . import goal_recovery_lab as prior
from .world import World


class RetargetOnPolicyTests(unittest.TestCase):
    def test_sealed_family_never_opened(self):
        self.assertTrue(all(s.split=='development' for s in lab.cases()))
        self.assertEqual({s.seed for s in lab.cases()}, {3001,3101,3201,3301})
        self.assertTrue(all(s.family not in ('archipelago','staggered_vault') for s in lab.cases()))

    def test_distinct_goals_and_legal_student_examples(self):
        model=prior.Ranker(619)
        x,m,y,meta=lab.student_corpus(model,619,128,'retarget',64)
        self.assertEqual(x.shape[0],len(m))
        self.assertTrue(np.all(m[np.arange(len(y)), y]))
        self.assertEqual(meta['split'],'train')
        self.assertGreater(meta['visited_states'],64)
        self.assertEqual(lab.fixture.active_phase(112,'retarget'),3)
        sp=lab.cases()[0]
        targets=lab.fixture.targets_for(sp,619,'retarget')
        self.assertEqual(len(targets),4)
        self.assertEqual(lab.fixture.active_phase(72,'retarget'),2)

    def test_identical_start_and_equal_optimizer_budget(self):
        models, hist=lab.train(619,1,1,64)
        self.assertEqual(models['static_adapt'].updates,models['retarget_adapt'].updates)
        self.assertEqual(models['frozen'].updates+3,models['static_adapt'].updates)
        for arm in ('static_adapt','retarget_adapt'):
            self.assertEqual(len(hist['on_policy_stages']),3)
            self.assertEqual(sum(x.size for x in models[arm].p.values()),961)

    def test_checkpoint_replay_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'evidence'
            lab.run(p,619,1,1,True,64)
            self.assertEqual(lab.verify(p)['episodes'],3)
            f=next((p/'traces').glob('*.json'))
            ep=json.loads(f.read_text());ep['rows'][0]['probabilities'][ep['rows'][0]['requested']]=0
            f.write_text(json.dumps(ep))
            with self.assertRaises(ValueError):lab.verify(p)

    def test_immutable_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.mkdir()
            with self.assertRaises(FileExistsError):lab.run(p,619,1,1,True,64)

if __name__ == '__main__':unittest.main()

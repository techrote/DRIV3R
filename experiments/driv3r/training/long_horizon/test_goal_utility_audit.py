"""Fast smoke tests for GOAL-UTILITY-001; no remote or paid compute."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
from . import goal_utility_audit as lab
from .world import Spec

ARTIFACT=Path('/mnt/data/driv3r_outcome_fork_20261117_evidence.zip')

class TestGoalUtility(unittest.TestCase):
    def test_only_new_development_seeds(self):
        specs=lab.development_specs()
        self.assertEqual(len(specs),8)
        self.assertEqual([s.seed%100 for s in specs],[1,2]*4)
        self.assertTrue(all(s.split=='development' for s in specs))
        self.assertTrue(all(s.family in ('pickets','ridges','islands','zigzag') for s in specs))

    def test_fork_legal_and_immutable_pair(self):
        sp=lab.development_specs()[0]
        q=lab.previous.fixture(sp,11);world=q['world']
        initial=world.state_sha()
        self.assertEqual(world.menu(12),['BACKTRACK','MOVE_S','RELOCATE_START','STOP'])
        self.assertEqual(q['spent'],11)
        self.assertEqual(len(q['prefix']),28)
        self.assertEqual(world.state_sha(),initial)
        self.assertFalse(world.occupied(lab.GOALS['north_exit']))
        self.assertFalse(world.occupied(lab.GOALS['south_spur']))

    def test_archive_restoration_and_paired_distributions(self):
        if not ARTIFACT.exists():self.skipTest('checkpoint archive not mounted')
        models,ident=lab.previous.restored(ARTIFACT,811)
        self.assertEqual(len(ident),4)
        for goal in lab.GOALS:
            move=lab.rollout(lab.development_specs()[0],811,'frozen',models['frozen'],goal,'MOVE_S')
            back=lab.rollout(lab.development_specs()[0],811,'frozen',models['frozen'],goal,'BACKTRACK')
            self.assertEqual(move['initial_sha256'],back['initial_sha256'])
            self.assertEqual(move['first_probs'],back['first_probs'])
            self.assertEqual(move['rows'][0]['event']['chosen'],'MOVE_S')
            self.assertEqual(back['rows'][0]['event']['chosen'],'BACKTRACK')

    def test_sealed_heldout_is_rejected(self):
        if not ARTIFACT.exists():self.skipTest('checkpoint archive not mounted')
        models,_=lab.previous.restored(ARTIFACT,811)
        with self.assertRaises(ValueError):
            lab.rollout(Spec('heldout','archipelago',9000,'wall'),811,'frozen',models['frozen'],'north_exit','BACKTRACK')

if __name__=='__main__': unittest.main()

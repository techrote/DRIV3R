"""Focused falsification and replay checks; no training or HELDOUT access."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from . import topology_goal_churn_lab as lab
from .world import Spec, specs
from .goal_conditioning_lab import GoalRanker

CHECKPOINTS=Path(__file__).resolve().parents[4]/'checkpoints'  # CLI can override for full archive verification

class TopologyTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Optional local archive; normal CI focuses on fixture invariants without checkpoints.
        cls.sp=specs('development',1)[0]
    def test_heldout_and_train_rejected(self):
        for split in ('train','heldout'):
            with self.assertRaises(ValueError):lab.overlay_world(specs(split,1)[0],'opposing_gates')
        with self.assertRaises(ValueError):lab.overlay_world(self.sp,'unlisted')
        with self.assertRaises(ValueError):lab.overlay_world(Spec(self.sp.split,self.sp.family,self.sp.seed,self.sp.mode,'trap_corridor'),'opposing_gates')
    def test_four_distinct_fixed_topologies(self):
        witnesses=[]
        for layout in lab.LAYOUTS:
            a=lab.overlay_world(self.sp,layout);b=lab.overlay_world(self.sp,layout)
            self.assertEqual(a.base_sha(),b.base_sha())
            self.assertEqual(a.menu(12),b.menu(12))
            self.assertEqual(a.logical_moves,0)
            self.assertTrue(all(not a.occupied(g) for g in lab.GOALS))
            self.assertTrue(all(p not in a.protected for p in a.ink))
            self.assertTrue(all(p not in a.materials for p in a.ink))
            witnesses.append(a.base_sha())
        self.assertEqual(len(set(witnesses)),4)
    def test_split_partition_and_fixed_clock(self):
        self.assertEqual(lab.SCHEDULE,(12,30))
        self.assertLess(lab.SCHEDULE[-1],lab.CAP)
        self.assertTrue(all(s.split=='development' for s in specs('development',1)))
    def test_original_replay_tampering(self):
        if not (CHECKPOINTS/'711-conditioned.npz').exists():self.skipTest('requires local binary checkpoint archive')
        model=GoalRanker.load(CHECKPOINTS/'711-conditioned.npz')
        row,tr=lab.episode(self.sp,'horizontal_switchback',0,711,'neural',model)
        self.assertEqual(lab.verify_trace(tr,model)[0],row['decisions'])
        self.assertEqual(row['illegal_material_edits'],0)
        bad=copy.deepcopy(tr)
        self.assertGreater(len(bad['entries']),0)
        bad['entries'][0]['menu'].append('MOVE_NONEXISTENT')
        with self.assertRaises(AssertionError):lab.verify_trace(bad,model)
        bad=copy.deepcopy(tr)
        bad['entries'][0]['receipt']['cursor_after']=[0,0]
        with self.assertRaises(AssertionError):lab.verify_trace(bad,model)
        bad=copy.deepcopy(tr)
        bad['entries'][0]['probabilities']={'STOP':1}
        with self.assertRaises(AssertionError):lab.verify_trace(bad,model)
        bad=copy.deepcopy(tr)
        self.assertTrue(bad['switches'])
        bad['switches'][0]['next_goal']=[0,0]
        with self.assertRaises(AssertionError):lab.verify_trace(bad,model)

if __name__=='__main__':unittest.main()

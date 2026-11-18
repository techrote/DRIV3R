"""Frozen-mask campaign regression and evidence integrity checks."""
import copy
import json
import unittest
from pathlib import Path
from . import unseen_mask_stress_lab as lab
from . import topology_adaptive_handoff_lab as baseline
from .goal_conditioning_lab import GoalRanker
from .world import specs

CHECKPOINTS=Path(__file__).resolve().parents[4]/'checkpoints'

class NovelMaskStudyTests(unittest.TestCase):
    def test_masks_are_distinct_from_prior_and_sealed_holdout(self):
        spec=specs('development',1)[0]
        old={baseline.overlay_world(spec,m).base_sha() for m in baseline.LAYOUTS}
        new=[lab.novel_world(spec,m).base_sha() for m in lab.MASKS]
        self.assertEqual(len(set(new)),len(new))
        self.assertFalse(old.intersection(new))
        for split in ('train','heldout'):
            with self.assertRaises(ValueError):lab.novel_world(specs(split,1)[0],lab.MASKS[0])
        self.assertEqual(baseline.SCHEDULE,(12,30))

    def test_all_starts_legally_accessible_without_mask_edits(self):
        for spec in specs('development',2):
            for mask in lab.MASKS:
                w=lab.novel_world(spec,mask)
                self.assertFalse(w.occupied(w.cursor))
                self.assertTrue(all(not w.occupied(g) for g in lab.GOALS))
                self.assertEqual(len(w.ink),0)
                self.assertEqual(w.logical_moves,0)
                self.assertTrue(all(lab.prefix_plan(w,g,64) is not None for g in lab.GOALS))

    def test_exact_replay_and_tamper_detection(self):
        cp=CHECKPOINTS/'711-conditioned.npz'
        if not cp.exists():self.skipTest('local immutable checkpoint absent')
        model=GoalRanker.load(cp)
        sp=specs('development',1)[0]
        with lab.use_novel_masks():
            for arm in lab.ARMS:
                row,tr=baseline.episode(sp,'broken_diamond',0,711,arm,model)
                c,v=baseline.verify_trace(tr,model)
                self.assertEqual(c,row['decisions'])
                self.assertGreaterEqual(v,0)
                self.assertEqual(row['illegal_material_edits'],0)
            bad=copy.deepcopy(tr)
            bad['entries'][0]['menu'].append('FORGED')
            with self.assertRaises(AssertionError):baseline.verify_trace(bad,model)

if __name__=='__main__':unittest.main()

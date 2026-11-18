"""Focused material transition, immutability and evidence-replay tests."""
import copy, json, unittest
from pathlib import Path
from .material_epoch_lab import apply_event, episode, verify_trace, PATCHES, LAYOUTS, SEEDS, ARMS
from .topology_adaptive_handoff_lab import overlay_world
from .goal_conditioning_lab import GoalRanker
from .world import Spec, MATERIAL_Z

class MaterialEpochTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).parents[3]  # unused source path; fixture is synthetic
        cls.sp=Spec('development','pickets',3000,'wall')
        cls.model=GoalRanker.load(Path(__file__).parents[4]/'checkpoints'/'711-conditioned.npz') if (Path(__file__).parents[4]/'checkpoints'/'711-conditioned.npz').exists() else None

    def test_material_event_receipts_are_stateful(self):
        for layout in LAYOUTS:
            w=overlay_world(self.sp,layout)
            first=apply_event(w,layout,20);second=apply_event(w,layout,36)
            self.assertNotEqual(first['pre_sha256'],first['post_sha256'])
            self.assertEqual(first['post_sha256'],second['pre_sha256'])
            self.assertEqual(len(first['operations']),2)
            self.assertFalse(w.ink & (set(w.materials)|w.protected))

    def test_closure_never_overwrites_painted_cell(self):
        w=overlay_world(self.sp,'opposing_gates');target=PATCHES['opposing_gates'][1]
        w.ink.add(target)
        event=apply_event(w,'opposing_gates',20)
        self.assertFalse(event['operations'][1]['applied'])
        self.assertEqual(event['operations'][1]['reason'],'blocked_by_ink_or_cursor')
        self.assertNotIn(target,w.materials)

    def test_tampering_detected_by_receipt_replay(self):
        if self.model is None:self.skipTest('local checkpoint not beside laboratory source')
        _,trace=episode(self.sp,'opposing_gates',0,711,'fixed24_epoch',self.model)
        self.assertGreater(len(trace['entries']),0)
        self.assertGreater(len(trace['material_events']),0)
        self.assertGreater(verify_trace(trace,self.model)[0],0)
        forged=copy.deepcopy(trace);forged['entries'][0]['action']='MOVE_N' if trace['entries'][0]['action']!='MOVE_N' else 'MOVE_E'
        with self.assertRaises(AssertionError):verify_trace(forged,self.model)
        forged=copy.deepcopy(trace);forged['material_events'][0]['operations'][0]['after']=999
        with self.assertRaises(AssertionError):verify_trace(forged,self.model)
        forged=copy.deepcopy(trace);forged['entries'][0]['probabilities']={'forged':1.0}
        with self.assertRaises(AssertionError):verify_trace(forged,self.model)

if __name__=='__main__':unittest.main()

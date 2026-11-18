"""Deterministic synthetic gate-oscillation qualification, including receipt tampering."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from . import gate_oscillation_lab as lab
from .world import Spec
from .goal_conditioning_lab import GoalRanker


class GateOscillationTests(unittest.TestCase):
    def setUp(self):
        self.sp = Spec('development', 'pickets', 3000, 'wall')
        self.layout = 'opposing_gates'
        self.checkpoints = Path(os.environ.get('DRIV3R_CHECKPOINTS', 'checkpoints'))

    def test_controls_and_event_schedule(self):
        for mode in lab.MODES:
            row, trace = lab.episode(self.sp, self.layout, 0, mode, 'lazy')
            self.assertTrue(row['arrived'], mode)
            self.assertEqual(row['illegal_edits'], 0)
            self.assertLessEqual(row['logical_moves'], 64)
            self.assertEqual(trace['events'][0]['kind'], 'close')
            if mode == 'rapid_bounce':
                self.assertEqual([e['kind'] for e in trace['events']], ['close','open','close','open'])
            else:
                self.assertEqual([e['kind'] for e in trace['events']], ['close','open'])

    def test_late_route_disruption_and_omitted_footprint_fail(self):
        for arm, success in [('blind',False),('partial_two',False),
                             ('intersect',True),('lazy',True),('always',True)]:
            row, tr = lab.episode(self.sp, self.layout, 0, 'late_reopen', arm)
            self.assertEqual(row['arrived'], success, arm)
            self.assertEqual(row['illegal_edits'], 0)

    def test_early_reopen_avoids_any_lazy_replanning(self):
        row, tr = lab.episode(self.sp, self.layout, 0, 'early_reopen', 'lazy')
        self.assertTrue(row['arrived'])
        self.assertEqual(row['planner_calls'], 1)
        self.assertEqual(lab.verify_trace(tr, {}), (row['decisions'], 0))

    def test_receipt_menu_and_event_tampering(self):
        _, original = lab.episode(self.sp, self.layout, 0, 'late_reopen', 'intersect')
        for key, value in [('action','STOP'),('menu',['STOP']),('pre_sha256','not-a-state')]:
            tr = copy.deepcopy(original)
            tr['entries'][0][key] = value
            with self.assertRaises(AssertionError):
                lab.verify_trace(tr, {})
        tr = copy.deepcopy(original)
        tr['events'][0]['cell'] = [0,0]
        with self.assertRaises(AssertionError):
            lab.verify_trace(tr, {})

    def test_heldout_sealed(self):
        held = Spec('heldout','archipelago',9000,'wall')
        with self.assertRaises(ValueError):
            lab.episode(held,self.layout,0,'early_reopen','lazy')

    def test_frozen_neural_probability_tamper(self):
        path = self.checkpoints/'711-conditioned.npz'
        if not path.exists():
            self.skipTest('Frozen checkpoint unavailable')
        model = GoalRanker.load(path)
        _, original = lab.episode(self.sp,self.layout,0,'late_reopen','neural',711,model)
        self.assertGreater(lab.verify_trace(original,{711:model})[1],0)
        tr = copy.deepcopy(original)
        ent = next(e for e in tr['entries'] if e['probabilities'] is not None)
        ent['probabilities']={'STOP':1.0}
        with self.assertRaises(AssertionError):
            lab.verify_trace(tr,{711:model})

    def test_complete_reexecution(self):
        if not all((self.checkpoints/f'{s}-conditioned.npz').exists() for s in lab.MODEL_SEEDS):
            self.skipTest('External frozen checkpoints unavailable')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            ev=lab.campaign(out,self.checkpoints)
            self.assertEqual(ev['episodes'],224)
            self.assertEqual(ev['summary'],lab.verify(out,self.checkpoints)['summary'])


if __name__=='__main__':
    unittest.main()

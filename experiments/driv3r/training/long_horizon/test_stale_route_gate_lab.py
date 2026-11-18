"""Source-faithful synthetic stale-route falsification tests; no Git/network."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from . import stale_route_gate_lab as lab
from .world import Spec
from .goal_conditioning_lab import GoalRanker


class StaleRouteGateTests(unittest.TestCase):
    def setUp(self):
        self.sp = Spec('development', 'pickets', 3000, 'wall')
        self.layout = 'opposing_gates'
        self.models_dir = Path(os.environ.get('DRIV3R_CHECKPOINTS', 'checkpoints'))

    def test_precommitted_future_vs_unrelated_gate(self):
        plan, points, sites = lab.scenario(self.sp, self.layout, lab.GOALS[0])
        self.assertGreater(len(plan), lab.GATE_INDEX + 2)
        self.assertEqual(points[lab.GATE_INDEX], sites['on_route'])
        self.assertNotIn(sites['off_route'], points)
        for hazard in lab.HAZARDS:
            row, trace = lab.one_episode(self.sp, self.layout, 0, hazard, 'stale_cache', None, None)
            self.assertEqual(trace['material_event']['remaining_route_intersection'], hazard == 'on_route')
            self.assertEqual(row['protected_material_edits'], 0)
            self.assertLessEqual(row['logical_moves'], 64)
            self.assertEqual(row['arrived'], hazard == 'off_route')

    def test_targeted_and_lazy_replanning_rescue_actual_blockage(self):
        for hazard in lab.HAZARDS:
            for arm in ('epoch_all', 'route_overlap', 'lazy_illegal'):
                row, trace = lab.one_episode(self.sp, self.layout, 0, hazard, arm, None, None)
                self.assertTrue(row['arrived'], (hazard, arm))
                self.assertEqual(row['planner_calls'], 2 if hazard == 'on_route' or arm == 'epoch_all' else 1)
                self.assertEqual(row['protected_material_edits'], 0)
                if hazard == 'on_route' and arm == 'lazy_illegal':
                    self.assertEqual(trace['row']['invalidations'][0]['tick'] - lab.EVENT_TICK, 3)

    def test_tampered_action_menu_and_event_are_rejected(self):
        _, trace = lab.one_episode(self.sp, self.layout, 0, 'on_route', 'route_overlap', None, None)
        models = {}
        self.assertGreater(lab.verify_trace(trace, models)[0], 0)
        forged = copy.deepcopy(trace)
        forged['entries'][0]['action'] = 'STOP'
        with self.assertRaises(AssertionError): lab.verify_trace(forged, models)
        forged = copy.deepcopy(trace)
        forged['entries'][0]['menu'] = ['STOP']
        with self.assertRaises(AssertionError): lab.verify_trace(forged, models)
        forged = copy.deepcopy(trace)
        forged['material_event']['cell'] = [80, 80]
        with self.assertRaises(AssertionError): lab.verify_trace(forged, models)

    def test_neural_vector_tampering_rejected_when_checkpoint_present(self):
        checkpoint = self.models_dir / '711-conditioned.npz'
        if not checkpoint.exists():
            self.skipTest('external frozen checkpoint not present locally')
        model = GoalRanker.load(checkpoint)
        _, trace = lab.one_episode(self.sp, self.layout, 0, 'on_route', 'neural_recovery', 711, model)
        self.assertGreater(lab.verify_trace(trace, {711: model})[1], 0)
        forged = copy.deepcopy(trace)
        index = next(i for i, x in enumerate(forged['entries']) if x['probabilities'] is not None)
        forged['entries'][index]['probabilities'] = {'STOP': 1.0}
        with self.assertRaises(AssertionError): lab.verify_trace(forged, {711: model})

    def test_full_campaign_replay_when_checkpoints_present(self):
        if not all((self.models_dir / f'{s}-conditioned.npz').exists() for s in lab.MODEL_SEEDS):
            self.skipTest('two external frozen checkpoint files required')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ev = lab.campaign(out, self.models_dir)
            self.assertEqual(ev['episodes'], 96)
            self.assertEqual(lab.verify(out, self.models_dir)['summary'], ev['summary'])


if __name__ == '__main__':
    unittest.main()

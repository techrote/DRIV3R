"""Bounded source-only checks and optional full evidence verification.

Run: python -m unittest experiments.driv3r.training.long_horizon.test_goal_native_gate_lab
Full receipt audit: DRIV3R_GOAL_GATE_EVIDENCE=/absolute/path python -m unittest ...
"""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from .goal_native_gate_lab import ARMS, SEEDS, episode, verify, verify_trace
from .goal_observability_lab import source_route, strict_step
from .world import MOVE, World, specs


class GoalNativeGateTests(unittest.TestCase):
    def test_no_heldout_worlds(self):
        sw = specs('development', 3)
        self.assertEqual(len(sw), 12)
        self.assertTrue(all(s.split == 'development' for s in sw))
        self.assertTrue({'pickets','ridges','islands','zigzag'} == {s.family for s in sw})

    def test_reference_goals_are_from_legal_moves(self):
        sp = specs('development', 1)[0]
        waypoints, route = source_route(sp)
        self.assertGreaterEqual(len(waypoints), 4)
        self.assertEqual(route.logical_moves, route.steps)
        self.assertTrue(set(waypoints).issubset(route.ink))
        self.assertFalse(route.ink & (set(route.materials) | route.protected))

    def test_strict_step_censors_oversized_relocation(self):
        w = World.create(specs('development', 1)[0])
        for i in range(7):
            choices = [a for a in w.menu() if a in MOVE]
            self.assertTrue(choices)
            w.apply(choices[0])
        self.assertIn('RELOCATE_START', w.menu())
        w.logical_moves = 63
        pre = w.state_sha()
        self.assertIsNone(strict_step(w,'RELOCATE_START'))
        self.assertEqual(w.state_sha(), pre)
        self.assertEqual(w.logical_moves, 63)

    def test_receipt_replay_and_tamper_rejected(self):
        sp = specs('development', 1)[0]
        targets, _ = source_route(sp)
        import numpy as np
        from .goal_conditioning_lab import GoalRanker
        m = GoalRanker(7)
        models = {'blind':m,'conditioned':m}
        result, trace = episode(sp, targets, models, 'conditioned', 7, False)
        self.assertEqual(result['protected_material_edits'], 0)
        self.assertLessEqual(result['logical_moves'], 64)
        self.assertEqual(verify_trace(trace,models)[0], result['decisions'])
        if trace['events']:
            corrupt = copy.deepcopy(trace)
            corrupt['events'][0]['legal'] = ['STOP']
            with self.assertRaises(AssertionError): verify_trace(corrupt,models)

    def test_full_archive_if_supplied(self):
        path = os.getenv('DRIV3R_GOAL_GATE_EVIDENCE')
        if not path: self.skipTest('optional immutable local archive not supplied')
        r = verify(Path(path))
        self.assertEqual(r['episodes'], 192)
        self.assertGreater(r['decisions'], 9000)
        self.assertGreater(r['probabilities'], 5000)


if __name__ == '__main__': unittest.main()

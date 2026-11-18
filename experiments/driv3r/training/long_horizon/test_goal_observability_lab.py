"""Source-only tests: no paid compute, checkpoint download or native CyberSand."""
import copy
import unittest
from .world import World, MOVE, specs, replay
from .policy import Ranker, FEATURES
from .goal_observability_lab import strict_step, source_route, conflict_states, shortest_path_action, NATIVE_ANALOG_CAP

class GoalObservabilityTests(unittest.TestCase):
    def test_world_goal_not_in_neural_features(self):
        self.assertFalse(any('goal' in x or 'target' in x for x in FEATURES))
        self.assertEqual(Ranker(1051).parameter_count, 625)

    def test_identical_observation_conflicting_optimum(self):
        w,goals,(a,b)=conflict_states(specs('development',1)[0])[0]
        self.assertNotEqual(goals[0],goals[1])
        self.assertNotEqual(a,b)
        self.assertIn(a,w.menu());self.assertIn(b,w.menu())
        model=Ranker(1051)
        before=w.state_sha()
        sel,p=model.choose(w,128)
        self.assertEqual((sel,p),model.choose(w,128))
        self.assertEqual(before,w.state_sha())
        self.assertLessEqual(sum(sel==x for x in (a,b)),1)

    def test_all_development_routes_without_structural_heldout(self):
        for sp in specs('development',3):
            waypoints,w=source_route(sp)
            self.assertEqual(sp.split,'development')
            self.assertGreaterEqual(len(waypoints),4)
            self.assertFalse(w.ink & (w.protected|set(w.materials)))
            self.assertEqual(w.state_sha(), replay(sp,w.decisions).state_sha())

    def test_external_cap_blocks_overshoot_without_mutating_menu(self):
        w=World.create(specs('development',1)[0]);before=w.menu()
        w.logical_moves=NATIVE_ANALOG_CAP
        self.assertIsNone(strict_step(w,before[0]))
        self.assertEqual(w.menu(),before)
        self.assertEqual(len(w.decisions),0)

    def test_nonlegal_action_rejected(self):
        w=World.create(specs('development',1)[0]);
        with self.assertRaises(ValueError):strict_step(w,'RELOCATE_START')
        self.assertEqual(w.steps,0)

    def test_goal_oracle_uses_offered_menu(self):
        w=World.create(specs('development',1)[0])
        goal=(w.cursor[0]+1,w.cursor[1])
        a=shortest_path_action(w,goal)
        self.assertIn(a,w.menu())
        self.assertEqual(a,'MOVE_E')

if __name__=='__main__':unittest.main()

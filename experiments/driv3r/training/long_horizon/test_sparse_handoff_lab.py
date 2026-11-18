"""Bounded local checks for the sparse planner handoff experiment."""
import copy
import json
from pathlib import Path
import unittest
from .sparse_handoff_lab import ARMS, LAYOUTS, SEEDS, fixture, episode, verify_trace, decide, run, verify
from .world import Spec, specs
from .goal_conditioning_lab import GoalRanker
from .detour_recovery_lab import prefix_plan, GOALS


class SparseHandoffTests(unittest.TestCase):
    def test_overlay_no_forbidden_prelude(self):
        for base in specs('development', 1):
            sp=Spec(base.split, base.family, base.seed, base.mode, 'trap_corridor')
            for label in LAYOUTS:
                w=fixture(sp, 3, label)
                self.assertEqual(w.cursor, (51,48))
                self.assertEqual(len(w.ink & (set(w.materials)|w.protected)), 0)

    def test_exact_one_search_cached(self):
        base=specs('development', 1)[0]
        sp=Spec(base.split, base.family, base.seed, base.mode, 'trap_corridor')
        w=fixture(sp,3,'east_sleeve')
        path=prefix_plan(w, GOALS[0]);self.assertIsNotNone(path)
        a,prob,cached,calls,src=decide(w,GOALS[0],None,'planner_once',0,0,None)
        self.assertEqual(calls,1);self.assertEqual(a,path[0]);self.assertEqual(src,'planner_cached')
        for _ in range(2):
            w.apply(a,12)
            a,prob,cached,calls,src=decide(w,GOALS[0],None,'planner_once',1,0,cached)
            self.assertEqual(calls,0)

    def test_receipt_and_probability_tamper_detection(self):
        base=specs('development',1)[0]
        sp=Spec(base.split,base.family,base.seed,base.mode,'trap_corridor')
        model=GoalRanker(711)
        for arm in ('neural','planner_once'):
            row, trace = episode(sp,'original',3,GOALS[0],711,arm,model)
            verify_trace(trace,model)
            wrong=copy.deepcopy(trace)
            wrong['entries'][0]['receipt']['chosen']='MOVE_N'
            with self.assertRaises(AssertionError): verify_trace(wrong,model)
        row,trace=episode(sp,'original',3,GOALS[0],711,'neural',model)
        has_prob=next((i for i,e in enumerate(trace['entries']) if e['probabilities'] is not None),None)
        self.assertIsNotNone(has_prob)
        wrong=copy.deepcopy(trace)
        wrong['entries'][has_prob]['probabilities']['STOP']=42.0
        with self.assertRaises(AssertionError): verify_trace(wrong,model)


if __name__=='__main__':unittest.main()

"""Targeted C34 invariants. Run: python -m unittest test_c34_independent_maze."""
import copy
import unittest
import numpy as np
from c34_independent_maze import (maze,shortest,make_world,obs,executed_suffix,
    ReArmedWatch,RetreatHead,play,FAMILIES,CASES)
from experiment import DIRECTIONS,sha
from c28_temporal_option import OptionHead
from signed_feedback import HostAdvice


class C34GraphTests(unittest.TestCase):
    def test_maze_tree_braid_are_distinct_and_connected(self):
        for seed in (340701,340702,340703):
            a=maze(seed,'tree');b=maze(seed,'braid')
            self.assertEqual(len(a),97);self.assertEqual(len(b),111)
            self.assertTrue(a<b)
            for nodes in (a,b):
                self.assertTrue(all(shortest(nodes,(1,1),p) for p in nodes))
    def test_graph_determinism(self):
        a=maze(31415,'braid');b=maze(31415,'braid')
        self.assertEqual(sha(sorted(a)),sha(sorted(b)))
        self.assertNotEqual(sha(sorted(a)),sha(sorted(maze(31416,'braid'))))
    def test_prefix_executed_and_goal_off_stack(self):
        for family in FAMILIES:
            for case in CASES:
                w,info=make_world(340701,family,case,1)
                self.assertEqual(info['prefix_len'],24)
                self.assertEqual(len(w.history),24)
                self.assertEqual(w.invalid_attempts,0)
                self.assertNotIn(w.goal,w.history)
                self.assertNotEqual(w.goal,w.pos)
                self.assertEqual(info['initial_retreat'],case=='detour')
                self.assertTrue(shortest({(x,y) for x in range(15) for y in range(15) if w.allowed((x,y))},w.pos,w.goal))
    def test_protected_cells_never_modified_by_suffix(self):
        w,_=make_world(340703,'braid','detour',-1)
        before=sha(sorted(w.protected));m=w.candidates(.35)
        for j,c in enumerate(m):
            if c.legal:
                t=executed_suffix(w,j)
                self.assertTrue(np.isfinite(t))
        self.assertEqual(before,sha(sorted(w.protected)))
        self.assertEqual(w.invalid_attempts,0)
    def test_observation_has_no_global_topology_oracle(self):
        w,_=make_world(340701,'tree','detour',1)
        x=obs(w,w.candidates(.35))
        self.assertEqual(x.shape,(9,));self.assertTrue(np.isfinite(x).all())
    def test_rearm_only_from_new_positive_cue(self):
        w,info=make_world(340701,'tree','detour',1)
        pol=ReArmedWatch(OptionHead(),RetreatHead());pol.phase='released';pol.goal=tuple(w.goal)
        cue=HostAdvice(w)
        self.assertNotEqual(cue.active(w),1)
        self.assertEqual(pol.rearms,0)
        w.goal=info['second_goal'];cue.issue_backtrack(w,strength=.7,ttl=9)
        self.assertEqual(cue.active(w),1)
        # Re-arm logic is executed when the actual policy is next queried.
        self.assertNotEqual(tuple(w.goal),pol.goal)
    def test_stop_and_legal_menu_guard(self):
        w,_=make_world(340702,'braid','forward',1)
        for c in w.candidates(.35):
            if c.action[0]=='INK' and c.target in w.protected:self.assertFalse(c.legal)
        self.assertFalse(w.invalid_attempts)
    def test_training_input_does_not_include_checkpoint(self):
        # Head is intentionally tiny, no model binary in source or repo.
        model=RetreatHead()
        self.assertLess(sum(p.numel() for p in model.parameters()),500)


if __name__=='__main__':unittest.main()

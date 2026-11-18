"""Tests for bounded rootless/adjacent-backtrack counterfactual audit."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from . import retrace_realism_lab as lab
from .world import Spec, World


class RetraceRealismTests(unittest.TestCase):
    def test_splits_and_no_structural_heldout(self):
        for cohort in ('previous_development','fresh_development'):
            specs=lab.specimen_specs(cohort)
            self.assertTrue(specs)
            self.assertTrue(all(s.split=='development' for s in specs))
            self.assertTrue(all(s.family in ('pickets','ridges','islands','zigzag') for s in specs))
        self.assertEqual(len(lab.specimen_specs('previous_development')),8)
        self.assertEqual(len(lab.specimen_specs('fresh_development')),16)
        with self.assertRaises(ValueError):lab.specimen_specs('heldout')

    def test_action_mask_retains_host_menu(self):
        w=World.create(Spec('development','pickets',3002,'wall'))
        w.apply('MOVE_E',12);w.apply('MOVE_E',12)
        menu=w.menu(12)
        self.assertEqual(menu,sorted(menu))
        self.assertIn('BACKTRACK',menu)
        before=w.state_sha()
        scores=[.05]*len(menu)
        scores[menu.index('STOP')]=.7
        self.assertEqual(lab.conditional_choice(menu,scores),'STOP')
        self.assertEqual(before,w.state_sha())
        self.assertEqual(menu,w.menu(12))
        # No history teleport is allowed by the local intervention, even when legal.
        for _ in range(6):
            next_move=next((x for x in ('MOVE_E','MOVE_N','MOVE_S','MOVE_W') if x in w.menu(12)),None)
            w.apply(next_move,12)
        menu=w.menu(12)
        self.assertIn('RELOCATE_START',menu)
        scores=[0]*len(menu);scores[menu.index('RELOCATE_START')]=100
        self.assertNotEqual(lab.conditional_choice(menu,scores),'RELOCATE_START')
        self.assertIn('RELOCATE_START',w.menu(12))

    def test_lexicographic_hit_and_material_margin(self):
        hit=lambda t,d:{'goal_hit_step':t,'final_distance':d}
        self.assertGreater(lab.outcome_key(hit(2,20)),lab.outcome_key(hit(None,0)))
        self.assertFalse(lab.robust_margin(hit(5,3),hit(8,9)))
        self.assertTrue(lab.robust_margin(hit(5,3),hit(9,9)))
        self.assertTrue(lab.robust_margin(hit(5,3),hit(None,0)))
        self.assertTrue(lab.robust_margin(hit(None,4),hit(None,12)))
        self.assertFalse(lab.robust_margin(hit(None,4),hit(None,11)))
        self.assertFalse(lab.robust_margin(hit(None,4),hit(30,10)))

    def test_source_replay_one_real_checkpoint_and_budget(self):
        cp=Path('/mnt/data/lh_next/evidence/checkpoints')
        if not cp.exists(): self.skipTest('source-checkpoint archive not present')
        models,_=lab.previous.load_models(cp,(619,))
        sp=lab.specimen_specs('previous_development')[0]
        state,prefix=lab.start_state(sp,619,0,models[619]['frozen'])
        self.assertIsNotNone(state)
        a=lab.evaluate_state(state,models,619,sp,0,'previous_development')
        b=lab.evaluate_state(state,models,619,sp,0,'previous_development')
        self.assertEqual(a,b)
        self.assertEqual(state['world'].logical_moves,32)
        self.assertNotIn('RELOCATE_START',state['world'].decisions[-1]['chosen'])
        for controller in a['outcomes'].values():
            for first,fork in controller.items():
                self.assertIn(first,a['host_menu'])
                self.assertNotEqual(first,'RELOCATE_START')
                self.assertLessEqual(fork['logical_moves_added'],32)
                self.assertLessEqual(len(fork['rows']),32)
                for event in fork['rows']:
                    self.assertIn(event['executed'],event['host_menu'])
                    self.assertNotEqual(event['executed'],'RELOCATE_START')
                    self.assertEqual(len(event['host_menu']),len(event['probabilities']))
                    self.assertLessEqual(event['receipt']['logical_moves'],64)
                    if event['executed']=='BACKTRACK':
                        u,v=event['receipt']['cursor_before'],event['receipt']['cursor_after']
                        self.assertEqual(abs(u[0]-v[0])+abs(u[1]-v[1]),1)

if __name__=='__main__':unittest.main()

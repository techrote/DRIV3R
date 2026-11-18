"""Adversarial and receipt tests for P15, research-only synthetic-W0 host."""
import hashlib
import json
import unittest
from pathlib import Path
import torch
import p15_topology_transfer as p15
import p12_recovery_preference as base

HERE=Path(__file__).resolve().parent

class TopologyTests(unittest.TestCase):
    def test_masks_predeclared_stable_and_distinct(self):
        shapes=[p15.masks(s) for s in p15.LAYOUTS]
        self.assertEqual(len(shapes),3)
        self.assertEqual(len({frozenset(v) for v in shapes}),3)
        self.assertTrue(all(len(s)>0 and (7,48) not in s for s in shapes))
        self.assertEqual(shapes[0],p15.masks('alternating_teeth'))

    def test_reject_unknown_mask(self):
        with self.assertRaises(ValueError):p15.masks('unknown')

    def test_exit_count_rejects_protected_and_out_of_bounds(self):
        mask={(2,1),(1,2),(0,1)}
        self.assertEqual(p15.exit_count((1,1),mask),1)
        self.assertEqual(p15.exit_count((0,0),set()),2)

    def test_stop_does_not_enter_protection(self):
        self.assertTrue(all((7,48) not in p15.masks(s) for s in p15.LAYOUTS))

    def test_choice_determinism_and_model_score_preservation(self):
        values=[.1,.2,.3,.4,.5,.6]
        actions=['LEFT','STRAIGHT']
        d={'LEFT':(8,48),'STRAIGHT':(7,47)}
        original=values.copy()
        for policy in ('position_veto','exit_aware_veto'):
            x=p15.sampled_move(values,actions,2711,'greebles',31,d,{(7,48)},set(),policy)
            y=p15.sampled_move(values,actions,2711,'greebles',31,d,{(7,48)},set(),policy)
            self.assertEqual(x,y)
            self.assertIn(x,actions)
        self.assertEqual(values,original)

    def test_empty_safe_actions_no_illegal_move(self):
        self.assertIsNone(p15.sampled_move([0.]*6,[],2711,'architecture',0,{},set(),set(),'exit_aware_veto'))

    def test_reject_tampered_ranker(self):
        with self.assertRaises(ValueError):p15.ranker_topology_transfer([],b'not a checkpoint')

    def test_p11_actual_source_replay_and_protection(self):
        p=HERE/'immutable-P11.zip'
        if not p.is_file():self.skipTest('Immutable P11 artifact not present in repository checkout')
        z,w0,head,_=base.open_pinned(p.read_bytes())
        source,_,_=base.extract_trace(z,'greebles',2819)
        originals=[]
        for policy in p15.POLICIES:
            rec,decisions,m=p15.rollout(w0,head,2819,'greebles','two_gate_cage',policy,source)
            self.assertTrue(all(d['host_legal'] for d in decisions))
            self.assertTrue(all(e['raw_legal'] for e in rec['events']))
            if policy!='original_late':self.assertEqual(m['protected_contacts'],0)
            if policy=='original_late':
                self.assertEqual(w0.png_bytes(rec),z.read('rollouts/greebles-2819-intact-late.png'))
            originals.append((m['novel_positions'],hashlib.sha256(w0.png_bytes(rec)).hexdigest()))
        # The mask is active: the source original differs from both filtered trajectories.
        self.assertNotEqual(originals[0],originals[1])

if __name__=='__main__':unittest.main()

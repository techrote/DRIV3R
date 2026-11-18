"""IN K-ALIAS-MEMORY-001 focused host-replay and causal-memory regression tests."""
from __future__ import annotations
import copy
import unittest
import numpy as np
from .world import Spec
from .ink_alias_memory_lab import (DEPTHS, GOAL, ACTIONS, state, world,
    limited_observation, history_features, train_probe, probe, make_cases,
    replay_receipt, run, verify_archive)
from .detour_recovery_lab import shortest_empty_path


class InkAliasMemoryTests(unittest.TestCase):
    def test_complete_aliases_and_opposing_optimal_actions(self):
        for depth in DEPTHS:
            sp=Spec('development','pickets',3000,'wall')
            east=state(sp,depth,'E');west=state(sp,depth,'W')
            self.assertEqual(limited_observation(east),limited_observation(west))
            self.assertNotEqual(east.state_sha(),west.state_sha())
            self.assertEqual(shortest_empty_path(east,east.cursor,GOAL)[0],'MOVE_W')
            self.assertEqual(shortest_empty_path(west,west.cursor,GOAL)[0],'MOVE_E')
            self.assertEqual(len(east.decisions),depth)

    def test_heldout_and_wrong_challenge_rejected(self):
        with self.assertRaises(ValueError):
            world(Spec('heldout','archipelago',9000,'wall'),36)
        with self.assertRaises(ValueError):
            world(Spec('development','pickets',3000,'wall','trap_corridor'),36)

    def test_paired_full_history_probe_but_no_history_ceiling(self):
        train=make_cases('train',1)
        weights,fit=train_probe([(w,label) for _,_,_,w,label in train],240)
        self.assertEqual(fit['updates'],240)
        dev=make_cases('development',1)
        self.assertEqual(sum(probe(w,weights)[0]==ACTIONS[label] for _,_,_,w,label in dev),len(dev))
        self.assertEqual(sum(label==0 for _,_,_,w,label in dev),len(dev)//2)

    def test_receipt_tampering_detected(self):
        w=state(Spec('development','pickets',3000,'wall'),36,'E')
        rec={'spec':vars(w.spec),'depth':36,'events':copy.deepcopy(w.decisions),
             'final_state_sha256':w.state_sha(),'trace_sha256':__import__('hashlib').sha256(b'bad').hexdigest()}
        with self.assertRaises(AssertionError):replay_receipt(rec)
        rec['trace_sha256']=__import__('experiments.driv3r.training.long_horizon.world',fromlist=['digest']).digest(w.decisions)
        self.assertEqual(replay_receipt(rec),36)
        rec['events'][0]['legal']=['STOP']
        with self.assertRaises(AssertionError):replay_receipt(rec)

    def test_forged_action_rejected(self):
        w=state(Spec('development','ridges',3100,'route'),36,'E')
        rec={'spec':vars(w.spec),'depth':36,'events':copy.deepcopy(w.decisions),
             'final_state_sha256':w.state_sha(),
             'trace_sha256':__import__('experiments.driv3r.training.long_horizon.world',fromlist=['digest']).digest(w.decisions)}
        rec['events'][-1]['chosen']='MOVE_N'
        with self.assertRaises(AssertionError):replay_receipt(rec)

if __name__=='__main__':unittest.main()

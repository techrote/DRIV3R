"""Topology transfer: source, replay and data-leak guardrail tests."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import unittest
from .topology_transfer_lab import (TRAIN_MOTIFS, DEV_MOTIFS, DEPTHS, ROTATIONS,
    ARMS, terrain, fixture, oracle, episode, verify_trace, verify, SEED_PAIRS)
from .world import Spec, specs, MOVE


class TopologyTransferTests(unittest.TestCase):
    def test_split_and_sealed_heldout(self):
        self.assertFalse(set(TRAIN_MOTIFS) & set(DEV_MOTIFS))
        train=specs('train',1)[0];dev=specs('development',1)[0]
        held=Spec('heldout','archipelago',9000,'route')
        with self.assertRaises(ValueError):fixture(train,DEV_MOTIFS[0],3,0,0)
        with self.assertRaises(ValueError):fixture(dev,TRAIN_MOTIFS[0],3,0,0)
        with self.assertRaises(ValueError):fixture(held,DEV_MOTIFS[0],3,0,0)

    def test_prelude_legal_and_base_stable(self):
        for sp in (specs('train',1)[0],specs('development',1)[0]):
            motifs=TRAIN_MOTIFS if sp.split=='train' else DEV_MOTIFS
            for motif in motifs:
                for quarter in ROTATIONS:
                    for depth in DEPTHS:
                        a,goal,_=fixture(sp,motif,depth,quarter,0)
                        b,goal2,_=fixture(sp,motif,depth,quarter,0)
                        self.assertEqual(a.state_sha(),b.state_sha())
                        self.assertEqual(goal,goal2)
                        self.assertEqual(a.logical_moves,depth)
                        self.assertTrue(a.ink.isdisjoint(set(a.materials)|a.protected))
                        self.assertIn(oracle(a,goal),a.menu(12))

    def test_new_motifs_diverge(self):
        sp=specs('development',1)[0]
        a,_,_=fixture(sp,DEV_MOTIFS[0],3,1,0)
        b,_,_=fixture(sp,DEV_MOTIFS[1],3,1,0)
        self.assertNotEqual(a.base_sha(),b.base_sha())
        self.assertNotEqual(a.materials,b.materials)

    def test_oracle_complete_for_constructed_development_tasks(self):
        for sp in specs('development',1):
            for motif in DEV_MOTIFS:
                for depth in DEPTHS:
                    for gi in (0,1):
                        w,goal,_=fixture(sp,motif,depth,(sp.seed+depth)%4,gi)
                        self.assertIsNotNone(oracle(w,goal))
                        self.assertEqual(oracle(w,goal),'BACKTRACK')

    def test_tampered_receipt_rejected(self):
        sp=specs('development',1)[0]
        row,trace=episode(sp,DEV_MOTIFS[0],5,1,0,'prefix_oracle',821,None)
        self.assertTrue(row['arrived'])
        self.assertEqual(verify_trace(trace,None)[0],5+row['actions'])
        tampered=copy.deepcopy(trace)
        tampered['events'][0]['chosen']='STOP'
        with self.assertRaises(AssertionError):verify_trace(tampered,None)

    def test_full_archive_if_provided(self):
        out=os.environ.get('DRIV3R_TOPOLOGY_EVIDENCE')
        cps=os.environ.get('DRIV3R_TOPOLOGY_OLD_CP')
        if not out or not cps:self.skipTest('Optional source-exact checkpoint archive absent')
        result=verify(Path(out),Path(cps))
        self.assertEqual(result['episodes'],256)
        self.assertGreater(result['vectors'],8000)


if __name__=='__main__':unittest.main()

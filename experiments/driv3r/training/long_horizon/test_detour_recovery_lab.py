"""DEVELOPMENT-only source checks; archive replay activated by env variable."""
import copy
import os
from pathlib import Path
import unittest
from .detour_recovery_lab import fixture, prefix_plan, episode, verify_trace, verify, GOALS
from .world import Spec, MOVE, World, specs


class DetourRecoveryTests(unittest.TestCase):
    def spec(self):
        sp=specs('development',1)[0]
        return Spec(sp.split,sp.family,sp.seed,sp.mode,'trap_corridor')

    def test_fixture_exact_reproducibility(self):
        a=fixture(self.spec(),5)
        b=fixture(self.spec(),5)
        self.assertEqual(a.state_sha(),b.state_sha())
        self.assertEqual(a.cursor,(53,48))
        self.assertEqual(a.logical_moves,5)
        self.assertEqual(len(a.decisions),5)
        self.assertEqual(len(a.ink & (a.protected | set(a.materials))),0)

    def test_no_heldout_and_recovery_path(self):
        for raw in specs('development',3):
            sp=Spec(raw.split,raw.family,raw.seed,raw.mode,'trap_corridor')
            for depth in (3,5):
                w=fixture(sp,depth)
                for goal in GOALS:
                    plan=prefix_plan(w,goal)
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan[0],'BACKTRACK')
                    self.assertTrue(set(plan).issubset(set(MOVE)|{'BACKTRACK'}))
                    self.assertLessEqual(len(plan)+depth,64)

    def test_all_actions_legal_and_tamper_rejected(self):
        sp=self.spec()
        r,t=episode(sp,5,GOALS[0],'prefix_search',711,None)
        self.assertTrue(r['arrived'])
        self.assertEqual(verify_trace(t,None)[0],r['continuation_actions'])
        self.assertEqual(r['synthetic_relocations'],0)
        tampered=copy.deepcopy(t)
        tampered['events'][0]['chosen']='STOP'
        with self.assertRaises(AssertionError):verify_trace(tampered,None)

    def test_structural_heldout_guard(self):
        sp=Spec('heldout','archipelago',9000,'route','trap_corridor')
        with self.assertRaises(ValueError):fixture(sp,5)

    def test_full_checkpoint_archive_if_supplied(self):
        out=os.environ.get('DRIV3R_DETOUR_EVIDENCE')
        cp=os.environ.get('DRIV3R_DETOUR_CHECKPOINTS')
        if not out or not cp:self.skipTest('Optional external immutable archive paths absent')
        r=verify(Path(out),Path(cp))
        self.assertEqual(r['episodes'],384)
        self.assertGreater(r['vectors'],8000)

if __name__=='__main__':unittest.main()

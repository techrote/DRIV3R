"""Synthetic host-legal LOOP-GUARD-001 invariants; no external artifacts required."""
import copy
import unittest
from . import loop_guard_lab as m
from . import sequential_probe_lab as base
from .world import specs

class TestLoopGuard(unittest.TestCase):
    def test_identical_initial_host_input_and_sensor_limit(self):
        pair=base.matched(specs('development',1)[0],0)
        self.assertEqual(pair['trap']['world'].menu(12),pair['open']['world'].menu(12))
        self.assertEqual(m.candidate(pair['trap']['world'],'MOVE_E'),(50,48))
        self.assertEqual(m.candidate(pair['trap']['world'],'BACKTRACK'),(48,48))

    def test_checkpoint_free_episode_replay_and_tamper(self):
        model=base.ProbeRanker(811)
        sp=specs('development',1)[0]
        for kind in m.KIND:
            for arm in m.ARMS:
                ep=m.episode(sp,0,kind,arm,model)
                self.assertEqual(m.verify_episode(ep,model),len(ep['rows']))
                self.assertLessEqual(ep['sensor_cells_read'],7)
                self.assertTrue(all(r['action'] in r['menu'] for r in ep['rows']))
                altered=copy.deepcopy(ep)
                altered['rows'][0]['probabilities'][altered['rows'][0]['action']]=0.12345678
                with self.assertRaises(ValueError):m.verify_episode(altered,model)

    def test_sealed_heldout_rejected(self):
        with self.assertRaises(ValueError):m.episode(specs('heldout',1)[0],0,'trap','raw',None)

    def test_invalid_arm_rejected(self):
        w=base.fixture(specs('development',1)[0],0,'trap')['world']
        with self.assertRaises(ValueError):m.decision(w,None,{},base.fresh_state(),'invalid',None,False)

if __name__=='__main__':unittest.main()

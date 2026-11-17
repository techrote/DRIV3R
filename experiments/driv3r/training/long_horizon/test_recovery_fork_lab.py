"""Safety, causality, split, replay and checkpoint tests for RECOVERY-FORK-001."""
import json
from pathlib import Path
import tempfile
import unittest
from . import recovery_fork_lab as f, goal_recovery_lab as host
from .world import specs, Spec, World


class RecoveryForkTests(unittest.TestCase):
    def test_legal_prefix_and_causal_recovery_difference(self):
        sp=specs('train',1)[0]
        w,cert=f.fork(sp,681,0)
        self.assertEqual(w.recoveries,12)
        self.assertEqual(w.menu(12),['MOVE_E','MOVE_W','STOP'])
        self.assertEqual(w.steps,35)
        self.assertLessEqual(cert['bad']['moves'],7)
        self.assertTrue(cert['bad']['forced_stop'])
        self.assertGreaterEqual(cert['good']['moves'],20)
        self.assertFalse(cert['good']['forced_stop'])
        replay=World.create(w.spec)
        for a in cert['prefix_actions']:replay.apply(a,12)
        self.assertEqual(replay.state_sha(),w.state_sha())
        self.assertFalse(w.ink.intersection(w.protected))
        self.assertFalse(w.ink.intersection(w.materials))

    def test_open_negative_control_is_legally_viable(self):
        sp=specs('development',1)[0]
        w,cert=f.open_fork(sp,f.DEV_SEED,0)
        self.assertEqual(w.spec.challenge,'normal')
        self.assertEqual(w.recoveries,12)
        self.assertIn('MOVE_E',w.menu(12))
        self.assertIn('MOVE_W',w.menu(12))
        self.assertGreaterEqual(cert['east']['moves'],20)
        self.assertGreaterEqual(cert['west']['moves'],20)

    def test_fork_never_uses_heldout(self):
        with self.assertRaises(ValueError):
            f.fork(Spec('heldout','archipelago',9000,'route'),681)
        self.assertTrue(all(s.split=='development' for s in f.development_cases()))

    def test_rejects_illegal_viability_move(self):
        w=World.create(specs('train',1)[0]);s=w.state_sha()
        with self.assertRaises(ValueError):f.viability(w,'BACKTRACK')
        self.assertEqual(w.state_sha(),s)

    def test_quick_train_restore_verify_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'evidence'
            rep=f.run(out,seed=683,base_updates=2,adapt_updates=2,
                      natural_limit=16,fork_variants=1,quick=True)
            receipt=f.verify(out)
            self.assertEqual(receipt['checkpoints'],3)
            self.assertEqual(receipt['fork_pairs'],4*2*3)
            self.assertTrue(rep['sealed_heldout_opened'] is False)
            p=next((out/'traces').glob('*.json'))
            p.write_text(p.read_text()+'X')
            with self.assertRaises(ValueError):f.verify(out)

if __name__=='__main__':unittest.main()

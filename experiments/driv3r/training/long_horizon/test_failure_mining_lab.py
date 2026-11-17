"""Focused reproducibility/partition/legality checks for FAILURE-MINING-001."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from . import failure_mining_lab as m, failure_outcome_lab as prev, goal_recovery_lab as host
from .world import World, specs


class MineTests(unittest.TestCase):
    def test_risk_does_not_alter_state(self):
        w=World.create(specs('train',1)[0]);state=w.state_sha()
        self.assertEqual(m.risk(w),0)
        self.assertEqual(w.state_sha(),state)

    def test_no_structural_heldout(self):
        self.assertTrue(all(s.split=='development' for s in m.cases()))
        self.assertTrue(all(3000<=s.seed<4000 for s in m.cases()))

    def test_counterfactual_is_nonmutating_and_legal(self):
        sp=specs('train',1)[0];w=World.create(sp)
        frozen=host.Ranker(4)
        from . import goal_retarget_lab as targets
        t=targets.targets_for(sp,44,'retarget');before=w.state_sha()
        a=next(a for a in w.menu(12) if a!='STOP')
        out=prev.student_suffix(w,a,t,0,36,frozen,8)
        self.assertGreater(out['completed'],0)
        self.assertEqual(before,w.state_sha())

    def test_evidence_manifest_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'evidence'
            m.run(out,seed=663,base_updates=2,adapt_updates=1,n=16,depth=8,quick=True)
            receipt=m.verify(out)
            self.assertEqual(receipt['checkpoints'],3)
            p=next((out/'traces').glob('*.json'))
            p.write_text(p.read_text()+'X')
            with self.assertRaises(ValueError):m.verify(out)

if __name__=='__main__':unittest.main()

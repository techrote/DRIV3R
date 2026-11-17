"""Focused negative-result qualification: legal pre-roll, split gate, checkpoint replay."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from . import recovery_debt_audit as audit
from .world import specs, Spec

class RecoveryDebtTests(unittest.TestCase):
    def test_actual_legal_recoveries(self):
        for spent in (10,11):
            for sp in specs('train',1):
                q=audit.fixture(sp,spent);events=q['prefix']
                self.assertEqual(len(events),6+2*spent)
                self.assertEqual(sum(e['chosen']=='BACKTRACK' for e in events),spent)
                self.assertEqual(q['world'].recoveries,spent)
                self.assertEqual(q['world'].menu(12),['BACKTRACK','MOVE_S','RELOCATE_START','STOP'])
                self.assertEqual(q['world'].cursor,(54,48))
    def test_split_and_recovery_guards(self):
        with self.assertRaises(ValueError):audit.fixture(Spec('heldout','archipelago',9000,'wall'),11)
        with self.assertRaises(ValueError):audit.fixture(specs('train',1)[0],12)
    def test_replay_and_manifest_tampering(self):
        archive=Path('/mnt/data/driv3r_outcome_fork_20261117_evidence.zip')
        if not archive.exists():self.skipTest('explicit predecessor evidence archive not mounted')
        models,_=audit.restored(archive,811)
        sp=specs('development',1)[0]
        a=audit.continuation(sp,11,811,'frozen',models['frozen'],'MOVE_S')
        b=audit.continuation(sp,11,811,'frozen',models['frozen'],'BACKTRACK')
        self.assertEqual(a['initial_sha256'],b['initial_sha256'])
        self.assertEqual(a['decisions'],len(a['rows']))
        self.assertEqual(b['decisions'],len(b['rows']))

if __name__=='__main__':unittest.main()
